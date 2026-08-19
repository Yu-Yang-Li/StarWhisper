"""Check an NGSS night configuration, exposure budget, and target list before anyone touches hardware.

Stdlib only. This script never opens a socket: no HTTP, no MQTT, no FTP, no NINA.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

SKILL = Path(__file__).resolve().parents[1]
API = json.loads((SKILL / "references" / "api.json").read_text(encoding="utf-8"))
EXAMPLE_CONFIG = SKILL / "references" / "observe_config.example.json"

RA_RANGE = (0.0, 360.0)
DEC_RANGE = (-90.0, 90.0)


def repo_root() -> Path | None:
    env = os.environ.get("STARWHISPER_ROOT")
    if env and Path(env, "NGSS").is_dir():
        return Path(env)
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "NGSS").is_dir() and (parent / "skills").is_dir():
            return parent
    return None


def resolve_config(explicit: Path | None) -> tuple[dict, str]:
    if explicit:
        return json.loads(explicit.read_text(encoding="utf-8")), str(explicit)
    root = repo_root()
    live = root / "NGSS" / "observe_config.json" if root else None
    if live and live.exists():
        return json.loads(live.read_text(encoding="utf-8")), "NGSS/observe_config.json"
    return json.loads(EXAMPLE_CONFIG.read_text(encoding="utf-8")), "bundled example (NGSS/observe_config.json not found)"


def check_config(config: dict) -> list[dict]:
    findings = []

    def fail(level, field, message):
        findings.append({"level": level, "field": field, "message": message})

    windows = config.get("time_windows") or {}
    if not windows:
        fail("error", "time_windows", "missing; no observable time is defined")
    for name, hours in windows.items():
        if not isinstance(hours, (int, float)) or hours <= 0:
            fail("error", f"time_windows.{name}", f"must be a positive number of hours, got {hours!r}")
    total = sum(v for v in windows.values() if isinstance(v, (int, float)))
    if total > 14:
        fail("warn", "time_windows", f"total {total} h is longer than any real night at mid latitude")

    d_moon = (config.get("constraints") or {}).get("d_moon")
    if d_moon is None:
        fail("error", "constraints.d_moon", "missing; targets would not be screened against the Moon")
    elif not 0 <= d_moon <= 180:
        fail("error", "constraints.d_moon", f"must be within 0-180 degrees, got {d_moon}")
    elif d_moon < 15:
        fail("warn", "constraints.d_moon", f"{d_moon} deg is permissive; sky background near the Moon will dominate")

    filters = config.get("filters")
    if not filters:
        fail("error", "filters", "missing or empty")

    exposure = config.get("exposure") or {}
    for key in ("count", "time"):
        value = exposure.get(key)
        if not isinstance(value, (int, float)) or value <= 0:
            fail("error", f"exposure.{key}", f"must be positive, got {value!r}")
    wait = exposure.get("wait")
    if wait is None or wait < 0:
        fail("error", "exposure.wait", f"must be zero or positive minutes, got {wait!r}")

    if "inherit" not in config:
        fail("warn", "inherit", "not set; the planner will not know whether to reuse the previous schedule")

    return findings


def budget(config: dict, slew_seconds: float) -> dict:
    exposure = config.get("exposure") or {}
    count = float(exposure.get("count") or 0)
    exp_time = float(exposure.get("time") or 0)
    wait_min = float(exposure.get("wait") or 0)
    n_filters = len(config.get("filters") or []) or 1
    per_filter = count * exp_time + max(count - 1, 0) * wait_min * 60.0
    per_target = n_filters * per_filter + slew_seconds
    windows = {k: float(v) for k, v in (config.get("time_windows") or {}).items() if isinstance(v, (int, float))}
    total_hours = sum(windows.values())
    total_seconds = total_hours * 3600.0
    return {
        "filters": config.get("filters"),
        "exposures_per_filter": count,
        "exposure_seconds": exp_time,
        "wait_minutes_between_exposures": wait_min,
        "slew_overhead_seconds": slew_seconds,
        "seconds_per_target": round(per_target, 1),
        "minutes_per_target": round(per_target / 60.0, 2),
        "window_hours": windows,
        "total_window_hours": round(total_hours, 2),
        "targets_that_fit": int(total_seconds // per_target) if per_target > 0 else 0,
        "note": "geometric capacity only; no airmass, moon, weather, or slew-path modelling",
    }


def read_targets(path: Path) -> tuple[list[dict], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = [f.strip().lower() for f in (reader.fieldnames or [])]
        rows = [{(k or "").strip().lower(): (v or "").strip() for k, v in row.items()} for row in reader]
    return rows, fields


def lint_targets(path: Path, cap: int) -> dict:
    rows, fields = read_targets(path)
    findings = []

    def fail(level, message, where=None):
        findings.append({"level": level, "message": message, "where": where})

    name_key = next((k for k in ("name", "objname", "target", "source_name") if k in fields), None)
    ra_key = next((k for k in ("ra", "ra_deg", "raj2000") if k in fields), None)
    dec_key = next((k for k in ("dec", "dec_deg", "decj2000") if k in fields), None)
    if not name_key:
        fail("error", f"no target-name column; looked for name/objname/target/source_name, found {fields}")
    if not ra_key or not dec_key:
        fail("error", f"no RA/Dec columns; looked for ra/dec variants, found {fields}")

    seen = {}
    for i, row in enumerate(rows, start=2):
        name = row.get(name_key or "", "")
        if name_key and not name:
            fail("error", "empty target name", f"line {i}")
        if name_key and name in seen:
            fail("error", f"duplicate target {name!r}, first seen at line {seen[name]}", f"line {i}")
        elif name_key:
            seen[name] = i
        for key, rng, label in ((ra_key, RA_RANGE, "RA"), (dec_key, DEC_RANGE, "Dec")):
            if not key:
                continue
            raw = row.get(key, "")
            try:
                value = float(raw)
            except ValueError:
                fail("error", f"{label} {raw!r} is not a number", f"line {i}")
                continue
            if not rng[0] <= value <= rng[1]:
                fail("error", f"{label} {value} outside {rng[0]}-{rng[1]}", f"line {i}")

    if cap and len(rows) > cap:
        fail("warn", f"{len(rows)} targets exceeds the {cap} that fit in the configured windows")

    return {
        "csv": str(path),
        "n_targets": len(rows),
        "columns": fields,
        "capacity": cap,
        "findings": findings,
        "errors": sum(1 for f in findings if f["level"] == "error"),
        "warnings": sum(1 for f in findings if f["level"] == "warn"),
    }


def cmd_check_config(args):
    config, source = resolve_config(args.config)
    findings = check_config(config)
    payload = {
        "source": source,
        "config": config,
        "findings": findings,
        "errors": sum(1 for f in findings if f["level"] == "error"),
        "warnings": sum(1 for f in findings if f["level"] == "warn"),
    }
    if args.json:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        print()
    else:
        print(f"config source: {source}")
        for f in findings:
            print(f"  [{f['level'].upper()}] {f['field']}: {f['message']}")
        print(f"errors={payload['errors']} warnings={payload['warnings']}")
    return 1 if payload["errors"] else 0


def cmd_budget(args):
    config, source = resolve_config(args.config)
    payload = budget(config, args.slew_seconds)
    payload["source"] = source
    if args.json:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        print()
        return 0
    print(f"config source: {source}")
    print(f"per target: {payload['minutes_per_target']} min ({payload['seconds_per_target']} s), filters={payload['filters']}")
    print(f"night windows: {payload['total_window_hours']} h -> {payload['targets_that_fit']} targets fit")
    print(f"! {payload['note']}")
    return 0


def cmd_lint_targets(args):
    config, _ = resolve_config(args.config)
    cap = budget(config, args.slew_seconds)["targets_that_fit"]
    payload = lint_targets(args.targets, cap)
    if args.json:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        print()
    else:
        print(f"{payload['n_targets']} targets, capacity {cap}")
        for f in payload["findings"]:
            where = f" ({f['where']})" if f.get("where") else ""
            print(f"  [{f['level'].upper()}] {f['message']}{where}")
        print(f"errors={payload['errors']} warnings={payload['warnings']}")
    return 1 if payload["errors"] else 0


def cmd_endpoints(args):
    payload = {
        "start": API["start"],
        "cwd": API["cwd"],
        "routes": API["routes"],
        "hardware": [r["path"] for r in API["routes"] if r["class"] == "hardware"],
        "safety": "this skill prints the contract; it does not call these endpoints",
    }
    if args.json:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        print()
        return 0
    print(f"start from {API['cwd']}/: {API['start']}")
    for r in API["routes"]:
        print(f"  {r['method']:6} {r['path']:34} {r['class']}")
    print(f"! {payload['safety']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="observe_config.json; defaults to NGSS/ then bundled example")
    parser.add_argument("--slew-seconds", type=float, default=60.0)
    parser.add_argument("--json", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("check-config", help="validate observe_config fields and ranges")
    p.set_defaults(func=cmd_check_config)

    p = sub.add_parser("budget", help="how many targets fit in the configured windows")
    p.set_defaults(func=cmd_budget)

    p = sub.add_parser("lint-targets", help="validate a target CSV against the night budget")
    p.add_argument("--targets", type=Path, required=True)
    p.set_defaults(func=cmd_lint_targets)

    p = sub.add_parser("endpoints", help="print the NGSS route contract, read / mutate / hardware")
    p.set_defaults(func=cmd_endpoints)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
