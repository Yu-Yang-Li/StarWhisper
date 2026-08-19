"""Screen SN Clock explosion-age predictions into an observing shortlist.

Stdlib only. Reads a published prediction table; never contacts TNS or a telescope.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

CONFIDENCE_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
TIER_STRICT = "strict_q84_within_2d"
TIER_Q50 = "q50_within_2d_only"
WEAK_PROVENANCE = "not_persisted_in_historical_snapshot"


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))


def repo_root() -> Path | None:
    env = os.environ.get("STARWHISPER_ROOT")
    if env and Path(env, "snclock").is_dir():
        return Path(env)
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "snclock").is_dir() and (parent / "skills").is_dir():
            return parent
    return None


def default_csv() -> Path:
    root = repo_root()
    if not root:
        raise SystemExit("no snclock/ found; pass --csv or set STARWHISPER_ROOT")
    tables = sorted((root / "snclock").glob("snclock_*.csv"))
    if not tables:
        raise SystemExit("snclock/ has no snclock_*.csv; pass --csv")
    return tables[-1]


def load(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            for key in ("h3_age_q16_days", "h3_age_q50_days", "h3_age_q84_days", "host_redshift"):
                text = (row.get(key) or "").strip()
                row[key] = float(text) if text else None
            row["discovery_dt"] = parse_utc(row["discovery_time_utc"])
            row["generated_dt"] = parse_utc(row["prediction_generated_utc"])
            rows.append(row)
    return rows


def age_now(row: dict, asof: datetime, quantile: str) -> float | None:
    base = row[f"h3_age_{quantile}_days"]
    if base is None:
        return None
    return base + (asof - row["discovery_dt"]).total_seconds() / 86400.0


def provenance(rows: list[dict]) -> dict:
    warnings = sorted({row["prediction_warning"] for row in rows if row.get("prediction_warning")})
    scope = sorted({row["scope_note_cn"] for row in rows if row.get("scope_note_cn")})
    return {
        "n_rows": len(rows),
        "model_id": sorted({row["model_id"] for row in rows}),
        "discovery_span_utc": [
            min(row["discovery_dt"] for row in rows).isoformat(),
            max(row["discovery_dt"] for row in rows).isoformat(),
        ],
        "prediction_warning": warnings,
        "scope_note_cn": scope,
        "boundary": "model age estimate, not a spectroscopic classification or a discovery claim",
    }


def emit(payload: dict, as_json: bool, table_rows: list[list[str]] | None = None, header: list[str] | None = None) -> int:
    if as_json:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2, default=str)
        print()
        return 0
    if table_rows is not None and header is not None:
        widths = [max(len(header[i]), *(len(r[i]) for r in table_rows)) if table_rows else len(header[i]) for i in range(len(header))]
        print("  ".join(h.ljust(widths[i]) for i, h in enumerate(header)))
        for row in table_rows:
            print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
    for note in payload.get("must_state", []):
        print(f"! {note}")
    return 0


def must_state(rows: list[dict], selected: list[dict]) -> list[str]:
    notes = []
    scope = sorted({row["scope_note_cn"] for row in rows if row.get("scope_note_cn")})
    notes.extend(scope)
    weak = sum(1 for row in selected if row.get("input_mode_record") == WEAK_PROVENANCE)
    if weak:
        notes.append(f"{weak}/{len(selected)} selected rows have input_mode_record={WEAK_PROVENANCE}; the input snapshot was not persisted")
    notes.append("H3 interval is fold-dispersion based; an age estimate is not a classification or a discovery")
    return notes


def cmd_describe(args, rows):
    tiers, confidence, groups = {}, {}, {}
    for row in rows:
        tiers[row["selection_tier"]] = tiers.get(row["selection_tier"], 0) + 1
        confidence[row["model_confidence_label"]] = confidence.get(row["model_confidence_label"], 0) + 1
        for g in (row.get("reporting_group") or "unknown").split(","):
            g = g.strip() or "unknown"
            groups[g] = groups.get(g, 0) + 1
    payload = {
        "csv": str(args.csv),
        "provenance": provenance(rows),
        "selection_tier": tiers,
        "model_confidence_label": confidence,
        "reporting_group": groups,
        "missing_host_redshift": sum(1 for row in rows if row["host_redshift"] is None),
        "must_state": must_state(rows, rows),
    }
    if not args.json:
        print(f"rows: {len(rows)}  span: {payload['provenance']['discovery_span_utc'][0]} .. {payload['provenance']['discovery_span_utc'][1]}")
        for key in ("selection_tier", "model_confidence_label"):
            print(f"{key}: " + ", ".join(f"{k}={v}" for k, v in sorted(payload[key].items())))
        print(f"missing host_redshift: {payload['missing_host_redshift']}")
    return emit(payload, args.json)


def apply_filters(rows: list[dict], args) -> list[dict]:
    out = []
    for row in rows:
        if args.tier == "strict" and row["selection_tier"] != TIER_STRICT:
            continue
        if args.tier == "q50" and row["selection_tier"] != TIER_Q50:
            continue
        if args.max_q50 is not None and (row["h3_age_q50_days"] is None or row["h3_age_q50_days"] > args.max_q50):
            continue
        if args.min_confidence and CONFIDENCE_ORDER.get(row["model_confidence_label"], -1) < CONFIDENCE_ORDER[args.min_confidence]:
            continue
        if args.max_redshift is not None and (row["host_redshift"] is None or row["host_redshift"] > args.max_redshift):
            continue
        if args.require_redshift and row["host_redshift"] is None:
            continue
        if args.exclude_weak_provenance and row.get("input_mode_record") == WEAK_PROVENANCE:
            continue
        out.append(row)
    return out


def shortlist_payload(args, rows, selected, asof, extra=None):
    items = []
    for row in sorted(selected, key=lambda r: r["h3_age_q50_days"]):
        item = {
            "source_name": row["source_name"],
            "tns_url": row["tns_url"],
            "discovery_time_utc": row["discovery_time_utc"],
            "h3_age_q16_q50_q84_days": [row["h3_age_q16_days"], row["h3_age_q50_days"], row["h3_age_q84_days"]],
            "selection_tier": row["selection_tier"],
            "model_confidence_label": row["model_confidence_label"],
            "host_redshift": row["host_redshift"],
            "reporting_group": row["reporting_group"],
            "input_mode_record": row["input_mode_record"],
        }
        if asof:
            item["age_now_q50_days"] = round(age_now(row, asof, "q50"), 3)
            item["age_now_q84_days"] = round(age_now(row, asof, "q84"), 3)
        items.append(item)
    payload = {
        "csv": str(args.csv),
        "asof_utc": asof.isoformat() if asof else None,
        "n_input": len(rows),
        "n_selected": len(selected),
        "selected": items,
        "provenance": provenance(rows),
        "must_state": must_state(rows, selected),
    }
    if extra:
        payload.update(extra)
    return payload


def render(payload, args, asof):
    header = ["source", "disc_utc", "q16", "q50", "q84", "tier", "conf", "z"]
    if asof:
        header += ["age_now_q50", "age_now_q84"]
    table = []
    for item in payload["selected"]:
        q16, q50, q84 = item["h3_age_q16_q50_q84_days"]
        row = [
            item["source_name"],
            item["discovery_time_utc"][:10],
            f"{q16:.2f}",
            f"{q50:.2f}",
            f"{q84:.2f}",
            "strict" if item["selection_tier"] == TIER_STRICT else "q50",
            item["model_confidence_label"],
            f"{item['host_redshift']:.4f}" if item["host_redshift"] is not None else "-",
        ]
        if asof:
            row += [f"{item['age_now_q50_days']:.2f}", f"{item['age_now_q84_days']:.2f}"]
        table.append(row)
    if not args.json:
        print(f"selected {payload['n_selected']}/{payload['n_input']}")
    return emit(payload, args.json, table, header)


def cmd_screen(args, rows):
    selected = apply_filters(rows, args)
    asof = parse_utc(args.asof) if args.asof else None
    payload = shortlist_payload(args, rows, selected, asof)
    return render(payload, args, asof)


def cmd_rank(args, rows):
    selected = sorted(apply_filters(rows, args), key=lambda r: (r["h3_age_q50_days"], r["h3_age_q84_days"]))
    if args.top:
        selected = selected[: args.top]
    payload = shortlist_payload(args, rows, selected, None, {"ranked_by": "h3_age_q50_days ascending, tie-break h3_age_q84_days"})
    return render(payload, args, None)


def cmd_window(args, rows):
    asof = parse_utc(args.asof) if args.asof else datetime.now(timezone.utc)
    selected = []
    for row in apply_filters(rows, args):
        value = age_now(row, asof, "q84" if args.conservative else "q50")
        if value is not None and value <= args.within_days:
            selected.append(row)
    payload = shortlist_payload(
        args,
        rows,
        selected,
        asof,
        {
            "window_days": args.within_days,
            "quantile_used": "q84" if args.conservative else "q50",
            "note": "age_now = h3 age at discovery + elapsed time since discovery",
        },
    )
    return render(payload, args, asof)


def cmd_audit(args, rows):
    asof = parse_utc(args.asof) if args.asof else datetime.now(timezone.utc)
    weak = [row["source_name"] for row in rows if row.get("input_mode_record") == WEAK_PROVENANCE]
    stale = [
        {"source_name": row["source_name"], "generated_utc": row["prediction_generated_utc"], "age_days": round((asof - row["generated_dt"]).total_seconds() / 86400.0, 2)}
        for row in rows
        if (asof - row["generated_dt"]) > timedelta(days=args.stale_after_days)
    ]
    strict = [row["source_name"] for row in rows if row["selection_tier"] == TIER_STRICT]
    inconsistent = [
        row["source_name"]
        for row in rows
        if (row["selection_tier"] == TIER_STRICT) != (str(row["conservative_q84_within_2d"]).lower() == "true")
    ]
    payload = {
        "csv": str(args.csv),
        "asof_utc": asof.isoformat(),
        "provenance": provenance(rows),
        "strict_tier_sources": strict,
        "weak_provenance_sources": weak,
        "weak_provenance_fraction": round(len(weak) / len(rows), 3) if rows else 0,
        "stale_predictions": stale,
        "stale_after_days": args.stale_after_days,
        "tier_flag_inconsistencies": inconsistent,
        "missing_host_redshift": [row["source_name"] for row in rows if row["host_redshift"] is None],
        "must_state": must_state(rows, rows),
    }
    if not args.json:
        print(f"strict tier: {len(strict)}  weak provenance: {len(weak)}/{len(rows)}  stale(>{args.stale_after_days}d): {len(stale)}")
        if inconsistent:
            print("tier flag inconsistencies: " + ", ".join(inconsistent))
    return emit(payload, args.json)


def add_filters(parser):
    parser.add_argument("--tier", choices=["strict", "q50", "any"], default="any")
    parser.add_argument("--max-q50", type=float, help="max h3_age_q50_days at discovery")
    parser.add_argument("--min-confidence", choices=["LOW", "MEDIUM", "HIGH"])
    parser.add_argument("--max-redshift", type=float)
    parser.add_argument("--require-redshift", action="store_true")
    parser.add_argument("--exclude-weak-provenance", action="store_true")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--json", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("describe", help="counts, provenance, and the scope caveat")
    p.set_defaults(func=cmd_describe)

    p = sub.add_parser("screen", help="filter into an observing shortlist")
    add_filters(p)
    p.add_argument("--asof", help="UTC timestamp to age the candidates to")
    p.set_defaults(func=cmd_screen)

    p = sub.add_parser("rank", help="rank youngest first")
    add_filters(p)
    p.add_argument("--top", type=int)
    p.set_defaults(func=cmd_rank)

    p = sub.add_parser("window", help="who is still within N days of explosion as of a time")
    add_filters(p)
    p.add_argument("--within-days", type=float, default=2.0)
    p.add_argument("--asof")
    p.add_argument("--conservative", action="store_true", help="use q84 instead of q50")
    p.set_defaults(func=cmd_window)

    p = sub.add_parser("audit", help="provenance, staleness, and tier-flag consistency")
    p.add_argument("--asof")
    p.add_argument("--stale-after-days", type=float, default=7.0)
    p.set_defaults(func=cmd_audit)

    args = parser.parse_args()
    args.csv = args.csv or default_csv()
    rows = load(args.csv)
    return args.func(args, rows)


if __name__ == "__main__":
    raise SystemExit(main())
