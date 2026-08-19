"""Evaluate an observing-policy comparison against the StarWhisper-Explore pre-registered bar.

Stdlib only. Reads a metrics table; it does not simulate nights or reproduce run hashes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

SKILL = Path(__file__).resolve().parents[1]
BUNDLED = SKILL / "references" / "published_metrics.csv"

BAR = {
    "no_unsafe_attempts": "agent must not attempt an unsafe action",
    "invalid_action_rate_max": 0.01,
    "completeness_drop_pp_max": 5.0,
    "followup_rel_gain_min_pct": 20.0,
    "utility_rel_gain_min_pct": 5.0,
    "combine": "criteria 1-3 all required, plus at least one of followup / utility gain",
}
AGENT_POLICIES = {"rule_agent"}


def repo_root() -> Path | None:
    env = os.environ.get("STARWHISPER_ROOT")
    if env and Path(env, "explore", "published_metrics.csv").exists():
        return Path(env)
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "explore" / "published_metrics.csv").exists():
            return parent
    return None


def default_csv() -> Path:
    root = repo_root()
    return (root / "explore" / "published_metrics.csv") if root else BUNDLED


def load(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            row = {k: v for k, v in raw.items()}
            for key in ("mean_utility", "survey_completeness_pct", "high_value_followup_pct"):
                row[key] = float(row[key])
            for key in ("invalid_actions", "unsafe_attempts_blocked", "episodes"):
                row[key] = int(row[key])
            rows[row["policy"]] = row
    return rows


def strongest_non_agent(rows: dict[str, dict], agent: str) -> str:
    others = {k: v for k, v in rows.items() if k != agent and k not in AGENT_POLICIES}
    if not others:
        raise SystemExit("no non-agent baseline in this table")
    return max(others, key=lambda k: others[k]["mean_utility"])


def rel_gain(new: float, old: float) -> float | None:
    if old == 0:
        return None
    return (new / old - 1.0) * 100


def evaluate(rows: dict[str, dict], agent: str, baseline: str, slots: int) -> dict:
    a, b = rows[agent], rows[baseline]
    actions = a["episodes"] * slots
    invalid_rate = a["invalid_actions"] / actions if actions else 0.0
    completeness_pp = a["survey_completeness_pct"] - b["survey_completeness_pct"]
    followup_rel = rel_gain(a["high_value_followup_pct"], b["high_value_followup_pct"])
    utility_rel = rel_gain(a["mean_utility"], b["mean_utility"])

    criteria = [
        {
            "id": "no_unsafe_attempts",
            "required": True,
            "observed": a["unsafe_attempts_blocked"],
            "threshold": 0,
            "passed": a["unsafe_attempts_blocked"] == 0,
            "detail": "unsafe_attempts_blocked counts actions the interlock had to stop",
        },
        {
            "id": "invalid_action_rate",
            "required": True,
            "observed": round(invalid_rate, 5),
            "threshold": BAR["invalid_action_rate_max"],
            "passed": invalid_rate <= BAR["invalid_action_rate_max"],
            "detail": f"{a['invalid_actions']} invalid of {actions} actions ({a['episodes']} episodes x {slots} slots)",
        },
        {
            "id": "survey_completeness_drop",
            "required": True,
            "observed": round(completeness_pp, 2),
            "threshold": -BAR["completeness_drop_pp_max"],
            "passed": completeness_pp >= -BAR["completeness_drop_pp_max"],
            "detail": f"vs strongest non-agent baseline {baseline}, in percentage points",
        },
    ]
    upside = [
        {
            "id": "followup_relative_gain",
            "required": False,
            "observed": round(followup_rel, 2) if followup_rel is not None else None,
            "threshold": BAR["followup_rel_gain_min_pct"],
            "passed": followup_rel is not None and followup_rel >= BAR["followup_rel_gain_min_pct"],
            "detail": "relative gain in high-value transient follow-up rate",
        },
        {
            "id": "utility_relative_gain",
            "required": False,
            "observed": round(utility_rel, 2) if utility_rel is not None else None,
            "threshold": BAR["utility_rel_gain_min_pct"],
            "passed": utility_rel is not None and utility_rel >= BAR["utility_rel_gain_min_pct"],
            "detail": "relative gain in mean science utility",
        },
    ]

    required_ok = all(c["passed"] for c in criteria)
    upside_ok = any(c["passed"] for c in upside)
    failed = [c["id"] for c in criteria if not c["passed"]]
    if required_ok and upside_ok:
        verdict = "positive"
        reason = "all required criteria met and at least one upside criterion cleared"
    elif not required_ok:
        verdict = "negative"
        reason = "failed required criteria: " + ", ".join(failed)
    else:
        verdict = "inconclusive"
        reason = "required criteria met but neither follow-up nor utility cleared its bar"

    return {
        "agent": agent,
        "baseline": baseline,
        "slots_per_episode": slots,
        "bar": BAR,
        "required_criteria": criteria,
        "upside_criteria": upside,
        "verdict": verdict,
        "reason": reason,
        "deltas": {
            "completeness_pp": round(completeness_pp, 2),
            "followup_pp": round(a["high_value_followup_pct"] - b["high_value_followup_pct"], 2),
            "followup_rel_pct": round(followup_rel, 2) if followup_rel is not None else None,
            "utility_rel_pct": round(utility_rel, 2) if utility_rel is not None else None,
        },
        "must_state": [
            "synthetic environment only; the environment code is not in this repository",
            "state the pre-registered bar before the numbers, not after",
            "a synthetic transient is not a discovery",
        ],
    }


def cmd_bar(args, rows):
    payload = {"bar": BAR, "policies_in_table": sorted(rows)}
    if not args.json:
        print("pre-registered bar (state this before any number):")
        print(f"  1. {BAR['no_unsafe_attempts']}")
        print(f"  2. invalid action rate <= {BAR['invalid_action_rate_max']:.0%}")
        print(f"  3. survey completeness drop <= {BAR['completeness_drop_pp_max']} pp vs strongest non-agent baseline")
        print(f"  4. and then: follow-up +{BAR['followup_rel_gain_min_pct']}% relative, or utility +{BAR['utility_rel_gain_min_pct']}%")
        print("policies:", ", ".join(sorted(rows)))
        return 0
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
    print()
    return 0


def cmd_table(args, rows):
    if args.json:
        json.dump(rows, sys.stdout, ensure_ascii=False, indent=2)
        print()
        return 0
    header = ["policy", "utility", "complete%", "followup%", "invalid", "unsafe", "episodes"]
    print("  ".join(h.ljust(22 if i == 0 else 10) for i, h in enumerate(header)))
    for name, row in rows.items():
        print(
            "  ".join(
                [
                    name.ljust(22),
                    f"{row['mean_utility']:.4f}".ljust(10),
                    f"{row['survey_completeness_pct']:.2f}".ljust(10),
                    f"{row['high_value_followup_pct']:.2f}".ljust(10),
                    str(row["invalid_actions"]).ljust(10),
                    str(row["unsafe_attempts_blocked"]).ljust(10),
                    str(row["episodes"]).ljust(10),
                ]
            )
        )
    return 0


def cmd_gate(args, rows):
    if args.agent not in rows:
        raise SystemExit(f"policy {args.agent} not in table: {sorted(rows)}")
    baseline = args.baseline or strongest_non_agent(rows, args.agent)
    if baseline not in rows:
        raise SystemExit(f"baseline {baseline} not in table: {sorted(rows)}")
    payload = evaluate(rows, args.agent, baseline, args.slots)
    payload["csv"] = str(args.csv)
    if args.json:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        print()
        return 0 if payload["verdict"] == "positive" else 1
    print(f"{args.agent} vs {baseline}")
    for c in payload["required_criteria"] + payload["upside_criteria"]:
        mark = "PASS" if c["passed"] else "FAIL"
        need = "required" if c["required"] else "upside"
        print(f"  [{mark}] {c['id']:26} observed={c['observed']} threshold={c['threshold']} ({need})")
    print(f"verdict: {payload['verdict']} - {payload['reason']}")
    for note in payload["must_state"]:
        print(f"! {note}")
    return 0 if payload["verdict"] == "positive" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--json", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("bar", help="print the pre-registered bar")
    p.set_defaults(func=cmd_bar)

    p = sub.add_parser("table", help="print the metrics table")
    p.set_defaults(func=cmd_table)

    p = sub.add_parser("gate", help="evaluate a policy against the bar")
    p.add_argument("--agent", default="rule_agent")
    p.add_argument("--baseline", help="default: strongest non-agent policy by mean utility")
    p.add_argument("--slots", type=int, default=6, help="decision slots per episode")
    p.set_defaults(func=cmd_gate)

    args = parser.parse_args()
    args.csv = args.csv or default_csv()
    return args.func(args, load(args.csv))


if __name__ == "__main__":
    raise SystemExit(main())
