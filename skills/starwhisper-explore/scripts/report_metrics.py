"""Print Explore published metrics and the pre-registered verdict. No simulation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

SKILL = Path(__file__).resolve().parents[1]
BUNDLED = SKILL / "references" / "published_metrics.csv"


def repo_root() -> Path | None:
    env = os.environ.get("STARWHISPER_ROOT")
    if env:
        p = Path(env)
        if (p / "explore" / "published_metrics.csv").exists():
            return p
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "explore" / "published_metrics.csv").exists() and (parent / "skills" / "starwhisper-index" / "SKILL.md").exists():
            return parent
    return None


def csv_path() -> Path:
    root = repo_root()
    if root:
        return root / "explore" / "published_metrics.csv"
    return BUNDLED


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    path = csv_path()
    rows = {row["policy"]: row for row in csv.DictReader(path.open(encoding="utf-8"))}
    rule = rows["rule_agent"]
    det = rows["deterministic_priority"]
    follow = float(rule["high_value_followup_pct"]) - float(det["high_value_followup_pct"])
    utility_rel = (float(rule["mean_utility"]) / float(det["mean_utility"]) - 1.0) * 100
    completeness = float(rule["survey_completeness_pct"]) - float(det["survey_completeness_pct"])
    payload = {
        "source": str(path.as_posix()),
        "environment_code_in_repo": False,
        "rows": rows,
        "rule_minus_deterministic": {
            "followup_pp": round(follow, 2),
            "utility_rel_pct": round(utility_rel, 2),
            "completeness_pp": round(completeness, 2),
        },
        "positive_bar": {
            "completeness_drop_pp_max": 5.0,
            "followup_rel_gain_min": 20.0,
            "utility_gain_min": 5.0,
        },
        "verdict": "stable_negative",
        "reason": "follow-up rose but survey completeness dropped 9.44 pp, past the 5 pp line",
    }
    if args.json:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        print()
        return 0
    print("verdict:", payload["verdict"])
    print("reason:", payload["reason"])
    print("rule-det followup_pp:", payload["rule_minus_deterministic"]["followup_pp"])
    print("rule-det completeness_pp:", payload["rule_minus_deterministic"]["completeness_pp"])
    print("environment_code_in_repo: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
