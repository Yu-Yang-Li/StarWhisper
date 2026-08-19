"""Route a StarWhisper question to a skill that does the work, or to a reference asset.

Stdlib only, no network.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CATALOG = json.loads((Path(__file__).resolve().parents[1] / "catalog.json").read_text(encoding="utf-8"))
BOUNDARY = "do not mix a published paper metric, the synthetic Explore table, and a hardware command in one claim"


def score(query: str, keywords: list[str]) -> int:
    q = query.casefold()
    return sum(1 for k in keywords if k.casefold() in q)


def rank(query: str, entries: list[dict]) -> list[dict]:
    scored = [(score(query, e["keywords"]), e) for e in entries]
    return [e for n, e in sorted(scored, key=lambda x: x[0], reverse=True) if n > 0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    skills = rank(args.query, CATALOG["skills"])[:3]
    assets = rank(args.query, CATALOG["assets"])[:3]

    if skills:
        action = "run the skill below"
    elif assets:
        action = "no skill runs this line; it is reference material, read the asset and say so"
    else:
        action = "no match; ask which line is meant, do not guess a number"

    payload = {
        "query": args.query,
        "action": action,
        "skills": [{k: v for k, v in s.items() if k != "keywords"} for s in skills],
        "assets": [{k: v for k, v in a.items() if k != "keywords"} for a in assets],
        "asset_map": "skills/starwhisper-index/references/asset-map.md",
        "boundary": BOUNDARY,
    }

    if args.json:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        print()
        return 0

    print(action)
    for s in payload["skills"]:
        print(f"  skill  {s['skill']}: {s['does']}")
        print(f"         {s['run']}")
    for a in payload["assets"]:
        print(f"  asset  {a['name']} -> {a['path']}")
    print(f"! {BOUNDARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
