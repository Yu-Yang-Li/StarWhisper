"""Route a Virtual Sitian question to a SitianClaw skill. Does not reimplement those skills."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SKILL = Path(__file__).resolve().parents[1]
CATALOG = json.loads((SKILL / "references" / "sitianclaw-skills.json").read_text(encoding="utf-8"))


def score(query: str, keywords: list[str]) -> int:
    q = query.casefold()
    return sum(1 for k in keywords if k.casefold() in q)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    ranked = sorted(((score(args.query, row["keywords"]), row) for row in CATALOG["skills"]), key=lambda x: x[0], reverse=True)
    hits = [row for n, row in ranked if n > 0][:3]
    if not hits:
        hits = [next(row for row in CATALOG["skills"] if row["id"] == "snc-transient-query")]
        note = "no keyword hit; default to snc-transient-query. Install SitianClaw, do not reimplement here."
    else:
        note = "run these in https://github.com/Yu-Yang-Li/SitianClaw ; StarWhisper only routes, do not reimplement here"
    payload = {
        "query": args.query,
        "matches": hits,
        "repo": CATALOG["repo"],
        "map": CATALOG["map"],
        "note": note,
    }
    if args.json:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        print()
        return 0
    print(payload["note"])
    for row in hits:
        print(row["id"])
    print("repo:", payload["repo"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
