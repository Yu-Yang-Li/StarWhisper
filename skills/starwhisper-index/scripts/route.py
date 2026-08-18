"""Route a StarWhisper question to one native skill. No network."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "catalog.json"


def score(query: str, keywords: list[str]) -> int:
    q = query.casefold()
    return sum(1 for k in keywords if k.casefold() in q)


def main() -> int:
    parser = argparse.ArgumentParser(description="Route a StarWhisper question to a skill.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    lines = json.loads(CATALOG.read_text(encoding="utf-8"))["lines"]
    ranked = sorted(((score(args.query, row["keywords"]), row) for row in lines), key=lambda x: x[0], reverse=True)
    hits = [row for n, row in ranked if n > 0][:3]
    if not hits:
        hits = [next(row for row in lines if row["id"] == "llm")]
        note = "no keyword hit; default to index/llm. Rephrase with NGSS, LC, Explore, Sitian, or ADS."
    else:
        note = "do not mix a paper metric, Explore table, and hardware command in one claim"
    payload = {"query": args.query, "matches": hits, "note": note}
    if args.json:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        print()
        return 0
    print(payload["note"])
    for row in hits:
        print(f"{row['skill']}\t{row['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
