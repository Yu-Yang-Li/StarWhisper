"""Print StarWhisper-Pulsar pointers. This line is not vendored here."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SKILL = Path(__file__).resolve().parents[1]
POINTERS = json.loads((SKILL / "references" / "pointers.json").read_text(encoding="utf-8"))


def main() -> int:
    json.dump(POINTERS, sys.stdout, ensure_ascii=False, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
