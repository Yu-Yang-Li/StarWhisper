"""Inventory LLM_Data JSON files against the published file list. Does not load a fake corpus."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

SKILL = Path(__file__).resolve().parents[1]
EXPECTED = json.loads((SKILL / "references" / "expected_files.json").read_text(encoding="utf-8"))


def repo_root() -> Path | None:
    env = os.environ.get("STARWHISPER_ROOT")
    if env and Path(env, "LLM_Data").exists():
        return Path(env)
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "LLM_Data").exists() and (parent / "skills" / "starwhisper-index" / "SKILL.md").exists():
            return parent
    return None


def main() -> int:
    root = repo_root()
    folder = root / "LLM_Data" if root else None
    files = []
    for name in EXPECTED["files"]:
        path = folder / name if folder else None
        present = bool(path and path.is_file())
        item = {"file": name, "present": present}
        if present:
            item["bytes"] = path.stat().st_size
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                item["n_records"] = len(data) if isinstance(data, list) else None
            except OSError as exc:
                item["error"] = str(exc)
        files.append(item)
    payload = {
        "folder": str(folder.as_posix()) if folder else None,
        "weights": EXPECTED["weights"],
        "note": EXPECTED["note"],
        "files": files,
        "present_count": sum(1 for row in files if row["present"]),
    }
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
