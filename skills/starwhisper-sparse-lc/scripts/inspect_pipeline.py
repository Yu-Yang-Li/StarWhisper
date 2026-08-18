"""Inspect the sparse-LC benchmark folder against the published pipeline. Does not train."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

SKILL = Path(__file__).resolve().parents[1]
PIPELINE = json.loads((SKILL / "references" / "pipeline.json").read_text(encoding="utf-8"))


def repo_root() -> Path | None:
    env = os.environ.get("STARWHISPER_ROOT")
    if env and Path(env, PIPELINE["folder"]).exists():
        return Path(env)
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / PIPELINE["folder"]).exists() and (parent / "skills" / "starwhisper-index" / "SKILL.md").exists():
            return parent
    return None


def main() -> int:
    root = repo_root()
    folder = root / PIPELINE["folder"] if root else None
    dirs = []
    for name in PIPELINE["expected_dirs"]:
        path = folder / name if folder else None
        dirs.append({"dir": name, "present": bool(path and path.is_dir())})
    payload = {
        "folder": str(folder.as_posix()) if folder else None,
        "present": bool(folder and folder.is_dir()),
        "weights": PIPELINE["weights"],
        "setting": PIPELINE["setting"],
        "dirs": dirs,
        "steps": PIPELINE["steps"],
        "note": "a test-set metric is not an explosion time or a discovery",
    }
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
