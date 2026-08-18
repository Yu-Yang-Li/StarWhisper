"""List GOTTA_Prototype files. Does not score new cutouts."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def repo_root() -> Path | None:
    env = os.environ.get("STARWHISPER_ROOT")
    if env and Path(env, "GOTTA_Prototype").exists():
        return Path(env)
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "GOTTA_Prototype").exists() and (parent / "skills" / "starwhisper-index" / "SKILL.md").exists():
            return parent
    return None


def main() -> int:
    root = repo_root()
    folder = root / "GOTTA_Prototype" if root else None
    files = sorted(str(p.relative_to(folder).as_posix()) for p in folder.rglob("*") if p.is_file())[:60] if folder and folder.exists() else []
    json.dump(
        {
            "folder": str(folder.as_posix()) if folder else None,
            "present": bool(folder and folder.is_dir()),
            "files": files,
            "note": "prototype scores are not broker alerts or spectroscopic types",
        },
        sys.stdout,
        ensure_ascii=False,
        indent=2,
    )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
