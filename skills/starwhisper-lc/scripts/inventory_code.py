"""List StarWhisper_LC test-code tree. Does not train."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def repo_root() -> Path | None:
    env = os.environ.get("STARWHISPER_ROOT")
    if env and Path(env, "StarWhisper_LC").exists():
        return Path(env)
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "StarWhisper_LC").exists() and (parent / "skills" / "starwhisper-index" / "SKILL.md").exists():
            return parent
    return None


def list_files(folder: Path, cap: int) -> list[str]:
    if not folder.exists():
        return []
    return sorted(str(p.relative_to(folder).as_posix()) for p in folder.rglob("*") if p.is_file())[:cap]


def main() -> int:
    root = repo_root()
    folder = root / "StarWhisper_LC" if root else None
    payload = {
        "folder": str(folder.as_posix()) if folder else None,
        "present": bool(folder and folder.is_dir()),
        "paper": "https://spj.science.org/doi/10.34133/icomputing.0110",
        "code_files": list_files(folder / "Code", 80) if folder else [],
        "result_files": list_files(folder / "Result", 40) if folder else [],
        "note": "test code, not a full training reproduction",
    }
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
