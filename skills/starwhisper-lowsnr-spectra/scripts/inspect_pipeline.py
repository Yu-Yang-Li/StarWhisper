"""Inspect Low-SNR stellar-spectra files. Does not train or invent a dataset dump."""

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
    items = []
    for rel in PIPELINE["expected"]:
        path = folder / rel if folder else None
        items.append({"path": rel, "present": bool(path and path.exists())})
    payload = {
        "folder": str(folder.as_posix()) if folder else None,
        "present": bool(folder and folder.is_dir()),
        "standalone": PIPELINE["standalone"],
        "weights": PIPELINE["weights"],
        "dataset": PIPELINE["dataset"],
        "snr_stages": PIPELINE["snr_stages"],
        "expected": items,
        "note": "a generated or denoised spectrum is not a new observation",
    }
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
