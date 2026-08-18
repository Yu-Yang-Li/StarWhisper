"""Locate AllSky-Camera-XL pipeline. Does not run inference unless --image is passed."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REQUIRED_OUTPUTS = [
    "pipeline_report.json",
    "scenario.json",
    "replanned_schedule.json",
    "replanned_sequence.ninaTargetSet",
]


def repo_root() -> Path | None:
    env = os.environ.get("STARWHISPER_ROOT")
    if env and Path(env, "AllSky-Camera-XL").exists():
        return Path(env)
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "AllSky-Camera-XL").exists() and (parent / "skills" / "starwhisper-index" / "SKILL.md").exists():
            return parent
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Absolute path to a raw all-sky jpg. Omit to inspect only.")
    parser.add_argument("--position-name")
    parser.add_argument("--save-overlay", action="store_true")
    args = parser.parse_args()
    root = repo_root()
    folder = root / "AllSky-Camera-XL" if root else None
    entry = folder / "run_pipeline.py" if folder else None
    payload = {
        "folder": str(folder.as_posix()) if folder else None,
        "entry": str(entry.as_posix()) if entry and entry.exists() else None,
        "packaged_skill": "AllSky-Camera-XL/skill/photo-to-replan/SKILL.md",
        "required_outputs": REQUIRED_OUTPUTS,
        "safety": "without --image this script does not run the pipeline or move a telescope",
    }
    if not args.image:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        print()
        return 0 if payload["entry"] else 0
    if not entry or not entry.exists():
        raise SystemExit("run_pipeline.py missing; clone AllSky-Camera-XL before running")
    cmd = [sys.executable, str(entry), "--image", args.image]
    if args.position_name:
        cmd += ["--position-name", args.position_name]
    if args.save_overlay:
        cmd.append("--save-overlay")
    print(" ".join(cmd), file=sys.stderr)
    return subprocess.call(cmd, cwd=str(folder))


if __name__ == "__main__":
    raise SystemExit(main())
