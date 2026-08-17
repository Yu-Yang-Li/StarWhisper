#!/usr/bin/env python3
"""Preflight a visual-target extraction run without exposing secrets.

This script checks whether the run has a readable visual target and an available
image-editing backend for true background/frame/icon extraction. It does not
call any provider and never prints token values.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from PIL import Image


def _image_info(path: Path) -> dict:
    if not path.exists():
        return {"exists": False, "readable": False, "path": str(path)}
    try:
        with Image.open(path) as im:
            return {"exists": True, "readable": True, "path": str(path), "width": im.width, "height": im.height}
    except Exception as exc:
        return {"exists": True, "readable": False, "path": str(path), "error": str(exc)}


def _prompt_status(run_root: Path, slide_id: str) -> dict:
    prompts = {}
    for name in ["visual-target", "background", "frame", "icons"]:
        path = run_root / "prompts" / f"{slide_id}-{name}.md"
        prompts[name] = {"path": str(path), "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0}
    return prompts


def preflight(target: Path, run_root: Path | None, slide_id: str, allow_chat_imagegen: bool) -> dict:
    target_info = _image_info(target)
    giiisp_set = bool(os.environ.get("GIIISP_AUTH_TOKEN"))
    openai_set = bool(os.environ.get("OPENAI_API_KEY"))
    backend_candidates = {
        "giiisp_env_token": giiisp_set,
        "giiisp_env_token_name": "GIIISP_AUTH_TOKEN",
        "openai_env_token": openai_set,
        "chat_imagegen_available_to_agent": allow_chat_imagegen,
    }

    blockers: list[str] = []
    if not target_info.get("readable"):
        blockers.append("visual target is missing or unreadable")
    if not giiisp_set and not openai_set and not allow_chat_imagegen:
        blockers.append(
            "no auditable image-editing backend is available in the local runtime; "
            "apply for GIIISP_AUTH_TOKEN at https://giiisp.com/#/mcp/authenticate "
            "or use the host platform's built-in image generation"
        )
    if allow_chat_imagegen:
        blockers.append(
            "chat imagegen may require the current target image to be attached or visible as an edit target; a local file path alone is not evidence of image input"
        )

    prompts = _prompt_status(run_root, slide_id) if run_root else None
    if prompts:
        missing = [name for name, info in prompts.items() if not info["exists"]]
        if missing:
            blockers.append(f"missing prompt files: {', '.join(missing)}")

    status = "pass" if not blockers else "blocked"
    return {
        "status": status,
        "slide_id": slide_id,
        "target": target_info,
        "backend_candidates": backend_candidates,
        "prompt_files": prompts,
        "blockers": blockers,
        "notes": [
            "This preflight only checks readiness. It does not generate or extract layers.",
            "Do not substitute programmatic layers for blocked image-model extraction in final delivery.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--slide-id", default="01")
    parser.add_argument("--allow-chat-imagegen", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report = preflight(args.target, args.run_root, args.slide_id, args.allow_chat_imagegen)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)
    if report["status"] == "blocked":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
