#!/usr/bin/env python3
"""Audit a manual or multimodal visible-text review for image-only decks.

This script does not perform OCR. It enforces that a human or multimodal review
has explicitly checked visible text against allow/deny constraints before a deck
is treated as release-ready.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PASS_VALUES = {"pass", "ok", "not_found", "none"}
WARN_VALUES = {"warn", "warning"}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def audit(review_path: Path, min_slides: int = 1, fail_on_warn: bool = False) -> dict:
    data = load_json(review_path)
    slides = data.get("slides", [])
    issues: list[str] = []
    warnings: list[str] = []

    if len(slides) < min_slides:
        issues.append(f"reviewed slide count below {min_slides}")

    required_fields = [
        "slide_id",
        "allowed_text",
        "forbidden_text",
        "unsupported_numbers",
        "unsupported_dates",
        "unsupported_names",
        "readability",
        "overall",
    ]

    for idx, slide in enumerate(slides, start=1):
        sid = slide.get("slide_id") or f"#{idx}"
        for field in required_fields:
            if field not in slide:
                issues.append(f"slide {sid}: missing {field}")

        for field in ["forbidden_text", "unsupported_numbers", "unsupported_dates", "unsupported_names"]:
            value = str(slide.get(field, "")).lower()
            if value not in PASS_VALUES:
                issues.append(f"slide {sid}: {field}={slide.get(field)!r}")

        for field in ["allowed_text", "readability", "overall"]:
            value = str(slide.get(field, "")).lower()
            if value in WARN_VALUES:
                warnings.append(f"slide {sid}: {field}=warn")
            elif value not in PASS_VALUES:
                issues.append(f"slide {sid}: {field}={slide.get(field)!r}")

        notes = slide.get("notes", [])
        if not isinstance(notes, list) or not notes:
            warnings.append(f"slide {sid}: missing review notes")

    if fail_on_warn and warnings:
        issues.extend(warnings)

    return {
        "review": str(review_path),
        "slide_count": len(slides),
        "status": "fail" if issues else ("warn" if warnings else "pass"),
        "issues": issues,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("review", type=Path)
    parser.add_argument("--min-slides", type=int, default=1)
    parser.add_argument("--fail-on-warn", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report = audit(args.review, args.min_slides, args.fail_on_warn)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)
    if report["status"] == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
