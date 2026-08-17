#!/usr/bin/env python3
"""Reject visual_style_contract.json / slide_spec.json that are too vague to
keep independent per-slide image generation calls visually consistent.

This is the automated substitute for "the Agent happened to write a good
contract this time": a deck cannot pass release QA on a style contract that
is just a short adjective-style label, or on slides that only carry a
freeform page_description with no structured design fields.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")
MIN_RICH_FIELD_CHARS = 8
REQUIRED_PALETTE_ROLES = ["background", "primary", "accent", "neutral"]
REQUIRED_RICHNESS_FIELDS = [
    "mood_and_lighting",
    "accent_usage_limit",
    "typography",
    "grid_and_spacing",
    "icon_style",
    "chart_rules",
    "rendering_descriptor",
]
REQUIRED_EXTRA_FIELDS = ["visual_elements", "visual_focus", "layout_notes"]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _text_len(value: Any) -> int:
    if isinstance(value, str):
        return len(value.strip())
    if isinstance(value, dict):
        return sum(_text_len(v) for v in value.values())
    if isinstance(value, list):
        return sum(_text_len(v) for v in value)
    return 0


def _extract_hex_values(palette: dict) -> dict[str, str | None]:
    found = {}
    for role in REQUIRED_PALETTE_ROLES:
        entry = palette.get(role)
        if isinstance(entry, dict):
            found[role] = entry.get("hex")
        elif isinstance(entry, str):
            match = re.search(r"#[0-9A-Fa-f]{6}", entry)
            found[role] = match.group(0) if match else None
        else:
            found[role] = None
    return found


def validate_contract(contract: dict, issues: list[dict]) -> None:
    palette = contract.get("palette")
    if not isinstance(palette, dict):
        issues.append({"level": "error", "message": "visual_style_contract.palette is missing or not an object"})
        palette = {}

    hex_values = _extract_hex_values(palette)
    for role, hex_value in hex_values.items():
        if role not in palette:
            issues.append({"level": "error", "message": f"palette missing role: {role}"})
            continue
        if not hex_value or not HEX_RE.match(hex_value):
            issues.append({
                "level": "error",
                "message": f"palette.{role} has no explicit hex color (found: {hex_value!r}); "
                           "a bare color word like '深蓝' is not enough to keep pages consistent",
            })

    for field in REQUIRED_RICHNESS_FIELDS:
        value = contract.get(field)
        if _text_len(value) < MIN_RICH_FIELD_CHARS:
            issues.append({
                "level": "error",
                "message": f"visual_style_contract.{field} is missing or too short to be a real constraint",
            })

    typography = contract.get("typography")
    if isinstance(typography, dict):
        if not typography.get("cjk_font"):
            issues.append({"level": "error", "message": "typography.cjk_font is missing an explicit font family name"})
        if not typography.get("latin_font"):
            issues.append({"level": "warning", "message": "typography.latin_font is missing an explicit font family name"})
    elif isinstance(typography, str):
        issues.append({
            "level": "warning",
            "message": "typography is a freeform string, not a structured object with cjk_font/latin_font; "
                       "prefer scripts/build_style_contract.py output",
        })

    style_source = str(contract.get("style_source") or "")
    if style_source not in {"preset_library", "template_image", "extracted_style", "user_instruction", "inferred_default"}:
        issues.append({"level": "warning", "message": f"unrecognized style_source: {style_source!r}"})
    if style_source == "inferred_default":
        issues.append({
            "level": "warning",
            "message": "style_source is inferred_default; prefer scripts/build_style_contract.py --preset "
                       "so the contract carries the full preset-library precision instead of an ad hoc guess",
        })


def validate_slide_extra_fields(spec: dict, issues: list[dict]) -> None:
    slides = spec.get("slides") or []
    for slide in slides:
        sid = str(slide.get("slide_id") or "?")
        extra = slide.get("extra_fields")
        if not isinstance(extra, dict) or not extra:
            issues.append({
                "level": "warning",
                "slide_id": sid,
                "message": "slide has no extra_fields (visual_elements/visual_focus/layout_notes); "
                           "falling back to freeform page_description only",
            })
            continue
        for field in REQUIRED_EXTRA_FIELDS:
            if _text_len(extra.get(field)) < MIN_RICH_FIELD_CHARS:
                issues.append({
                    "level": "error",
                    "slide_id": sid,
                    "message": f"extra_fields.{field} is missing or too short",
                })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path, help="Path to visual_style_contract.json")
    parser.add_argument("--slide-spec", type=Path, help="Optional slide_spec.json to also check per-slide extra_fields")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    issues: list[dict] = []
    contract = _load_json(args.contract)
    validate_contract(contract, issues)

    if args.slide_spec:
        spec = _load_json(args.slide_spec)
        validate_slide_extra_fields(spec, issues)

    status = "pass"
    if any(item["level"] == "error" for item in issues):
        status = "fail"
    elif issues:
        status = "warn"

    report = {
        "schema": "visual_deck_style_richness_v1",
        "status": status,
        "issue_count": len(issues),
        "issues": issues,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 1 if status == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
