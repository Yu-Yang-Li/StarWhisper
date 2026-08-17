#!/usr/bin/env python3
"""Build a precise visual_style_contract.json from the built-in preset library.

Replaces free-form, one-line style descriptions ("clean, modern, blue and
white") with a contract that carries exact hex colors, named fonts, spacing
and accent-usage percentages. This is what actually keeps independent
per-slide text-to-image calls visually consistent; vague adjectives leave the
image model to reinterpret "deep blue" differently on every call.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from style_presets import DEFAULT_PRESET_KEY, get_preset, list_presets


def build_contract(preset_key: str, resolution: str, extra_forbidden: list[str] | None = None) -> dict[str, Any]:
    preset = get_preset(preset_key)
    contract = {
        "style_source": "preset_library",
        "preset_key": preset_key,
        **preset,
        "resolution": resolution,
    }
    if extra_forbidden:
        merged = list(dict.fromkeys(list(contract.get("forbidden_patterns", [])) + list(extra_forbidden)))
        contract["forbidden_patterns"] = merged
    return contract


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        default=DEFAULT_PRESET_KEY,
        choices=list_presets(),
        help=f"Style preset key. Default: {DEFAULT_PRESET_KEY}.",
    )
    parser.add_argument(
        "--resolution",
        default="默认 1K；仅在用户明确要求高清重生成时使用 2K",
        help="Resolution policy string recorded in the contract.",
    )
    parser.add_argument("--extra-forbidden", nargs="*", default=[], help="Additional forbidden visual/text patterns to merge in.")
    parser.add_argument("--out", type=Path, help="Output path for visual_style_contract.json. Prints to stdout if omitted.")
    parser.add_argument("--list", action="store_true", help="List available preset keys and exit.")
    args = parser.parse_args()

    if args.list:
        for key in list_presets():
            print(key)
        return 0

    contract = build_contract(args.preset, args.resolution, args.extra_forbidden)
    text = json.dumps(contract, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
