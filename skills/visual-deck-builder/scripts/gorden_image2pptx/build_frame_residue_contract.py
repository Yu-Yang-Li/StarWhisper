#!/usr/bin/env python3
"""Build a frame-residue audit contract from layout/icon ownership evidence.

This helper does not guess that all slides have the same icon colors. By
default it emits bbox-only regions, which the residue audit will skip. Pass one
or more ``--color-family`` values only for slide families where the extraction
plan explicitly says those movable decorations must not remain in the frame.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SLIDE_KEYS = {"background", "frame", "shapes", "icons", "texts"}
COLOR_FAMILIES = {"teal_green", "warm", "red_or_orange", "dark_saturated", "any_saturated"}


def _load_layout(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "slides" not in data:
        slide = {k: data[k] for k in SLIDE_KEYS if k in data}
        deck = {k: v for k, v in data.items() if k not in SLIDE_KEYS}
        deck["slides"] = [slide]
        data = deck
    return data


def _bbox_from_icon(deck: dict, icon: dict) -> list[float] | None:
    if isinstance(icon.get("source_bbox"), list) and len(icon["source_bbox"]) == 4:
        return [float(v) for v in icon["source_bbox"]]
    ref_w = float(deck.get("ref_width") or 0)
    ref_h = float(deck.get("ref_height") or 0)
    if not ref_w or not ref_h:
        return None
    try:
        return [
            float(icon["x"]) * ref_w,
            float(icon["y"]) * ref_h,
            float(icon["w"]) * ref_w,
            float(icon["h"]) * ref_h,
        ]
    except KeyError:
        return None


def _pad_box(box: list[float], padding: float, ref_w: float | None, ref_h: float | None) -> list[float]:
    x, y, w, h = box
    x0 = x - padding
    y0 = y - padding
    x1 = x + w + padding
    y1 = y + h + padding
    if ref_w:
        x0 = max(0.0, x0)
        x1 = min(ref_w, x1)
    if ref_h:
        y0 = max(0.0, y0)
        y1 = min(ref_h, y1)
    return [round(x0, 3), round(y0, 3), round(max(0.0, x1 - x0), 3), round(max(0.0, y1 - y0), 3)]


def _checks(color_families: list[str], args: argparse.Namespace) -> list[dict]:
    checks = []
    for color_family in color_families:
        if color_family not in COLOR_FAMILIES:
            raise ValueError(f"unsupported color family: {color_family}")
        checks.append({
            "name": f"no_{color_family}_movable_decor_residue_in_frame",
            "detector": "saturated_components",
            "color_family": color_family,
            "min_area": args.min_area,
            "min_side": args.min_side,
            "max_aspect": args.max_aspect,
            "min_alpha": args.min_alpha,
        })
    return checks


def _regions_from_expected(expected_path: Path, color_families: list[str], args: argparse.Namespace) -> list[dict]:
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    regions = []
    for region in expected.get("regions", []):
        box = region.get("bbox", region.get("source_bbox"))
        if not isinstance(box, list) or len(box) != 4:
            continue
        item = {
            "name": f"{region.get('name', 'region')}_frame_should_not_contain_movable_decor",
            "bbox": [float(v) for v in box],
            "source": "icon_coverage_expected",
        }
        checks = _checks(color_families, args)
        if checks:
            item["forbidden_residue"] = checks
        regions.append(item)
    return regions


def _regions_from_icons(layout_path: Path, color_families: list[str], args: argparse.Namespace) -> list[dict]:
    deck = _load_layout(layout_path)
    slides = deck.get("slides") or []
    slide_index = args.slide_index - 1
    if slide_index < 0 or slide_index >= len(slides):
        raise SystemExit(f"slide index out of range: {args.slide_index}")
    slide = slides[slide_index]
    ref_w = float(deck.get("ref_width") or 0) or None
    ref_h = float(deck.get("ref_height") or 0) or None
    checks = _checks(color_families, args)
    regions = []
    for index, icon in enumerate(slide.get("icons") or []):
        box = _bbox_from_icon(deck, icon)
        if box is None:
            continue
        name = str(icon.get("name") or icon.get("file") or f"icon_{index}")
        item = {
            "name": f"{name}_frame_should_not_contain_movable_decor",
            "bbox": _pad_box(box, args.padding, ref_w, ref_h),
            "source": "layout_icon_source_bbox",
        }
        if checks:
            item["forbidden_residue"] = checks
        regions.append(item)
    return regions


def build_contract(
    layout_path: Path,
    *,
    icon_coverage_expected: Path | None = None,
    slide_index: int = 1,
    color_family: list[str] | None = None,
    padding: float = 8.0,
    min_area: int = 2500,
    min_side: int = 40,
    max_aspect: float = 3.0,
    min_alpha: int = 50,
) -> dict:
    args = argparse.Namespace(
        slide_index=slide_index,
        color_family=color_family or [],
        padding=padding,
        min_area=min_area,
        min_side=min_side,
        max_aspect=max_aspect,
        min_alpha=min_alpha,
    )
    if icon_coverage_expected:
        regions = _regions_from_expected(icon_coverage_expected, args.color_family, args)
        source = str(icon_coverage_expected)
    else:
        regions = _regions_from_icons(layout_path, args.color_family, args)
        source = str(layout_path)
    return {
        "version": 1,
        "layout": str(layout_path),
        "source": source,
        "mode": "explicit_forbidden_residue" if args.color_family else "bbox_skeleton_review_required",
        "regions": regions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("layout", type=Path, help="layered layout JSON used for reconstruction")
    parser.add_argument("--icon-coverage-expected", type=Path, help="reuse broad expected icon/decor regions")
    parser.add_argument("--slide-index", type=int, default=1)
    parser.add_argument("--color-family", action="append", default=[], choices=sorted(COLOR_FAMILIES))
    parser.add_argument("--padding", type=float, default=8.0, help="source-pixel padding when deriving regions from icon boxes")
    parser.add_argument("--min-area", type=int, default=2500)
    parser.add_argument("--min-side", type=int, default=40)
    parser.add_argument("--max-aspect", type=float, default=3.0)
    parser.add_argument("--min-alpha", type=int, default=50)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    contract = build_contract(
        args.layout,
        icon_coverage_expected=args.icon_coverage_expected,
        slide_index=args.slide_index,
        color_family=args.color_family,
        padding=args.padding,
        min_area=args.min_area,
        min_side=args.min_side,
        max_aspect=args.max_aspect,
        min_alpha=args.min_alpha,
    )
    text = json.dumps(contract, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
