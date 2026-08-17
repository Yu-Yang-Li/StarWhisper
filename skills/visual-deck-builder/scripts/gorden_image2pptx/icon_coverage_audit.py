#!/usr/bin/env python3
"""Audit icon/decor coverage against expected source-image regions.

The reconstruction path can look superficially editable while missing dense
micro-icons or entire status regions. This audit checks icon source boxes
against an explicit region contract so coverage failures are reproducible.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path


def _load_layout(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "slides" not in data:
        slide_keys = {"background", "frame", "shapes", "icons", "texts"}
        slide = {k: data[k] for k in slide_keys if k in data}
        deck = {k: v for k, v in data.items() if k not in slide_keys}
        deck["slides"] = [slide]
        data = deck
    return data


def _bbox_from_icon(deck: dict, icon: dict) -> list[float] | None:
    if icon.get("source_bbox") and len(icon["source_bbox"]) == 4:
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


def _intersect_area(a: list[float], b: list[float]) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    x0, y0 = max(ax0, bx0), max(ay0, by0)
    x1, y1 = min(ax1, bx1), min(ay1, by1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def _center_in(box: list[float], region: list[float]) -> bool:
    x, y, w, h = box
    rx, ry, rw, rh = region
    cx, cy = x + w / 2, y + h / 2
    return rx <= cx <= rx + rw and ry <= cy <= ry + rh


def _matches_region(icon_box: list[float], region_box: list[float], min_ratio: float, require_center: bool) -> bool:
    if require_center and _center_in(icon_box, region_box):
        return True
    icon_area = max(1.0, icon_box[2] * icon_box[3])
    return (_intersect_area(icon_box, region_box) / icon_area) >= min_ratio


def _name_matches(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def audit(layout_path: Path, expected_path: Path, fail_on_warn: bool = False) -> dict:
    deck = _load_layout(layout_path)
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    slides = deck.get("slides") or []
    slide_index = int(expected.get("slide_index", 1)) - 1
    if slide_index < 0 or slide_index >= len(slides):
        return {
            "status": "fail",
            "layout": str(layout_path),
            "expected": str(expected_path),
            "issues": [{"level": "error", "message": f"slide_index out of range: {slide_index + 1}"}],
            "regions": [],
        }

    icons = slides[slide_index].get("icons") or []
    icon_records = []
    issues = []
    for index, icon in enumerate(icons):
        box = _bbox_from_icon(deck, icon)
        if box is None:
            issues.append({"level": "error", "message": f"icons[{index}] lacks source_bbox and cannot derive source box"})
            continue
        icon_records.append({
            "index": index,
            "name": str(icon.get("name") or icon.get("file") or f"icon_{index}"),
            "bbox": box,
        })

    overall_min = expected.get("overall_min_icons")
    if overall_min is not None and len(icon_records) < int(overall_min):
        issues.append({
            "level": "error",
            "message": "overall icon count below expected minimum",
            "details": {"actual": len(icon_records), "minimum": int(overall_min)},
        })

    region_reports = []
    assigned = set()
    for region in expected.get("regions", []):
        name = str(region["name"])
        region_box = [float(v) for v in region.get("bbox", region.get("source_bbox", []))]
        if len(region_box) != 4:
            issues.append({"level": "error", "region": name, "message": "region missing bbox"})
            continue
        min_ratio = float(region.get("min_intersection_ratio", expected.get("min_intersection_ratio", 0.15)))
        require_center = bool(region.get("require_center", expected.get("require_center", True)))
        matches = [
            item for item in icon_records
            if _matches_region(item["bbox"], region_box, min_ratio, require_center)
        ]
        for item in matches:
            assigned.add(item["index"])

        min_icons = int(region.get("min_icons", 0))
        if len(matches) < min_icons:
            issues.append({
                "level": "error",
                "region": name,
                "message": "region icon count below expected minimum",
                "details": {"actual": len(matches), "minimum": min_icons},
            })

        names = [item["name"] for item in matches]
        for required in region.get("required_names", []):
            patterns = required if isinstance(required, list) else [str(required)]
            if not any(_name_matches(icon_name, patterns) for icon_name in names):
                issues.append({
                    "level": "error",
                    "region": name,
                    "message": "required icon name missing",
                    "details": {"required": patterns},
                })

        for recommended in region.get("recommended_names", []):
            patterns = recommended if isinstance(recommended, list) else [str(recommended)]
            if not any(_name_matches(icon_name, patterns) for icon_name in names):
                issues.append({
                    "level": "warning",
                    "region": name,
                    "message": "recommended icon name missing",
                    "details": {"recommended": patterns},
                })

        region_reports.append({
            "name": name,
            "bbox": region_box,
            "icon_count": len(matches),
            "icons": names,
        })

    max_unassigned = expected.get("max_unassigned_icons")
    unassigned = [item for item in icon_records if item["index"] not in assigned]
    if max_unassigned is not None and len(unassigned) > int(max_unassigned):
        issues.append({
            "level": "warning",
            "message": "more icons outside expected regions than allowed",
            "details": {"actual": len(unassigned), "maximum": int(max_unassigned)},
        })

    has_error = any(issue["level"] == "error" for issue in issues)
    has_warning = any(issue["level"] == "warning" for issue in issues)
    status = "fail" if has_error or (fail_on_warn and has_warning) else ("warn" if has_warning else "pass")
    return {
        "status": status,
        "layout": str(layout_path),
        "expected": str(expected_path),
        "slide_index": slide_index + 1,
        "icon_count": len(icon_records),
        "issue_count": len(issues),
        "issues": issues,
        "regions": region_reports,
        "unassigned_icons": [item["name"] for item in unassigned],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("layout", type=Path)
    parser.add_argument("expected", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--fail-on-warn", action="store_true")
    args = parser.parse_args()

    report = audit(args.layout, args.expected, fail_on_warn=args.fail_on_warn)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)
    if report["status"] == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
