#!/usr/bin/env python3
"""Audit frame layers for explicitly forbidden icon/decor residue.

Frame extraction should keep structure and remove ordinary icons. Dense slides
can pass icon coverage while still failing visually because the frame layer
contains icon bases, badges, or other movable decor residue. This audit is
contract-driven: it only scans regions that explicitly declare
``forbidden_residue`` checks. A generic bbox alone is not a rule.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path


SLIDE_KEYS = {"background", "frame", "shapes", "icons", "texts"}


def _load_deck(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "slides" not in data:
        slide = {k: data[k] for k in SLIDE_KEYS if k in data}
        deck = {k: v for k, v in data.items() if k not in SLIDE_KEYS}
        deck["slides"] = [slide]
        data = deck
    return data


def _resolve(assets_dir: Path, file_name: str) -> Path:
    path = Path(file_name)
    return path if path.is_absolute() else assets_dir / path


def _region_box(region: dict) -> list[float]:
    box = region.get("bbox", region.get("source_bbox"))
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError(f"region missing bbox: {region.get('name', '<unnamed>')}")
    return [float(v) for v in box]


def _scale_box(box: list[float], frame_w: int, frame_h: int, ref_w: float, ref_h: float) -> tuple[int, int, int, int]:
    x, y, w, h = box
    sx = frame_w / ref_w
    sy = frame_h / ref_h
    return (
        max(0, int(round(x * sx))),
        max(0, int(round(y * sy))),
        min(frame_w, int(round((x + w) * sx))),
        min(frame_h, int(round((y + h) * sy))),
    )


def _is_saturated_residue(pixel: tuple[int, int, int, int], min_alpha: int, color_family: str) -> bool:
    r, g, b, a = pixel
    if a < min_alpha:
        return False
    mx = max(r, g, b)
    mn = min(r, g, b)
    saturation = mx - mn
    if color_family == "teal_green":
        # Typical corporate teal/green icon bases and status badges.
        return g > 70 and b > 55 and r < 105 and (g - r) > 20 and (b - r) > 15
    if color_family == "warm":
        return r > 115 and g > 55 and b < 110 and (r - b) > 35
    if color_family == "red_or_orange":
        return r > 125 and (r - g) > 20 and (r - b) > 35
    if color_family == "dark_saturated":
        return mx < 150 and saturation > 40 and a >= min_alpha
    if color_family == "any_saturated":
        return saturation > 55 and mx > 70 and mn < 180
    raise ValueError(f"unsupported color_family: {color_family}")


def _components(img, box: tuple[int, int, int, int], min_alpha: int, color_family: str) -> list[dict]:
    px = img.load()
    x0, y0, x1, y1 = box
    seen: set[tuple[int, int]] = set()
    found: list[dict] = []

    for y in range(y0, y1):
        for x in range(x0, x1):
            if (x, y) in seen or not _is_saturated_residue(px[x, y], min_alpha, color_family):
                continue
            q: deque[tuple[int, int]] = deque([(x, y)])
            seen.add((x, y))
            xs: list[int] = []
            ys: list[int] = []
            while q:
                cx, cy = q.popleft()
                xs.append(cx)
                ys.append(cy)
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if (
                        x0 <= nx < x1
                        and y0 <= ny < y1
                        and (nx, ny) not in seen
                        and _is_saturated_residue(px[nx, ny], min_alpha, color_family)
                    ):
                        seen.add((nx, ny))
                        q.append((nx, ny))
            bbox = [min(xs), min(ys), max(xs) + 1, max(ys) + 1]
            found.append({
                "area": len(xs),
                "bbox_px": bbox,
                "width_px": bbox[2] - bbox[0],
                "height_px": bbox[3] - bbox[1],
            })
    return found


def audit(
    layout_path: Path,
    regions_path: Path,
    *,
    slide_index: int = 1,
    min_area: int = 2500,
    min_side: int = 40,
    max_aspect: float = 3.0,
    min_alpha: int = 50,
) -> dict:
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("Pillow is required (pip install pillow)") from exc

    deck = _load_deck(layout_path)
    slides = deck.get("slides") or []
    if slide_index < 1 or slide_index > len(slides):
        raise SystemExit(f"slide index out of range: {slide_index}")
    slide = slides[slide_index - 1]
    assets_dir = Path(deck.get("assets_dir") or layout_path.parent)
    frame_file = slide.get("frame")
    if not frame_file:
        return {
            "status": "fail",
            "layout": str(layout_path),
            "issues": [{"level": "error", "message": "slide has no frame layer"}],
            "regions": [],
        }

    frame_path = _resolve(assets_dir, str(frame_file))
    with Image.open(frame_path) as image:
        frame = image.convert("RGBA")
    frame_w, frame_h = frame.size
    ref_w = float(deck.get("ref_width") or frame_w)
    ref_h = float(deck.get("ref_height") or frame_h)

    spec = json.loads(regions_path.read_text(encoding="utf-8"))
    regions = spec.get("regions", spec if isinstance(spec, list) else [])
    issues: list[dict] = []
    reports: list[dict] = []

    for region in regions:
        name = str(region.get("name", "region"))
        source_box = _region_box(region)
        pixel_box = _scale_box(source_box, frame_w, frame_h, ref_w, ref_h)
        checks = region.get("forbidden_residue") or spec.get("default_forbidden_residue") or []
        if isinstance(checks, dict):
            checks = [checks]

        check_reports = []
        suspicious_total = 0
        for check in checks:
            detector = str(check.get("detector", "saturated_components"))
            if detector != "saturated_components":
                raise ValueError(f"unsupported detector in {name}: {detector}")
            check_min_alpha = int(check.get("min_alpha", min_alpha))
            check_min_area = int(check.get("min_area", min_area))
            check_min_side = int(check.get("min_side", min_side))
            check_max_aspect = float(check.get("max_aspect", max_aspect))
            color_family = str(check.get("color_family", "teal_green"))
            components = _components(frame, pixel_box, min_alpha=check_min_alpha, color_family=color_family)
            suspicious = []
            for comp in components:
                w = comp["width_px"]
                h = comp["height_px"]
                aspect = max(w / max(h, 1), h / max(w, 1))
                if comp["area"] >= check_min_area and min(w, h) >= check_min_side and aspect <= check_max_aspect:
                    suspicious.append(comp)
            suspicious_total += len(suspicious)
            check_report = {
                "name": str(check.get("name", detector)),
                "detector": detector,
                "color_family": color_family,
                "component_count": len(components),
                "suspicious_count": len(suspicious),
                "suspicious_components": suspicious,
                "thresholds": {
                    "min_area": check_min_area,
                    "min_side": check_min_side,
                    "max_aspect": check_max_aspect,
                    "min_alpha": check_min_alpha,
                },
            }
            check_reports.append(check_report)
            if suspicious:
                issues.append({
                    "level": "error",
                    "region": name,
                    "message": "frame contains forbidden icon/decor residue",
                    "details": check_report,
                })

        reports.append({
            "name": name,
            "source_bbox": source_box,
            "frame_bbox_px": list(pixel_box),
            "check_count": len(check_reports),
            "skipped": not bool(check_reports),
            "suspicious_count": suspicious_total,
            "checks": check_reports,
        })

    status = "fail" if any(issue["level"] == "error" for issue in issues) else "pass"
    return {
        "status": status,
        "layout": str(layout_path),
        "frame": str(frame_path),
        "slide_index": slide_index,
        "frame_size": [frame_w, frame_h],
        "issues": issues,
        "regions": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("layout", type=Path)
    parser.add_argument(
        "regions",
        type=Path,
        help="JSON list or {regions:[...]} with source-coordinate bboxes and explicit forbidden_residue checks.",
    )
    parser.add_argument("--slide-index", type=int, default=1)
    parser.add_argument("--min-area", type=int, default=2500)
    parser.add_argument("--min-side", type=int, default=40)
    parser.add_argument("--max-aspect", type=float, default=3.0)
    parser.add_argument("--min-alpha", type=int, default=50)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report = audit(
        args.layout,
        args.regions,
        slide_index=args.slide_index,
        min_area=args.min_area,
        min_side=args.min_side,
        max_aspect=args.max_aspect,
        min_alpha=args.min_alpha,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)
    if report["status"] == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
