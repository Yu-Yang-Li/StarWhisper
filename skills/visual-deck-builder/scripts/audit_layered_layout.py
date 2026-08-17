#!/usr/bin/env python3
"""Audit layered_deck.json for dense-slide layout risks before delivery."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _box(item: dict) -> tuple[float, float, float, float] | None:
    try:
        x = float(item["x"])
        y = float(item["y"])
        w = float(item["w"])
        h = float(item["h"])
    except (KeyError, TypeError, ValueError):
        return None
    return x, y, w, h


def _area(box: tuple[float, float, float, float]) -> float:
    return max(box[2], 0.0) * max(box[3], 0.0)


def _intersection(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0 = max(ax, bx)
    y0 = max(ay, by)
    x1 = min(ax + aw, bx + bw)
    y1 = min(ay + ah, by + bh)
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _display_units(text: str) -> float:
    total = 0.0
    for char in text:
        if char.isspace():
            total += 0.3
        elif ord(char) > 127:
            total += 1.0
        else:
            total += 0.55
    return total


def _text_capacity_issue(text: dict, slide_w: float, slide_h: float) -> tuple[bool, dict]:
    box = _box(text)
    if not box:
        return True, {"message": "text item has invalid box"}
    _, _, w, h = box
    size = float(text.get("size") or text.get("font_size") or 14)
    content = str(text.get("text") or "")
    if not content.strip() or size <= 0:
        return False, {}

    box_w_in = w * slide_w
    box_h_in = h * slide_h
    char_w_in = max(size / 72.0 * 0.58, 0.01)
    line_capacity = max(box_w_in / char_w_in, 1.0)
    estimated_lines = max(1, math.ceil(_display_units(content) / line_capacity))
    required_h_in = estimated_lines * (size / 72.0) * 1.22
    cramped = box_h_in < required_h_in
    return cramped, {
        "message": "text box likely too small for editable text",
        "text": content,
        "font_size": size,
        "box_height_in": round(box_h_in, 3),
        "required_height_in": round(required_h_in, 3),
        "estimated_lines": estimated_lines,
    }


def audit(layered_path: Path) -> dict:
    deck = _load_json(layered_path)
    slide_w = float(deck.get("slide_width_in") or 13.333333)
    slide_h = float(deck.get("slide_height_in") or 7.5)
    issues: list[dict] = []
    slides_out: list[dict] = []

    for slide_index, slide in enumerate(deck.get("slides") or [], start=1):
        sid = str(slide.get("slide_id") or slide_index)
        texts = [item for item in (slide.get("texts") or []) if isinstance(item, dict)]
        icons = [item for item in (slide.get("icons") or []) if isinstance(item, dict)]
        slide_issues: list[dict] = []

        for index, text in enumerate(texts):
            box = _box(text)
            if not box:
                issue = {"level": "error", "slide_id": sid, "message": f"texts[{index}] has invalid box"}
                issues.append(issue)
                slide_issues.append(issue)
                continue
            x, y, w, h = box
            if min(x, y) < 0.025 or x + w > 0.975 or y + h > 0.975:
                issue = {"level": "warning", "slide_id": sid, "message": f"texts[{index}] is close to slide edge"}
                issues.append(issue)
                slide_issues.append(issue)
            cramped, details = _text_capacity_issue(text, slide_w, slide_h)
            if cramped:
                issue = {"level": "warning", "slide_id": sid, "message": details["message"], "details": details}
                issues.append(issue)
                slide_issues.append(issue)

        text_boxes = [(_box(text), index, str(text.get("text") or "")) for index, text in enumerate(texts)]
        text_boxes = [(box, index, label) for box, index, label in text_boxes if box]
        for left_pos, (left_box, left_index, left_label) in enumerate(text_boxes):
            for right_box, right_index, right_label in text_boxes[left_pos + 1:]:
                overlap = _intersection(left_box, right_box)
                if overlap <= 0:
                    continue
                ratio = overlap / max(min(_area(left_box), _area(right_box)), 1e-9)
                if ratio > 0.08:
                    issue = {
                        "level": "error",
                        "slide_id": sid,
                        "message": f"texts[{left_index}] overlaps texts[{right_index}]",
                        "details": {"overlap_ratio": round(ratio, 3), "left": left_label, "right": right_label},
                    }
                    issues.append(issue)
                    slide_issues.append(issue)

        icon_boxes = [(_box(icon), index) for index, icon in enumerate(icons)]
        icon_boxes = [(box, index) for box, index in icon_boxes if box]
        for text_box, text_index, text_label in text_boxes:
            for icon_box, icon_index in icon_boxes:
                overlap = _intersection(text_box, icon_box)
                if overlap <= 0:
                    continue
                ratio = overlap / max(min(_area(text_box), _area(icon_box)), 1e-9)
                if ratio > 0.05:
                    issue = {
                        "level": "warning",
                        "slide_id": sid,
                        "message": f"icons[{icon_index}] crowds texts[{text_index}]",
                        "details": {"overlap_ratio": round(ratio, 3), "text": text_label},
                    }
                    issues.append(issue)
                    slide_issues.append(issue)

        slides_out.append({
            "slide_id": sid,
            "text_count": len(texts),
            "icon_count": len(icons),
            "issue_count": len(slide_issues),
        })

    status = "pass"
    if any(issue["level"] == "error" for issue in issues):
        status = "fail"
    elif issues:
        status = "warn"

    return {
        "status": status,
        "layered_deck": str(layered_path),
        "slide_count": len(deck.get("slides") or []),
        "issue_count": len(issues),
        "issues": issues,
        "slides": slides_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("layered_deck", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--fail-on-warn", action="store_true")
    args = parser.parse_args()

    report = audit(args.layered_deck)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)
    if report["status"] == "fail" or (args.fail_on_warn and report["status"] == "warn"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
