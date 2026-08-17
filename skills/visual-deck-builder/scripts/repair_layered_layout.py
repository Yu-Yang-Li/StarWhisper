#!/usr/bin/env python3
"""Repair simple layered_deck.json placement issues without changing text content."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from audit_layered_layout import _area, _box, _intersection, audit


def _overlap_ratio(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    return _intersection(a, b) / max(min(_area(a), _area(b)), 1e-9)


def _inside(box: tuple[float, float, float, float]) -> bool:
    x, y, w, h = box
    return w > 0 and h > 0 and x >= 0 and y >= 0 and x + w <= 1 and y + h <= 1


def _collides_with_text(box: tuple[float, float, float, float], texts: list[dict], threshold: float = 0.05) -> bool:
    for text in texts:
        text_box = _box(text)
        if text_box and _overlap_ratio(box, text_box) > threshold:
            return True
    return False


def _candidate_icon_boxes(icon: dict, text: dict, gap: float) -> list[tuple[float, float, float, float, str]]:
    icon_box = _box(icon)
    text_box = _box(text)
    if not icon_box or not text_box:
        return []
    ix, iy, iw, ih = icon_box
    tx, ty, tw, th = text_box
    y_centered = ty + max(0.0, (th - ih) / 2)
    x_centered = tx + max(0.0, (tw - iw) / 2)
    return [
        (tx + tw + gap, y_centered, iw, ih, "right_of_text"),
        (ix + iw + gap, iy, iw, ih, "nudge_right"),
        (tx - iw - gap, y_centered, iw, ih, "left_of_text"),
        (x_centered, ty - ih - gap, iw, ih, "above_text"),
        (x_centered, ty + th + gap, iw, ih, "below_text"),
    ]


def repair(deck_path: Path, out_path: Path, gap: float = 0.01, max_passes: int = 4) -> dict:
    original = json.loads(deck_path.read_text(encoding="utf-8"))
    deck = copy.deepcopy(original)
    changes: list[dict] = []

    for _ in range(max_passes):
        changed_this_pass = False
        for slide in deck.get("slides") or []:
            sid = str(slide.get("slide_id") or "")
            texts = [item for item in (slide.get("texts") or []) if isinstance(item, dict)]
            icons = [item for item in (slide.get("icons") or []) if isinstance(item, dict)]
            for icon_index, icon in enumerate(icons):
                icon_box = _box(icon)
                if not icon_box:
                    continue
                crowded_texts = []
                for text_index, text in enumerate(texts):
                    text_box = _box(text)
                    if text_box and _overlap_ratio(icon_box, text_box) > 0.05:
                        crowded_texts.append((text_index, text))
                if not crowded_texts:
                    continue

                candidate_boxes: list[tuple[float, float, float, float, str]] = []
                for _, text in crowded_texts:
                    candidate_boxes.extend(_candidate_icon_boxes(icon, text, gap))
                for x, y, w, h, reason in candidate_boxes:
                    candidate = (x, y, w, h)
                    if not _inside(candidate):
                        continue
                    if _collides_with_text(candidate, texts):
                        continue
                    before = {"x": icon.get("x"), "y": icon.get("y"), "w": icon.get("w"), "h": icon.get("h")}
                    icon["x"], icon["y"], icon["w"], icon["h"] = x, y, w, h
                    changes.append({
                        "slide_id": sid,
                        "item": f"icons[{icon_index}]",
                        "reason": reason,
                        "before": before,
                        "after": {"x": x, "y": y, "w": w, "h": h},
                    })
                    changed_this_pass = True
                    break
        if not changed_this_pass:
            break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(deck, ensure_ascii=False, indent=2), encoding="utf-8")
    after = audit(out_path)
    return {
        "status": after["status"],
        "source": str(deck_path),
        "repaired": str(out_path),
        "change_count": len(changes),
        "changes": changes,
        "post_repair_audit": after,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("layered_deck", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--gap", type=float, default=0.01)
    args = parser.parse_args()

    report = repair(args.layered_deck, args.out, args.gap)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text)
    if report["status"] == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
