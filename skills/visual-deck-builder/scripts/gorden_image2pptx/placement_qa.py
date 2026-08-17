#!/usr/bin/env python3
"""Draw placement QA overlays for Image2PPTX layouts.

Use after writing layout.json/deck.json and rendering previews. The script draws
the same icon/text boxes on the source slide and on the composed preview so
placement errors are visible before handoff.

Usage:
    python3 scripts/placement_qa.py source.png layout.json --slide-index 1 \
      --preview out/preview/slide_01.png --out-dir qa/placement
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _die(msg: str, code: int = 2):
    import sys
    print(f"Error: {msg}", file=sys.stderr)
    raise SystemExit(code)


def _load_deck(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if "slides" not in data:
        slide_keys = {"background", "frame", "shapes", "icons", "texts"}
        slide = {k: data[k] for k in slide_keys if k in data}
        deck = {k: v for k, v in data.items() if k not in slide_keys}
        deck["slides"] = [slide]
        data = deck
    data.setdefault("units", "fraction")
    return data


def _box(deck, item, width: int, height: int):
    units = deck.get("units", "fraction")
    if units == "px":
        ref_w = float(deck.get("ref_width") or width)
        ref_h = float(deck.get("ref_height") or height)
        x = float(item["x"]) / ref_w * width
        y = float(item["y"]) / ref_h * height
        w = float(item["w"]) / ref_w * width
        h = float(item["h"]) / ref_h * height
    else:
        x = float(item["x"]) * width
        y = float(item["y"]) * height
        w = float(item["w"]) * width
        h = float(item["h"]) * height
    return int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))


def _label_text(item, fallback: str):
    if item.get("file"):
        return Path(str(item["file"])).name
    text = str(item.get("text", fallback)).replace("\n", " ")
    return text[:18] + ("..." if len(text) > 18 else "")


def _draw_boxes(image, deck, slide, title: str):
    from PIL import ImageDraw, ImageFont

    out = image.convert("RGBA")
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 15)
    except Exception:
        font = ImageFont.load_default()

    # Label background.
    draw.rectangle([0, 0, min(out.width, 520), 32], fill=(0, 0, 0, 170))
    draw.text((10, 7), title, fill=(255, 255, 255, 255), font=font)

    for idx, item in enumerate(slide.get("icons", []), 1):
        x0, y0, x1, y1 = _box(deck, item, out.width, out.height)
        draw.rectangle([x0, y0, x1, y1], outline=(255, 202, 40, 255), width=3)
        draw.text((x0 + 3, max(34, y0 + 3)), f"I{idx} {_label_text(item, '')}",
                  fill=(255, 202, 40, 255), font=font)

    for idx, item in enumerate(slide.get("texts", []), 1):
        x0, y0, x1, y1 = _box(deck, item, out.width, out.height)
        draw.rectangle([x0, y0, x1, y1], outline=(0, 210, 255, 255), width=2)
        draw.text((x0 + 3, max(34, y0 + 3)), f"T{idx} {_label_text(item, '')}",
                  fill=(0, 210, 255, 255), font=font)
    return out.convert("RGB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", help="Source slide image.")
    ap.add_argument("layout", help="layout.json / deck.json.")
    ap.add_argument("--slide-index", type=int, default=1, help="1-based slide index in deck.")
    ap.add_argument("--preview", help="Optional composed preview PNG.")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    try:
        from PIL import Image
    except ImportError:
        _die("Pillow is required (pip install pillow)")

    source = Path(args.source)
    layout = Path(args.layout)
    out_dir = Path(args.out_dir)
    if not source.exists():
        _die(f"source not found: {source}")
    if not layout.exists():
        _die(f"layout not found: {layout}")
    deck = _load_deck(layout)
    if args.slide_index < 1 or args.slide_index > len(deck["slides"]):
        _die(f"--slide-index out of range: {args.slide_index}")
    slide = deck["slides"][args.slide_index - 1]

    out_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as im:
        source_qa = _draw_boxes(im, deck, slide, f"source boxes | slide {args.slide_index:02d}")
    source_out = out_dir / f"slide_{args.slide_index:02d}_source_boxes.png"
    source_qa.save(source_out)
    print(f"Wrote {source_out}")

    if args.preview:
        preview = Path(args.preview)
        if not preview.exists():
            _die(f"preview not found: {preview}")
        with Image.open(preview) as im:
            preview_qa = _draw_boxes(im, deck, slide, f"preview boxes | slide {args.slide_index:02d}")
        preview_out = out_dir / f"slide_{args.slide_index:02d}_preview_boxes.png"
        preview_qa.save(preview_out)
        print(f"Wrote {preview_out}")

        # Resize source to preview size and produce a quick blend overlay.
        with Image.open(source) as src, Image.open(preview) as prv:
            src = src.convert("RGB").resize(prv.size)
            blend = Image.blend(src, prv.convert("RGB"), 0.5)
        overlay_out = out_dir / f"slide_{args.slide_index:02d}_source_preview_blend.png"
        blend.save(overlay_out)
        print(f"Wrote {overlay_out}")


if __name__ == "__main__":
    main()
