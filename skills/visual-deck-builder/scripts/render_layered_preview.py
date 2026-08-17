#!/usr/bin/env python3
"""Render QA previews from visual-deck-builder layered_deck.json.

This is a QA renderer only. It must not be used as the source of PPT layer
assets or as a replacement for image-model-generated background/frame/icons.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _resolve(base: Path, maybe_path: str | None) -> Path | None:
    if not maybe_path:
        return None
    p = Path(maybe_path)
    if not p.is_absolute():
        p = base / p
    return p


def _box(item: dict, width: int, height: int) -> tuple[int, int, int, int]:
    x = float(item["x"])
    y = float(item["y"])
    w = float(item["w"])
    h = float(item["h"])
    return int(x * width), int(y * height), int(w * width), int(h * height)


def _font(size_px: int, bold: bool = False, font_name: str | None = None) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if font_name:
        candidates.append(font_name)
    candidates.extend([
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ])
    for candidate in candidates:
        try:
            if Path(candidate).exists():
                return ImageFont.truetype(candidate, size_px)
        except Exception:
            continue
    return ImageFont.load_default()


def _rgb(value: str) -> tuple[int, int, int]:
    value = (value or "111111").strip().lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    if len(value) != 6:
        value = "111111"
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _draw_text(draw: ImageDraw.ImageDraw, item: dict, width: int, height: int, show_boxes: bool) -> None:
    x, y, w, h = _box(item, width, height)
    size_pt = float(item.get("size", item.get("font_size", 18)))
    size_px = max(8, int(size_pt * height / 540))
    font = _font(size_px, bool(item.get("bold", False)), item.get("font"))
    color = _rgb(str(item.get("color", "111111")))
    if show_boxes:
        draw.rectangle((x, y, x + w, y + h), outline=(220, 40, 40), width=2)
    text = str(item.get("text") or "")
    if not text:
        return
    chars_per_line = max(1, int(w / max(size_px * 0.62, 1)))
    lines = []
    for raw_line in text.splitlines() or [text]:
        lines.extend(textwrap.wrap(raw_line, width=chars_per_line) or [""])
    line_gap = int(size_px * 1.18)
    yy = y
    for line in lines:
        if yy + line_gap > y + h + line_gap:
            break
        draw.text((x, yy), line, fill=color, font=font)
        yy += line_gap


def render(deck_path: Path, out_dir: Path, width: int = 1600, show_boxes: bool = False) -> list[Path]:
    deck = json.loads(deck_path.read_text(encoding="utf-8"))
    base = Path(deck.get("assets_dir") or deck_path.parent)
    if not base.is_absolute():
        base = deck_path.parent / base
    slide_w = float(deck.get("slide_width_in", 13.333333))
    slide_h = float(deck.get("slide_height_in", 7.5))
    height = int(round(width * slide_h / slide_w))
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    for index, slide in enumerate(deck.get("slides") or [], start=1):
        bg_path = _resolve(base, slide.get("background"))
        if not bg_path or not bg_path.exists():
            raise SystemExit(f"slide {index}: missing background: {bg_path}")
        canvas = Image.open(bg_path).convert("RGBA").resize((width, height), Image.Resampling.LANCZOS)

        frame_path = _resolve(base, slide.get("frame"))
        if frame_path:
            if not frame_path.exists():
                raise SystemExit(f"slide {index}: missing frame: {frame_path}")
            frame = Image.open(frame_path).convert("RGBA").resize((width, height), Image.Resampling.LANCZOS)
            canvas.alpha_composite(frame)

        for icon in slide.get("icons") or []:
            icon_path = _resolve(base, icon.get("file"))
            if not icon_path or not icon_path.exists():
                raise SystemExit(f"slide {index}: missing icon: {icon_path}")
            x, y, w, h = _box(icon, width, height)
            img = Image.open(icon_path).convert("RGBA").resize((w, h), Image.Resampling.LANCZOS)
            canvas.alpha_composite(img, (x, y))

        draw = ImageDraw.Draw(canvas)
        for item in slide.get("texts") or []:
            _draw_text(draw, item, width, height, show_boxes)

        sid = str(slide.get("slide_id") or f"{index:02d}")
        out = out_dir / f"slide-{sid}.png"
        canvas.convert("RGB").save(out)
        outputs.append(out)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("deck", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--show-boxes", action="store_true")
    args = parser.parse_args()
    for out in render(args.deck, args.out_dir, args.width, args.show_boxes):
        print(out)


if __name__ == "__main__":
    main()
