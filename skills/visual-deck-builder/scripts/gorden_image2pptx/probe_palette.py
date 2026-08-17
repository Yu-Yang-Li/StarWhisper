#!/usr/bin/env python3
"""Probe a slide image and recommend a safe chroma-key color for icon extraction.

Why: in Part 2 we extract icons onto a flat chroma-key canvas and then remove that
color with `remove_chroma_key.py`. If the slide itself contains the key color
(e.g. a green slide keyed on green), the icons get eaten by the matte. This tool
samples the slide, reports which candidate key colors collide with the artwork,
and prints the first safe one.

Usage:
    python3 scripts/probe_palette.py SLIDE.png
    python3 scripts/probe_palette.py SLIDE.png --json

Output (human): a recommendation line `recommended_key: #ff00ff` plus per-candidate
coverage. With --json: a single machine-readable JSON line.
"""
from __future__ import annotations

import argparse
import colorsys
import json
import sys
from pathlib import Path

# Candidate key colors in priority order. Green first (best matte behaviour in
# remove_chroma_key.py), then magenta, orange, red, cyan. Each entry is
# (name, hex, target_hue_degrees).
CANDIDATES = [
    ("green", "#00ff00", 120.0),
    ("magenta", "#ff00ff", 300.0),
    ("orange", "#ff7a00", 30.0),
    ("red", "#ff0033", 350.0),
    ("cyan", "#00ffff", 180.0),
]

# A pixel "collides" with a candidate when it is a vivid pixel whose hue is near
# the candidate hue. We only count vivid pixels (high saturation + mid/high
# value) because the matte only struggles with saturated key-like regions.
HUE_TOLERANCE_DEG = 30.0
MIN_SATURATION = 0.35
MIN_VALUE = 0.30
# Avoid a candidate if vivid pixels near its hue exceed this fraction of the image.
COLLISION_FRACTION = 0.012


def _load_pixels(path: Path, max_side: int = 240):
    try:
        from PIL import Image
    except ImportError:
        print("Error: Pillow is required (pip3 install pillow).", file=sys.stderr)
        raise SystemExit(2)
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        raw = im.tobytes()
        pixels = [(raw[i], raw[i + 1], raw[i + 2]) for i in range(0, len(raw), 3)]
        return pixels, im.size


def _hue_distance(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def analyze(path: Path):
    pixels, (w, h) = _load_pixels(path)
    total = max(1, len(pixels))
    coverage = {name: 0 for name, _, _ in CANDIDATES}

    for r, g, b in pixels:
        hh, _, _ = colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
        hue = hh * 360.0
        mx = max(r, g, b)
        mn = min(r, g, b)
        sat = 0.0 if mx == 0 else (mx - mn) / mx
        val = mx / 255.0
        if sat < MIN_SATURATION or val < MIN_VALUE:
            continue
        for name, _hex, target in CANDIDATES:
            if _hue_distance(hue, target) <= HUE_TOLERANCE_DEG:
                coverage[name] += 1

    cov_frac = {name: coverage[name] / total for name, _, _ in CANDIDATES}
    recommended = None
    for name, hexval, _ in CANDIDATES:
        if cov_frac[name] < COLLISION_FRACTION:
            recommended = (name, hexval)
            break
    if recommended is None:
        # Everything collides; pick the least-used candidate.
        name = min(cov_frac, key=cov_frac.get)
        recommended = (name, dict((n, h) for n, h, _ in CANDIDATES)[name])

    return {
        "image": str(path),
        "size": [w, h],
        "coverage_fraction": {k: round(v, 5) for k, v in cov_frac.items()},
        "collision_threshold": COLLISION_FRACTION,
        "recommended_key_name": recommended[0],
        "recommended_key": recommended[1],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", help="Slide image to probe.")
    ap.add_argument("--json", action="store_true", help="Emit a single JSON line.")
    args = ap.parse_args()

    path = Path(args.image)
    if not path.exists():
        print(f"Error: image not found: {path}", file=sys.stderr)
        raise SystemExit(2)

    result = analyze(path)
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
        return

    print(f"image: {result['image']}  size: {result['size'][0]}x{result['size'][1]}")
    print("vivid coverage per candidate key color:")
    for name, hexval, _ in CANDIDATES:
        frac = result["coverage_fraction"][name]
        flag = "COLLIDES" if frac >= COLLISION_FRACTION else "ok"
        print(f"  {name:8} {hexval}  {frac*100:6.2f}%  {flag}")
    print(f"recommended_key: {result['recommended_key']}  ({result['recommended_key_name']})")


if __name__ == "__main__":
    main()
