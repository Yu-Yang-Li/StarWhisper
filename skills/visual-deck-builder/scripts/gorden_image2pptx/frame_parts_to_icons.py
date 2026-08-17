#!/usr/bin/env python3
"""Convert frame_parts/icons_manifest.json into layout.icons frame_part entries.

This is an optional helper for Image2PPTX runs where the user explicitly asks
to split the framework layer into movable pieces. The default workflow keeps
the whole transparent frame.png as a single full-slide layer.

The frame part manifest stores each cutout's bbox in frame.png coordinates.
This script maps those bboxes into the slide reference coordinate system so
placing every cutout at the emitted x/y/w/h reconstructs the original frame.png
layout deterministically.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _die(msg: str) -> None:
    print(f"Error: {msg}", file=sys.stderr)
    raise SystemExit(2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest", help="frame_parts/icons_manifest.json from slice_grid.py --components.")
    ap.add_argument("--ref-width", type=float, required=True, help="Source slide image width in pixels.")
    ap.add_argument("--ref-height", type=float, required=True, help="Source slide image height in pixels.")
    ap.add_argument("--path-prefix", default=None,
                    help='File prefix for layout paths. Defaults to manifest parent name, e.g. "frame_parts".')
    ap.add_argument("--units", choices=("px", "fraction"), default="fraction",
                    help="Output x/y/w/h units. Image2PPTX defaults to fraction for PPT-size-independent placement.")
    ap.add_argument("--out", default=None,
                    help="Output JSON file. Defaults to frame_parts_layout_icons.json next to the manifest.")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        _die(f"manifest not found: {manifest_path}")

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data.get("mode") != "components":
        _die("manifest mode must be 'components'; run slice_grid.py with --components for frame parts")
    if "size" not in data or len(data["size"]) != 2:
        _die("manifest missing size:[frame_w,frame_h]")

    frame_w, frame_h = map(float, data["size"])
    if frame_w <= 0 or frame_h <= 0:
        _die(f"invalid frame size: {data['size']}")

    path_prefix = args.path_prefix
    if path_prefix is None:
        path_prefix = manifest_path.parent.name
    path_prefix = path_prefix.strip("/")

    icons = []
    for i, item in enumerate(data.get("icons", []), 1):
        bbox = item.get("bbox")
        if not bbox or len(bbox) != 4:
            _die(f"item {i} missing bbox")
        l, t, r, b = map(float, bbox)
        bw, bh = r - l, b - t
        if bw <= 0 or bh <= 0:
            _die(f"item {i} has invalid bbox: {bbox}")

        # For frame reconstruction the cutout canvas must be exactly the bbox
        # canvas. --square would add transparent margins and break placement.
        iw = float(item.get("width", bw))
        ih = float(item.get("height", bh))
        if round(iw) != round(bw) or round(ih) != round(bh):
            _die(
                f"item {i} cutout size {iw:g}x{ih:g} does not match bbox {bw:g}x{bh:g}; "
                "do not use --square for frame_parts"
            )

        if args.units == "fraction":
            x, y, w, h = l / frame_w, t / frame_h, bw / frame_w, bh / frame_h
            source_bbox = [
                l / frame_w * args.ref_width,
                t / frame_h * args.ref_height,
                bw / frame_w * args.ref_width,
                bh / frame_h * args.ref_height,
            ]
        else:
            x = l / frame_w * args.ref_width
            y = t / frame_h * args.ref_height
            w = bw / frame_w * args.ref_width
            h = bh / frame_h * args.ref_height
            source_bbox = [x, y, w, h]

        file_name = item["file"]
        file_path = f"{path_prefix}/{file_name}" if path_prefix else file_name
        icons.append({
            "file": file_path,
            "role": "frame_part",
            "source_label": f"frame_part_{i:03d}",
            "x": round(x, 4),
            "y": round(y, 4),
            "w": round(w, 4),
            "h": round(h, 4),
            "source_bbox": [round(v, 4) for v in source_bbox],
        })

    out_path = Path(args.out) if args.out else manifest_path.parent / "frame_parts_layout_icons.json"
    out_path.write_text(json.dumps(icons, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(icons)} frame_part icon entries to {out_path}")


if __name__ == "__main__":
    main()
