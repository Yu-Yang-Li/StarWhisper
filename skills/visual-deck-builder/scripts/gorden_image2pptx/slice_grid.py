#!/usr/bin/env python3
"""Slice a transparent icon/frame image into individual alpha-trimmed PNGs.

Pipeline position (Part 2): after you generate a multi-icon chroma-key sheet
and run `chroma_key.py` to make it transparent, this tool cuts it into individual
icon PNGs. Prefer `--auto` for generated icon sheets: it segments by transparent
gaps and is more tolerant when AI does not place icons perfectly inside cells.
Optional: when the user explicitly asks to split a full-slide framework image
into movable pieces, use `--components`. It slices by connected non-transparent
regions so each transparent-separated frame part becomes a movable PNG. The
default Image2PPTX workflow keeps the whole transparent frame.png.

Usage:
    python3 scripts/slice_grid.py GRID_TRANSPARENT.png OUT_DIR --grid 4x4
    python3 scripts/slice_grid.py deco_transparent.png out/deco --grid 2x2 --prefix deco
    python3 scripts/slice_grid.py icons_t_1.png out/icons --auto --pad 24 --contact-sheet
    python3 scripts/slice_grid.py frame.png out/frame_parts --components --prefix fp

Notes:
- Input MUST already be a transparent PNG (run remove_chroma_key.py first).
- Cells are cut by equal fractions, so the source grid must be evenly divided.
- Near-empty cells (mostly transparent) are skipped and reported.
- Emits a manifest JSON describing each saved cutout (grid position + size).
- Optional --contact-sheet emits a labeled preview sheet for visual QA.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse_grid(value: str):
    try:
        rows, cols = value.lower().split("x")
        return int(rows), int(cols)
    except Exception:
        print(f"Error: --grid must look like 4x4, got {value!r}", file=sys.stderr)
        raise SystemExit(2)


def _alpha_bbox(im, min_alpha: int):
    """Return (l, t, r, b) bounding box of pixels with alpha > min_alpha, or None."""
    alpha = im.getchannel("A")
    # Threshold the alpha channel then use getbbox for speed.
    mask = alpha.point(lambda a: 255 if a > min_alpha else 0)
    return mask.getbbox()


def _runs(flags, min_gap):
    """Yield (start, end) index ranges of True runs, merging gaps shorter than min_gap."""
    runs = []
    start = None
    for i, v in enumerate(flags):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append([start, i])
            start = None
    if start is not None:
        runs.append([start, len(flags)])
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < min_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return merged


def _auto_segment(im, min_alpha, pad, min_side_frac):
    """Projection-based segmentation: split content into rows by horizontal gaps,
    then each row into items by vertical gaps. Robust when the AI grid is imperfect
    or content is not centered in its cells. Returns list of (bbox, row, col)."""
    try:
        import numpy as np
    except ImportError:
        _die_auto()
    W, H = im.size
    arr = np.asarray(im.getchannel("A"))
    mask = arr > min_alpha
    min_px_row = max(1, int(0.003 * W))
    min_px_col = max(1, int(0.003 * H))
    row_gap = max(2, int(0.02 * H))
    col_gap = max(2, int(0.02 * W))
    min_w = max(2, int(min_side_frac * W))
    min_h = max(2, int(min_side_frac * H))

    row_has = (mask.sum(axis=1) > min_px_row).tolist()
    out = []
    for ri, (y0, y1) in enumerate(_runs(row_has, row_gap), 1):
        band = mask[y0:y1]
        col_has = (band.sum(axis=0) > min_px_col).tolist()
        for ci, (x0, x1) in enumerate(_runs(col_has, col_gap), 1):
            sub = band[:, x0:x1]
            ys = np.where(sub.any(axis=1))[0]
            xs = np.where(sub.any(axis=0))[0]
            if len(ys) == 0 or len(xs) == 0:
                continue
            bx0 = x0 + int(xs[0]); bx1 = x0 + int(xs[-1]) + 1
            by0 = y0 + int(ys[0]); by1 = y0 + int(ys[-1]) + 1
            if (bx1 - bx0) < min_w or (by1 - by0) < min_h:
                continue
            l = max(0, bx0 - pad); t = max(0, by0 - pad)
            r = min(W, bx1 + pad); b = min(H, by1 + pad)
            out.append(((l, t, r, b), ri, ci))
    return out


def _die_auto():
    print("Error: --auto needs numpy (pip3 install numpy).", file=sys.stderr)
    raise SystemExit(2)


def _component_segment(im, min_alpha, pad, min_area):
    """Connected-component segmentation by alpha transparency.

    Returns list of (bbox, index) sorted top-to-bottom, left-to-right. Components
    are based only on non-transparent pixel connectivity; no semantic grouping is
    applied.
    """
    try:
        import numpy as np
    except ImportError:
        _die_auto()

    W, H = im.size
    mask = np.asarray(im.getchannel("A")) > min_alpha
    out = []

    try:
        from scipy import ndimage  # type: ignore
        labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
        objects = ndimage.find_objects(labels)
        for label_id, slc in enumerate(objects, 1):
            if slc is None:
                continue
            ys, xs = slc
            area = int((labels[ys, xs] == label_id).sum())
            if area < min_area:
                continue
            x0, x1 = int(xs.start), int(xs.stop)
            y0, y1 = int(ys.start), int(ys.stop)
            out.append(((
                max(0, x0 - pad), max(0, y0 - pad),
                min(W, x1 + pad), min(H, y1 + pad)), label_id, area))
    except Exception:
        from collections import deque
        visited = np.zeros(mask.shape, dtype=bool)
        ys, xs = np.where(mask)
        neighbors = ((-1, -1), (0, -1), (1, -1), (-1, 0),
                     (1, 0), (-1, 1), (0, 1), (1, 1))
        label_id = 0
        for y, x in zip(ys.tolist(), xs.tolist()):
            if visited[y, x]:
                continue
            label_id += 1
            q = deque([(x, y)])
            visited[y, x] = True
            x0 = x1 = x
            y0 = y1 = y
            area = 0
            while q:
                cx, cy = q.popleft()
                area += 1
                x0 = min(x0, cx); x1 = max(x1, cx)
                y0 = min(y0, cy); y1 = max(y1, cy)
                for dx, dy in neighbors:
                    nx, ny = cx + dx, cy + dy
                    if nx < 0 or ny < 0 or nx >= W or ny >= H:
                        continue
                    if visited[ny, nx] or not mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    q.append((nx, ny))
            if area < min_area:
                continue
            out.append(((
                max(0, x0 - pad), max(0, y0 - pad),
                min(W, x1 + 1 + pad), min(H, y1 + 1 + pad)), label_id, area))

    out.sort(key=lambda item: (item[0][1], item[0][0]))
    return [((l, t, r, b), i + 1, area) for i, ((l, t, r, b), _label, area) in enumerate(out)]


def _write_contact_sheet(icon_paths, out_path: Path):
    """Write a labeled visual index of sliced icons for mandatory QA."""
    if not icon_paths:
        return
    from PIL import Image, ImageDraw, ImageFont
    thumb_w, thumb_h = 220, 180
    cols = min(4, max(1, len(icon_paths)))
    rows = (len(icon_paths) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    font = None
    for candidate in (
        "/System/Library/Fonts/PingFang.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            font = ImageFont.truetype(candidate, 16)
            break
        except Exception:
            pass
    for i, path in enumerate(icon_paths):
        x = (i % cols) * thumb_w
        y = (i // cols) * thumb_h
        draw.rectangle([x, y, x + thumb_w - 1, y + thumb_h - 1], outline=(180, 180, 180))
        with Image.open(path) as im0:
            im = im0.convert("RGBA")
            im.thumbnail((thumb_w - 24, thumb_h - 48))
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            bg.alpha_composite(im)
        sheet.paste(bg.convert("RGB"), (x + (thumb_w - im.width) // 2, y + 10))
        draw.text((x + 8, y + thumb_h - 30), path.name, fill=(0, 0, 0), font=font)
    sheet.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="Transparent grid PNG (already chroma-keyed).")
    ap.add_argument("out_dir", help="Directory for the sliced icon PNGs.")
    ap.add_argument("--grid", default="4x4", help="Grid as ROWSxCOLS, e.g. 4x4 or 2x2.")
    ap.add_argument("--auto", action="store_true",
                    help="Ignore --grid; auto-segment items by transparent gaps (robust to imperfect grids).")
    ap.add_argument("--components", action="store_true",
                    help="Ignore --grid/--auto; slice connected non-transparent components, for frame parts.")
    ap.add_argument("--prefix", default="icon", help="Filename prefix for cutouts.")
    ap.add_argument("--min-alpha", type=int, default=12,
                    help="Alpha above this counts as content when trimming (0-255).")
    ap.add_argument("--pad", type=int, default=16,
                    help="Transparent padding (px) kept around each trimmed icon.")
    ap.add_argument("--min-coverage", type=float, default=0.004,
                    help="Skip a cell if content covers less than this fraction of it.")
    ap.add_argument("--min-area", type=int, default=16,
                    help="With --components, skip connected components smaller than this pixel area.")
    ap.add_argument("--square", action="store_true",
                    help="Pad each cutout to a square canvas (keeps aspect, centers).")
    ap.add_argument("--contact-sheet", action="store_true",
                    help="Write icons_contact_sheet.png for visual QA of clipped/missing cutouts.")
    args = ap.parse_args()

    try:
        from PIL import Image
    except ImportError:
        print("Error: Pillow is required (pip3 install pillow).", file=sys.stderr)
        raise SystemExit(2)

    src = Path(args.input)
    if not src.exists():
        print(f"Error: input not found: {src}", file=sys.stderr)
        raise SystemExit(2)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(src) as im0:
        im = im0.convert("RGBA")
    W, H = im.size

    if args.components:
        manifest = {"source": str(src), "mode": "components", "size": [W, H], "icons": []}
        segments = _component_segment(im, args.min_alpha, args.pad, max(1, args.min_area))
        saved = []
        for bbox, idx, area in segments:
            cutout = im.crop(bbox)
            if args.square:
                side = max(cutout.size)
                canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
                canvas.paste(cutout, ((side - cutout.size[0]) // 2,
                                      (side - cutout.size[1]) // 2), cutout)
                cutout = canvas
            name = f"{args.prefix}_{idx:03d}.png"
            edge_touch = {
                "left": bbox[0] <= args.pad,
                "top": bbox[1] <= args.pad,
                "right": bbox[2] >= W - args.pad,
                "bottom": bbox[3] >= H - args.pad,
            }
            out_path = out_dir / name
            cutout.save(out_path)
            saved.append(out_path)
            manifest["icons"].append({
                "file": name, "index": idx,
                "width": cutout.size[0], "height": cutout.size[1],
                "aspect": round(cutout.size[0] / cutout.size[1], 4),
                "bbox": list(bbox), "area": area,
                "edge_touch": edge_touch})
        (out_dir / "icons_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.contact_sheet:
            _write_contact_sheet(saved, out_dir / "icons_contact_sheet.png")
            print(f"Contact sheet: {out_dir/'icons_contact_sheet.png'}")
        print(f"[components] Saved {len(manifest['icons'])} items to {out_dir}")
        print(f"Manifest: {out_dir/'icons_manifest.json'}")
        return

    if args.auto:
        manifest = {"source": str(src), "mode": "auto", "size": [W, H], "icons": []}
        segments = _auto_segment(im, args.min_alpha, args.pad, max(0.01, args.min_coverage ** 0.5))
        saved = []
        for bbox, r, c in segments:
            icon = im.crop(bbox)
            if args.square:
                side = max(icon.size)
                canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
                canvas.paste(icon, ((side - icon.size[0]) // 2,
                                    (side - icon.size[1]) // 2), icon)
                icon = canvas
            name = f"{args.prefix}_r{r}c{c}.png"
            edge_touch = {
                "left": bbox[0] <= args.pad,
                "top": bbox[1] <= args.pad,
                "right": bbox[2] >= W - args.pad,
                "bottom": bbox[3] >= H - args.pad,
            }
            out_path = out_dir / name
            icon.save(out_path)
            saved.append(out_path)
            manifest["icons"].append({
                "file": name, "row": r, "col": c,
                "width": icon.size[0], "height": icon.size[1],
                "aspect": round(icon.size[0] / icon.size[1], 4),
                "bbox": list(bbox),
                "edge_touch": edge_touch})
        (out_dir / "icons_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.contact_sheet:
            _write_contact_sheet(saved, out_dir / "icons_contact_sheet.png")
            print(f"Contact sheet: {out_dir/'icons_contact_sheet.png'}")
        print(f"[auto] Saved {len(manifest['icons'])} items to {out_dir}")
        print(f"Manifest: {out_dir/'icons_manifest.json'}")
        return

    rows, cols = _parse_grid(args.grid)
    cell_w = W / cols
    cell_h = H / rows

    manifest = {"source": str(src), "grid": [rows, cols], "size": [W, H], "icons": []}
    skipped = []
    saved = []

    for r in range(rows):
        for c in range(cols):
            left = int(round(c * cell_w))
            top = int(round(r * cell_h))
            right = int(round((c + 1) * cell_w))
            bottom = int(round((r + 1) * cell_h))
            cell = im.crop((left, top, right, bottom))
            cw, ch = cell.size

            bbox = _alpha_bbox(cell, args.min_alpha)
            if bbox is None:
                skipped.append([r, c])
                continue
            bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if (bw * bh) < args.min_coverage * (cw * ch):
                skipped.append([r, c])
                continue

            # Trim to content with padding (clamped to the cell). If content
            # touches a cell edge, the manifest flags it so the asset can be
            # regenerated or re-sliced with --auto before it reaches compose.
            edge_touch = {
                "left": bbox[0] <= args.pad,
                "top": bbox[1] <= args.pad,
                "right": bbox[2] >= cw - args.pad,
                "bottom": bbox[3] >= ch - args.pad,
            }
            pl = max(0, bbox[0] - args.pad)
            pt = max(0, bbox[1] - args.pad)
            pr = min(cw, bbox[2] + args.pad)
            pb = min(ch, bbox[3] + args.pad)
            icon = cell.crop((pl, pt, pr, pb))

            if args.square:
                side = max(icon.size)
                canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
                canvas.paste(icon, ((side - icon.size[0]) // 2,
                                    (side - icon.size[1]) // 2), icon)
                icon = canvas

            name = f"{args.prefix}_r{r+1}c{c+1}.png"
            out_path = out_dir / name
            icon.save(out_path)
            saved.append(out_path)
            manifest["icons"].append({
                "file": name,
                "row": r + 1,
                "col": c + 1,
                "width": icon.size[0],
                "height": icon.size[1],
                "aspect": round(icon.size[0] / icon.size[1], 4),
                "edge_touch": edge_touch,
            })

    manifest["skipped_empty_cells"] = skipped
    (out_dir / "icons_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.contact_sheet:
        _write_contact_sheet(saved, out_dir / "icons_contact_sheet.png")
        print(f"Contact sheet: {out_dir/'icons_contact_sheet.png'}")

    print(f"Saved {len(manifest['icons'])} icons to {out_dir}")
    if skipped:
        print(f"Skipped {len(skipped)} empty/near-empty cells: {skipped}")
    print(f"Manifest: {out_dir/'icons_manifest.json'}")


if __name__ == "__main__":
    main()
