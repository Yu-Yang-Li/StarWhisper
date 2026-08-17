#!/usr/bin/env python3
"""Color-preserving chroma-key removal for infographic assets (frames / icons / decorations).

Why a dedicated script (instead of a generic green-screen remover):
  Generic removers tend to (a) DESATURATE content — capping the key channel on every
  semi-key pixel turns reds/oranges grey — and (b) EAT thin bright lines / glows when an
  aggressive soft-matte + edge erosion is used. For PPT skeletons that is fatal: the user
  loses divider lines, glow rings and accent fills.

This script instead:
  * keeps a FLAT key-color matte (hard core + anti-aliased ramp), so content colors that
    are far from the key are preserved 100% (no desaturation of reds / navy / grey / white);
  * reconstructs anti-aliased edge colors by unmixing the flat key color before despill,
    so thin strokes keep their original color instead of becoming dirty grey/green;
    recovery is limited to medium/high-alpha edge pixels to avoid magenta/purple halos
    caused by over-amplifying near-transparent pixels;
  * de-spills ONLY key-dominant pixels (e.g. green-dominant) by pulling the key channel down
    to the max of the other channels — this is a no-op for non-key colors, so it never
    greys-out content;
  * boosts anti-aliased edge alpha for frame/icon presets, so hairlines and arcs do not
    break into dotted fragments after transparency;
  * repairs tiny alpha gaps with binary close + alpha floor instead of filtering the
    grayscale alpha directly, so horizontal/vertical lines stay straight and arcs stay smooth;
  * does NOT erode the matte by default, so 1px lines and glow survive;
  * lets you opt into a tiny contract/feather when you specifically want to kill a fringe.

Usage:
    python3 chroma_key.py --input frame_raw.png --out frame.png            # auto green key
    python3 chroma_key.py --input ic.png --out ic.png --key-color #ff00ff  # magenta key
    python3 chroma_key.py --input f.png --out f.png --t-low 40 --t-high 110 --contract 1
    python3 chroma_key.py --input icons_raw.png --out icons_t.png --preset icon-safe --scale 2
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover
    print("Error: numpy is required (pip install numpy).", file=sys.stderr)
    raise SystemExit(1)
try:
    from PIL import Image, ImageFilter
except ImportError:  # pragma: no cover
    print("Error: Pillow is required (pip install pillow).", file=sys.stderr)
    raise SystemExit(1)

Color = Tuple[int, int, int]


def _die(msg: str, code: int = 1) -> None:
    print(f"Error: {msg}", file=sys.stderr)
    raise SystemExit(code)


def _parse_hex(raw: str) -> Color:
    m = re.fullmatch(r"#?([0-9a-fA-F]{6})", raw.strip())
    if not m:
        _die("key color must be hex RGB like #00ff00")
    h = m.group(1)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _sample_border_key(arr: np.ndarray, mode: str) -> Color:
    """Median color of the image border / corners (the flat key background)."""
    h, w = arr.shape[:2]
    rgb = arr[..., :3]
    if mode == "corners":
        p = max(1, min(w, h) // 12)
        patches = [rgb[:p, :p], rgb[:p, -p:], rgb[-p:, :p], rgb[-p:, -p:]]
        samples = np.concatenate([p.reshape(-1, 3) for p in patches], axis=0)
    else:  # border band
        b = max(2, min(w, h) // 60)
        samples = np.concatenate([
            rgb[:b, :].reshape(-1, 3), rgb[-b:, :].reshape(-1, 3),
            rgb[:, :b].reshape(-1, 3), rgb[:, -b:].reshape(-1, 3),
        ], axis=0)
    med = np.median(samples.astype(np.float64), axis=0)
    return int(round(med[0])), int(round(med[1])), int(round(med[2]))


def _spill_channels(key: Color) -> List[int]:
    """Channels that carry the key color's energy (e.g. [1] for green, [0,2] for magenta)."""
    km = max(key)
    if km < 110:
        return []
    return [i for i, v in enumerate(key) if v >= km - 24 and v >= 110]


def remove_key(
    arr: np.ndarray,
    key: Color,
    t_low: float,
    t_high: float,
    despill: bool,
    contract: int,
    feather: float,
    edge_recover: bool,
    alpha_close: int,
    alpha_gamma: float,
    alpha_repair_floor: int,
    alpha_cutoff: int,
    alpha_opaque: int,
    edge_recover_min: float,
) -> Tuple[np.ndarray, dict]:
    h, w = arr.shape[:2]
    rgb = arr[..., :3].astype(np.float32)
    src_a = arr[..., 3].astype(np.float32) / 255.0
    key_arr = np.array(key, dtype=np.float32)

    # Chebyshev distance to the flat key color -> anti-aliased matte.
    dist = np.max(np.abs(rgb - key_arr), axis=2)
    if t_high <= t_low:
        t_high = t_low + 1.0
    ramp = (dist - t_low) / (t_high - t_low)
    ramp = np.clip(ramp, 0.0, 1.0)
    alpha = ramp * ramp * (3.0 - 2.0 * ramp)  # smoothstep

    spill = _spill_channels(key)
    out_rgb = rgb.copy()

    if edge_recover:
        # Recover foreground color from anti-aliased edge pixels:
        # observed = alpha * foreground + (1 - alpha) * key.
        # This removes the green/magenta contamination that otherwise makes
        # icon outlines and hairlines look ragged after transparency is applied.
        a = np.clip(alpha[..., None], 0.0, 1.0)
        edge = (a[..., 0] > edge_recover_min) & (a[..., 0] < 0.995)
        if np.any(edge):
            safe_a = np.maximum(a, max(0.08, edge_recover_min))
            recovered = (rgb - (1.0 - a) * key_arr) / safe_a
            out_rgb[edge] = recovered[edge]
            out_rgb = np.clip(out_rgb, 0.0, 255.0)

    if despill and spill:
        non_spill = [i for i in range(3) if i not in spill]
        # Anchor = strongest non-key channel; for multi-channel keys use the per-pixel max.
        anchor = (np.max(out_rgb[..., non_spill], axis=2) if non_spill
                  else np.zeros((h, w), np.float32))
        for c in spill:
            # Pull the key channel down to the anchor only where it exceeds it
            # (key-dominant). No-op for reds/navy/grey/white -> colors preserved.
            out_rgb[..., c] = np.minimum(out_rgb[..., c], anchor)

    if alpha_gamma > 0 and alpha_gamma != 1.0:
        # gamma < 1 raises semi-transparent edge alpha, preserving thin strokes
        # and circular arcs that otherwise become broken after PowerPoint scaling.
        alpha = np.power(np.clip(alpha, 0.0, 1.0), alpha_gamma)

    a8 = (alpha * src_a * 255.0).astype(np.uint8)
    if alpha_cutoff > 0:
        a8[a8 < alpha_cutoff] = 0
    if alpha_opaque < 255:
        a8[a8 >= alpha_opaque] = 255
    if alpha_close > 0:
        a8 = _repair_alpha_gaps(a8, alpha_close, alpha_repair_floor)

    out = np.dstack([np.clip(out_rgb, 0, 255).astype(np.uint8), a8])
    img = Image.fromarray(out, "RGBA")

    if contract > 0:
        ch = img.getchannel("A")
        for _ in range(contract):
            ch = ch.filter(ImageFilter.MinFilter(3))
        img.putalpha(ch)
    if feather > 0:
        img.putalpha(img.getchannel("A").filter(ImageFilter.GaussianBlur(feather)))

    fa = np.asarray(img.getchannel("A"))
    stats = {
        "key": "#%02x%02x%02x" % key,
        "transparent": int((fa == 0).sum()),
        "opaque": int((fa == 255).sum()),
        "partial": int(((fa > 0) & (fa < 255)).sum()),
        "total": int(fa.size),
    }
    return np.asarray(img), stats


def _repair_alpha_gaps(a8: np.ndarray, rounds: int, repair_floor: int) -> np.ndarray:
    """Fill pinholes/gaps without reshaping existing anti-aliased edges.

    A direct max/min filter on the grayscale alpha channel makes straight lines
    and arcs uneven. Instead, close a binary foreground mask, then only raise
    newly covered gap pixels to a floor alpha.
    """
    repair_floor = int(np.clip(repair_floor, 0, 255))
    if repair_floor <= 0:
        return a8
    orig = a8 > 4
    base = Image.fromarray((orig.astype(np.uint8) * 255), "L")
    closed = base
    for _ in range(max(0, int(rounds))):
        closed = closed.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))
    fill = np.asarray(closed) > 0
    repaired = a8.copy()
    gap = fill & ~orig
    repaired[gap] = repair_floor
    return repaired


def _apply_preset(args) -> None:
    """Tune defaults for frame layers vs icon sheets without hiding explicit flags."""
    if args.preset == "default":
        return
    if args.preset == "icon-safe":
        # Icon sheets need softer alpha on anti-aliased borders and zero erosion.
        # This keeps circular badge outlines and art-word strokes intact after
        # scaling down in PowerPoint.
        if args.t_low == ap_defaults["t_low"]:
            args.t_low = 24.0
        if args.t_high == ap_defaults["t_high"]:
            args.t_high = 86.0
        if args.pad_alpha is None:
            args.pad_alpha = True
        if args.alpha_gamma == ap_defaults["alpha_gamma"]:
            args.alpha_gamma = 0.70
        if args.alpha_repair_floor == ap_defaults["alpha_repair_floor"]:
            args.alpha_repair_floor = 220
        if args.alpha_cutoff == ap_defaults["alpha_cutoff"]:
            args.alpha_cutoff = 14
        if args.alpha_opaque == ap_defaults["alpha_opaque"]:
            args.alpha_opaque = 248
        if args.edge_recover_min == ap_defaults["edge_recover_min"]:
            args.edge_recover_min = 0.20
        args.contract = 0
        args.feather = 0.0
        if args.alpha_close == 0:
            args.alpha_close = 1
    elif args.preset == "frame-safe":
        # Frame layers often contain hairline dividers. Keep a harder matte,
        # but still never erode unless explicitly requested.
        if args.t_low == ap_defaults["t_low"]:
            args.t_low = 32.0
        if args.t_high == ap_defaults["t_high"]:
            args.t_high = 100.0
        if args.alpha_gamma == ap_defaults["alpha_gamma"]:
            args.alpha_gamma = 0.72
        if args.alpha_repair_floor == ap_defaults["alpha_repair_floor"]:
            args.alpha_repair_floor = 235
        if args.alpha_cutoff == ap_defaults["alpha_cutoff"]:
            args.alpha_cutoff = 18
        if args.alpha_opaque == ap_defaults["alpha_opaque"]:
            args.alpha_opaque = 246
        if args.edge_recover_min == ap_defaults["edge_recover_min"]:
            args.edge_recover_min = 0.26
        args.contract = 0
        if args.alpha_close == 0:
            args.alpha_close = 1


ap_defaults = {
    "t_low": 38.0,
    "t_high": 110.0,
    "alpha_gamma": 1.0,
    "alpha_repair_floor": 220,
    "alpha_cutoff": 0,
    "alpha_opaque": 255,
    "edge_recover_min": 0.08,
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Color-preserving chroma-key removal.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True, help="output .png (alpha preserved)")
    ap.add_argument("--key-color", default="#00ff00", help="hex key color when --auto-key none")
    ap.add_argument("--auto-key", choices=["none", "border", "corners"], default="border",
                    help="sample key color from border/corners (default border)")
    ap.add_argument("--t-low", type=float, default=38.0,
                    help="distance<=t-low -> fully transparent (default 38)")
    ap.add_argument("--t-high", type=float, default=110.0,
                    help="distance>=t-high -> fully opaque; the gap is the anti-aliased edge (default 110)")
    ap.add_argument("--preset", choices=["default", "icon-safe", "frame-safe"], default="default",
                    help="quality preset; icon-safe preserves circular badge/art-word borders")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="upscale input before matting; 2 improves lines/arcs and PPT downsampling smoothness")
    ap.add_argument("--pad-alpha", action="store_true", default=None,
                    help="after matting, add a transparent safety border around the output canvas")
    ap.add_argument("--no-despill", dest="despill", action="store_false",
                    help="disable green/key spill cleanup on key-dominant pixels")
    ap.add_argument("--no-edge-recover", dest="edge_recover", action="store_false",
                    help="disable anti-aliased edge color reconstruction")
    ap.add_argument("--alpha-close", type=int, default=0,
                    help="repair tiny alpha gaps by N px before optional contract/feather (default 0; presets use 1)")
    ap.add_argument("--alpha-gamma", type=float, default=1.0,
                    help="gamma applied to foreground alpha; <1 boosts anti-aliased edges (presets use <1)")
    ap.add_argument("--alpha-repair-floor", type=int, default=220,
                    help="minimum alpha for pixels filled by --alpha-close repair (0-255)")
    ap.add_argument("--alpha-cutoff", type=int, default=0,
                    help="alpha below this is forced transparent; presets remove low-alpha speckles")
    ap.add_argument("--alpha-opaque", type=int, default=255,
                    help="alpha at/above this is forced opaque; presets stabilize solid strokes")
    ap.add_argument("--edge-recover-min", type=float, default=0.08,
                    help="minimum alpha for edge color unmixing; higher reduces magenta/purple halos")
    ap.add_argument("--contract", type=int, default=0,
                    help="erode matte by N px to kill a fringe (default 0; keep 0 to preserve thin lines)")
    ap.add_argument("--feather", type=float, default=0.0, help="gaussian blur alpha radius (default 0)")
    ap.add_argument("--force", action="store_true")
    ap.set_defaults(despill=True, edge_recover=True)
    args = ap.parse_args()
    _apply_preset(args)

    src = Path(args.input)
    if not src.exists():
        _die(f"input not found: {src}")
    out = Path(args.out)
    if out.exists() and not args.force:
        _die(f"output exists: {out} (use --force)")
    if out.suffix.lower() != ".png":
        _die("--out must be a .png so alpha is preserved")

    im = Image.open(src).convert("RGBA")
    if args.scale <= 0:
        _die("--scale must be > 0")
    if args.scale != 1.0:
        nw = max(1, int(round(im.width * args.scale)))
        nh = max(1, int(round(im.height * args.scale)))
        im = im.resize((nw, nh), Image.Resampling.LANCZOS)
    arr = np.asarray(im)
    key = (_parse_hex(args.key_color) if args.auto_key == "none"
           else _sample_border_key(arr, args.auto_key))
    result, stats = remove_key(arr, key, args.t_low, args.t_high,
                               args.despill, args.contract, args.feather,
                               args.edge_recover, max(0, args.alpha_close),
                               args.alpha_gamma, args.alpha_repair_floor,
                               args.alpha_cutoff, args.alpha_opaque,
                               args.edge_recover_min)
    if args.pad_alpha:
        result = np.pad(result, ((4, 4), (4, 4), (0, 0)), mode="constant", constant_values=0)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result, "RGBA").save(out)
    pct = 100.0 * stats["transparent"] / stats["total"]
    print(f"Wrote {out}")
    print(f"Key {stats['key']} | transparent {stats['transparent']}/{stats['total']} "
          f"({pct:.1f}%) | partial {stats['partial']} | opaque {stats['opaque']}")
    if stats["transparent"] == 0:
        print("Warning: nothing matched the key color — check --key-color / --t-low.", file=sys.stderr)


if __name__ == "__main__":
    main()
