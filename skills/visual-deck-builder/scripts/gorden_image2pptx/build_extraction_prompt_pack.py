#!/usr/bin/env python3
"""Build coverage-driven image-model extraction prompt packs.

Use after `icon_coverage_audit.py` or visual acceptance identifies dense-region
loss. The script does not generate images; it creates precise prompt files for
the next image editing pass so the run can target missing regions instead of
repeating a broad "extract all icons" request.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


FRAME_HINT_TERMS = (
    "radar", "axis", "marker", "micro", "symbol", "legend", "dot",
    "chart", "strip", "separator", "bottom", "status",
)


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "region"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _region_issue_map(report: dict) -> dict[str, list[dict]]:
    mapped: dict[str, list[dict]] = {}
    for issue in report.get("issues", []):
        region = issue.get("region")
        if region:
            mapped.setdefault(str(region), []).append(issue)
    return mapped


def _needs_frame_pass(region: dict, issues: list[dict]) -> bool:
    text = " ".join([
        str(region.get("name", "")),
        " ".join(str(item) for item in region.get("recommended_names", [])),
        " ".join(issue.get("message", "") for issue in issues),
        " ".join(str(issue.get("details", "")) for issue in issues),
    ]).lower()
    return any(term in text for term in FRAME_HINT_TERMS)


def _required_summary(region: dict, issues: list[dict]) -> str:
    parts = []
    if region.get("required_names"):
        parts.append(f"required icon/decor names: {json.dumps(region['required_names'], ensure_ascii=False)}")
    if region.get("recommended_names"):
        parts.append(f"recommended icon/decor names: {json.dumps(region['recommended_names'], ensure_ascii=False)}")
    for issue in issues:
        details = issue.get("details") or {}
        if details:
            parts.append(f"{issue.get('message')}: {json.dumps(details, ensure_ascii=False)}")
        else:
            parts.append(str(issue.get("message", "")))
    return "\n".join(f"- {part}" for part in parts) or "- Re-extract all missing visible elements in this region."


def _icon_prompt(deck_language: str, chroma: str, region: dict, issues: list[dict]) -> str:
    name = region["name"]
    bbox = region.get("bbox") or region.get("source_bbox")
    min_icons = region.get("min_icons", 0)
    return f"""# B4 Region Icon Sheet: {name}

Use the image that was just opened in the conversation as the only edit target. Do not infer from this text alone, and do not use any local file path as a substitute for the visible image.

Task: extract the missing movable icon/decor elements from source region `{name}`.

- Source region bbox in the original image coordinate system: `{bbox}`.
- Minimum required icon/decor count for this region: `{min_icons}`.
- Deck language: `{deck_language}`.
- Output format: one transparent-cutout icon sheet on a flat chroma background `{chroma}`.
- Arrange elements in a clean 4x4 grid when there are many small elements; use 2x2 only for large objects.
- Do not draw grid lines, separators, labels, captions, or explanatory text.
- Preserve original colors, strokes, shadows, and visual weight.
- Do not include ordinary page text. Text that is part of an icon or decorative badge may remain as pixels.
- Do not include frame/card borders, chart axes, radar grids, or panel fills unless they are visually inseparable from a movable icon.
- Leave padding around every element so `slice_grid.py --auto --pad 24` can cut cleanly.

Coverage requirements:
{_required_summary(region, issues)}

Reject and regenerate if any required element is missing, clipped, merged into another element, duplicated from the frame layer, or replaced by a generic icon.
"""


def _frame_prompt(deck_language: str, chroma: str, region: dict, issues: list[dict]) -> str:
    name = region["name"]
    bbox = region.get("bbox") or region.get("source_bbox")
    return f"""# B3 Frame Detail Pass: {name}

Use the image that was just opened in the conversation as the only edit target. Do not infer from this text alone, and do not use any local file path as a substitute for the visible image.

Task: regenerate only the structural/detail frame elements for source region `{name}`.

- Source region bbox in the original image coordinate system: `{bbox}`.
- Deck language: `{deck_language}`.
- Output format: full-slide frame/detail image on flat chroma background `{chroma}`.
- Keep the same slide size and 1:1 positions from the source image.
- Include structure, chart/radar geometry, fine divider lines, grid lines, dots, micro-markers, ribbons, status-strip separators, fills, and non-text scaffolding that belongs visually to the frame.
- Exclude ordinary readable text and exclude movable standalone icons that should be in B4 icon sheets.
- Preserve original colors, opacity, line weight, rounded corners, and shadows.
- Do not create new panels, labels, placeholder circles, or generic decorations.

Coverage requirements:
{_required_summary(region, issues)}

Reject and regenerate if the region becomes simpler than the source, loses small marks, or creates duplicate standalone icons that should be in the icon layer.
"""


def build_pack(report_path: Path, expected_path: Path, out_dir: Path, deck_language: str = "zh", chroma: str = "#ff00ff") -> dict:
    report = _load_json(report_path)
    expected = _load_json(expected_path)
    issue_map = _region_issue_map(report)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = []
    for region in expected.get("regions", []):
        name = str(region.get("name", "region"))
        issues = issue_map.get(name, [])
        region_report = next((item for item in report.get("regions", []) if item.get("name") == name), {})
        actual = int(region_report.get("icon_count", 0))
        minimum = int(region.get("min_icons", 0))
        has_missing_names = any(issue.get("region") == name and "missing" in issue.get("message", "") for issue in issues)
        if actual >= minimum and not issues:
            continue

        slug = _slug(name)
        if actual < minimum or has_missing_names:
            path = out_dir / f"b4-icons-{slug}.md"
            path.write_text(_icon_prompt(deck_language, chroma, region, issues), encoding="utf-8")
            prompts.append({
                "region": name,
                "kind": "icons",
                "prompt_file": str(path),
                "actual_icons": actual,
                "minimum_icons": minimum,
                "issue_count": len(issues),
            })

        if _needs_frame_pass(region, issues):
            path = out_dir / f"b3-frame-detail-{slug}.md"
            path.write_text(_frame_prompt(deck_language, chroma, region, issues), encoding="utf-8")
            prompts.append({
                "region": name,
                "kind": "frame_detail",
                "prompt_file": str(path),
                "actual_icons": actual,
                "minimum_icons": minimum,
                "issue_count": len(issues),
            })

    index = {
        "status": "pass" if prompts else "noop",
        "source_report": str(report_path),
        "expected": str(expected_path),
        "deck_language": deck_language,
        "chroma": chroma,
        "prompt_count": len(prompts),
        "prompts": prompts,
    }
    (out_dir / "prompt-pack-index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("coverage_report", type=Path)
    parser.add_argument("expected", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--language", default="zh")
    parser.add_argument("--chroma", default="#ff00ff")
    args = parser.parse_args()

    index = build_pack(args.coverage_report, args.expected, args.out_dir, deck_language=args.language, chroma=args.chroma)
    print(json.dumps(index, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
