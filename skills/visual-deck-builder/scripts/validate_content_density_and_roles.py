#!/usr/bin/env python3
"""Validate slide_spec.json against the content rules banana-slides encodes
in its outline/description prompts, independent of how the spec was authored
(generate_slide_spec.py, or hand-written by an agent).

Checks (each with a banana-slides source citation in the message):

1. Text density floor: a slide's body_text must not be reduced to isolated
   2-4 character labels; it should read closer to banana's DEFAULT
   `DETAIL_LEVEL_SPECS` entry (2-6 short phrases/sentences per page).
2. Cover page anatomy: the first slide should carry subtitle/presenter-style
   content, not just a bare title (prompts.py:292/340/591/642).
3. Page-role diversity: a deck with 4+ slides should not be 100% diagram-role
   pages, and should not repeat the same page_role 4+ times in a row
   (prompts.py:1293 template_role, 1524 anti-repetition rule).
4. Layout profile contract: each slide should carry the banana-slides 9-field
   template-analysis schema equivalent (`layout_profile`), so downstream image
   prompts can reason about content capacity, text regions, image regions, and
   visual density instead of relying on one freeform layout sentence.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

MIN_BODY_TEXT_ITEMS = 2
MIN_ITEM_CHARS_FOR_PHRASE = 4  # below this, an item reads as a bare keyword tag, not a phrase
MIN_PHRASE_ITEMS_REQUIRED = 2  # at least this many items must clear the phrase-length bar
DIAGRAM_ROLES = {"data", "comparison", "timeline"}
LAYOUT_PROFILE_FIELDS = {
    "template_role",
    "layout_structure",
    "content_capacity",
    "text_regions",
    "image_regions",
    "visual_density",
    "style_keywords",
    "color_palette",
    "notes",
}
TEMPLATE_ROLES = {"cover", "content", "section_divider", "summary", "data", "comparison", "timeline", "other"}
CAPACITY_VALUES = {"low", "medium", "high"}
REGION_POSITIONS = {"top", "center", "bottom", "left", "right"}
REGION_SIZES = {"small", "medium", "large"}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_density(slide, issues):
    sid = slide.get("slide_id", "?")
    body = slide.get("body_text") or []
    if isinstance(body, str):
        body = [body]
    if len(body) < MIN_BODY_TEXT_ITEMS:
        issues.append({
            "level": "error",
            "slide_id": sid,
            "message": f"body_text has only {len(body)} item(s); banana-slides' default density floor "
                       "expects 2-6 phrases/sentences per page (prompts.py:56), not a near-empty page",
        })
        return
    phrase_like = [item for item in body if len(str(item).strip()) >= MIN_ITEM_CHARS_FOR_PHRASE]
    if len(phrase_like) < MIN_PHRASE_ITEMS_REQUIRED:
        issues.append({
            "level": "error",
            "slide_id": sid,
            "message": f"body_text reads as isolated short tags (only {len(phrase_like)} item(s) >= "
                       f"{MIN_ITEM_CHARS_FOR_PHRASE} chars); banana-slides default density is "
                       "'每条要点控制在15-20字以内，优先使用短语而非完整句子' (prompts.py:56), not bare keywords",
        })


FABRICATED_DATE_PATTERN = re.compile(r"(19|20)\d{2}\s*年|(19|20)\d{2}\s*[-/]\s*(0?[1-9]|1[0-2])|Q[1-4]\s*(19|20)?\d{2}")


def check_cover(slides, issues, deck_title=""):
    if not slides:
        return
    cover = slides[0]
    sid = cover.get("slide_id", "01")
    page_role = cover.get("page_role")
    if page_role and page_role != "cover":
        issues.append({"level": "warning", "slide_id": sid, "message": f"first slide page_role is '{page_role}', expected 'cover'"})
    body = cover.get("body_text") or []
    if isinstance(body, str):
        body = [body]
    extra = cover.get("extra_fields") or {}
    layout_notes = str(extra.get("layout_notes") or "")
    has_subtitle_hint = any(
        keyword in " ".join(str(item) for item in body) + layout_notes
        for keyword in ["副标题", "演讲人", "团队", "日期", "presenter", "subtitle"]
    )
    if not has_subtitle_hint:
        issues.append({
            "level": "error",
            "slide_id": sid,
            "message": "cover slide has no subtitle/presenter-style content; banana-slides requires the first "
                       "page to contain 'title, subtitle, and presenter information' (prompts.py:292/340/591/642)",
        })
    combined_text = " ".join(str(item) for item in body) + " " + layout_notes
    fabricated = [m for m in FABRICATED_DATE_PATTERN.findall(combined_text) if not FABRICATED_DATE_PATTERN.search(deck_title)]
    if fabricated:
        issues.append({
            "level": "error",
            "slide_id": sid,
            "message": f"cover slide body_text/layout_notes contains a specific year/date ({combined_text.strip()[:80]}...) "
                       "that was not supplied by the user; this is very likely an LLM-fabricated date, not a real one. "
                       "Remove it or replace with a generic placeholder (e.g. team name only, no date).",
        })


def check_role_diversity(slides, issues):
    roles = [s.get("page_role") for s in slides if s.get("page_role")]
    if len(roles) < len(slides):
        issues.append({
            "level": "warning",
            "message": f"{len(slides) - len(roles)} slide(s) are missing page_role; role-diversity audit is incomplete",
        })
    if not roles:
        return
    if len(slides) >= 4 and not any(role == "content" for role in roles):
        issues.append({
            "level": "error",
            "message": "deck has 4+ slides but no page_role='content' (text-forward) slide; banana-slides' "
                       "template_role taxonomy treats 'content' as a first-class page type (prompts.py:1293), "
                       "not an optional extra",
        })
    diagram_count = sum(1 for role in roles if role in DIAGRAM_ROLES)
    if len(roles) >= 4 and diagram_count == len(roles):
        issues.append({
            "level": "error",
            "message": "every slide uses a diagram-style page_role (data/comparison/timeline); a deck should "
                       "read as a presentation, not a chain of infographics",
        })
    run_role, run_length = None, 0
    for role in roles:
        if role == run_role and role != "cover":
            run_length += 1
        else:
            run_role, run_length = role, 1
        if run_length >= 4:
            issues.append({
                "level": "warning",
                "message": f"page_role='{run_role}' repeats 4+ times in a row; banana-slides explicitly avoids "
                           "5 consecutive identical templates (prompts.py:1524)",
            })


def _check_regions(slide_id, profile, key, issues):
    regions = profile.get(key)
    if not isinstance(regions, list):
        issues.append({
            "level": "error",
            "slide_id": slide_id,
            "message": f"layout_profile.{key} must be an array, matching banana-slides template-analysis schema (prompts.py:1296/1299)",
        })
        return
    for idx, region in enumerate(regions):
        if not isinstance(region, dict):
            issues.append({"level": "error", "slide_id": slide_id, "message": f"layout_profile.{key}[{idx}] must be an object"})
            continue
        for field in ("name", "position", "size"):
            if not region.get(field):
                issues.append({"level": "error", "slide_id": slide_id, "message": f"layout_profile.{key}[{idx}] missing '{field}'"})
        if region.get("position") and region["position"] not in REGION_POSITIONS:
            issues.append({"level": "error", "slide_id": slide_id, "message": f"layout_profile.{key}[{idx}].position='{region['position']}' is not a banana enum"})
        if region.get("size") and region["size"] not in REGION_SIZES:
            issues.append({"level": "error", "slide_id": slide_id, "message": f"layout_profile.{key}[{idx}].size='{region['size']}' is not a banana enum"})


def check_layout_profile(slide, issues):
    sid = slide.get("slide_id", "?")
    role = slide.get("page_role")
    profile = slide.get("layout_profile")
    if not isinstance(profile, dict):
        issues.append({
            "level": "error",
            "slide_id": sid,
            "message": "slide is missing layout_profile; banana-slides template-analysis prompt uses a 9-field schema "
                       "(template_role/layout_structure/content_capacity/text_regions/image_regions/visual_density/style_keywords/color_palette/notes)",
        })
        return
    missing = sorted(LAYOUT_PROFILE_FIELDS - set(profile))
    if missing:
        issues.append({
            "level": "error",
            "slide_id": sid,
            "message": f"layout_profile missing field(s): {', '.join(missing)}; must mirror banana-slides 9-field schema (prompts.py:1293)",
        })
    template_role = profile.get("template_role")
    if template_role not in TEMPLATE_ROLES:
        issues.append({"level": "error", "slide_id": sid, "message": f"layout_profile.template_role='{template_role}' is not a banana enum"})
    expected_role = "section_divider" if role == "section_divider" else role
    if role and template_role and template_role != expected_role:
        issues.append({
            "level": "warning",
            "slide_id": sid,
            "message": f"layout_profile.template_role='{template_role}' does not match page_role='{role}'",
        })
    if profile.get("content_capacity") not in CAPACITY_VALUES:
        issues.append({"level": "error", "slide_id": sid, "message": f"layout_profile.content_capacity='{profile.get('content_capacity')}' is not low/medium/high"})
    if profile.get("visual_density") not in CAPACITY_VALUES:
        issues.append({"level": "error", "slide_id": sid, "message": f"layout_profile.visual_density='{profile.get('visual_density')}' is not low/medium/high"})
    if not profile.get("layout_structure"):
        issues.append({"level": "error", "slide_id": sid, "message": "layout_profile.layout_structure is empty"})
    _check_regions(sid, profile, "text_regions", issues)
    _check_regions(sid, profile, "image_regions", issues)
    if not isinstance(profile.get("style_keywords"), list):
        issues.append({"level": "error", "slide_id": sid, "message": "layout_profile.style_keywords must be an array"})
    if not isinstance(profile.get("color_palette"), list):
        issues.append({"level": "error", "slide_id": sid, "message": "layout_profile.color_palette must be an array"})
    text_region_names = " ".join(str(region.get("name", "")) for region in profile.get("text_regions", []) if isinstance(region, dict))
    image_region_names = " ".join(str(region.get("name", "")) for region in profile.get("image_regions", []) if isinstance(region, dict))
    if role == "content" and not any(marker in text_region_names for marker in ("body", "text", "正文", "left_body", "right_body")):
        issues.append({
            "level": "error",
            "slide_id": sid,
            "message": "content slide layout_profile must include a body/text region; otherwise the page can regress into a diagram-only visual",
        })
    if role == "cover" and any(marker in image_region_names for marker in ("chart", "diagram", "timeline", "flow")):
        issues.append({
            "level": "error",
            "slide_id": sid,
            "message": "cover layout_profile plans a chart/diagram/timeline/flow image region; banana cover prompt keeps page 1 simple",
        })


def validate(spec_path: Path) -> dict:
    spec = read_json(spec_path)
    slides = spec.get("slides") or []
    issues: list[dict] = []
    deck_title = str((spec.get("deck") or {}).get("title") or "")
    for slide in slides:
        check_density(slide, issues)
        check_layout_profile(slide, issues)
    check_cover(slides, issues, deck_title)
    check_role_diversity(slides, issues)

    status = "pass"
    if any(item["level"] == "error" for item in issues):
        status = "fail"
    elif issues:
        status = "warn"
    return {
        "schema": "visual_deck_content_density_roles_v1",
        "status": status,
        "slide_count": len(slides),
        "issue_count": len(issues),
        "issues": issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slide_spec", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report = validate(args.slide_spec)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
