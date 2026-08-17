#!/usr/bin/env python3
"""Build a deck slide image prompt from slide_spec and visual_style_contract.

This script is intentionally provider-agnostic. It prepares the text prompt only;
the existing image backend remains responsible for generation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


INTERNAL_FORBIDDEN = [
    "Purpose",
    "Visual brief",
    "visual_brief",
    "must_show",
    "slide_spec",
    "image_prompt",
    "page_description",
    "render_manifest",
    "extra_fields",
    "visual_elements",
    "visual_focus",
    "layout_notes",
    "layout_profile",
    "template_role",
    "layout_structure",
    "content_capacity",
    "text_regions",
    "image_regions",
    "visual_density",
]

# Only these extra_fields keys are design-facing; speaker_notes is presenter-only
# and must never reach the image prompt (mirrors banana-slides' image_prompt_extra_fields
# excluding 演讲者备注 by default).
IMAGE_SAFE_EXTRA_FIELDS = ["visual_elements", "visual_focus", "layout_notes"]
EXTRA_FIELD_LABELS = {
    "visual_elements": "视觉元素",
    "visual_focus": "视觉焦点",
    "layout_notes": "排版布局",
}

PPT_LANGUAGE_TEXT = {
    "zh": "PPT文字请使用全中文。",
    "ja": "PPTのテキストは全て日本語で出力してください。",
    "en": "Use English for PPT text.",
    "auto": "",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(f"- {as_text(item)}" for item in value if as_text(item))
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value).strip()


def find_slide(spec: dict[str, Any], slide_id: str) -> dict[str, Any]:
    for slide in spec.get("slides", []):
        if str(slide.get("slide_id", "")).zfill(2) == str(slide_id).zfill(2):
            return slide
    raise SystemExit(f"slide_id not found: {slide_id}")


def visible_text(slide: dict[str, Any]) -> list[str]:
    values: list[str] = []
    title = as_text(slide.get("title"))
    if title:
        values.append(title)
    body = slide.get("body_text", [])
    if isinstance(body, str):
        values.append(body)
    elif isinstance(body, list):
        values.extend(as_text(item) for item in body if as_text(item))
    for item in slide.get("must_show", []) or []:
        text = as_text(item)
        if text and text not in values:
            values.append(text)
    return values


def format_palette(palette: Any) -> str:
    if not isinstance(palette, dict):
        return as_text(palette)
    lines = []
    for role, entry in palette.items():
        if isinstance(entry, dict):
            name = as_text(entry.get("name"))
            hex_value = as_text(entry.get("hex"))
            label = f"{name} {hex_value}".strip() if (name or hex_value) else as_text(entry)
        else:
            label = as_text(entry)
        if label:
            lines.append(f"- {role}: {label}")
    return "\n".join(lines)


def format_typography(typography: Any) -> str:
    if not isinstance(typography, dict):
        return as_text(typography)
    order = ["cjk_font", "latin_font", "title_weight", "body_weight"]
    lines = []
    for key in order:
        if typography.get(key):
            lines.append(f"- {key}: {as_text(typography[key])}")
    for key, value in typography.items():
        if key not in order and value:
            lines.append(f"- {key}: {as_text(value)}")
    return "\n".join(lines)


def style_block(contract: dict[str, Any]) -> str:
    parts = []
    if contract.get("style_source"):
        parts.append(f"style_source: {as_text(contract['style_source'])}")
    if contract.get("style_name"):
        parts.append(f"style_name: {as_text(contract['style_name'])}")
    if contract.get("mood_and_lighting"):
        parts.append(f"mood_and_lighting: {as_text(contract['mood_and_lighting'])}")
    if contract.get("palette"):
        parts.append("palette:\n" + format_palette(contract["palette"]))
    if contract.get("accent_usage_limit"):
        parts.append(f"accent_usage_limit: {as_text(contract['accent_usage_limit'])}")
    if contract.get("typography"):
        parts.append("typography:\n" + format_typography(contract["typography"]))
    if contract.get("layout_density"):
        parts.append(f"layout_density: {as_text(contract['layout_density'])}")
    if contract.get("grid_and_spacing"):
        parts.append(f"grid_and_spacing: {as_text(contract['grid_and_spacing'])}")
    if contract.get("visual_language"):
        parts.append(f"visual_language: {as_text(contract['visual_language'])}")
    if contract.get("icon_style"):
        parts.append(f"icon_style: {as_text(contract['icon_style'])}")
    if contract.get("chart_rules") or contract.get("diagram_chart_rules"):
        parts.append(f"chart_rules: {as_text(contract.get('chart_rules') or contract.get('diagram_chart_rules'))}")
    if contract.get("cover_treatment"):
        parts.append(f"cover_treatment: {as_text(contract['cover_treatment'])}")
    if contract.get("rendering_descriptor"):
        parts.append(f"rendering_descriptor: {as_text(contract['rendering_descriptor'])}")
    if contract.get("reference_policy"):
        parts.append(f"reference_policy: {as_text(contract['reference_policy'])}")
    if contract.get("forbidden_patterns"):
        parts.append(f"forbidden_patterns: {as_text(contract['forbidden_patterns'])}")
    return "\n".join(parts)


def get_ppt_language_instruction(language: str | None) -> str:
    """Mirror banana-slides' get_ppt_language_instruction() mapping locally."""
    return PPT_LANGUAGE_TEXT.get(str(language or "zh"), PPT_LANGUAGE_TEXT["zh"])


def build_page_style_block(page_style_text: str) -> str:
    if not page_style_text.strip():
        return ""
    return (
        "\n\n<page_style>\n"
        f"{page_style_text.strip()}\n"
        "</page_style>\n"
        "- 必须遵循上述 page_style 中的视觉风格、配色、版式语言。"
    )


def build_material_images_note(has_material_images: bool, has_template: bool) -> str:
    if not has_material_images:
        return ""
    prefix = "除了模板参考图片（用于风格参考）外，还提供了额外的素材图片。" if has_template else "用户提供了额外的素材图片。"
    return (
        "\n\n提示："
        + prefix
        + "这些素材图片是可供挑选和使用的元素，你可以从这些素材图片中选择合适的图片、图标、图表或其他视觉元素"
        "直接整合到生成的PPT页面中。请根据页面内容的需要，智能地选择和组合这些素材图片中的元素。"
    )


def build_extra_requirements_text(slide: dict[str, Any], deck: dict[str, Any]) -> str:
    extra_requirements = (
        slide.get("extra_requirements")
        or slide.get("image_extra_requirements")
        or deck.get("extra_requirements")
        or deck.get("image_extra_requirements")
        or ""
    )
    text = as_text(extra_requirements)
    if not text:
        return ""
    return f"\n\n额外要求（请务必遵循）：\n{text}\n"


def extract_layout_requirement(slide: dict[str, Any]) -> str:
    """Pull the composition-critical part of layout_notes (grid/column/row structure,
    e.g. '2x2 four-quadrant matrix') into its own top-level block so it does not get
    diluted inside the much longer, mostly-repeated visual_style_contract text. A single
    clause buried in a long paragraph competes poorly against ~30 lines of boilerplate
    style text that repeats verbatim across every slide in the deck."""
    extra = slide.get("extra_fields") or {}
    layout_notes = as_text(extra.get("layout_notes"))
    profile = slide.get("layout_profile") or {}
    if not isinstance(profile, dict):
        return layout_notes
    parts = []
    if profile.get("layout_structure"):
        parts.append(f"layout_structure: {as_text(profile.get('layout_structure'))}")
    if profile.get("template_role"):
        parts.append(f"template_role: {as_text(profile.get('template_role'))}")
    if profile.get("content_capacity"):
        parts.append(f"content_capacity: {as_text(profile.get('content_capacity'))}")
    if profile.get("visual_density"):
        parts.append(f"visual_density: {as_text(profile.get('visual_density'))}")
    if profile.get("text_regions") is not None:
        parts.append("text_regions:\n" + as_text(profile.get("text_regions")))
    if profile.get("image_regions") is not None:
        parts.append("image_regions:\n" + as_text(profile.get("image_regions")))
    if profile.get("notes"):
        parts.append(f"notes: {as_text(profile.get('notes'))}")
    if layout_notes:
        parts.append("layout_notes:\n" + layout_notes)
    return "\n".join(part for part in parts if part.strip())


def layout_profile_block(slide: dict[str, Any]) -> str:
    profile = slide.get("layout_profile")
    if not isinstance(profile, dict):
        return ""
    return json.dumps(profile, ensure_ascii=False, indent=2)


def build_page_description(slide: dict[str, Any]) -> str:
    """Prefer the structured extra_fields brief; fall back to freeform page_description
    for decks written before extra_fields existed. speaker_notes is intentionally
    excluded here even when present, matching IMAGE_SAFE_EXTRA_FIELDS."""
    extra = slide.get("extra_fields")
    title = as_text(slide.get("title"))
    if isinstance(extra, dict) and any(as_text(extra.get(key)) for key in IMAGE_SAFE_EXTRA_FIELDS):
        parts = [f"页面标题：{title}" if title else ""]
        parts.append("页面文字：\n" + as_text(slide.get("body_text")))
        for key in IMAGE_SAFE_EXTRA_FIELDS:
            value = as_text(extra.get(key))
            if value:
                parts.append(f"{EXTRA_FIELD_LABELS[key]}：\n{value}")
        return "\n".join(part for part in parts if part.strip())

    return as_text(slide.get("page_description")) or "\n".join(
        part
        for part in [
            f"页面标题：{title}" if title else "",
            "页面文字：\n" + as_text(slide.get("body_text")),
            "视觉意图：\n" + as_text(slide.get("visual_brief")),
        ]
        if part.strip()
    )


def build_prompt(spec: dict[str, Any], contract: dict[str, Any], slide: dict[str, Any]) -> str:
    deck = spec.get("deck", {})
    language = deck.get("language", "zh")
    aspect_ratio = deck.get("aspect_ratio", "16:9")
    style_source = as_text(contract.get("style_source"))
    has_template = style_source in {"template_image", "style_reference", "extracted_style"} or bool(
        contract.get("template_image") or contract.get("reference_image")
    )
    slide_id = as_text(slide.get("slide_id"))
    title = as_text(slide.get("title"))
    page_desc = build_page_description(slide)
    layout_requirement = extract_layout_requirement(slide)
    layout_profile = layout_profile_block(slide)
    layout_profile_section = f"<layout_profile>\n{layout_profile}\n</layout_profile>\n" if layout_profile else ""
    page_style_text = style_block(contract)
    page_style_block = build_page_style_block(page_style_text)
    exact_text = visible_text(slide)
    reference_guard = deck.get("reference_guard", {}) if isinstance(deck.get("reference_guard"), dict) else {}
    forbidden = list(
        dict.fromkeys(
            INTERNAL_FORBIDDEN
            + list(contract.get("forbidden_patterns", []) or [])
            + list(reference_guard.get("forbidden_terms", []) or [])
        )
    )
    cover_note = ""
    if str(slide_id).zfill(2) == "01":
        cover_note = "\n- 这是封面页，必须保持简洁：只突出标题、副标题和汇报人/团队信息；不要生成流程图、机制图、数据图或密集装饰。"

    material_refs = (
        slide.get("material_references")
        or slide.get("source_figures")
        or slide.get("materials")
        or []
    )
    material_note = ""
    if material_refs:
        material_note = (
            "\n<material_references>\n"
            f"{as_text(material_refs)}\n"
            "</material_references>\n"
            "素材说明：这些素材只作为可选视觉元素或图表依据，请按页面需要选择整合，不要把素材路径、文件名或说明文字直接渲染到页面上。\n"
        )
    material_images_note = build_material_images_note(bool(material_refs), has_template)
    extra_req_text = build_extra_requirements_text(slide, deck)

    template_style_guideline = "- 配色和设计语言和模板图片严格相似。" if has_template else "- 严格按照风格描述进行设计。"
    forbidden_template_text_guideline = "- 只参考风格设计，禁止出现模板中的文字。\n" if has_template else ""
    reference_guard_guideline = (
        "- 参考图仅用于统一整体配色、字体质感、卡片/图标视觉语言与装饰纹理这类风格锚点；"
        "本页的具体构图、图表类型、节点数量和文字内容必须严格按照 page_description 重新设计，"
        "不要照抄参考图里的版式结构、图形数量或具体内容，也不要复制参考图中的文字。"
        if has_template
        else "- 没有模板图时也必须保持统一、具体、专业的页面风格。"
    )
    resolution = as_text(contract.get("resolution")) or "4K 或当前图片服务支持的最高分辨率"

    layout_requirement_block = (
        f"\n<layout_requirement priority=\"highest\">\n"
        f"{layout_requirement}\n"
        "这是本页最重要的结构性要求，优先级高于下面 page_style 里的通用风格描述；"
        "如果这里描述了具体的网格/行列数（如 2x2、三栏、横向五步），最终构图必须严格是这个行列结构，"
        "不能因为风格契约或参考图的视觉习惯而改成其他行列数或方向。\n"
        "</layout_requirement>\n"
        if layout_requirement
        else ""
    )

    cover_prompt = (
        "**注意：当前页面为ppt的封面页，请你采用专业的封面设计美学技巧，务必凸显出页面标题，分清主次，确保一下就能抓住观众的注意力。**"
        if str(slide_id).zfill(2) == "01"
        else ""
    )

    prompt = f"""你是一位专家级UI UX演示设计师，专注于生成设计良好的PPT页面。
当前PPT页面的页面描述如下:

<page_description>
{page_desc}
</page_description>
{page_style_block}
{layout_profile_section}
{layout_requirement_block}

{material_note}
<exact_visible_text>
{chr(10).join(f"- {text}" for text in exact_text)}
</exact_visible_text>

<design_guidelines>
- 生成一张完整成品 PPT 页面，不要生成草稿、线框图、多个方案或解释文字。
- 要求文字清晰锐利, 画面为{resolution}，{aspect_ratio}比例。
{template_style_guideline}
{reference_guard_guideline}
- 根据内容和要求自动设计最完美的构图，不重不漏地渲染"页面文字"段落中的文本。
- 如非必要，禁止出现 markdown 格式符号（如 # 和 * 等）。
{forbidden_template_text_guideline}
- 所有可见正文必须遵循下方 PPT 语言限制，除非 exact_visible_text 中明确要求其他语言。
- 文字必须清晰锐利，标题、正文、注释层级分明，避免拥挤、错字、伪英文和随机字符。
- exact_visible_text 是允许出现的文字白名单，不是重复次数要求；除非页面描述明确要求重复，同一个标签只能出现一次。
- 严禁在 exact_visible_text 之外新增任何看起来像文字的内容，包括小标签、按钮文案、图例脚注、界面模拟文字或装饰性文字；如果视觉元素里提到"UI概念图""界面示意"，用无文字的线框、图标、占位色块或几何图形表示界面结构，不要编造具体文字或伪造看似有意义的字符组合。
- page_style 里的 accent_usage_limit 是硬约束：强调色/主色只能用于图标高亮、细线条、关键数字或极小面积的状态标记，不能作为整张卡片、整块背景或大色块的填充色；卡片主体必须使用背景色或浅色调，保持轻量描边风格。
- 根据页面描述自动设计最合适的构图，不重不漏地渲染 exact_visible_text 和必要的极短图例标签。
- 如果存在 layout_requirement 区块，其中描述的具体网格/行列/分栏结构是硬性要求，必须严格遵守，不能简化成其他排列方式。
- layout_profile 是结构化版式契约：content_capacity 和 visual_density 决定页面承载量；text_regions 决定文字必须落在哪些区域；image_regions 决定图形/配图只能服务于这些区域。不要把 content 页画成没有正文区域的概念图。
- 如果 text_regions 与 image_regions 同时存在，优先保证 text_regions 里的标题、正文、图例等文字清晰可读，再安排图形区域；不要让图形侵占正文区。
- 流程图、机制图、决策图中必须遵守“一个逻辑步骤 = 一个视觉节点”；不要为了填充空间复制同名卡片、重复同名分支或增加语义相同的节点。
- 每一段可见文字必须放在对应卡片、节点、标题区或图例区域内部，不得出现漂浮在卡片外、压在线条上、贴近边缘或脱离语义归属的文字。
- 如果页面描述包含分支、质量门、候选图、原图、人工判断等路径，分支含义必须彼此区分清楚，不能把两个不同路径画成重复卡片。
- 版式要像可直接放入正式汇报的 PPT 页面：有明确焦点、稳定留白、统一配色、完整视觉层级。
- 如果需要图表、流程、机制图或对比结构，使用简洁可读的专业图形，不要堆砌装饰。
- 图标必须服从 page_style 的 icon_style；如果要求线性描边图标，就不要使用实心圆点、徽章、贴纸或杂乱装饰来替代。
- 禁止出现提示词字段名、内部元数据、水印、logo、占位符和模板原文。{cover_note}
</design_guidelines>
{get_ppt_language_instruction(language)}
{material_images_note}{extra_req_text}

{cover_prompt}

<forbidden_visible_text>
{chr(10).join(f"- {text}" for text in forbidden if as_text(text))}
</forbidden_visible_text>
"""
    return prompt.strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slide_spec", type=Path)
    parser.add_argument("visual_style_contract", type=Path)
    parser.add_argument("--slide-id", required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    spec = load_json(args.slide_spec)
    contract = load_json(args.visual_style_contract)
    slide = find_slide(spec, args.slide_id)
    prompt = build_prompt(spec, contract, slide)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(prompt, encoding="utf-8")
    else:
        print(prompt, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
