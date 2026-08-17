#!/usr/bin/env python3
"""Real content-planning generator, with four input modes mirroring banana-slides'
`backend/services/prompts.py` content-ingestion functions:

1. `--topic` (default): topic -> LLM outline -> LLM per-page description.
   Mirrors `get_outline_generation_prompt` + `get_page_description_prompt`.
2. `--outline-text <file>`: user already wrote an outline; parse it into structured
   form WITHOUT rewriting any wording, then LLM-write per-page description from it.
   Mirrors `get_outline_parsing_prompt` (prompts.py:352).
3. `--description-text <file>`: user provided a full article/description with no
   explicit page split; LLM derives an outline from it, then LLM-writes per-page
   description as usual. Mirrors `get_description_to_outline_prompt` (prompts.py:413).
4. `--existing-page-text <file>` (requires an outline from mode 1/2/3, or
   `--outline-text` combined): user already wrote finished, page-by-page copy;
   split it to match the outline WITHOUT rewriting wording (no description-writing
   LLM call). Mirrors `get_description_split_prompt` (prompts.py:680).

Modes 2-4 exist because mode 1 alone only covers "give a topic, let the LLM plan
everything" -- if the user already has an outline, a full article, or finished
per-page copy, forcing it all back into a bare topic string and re-generating from
scratch would silently discard or paraphrase content the user already wrote.

All modes still enforce the content-quality floor that was previously missing from
this skill entirely:

- the cover page carries only title + subtitle + presenter/team/date, never a
  diagram (banana-slides prompts.py:292/340/591/642), and never a fabricated
  specific year/date;
- body text follows a density floor equivalent to banana's DEFAULT
  `DETAIL_LEVEL_SPECS` entry (2-6 short sentences/phrases per page, not
  isolated 2-4 character labels);
- every page is assigned a `page_role` (cover/content/data/comparison/
  timeline/summary/section_divider), mirroring banana's `template_role`
  9-field template-analysis schema (prompts.py:1293), and a deck with 4+
  pages must include at least one `content` (text-forward) page instead of
  defaulting every page to a diagram.

Uses the same DashScope-compatible chat endpoint already wired for VLM
review and the Stylist pass (text-only calls, no image).
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import ssl
import time
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
DEFAULT_MODEL = "qwen3.7-plus"

PAGE_ROLES = ["cover", "content", "data", "comparison", "timeline", "summary", "section_divider"]
CONTENT_CAPACITIES = {"low", "medium", "high"}
VISUAL_DENSITIES = {"low", "medium", "high"}
REGION_POSITIONS = {"top", "center", "bottom", "left", "right"}
REGION_SIZES = {"small", "medium", "large"}
DETAIL_LEVEL_SPECS = {
    "concise": "文字极致地压缩和精简，每条要点用一个核心词语或数据代替，例如效率↑80%",
    "default": "清晰明了，每条要点控制在15-20字以内，优先使用短语而非完整句子；落地到页面的文字建议在2-6句之内，避免冗长和复杂表述，为演示服务，而不是代替演讲人叙述。",
    "detailed": "忠于原文的基础上做到内容详实，逻辑清晰。",
}


OUTLINE_JSON_FORMAT = """\
1. Simple format (for short PPTs without major sections):
[
  {"title": "title1", "intent": "slide purpose", "page_role": "cover|content|data|comparison|timeline|summary|section_divider", "points": ["point1", "point2"], "layout_profile": {...}},
  {"title": "title2", "intent": "slide purpose", "page_role": "content|data|comparison|timeline|summary|section_divider", "points": ["point1", "point2"], "layout_profile": {...}}
]

2. Part-based format (for longer PPTs with major sections):
[
  {
    "part": "Part 1: Introduction",
    "pages": [
      {"title": "Welcome", "intent": "slide purpose", "page_role": "cover", "points": ["point1", "point2"], "layout_profile": {...}},
      {"title": "Overview", "intent": "slide purpose", "page_role": "content", "points": ["point1", "point2"], "layout_profile": {...}}
    ]
  },
  {
    "part": "Part 2: Main Content",
    "pages": [
      {"title": "Topic 1", "intent": "slide purpose", "page_role": "data", "points": ["point1", "point2"], "layout_profile": {...}},
      {"title": "Topic 2", "intent": "slide purpose", "page_role": "comparison", "points": ["point1", "point2"], "layout_profile": {...}}
    ]
  }
]"""


LAYOUT_PROFILE_INSTRUCTION = """\
## 版式结构字段（必须输出，源码级对齐 banana-slides get_template_analysis_prompt 的 9 字段）
每页必须额外输出 layout_profile。它不是给用户看的文案，而是后续图像 prompt 和模板匹配使用的结构化版式契约：
{
  "template_role": "cover|content|section_divider|summary|data|comparison|timeline|other",
  "layout_structure": "kebab-case 版式摘要，如 title-top-two-column / centered-title-subtitle-presenter / horizontal-timeline-five-nodes",
  "content_capacity": "low|medium|high",
  "text_regions": [
    {"name": "title|subtitle|presenter|body|left_body|right_body|node_labels|legend", "position": "top|center|bottom|left|right", "size": "small|medium|large"}
  ],
  "image_regions": [
    {"name": "hero|chart|diagram|node_icons|support_visual|background", "position": "top|center|bottom|left|right", "size": "small|medium|large"}
  ],
  "visual_density": "low|medium|high",
  "style_keywords": ["最多 5 个英文形容词，如 academic / clean / minimalist / professional"],
  "color_palette": ["最多 5 个主色 hex，如 #FFFFFF"],
  "notes": "一两句话补充固定版式约束，不超过 80 字"
}

硬约束：
- text_regions 和 image_regions 必须存在；没有图片区域时 image_regions 输出 []。
- template_role 必须和 page_role 对应；section_divider 可对应 page_role=section_divider。
- content 页必须明确 body/left_body/right_body 等文字区域，不能只写 diagram/chart。
- cover 页必须包含 title、subtitle、presenter/team/date 信息区，image_regions 只能是 background 或极弱装饰，不得规划流程图/机制图/数据图。
"""

OUTLINE_SYSTEM_PROMPT = """\
## 角色
你是一名 PPT 内容策划师，负责把用户的主题拆成一份大纲。

## 输出要求
严格输出 JSON 数组，不要 Markdown 代码块，不要解释性文字。
你可以按 banana-slides 的两种大纲格式组织内容：

{outline_json_format}

选择最适合内容的格式；短 PPT 用 simple format，长 PPT 或有清楚章节时用 part-based format。
无论使用哪种格式，每一个 page 对象都必须包含：
{
  "title": "页面标题",
  "intent": "一句话说明这一页要讲什么、为什么需要这一页",
  "page_role": "cover|content|data|comparison|timeline|summary|section_divider",
  "points": ["要点1", "要点2", "..."],
  "layout_profile": {
    "template_role": "cover|content|section_divider|summary|data|comparison|timeline|other",
    "layout_structure": "kebab-case 版式摘要",
    "content_capacity": "low|medium|high",
    "text_regions": [{"name": "title", "position": "top|center|bottom|left|right", "size": "small|medium|large"}],
    "image_regions": [],
    "visual_density": "low|medium|high",
    "style_keywords": ["academic", "clean"],
    "color_palette": ["#FFFFFF"],
    "notes": "一两句话补充版式约束"
  }
}

{layout_profile_instruction}

## 铁律
1. 第一页 page_role 必须是 "cover"；封面页只需要标题 + 副标题方向 + 演讲人/团队/日期方向，points 里只放这两类信息的建议内容，不要放正文论点，不要放图表构图想法。
   - 严禁编造具体的年份、月份或日期（如"2023年10月""2024年Q3"）。用户没有提供真实日期时，演讲人/团队信息只能写团队名称或角色，不写任何具体年月；如果确实需要日期占位，只能写通用词（如"内部评审"），不能编造看起来真实的具体时间点。
2. 页面角色要根据内容真实选择，不能每一页都选同一个角色。如果总页数 >= 4，至少要有 1 页 page_role="content"（以文字论述为主，不是图表/流程图为主的页面）。
3. "content" 角色的页面适合放置需要文字说明、背景介绍、论述、总结陈词类内容；"data/comparison/timeline" 适合放置数据对比、流程步骤、时间线这类适合图形化的内容；不要把每一页都塞成图表。
4. 大纲阶段只定方向和意图，不要写最终成稿文案。
5. 页数以用户指定为准；未指定时选择能讲清楚主题的合理页数（4-10 页）。
6. 如果内容有清楚的大章节，使用 part-based format；否则使用 simple format。不要为了形式强行分章节。

只输出 JSON 数组本身。
"""

OUTLINE_PARSING_SYSTEM_PROMPT = """\
## 角色
你是一名 PPT 大纲整理助手，负责把用户已经写好的大纲原文转换成结构化 JSON，只做格式重组，不做内容创作。

## 铁律（不可违反）
1. 绝对不能修改、改写、润色或替换用户原文中的任何文字表述；标题和要点必须逐字摘录原文。
2. 绝对不能新增用户原文里没有出现过的内容、论点或想法。
3. 绝对不能删除用户原文里的内容；如果原文有些页面信息不全，也原样保留，不替用户编造补全。
4. 只允许做的事情：把原文重新组织成一页一页的结构，并为每页归纳一个 page_role 分类（这属于分类整理，不算改写原文）。
5. 如果原文本身没有写出封面页的副标题/演讲人信息，不要替用户编造这些信息；第一页 page_role 仍标注为 "cover"，points 只放原文中确实出现的内容。

## 输出格式
严格输出 JSON 数组，不要 Markdown 代码块，不要解释性文字。
你可以按 banana-slides 的两种大纲格式组织内容：

{outline_json_format}

选择最适合原文结构的格式；如果原文有清楚章节/部分，使用 part-based format。
无论使用哪种格式，每一个 page 对象都必须包含：
{
  "title": "页面标题（原文摘录，逐字不改）",
  "intent": "根据原文内容归纳这一页的定位（这是你可以做的归纳判断，不算改写原文正文）",
  "page_role": "cover|content|data|comparison|timeline|summary|section_divider",
  "points": ["原文中的要点，逐条摘录，逐字不改"],
  "layout_profile": {
    "template_role": "cover|content|section_divider|summary|data|comparison|timeline|other",
    "layout_structure": "kebab-case 版式摘要",
    "content_capacity": "low|medium|high",
    "text_regions": [{"name": "title", "position": "top|center|bottom|left|right", "size": "small|medium|large"}],
    "image_regions": [],
    "visual_density": "low|medium|high",
    "style_keywords": ["academic", "clean"],
    "color_palette": ["#FFFFFF"],
    "notes": "一两句话补充版式约束"
  }
}

{layout_profile_instruction}

只输出 JSON 数组本身。
"""

DESCRIPTION_TO_OUTLINE_SYSTEM_PROMPT = """\
## 角色
你是一名 PPT 内容策划师，负责从用户提供的一段完整描述/文章文本中，提炼出适合做成 PPT 的大纲结构。

## 任务
分析这段文本的内容和逻辑结构，判断：
1. 这段内容自然地应该分成几页（不要为了凑页数而拆分或合并，也不要机械按字数切分）
2. 每页的标题
3. 每页的核心要点归纳（这是你对原文内容的提炼总结，不是逐字摘抄）

## 铁律
1. 第一页 page_role 必须是 "cover"；封面页只需要标题 + 副标题方向 + 演讲人/团队/日期方向，points 里只放这两类信息的建议内容，不要放正文论点。
   - 严禁编造具体的年份、月份或日期。除非原文本身包含明确日期，否则演讲人/团队信息只能写角色或团队名称，不写具体年月。
2. 页面角色要根据内容真实选择，不能每一页都选同一个角色。如果总页数 >= 4，至少要有 1 页 page_role="content"。
3. 大纲阶段只提炼结构和方向，不要把原文整段照抄进 points，points 应该是简洁的归纳短语。
4. 页数由文本内容的自然结构和篇幅决定，未特别要求页数时选择能讲清楚这段内容的合理页数。

## 输出格式
严格输出 JSON 数组，不要 Markdown 代码块，不要解释性文字。
你可以按 banana-slides 的两种大纲格式组织内容：

{outline_json_format}

选择最适合文本逻辑的格式；如果文本有清楚章节/部分，使用 part-based format。
无论使用哪种格式，每一个 page 对象都必须包含：
{
  "title": "页面标题",
  "intent": "一句话说明这一页要讲什么",
  "page_role": "cover|content|data|comparison|timeline|summary|section_divider",
  "points": ["提炼出的要点1", "要点2", "..."],
  "layout_profile": {
    "template_role": "cover|content|section_divider|summary|data|comparison|timeline|other",
    "layout_structure": "kebab-case 版式摘要",
    "content_capacity": "low|medium|high",
    "text_regions": [{"name": "title", "position": "top|center|bottom|left|right", "size": "small|medium|large"}],
    "image_regions": [],
    "visual_density": "low|medium|high",
    "style_keywords": ["academic", "clean"],
    "color_palette": ["#FFFFFF"],
    "notes": "一两句话补充版式约束"
  }
}

{layout_profile_instruction}

只输出 JSON 数组本身。
"""

DESCRIPTION_SPLIT_SYSTEM_PROMPT = """\
## 角色
你是一名 PPT 文案整理助手，负责把用户已经写好的完整逐页文案，按照给定大纲切分、归类成每一页对应的结构化内容，不改写用户的原文文字。

## 铁律（不可违反）
1. 绝对不能修改、改写、润色或替换用户原文中的任何文字表述；body_text 必须逐字或逐句摘录原文，不能替用户重新表达。
2. 绝对不能新增用户原文里没有的内容、数据或视觉设计描述；如果原文没有提到具体的排版/风格/素材要求，extra_fields 里对应字段可以留空字符串，不要凭空编造。
3. 如果某一页在原文中找不到明确对应的内容，body_text 可以只包含大纲里的要点，不要替用户编造更多正文。
4. 只允许做的事情：判断原文里的哪一段落属于哪一页，并识别原文中确实提到的排版/风格/素材描述放进 extra_fields。

## 输出格式
严格输出 JSON 数组，元素数量和顺序必须和输入大纲完全一致，不要省略任何一页：
{
  "slide_id": "必须和输入的 slide_id 完全一致",
  "body_text": ["原文摘录的正文条目，逐字不改写"],
  "must_show": ["从原文摘录的关键词，用于图片生成时的白名单"],
  "layout_profile": {
    "template_role": "cover|content|section_divider|summary|data|comparison|timeline|other",
    "layout_structure": "kebab-case 版式摘要",
    "content_capacity": "low|medium|high",
    "text_regions": [{"name": "title", "position": "top|center|bottom|left|right", "size": "small|medium|large"}],
    "image_regions": [],
    "visual_density": "low|medium|high",
    "style_keywords": ["academic", "clean"],
    "color_palette": ["#FFFFFF"],
    "notes": "只记录原文确实提到或由大纲结构直接推出的版式约束"
  },
  "extra_fields": {
    "visual_elements": "原文中确实提到的视觉元素描述；原文没提到就留空字符串",
    "visual_focus": "原文中确实提到的视觉焦点；原文没提到就留空字符串",
    "layout_notes": "原文中确实提到的排版描述；原文没提到就留空字符串",
    "speaker_notes": "原文中确实提到的演讲备注；原文没提到就留空字符串"
  }
}

只输出 JSON 数组本身。
"""

def build_description_system_prompt(detail_level: str = "default", page_index: int | None = None) -> str:
    detail_spec = DETAIL_LEVEL_SPECS.get(detail_level, DETAIL_LEVEL_SPECS["default"])
    cover_instruction = (
        "\n**除非特殊要求，第一页的内容需要保持极简，只放标题副标题以及演讲人等（输出到标题后）, 不添加任何素材。**\n"
        if page_index == 1
        else ""
    )
    return f"""\
我们正在为PPT的每一页生成内容描述。
{cover_instruction}

## 文字密度规范（必须遵守）
细致程度要求：{detail_spec}
不允许整页只有 1-2 个孤立的 2-4 字关键词标签，除非 detail_level=concise 且该页确实只适合极短关键词。

{LAYOUT_PROFILE_INSTRUCTION}

## 页面类型与构图倾向
- page_role="cover"：只有标题、副标题（一句话概括主题/价值主张）、演讲人或团队或日期信息；不写正文论点；layout_notes 里必须明确写出"副标题"和"演讲人/团队/日期"分别放在什么位置。
  - 严禁编造具体年份、月份或日期（如"2023年10月""2024年Q3""2025年"）。除非用户在主题/需求里明确给出了真实日期，演讲人信息只能写团队名称或角色（如"产品体验团队""内部评审组"），不得附加任何编造出来的具体时间点。
- page_role="content"：以文字论述为主，layout_notes 必须描述一个文字主导的版式（例如左右两栏文字、上下分段落、要点列表配极简图标），不能写成"横向流程图"这类图表主导的构图。
- page_role 是 "data"/"comparison"/"timeline"/"section_divider"/"summary" 时：可以用图表/流程图/矩阵/时间线等图形化构图，但仍要保证 exact_visible_text 达到密度规范。

## 每页需要输出
{{
  "slide_id": "必须和输入的 slide_id 完全一致",
  "body_text": ["按密度规范写的正文文字条目，cover 页是 [副标题文案, 演讲人/团队/日期文案]"],
  "must_show": ["页面里必须原样出现的关键词，取自 body_text 或大纲要点"],
  "layout_profile": {{
    "template_role": "cover|content|section_divider|summary|data|comparison|timeline|other",
    "layout_structure": "kebab-case 版式摘要",
    "content_capacity": "low|medium|high",
    "text_regions": [{{"name": "title", "position": "top|center|bottom|left|right", "size": "small|medium|large"}}],
    "image_regions": [],
    "visual_density": "low|medium|high",
    "style_keywords": ["academic", "clean"],
    "color_palette": ["#FFFFFF"],
    "notes": "一两句话补充版式约束"
  }},
  "extra_fields": {{
    "visual_elements": "这一页需要的具体视觉元素",
    "visual_focus": "这一页最应该第一眼被看到的焦点",
    "layout_notes": "具体版式：明确写出标题区/正文文字区/配图区各自的位置和相对大小（例如“标题区在顶部；正文文字区占左侧60%宽度，中等容量；配图区占右侧40%宽度，小尺寸点缀”），cover 页要写清楚副标题和演讲人信息各自的位置",
    "speaker_notes": "演讲人备注，不会进图"
  }}
}}

## 重要提示
- 生成的"页面文字"部分会直接渲染到PPT页面上，因此请务必不要包含任何额外的说明性文字或注释，也不要把用户的设计意图显式地放在页面文字中。
- body_text 会直接决定页面上出现的文字，要写真正会印在页面上的文案；设计意图放在 extra_fields 里。
- 如果参考文件中包含以 /files/ 开头的本地文件URL图片（例如 /files/mineru/xxx/image.png），请在 extra_fields.visual_elements 或 material_references 中保留这些图片线索，后续图片生成会使用它们。
- 输出必须是严格 JSON 对象，只输出当前这一页，不要 Markdown 代码块，不要解释性文字。
"""

OUTLINE_SYSTEM_PROMPT = OUTLINE_SYSTEM_PROMPT.replace("{layout_profile_instruction}", LAYOUT_PROFILE_INSTRUCTION)
OUTLINE_PARSING_SYSTEM_PROMPT = OUTLINE_PARSING_SYSTEM_PROMPT.replace("{layout_profile_instruction}", LAYOUT_PROFILE_INSTRUCTION)
DESCRIPTION_TO_OUTLINE_SYSTEM_PROMPT = DESCRIPTION_TO_OUTLINE_SYSTEM_PROMPT.replace("{layout_profile_instruction}", LAYOUT_PROFILE_INSTRUCTION)
OUTLINE_SYSTEM_PROMPT = OUTLINE_SYSTEM_PROMPT.replace("{outline_json_format}", OUTLINE_JSON_FORMAT)
OUTLINE_PARSING_SYSTEM_PROMPT = OUTLINE_PARSING_SYSTEM_PROMPT.replace("{outline_json_format}", OUTLINE_JSON_FORMAT)
DESCRIPTION_TO_OUTLINE_SYSTEM_PROMPT = DESCRIPTION_TO_OUTLINE_SYSTEM_PROMPT.replace("{outline_json_format}", OUTLINE_JSON_FORMAT)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def format_reference_files_xml(reference_files_content: list[dict[str, str]] | None) -> str:
    """Mirror banana-slides `_format_reference_files_xml`: prepend uploaded files
    as XML before content-planning prompts."""
    if not reference_files_content:
        return ""
    xml_parts = ["<uploaded_files>"]
    for file_info in reference_files_content:
        filename = file_info.get("filename", "unknown")
        content = file_info.get("content", "")
        xml_parts.append(f'  <file name="{filename}">')
        xml_parts.append("    <content>")
        xml_parts.append(content)
        xml_parts.append("    </content>")
        xml_parts.append("  </file>")
    xml_parts.append("</uploaded_files>")
    xml_parts.append("")
    return "\n".join(xml_parts)


def build_prompt(prompt_text: str, reference_files_content: list[dict[str, str]] | None = None) -> str:
    return format_reference_files_xml(reference_files_content) + prompt_text


def load_reference_files(paths: list[Path] | None) -> list[dict[str, str]]:
    refs = []
    for path in paths or []:
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = path.read_text(encoding="utf-8", errors="replace")
        refs.append({"filename": path.name, "content": content})
    return refs


def call_dashscope_json(model: str, api_key: str, system_prompt: str, user_prompt: str, timeout: int) -> dict:
    # Deliberately no response_format=json_object: the outline/description
    # payloads are top-level JSON *arrays*, and DashScope's json_object mode
    # requires a top-level object. The system prompts already demand strict
    # JSON output; extract_json_payload() strips optional markdown fences.
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.4,
    }
    request = Request(
        ENDPOINT,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + api_key},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            text = response.read().decode("utf-8", errors="replace")
            return {"status_code": response.status, "json": json.loads(text), "text": text}
    except HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = {"raw_text": text}
        return {"status_code": exc.code, "json": data, "text": text}
    except URLError as exc:
        # Includes SSL-layer drops (e.g. "UNEXPECTED_EOF_WHILE_READING") wrapped by urlopen;
        # classify by the underlying reason's type name so the retry wrapper can catch them too.
        reason = getattr(exc, "reason", None)
        return {
            "status_code": None,
            "json": {"error": str(exc), "type": type(reason).__name__ if reason is not None else "URLError"},
            "text": str(exc),
        }
    except (ConnectionError, http.client.HTTPException, TimeoutError, OSError) as exc:
        return {"status_code": None, "json": {"error": str(exc), "type": type(exc).__name__}, "text": str(exc)}


def is_transient_network_error(response: dict) -> bool:
    if response.get("status_code") is not None:
        return False
    error_type = ""
    error_text = ""
    if isinstance(response.get("json"), dict):
        error_type = str(response["json"].get("type") or "")
    error_text = str(response.get("text") or "")
    if error_type in {
        "RemoteDisconnected",
        "ConnectionResetError",
        "ConnectionAbortedError",
        "BrokenPipeError",
        "TimeoutError",
        "IncompleteRead",
        "SSLError",
        "SSLEOFError",
        "URLError",
    }:
        return True
    # Fallback substring match: some SSL/urllib exceptions surface only in the message text.
    return any(marker in error_text for marker in ("SSL", "EOF occurred", "Remote end closed", "Connection reset"))


def call_dashscope_json_with_retry(model, api_key, system_prompt, user_prompt, timeout, attempts=4, base_backoff_seconds=8):
    last = None
    for attempt in range(1, max(1, attempts) + 1):
        last = call_dashscope_json(model, api_key, system_prompt, user_prompt, timeout)
        if not is_transient_network_error(last):
            return last
        if attempt < attempts:
            time.sleep(base_backoff_seconds * (2 ** (attempt - 1)))
    return last


def extract_json_payload(response: dict):
    content = response["json"]["choices"][0]["message"]["content"]
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)


def default_layout_profile(page_role: str) -> dict:
    role = page_role if page_role in PAGE_ROLES else "content"
    template_role = "section_divider" if role == "section_divider" else role
    defaults = {
        "cover": {
            "layout_structure": "centered-title-subtitle-presenter",
            "content_capacity": "low",
            "text_regions": [
                {"name": "title", "position": "center", "size": "large"},
                {"name": "subtitle", "position": "center", "size": "medium"},
                {"name": "presenter", "position": "bottom", "size": "small"},
            ],
            "image_regions": [{"name": "background", "position": "center", "size": "large"}],
            "visual_density": "low",
            "notes": "封面保持简洁，只承载标题、副标题和汇报人/团队信息。",
        },
        "content": {
            "layout_structure": "title-top-two-column-text",
            "content_capacity": "medium",
            "text_regions": [
                {"name": "title", "position": "top", "size": "medium"},
                {"name": "left_body", "position": "left", "size": "medium"},
                {"name": "right_body", "position": "right", "size": "medium"},
            ],
            "image_regions": [],
            "visual_density": "medium",
            "notes": "文字主导页面，优先保证正文区域清晰，不退化成概念图。",
        },
        "data": {
            "layout_structure": "title-top-chart-center",
            "content_capacity": "high",
            "text_regions": [
                {"name": "title", "position": "top", "size": "medium"},
                {"name": "chart_labels", "position": "center", "size": "small"},
            ],
            "image_regions": [{"name": "chart", "position": "center", "size": "large"}],
            "visual_density": "high",
            "notes": "数据或结构图页面，图形区域必须服务于页面文字和指标。",
        },
        "comparison": {
            "layout_structure": "title-top-two-column-comparison",
            "content_capacity": "high",
            "text_regions": [
                {"name": "title", "position": "top", "size": "medium"},
                {"name": "left_body", "position": "left", "size": "medium"},
                {"name": "right_body", "position": "right", "size": "medium"},
            ],
            "image_regions": [{"name": "comparison_cards", "position": "center", "size": "large"}],
            "visual_density": "high",
            "notes": "左右或矩阵对比，两个比较对象必须清楚分区。",
        },
        "timeline": {
            "layout_structure": "horizontal-timeline-five-nodes",
            "content_capacity": "high",
            "text_regions": [
                {"name": "title", "position": "top", "size": "medium"},
                {"name": "node_labels", "position": "center", "size": "small"},
            ],
            "image_regions": [{"name": "node_icons", "position": "center", "size": "small"}],
            "visual_density": "high",
            "notes": "时间线或流程节点等距排列，节点数量以页面文字为准。",
        },
        "summary": {
            "layout_structure": "title-top-summary-cards",
            "content_capacity": "medium",
            "text_regions": [
                {"name": "title", "position": "top", "size": "medium"},
                {"name": "summary_body", "position": "center", "size": "medium"},
            ],
            "image_regions": [{"name": "support_visual", "position": "right", "size": "small"}],
            "visual_density": "medium",
            "notes": "结论页突出少量关键判断，避免复杂流程图。",
        },
        "section_divider": {
            "layout_structure": "section-title-large-divider",
            "content_capacity": "low",
            "text_regions": [
                {"name": "title", "position": "center", "size": "large"},
                {"name": "section_label", "position": "top", "size": "small"},
            ],
            "image_regions": [],
            "visual_density": "low",
            "notes": "章节页低密度，承担节奏切换，不承载复杂正文。",
        },
    }
    profile = {
        "template_role": template_role,
        "style_keywords": ["academic", "clean", "professional"],
        "color_palette": [],
    }
    profile.update(defaults.get(role, defaults["content"]))
    return profile


def normalize_region(region: dict, fallback_name: str) -> dict:
    if not isinstance(region, dict):
        region = {}
    position = str(region.get("position") or "center")
    size = str(region.get("size") or "medium")
    return {
        "name": str(region.get("name") or fallback_name),
        "position": position if position in REGION_POSITIONS else "center",
        "size": size if size in REGION_SIZES else "medium",
    }


def normalize_layout_profile(value, page_role: str) -> dict:
    base = default_layout_profile(page_role)
    if isinstance(value, dict):
        base.update({k: v for k, v in value.items() if v not in (None, "")})
    role = str(base.get("template_role") or page_role or "content")
    if role == "section-divider":
        role = "section_divider"
    if role not in {"cover", "content", "section_divider", "summary", "data", "comparison", "timeline", "other"}:
        role = page_role if page_role in PAGE_ROLES else "content"
    base["template_role"] = "section_divider" if role == "section_divider" else role
    base["layout_structure"] = str(base.get("layout_structure") or default_layout_profile(page_role)["layout_structure"])
    if base.get("content_capacity") not in CONTENT_CAPACITIES:
        base["content_capacity"] = default_layout_profile(page_role)["content_capacity"]
    if base.get("visual_density") not in VISUAL_DENSITIES:
        base["visual_density"] = default_layout_profile(page_role)["visual_density"]
    text_regions = base.get("text_regions")
    if not isinstance(text_regions, list):
        text_regions = default_layout_profile(page_role)["text_regions"]
    image_regions = base.get("image_regions")
    if not isinstance(image_regions, list):
        image_regions = default_layout_profile(page_role)["image_regions"]
    base["text_regions"] = [normalize_region(item, f"text_{idx + 1}") for idx, item in enumerate(text_regions)]
    base["image_regions"] = [normalize_region(item, f"image_{idx + 1}") for idx, item in enumerate(image_regions)]
    style_keywords = base.get("style_keywords")
    if not isinstance(style_keywords, list):
        style_keywords = []
    color_palette = base.get("color_palette")
    if not isinstance(color_palette, list):
        color_palette = []
    base["style_keywords"] = [str(item) for item in style_keywords[:5]]
    base["color_palette"] = [str(item) for item in color_palette[:5]]
    base["notes"] = str(base.get("notes") or "")[:120]
    return base


def canonical_slide_id(value, fallback_index: int | None = None) -> str:
    raw = "" if value is None else str(value).strip()
    if raw.isdigit():
        return raw.zfill(2)
    if raw:
        return raw
    if fallback_index is not None:
        return str(fallback_index).zfill(2)
    return ""


def normalize_outline_structure(outline) -> list[dict]:
    """Accept banana-slides simple or part-based outline JSON and return flat pages.

    Upstream allows either `[page, ...]` or `[{part, pages:[page, ...]}, ...]`.
    This skill keeps a flat `slides` array, so the part label is preserved on
    each page instead of discarding the structure.
    """
    if isinstance(outline, dict):
        if isinstance(outline.get("pages"), list):
            outline = [outline]
        elif isinstance(outline.get("outline"), list):
            outline = outline["outline"]
    if not isinstance(outline, list) or not outline:
        raise ValueError("outline response is not a non-empty JSON array")

    flat_pages = []
    for entry in outline:
        if not isinstance(entry, dict):
            raise ValueError("outline entries must be JSON objects")
        pages = entry.get("pages")
        if isinstance(pages, list):
            part = entry.get("part") or entry.get("section") or entry.get("title") or ""
            for page in pages:
                if not isinstance(page, dict):
                    raise ValueError("part-based outline pages must be JSON objects")
                item = dict(page)
                if part and not item.get("part"):
                    item["part"] = str(part)
                flat_pages.append(item)
        else:
            flat_pages.append(dict(entry))

    if not flat_pages:
        raise ValueError("outline response contains no pages")
    return flat_pages


def generate_outline(model, api_key, topic, audience, page_count_hint, language, timeout, reference_files_content=None):
    user_prompt = (
        f"主题/用户需求：{topic}\n"
        f"目标受众：{audience or '未特别说明，请自行判断合适的受众语气'}\n"
        f"期望页数：{page_count_hint or '未指定，请自行选择 4-10 之间的合理页数'}\n"
        f"输出语言：{language}\n"
    )
    response = call_dashscope_json_with_retry(model, api_key, OUTLINE_SYSTEM_PROMPT, build_prompt(user_prompt, reference_files_content), timeout)
    if not (response.get("status_code") and 200 <= response["status_code"] < 300):
        return None, {"reason": "outline dashscope request failed", "details": response}
    try:
        outline = extract_json_payload(response)
        outline = normalize_outline_structure(outline)
    except Exception as exc:
        return None, {"reason": "failed to parse outline response", "details": {"error": str(exc), "raw": response.get("text", "")[:1500]}}
    return outline, None


def parse_outline_from_text(model, api_key, outline_text, language, timeout, reference_files_content=None):
    """Mirrors banana `get_outline_parsing_prompt` (prompts.py:352): reorganize
    a user-supplied outline into structured JSON without rewriting any wording."""
    user_prompt = f"用户提供的大纲原文：\n{outline_text}\n\n输出语言：{language}\n"
    response = call_dashscope_json_with_retry(model, api_key, OUTLINE_PARSING_SYSTEM_PROMPT, build_prompt(user_prompt, reference_files_content), timeout)
    if not (response.get("status_code") and 200 <= response["status_code"] < 300):
        return None, {"reason": "outline-parsing dashscope request failed", "details": response}
    try:
        outline = extract_json_payload(response)
        outline = normalize_outline_structure(outline)
    except Exception as exc:
        return None, {"reason": "failed to parse outline-parsing response", "details": {"error": str(exc), "raw": response.get("text", "")[:1500]}}
    return outline, None


def generate_outline_from_description(model, api_key, description_text, page_count_hint, language, timeout, reference_files_content=None):
    """Mirrors banana `get_description_to_outline_prompt` (prompts.py:413): derive
    an outline structure from a full article/description text with no explicit page split."""
    user_prompt = (
        f"用户提供的完整描述/文章文本：\n{description_text}\n\n"
        f"期望页数：{page_count_hint or '未指定，请根据内容自然结构判断合理页数'}\n"
        f"输出语言：{language}\n"
    )
    response = call_dashscope_json_with_retry(model, api_key, DESCRIPTION_TO_OUTLINE_SYSTEM_PROMPT, build_prompt(user_prompt, reference_files_content), timeout)
    if not (response.get("status_code") and 200 <= response["status_code"] < 300):
        return None, {"reason": "description-to-outline dashscope request failed", "details": response}
    try:
        outline = extract_json_payload(response)
        outline = normalize_outline_structure(outline)
    except Exception as exc:
        return None, {"reason": "failed to parse description-to-outline response", "details": {"error": str(exc), "raw": response.get("text", "")[:1500]}}
    return outline, None


def split_description_into_pages(model, api_key, outline_with_ids, existing_page_text, language, timeout, reference_files_content=None):
    """Mirrors banana `get_description_split_prompt` (prompts.py:680): split a
    user's already-finished, page-by-page copy to match an outline without
    rewriting wording (no content-generation LLM call happens here)."""
    user_prompt = (
        f"给定大纲（按顺序）：\n{json.dumps(outline_with_ids, ensure_ascii=False, indent=2)}\n\n"
        f"用户已经写好的完整逐页文案原文：\n{existing_page_text}\n\n"
        f"输出语言：{language}\n"
    )
    response = call_dashscope_json_with_retry(model, api_key, DESCRIPTION_SPLIT_SYSTEM_PROMPT, build_prompt(user_prompt, reference_files_content), timeout)
    if not (response.get("status_code") and 200 <= response["status_code"] < 300):
        return None, {"reason": "description-split dashscope request failed", "details": response}
    try:
        descriptions = extract_json_payload(response)
        if not isinstance(descriptions, list) or not descriptions:
            raise ValueError("description-split response is not a non-empty JSON array")
    except Exception as exc:
        return None, {"reason": "failed to parse description-split response", "details": {"error": str(exc), "raw": response.get("text", "")[:1500]}}
    return descriptions, None


def generate_descriptions(model, api_key, outline_with_ids, topic, language, timeout, detail_level="default", reference_files_content=None):
    descriptions = []
    outline_text = json.dumps(outline_with_ids, ensure_ascii=False, indent=2)
    for idx, page_outline in enumerate(outline_with_ids, start=1):
        user_prompt = (
            f"用户的原始需求是：\n{topic}\n\n"
            f"我们已经有了完整的大纲：\n{outline_text}\n\n"
            f"现在请为第 {idx} 页生成描述：\n{json.dumps(page_outline, ensure_ascii=False, indent=2)}\n\n"
            f"输出语言：{language}\n"
        )
        response = call_dashscope_json_with_retry(
            model,
            api_key,
            build_description_system_prompt(detail_level, idx),
            build_prompt(user_prompt, reference_files_content),
            timeout,
        )
        if not (response.get("status_code") and 200 <= response["status_code"] < 300):
            return None, {"reason": f"description dashscope request failed for page {idx}", "details": response}
        try:
            description = extract_json_payload(response)
            if not isinstance(description, dict):
                raise ValueError("page description response is not a JSON object")
            description["slide_id"] = canonical_slide_id(description.get("slide_id"), idx)
            descriptions.append(description)
        except Exception as exc:
            return None, {"reason": f"failed to parse description response for page {idx}", "details": {"error": str(exc), "raw": response.get("text", "")[:1500]}}
    return descriptions, None


def assemble_slides(outline, descriptions):
    by_id = {canonical_slide_id(item.get("slide_id")): item for item in descriptions if isinstance(item, dict)}
    slides = []
    for idx, page in enumerate(outline, start=1):
        sid = str(idx).zfill(2)
        desc = by_id.get(sid, {})
        page_role = str(page.get("page_role") or "content")
        if page_role not in PAGE_ROLES:
            page_role = "content"
        layout_profile = normalize_layout_profile(desc.get("layout_profile") or page.get("layout_profile"), page_role)
        slides.append(
            {
                "slide_id": sid,
                "purpose": page.get("intent") or "",
                "title": page.get("title") or "",
                "part": page.get("part") or "",
                "page_role": page_role,
                "content_density": layout_profile.get("visual_density"),
                "layout_profile": layout_profile,
                "body_text": desc.get("body_text") or page.get("points") or [],
                "must_show": desc.get("must_show") or [],
                "extra_fields": desc.get("extra_fields") or {},
                "delivery_mode": "image_only",
                "image_prompt": "",
                "rendered_image": "",
                "editable_text": [],
                "evidence": [],
                "status": "planned",
            }
        )
    return slides


def audit_role_diversity(slides):
    """Cheap deck-level equivalent of banana's 'avoid 5 consecutive identical
    templates' rule (prompts.py:1524) plus the 'must have a content page' rule."""
    issues = []
    if slides and slides[0].get("page_role") != "cover":
        issues.append("first slide page_role is not 'cover'")
    if len(slides) >= 4 and not any(s.get("page_role") == "content" for s in slides):
        issues.append("deck has 4+ slides but no page_role='content' slide")
    run_length = 1
    for i in range(1, len(slides)):
        if slides[i].get("page_role") == slides[i - 1].get("page_role") and slides[i].get("page_role") != "cover":
            run_length += 1
            if run_length >= 4:
                issues.append(f"slides {i - run_length + 2}-{i + 1} repeat page_role='{slides[i]['page_role']}' 4+ times in a row")
        else:
            run_length = 1
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", help="Deck topic / idea prompt (mode 1, default).")
    parser.add_argument("--outline-text", type=Path, help="Path to a text file containing a user-written outline (mode 2: parse, don't rewrite).")
    parser.add_argument("--description-text", type=Path, help="Path to a text file containing a full article/description with no explicit page split (mode 3: derive outline, then write descriptions).")
    parser.add_argument("--existing-page-text", type=Path, help="Path to a text file containing the user's already-finished, page-by-page copy (mode 4: split to match outline, no rewriting). Requires --outline-text or --description-text to establish the outline first.")
    parser.add_argument("--reference-file", type=Path, action="append", default=[], help="Uploaded/source reference file to prepend as banana-slides-style <uploaded_files> XML. Can be repeated.")
    parser.add_argument("--title", help="Deck title; defaults to topic/derived outline's first-page title.")
    parser.add_argument("--audience", default="")
    parser.add_argument("--page-count-hint", type=int, default=None)
    parser.add_argument("--language", default="zh")
    parser.add_argument("--detail-level", default="default", choices=sorted(DETAIL_LEVEL_SPECS), help="Text density mode mirroring banana-slides DETAIL_LEVEL_SPECS.")
    parser.add_argument("--aspect-ratio", default="16:9")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=int, default=180, help="Per-request read timeout; the description stage generates a larger JSON payload than the outline stage and can take longer.")
    parser.add_argument("--out", type=Path, required=True, help="Output slide_spec.json path (deck/slides only; caller fills visual_style_contract reference).")
    args = parser.parse_args()

    input_modes = [bool(args.topic), bool(args.outline_text), bool(args.description_text)]
    if sum(input_modes) != 1:
        print(json.dumps({"status": "blocked", "reason": "exactly one of --topic / --outline-text / --description-text is required"}, ensure_ascii=False, indent=2))
        return 2

    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        blocker = {"status": "blocked", "reason": "missing DASHSCOPE_API_KEY"}
        print(json.dumps(blocker, ensure_ascii=False, indent=2))
        return 2
    reference_files_content = load_reference_files(args.reference_file)

    route = ["topic_only"]
    content_source_note = args.topic or ""
    if args.outline_text:
        outline_text = args.outline_text.read_text(encoding="utf-8")
        outline, err = parse_outline_from_text(args.model, api_key, outline_text, args.language, args.timeout, reference_files_content)
        stage = "outline-parsing"
        route = ["user_outline_text"]
        content_source_note = f"(用户提供的大纲文本，来自 {args.outline_text})"
    elif args.description_text:
        description_text = args.description_text.read_text(encoding="utf-8")
        outline, err = generate_outline_from_description(args.model, api_key, description_text, args.page_count_hint, args.language, args.timeout, reference_files_content)
        stage = "description-to-outline"
        route = ["user_description_text"]
        content_source_note = f"(从用户提供的完整描述文本提炼大纲，来自 {args.description_text})"
    else:
        outline, err = generate_outline(args.model, api_key, args.topic, args.audience, args.page_count_hint, args.language, args.timeout, reference_files_content)
        stage = "outline"
    if err:
        print(json.dumps({"status": "blocked", "stage": stage, **err}, ensure_ascii=False, indent=2))
        return 2

    outline_with_ids = [{**page, "slide_id": str(i + 1).zfill(2)} for i, page in enumerate(outline)]

    if args.existing_page_text:
        existing_page_text = args.existing_page_text.read_text(encoding="utf-8")
        descriptions, err = split_description_into_pages(args.model, api_key, outline_with_ids, existing_page_text, args.language, args.timeout, reference_files_content)
        desc_stage = "description-split"
        route = route + ["existing_page_text"]
    else:
        descriptions, err = generate_descriptions(args.model, api_key, outline_with_ids, content_source_note, args.language, args.timeout, args.detail_level, reference_files_content)
        desc_stage = "description"
    if err:
        print(json.dumps({"status": "blocked", "stage": desc_stage, **err}, ensure_ascii=False, indent=2))
        return 2

    slides = assemble_slides(outline_with_ids, descriptions)
    role_issues = audit_role_diversity(slides)

    deck_title = args.title or args.topic or (outline_with_ids[0].get("title") if outline_with_ids else None) or "未命名汇报"
    spec = {
        "deck": {
            "title": deck_title,
            "language": args.language,
            "aspect_ratio": args.aspect_ratio,
            "audience": args.audience or "未特别说明",
            "route": route,
        },
        "slides": slides,
        "generation_meta": {
            "schema": "visual_deck_slide_spec_generation_v1",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "model": args.model,
            "provider": "dashscope",
            "input_mode": route[0],
            "detail_level": args.detail_level,
            "reference_files": [{"filename": item.get("filename", ""), "content_chars": len(item.get("content", ""))} for item in reference_files_content],
            "role_diversity_issues": role_issues,
        },
    }
    write_json(args.out, spec)
    print(json.dumps({"status": "completed", "out": str(args.out), "slide_count": len(slides), "route": route, "role_diversity_issues": role_issues}, ensure_ascii=False, indent=2))
    return 0 if not role_issues else 3


if __name__ == "__main__":
    raise SystemExit(main())
