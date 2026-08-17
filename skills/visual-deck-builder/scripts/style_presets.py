"""Rich, banana-slides-density style presets for visual_style_contract.json.

Each preset specifies concrete hex colors, exact font family names, spacing
and accent-usage percentages, icon/chart rendering rules, and a rendering
descriptor. This exists because a short label like "clean, modern, blue and
white" leaves too much room for the image model to drift between pages; a
model given exact hex values and named fonts reproduces them far more
consistently across independent generation calls.

These are original presets written for this skill, not copied text from any
other product's preset library. They aim for comparable *precision density*
(explicit colors/fonts/percentages), not identical wording or taxonomy.
"""

from __future__ import annotations

PRESETS: dict[str, dict] = {
    "scientific_tool_report": {
        "style_name": "科研工具产品汇报",
        "mood_and_lighting": (
            "克制、专业、可信的产品评审氛围。全篇使用均匀漫射光的扁平插画风格，"
            "没有强烈的阴影或高光，没有摄影质感，避免任何营销海报式的戏剧化构图。"
        ),
        "palette": {
            "background": {"name": "暖白纸感背景", "hex": "#F7F4EC"},
            "primary": {"name": "深蓝", "hex": "#13315C"},
            "accent": {"name": "青绿色", "hex": "#2FA39A"},
            "neutral": {"name": "炭黑文字", "hex": "#1F2430"},
            "divider": {"name": "浅灰分割线", "hex": "#D9D4C7"},
        },
        "accent_usage_limit": "强调色（青绿色）使用面积不超过单页总面积的 10%，只用于图标高亮、连接线、关键数字和状态标记；不作为大面积色块。",
        "typography": {
            "cjk_font": "思源黑体 / Noto Sans SC",
            "latin_font": "Inter",
            "title_weight": "Bold，字号约为正文字号的 2.2-2.6 倍，仅左上或居中一种对齐方式贯穿全篇",
            "body_weight": "Regular 或 Medium，短标签优先于长句",
        },
        "layout_density": "中等密度：每页一个主视觉结构，页边距约占画布宽度的 6%，元素之间保持清晰留白，避免拥挤。",
        "grid_and_spacing": "统一使用隐形网格对齐；同类卡片宽高一致，卡片间距约为画布宽度的 2%-3%。",
        "visual_language": "扁平科研产品图表、流程线、克制的线性图标、轻量圆角卡片；不使用营销海报风、不使用真实照片拼贴。",
        "icon_style": "线性描边图标，线宽视觉上均匀一致；卡片圆角半径约为画布宽度的 1%-1.5%，不使用尖角卡片。",
        "chart_rules": "流程、质量门、产物卡片等结构使用简洁可读的专业图形（横向/纵向流程、分栏对比、简单网格）；避免 3D 图表、避免不必要的图例。",
        "cover_treatment": "封面极简：左上或居中大标题、副标题与汇报人/团队信息清晰分层；只允许极弱背景纹理或细线装饰，不规划流程图、机制图、数据图或复杂视觉母题。",
        "forbidden_patterns": ["watermark", "logo", "fake English", "prompt labels", "markdown symbols", "stock photo", "3D 拟真材质"],
        "rendering_descriptor": "超高清扁平矢量插画渲染，边缘锐利无锯齿，色彩克制，整体呈现严谨的企业级科研工具评审美学。",
    },
    "academic_formal_report": {
        "style_name": "学术严谨报告",
        "mood_and_lighting": (
            "安静、克制、纸质印刷质感，没有任何炫光或过度设计；画面二维平面呈现，不做三维立体处理，不出现书本装订线或阴影边框。"
        ),
        "palette": {
            "background": {"name": "米白印刷纸色", "hex": "#F8F6F1"},
            "primary": {"name": "墨黑", "hex": "#202124"},
            "accent": {"name": "学术深红", "hex": "#8A2E2E"},
            "neutral": {"name": "深炭灰文字", "hex": "#33363B"},
            "divider": {"name": "细线灰", "hex": "#C9C4B8"},
        },
        "accent_usage_limit": "强调色（学术深红）占比不超过单页面积的 5%，仅用于关键结论、图表高亮或引用标记。",
        "typography": {
            "cjk_font": "思源宋体 / Noto Serif SC",
            "latin_font": "Times New Roman 或 Georgia 风格的衬线体",
            "title_weight": "Bold 衬线体，字号约为正文的 2 倍，居中或左对齐均可但全篇统一",
            "body_weight": "Regular 衬线体，行距宽松",
        },
        "layout_density": "版式遵循经典排版原则，页边距宽阔（约画布宽度的 8%），信息密度低于商业风格页面。",
        "grid_and_spacing": "采用左右分栏或上下结构的严谨对齐方式，栏间距约为画布宽度的 4%。",
        "visual_language": "精细黑色线条框、标准学术表格样式、黑白线稿插图；避免任何卡通化或营销化视觉元素。",
        "icon_style": "极简黑色线稿图标，不使用彩色填充图标，不使用圆角卡片，使用直角矩形框线。",
        "chart_rules": "图表以黑白或深红单色为主，坐标轴、图例线条精细，不使用渐变或立体效果。",
        "cover_treatment": "封面居中排布标题、副标题与作者/机构信息，四周留白宽阔，不加装饰性图形。",
        "forbidden_patterns": ["watermark", "logo", "fake English", "prompt labels", "markdown symbols", "3D effect", "photo collage", "page border", "drop shadow"],
        "rendering_descriptor": "超高分辨率扫描印刷风格，字体抗锯齿细腻，线条锐利，呈现精装学术期刊内页质感。",
    },
    "modern_tech_briefing": {
        "style_name": "现代科技简报",
        "mood_and_lighting": (
            "冷静而有科技感，暗底配合自发光描边元素，避免赛博朋克式的夸张霓虹，保持商务可读性。"
        ),
        "palette": {
            "background": {"name": "深空蓝黑", "hex": "#0E1A2B"},
            "primary": {"name": "冰蓝", "hex": "#3DA9FC"},
            "accent": {"name": "荧光青", "hex": "#2FE6D0"},
            "neutral": {"name": "近白文字", "hex": "#EAF1FB"},
            "divider": {"name": "低对比蓝灰", "hex": "#2A3B52"},
        },
        "accent_usage_limit": "荧光青仅用于关键路径、数据高亮和状态图标，占比不超过单页面积的 8%。",
        "typography": {
            "cjk_font": "思源黑体 Medium / Noto Sans SC",
            "latin_font": "Inter 或等宽字体（用于代码/参数展示）",
            "title_weight": "Bold，字号约为正文的 2.3 倍，左对齐",
            "body_weight": "Regular，短句为主",
        },
        "layout_density": "中等偏紧凑，允许非对称动态构图，但每页仍只保留一个主视觉焦点。",
        "grid_and_spacing": "网格线可见但极淡（透明度低于 12%），元素间距约为画布宽度的 2%。",
        "visual_language": "线框几何体、发光连接线、卡片式信息面板；避免真实照片、避免复杂粒子特效堆砌。",
        "icon_style": "描边发光图标，线宽均匀，卡片使用半透明深色背景与细描边，圆角半径约画布宽度 1%。",
        "chart_rules": "使用简洁的流程图、雷达图或进度条，配色限定在主色/强调色范围内，不引入无关色系。",
        "cover_treatment": "封面居左，标题、副标题与汇报人/团队信息清晰分层；可使用一条极细发光引导线作为背景节奏，不使用系统架构图或复杂剪影。",
        "forbidden_patterns": ["watermark", "logo", "fake English", "prompt labels", "markdown symbols", "cluttered particle effects", "stock photo"],
        "rendering_descriptor": "暗色调数字产品发布会风格渲染，描边发光克制，对比清晰，适合科技/工具类产品对外简报。",
    },
    "business_minimal_report": {
        "style_name": "商务极简汇报",
        "mood_and_lighting": (
            "克制、稳重、国际化咨询公司风格；均匀漫射光，无方向性主光，无戏剧化明暗对比。"
        ),
        "palette": {
            "background": {"name": "海军蓝", "hex": "#0B1F3B"},
            "primary": {"name": "纯白", "hex": "#FFFFFF"},
            "accent": {"name": "天蓝", "hex": "#38BDF8"},
            "neutral": {"name": "浅灰辅助", "hex": "#E5E7EB"},
            "divider": {"name": "低对比灰蓝分割线", "hex": "#1E3352"},
        },
        "accent_usage_limit": "天蓝仅用于关键数字、结论关键词、关键路径端点，占比不超过单页面积的 3%。",
        "typography": {
            "cjk_font": "思源黑体 Heavy/Bold（标题）与 Regular（正文）",
            "latin_font": "Roboto Bold（标题）/ Roboto Regular（正文）",
            "title_weight": "Bold，字号约为正文的 2.4 倍",
            "body_weight": "Regular/Light，避免同层级混用不同字重",
        },
        "layout_density": "严格模块化网格系统，页面分区固定为标题区/主图区/要点区/结论区。",
        "grid_and_spacing": "分区边界使用 1px 细线（#E5E7EB），页边距约为画布宽度的 6%。",
        "visual_language": "白色线稿矢量插画，关键部件用天蓝点亮；禁止彩色照片、复杂纹理、拟物材质。",
        "icon_style": "统一线宽的白色线稿图标，转角规整，工程图式简洁。",
        "chart_rules": "默认允许柱状图、折线图、流程/架构框图；除非分类不超过 5 类且必须表达占比，否则禁止饼图。",
        "cover_treatment": "封面居中或左对齐大标题，副标题与汇报人/团队信息完整可读；只使用一条极细天蓝色分割线或路径线作为弱装饰，正文极简。",
        "forbidden_patterns": ["watermark", "logo", "fake English", "prompt labels", "markdown symbols", "gradient glow", "skeuomorphic texture", "pie chart overuse"],
        "rendering_descriptor": "超高清矢量插画与商务信息图风格，线条干净、层级稳定，适用于正式企业级评审场景。",
    },
}

DEFAULT_PRESET_KEY = "scientific_tool_report"


def list_presets() -> list[str]:
    return list(PRESETS.keys())


def get_preset(key: str) -> dict:
    if key not in PRESETS:
        raise KeyError(f"unknown style preset: {key}; available: {', '.join(list_presets())}")
    return PRESETS[key]
