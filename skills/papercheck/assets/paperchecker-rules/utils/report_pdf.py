from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Optional

import fitz


# 优先选可稳定显示中文的系统字体（macOS）
FONT_CANDIDATES = [
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
]


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def _find_font_for_text(text: str) -> Optional[str]:
    if not _contains_cjk(text):
        return None
    for font_path in FONT_CANDIDATES:
        if Path(font_path).exists():
            return font_path
    return None


def _normalize_md_line(line: str) -> str:
    normalized = (line or "").replace("\t", "    ").replace("<br>", " ")
    normalized = re.sub(r"^\s{0,3}#{1,6}\s*", "", normalized)
    normalized = re.sub(r"\*\*(.*?)\*\*", r"\1", normalized)
    normalized = re.sub(r"`([^`]*)`", r"\1", normalized)
    return normalized.rstrip()


def _char_width(ch: str) -> int:
    if ch == "\t":
        return 4
    return 2 if unicodedata.east_asian_width(ch) in {"F", "W"} else 1


def _wrap_line(line: str, max_units: int) -> list[str]:
    if not line:
        return [""]

    chunks: list[str] = []
    current: list[str] = []
    width = 0
    for ch in line:
        ch_width = _char_width(ch)
        if width + ch_width > max_units and current:
            chunks.append("".join(current))
            current = [ch]
            width = ch_width
        else:
            current.append(ch)
            width += ch_width

    if current:
        chunks.append("".join(current))
    return chunks


def save_markdown_as_pdf(markdown_content: str, output_path: str) -> str:
    """
    将 Markdown 文本渲染为 PDF 文件（轻量文本渲染，确保可稳定导出）。
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    page_width, page_height = fitz.paper_size("a4")
    margin_left = 42
    margin_top = 48
    margin_bottom = 48
    line_height = 16
    max_units = 96
    font_size = 10.5
    font_alias = "helv"

    font_path = _find_font_for_text(markdown_content)

    doc = fitz.open()
    page = doc.new_page(width=page_width, height=page_height)
    if font_path:
        font_alias = "F0"
        page.insert_font(fontname=font_alias, fontfile=font_path)

    y = margin_top
    for raw_line in (markdown_content or "").splitlines():
        line = _normalize_md_line(raw_line)
        wrapped_lines = _wrap_line(line, max_units) if line else [""]

        for wrapped in wrapped_lines:
            if y + line_height > page_height - margin_bottom:
                page = doc.new_page(width=page_width, height=page_height)
                if font_path:
                    page.insert_font(fontname=font_alias, fontfile=font_path)
                y = margin_top

            # 空行用更小高度保留段落感
            if wrapped == "":
                y += int(line_height * 0.65)
                continue

            page.insert_text(
                (margin_left, y),
                wrapped,
                fontsize=font_size,
                fontname=font_alias,
                color=(0, 0, 0),
            )
            y += line_height

    doc.save(str(output))
    doc.close()
    return str(output)
