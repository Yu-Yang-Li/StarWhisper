#!/usr/bin/env python3
"""Render a simple Chinese Markdown audit report to PDF with ReportLab."""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet


FONT_CANDIDATES = [
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]


def register_font() -> str:
    for candidate in FONT_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            pdfmetrics.registerFont(TTFont("AuditCN", str(path)))
            return "AuditCN"
    return "Helvetica"


def inline_markup(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r'<font color="#444444">\1</font>', text)
    text = re.sub(r"\[([^\]]+)\]\(([^\)]+)\)", r"\1 (\2)", text)
    return text


def build_pdf(input_path: Path, output_path: Path) -> None:
    font = register_font()
    styles = getSampleStyleSheet()
    base = ParagraphStyle("BaseCN", parent=styles["Normal"], fontName=font, fontSize=9.2, leading=14, spaceAfter=4)
    h1 = ParagraphStyle("H1CN", parent=base, fontSize=18, leading=24, alignment=TA_CENTER, spaceAfter=10)
    h2 = ParagraphStyle("H2CN", parent=base, fontSize=14, leading=20, spaceBefore=8, spaceAfter=6)
    h3 = ParagraphStyle("H3CN", parent=base, fontSize=11.5, leading=17, spaceBefore=6, spaceAfter=4)
    mono = ParagraphStyle("MonoCN", parent=base, fontName=font, fontSize=7.8, leading=11, leftIndent=6, rightIndent=6, backColor=HexColor("#f6f6f6"))
    small = ParagraphStyle("SmallCN", parent=base, fontSize=8.5, leading=12)

    story = []
    for raw in input_path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if not line:
            story.append(Spacer(1, 2))
        elif line.startswith("# "):
            story.append(Paragraph(inline_markup(line[2:]), h1))
        elif line.startswith("## "):
            story.append(Paragraph(inline_markup(line[3:]), h2))
        elif line.startswith("### "):
            story.append(Paragraph(inline_markup(line[4:]), h3))
        elif line.startswith("|"):
            story.append(Preformatted(line, mono))
        elif line.startswith("- [ ]"):
            story.append(Paragraph("□ " + inline_markup(line[5:].strip()), small))
        elif line.startswith("- "):
            story.append(Paragraph("• " + inline_markup(line[2:].strip()), base))
        else:
            story.append(Paragraph(inline_markup(line), base))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )
    doc.build(story)
    print(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a Markdown audit report to PDF.")
    parser.add_argument("--input", required=True, help="Input Markdown path")
    parser.add_argument("--output", required=True, help="Output PDF path")
    args = parser.parse_args()
    build_pdf(Path(args.input).expanduser().resolve(), Path(args.output).expanduser().resolve())


if __name__ == "__main__":
    main()
