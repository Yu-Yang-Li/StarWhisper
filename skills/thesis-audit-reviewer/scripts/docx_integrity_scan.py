#!/usr/bin/env python3
"""Scan DOCX internals for objects that plain paragraph text extraction loses."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def read_xml(zf: zipfile.ZipFile, name: str) -> ET.Element | None:
    try:
        return ET.fromstring(zf.read(name))
    except KeyError:
        return None
    except ET.ParseError:
        return None


def text_of(element: ET.Element) -> str:
    parts: list[str] = []
    for node in element.iter():
        if node.tag == f"{{{NS['w']}}}t" and node.text:
            parts.append(node.text)
    return "".join(parts).strip()


def count(root: ET.Element | None, query: str) -> int:
    if root is None:
        return 0
    return len(root.findall(f".//{query}", NS))


def scan_docx(path: Path) -> dict:
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        document = read_xml(zf, "word/document.xml")
        comments = read_xml(zf, "word/comments.xml")
        footnotes = read_xml(zf, "word/footnotes.xml")
        endnotes = read_xml(zf, "word/endnotes.xml")

        paragraphs = document.findall(".//w:p", NS) if document is not None else []
        math_paragraphs = []
        for idx, paragraph in enumerate(paragraphs, start=1):
            math_count = len(paragraph.findall(".//m:oMath", NS)) + len(paragraph.findall(".//m:oMathPara", NS))
            if math_count:
                excerpt = text_of(paragraph)
                math_paragraphs.append({"paragraph_index": idx, "math_objects": math_count, "text_excerpt": excerpt[:160]})

        media_files = sorted(name for name in names if name.startswith("word/media/") and not name.endswith("/"))
        result = {
            "docx": str(path),
            "paragraph_count": len(paragraphs),
            "table_count": count(document, "w:tbl"),
            "ooxml_math_count": count(document, "m:oMath") + count(document, "m:oMathPara"),
            "paragraphs_with_math_count": len(math_paragraphs),
            "paragraphs_with_math": math_paragraphs[:50],
            "drawing_count": count(document, "w:drawing"),
            "legacy_pict_count": count(document, "w:pict"),
            "media_file_count": len(media_files),
            "media_files": media_files[:100],
            "hyperlink_count": count(document, "w:hyperlink"),
            "field_code_count": count(document, "w:fldChar"),
            "bookmark_count": count(document, "w:bookmarkStart"),
            "comment_reference_count": count(document, "w:commentReference"),
            "comment_count": count(comments, "w:comment"),
            "footnote_count": max(0, count(footnotes, "w:footnote") - 2),
            "endnote_count": max(0, count(endnotes, "w:endnote") - 2),
            "revision_insert_count": count(document, "w:ins"),
            "revision_delete_count": count(document, "w:del"),
            "has_comments_xml": "word/comments.xml" in names,
            "has_footnotes_xml": "word/footnotes.xml" in names,
            "has_endnotes_xml": "word/endnotes.xml" in names,
        }
        result["risk_flags"] = []
        if result["ooxml_math_count"]:
            result["risk_flags"].append("ooxml_math_lost_by_plain_text")
        if result["media_file_count"] or result["drawing_count"] or result["legacy_pict_count"]:
            result["risk_flags"].append("visual_objects_require_render_check")
        if result["comment_count"] or result["revision_insert_count"] or result["revision_delete_count"]:
            result["risk_flags"].append("comments_or_revisions_require_ooxml_or_word_check")
        if result["field_code_count"]:
            result["risk_flags"].append("field_codes_cross_references_or_toc_require_check")
        return result


def markdown_report(result: dict) -> str:
    lines = [
        "# DOCX Integrity Scan",
        "",
        f"- docx: {result['docx']}",
        f"- paragraph_count: {result['paragraph_count']}",
        f"- table_count: {result['table_count']}",
        f"- ooxml_math_count: {result['ooxml_math_count']}",
        f"- paragraphs_with_math_count: {result['paragraphs_with_math_count']}",
        f"- drawing_count: {result['drawing_count']}",
        f"- legacy_pict_count: {result['legacy_pict_count']}",
        f"- media_file_count: {result['media_file_count']}",
        f"- hyperlink_count: {result['hyperlink_count']}",
        f"- field_code_count: {result['field_code_count']}",
        f"- comment_count: {result['comment_count']}",
        f"- footnote_count: {result['footnote_count']}",
        f"- endnote_count: {result['endnote_count']}",
        f"- revision_insert_count: {result['revision_insert_count']}",
        f"- revision_delete_count: {result['revision_delete_count']}",
        "",
        "## Risk Flags",
        "",
    ]
    if result["risk_flags"]:
        lines.extend(f"- {flag}" for flag in result["risk_flags"])
    else:
        lines.append("- none")
    lines.extend(["", "## Paragraphs With Math", ""])
    for item in result["paragraphs_with_math"]:
        lines.append(f"- p{item['paragraph_index']}: math={item['math_objects']}; text={item['text_excerpt']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan DOCX for OOXML objects that may be invisible to plain text extraction.")
    parser.add_argument("--docx", required=True, help="DOCX path")
    parser.add_argument("--json-output", default="", help="Optional JSON output path")
    parser.add_argument("--markdown-output", default="", help="Optional Markdown output path")
    args = parser.parse_args()

    path = Path(args.docx).expanduser().resolve()
    result = scan_docx(path)
    if args.json_output:
        out = Path(args.json_output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.markdown_output:
        out = Path(args.markdown_output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown_report(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
