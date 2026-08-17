#!/usr/bin/env python3
"""Extract PDF text locally when MinerU is unavailable.

The fallback order mirrors the PaperCheck runtime:
1. Try pymupdf4llm.to_markdown() for Markdown-like layout text.
2. Fall back to PyMuPDF/fitz page.get_text(sort=True) per page.

This script does not call external services and does not require API keys.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def try_pymupdf4llm(pdf_path: Path) -> tuple[str, str | None]:
    try:
        import pymupdf4llm  # type: ignore
    except Exception as exc:
        return "", f"pymupdf4llm_unavailable: {exc}"
    try:
        markdown = pymupdf4llm.to_markdown(str(pdf_path))
    except Exception as exc:
        return "", f"pymupdf4llm_failed: {exc}"
    if not str(markdown).strip():
        return "", "pymupdf4llm_empty_output"
    return str(markdown), None


def fitz_page_text(pdf_path: Path) -> tuple[list[dict[str, Any]], str | None]:
    try:
        import fitz  # type: ignore
    except Exception as exc:
        return [], f"fitz_unavailable: {exc}"
    pages: list[dict[str, Any]] = []
    try:
        doc = fitz.open(pdf_path)
        for index, page in enumerate(doc, start=1):
            text = page.get_text("text", sort=True)
            pages.append(
                {
                    "pdf_page": index,
                    "width": float(page.rect.width),
                    "height": float(page.rect.height),
                    "text": text.strip(),
                    "char_count": len(text.strip()),
                }
            )
        doc.close()
    except Exception as exc:
        return pages, f"fitz_failed: {exc}"
    return pages, None


def markdown_from_pages(pages: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for page in pages:
        chunks.append(f"\n\n<!-- pdf_page: {page['pdf_page']} -->\n\n")
        chunks.append(str(page.get("text") or "").strip())
    return "\n".join(chunks).strip() + "\n"


def quality_label(method: str, pages: list[dict[str, Any]]) -> str:
    if method == "pymupdf4llm":
        return "local_markdown_fallback"
    if not pages:
        return "blocked_no_local_text"
    non_empty = sum(1 for page in pages if page.get("char_count", 0) > 20)
    if non_empty == 0:
        return "likely_scanned_or_image_pdf"
    if non_empty < max(1, len(pages) // 2):
        return "partial_text_layer"
    return "local_text_layer"


def main() -> None:
    parser = argparse.ArgumentParser(description="Local PDF fallback extraction for thesis audit.")
    parser.add_argument("--file", required=True, help="PDF file path")
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()

    pdf_path = Path(args.file).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    markdown, markdown_warning = try_pymupdf4llm(pdf_path)
    pages, fitz_warning = fitz_page_text(pdf_path)
    method = "pymupdf4llm" if markdown else "fitz_page_text"
    if not markdown:
        markdown = markdown_from_pages(pages)

    markdown_path = out_dir / "extracted_text.md"
    pages_path = out_dir / "page_text.json"
    status_path = out_dir / "pdf_fallback_status.json"
    markdown_path.write_text(markdown, encoding="utf-8")
    pages_path.write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")

    status = {
        "source_pdf": str(pdf_path),
        "method": method,
        "quality": quality_label(method, pages),
        "markdown_output": str(markdown_path),
        "page_text_output": str(pages_path),
        "page_count": len(pages),
        "non_empty_pages": sum(1 for page in pages if page.get("char_count", 0) > 20),
        "warnings": [item for item in [markdown_warning, fitz_warning] if item],
        "audit_note": (
            "Local fallback is usable for text-layer PDFs. For scanned PDFs, formulas, complex tables, "
            "multi-column layout, headers/footers, or reference lists split across pages, verify against "
            "the original PDF with an appropriate viewer/parser or mark the area as blocked/residual risk."
        ),
    }
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
