#!/usr/bin/env python3
"""Split MinerU VLM content_list_v2.json into page-level Markdown files."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def textify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "".join(textify(item) for item in value)
    if isinstance(value, dict):
        parts: list[str] = []
        for key, item in value.items():
            if key in {"image_source"}:
                continue
            parts.append(textify(item))
        return "".join(parts)
    return str(value)


def normalize_text(text: str) -> str:
    text = text.replace("\r", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pages(path: Path) -> list[list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected a top-level list")
    if data and isinstance(data[0], list):
        return data

    pages: dict[int, list[dict[str, Any]]] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        page_idx = int(item.get("page_idx", 0))
        pages.setdefault(page_idx, []).append(item)
    if not pages:
        return []
    return [pages.get(i, []) for i in range(max(pages) + 1)]


def split_pages(content_list: Path, out_dir: Path, summary_path: Path | None) -> None:
    pages = load_pages(content_list)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_lines: list[str] = []
    for page_no, page in enumerate(pages, start=1):
        lines = [f"# PDF物理页 p{page_no}\n"]
        type_counter: Counter[str] = Counter()
        for obj_no, obj in enumerate(page, start=1):
            typ = obj.get("type", "")
            type_counter[typ] += 1
            content = normalize_text(textify(obj.get("content")))
            if not content:
                content = normalize_text(
                    textify(obj.get("text") or obj.get("table_body") or obj.get("img_caption") or obj.get("table_caption"))
                )
            if not content:
                continue
            lines.append(f"## 对象 {obj_no} | type={typ} | bbox={obj.get('bbox')}\n")
            lines.append(content + "\n")
        (out_dir / f"page_{page_no:03d}.md").write_text("\n".join(lines), encoding="utf-8")
        summary_lines.append(f"- p{page_no}: {len(page)} objects; types={dict(type_counter)}")
    if summary_path:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print("pages", len(pages))
    print("out_dir", out_dir)
    if summary_path:
        print("summary", summary_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Split MinerU content_list_v2.json into per-page Markdown files.")
    parser.add_argument("--content-list", required=True, help="Path to content_list_v2.json or *_content_list.json")
    parser.add_argument("--out", required=True, help="Output directory for page_XXX.md")
    parser.add_argument("--summary", default=None, help="Optional summary Markdown path")
    args = parser.parse_args()

    content_list = Path(args.content_list).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    summary = Path(args.summary).expanduser().resolve() if args.summary else out_dir.parent / "vlm_object_summary.md"
    split_pages(content_list, out_dir, summary)


if __name__ == "__main__":
    main()
