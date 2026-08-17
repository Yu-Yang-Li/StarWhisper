#!/usr/bin/env python
"""Create a Scispark workspace skeleton."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def topic_slug(text: str) -> str:
    # P2-6: 关键词取前20字符做slug，空格替换为-，其他非字母数字/中文/下划线/短横线去除
    if not text:
        return "scispark-topic"
    head = str(text).strip()[:20]
    head = re.sub(r"\s+", "-", head)
    head = re.sub(r"[^\w\u4e00-\u9fff\-]+", "", head, flags=re.UNICODE)
    head = re.sub(r"-+", "-", head).strip("-")
    return head or "scispark-topic"


def default_root(keyword: str) -> Path:
    # P2-6: 默认root改为当前目录下 ./scispark/<topic-slug>
    return Path(".") / "scispark" / topic_slug(keyword)


STAGE_FILES = [
    "01_fact_extraction.md",
    "02_hypothesis.md",
    "03_initial_idea.md",
    "04_technical_optimization.md",
    "05_moa_optimization.md",
    "06_human_ai_collaboration.md",
]


CSV_HEADER = [
    "id",
    "title",
    "authors",
    "year",
    "venue",
    "doi",
    "arxiv_id",
    "url",
    "pdf_url",
    "source_api",
    "query",
    "stage",
    "usage",
    "verification_status",
    "notes",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("keyword")
    parser.add_argument("--root", default=None,
                        help="输出根目录，默认为 ./scispark/<topic-slug>")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    # P2-6: 默认root不再依赖硬编码中文路径；--root显式传则用传入值
    if args.root:
        out = Path(args.root) / topic_slug(args.keyword)
    else:
        out = default_root(args.keyword)

    out.mkdir(parents=True, exist_ok=True)  # P2-6: 自动mkdir -p
    (out / "experts").mkdir(exist_ok=True)
    (out / "slides").mkdir(exist_ok=True)

    for name in STAGE_FILES:
        path = out / name
        if args.force or not path.exists():
            path.write_text(f"# {name[:-3]}\n\n", encoding="utf-8")

    csv_path = out / "literature.csv"
    if args.force or not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)

    state = out / "scispark-state.json"
    if args.force or not state.exists():
        state.write_text(
            '{\n  "keyword": "%s",\n  "current_stage": 1,\n  "status": "initialized"\n}\n'
            % args.keyword.replace('"', '\\"'),
            encoding="utf-8",
        )

    print(out.resolve())


if __name__ == "__main__":
    main()
