#!/usr/bin/env python3
"""Create a standard thesis audit workspace.

The script is intentionally conservative: it creates missing files and refuses
to overwrite existing files unless --force is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from datetime import date
from pathlib import Path


OBJECT_LEDGER_FIELDS = [
    "paper_id",
    "object_ledger_id",
    "object_type",
    "pdf_page",
    "thesis_page",
    "page_region",
    "section",
    "object_id",
    "object_title",
    "original_excerpt",
    "extraction_confidence",
    "needs_visual_check",
    "notes",
]

REVIEW_MATRIX_FIELDS = [
    "matrix_id",
    "source_doc",
    "source_rule_id",
    "source_requirement",
    "object_ledger_id",
    "pdf_page",
    "section",
    "check_item",
    "status",
    "evidence_excerpt",
    "issue_id",
    "notes",
]

VERIFIABLE_CLAIM_FIELDS = [
    "claim_id",
    "paper_id",
    "location",
    "claim_type",
    "claim_text",
    "value",
    "unit",
    "subject",
    "method_id",
    "verification_action",
    "status",
    "verification_notes",
    "issue_id",
]

REFERENCE_AUDIT_FIELDS = [
    "ref_id",
    "pdf_page",
    "original_entry",
    "type_mark",
    "cited_in_text",
    "function",
    "author",
    "year",
    "title",
    "source",
    "volume_issue",
    "pages",
    "doi_url",
    "access_date",
    "coverage_years",
    "matches_method_or_data_source",
    "issue",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pdf_page_count(path: Path) -> str:
    if path.suffix.lower() != ".pdf":
        return ""
    try:
        import fitz  # type: ignore
    except Exception:
        return ""
    try:
        doc = fitz.open(path)
        count = str(doc.page_count)
        doc.close()
        return count
    except Exception:
        return ""


def write_text(path: Path, text: str, force: bool) -> None:
    if path.exists() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, fields: list[str], force: bool) -> None:
    if path.exists() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(fields)


def source_manifest(args: argparse.Namespace, source: Path | None) -> str:
    size = str(source.stat().st_size) if source and source.exists() else ""
    digest = sha256_file(source) if source and source.exists() else ""
    pages = pdf_page_count(source) if source and source.exists() else ""
    source_path = str(source) if source else ""
    checklist_lines = "\n".join(f"- {item}" for item in args.checklist)
    return f"""# Source Manifest

- paper_id: {args.paper_id}
- title: {args.title}
- source_file: {source_path}
- file_size_bytes: {size}
- sha256: {digest}
- pdf_page_count: {pages}
- input_type: {resolve_input_type(args, source)}
- output_mode: {resolve_output_mode(args, source)}
- parse_method: {args.parse_method}
- review_basis: {args.review_basis}
- audit_date: {date.today().isoformat()}

## Checklist / Reference Documents

{checklist_lines or "- pending"}

## Input Risks

- pending
"""


def resolve_input_type(args: argparse.Namespace, source: Path | None) -> str:
    if args.input_type != "auto":
        return args.input_type
    if not source:
        return "unknown"
    suffix = source.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in {".doc", ".docx"}:
        return "docx"
    return suffix.lstrip(".") or "unknown"


def resolve_output_mode(args: argparse.Namespace, source: Path | None) -> str:
    if args.output_mode != "auto":
        return args.output_mode
    input_type = resolve_input_type(args, source)
    if input_type == "pdf":
        return "pdf_report"
    if input_type in {"doc", "docx"}:
        return "commented_docx"
    return "audit_report"


def page_index(args: argparse.Namespace) -> str:
    return f"""# Page Index

paper_id: {args.paper_id}

| pdf_page | thesis_page | page_type | section | extraction_quality | summary | visual_check_needed | notes |
|---:|---|---|---|---|---|---|---|
"""


def structure_map(args: argparse.Namespace) -> str:
    return f"""# Structure Map

paper_id: {args.paper_id}

| section_id | title | pdf_page_start | pdf_page_end | thesis_page_range | role | notes |
|---|---|---:|---:|---|---|---|
"""


def method_profile(args: argparse.Namespace) -> str:
    return f"""# Method Profile

paper_id: {args.paper_id}

| method_id | method_name | location | role | input_data | premises | outputs | supports_main_claim | public_or_internal_recheck | required_actions | status |
|---|---|---|---|---|---|---|---|---|---|---|
"""


def issue_database(args: argparse.Namespace) -> str:
    return f"""# Issue Database

paper_id: {args.paper_id}

Use one block per formal issue:

```text
### ISSUE-000: title

- severity:
- priority:
- certainty:
- matrix_id:
- object_ledger_id:
- claim_id:
- method_id:
- location:
- original_excerpt:
- problem:
- basis:
- modification_requirement:
- factcheck_status:
```
"""


def external_factcheck(args: argparse.Namespace) -> str:
    return f"""# External Factcheck

paper_id: {args.paper_id}

| fact_id | claim_id | method_id | location | paper_claim | fact_type | verification_status | source | source_date | checked_date | result | required_action |
|---|---|---|---|---|---|---|---|---|---|---|---|
"""


def coverage_report(args: argparse.Namespace) -> str:
    return f"""# Coverage Report

paper_id: {args.paper_id}

## Coverage Summary

| category | identified | checked | pass | issue | needs_factcheck | needs_internal_crosscheck | needs_method_premise_check | needs_recalculation | needs_author_source | blocked | notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|

## Blocked / Residual Risk

- pending
"""


def final_report(args: argparse.Namespace) -> str:
    return f"""# 学位论文规范审阅报告

## 封面信息

- 论文题名：{args.title}
- 文件名：
- 审核日期：{date.today().isoformat()}
- 审核依据：{args.review_basis}
- 审核方式：
- 定位口径：
- 审核结论：

## 一、总体审核结论

pending

## 二、必须修改问题汇总

| 序号 | 问题类别 | 定位 | 问题概述 | 修改优先级 |
|---|---|---|---|---|

## 三、逐处批注式审核意见

### 批注 AUD-000：示例占位，交付前删除

- **追溯编号**：
- **定位**：
- **原文摘录**：
- **批注意见**：
- **审核依据**：
- **修改要求**：
- **优先级**：
- **确定性**：

## 四、分项审核意见

### 4.1 摘要与关键词

### 4.2 目录、图表目录与编号

### 4.3 绪论与研究问题

### 4.4 文献综述与理论基础

### 4.5 研究设计与资料来源

### 4.6 正文章节论证

### 4.7 模型、公式、变量与数据

### 4.8 结论、管理启示与政策建议

### 4.9 参考文献与引用规范

### 4.10 可验证声明、方法前提与核心结果

## 五、可验证声明、方法前提与事实核验表

| 序号 | 追溯编号 | 类型 | 定位 | 原文声明 | 核验/复核动作 | 核验结果 | 来源或计算说明 | 处理意见 |
|---|---|---|---|---|---|---|---|---|

## 六、建议修改顺序

pending

## 七、复核清单

- [ ] pending

## 八、覆盖与剩余风险

pending
"""


def comment_list(args: argparse.Namespace) -> str:
    return f"""# 批注清单

paper_id: {args.paper_id}
title: {args.title}
review_basis: {args.review_basis}

用于 DOC/DOCX 输入的批注交付追溯。若已生成批注版 DOCX，每条批注应能追溯到此处或 issue database。

| comment_id | location | original_excerpt | comment | basis | required_action | priority | certainty | issue_id |
|---|---|---|---|---|---|---|---|---|
"""


def create_workspace(args: argparse.Namespace) -> None:
    root = Path(args.out).expanduser().resolve()
    source = Path(args.source_file).expanduser().resolve() if args.source_file else None
    work = root / "work" / args.paper_id
    output = root / "outputs" / args.paper_id
    for folder in (root / "inputs", work, output, root / "logs"):
        folder.mkdir(parents=True, exist_ok=True)

    write_text(work / "00_source_manifest.md", source_manifest(args, source), args.force)
    write_text(work / "01_page_index.md", page_index(args), args.force)
    write_text(work / "02_structure_map.md", structure_map(args), args.force)
    write_text(work / "03_method_profile.md", method_profile(args), args.force)
    write_csv(work / "04_object_ledger.csv", OBJECT_LEDGER_FIELDS, args.force)
    write_csv(work / "05_verifiable_claims.csv", VERIFIABLE_CLAIM_FIELDS, args.force)
    write_csv(work / "06_review_matrix.csv", REVIEW_MATRIX_FIELDS, args.force)
    write_text(work / "07_issue_database.md", issue_database(args), args.force)
    write_text(work / "08_external_factcheck.md", external_factcheck(args), args.force)
    write_csv(work / "09_reference_audit.csv", REFERENCE_AUDIT_FIELDS, args.force)
    write_text(work / "10_coverage_report.md", coverage_report(args), args.force)
    write_text(work / "11_final_review_report.md", final_report(args), args.force)
    mode = resolve_output_mode(args, source)
    if mode == "pdf_report":
        write_text(output / "审查报告.md", final_report(args), args.force)
    elif mode == "commented_docx":
        write_text(output / "批注清单.md", comment_list(args), args.force)
        write_text(
            output / "README_批注版DOCX生成说明.md",
            "批注版论文.docx 应由 Documents 工具在原 DOC/DOCX 上写入批注后生成；本脚本只创建追溯清单和工作区。\n",
            args.force,
        )
    else:
        write_text(output / "审查报告.md", final_report(args), args.force)
    print(root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize a thesis audit workspace.")
    parser.add_argument("--out", required=True, help="Output project directory")
    parser.add_argument("--paper-id", required=True, help="Stable paper id, e.g. HM_HarmonyOS")
    parser.add_argument("--title", required=True, help="Paper title")
    parser.add_argument("--source-file", default="", help="Optional source PDF/DOCX path")
    parser.add_argument("--input-type", default="auto", choices=["auto", "pdf", "doc", "docx", "unknown"], help="Input type")
    parser.add_argument("--output-mode", default="auto", choices=["auto", "pdf_report", "commented_docx", "audit_report"], help="Final delivery mode")
    parser.add_argument("--checklist", action="append", default=[], help="Checklist or reference document path/name")
    parser.add_argument("--review-basis", default="references/default_review_sources.md", help="Default or custom review basis")
    parser.add_argument("--parse-method", default="pending", help="Planned or actual parse method")
    parser.add_argument("--force", action="store_true", help="Overwrite existing scaffold files")
    args = parser.parse_args()
    create_workspace(args)


if __name__ == "__main__":
    main()
