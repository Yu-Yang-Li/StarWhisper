#!/usr/bin/env python3
"""Validate a Markdown thesis audit report against the skill contract."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


BASE_REQUIRED_SECTIONS = [
    "## 一、总体审核结论",
    "## 二、必须修改问题汇总",
    "## 三、逐处批注式审核意见",
    "## 四、分项审核意见",
    "## 五、可验证声明、方法前提与事实核验表",
    "## 六、建议修改顺序",
    "## 七、复核清单",
]

STRICT_REQUIRED_SECTIONS = BASE_REQUIRED_SECTIONS + [
    "## 八、覆盖与剩余风险",
]

COMMENT_FIELDS = [
    "追溯编号",
    "定位",
    "原文摘录",
    "批注意见",
    "审核依据",
    "修改要求",
    "优先级",
    "确定性",
]

FORBIDDEN_PATTERNS = [
    r"我觉得",
    r"感觉",
    r"大概",
    r"可能不太好",
    r"有点",
]


def split_comment_blocks(text: str) -> list[tuple[str, str]]:
    matches = list(re.finditer(r"^###\s+批注[^\n]*", text, flags=re.M))
    blocks: list[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        blocks.append((match.group(0), text[start:end]))
    return blocks


def validate(text: str, strict: bool) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    required = STRICT_REQUIRED_SECTIONS if strict else BASE_REQUIRED_SECTIONS

    for section in required:
        if section not in text:
            errors.append(f"missing_section:{section}")

    blocks = split_comment_blocks(text)
    if not blocks:
        errors.append("missing_comment_blocks")

    for title, block in blocks:
        if "示例占位，交付前删除" in title or "示例占位" in block:
            errors.append(f"placeholder_comment_not_removed:{title}")
        for field in COMMENT_FIELDS:
            pattern = rf"\*\*{re.escape(field)}\*\*\s*[：:]"
            if not re.search(pattern, block):
                errors.append(f"missing_comment_field:{title}:{field}")
        if strict and not re.search(r"(ISSUE-|MATRIX-|OBJ-|FACT-|REF-|[A-Z]+-\d{3})", block):
            errors.append(f"missing_trace_identifier:{title}")

    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, text):
            errors.append(f"forbidden_loose_language:{pattern}")

    fact_section = text.split("## 五、可验证声明、方法前提与事实核验表", 1)
    if len(fact_section) == 2:
        fact_body = fact_section[1].split("\n## ", 1)[0]
        if "|" not in fact_body:
            errors.append("verifiable_claim_factcheck_table_missing")
    elif strict:
        errors.append("verifiable_claim_factcheck_section_missing")

    if strict and "blocked" not in text.lower() and "剩余风险" not in text:
        warnings.append("strict_report_should_explain_blocked_or_residual_risk")

    if strict:
        needed_terms = ["可验证声明", "方法前提", "复核"]
        for term in needed_terms:
            if term not in text:
                warnings.append(f"strict_report_should_discuss:{term}")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "comment_count": len(blocks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a Markdown thesis audit report.")
    parser.add_argument("--report", required=True, help="Markdown report path")
    parser.add_argument("--strict", action="store_true", help="Require completion-gate sections and trace ids")
    parser.add_argument("--json", action="store_true", help="Print JSON")
    args = parser.parse_args()

    path = Path(args.report).expanduser().resolve()
    text = path.read_text(encoding="utf-8")
    result = validate(text, args.strict)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("ok", result["ok"])
        print("comment_count", result["comment_count"])
        for error in result["errors"]:
            print("error", error)
        for warning in result["warnings"]:
            print("warning", warning)

    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
