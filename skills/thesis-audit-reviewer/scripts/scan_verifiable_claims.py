#!/usr/bin/env python3
"""Extract candidate verifiable claims from text/Markdown/DOCX.

This is a candidate generator, not a final fact checker. AI must clean,
deduplicate, and complete the ledger during review.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

PATTERNS = [
    ("percent", re.compile(r"\d+(?:\.\d+)?\s*%")),
    ("metric_score", re.compile(r"\d+(?:\.\d+)?\s*(?:BLEU|F1|ROUGE|AUC|AP|mAP|accuracy|acc\\.?|precision|recall|perplexity|PPL)\\b", re.IGNORECASE)),
    (
        "money",
        re.compile(
            r"(?:[$¥￥]\s*\d+(?:\.\d+)?\s*(?:万|亿|百万|千万|百亿|万亿)?\s*(?:美元|人民币|元|亿元|亿美元|万元)?|\d+(?:\.\d+)?\s*(?:万|亿|百万|千万|百亿|万亿)?\s*(?:美元|人民币|元|亿元|亿美元|万元))"
        ),
    ),
    ("date", re.compile(r"\d{4}[年/-]\d{1,2}(?:[月/-]\d{1,2}日?)?|\d{4}年|\b(?:19|20)\d{2}\b")),
    ("sample", re.compile(r"\d+(?:\.\d+)?\s*(?:个年份|年观测|个|家|人|份|条|篇|组|次|名|项|天|个月|季度|样本|观测值)")),
    (
        "sample",
        re.compile(
            r"\d+(?:\.\d+)?\s*(?:k|K|m|M|b|B|thousand|million|billion)?\s*"
            r"(?:samples?|examples?|instances?|sentences?|tokens?|words?|documents?|images?|classes?|categories?|"
            r"layers?|heads?|GPUs?|TPUs?|CPUs?|parameters?|steps?|epochs?|hours?|days?|models?|tasks?|datasets?)\b",
            re.IGNORECASE,
        ),
    ),
    ("cross_reference", re.compile(r"第[一二三四五六七八九十百\d]+章|表\s*\d+(?:[-－]\d+)?|图\s*\d+(?:[-－]\d+)?|公式\s*\(?\d+(?:[-－]\d+)?\)?|附录\s*[A-Za-z一二三四五六七八九十\d]+")),
    ("cross_reference", re.compile(r"\b(?:Table|Fig\\.?|Figure|Equation|Eq\\.?|Appendix|Section|Sec\\.?)\s*[A-Za-z]?\d+(?:[.-]\d+)?\b", re.IGNORECASE)),
]

FIELDS = [
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


def read_docx(path: Path) -> list[tuple[str, str]]:
    with zipfile.ZipFile(path) as zf:
        root = ET.fromstring(zf.read("word/document.xml"))
    rows: list[tuple[str, str]] = []
    for idx, para in enumerate(root.findall(".//w:p", NS), start=1):
        text = "".join(node.text or "" for node in para.findall(".//w:t", NS)).strip()
        if text:
            rows.append((f"paragraph:{idx}", text))
    return rows


def read_text(path: Path) -> list[tuple[str, str]]:
    if path.suffix.lower() == ".docx":
        return read_docx(path)
    rows = []
    for idx, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        line = line.strip()
        if line:
            rows.append((f"line:{idx}", line))
    return rows


def context(text: str, start: int, end: int, width: int = 90) -> str:
    left = max(0, start - width)
    right = min(len(text), end + width)
    return text[left:right].replace("\n", " ").strip()


def verification_action(claim_type: str, excerpt: str) -> str:
    if claim_type == "cross_reference":
        return "internal_crosscheck"
    if claim_type == "date":
        if re.search(r"窗口|事件|样本|期间|实施|发布|上市|交易|估计", excerpt):
            return "method_premise_check"
        return "external_factcheck"
    if claim_type in {"money", "percent", "number", "sample"}:
        if re.search(r"计算|得分|收益|效率|回归|显著|模型|贡献率|指标|均值|增长率|结果", excerpt):
            return "minimal_recalculation"
        return "internal_crosscheck"
    return "external_factcheck"


def status_for(action: str) -> str:
    return {
        "internal_crosscheck": "needs_internal_crosscheck",
        "external_factcheck": "needs_factcheck",
        "method_premise_check": "needs_method_premise_check",
        "minimal_recalculation": "needs_recalculation",
        "author_source_required": "needs_author_source",
        "blocked": "blocked",
    }.get(action, "pending")


def unit_for(value: str) -> str:
    match = re.search(
        r"(美元|人民币|亿元|亿美元|万元|元|%|个|家|人|份|条|篇|年|组|次|名|项|天|个月|季度|样本|观测值|"
        r"BLEU|F1|ROUGE|AUC|AP|mAP|accuracy|acc\\.?|precision|recall|perplexity|PPL|"
        r"samples?|examples?|instances?|sentences?|tokens?|words?|documents?|images?|classes?|categories?|"
        r"layers?|heads?|GPUs?|TPUs?|CPUs?|parameters?|steps?|epochs?|hours?|days?|models?|tasks?|datasets?)",
        value,
        re.IGNORECASE,
    )
    return match.group(1) if match else ""


def scan(path: Path, paper_id: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for location, text in read_text(path):
        for claim_type, pattern in PATTERNS:
            for match in pattern.finditer(text):
                value = match.group(0).strip()
                excerpt = context(text, match.start(), match.end())
                key = (location, claim_type, value)
                if key in seen:
                    continue
                seen.add(key)
                action = verification_action(claim_type, excerpt)
                rows.append(
                    {
                        "claim_id": f"CLAIM-{len(rows) + 1:03d}",
                        "paper_id": paper_id,
                        "location": location,
                        "claim_type": claim_type,
                        "claim_text": excerpt,
                        "value": value,
                        "unit": unit_for(value),
                        "subject": "",
                        "method_id": "",
                        "verification_action": action,
                        "status": status_for(action),
                        "verification_notes": "candidate_from_script; AI must clean/deduplicate/complete",
                        "issue_id": "",
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan text/Markdown/DOCX for candidate verifiable claims.")
    parser.add_argument("--input", required=True, help="Input text, Markdown, or DOCX path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--paper-id", default="", help="Paper id; defaults to input stem")
    parser.add_argument("--json-output", default="", help="Optional JSON output path")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    paper_id = args.paper_id or input_path.stem
    rows = scan(input_path, paper_id)
    write_csv(Path(args.output).expanduser().resolve(), rows)
    if args.json_output:
        out = Path(args.json_output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"input": str(input_path), "output": args.output, "claim_count": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
