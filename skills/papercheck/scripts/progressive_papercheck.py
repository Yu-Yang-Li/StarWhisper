#!/usr/bin/env python
"""Emit PaperCheck progress events as JSONL."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SKILL_ROOT = Path(__file__).resolve().parents[1]
RULES_ROOT = SKILL_ROOT / "assets" / "paperchecker-rules"
AUTH_URL = "https://mineru.net"
NUMBERED_CITATION_RE = re.compile(r"\[(\d+)(?:\s*-\s*(\d+))?\]")


STARTED = time.monotonic()


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def elapsed_ms() -> int:
    return int((time.monotonic() - STARTED) * 1000)


def short_message(event: dict[str, Any]) -> str:
    name = event.get("event")
    if name == "papercheck_started":
        return f"开始检查，模式：{event.get('mode', 'subjective')}"
    if name == "mode_selected":
        return f"已选择 {event.get('mode')} 模式"
    if name == "env_check_started":
        return "检查运行环境"
    if name == "env_check_complete":
        return "环境检查完成"
    if name == "parser_selected":
        parser = event.get("parser")
        if parser == "docx_evidence_extractor":
            return "读取 DOCX 引用和参考文献"
        if parser == "pdf_mineru_then_pymupdf":
            if event.get("mineru_configured"):
                return "读取 PDF：优先使用 MinerU"
            return f"读取 PDF：未配置 MinerU API key，请到 {AUTH_URL} 申请；先使用本地 fallback"
        return "已选择解析器"
    if name == "evidence_extract_started":
        return "开始抽取论文证据"
    if name == "pdf_extract_started":
        return "开始抽取 PDF 文本"
    if name == "evidence_ready":
        parts = [
            f"引用 {event.get('citation_count', 0)}",
            f"参考文献 {event.get('reference_count', 0)}",
        ]
        if "missing_citations_count" in event:
            parts.append(f"缺失引用 {event.get('missing_citations_count', 0)}")
        if "unused_references_count" in event:
            parts.append(f"未使用参考 {event.get('unused_references_count', 0)}")
        return "证据抽取完成：" + "，".join(parts)
    if name == "rules_check_started":
        return "开始规则检查，通常需要几秒"
    if name == "rules_check_complete":
        bits = []
        if event.get("match_rate") is not None:
            bits.append(f"匹配率 {event.get('match_rate')}")
        if event.get("unused_references_count") is not None:
            bits.append(f"未使用参考 {event.get('unused_references_count')}")
        if event.get("reference_format_issue_count") is not None:
            bits.append(f"格式问题 {event.get('reference_format_issue_count')}")
        return "规则检查完成" + (("：" + "，".join(bits)) if bits else "")
    if name == "semantic_review_skipped":
        return "quick 模式跳过语义审阅"
    if name == "model_review_ready":
        count = event.get("review_item_count")
        return f"已准备语义审阅项：{count}" if count is not None else "已准备语义审阅"
    if name == "source_verification_ready":
        return "full 模式会标注来源核验边界"
    if name == "report_ready":
        return "报告已生成"
    if name == "papercheck_complete":
        return "检查完成"
    if name == "input_error":
        return f"输入文件有问题：{event.get('reason')}"
    if name == "evidence_failed":
        return f"证据抽取失败，请检查文件或配置 MinerU API key：{AUTH_URL}"
    if name == "evidence_warning":
        return str(event.get("summary") or "证据不足，需要确认文档格式")
    if name == "pdf_fallback_notice":
        return f"未配置 MinerU API key，请到 {event.get('auth_url', AUTH_URL)} 申请；PDF 先使用本地 fallback，结果需复核"
    if name == "rules_check_failed":
        return "规则检查未完成，保留已抽取证据"
    return str(name or "papercheck_event")


def emit(event: dict[str, Any]) -> None:
    event.setdefault("ts", now_iso())
    event.setdefault("elapsed_ms", elapsed_ms())
    event.setdefault("message", short_message(event))
    print(json.dumps(event, ensure_ascii=False, separators=(",", ":")), flush=True)


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def run_env_check() -> dict[str, Any]:
    command = [sys.executable, str(SKILL_ROOT / "scripts" / "check_papercheck_env.py")]
    completed = subprocess.run(
        command,
        cwd=str(SKILL_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError:
        report = {
            "warnings": ["check_papercheck_env.py did not return valid JSON"],
            "stderr": completed.stderr[-500:],
        }
    report["exit_code"] = completed.returncode
    return report


def env_summary(report: dict[str, Any]) -> dict[str, Any]:
    imports = {
        item.get("module"): bool(item.get("importable"))
        for item in report.get("python_imports", [])
        if isinstance(item, dict)
    }
    pdf = report.get("pdf_extraction", {}) if isinstance(report.get("pdf_extraction"), dict) else {}
    return {
        "exit_code": report.get("exit_code"),
        "warnings": report.get("warnings", []),
        "pdf_extraction": {
            "mineru_configured": bool(pdf.get("configured")),
            "fitz_importable": bool(imports.get("fitz")),
            "pymupdf4llm_importable": bool(imports.get("pymupdf4llm")),
            "fallback": pdf.get("fallback"),
            "fallback_quality": pdf.get("fallback_quality"),
            "user_action": pdf.get("user_action"),
            "auth_url": AUTH_URL,
        },
    }


def default_out(input_path: Path) -> Path:
    if input_path.suffix.lower() == ".pdf":
        return input_path.with_suffix(".papercheck-pdf-evidence.json")
    return input_path.with_suffix(".citation-evidence.json")


def expand_numbers(text: str) -> list[int]:
    numbers: list[int] = []
    for match in NUMBERED_CITATION_RE.finditer(text or ""):
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if end < start or end - start > 200:
            numbers.append(start)
            continue
        numbers.extend(range(start, end + 1))
    return numbers


def reference_number(text: str) -> int | None:
    match = re.match(r"^\s*\[(\d+)\]", text or "")
    return int(match.group(1)) if match else None


def build_review_items_from_pdf(citations: list[dict[str, Any]], references: list[dict[str, Any]]) -> dict[str, Any]:
    reference_by_number = {}
    for reference in references:
        number = reference_number(reference.get("text", ""))
        if number is not None and number not in reference_by_number:
            reference_by_number[number] = reference.get("text", "")

    cited_numbers: set[int] = set()
    contexts_by_number: dict[int, list[dict[str, Any]]] = {}
    for citation in citations:
        for number in expand_numbers(citation.get("text", "")):
            cited_numbers.add(number)
            contexts_by_number.setdefault(number, []).append(
                {
                    "context": citation.get("context", ""),
                    "matched_text": citation.get("text", ""),
                    "match_type": "pdf_extractor",
                }
            )

    items = []
    for number in sorted(cited_numbers):
        contexts = contexts_by_number.get(number, [])[:3]
        items.append(
            {
                "citation": f"[{number}]",
                "number": number,
                "reference": reference_by_number.get(number),
                "reference_found": number in reference_by_number,
                "contexts": contexts,
                "context_count": len(contexts_by_number.get(number, [])),
                "needs_model_review": bool(contexts and number in reference_by_number),
            }
        )
    reference_numbers = set(reference_by_number)
    return {
        "items": items,
        "missing_citations": [f"[{number}]" for number in sorted(cited_numbers - reference_numbers)],
        "unused_references": [f"[{number}]" for number in sorted(reference_numbers - cited_numbers)],
    }


def mode_description(mode: str) -> str:
    return {
        "quick": "Run structural evidence extraction and bundled rules checks only.",
        "subjective": "Run rules plus prepare current-model citation support review items.",
        "full": "Run rules plus prepare current-model review and source-verification limits.",
    }[mode]


def run_rules_report(input_path: Path) -> dict[str, Any]:
    sys.path.insert(0, str(RULES_ROOT))
    from contracts.v2_contract import build_analysis_contract
    from services.report_service import analyze_document

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
        analysis = analyze_document(
            str(input_path),
            author_format="full",
            citation_standard="ucas",
        )
    raw_report = analysis.get("raw_report", {})
    contract = build_analysis_contract(
        raw_report=raw_report,
        status="succeeded",
        metadata={
            "author_format": "full",
            "citation_standard": "ucas",
            "entrypoint": "progressive_papercheck.rules",
        },
    )
    contract["run"]["started_at"] = analysis.get("started_at")
    contract["run"]["finished_at"] = analysis.get("finished_at")
    contract["run"]["duration_ms"] = analysis.get("duration_ms", 0)
    return {
        "contract": contract,
        "stdout": stdout_buffer.getvalue().strip(),
        "stderr": stderr_buffer.getvalue().strip(),
    }


def evidence_summary(evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "citation_count": evidence.get("citation_count", 0),
        "reference_count": evidence.get("reference_count", 0),
        "missing_citations": evidence.get("missing_citations", []),
        "unused_references": evidence.get("unused_references", []),
        "model_review_count": len([item for item in evidence.get("items", []) if item.get("needs_model_review")]),
    }


def evidence_warnings(evidence: dict[str, Any], suffix: str) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []
    citation_count = int(evidence.get("citation_count") or 0)
    reference_count = int(evidence.get("reference_count") or 0)
    if citation_count == 0 and reference_count == 0:
        warnings.append(
            {
                "code": "no_citation_evidence",
                "summary": "未检测到编号引用或参考文献，请确认文档是否包含规范引用段落",
                "detail": "This usually means the document has no [1] style citations, the reference section heading is missing, or the file is not an academic paper body.",
            }
        )
    elif citation_count == 0:
        warnings.append(
            {
                "code": "no_citations",
                "summary": "未检测到正文编号引用，请确认引用格式",
                "detail": "PaperCheck currently audits numbered citations such as [1] and [2-4].",
            }
        )
    elif reference_count == 0:
        warnings.append(
            {
                "code": "no_references",
                "summary": "未检测到参考文献列表，请确认参考文献标题和编号格式",
                "detail": "Expected a References or 参考文献 section with numbered entries.",
            }
        )
    if suffix == ".pdf":
        metadata = evidence.get("metadata", {}) if isinstance(evidence.get("metadata"), dict) else {}
        warning = metadata.get("pdf_extraction_warning")
        method = metadata.get("pdf_extraction_method")
        if warning:
            warnings.append(
                {
                    "code": "pdf_fallback",
                    "summary": f"PDF 使用 {method or 'local'} fallback，复杂版式结果需要复核",
                    "detail": warning,
                }
            )
    return warnings


def rules_summary(rules_contract: dict[str, Any] | None) -> dict[str, Any]:
    if not rules_contract:
        return {}
    summary = rules_contract.get("summary", {}) if isinstance(rules_contract.get("summary"), dict) else {}
    return {
        "total_citations": summary.get("total_citations"),
        "total_references": summary.get("total_references"),
        "match_rate": summary.get("match_rate"),
        "unused_references_count": summary.get("unused_references_count"),
        "reference_format_issue_count": summary.get("reference_format_issue_count"),
        "citation_style_issue_count": summary.get("citation_style_issue_count"),
        "high_confidence_issue_count": summary.get("high_confidence_issue_count"),
    }


def first_context(item: dict[str, Any]) -> str:
    contexts = item.get("contexts", [])
    if not contexts:
        return ""
    context = contexts[0].get("context", "")
    return context.replace("\n", " ").strip()


def write_markdown_report(
    *,
    report_path: Path,
    input_path: Path,
    mode: str,
    env_report: dict[str, Any],
    evidence: dict[str, Any],
    rules_contract: dict[str, Any] | None,
) -> None:
    pdf = env_report.get("pdf_extraction", {}) if isinstance(env_report.get("pdf_extraction"), dict) else {}
    ev = evidence_summary(evidence)
    warnings = evidence_warnings(evidence, input_path.suffix.lower())
    rs = rules_summary(rules_contract)
    lines = [
        "# PaperCheck Audit Report",
        "",
        f"- Input: `{input_path}`",
        f"- Mode: `{mode}` - {mode_description(mode)}",
        f"- Generated: `{now_iso()}`",
        f"- Evidence source: `{evidence.get('source_docx') or evidence.get('source_pdf')}`",
        "",
        "## Runtime",
        "",
        f"- MinerU configured: `{bool(pdf.get('configured'))}`",
    ]
    if input_path.suffix.lower() == ".pdf":
        metadata = evidence.get("metadata", {}) if isinstance(evidence.get("metadata"), dict) else {}
        lines.extend(
            [
                f"- PDF extraction method: `{metadata.get('pdf_extraction_method', 'unknown')}`",
                f"- PDF warning: {metadata.get('pdf_extraction_warning') or 'none'}",
            ]
        )
        if not pdf.get("configured"):
            lines.append(f"- User action: {pdf.get('user_action') or 'Configure MINERU_API_KEY for higher-confidence PDF extraction.'}")
    lines.extend(["", "## Result Summary", ""])
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning['summary']}")
    else:
        lines.append("- Evidence extraction and rules checks completed. Review the issue counts below.")
    lines.extend(
        [
            "",
            "## Evidence Summary",
            "",
            f"- Citations found: `{ev['citation_count']}`",
            f"- References found: `{ev['reference_count']}`",
            f"- Missing citations: `{', '.join(ev['missing_citations']) if ev['missing_citations'] else 'none'}`",
            f"- Unused references: `{', '.join(ev['unused_references']) if ev['unused_references'] else 'none'}`",
            f"- Current-model review items: `{ev['model_review_count']}`",
        ]
    )
    if rs:
        lines.extend(
            [
                "",
                "## Rules Summary",
                "",
                f"- Rule citations: `{rs.get('total_citations')}`",
                f"- Rule references: `{rs.get('total_references')}`",
                f"- Match rate: `{rs.get('match_rate')}`",
                f"- Unused references: `{rs.get('unused_references_count')}`",
                f"- Reference format issues: `{rs.get('reference_format_issue_count')}`",
                f"- Citation style issues: `{rs.get('citation_style_issue_count')}`",
                f"- High-confidence issues: `{rs.get('high_confidence_issue_count')}`",
            ]
        )
    else:
        lines.extend(["", "## Rules Summary", "", "- Rules report was not available."])

    lines.extend(["", "## Current-Model Review Queue", ""])
    review_items = [item for item in evidence.get("items", []) if item.get("needs_model_review")]
    if mode == "quick":
        lines.append("- Skipped in quick mode.")
    elif not review_items:
        lines.append("- No citation with both reference entry and local context was ready for semantic review.")
    else:
        for item in review_items[:30]:
            lines.extend(
                [
                    f"### {item.get('citation')}",
                    "",
                    f"- Reference: {item.get('reference')}",
                    f"- Context: {first_context(item)}",
                    "- Required judgment: decide whether the citation is topically supported by the local context and reference entry. Mark broad or uncertain matches as `待人工确认`.",
                    "",
                ]
            )

    if mode == "full":
        lines.extend(
            [
                "## Full-Mode Limits",
                "",
                "- This skill package does not require provider model keys. It can judge citation support from extracted context and reference text with the mounted Codex model.",
                "- Strong source-content claims still require supplied source PDFs, DOI/OA retrieval evidence, or another verified paper source. Without that evidence, mark the item as `待人工确认` rather than claiming the cited paper proves the sentence.",
            ]
        )
    lines.extend(
        [
            "",
            "## Audit Limit",
            "",
            "PaperCheck is an audit assistant. It is not academic, legal, or publication certification.",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def extract_docx(input_path: Path, out_path: Path, window: int, max_contexts: int) -> int:
    command = [
        sys.executable,
        str(SKILL_ROOT / "scripts" / "extract_citation_evidence.py"),
        str(input_path),
        "--out",
        str(out_path),
        "--window",
        str(window),
        "--max-contexts",
        str(max_contexts),
    ]
    completed = subprocess.run(
        command,
        cwd=str(SKILL_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if completed.stdout.strip():
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError:
            payload = {"stdout": completed.stdout[-500:]}
        emit({"event": "extractor_stdout", "payload": payload})
    if completed.stderr.strip():
        emit({"event": "extractor_stderr", "stderr_excerpt": completed.stderr[-500:]})
    return completed.returncode


def extract_pdf(input_path: Path, out_path: Path) -> None:
    sys.path.insert(0, str(RULES_ROOT))
    from core.extractor.pdf_extractor import PDFExtractor

    document = PDFExtractor().extract(str(input_path))
    citations = [
        {
            "text": citation.text,
            "format_type": citation.format_type,
            "context": citation.context,
            "author": citation.author,
            "year": citation.year,
        }
        for citation in document.citations
    ]
    references = [
        {"text": reference.text, "author": reference.author, "year": reference.year}
        for reference in document.references
    ]
    review_payload = build_review_items_from_pdf(citations, references)
    payload = {
        "source_pdf": str(input_path),
        "paragraph_count": len(document.content),
        "table_count": len(document.tables),
        "citation_count": len(document.citations),
        "reference_count": len(document.references),
        "metadata": document.metadata,
        "content": document.content,
        "tables": document.tables,
        "citations": citations,
        "references": references,
        **review_payload,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Emit PaperCheck progress events as JSONL.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--report", type=Path, help="Write an integrated Markdown report.")
    parser.add_argument("--mode", choices=["quick", "subjective", "full"], default="subjective")
    parser.add_argument("--window", type=int, default=1)
    parser.add_argument("--max-contexts", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    input_path = args.input
    out_path = args.out or default_out(input_path)
    report_path = args.report
    suffix = input_path.suffix.lower()

    emit({"event": "papercheck_started", "input": str(input_path), "output": str(out_path), "mode": args.mode})
    emit({"event": "mode_selected", "mode": args.mode, "description": mode_description(args.mode)})

    if not input_path.exists():
        emit({"event": "input_error", "reason": "file_not_found", "input": str(input_path)})
        return 2
    if suffix not in {".docx", ".pdf"}:
        emit({"event": "input_error", "reason": "unsupported_file_type", "suffix": suffix})
        return 2

    emit({"event": "env_check_started"})
    env_report = run_env_check()
    summary = env_summary(env_report)
    emit({"event": "env_check_complete", **summary})

    if suffix == ".docx":
        emit({
            "event": "parser_selected",
            "parser": "docx_evidence_extractor",
            "note": "DOCX evidence extraction is local and does not need provider API keys.",
        })
        if args.dry_run:
            emit({"event": "papercheck_complete", "status": "dry_run", "output": str(out_path)})
            return 0
        emit({"event": "evidence_extract_started", "input": str(input_path)})
        code = extract_docx(input_path, out_path, args.window, args.max_contexts)
        if code != 0:
            emit({"event": "evidence_failed", "status": "failed", "exit_code": code})
            return code
        evidence = read_json(out_path)
        rules_contract = None
        emit({
            "event": "evidence_ready",
            "status": "ready",
            "output": str(out_path),
            "citation_count": evidence.get("citation_count"),
            "reference_count": evidence.get("reference_count"),
            "missing_citations_count": len(evidence.get("missing_citations", [])),
            "unused_references_count": len(evidence.get("unused_references", [])),
        })
        for warning in evidence_warnings(evidence, suffix):
            emit({"event": "evidence_warning", **warning})
        emit({"event": "rules_check_started", "citation_standard": "ucas", "author_format": "full"})
        try:
            rules_result = run_rules_report(input_path)
            rules_contract = rules_result["contract"]
            if rules_result.get("stdout"):
                emit({"event": "rules_stdout", "stdout_excerpt": rules_result["stdout"][-1000:]})
            if rules_result.get("stderr"):
                emit({"event": "rules_stderr", "stderr_excerpt": rules_result["stderr"][-1000:]})
            emit({"event": "rules_check_complete", "status": "ready", **rules_summary(rules_contract)})
        except Exception as exc:
            emit({"event": "rules_check_failed", "status": "warning", "reason": str(exc)})
        if args.mode == "quick":
            emit({"event": "semantic_review_skipped", "mode": args.mode})
        else:
            emit({
                "event": "model_review_ready",
                "review_item_count": evidence_summary(evidence)["model_review_count"],
                "note": "Use the mounted Codex model to judge evidence items marked needs_model_review.",
            })
        if args.mode == "full":
            emit({"event": "source_verification_ready", "note": "Use supplied source PDFs or verified DOI/OA retrieval before claiming cited papers prove a sentence."})
        if report_path:
            write_markdown_report(
                report_path=report_path,
                input_path=input_path,
                mode=args.mode,
                env_report=env_report,
                evidence=evidence,
                rules_contract=rules_contract,
            )
            emit({"event": "report_ready", "report": str(report_path)})
        emit({"event": "papercheck_complete", "status": "ready", "output": str(out_path), "report": str(report_path) if report_path else None})
        return 0

    emit({
        "event": "parser_selected",
        "parser": "pdf_mineru_then_pymupdf",
        "mineru_configured": summary["pdf_extraction"]["mineru_configured"],
        "local_fallback_available": summary["pdf_extraction"]["fitz_importable"] or summary["pdf_extraction"]["pymupdf4llm_importable"],
        "fallback_quality": summary["pdf_extraction"]["fallback_quality"],
        "user_action": "" if summary["pdf_extraction"]["mineru_configured"] else summary["pdf_extraction"]["user_action"],
    })
    if not summary["pdf_extraction"]["mineru_configured"]:
        emit({
            "event": "pdf_fallback_notice",
            "method": "pymupdf4llm/PyMuPDF",
            "auth_url": AUTH_URL,
            "summary": "PDF 未配置 MinerU，正在使用本地 fallback；文本层 PDF 可用，扫描件和复杂版式需要复核。",
        })
    if args.dry_run:
        emit({"event": "papercheck_complete", "status": "dry_run", "output": str(out_path)})
        return 0
    emit({"event": "pdf_extract_started", "input": str(input_path)})
    try:
        stdout_buffer = io.StringIO()
        with contextlib.redirect_stdout(stdout_buffer):
            extract_pdf(input_path, out_path)
        captured_stdout = stdout_buffer.getvalue().strip()
        if captured_stdout:
            emit({"event": "extractor_stdout", "stdout_excerpt": captured_stdout[-1000:]})
    except Exception as exc:
        emit({
            "event": "evidence_failed",
            "status": "failed",
            "reason": str(exc),
            "user_action": "Configure MINERU_API_KEY or install PyMuPDF and pymupdf4llm, then retry.",
        })
        return 1
    evidence = read_json(out_path)
    metadata = evidence.get("metadata", {}) if isinstance(evidence.get("metadata"), dict) else {}
    rules_contract = None
    emit({
        "event": "evidence_ready",
        "status": "ready",
        "output": str(out_path),
        "citation_count": evidence.get("citation_count"),
        "reference_count": evidence.get("reference_count"),
        "paragraph_count": evidence.get("paragraph_count"),
        "pdf_extraction_method": metadata.get("pdf_extraction_method"),
        "pdf_extraction_warning": metadata.get("pdf_extraction_warning"),
    })
    for warning in evidence_warnings(evidence, suffix):
        emit({"event": "evidence_warning", **warning})
    emit({"event": "rules_check_started", "citation_standard": "ucas", "author_format": "full"})
    try:
        rules_result = run_rules_report(input_path)
        rules_contract = rules_result["contract"]
        if rules_result.get("stdout"):
            emit({"event": "rules_stdout", "stdout_excerpt": rules_result["stdout"][-1000:]})
        if rules_result.get("stderr"):
            emit({"event": "rules_stderr", "stderr_excerpt": rules_result["stderr"][-1000:]})
        emit({"event": "rules_check_complete", "status": "ready", **rules_summary(rules_contract)})
    except Exception as exc:
        emit({"event": "rules_check_failed", "status": "warning", "reason": str(exc)})
    if args.mode == "quick":
        emit({"event": "semantic_review_skipped", "mode": args.mode})
    else:
        emit({
            "event": "model_review_ready",
            "review_item_count": evidence_summary(evidence)["model_review_count"],
            "note": "PDF findings should be labeled fallback/needs-review unless MinerU or source-content verification is used.",
        })
    if args.mode == "full":
        emit({"event": "source_verification_ready", "note": "Use supplied source PDFs or verified DOI/OA retrieval before claiming cited papers prove a sentence."})
    if report_path:
        write_markdown_report(
            report_path=report_path,
            input_path=input_path,
            mode=args.mode,
            env_report=env_report,
            evidence=evidence,
            rules_contract=rules_contract,
        )
        emit({"event": "report_ready", "report": str(report_path)})
    emit({"event": "papercheck_complete", "status": "ready", "output": str(out_path), "report": str(report_path) if report_path else None})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
