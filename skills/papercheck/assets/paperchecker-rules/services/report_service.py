from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from contracts.report_contract import (
    build_contract_payload,
    normalize_author_format,
    normalize_citation_standard,
)
from core.processors.citation_processor import CitationProcessor
from services.exceptions import ServiceValidationError
from utils.report_markdown import build_markdown_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUPPORTED_EXTENSIONS = {".doc", ".docx", ".pdf", ".md"}



def _resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()



def validate_input_file(file_path: str) -> Path:
    resolved = _resolve_path(file_path)
    if not resolved.exists():
        raise ServiceValidationError(f"输入文件不存在: {resolved}")
    if resolved.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ServiceValidationError(
            f"不支持的文件类型: {resolved.suffix}. 支持类型: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    if resolved.stat().st_size <= 0:
        raise ServiceValidationError("输入文件为空（0 字节），请重新上传或重新下载原始文件")
    return resolved



def _prepare_output_path(path: Optional[str], default_dir: str, default_name: str) -> Path:
    if path:
        output = _resolve_path(path)
    else:
        output = PROJECT_ROOT / default_dir / default_name
    output.parent.mkdir(parents=True, exist_ok=True)
    return output



def analyze_document(
    file_path: str,
    author_format: str = "full",
    citation_standard: str = "legacy",
) -> Dict[str, Any]:
    normalized_author_format = normalize_author_format(author_format)
    normalized_citation_standard = normalize_citation_standard(citation_standard)
    resolved_input = validate_input_file(file_path)

    started = datetime.now()
    processor = CitationProcessor()
    raw_report = processor.process(
        str(resolved_input),
        author_format=normalized_author_format,
        citation_standard=normalized_citation_standard,
    )
    raw_report.setdefault("citation_standard", normalized_citation_standard)
    finished = datetime.now()
    duration_ms = int((finished - started).total_seconds() * 1000)

    return {
        "raw_report": raw_report,
        "input_path": str(resolved_input),
        "author_format": normalized_author_format,
        "citation_standard": normalized_citation_standard,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_ms": duration_ms,
    }



def _dump_json(payload: Dict[str, Any], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)



def _dump_markdown(raw_report: Dict[str, Any], output_path: Path) -> str:
    markdown = build_markdown_report(raw_report)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(markdown)
    return markdown



def _dump_pdf(markdown: str, output_path: Path) -> None:
    from utils.report_pdf import save_markdown_as_pdf

    save_markdown_as_pdf(markdown, str(output_path))



def analyze_and_export(
    *,
    file_path: str,
    author_format: str = "full",
    citation_standard: str = "legacy",
    output_json_path: Optional[str] = None,
    output_md_path: Optional[str] = None,
    output_pdf_path: Optional[str] = None,
    include_raw_in_contract: bool = False,
) -> Dict[str, Any]:
    analysis = analyze_document(
        file_path=file_path,
        author_format=author_format,
        citation_standard=citation_standard,
    )
    raw_report = analysis["raw_report"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(analysis["input_path"]).stem
    json_path = _prepare_output_path(
        output_json_path,
        default_dir="reports_json",
        default_name=f"{stem}_{timestamp}.json",
    )
    md_path = _prepare_output_path(
        output_md_path,
        default_dir="reports_md",
        default_name=f"{stem}_{timestamp}.md",
    )
    if output_pdf_path:
        pdf_path = _prepare_output_path(
            output_pdf_path,
            default_dir="reports_pdf",
            default_name=f"{stem}_{timestamp}.pdf",
        )
    elif output_md_path:
        pdf_path = _prepare_output_path(
            str(md_path.with_suffix(".pdf")),
            default_dir="reports_pdf",
            default_name=f"{stem}_{timestamp}.pdf",
        )
    else:
        pdf_path = _prepare_output_path(
            None,
            default_dir="reports_pdf",
            default_name=f"{stem}_{timestamp}.pdf",
        )

    _dump_json(raw_report, json_path)
    markdown = _dump_markdown(raw_report, md_path)
    _dump_pdf(markdown, pdf_path)

    artifacts = {
        "report_json_path": str(json_path),
        "report_md_path": str(md_path),
        "report_pdf_path": str(pdf_path),
    }
    contract_payload = build_contract_payload(
        raw_report,
        input_path=analysis["input_path"],
        author_format=analysis["author_format"],
        citation_standard=analysis["citation_standard"],
        status="succeeded",
        started_at=analysis["started_at"],
        finished_at=analysis["finished_at"],
        duration_ms=analysis["duration_ms"],
        artifacts=artifacts,
        include_raw_report=include_raw_in_contract,
    )

    return {
        "raw_report": raw_report,
        "contract_payload": contract_payload,
        "artifacts": artifacts,
    }



def run_smoke_suite(
    *,
    manifest: str = "tests/smoke_cases_manifest.json",
    output_dir: Optional[str] = None,
    author_format: str = "full",
    citation_standard: str = "legacy",
) -> Dict[str, Any]:
    normalized_author_format = normalize_author_format(author_format)
    normalized_citation_standard = normalize_citation_standard(citation_standard)
    resolved_manifest = _resolve_path(manifest)
    if not resolved_manifest.exists():
        raise ServiceValidationError(f"smoke manifest 不存在: {resolved_manifest}")

    resolved_output_dir = _resolve_path(output_dir) if output_dir else (PROJECT_ROOT / "audit_runs" / f"{datetime.now().strftime('%Y-%m-%d')}_smoke_cli")
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tests" / "run_smoke_suite.py"),
        "--manifest",
        str(resolved_manifest),
        "--output-dir",
        str(resolved_output_dir),
        "--author-format",
        normalized_author_format,
        "--citation-standard",
        normalized_citation_standard,
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True)

    parsed_flags: Dict[str, str] = {}
    combined_output = f"{completed.stdout}\n{completed.stderr}"
    for line in combined_output.splitlines():
        if line.startswith("SMOKE_") and "=" in line:
            key, value = line.split("=", 1)
            parsed_flags[key.strip()] = value.strip()

    smoke_summary_json = parsed_flags.get("SMOKE_SUMMARY_JSON")
    smoke_summary: Optional[Dict[str, Any]] = None
    if smoke_summary_json:
        summary_path = Path(smoke_summary_json)
        if not summary_path.is_absolute():
            summary_path = (PROJECT_ROOT / summary_path).resolve()
        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as f:
                smoke_summary = json.load(f)

    return {
        "return_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "parsed_flags": parsed_flags,
        "smoke_summary": smoke_summary,
    }
