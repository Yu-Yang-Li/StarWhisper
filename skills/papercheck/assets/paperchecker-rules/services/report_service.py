"""Run the bundled citation checks on a local document."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from contracts.report_contract import normalize_author_format, normalize_citation_standard
from core.processors.citation_processor import CitationProcessor
from services.exceptions import ServiceValidationError

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
