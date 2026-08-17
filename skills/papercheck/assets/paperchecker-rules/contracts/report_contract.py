from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

CONTRACT_VERSION = "1.0.0"
SUPPORTED_AUTHOR_FORMATS = {"full", "abbrev"}
SUPPORTED_CITATION_STANDARDS = {"legacy", "ucas"}


def normalize_author_format(author_format: Optional[str]) -> str:
    if not author_format:
        return "full"
    value = str(author_format).strip().lower()
    if value in SUPPORTED_AUTHOR_FORMATS:
        return value
    return "full"


def normalize_citation_standard(citation_standard: Optional[str]) -> str:
    if not citation_standard:
        return "legacy"
    value = str(citation_standard).strip().lower()
    if value in SUPPORTED_CITATION_STANDARDS:
        return value
    return "legacy"


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _ensure_list(value: Any) -> List[Dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return []


def _collect_issue_layer_counts(issues: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    strength = {"strong_rule": 0, "heuristic_rule": 0, "unclassified": 0}
    confidence = {"high": 0, "medium": 0, "low": 0, "unclassified": 0}
    for issue in issues:
        rule_strength = issue.get("rule_strength")
        if rule_strength in {"strong_rule", "heuristic_rule"}:
            strength[rule_strength] += 1
        else:
            strength["unclassified"] += 1

        confidence_tier = issue.get("confidence_tier")
        if confidence_tier in {"high", "medium", "low"}:
            confidence[confidence_tier] += 1
        else:
            confidence["unclassified"] += 1
    return {"strength": strength, "confidence": confidence}


def build_summary(raw_report: Dict[str, Any]) -> Dict[str, Any]:
    reference_format_issues = _ensure_list(raw_report.get("reference_format_issues"))
    citation_style_issues = _ensure_list(raw_report.get("citation_style_issues"))
    all_issues = reference_format_issues + citation_style_issues
    layer_counts = _collect_issue_layer_counts(all_issues)
    strength_counts = layer_counts["strength"]
    confidence_counts = layer_counts["confidence"]

    return {
        "total_citations": _safe_int(raw_report.get("total_citations")),
        "total_references": _safe_int(raw_report.get("total_references")),
        "matched_count": _safe_int(raw_report.get("matched_count")),
        "unmatched_count": _safe_int(raw_report.get("unmatched_count")),
        "corrected_count": _safe_int(raw_report.get("corrected_count")),
        "formatted_count": _safe_int(raw_report.get("formatted_count")),
        "citation_formatted_count": _safe_int(raw_report.get("citation_formatted_count")),
        "reference_formatted_count": _safe_int(raw_report.get("reference_formatted_count")),
        "unused_references_count": _safe_int(raw_report.get("unused_references_count")),
        "reference_format_issue_count": _safe_int(
            raw_report.get("reference_format_issue_count"),
            default=len(reference_format_issues),
        )
        if raw_report.get("reference_format_issue_count") is not None
        else len(reference_format_issues),
        "citation_style_issue_count": _safe_int(
            raw_report.get("citation_style_issue_count"),
            default=len(citation_style_issues),
        )
        if raw_report.get("citation_style_issue_count") is not None
        else len(citation_style_issues),
        "unmatched_true_missing_count": _safe_int(raw_report.get("unmatched_true_missing_count")),
        "unmatched_ambiguous_count": _safe_int(raw_report.get("unmatched_ambiguous_count")),
        "unmatched_extraction_noise_count": _safe_int(raw_report.get("unmatched_extraction_noise_count")),
        "unmatched_breakdown": raw_report.get(
            "unmatched_breakdown",
            {
                "true_missing": _safe_int(raw_report.get("unmatched_true_missing_count")),
                "ambiguous": _safe_int(raw_report.get("unmatched_ambiguous_count")),
                "extraction_noise": _safe_int(raw_report.get("unmatched_extraction_noise_count")),
            },
        ),
        "match_rate": raw_report.get("match_rate", "0%"),
        "citation_style": raw_report.get("citation_style", "unknown"),
        "citation_standard": normalize_citation_standard(raw_report.get("citation_standard")),
        "citation_numeric_count": _safe_int(raw_report.get("citation_numeric_count")),
        "citation_author_year_count": _safe_int(raw_report.get("citation_author_year_count")),
        "strong_rule_issue_count": _safe_int(
            raw_report.get("strong_rule_issue_count"),
            default=strength_counts["strong_rule"],
        )
        if raw_report.get("strong_rule_issue_count") is not None
        else strength_counts["strong_rule"],
        "heuristic_rule_issue_count": _safe_int(
            raw_report.get("heuristic_rule_issue_count"),
            default=strength_counts["heuristic_rule"],
        )
        if raw_report.get("heuristic_rule_issue_count") is not None
        else strength_counts["heuristic_rule"],
        "unclassified_rule_issue_count": _safe_int(
            raw_report.get("unclassified_rule_issue_count"),
            default=strength_counts["unclassified"],
        )
        if raw_report.get("unclassified_rule_issue_count") is not None
        else strength_counts["unclassified"],
        "high_confidence_issue_count": _safe_int(
            raw_report.get("high_confidence_issue_count"),
            default=confidence_counts["high"],
        )
        if raw_report.get("high_confidence_issue_count") is not None
        else confidence_counts["high"],
        "medium_confidence_issue_count": _safe_int(
            raw_report.get("medium_confidence_issue_count"),
            default=confidence_counts["medium"],
        )
        if raw_report.get("medium_confidence_issue_count") is not None
        else confidence_counts["medium"],
        "low_confidence_issue_count": _safe_int(
            raw_report.get("low_confidence_issue_count"),
            default=confidence_counts["low"],
        )
        if raw_report.get("low_confidence_issue_count") is not None
        else confidence_counts["low"],
        "unclassified_confidence_issue_count": _safe_int(
            raw_report.get("unclassified_confidence_issue_count"),
            default=confidence_counts["unclassified"],
        )
        if raw_report.get("unclassified_confidence_issue_count") is not None
        else confidence_counts["unclassified"],
        "rule_strength_counts": raw_report.get("rule_strength_counts", strength_counts),
        "confidence_tier_counts": raw_report.get("confidence_tier_counts", confidence_counts),
    }


def _extract_unmatched_results(raw_report: Dict[str, Any]) -> List[Dict[str, Any]]:
    results = _ensure_list(raw_report.get("results"))
    return [item for item in results if not bool(item.get("matched"))]


def build_issue_buckets(raw_report: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    citation_formatting = _ensure_list(raw_report.get("citation_formatting_needed"))
    if not citation_formatting:
        citation_formatting = _ensure_list(raw_report.get("formatting_needed"))
    return {
        "corrections_needed": _ensure_list(raw_report.get("corrections_needed")),
        "formatting_needed": citation_formatting,
        "citation_formatting_needed": citation_formatting,
        "reference_formatting_needed": _ensure_list(raw_report.get("reference_formatting_needed")),
        "reference_format_issues": _ensure_list(raw_report.get("reference_format_issues")),
        "citation_style_issues": _ensure_list(raw_report.get("citation_style_issues")),
        "unused_references": _ensure_list(raw_report.get("unused_references")),
        "unmatched_citations": _extract_unmatched_results(raw_report),
        "unmatched_classification": _ensure_list(raw_report.get("unmatched_classification")),
    }


def build_run_block(
    input_path: str,
    author_format: Optional[str],
    citation_standard: Optional[str],
    status: str,
    run_id: str,
    started_at: str,
    finished_at: str,
    duration_ms: int,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "status": status,
        "input_path": str(Path(input_path)),
        "author_format": normalize_author_format(author_format),
        "citation_standard": normalize_citation_standard(citation_standard),
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_ms": _safe_int(duration_ms),
        "error": error,
    }


def build_contract_payload(
    raw_report: Dict[str, Any],
    *,
    input_path: str,
    author_format: Optional[str],
    citation_standard: Optional[str] = None,
    status: str = "succeeded",
    run_id: Optional[str] = None,
    started_at: Optional[str] = None,
    finished_at: Optional[str] = None,
    duration_ms: int = 0,
    artifacts: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    include_raw_report: bool = True,
) -> Dict[str, Any]:
    started = started_at or datetime.now().isoformat()
    finished = finished_at or datetime.now().isoformat()
    run_identity = run_id or f"run_{int(datetime.now().timestamp())}"

    payload: Dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "run": build_run_block(
            input_path=input_path,
            author_format=author_format,
            citation_standard=citation_standard,
            status=status,
            run_id=run_identity,
            started_at=started,
            finished_at=finished,
            duration_ms=duration_ms,
            error=error,
        ),
        "summary": build_summary(raw_report),
        "issues": build_issue_buckets(raw_report),
        "artifacts": artifacts or {},
    }
    if include_raw_report:
        payload["raw_report"] = raw_report
    return payload
