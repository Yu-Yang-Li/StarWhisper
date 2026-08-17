from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from contracts.report_contract import build_issue_buckets, build_summary

CONTRACT_V2_VERSION = "2.0.0"
AGENT_COMMAND_SCHEMA_VERSION = "v1"


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _ensure_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _ensure_list(value: Any):
    return value if isinstance(value, list) else []


def build_error(*, code: str, message: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = {
        "code": code,
        "message": message,
    }
    if details:
        payload["details"] = details
    return payload


def build_run(
    *,
    operation: str,
    status: str,
    run_id: Optional[str] = None,
    started_at: Optional[str] = None,
    finished_at: Optional[str] = None,
    duration_ms: int = 0,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    now = datetime.now().isoformat()
    return {
        "run_id": run_id or f"{operation}_{int(datetime.now().timestamp())}",
        "operation": operation,
        "status": status,
        "started_at": started_at or now,
        "finished_at": finished_at or now,
        "duration_ms": _safe_int(duration_ms),
        "metadata": _ensure_dict(metadata),
    }


def build_contract(
    *,
    operation: str,
    status: str,
    summary: Optional[Dict[str, Any]] = None,
    issues: Optional[Dict[str, Any]] = None,
    evidence: Optional[Dict[str, Any]] = None,
    artifacts: Optional[Dict[str, Any]] = None,
    error: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
    started_at: Optional[str] = None,
    finished_at: Optional[str] = None,
    duration_ms: int = 0,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "contract_version": CONTRACT_V2_VERSION,
        "agent_command_schema_version": AGENT_COMMAND_SCHEMA_VERSION,
        "run": build_run(
            operation=operation,
            status=status,
            run_id=run_id,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            metadata=metadata,
        ),
        "summary": _ensure_dict(summary),
        "issues": _ensure_dict(issues),
        "evidence": _ensure_dict(evidence),
        "artifacts": _ensure_dict(artifacts),
        "error": error,
    }


def build_analysis_contract(
    *,
    raw_report: Dict[str, Any],
    status: str,
    metadata: Optional[Dict[str, Any]] = None,
    artifacts: Optional[Dict[str, Any]] = None,
    error: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return build_contract(
        operation="analysis.report",
        status=status,
        summary=build_summary(raw_report),
        issues=build_issue_buckets(raw_report),
        evidence={
            "results": _ensure_list(raw_report.get("results")),
            "unused_references": _ensure_list(raw_report.get("unused_references")),
        },
        artifacts=artifacts,
        error=error,
        metadata=metadata,
    )
