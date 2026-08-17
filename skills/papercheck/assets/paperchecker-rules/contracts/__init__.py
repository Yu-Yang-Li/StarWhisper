"""PaperChecker contract helpers."""

from .report_contract import (
    CONTRACT_VERSION,
    build_contract_payload,
    build_summary,
    normalize_author_format,
    normalize_citation_standard,
)
from .v2_contract import (
    AGENT_COMMAND_SCHEMA_VERSION,
    CONTRACT_V2_VERSION,
    build_analysis_contract,
    build_contract,
    build_error,
    build_run,
)

__all__ = [
    "CONTRACT_VERSION",
    "build_contract_payload",
    "build_summary",
    "normalize_author_format",
    "normalize_citation_standard",
    "CONTRACT_V2_VERSION",
    "AGENT_COMMAND_SCHEMA_VERSION",
    "build_run",
    "build_error",
    "build_contract",
    "build_analysis_contract",
]
