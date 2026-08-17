#!/usr/bin/env python
"""Check PaperCheck skill-native runtime without printing secrets."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RULES_REPO = Path(
    os.environ.get("PAPERCHECK_RULES_REPO", str(SKILL_ROOT / "assets" / "paperchecker-rules"))
)
PYTHON_IMPORTS = (
    "docx",
    "fastapi",
    "fitz",
    "uvicorn",
    "requests",
    "pymupdf4llm",
)
MINERU_AUTH_URL = "https://mineru.net"


def _load_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def exists(path: Path) -> dict:
    return {"path": str(path), "exists": path.exists()}


def import_status(module: str) -> dict:
    return {"module": module, "importable": importlib.util.find_spec(module) is not None}


def mineru_status(rules_repo: Path) -> dict:
    env_present = bool(os.getenv("MINERU_API_KEY"))
    config_path = rules_repo / "config" / "config.json"
    config = _load_json(config_path)
    mineru_config = config.get("mineru_config") if isinstance(config.get("mineru_config"), dict) else {}
    config_present = bool((mineru_config or {}).get("api_key"))
    configured = env_present or config_present
    return {
        "configured": configured,
        "env_mineru_api_key_set": env_present,
        "config_mineru_api_key_set": config_present,
        "config_path": str(config_path),
        "primary": "MinerU API converts PDF to Markdown with layout-aware extraction.",
        "fallback": (
            "Local fallback first tries pymupdf4llm.to_markdown() for Markdown-like layout text, then "
            "falls back to PyMuPDF/fitz page.get_text(sort=True) per page when MinerU is unavailable."
        ),
        "fallback_quality": (
            "Good for text-layer PDFs and usually better with pymupdf4llm than raw page text because headings, lists, "
            "and some table structure are retained. Still weak for scanned PDFs, complex multi-column layout, heavy "
            "headers/footers, formulas, and reference lists broken across pages. It is the best no-key built-in fallback here, "
            "but not the best overall extraction path; MinerU or another OCR/layout parser is preferred for production PDF audits."
        ),
        "user_action": (
            "Set MINERU_API_KEY or fill assets/paperchecker-rules/config/config.json mineru_config.api_key "
            f"after applying for MinerU access at {MINERU_AUTH_URL}. If no key is available, install PyMuPDF and pymupdf4llm "
            "from assets/paperchecker-rules/requirements.txt; PaperCheck will continue with local fallback and should label PDF findings as fallback/needs-review."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check PaperCheck skill-native runtime.")
    parser.add_argument("--rules-repo", type=Path, default=DEFAULT_RULES_REPO)
    args = parser.parse_args()

    rules_repo = args.rules_repo
    required = [
        SKILL_ROOT / "SKILL.md",
        SKILL_ROOT / "scripts" / "extract_citation_evidence.py",
        rules_repo / "README.md",
        rules_repo / "requirements.txt",
        rules_repo / "run_server.py",
        rules_repo / "app" / "main.py",
        rules_repo / "core" / "checker" / "citation_checker.py",
    ]

    report = {
        "skill_root": str(SKILL_ROOT),
        "rules_repo": exists(rules_repo),
        "required_files": [exists(path) for path in required],
        "python_imports": [import_status(module) for module in PYTHON_IMPORTS],
        "pdf_extraction": mineru_status(rules_repo),
        "python": shutil.which("python") or shutil.which("py") or sys.executable,
        "warnings": [],
        "notes": [
            "No provider API key is required. The mounted Codex model performs semantic review from extracted evidence."
        ],
    }

    missing_imports = [
        item["module"] for item in report["python_imports"] if not item["importable"]
    ]
    if missing_imports:
        report["warnings"].append(
            "Missing Python imports: "
            + ", ".join(missing_imports)
            + ". Install the bundled rules requirements if needed."
        )
    if not report["pdf_extraction"]["configured"]:
        report["warnings"].append(
            "MinerU API key is not configured. PDF uploads will fall back to local pymupdf4llm/PyMuPDF extraction if installed; "
            "tell the user this is usable for text PDFs but less accurate than MinerU/OCR layout extraction."
        )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    missing = [path for path in required if not path.exists()]
    return 1 if missing or missing_imports else 0


if __name__ == "__main__":
    raise SystemExit(main())
