#!/usr/bin/env python3
"""Check the thesis audit reviewer skill runtime."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path


SKILL_DIR = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "SKILL.md",
    "requirements.txt",
    "references/default_review_sources.md",
    "references/input_output_policy.md",
    "references/audit_operating_protocol.md",
    "references/docx_integrity_gate.md",
    "references/method_and_fact_audit.md",
    "references/work_products.md",
    "references/completion_gate.md",
    "references/standard_report_template.md",
    "scripts/init_audit_workspace.py",
    "scripts/docx_integrity_scan.py",
    "scripts/scan_verifiable_claims.py",
    "scripts/mineru_vlm_extract.py",
    "scripts/split_mineru_vlm_pages.py",
    "scripts/pdf_local_fallback_extract.py",
    "scripts/render_md_report_pdf.py",
    "scripts/validate_audit_report.py",
]

PYTHON_MODULES = [
    ("requests", "requests"),
    ("PyMuPDF", "fitz"),
    ("reportlab", "reportlab"),
]

OPTIONAL_PYTHON_MODULES = [
    ("pymupdf4llm", "pymupdf4llm", "higher-quality local PDF Markdown fallback"),
]

OPTIONAL_COMMANDS = [
    ("textutil", "macOS DOC/DOCX text conversion"),
    ("soffice", "LibreOffice fallback for DOC/DOCX conversion"),
    ("pdftotext", "Poppler PDF text fallback"),
]


def module_ok(import_name: str) -> bool:
    return importlib.util.find_spec(import_name) is not None


def main() -> None:
    parser = argparse.ArgumentParser(description="Check thesis audit reviewer runtime.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--soft", action="store_true", help="Always exit 0")
    args = parser.parse_args()

    files = [{"path": item, "ok": (SKILL_DIR / item).exists()} for item in REQUIRED_FILES]
    modules = [{"name": name, "import": import_name, "ok": module_ok(import_name)} for name, import_name in PYTHON_MODULES]
    optional_modules = [
        {"name": name, "import": import_name, "ok": module_ok(import_name), "purpose": purpose}
        for name, import_name, purpose in OPTIONAL_PYTHON_MODULES
    ]
    commands = [{"name": name, "purpose": purpose, "ok": shutil.which(name) is not None} for name, purpose in OPTIONAL_COMMANDS]
    errors = [f"missing_file:{item['path']}" for item in files if not item["ok"]]
    errors += [f"missing_module:{item['import']}" for item in modules if not item["ok"]]
    result = {
        "ok": not errors,
        "skill_dir": str(SKILL_DIR),
        "python": sys.version.split()[0],
        "files": files,
        "modules": modules,
        "optional_modules": optional_modules,
        "optional_commands": commands,
        "errors": errors,
    }

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("ok", result["ok"])
        for error in errors:
            print("error", error)
        for item in optional_modules:
            print("optional_module", item["name"], item["ok"], "-", item["purpose"])
        for item in commands:
            print("optional", item["name"], item["ok"], "-", item["purpose"])

    raise SystemExit(0 if result["ok"] or args.soft else 1)


if __name__ == "__main__":
    main()
