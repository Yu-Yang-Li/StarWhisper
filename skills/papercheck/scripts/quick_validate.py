#!/usr/bin/env python
"""Self-contained quick validator for the packaged PaperCheck skill."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_DIRS = {
    ".git",
    ".github",
    ".idea",
    ".pytest_cache",
    ".ruff_cache",
    ".vscode",
    "__pycache__",
    "node_modules",
    "pdf_cache",
    "temp_uploads",
}
FORBIDDEN_EXTS = {".pyc", ".pyo", ".doc", ".docx", ".pdf", ".zip", ".7z", ".rar"}
FORBIDDEN_FILES = {"TEST" + "_ONLY_KEYS", ".DS_Store"}
SECRET_RE = re.compile(
    "("
    + "s" + "k-" + r"[A-Za-z0-9]{12,}|"
    + "a" + "rk-" + r"[A-Za-z0-9-]{12,}|"
    + "TEST" + "_ONLY_KEYS"
    + ")"
)


def run(cmd: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        cmd,
        cwd=str(SKILL_ROOT),
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def fail_if(condition: bool, message: str, failures: list[str]) -> None:
    if condition:
        failures.append(message)


def parse_skill_header() -> dict[str, str]:
    text = read_text(SKILL_ROOT / "SKILL.md")
    parts = text.split("---", 2)
    if len(parts) < 3 or parts[0] != "":
        raise ValueError("SKILL.md must start with closed YAML frontmatter.")
    header: dict[str, str] = {}
    for line in parts[1].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            header[key.strip()] = value.strip().strip("'\"")
    return header


def scan_residue() -> list[str]:
    issues: list[str] = []
    for path in SKILL_ROOT.rglob("*"):
        rel = path.relative_to(SKILL_ROOT).as_posix()
        parts = set(path.relative_to(SKILL_ROOT).parts)
        if parts & FORBIDDEN_DIRS:
            issues.append(f"forbidden residue: {rel}")
            continue
        if path.is_file():
            if path.name in FORBIDDEN_FILES or path.suffix.lower() in FORBIDDEN_EXTS:
                issues.append(f"forbidden packaged file: {rel}")
            try:
                if path.stat().st_size <= 2_000_000 and SECRET_RE.search(read_text(path)):
                    issues.append(f"secret-like token found: {rel}")
            except OSError:
                issues.append(f"unreadable file: {rel}")
    return issues


def importable(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def validate(smoke_input: Path | None, skip_tests: bool) -> list[str]:
    failures: list[str] = []
    required = [
        "SKILL.md",
        "scripts/extract_citation_evidence.py",
        "scripts/progressive_papercheck.py",
        "scripts/check_papercheck_env.py",
        "assets/paperchecker-rules/README.md",
        "assets/paperchecker-rules/requirements.txt",
        "assets/paperchecker-rules/run_server.py",
        "assets/paperchecker-rules/app/main.py",
        "assets/paperchecker-rules/core/extractor/pdf_extractor.py",
    ]
    for rel in required:
        fail_if(not (SKILL_ROOT / rel).exists(), f"missing required file: {rel}", failures)

    try:
        header = parse_skill_header()
        fail_if(header.get("name") != "papercheck", "SKILL.md name must be papercheck", failures)
        fail_if(len(header.get("description", "")) < 40, "SKILL.md description is too thin", failures)
    except Exception as exc:
        failures.append(str(exc))

    failures.extend(scan_residue())
    for module in ["docx", "fastapi", "fitz", "pymupdf4llm", "requests", "uvicorn"]:
        fail_if(not importable(module), f"Python module not importable: {module}", failures)

    if failures:
        return failures

    env = run([sys.executable, str(SKILL_ROOT / "scripts" / "check_papercheck_env.py")])
    fail_if(env.returncode != 0, "check_papercheck_env.py failed", failures)
    if env.returncode == 0:
        try:
            env_report = json.loads(env.stdout)
            user_action = env_report.get("pdf_extraction", {}).get("user_action") or ""
            if not env_report.get("pdf_extraction", {}).get("configured"):
                fail_if("https://mineru.net" not in user_action, "MinerU action must mention https://mineru.net", failures)
        except json.JSONDecodeError:
            failures.append("check_papercheck_env.py did not emit valid JSON")

    help_run = run([sys.executable, str(SKILL_ROOT / "scripts" / "progressive_papercheck.py"), "--help"])
    fail_if(help_run.returncode != 0, "progressive_papercheck.py --help failed", failures)
    fail_if("--mode {quick,subjective,full}" not in help_run.stdout, "progressive help missing mode choices", failures)

    if smoke_input:
        smoke = run(
            [sys.executable, str(SKILL_ROOT / "scripts" / "progressive_papercheck.py"), str(smoke_input), "--dry-run"],
            timeout=180,
        )
        fail_if(smoke.returncode != 0, "progressive_papercheck.py smoke dry-run failed", failures)
        fail_if("模式：subjective" not in smoke.stdout, "dry-run stream does not prove subjective default", failures)
        if smoke_input.suffix.lower() == ".pdf":
            fail_if(
                "未配置 MinerU API key，请到 https://mineru.net 申请" not in smoke.stdout,
                "PDF dry-run stream missing MinerU action",
                failures,
            )

    if not skip_tests and (SKILL_ROOT / "tests").exists():
        tests = run([sys.executable, "-m", "pytest", str(SKILL_ROOT / "tests"), "-q", "-p", "no:cacheprovider"], timeout=180)
        fail_if(tests.returncode != 0, "pytest failed:\n" + (tests.stdout + tests.stderr).strip(), failures)

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate packaged PaperCheck skill.")
    parser.add_argument("--smoke-input", type=Path)
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    failures = validate(args.smoke_input, args.skip_tests)
    print(json.dumps({"skill": str(SKILL_ROOT), "status": "fail" if failures else "pass", "failures": failures}, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
