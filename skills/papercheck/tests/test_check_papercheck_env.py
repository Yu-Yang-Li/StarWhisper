import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_papercheck_env.py"


def run_check_env(tmp_path, env=None):
    rules_repo = tmp_path / "rules"
    (rules_repo / "config").mkdir(parents=True)
    for relative in [
        "README.md",
        "requirements.txt",
        "run_server.py",
        "app/main.py",
        "core/checker/citation_checker.py",
    ]:
        path = rules_repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    (rules_repo / "config" / "config.json").write_text(
        json.dumps({"mineru_config": {"api_key": ""}}),
        encoding="utf-8",
    )

    merged_env = os.environ.copy()
    merged_env.pop("MINERU_API_KEY", None)
    merged_env.update(env or {})
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--rules-repo", str(rules_repo)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=merged_env,
    )
    return completed


def test_check_env_warns_when_mineru_is_missing_and_describes_fallback(tmp_path):
    completed = run_check_env(tmp_path)
    report = json.loads(completed.stdout)

    assert report["pdf_extraction"]["configured"] is False
    assert "PyMuPDF/fitz" in report["pdf_extraction"]["fallback"]
    assert "best no-key built-in fallback" in report["pdf_extraction"]["fallback_quality"]
    assert "MINERU_API_KEY" in report["pdf_extraction"]["user_action"]
    assert any("MinerU API key is not configured" in warning for warning in report["warnings"])


def test_check_env_reports_mineru_configured_without_printing_secret(tmp_path):
    completed = run_check_env(tmp_path, env={"MINERU_API_KEY": "test-mineru-secret"})
    report = json.loads(completed.stdout)

    assert report["pdf_extraction"]["configured"] is True
    assert report["pdf_extraction"]["env_mineru_api_key_set"] is True
    assert "test-mineru-secret" not in completed.stdout
    assert "test-mineru-secret" not in completed.stderr
