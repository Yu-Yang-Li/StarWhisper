import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "report_metrics.py"


def test_published_table_is_stable_negative():
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--json"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    payload = json.loads(completed.stdout)
    assert payload["verdict"] == "stable_negative"
    assert payload["environment_code_in_repo"] is False
    delta = payload["rule_minus_deterministic"]
    assert delta["followup_pp"] == 23.52
    assert delta["completeness_pp"] == -9.44
    assert abs(delta["completeness_pp"]) > payload["positive_bar"]["completeness_drop_pp_max"]
    assert payload["rows"]["rule_agent"]["episodes"] == "90"
