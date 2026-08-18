import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "inspect_stack.py"


def test_inspect_never_marks_nina_as_safe():
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--json"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    payload = json.loads(completed.stdout)
    assert payload["mode"] == "inspect"
    assert "/manipulate_nina/{action}" in payload["hardware_blocked"]
    assert "/ftp_transfer" in payload["hardware_blocked"]
    assert "does not start uvicorn" in payload["safety"]
    paths = {row["path"] for row in payload["routes"]}
    assert "/plan_observation" in paths
    assert "/look_config" in paths
