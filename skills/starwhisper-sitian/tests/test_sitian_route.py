import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "route.py"


def test_clock_query_routes_to_explosion_time():
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--query", "超新星时钟 explosion epoch", "--json"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    payload = json.loads(completed.stdout)
    assert payload["matches"][0]["id"] == "snc-explosion-time"
    assert "SitianClaw" in payload["repo"]
    assert "do not reimplement" in payload["note"]
