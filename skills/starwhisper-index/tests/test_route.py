import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "route.py"


def run(query: str) -> dict:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--query", query, "--json"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(completed.stdout)


def test_telescope_query_routes_to_ngss():
    payload = run("NGSS 夜计划 NINA")
    assert payload["matches"][0]["skill"] == "starwhisper-telescope"
    assert payload["matches"][0]["path"] == "NGSS"


def test_explore_query_does_not_claim_hardware():
    payload = run("Explore 稳定负结果 巡天完成度")
    assert payload["matches"][0]["skill"] == "starwhisper-explore"
    assert "hardware" in payload["note"]


def test_emptyish_query_defaults_without_crash():
    payload = run("zzz-no-such-line")
    assert payload["matches"]
    assert payload["matches"][0]["skill"] == "starwhisper-llm"
