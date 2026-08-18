import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ads_first_search.py"


def test_dry_run_prints_ads_then_arxiv_without_network():
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--query", "StarWhisper Telescope NGSS", "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    payload = json.loads(completed.stdout)
    assert payload["route"][0] == "nasa-ads"
    assert payload["docs"] == []
    assert "no papers were invented" in payload["safety"]
    ads = payload["planned_requests"]["ads"]
    assert ads["url"] == "https://api.adsabs.harvard.edu/v1/search/query"
    assert "StarWhisper Telescope NGSS" in ads["params"]["q"]
    arxiv = payload["planned_requests"]["arxiv"]
    assert arxiv["params"]["search_query"].startswith("cat:astro-ph")
