import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "screen_snclock.py"
ASOF = "2026-08-16T04:00:00Z"


def run(*args) -> dict:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--json", *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(completed.stdout)


def test_describe_reports_scope_caveat():
    payload = run("describe")
    assert payload["provenance"]["n_rows"] == 22
    assert payload["selection_tier"]["strict_q84_within_2d"] == 2
    assert any("2026-06-24" in note for note in payload["must_state"])
    assert any("not a classification or a discovery" in note for note in payload["must_state"])


def test_strict_tier_keeps_only_conservative_rows():
    payload = run("screen", "--tier", "strict")
    assert payload["n_selected"] == 2
    for item in payload["selected"]:
        q84 = item["h3_age_q16_q50_q84_days"][2]
        assert q84 <= 2.0


def test_rank_is_youngest_first():
    payload = run("rank", "--top", "5")
    ages = [item["h3_age_q16_q50_q84_days"][1] for item in payload["selected"]]
    assert ages == sorted(ages)
    assert len(ages) == 5


def test_age_grows_with_elapsed_time():
    payload = run("screen", "--tier", "strict", "--asof", ASOF)
    for item in payload["selected"]:
        q50_at_discovery = item["h3_age_q16_q50_q84_days"][1]
        assert item["age_now_q50_days"] > q50_at_discovery


def test_window_narrows_as_of_a_later_time():
    wide = run("window", "--within-days", "40", "--asof", ASOF)
    narrow = run("window", "--within-days", "3", "--asof", ASOF)
    assert narrow["n_selected"] < wide["n_selected"]
    assert narrow["quantile_used"] == "q50"


def test_empty_result_is_reported_not_widened():
    payload = run("screen", "--tier", "strict", "--min-confidence", "HIGH")
    assert payload["n_selected"] == 0
    assert payload["selected"] == []


def test_audit_flags_weak_provenance_and_consistent_tiers():
    payload = run("audit", "--asof", ASOF)
    assert len(payload["weak_provenance_sources"]) == 18
    assert payload["tier_flag_inconsistencies"] == []
    assert payload["missing_host_redshift"] == []
