import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "plan_night.py"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def run(*args, expect: int | None = None) -> dict:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--json", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if expect is not None:
        assert completed.returncode == expect, completed.stderr
    return json.loads(completed.stdout)


def test_reference_config_is_clean():
    payload = run("check-config", expect=0)
    assert payload["errors"] == 0


def test_broken_config_is_rejected(tmp_path):
    bad = tmp_path / "observe_config.json"
    bad.write_text(
        json.dumps({"time_windows": {}, "constraints": {"d_moon": 400}, "filters": [], "exposure": {"count": 0, "time": -1, "wait": -2}}),
        encoding="utf-8",
    )
    payload = run("--config", str(bad), "check-config", expect=1)
    fields = {f["field"] for f in payload["findings"] if f["level"] == "error"}
    assert {"time_windows", "constraints.d_moon", "filters", "exposure.count", "exposure.time", "exposure.wait"} <= fields


def test_budget_matches_the_documented_formula():
    payload = run("budget", expect=0)
    # 1 filter x (3 x 120 s + 2 x 60 s) + 60 s slew = 540 s per target
    assert payload["seconds_per_target"] == 540.0
    assert payload["total_window_hours"] == 6.5
    assert payload["targets_that_fit"] == 43


def test_slew_overhead_reduces_capacity():
    base = run("budget", expect=0)
    slower = run("--slew-seconds", "300", "budget", expect=0)
    assert slower["targets_that_fit"] < base["targets_that_fit"]


def test_target_list_problems_are_caught():
    payload = run("lint-targets", "--targets", str(FIXTURES / "targets_bad.csv"), expect=1)
    messages = " | ".join(f["message"] for f in payload["findings"])
    assert "duplicate target" in messages
    assert "RA 999.0 outside" in messages
    assert "Dec -120.0 outside" in messages
    assert "not a number" in messages


def test_clean_target_list_passes():
    payload = run("lint-targets", "--targets", str(FIXTURES / "targets_ok.csv"), expect=0)
    assert payload["errors"] == 0
    assert payload["n_targets"] == 3


def test_hardware_routes_are_labelled():
    payload = run("endpoints", expect=0)
    assert "/manipulate_nina/{action}" in payload["hardware"]
    assert "/ftp_transfer" in payload["hardware"]
    assert "does not call these endpoints" in payload["safety"]
