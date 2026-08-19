import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "eval_gate.py"


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


def criterion(payload: dict, name: str) -> dict:
    for c in payload["required_criteria"] + payload["upside_criteria"]:
        if c["id"] == name:
            return c
    raise AssertionError(f"{name} not evaluated")


def test_published_rule_agent_is_a_stable_negative():
    payload = run("gate", "--agent", "rule_agent", expect=1)
    assert payload["baseline"] == "deterministic_priority"
    assert payload["verdict"] == "negative"
    assert criterion(payload, "survey_completeness_drop")["passed"] is False
    assert payload["deltas"]["completeness_pp"] == -9.44


def test_followup_clears_its_bar_even_though_the_gate_fails():
    payload = run("gate", "--agent", "rule_agent", expect=1)
    assert criterion(payload, "followup_relative_gain")["passed"] is True
    assert payload["deltas"]["followup_rel_pct"] == 47.22
    assert criterion(payload, "utility_relative_gain")["passed"] is False


def test_random_policy_fails_safety_and_validity():
    payload = run("gate", "--agent", "random", "--baseline", "deterministic_priority", expect=1)
    assert criterion(payload, "no_unsafe_attempts")["passed"] is False
    assert criterion(payload, "invalid_action_rate")["passed"] is False


def test_baseline_defaults_to_strongest_non_agent_by_utility():
    payload = run("gate", "--agent", "rule_agent", expect=1)
    assert payload["baseline"] == "deterministic_priority"


def test_a_passing_policy_is_reported_as_positive(tmp_path):
    csv = tmp_path / "metrics.csv"
    csv.write_text(
        "policy,mean_utility,survey_completeness_pct,high_value_followup_pct,invalid_actions,unsafe_attempts_blocked,episodes\n"
        "deterministic_priority,4.0000,61.11,49.81,0,0,90\n"
        "rule_agent,4.4000,60.00,70.00,0,0,90\n",
        encoding="utf-8",
    )
    payload = run("--csv", str(csv), "gate", "--agent", "rule_agent", expect=0)
    assert payload["verdict"] == "positive"


def test_required_pass_without_upside_is_inconclusive(tmp_path):
    csv = tmp_path / "metrics.csv"
    csv.write_text(
        "policy,mean_utility,survey_completeness_pct,high_value_followup_pct,invalid_actions,unsafe_attempts_blocked,episodes\n"
        "deterministic_priority,4.0000,61.11,49.81,0,0,90\n"
        "rule_agent,4.0100,60.00,50.00,0,0,90\n",
        encoding="utf-8",
    )
    payload = run("--csv", str(csv), "gate", "--agent", "rule_agent", expect=1)
    assert payload["verdict"] == "inconclusive"


def test_bar_is_available_before_any_number():
    payload = run("bar", expect=0)
    assert payload["bar"]["completeness_drop_pp_max"] == 5.0
    assert payload["bar"]["followup_rel_gain_min_pct"] == 20.0
