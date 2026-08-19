import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "eval_varlen.py"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def run(*args, expect=None) -> dict:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--json", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if expect is not None:
        assert completed.returncode == expect, completed.stderr or completed.stdout
    return json.loads(completed.stdout)


def test_table_has_eleven_published_configs():
    payload = run("table", expect=0)
    assert payload["n"] == 11


def test_varlen_best_is_matched_finetune():
    payload = run("best", "--pool", "varlen", expect=0)
    assert payload["best"]["exp_id"] == "e2e_tf_matched_ft"
    assert payload["best"]["pool"] == "varlen"


def test_fifty_obs_winner_is_not_treated_as_varlen_best():
    fifty = run("best", "--pool", "50obs", expect=0)
    varlen = run("best", "--pool", "varlen", expect=0)
    assert fifty["best"]["exp_id"] == "e2e_tf_matched_50"
    assert fifty["best"]["exp_id"] != varlen["best"]["exp_id"]
    assert fifty["best"]["macro_f1"] > varlen["best"]["macro_f1"]


def test_matched_scratch_stays_in_the_varlen_table():
    payload = run("table", "--pool", "varlen", expect=0)
    ids = [row["exp_id"] for row in payload["rows"]]
    assert "e2e_tf_matched_scratch" in ids
    scratch = next(row for row in payload["rows"] if row["exp_id"] == "e2e_tf_matched_scratch")
    assert scratch["macro_f1"] < 0.6


def test_same_pool_compare_is_allowed():
    payload = run("compare", "--a", "e2e_tf_matched_ft", "--b", "xgb_1117", expect=0)
    assert payload["comparable"] is True
    assert payload["delta"]["macro_f1"] > 0


def test_cross_pool_compare_is_rejected_as_incomparable():
    payload = run("compare", "--a", "e2e_tf_matched_50", "--b", "e2e_tf_matched_ft", expect=1)
    assert payload["comparable"] is False
    assert "do not treat the delta as a method gain" in payload["must_state"][0]


def test_unknown_raw_label_fails():
    payload = run("labels", "--raw", "BYDra,NotAClass", expect=1)
    assert payload["unknown"] == ["NotAClass"]
    assert any(row["merged"] == "Active" for row in payload["mapped"])


def test_contract_check_rejects_bad_rows():
    payload = run("check", "--csv", str(FIXTURES / "bad.csv"), expect=1)
    assert payload["errors"] >= 1


def test_contract_check_accepts_merged_and_raw_labels():
    payload = run("check", "--csv", str(FIXTURES / "ok.csv"), expect=0)
    assert payload["errors"] == 0
    assert payload["n_rows"] == 4
