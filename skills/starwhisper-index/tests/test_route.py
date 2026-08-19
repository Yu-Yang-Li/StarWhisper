import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "route.py"
CATALOG = json.loads((ROOT / "catalog.json").read_text(encoding="utf-8"))


def run(query: str) -> dict:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--query", query, "--json"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(completed.stdout)


def test_night_question_routes_to_a_runnable_skill():
    payload = run("observe_config 一夜能排多少目标")
    assert payload["skills"][0]["skill"] == "starwhisper-night-plan"
    assert "plan_night.py" in payload["skills"][0]["run"]


def test_candidate_question_routes_to_snclock():
    payload = run("这批年轻超新星候选还在两天内吗")
    assert payload["skills"][0]["skill"] == "starwhisper-snclock"


def test_policy_question_routes_to_the_gate():
    payload = run("规则策略过线了吗 巡天完成度")
    assert payload["skills"][0]["skill"] == "starwhisper-explore"


def test_reference_only_line_is_marked_as_not_runnable():
    payload = run("Kepler 光变分类的代码在哪")
    assert payload["skills"] == []
    assert payload["assets"][0]["path"] == "StarWhisper_LC/"
    assert "reference material" in payload["action"]


def test_unmatched_query_does_not_guess():
    payload = run("zzz-nothing-here")
    assert payload["skills"] == []
    assert payload["assets"] == []
    assert "do not guess" in payload["action"]


def test_every_catalog_skill_directory_exists():
    skills_dir = ROOT.parent
    for entry in CATALOG["skills"]:
        assert (skills_dir / entry["skill"] / "SKILL.md").exists(), entry["skill"]
