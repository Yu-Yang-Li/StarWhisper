"""Keep installed offline tools usable across the repository directory move."""

import importlib.util
import json
from pathlib import Path

import pytest


SKILLS = Path(__file__).resolve().parents[1]
CASES = [
    ("starwhisper-snclock", "screen_snclock.py", "snclock", "snclock_20261020.csv"),
    ("starwhisper-explore", "eval_gate.py", "explore", "published_metrics.csv"),
    ("starwhisper-night-plan", "plan_night.py", "ngss", "observe_config.json"),
]


def load_tool(skill, script):
    spec = importlib.util.spec_from_file_location(skill, SKILLS / skill / "scripts" / script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_input(root, project, filename, legacy=False):
    folder = root / ("NGSS" if project == "ngss" else project) if legacy else root / "projects" / project
    folder.mkdir(parents=True)
    path = folder / filename
    path.write_text('{"source": "test input"}' if filename.endswith(".json") else "test\n", encoding="utf-8")
    return path


def selected_input(tool, project):
    if project == "ngss":
        config, source = tool.resolve_config(None)
        assert config == {"source": "test input"}
        return source.replace("\\", "/")
    return tool.default_csv()


@pytest.mark.parametrize("skill,script,project,filename", CASES)
@pytest.mark.parametrize("legacy", [False, True], ids=["projects", "legacy"])
@pytest.mark.parametrize("discovery", ["environment", "checkout"])
def test_project_inputs_are_found(tmp_path, monkeypatch, skill, script, project, filename, legacy, discovery):
    tool = load_tool(skill, script)
    root = tmp_path / "repository"
    (root / "skills").mkdir(parents=True)
    expected = make_input(root, project, filename, legacy)
    monkeypatch.delenv("STARWHISPER_ROOT", raising=False)
    if discovery == "environment":
        monkeypatch.setenv("STARWHISPER_ROOT", str(root))
        monkeypatch.setattr(tool, "__file__", str(tmp_path / "installed" / skill / "scripts" / script))
    else:
        monkeypatch.setattr(tool, "__file__", str(root / "skills" / skill / "scripts" / script))
    actual = selected_input(tool, project)
    if project == "ngss":
        assert actual == expected.relative_to(root).as_posix()
    else:
        assert actual == expected


@pytest.mark.parametrize("skill,script,project,filename", CASES)
def test_new_project_directory_takes_precedence(tmp_path, monkeypatch, skill, script, project, filename):
    tool = load_tool(skill, script)
    expected = make_input(tmp_path, project, filename)
    make_input(tmp_path, project, filename, legacy=True)
    monkeypatch.setenv("STARWHISPER_ROOT", str(tmp_path))
    actual = selected_input(tool, project)
    assert actual == (expected.relative_to(tmp_path).as_posix() if project == "ngss" else expected)


def test_installed_tools_keep_their_missing_data_behavior(tmp_path, monkeypatch):
    monkeypatch.delenv("STARWHISPER_ROOT", raising=False)
    for skill, script, project, _ in CASES:
        tool = load_tool(skill, script)
        monkeypatch.setattr(tool, "__file__", str(tmp_path / "installed" / skill / "scripts" / script))
        if project == "snclock":
            with pytest.raises(SystemExit, match="pass --csv"):
                tool.default_csv()
        elif project == "explore":
            assert tool.default_csv() == tool.BUNDLED
        else:
            config, source = tool.resolve_config(None)
            assert config == json.loads(tool.EXAMPLE_CONFIG.read_text(encoding="utf-8"))
            assert source.startswith("bundled example")


def test_ngss_flat_configuration_uses_per_target_overhead(tmp_path):
    tool = load_tool("starwhisper-night-plan", "plan_night.py")
    path = tmp_path / "observe_config.json"
    path.write_text(json.dumps({
        "inherit": True, "early_night": 0.5, "midnight": 2.0,
        "midmorning": 2.0, "early_morning": 2.0, "d_moon": 15,
        "FilterType": ["L"], "TotalExposureCount": 3, "ExposureTime": 120, "WaitTime": 1.0,
    }), encoding="utf-8")
    config, _ = tool.resolve_config(path)
    assert tool.check_config(config) == []
    result = tool.budget(config, None)
    assert result["seconds_per_target"] == 420.0
    assert result["targets_that_fit"] == 55
    assert result["wait_minutes_between_exposures"] == 0
    assert tool.budget(config, 90)["seconds_per_target"] == 450.0


def test_ngss_flat_configuration_preserves_invalid_values(tmp_path):
    tool = load_tool("starwhisper-night-plan", "plan_night.py")
    path = tmp_path / "observe_config.json"
    path.write_text(json.dumps({"early_night": -1, "FilterType": [], "ExposureTime": -2,
                                "TotalExposureCount": 0, "WaitTime": -1}), encoding="utf-8")
    config, _ = tool.resolve_config(path)
    findings = tool.check_config(config)
    assert any(item["field"] == "constraints.d_moon" for item in findings)
    assert any(item["field"] == "exposure.target_overhead_seconds" for item in findings)
    assert sum(item["level"] == "error" for item in findings) >= 5
