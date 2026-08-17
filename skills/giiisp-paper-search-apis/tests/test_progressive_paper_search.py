import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "progressive_paper_search.py"


def run_progressive(*args, env=None):
    merged_env = {**os.environ, **(env or {})}
    # Keep the no-token scenario hermetic on machines that have a real token configured.
    if not (env and "GIIISP_AUTH_TOKEN" in env):
        merged_env.pop("GIIISP_AUTH_TOKEN", None)
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=merged_env,
    )
    return [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]


def assert_timestamped(events):
    assert events
    for event in events:
        assert "T" in event["ts"]
        assert isinstance(event["elapsed_ms"], int)
        assert event["elapsed_ms"] >= 0


def test_dry_run_emits_start_request_and_complete_events_without_network():
    events = run_progressive("--query", "test query", "--mode", "arxiv-title", "--dry-run")

    assert_timestamped(events)
    assert [event["event"] for event in events] == [
        "search_started",
        "auth_status",
        "request_prepared",
        "search_complete",
    ]
    assert events[1]["provider"] == "giiisp"
    assert events[1]["ok"] is False
    assert events[1]["auth_url"] == "https://giiisp.com/#/mcp/authenticate"
    assert events[1]["user_action"] == "申请或刷新 Giiisp MCP 认证后设置 GIIISP_AUTH_TOKEN"
    assert events[2]["request"]["url"] == "https://giiisp.com/first/paper/searchArxivByTitle"
    assert events[2]["request"]["body"] == {"key": "test query", "pageNum": 1, "pageSize": 10}
    assert events[-1]["dry_run"] is True


def test_expansion_plan_emits_each_expanded_route_and_page_in_order():
    events = run_progressive(
        "--query",
        "test query",
        "--mode",
        "arxiv-title",
        "--expand",
        "arxiv-abstract,oa",
        "--max-pages",
        "2",
        "--dry-run",
    )

    assert_timestamped(events)
    request_events = [event for event in events if event["event"] == "request_prepared"]
    assert [event["route"]["mode"] for event in request_events] == [
        "arxiv-title",
        "arxiv-title",
        "arxiv-abstract",
        "arxiv-abstract",
        "oa",
        "oa",
    ]
    assert [event["route"]["page_num"] for event in request_events] == [1, 2, 1, 2, 1, 2]
    assert events[-1]["planned_requests"] == 6


def test_auth_status_reports_configured_token_without_leaking_value():
    events = run_progressive(
        "--query",
        "test query",
        "--mode",
        "arxiv-title",
        "--dry-run",
        env={"GIIISP_AUTH_TOKEN": "test-token-not-leaked"},
    )

    assert_timestamped(events)
    auth_event = next(event for event in events if event["event"] == "auth_status")
    assert auth_event["ok"] is True
    assert auth_event["env_var"] == "GIIISP_AUTH_TOKEN"
    assert "test-token-not-leaked" not in json.dumps(events)
