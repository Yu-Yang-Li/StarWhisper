import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "stream_deep_research.py"


def run_stream(log_path, *extra_args):
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--from-log", str(log_path), *extra_args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]


def assert_timestamped(events):
    assert events
    for event in events:
        assert "T" in event["ts"]
        assert isinstance(event["elapsed_ms"], int)
        assert event["elapsed_ms"] >= 0


def test_sse_log_is_forwarded_as_progress_events_before_final_summary(tmp_path):
    log_path = tmp_path / "deep_research.sse"
    log_path.write_text(
        "\n".join(
            [
                "event: phase",
                'data: {"phase":"KeywordPlanning"}',
                "",
                "event: keywords",
                'data: {"keywords":["agentic science","benchmark"]}',
                "",
                "event: private_search_summary",
                'data: {"totalResults":31,"failures":[]}',
                "",
                "event: references",
                'data: {"references":[{"title":"Paper A","url":"https://example.test/a","sourceApi":"searchArxiv"}]}',
                "",
                "event: delta",
                'data: {"content":"partial answer"}',
                "",
            ]
        ),
        encoding="utf-8",
    )

    events = run_stream(log_path)

    assert_timestamped(events)
    assert [event["event"] for event in events] == [
        "stream_started",
        "phase",
        "keywords_ready",
        "private_search_summary",
        "references_ready",
        "answer_delta",
        "stream_final",
    ]
    assert events[4]["references_count"] == 1
    assert events[4]["first_references"][0]["title"] == "Paper A"
    assert events[-1]["answer_status"] == "incomplete"
    assert events[-1]["user_action"] == "先展示已返回 references 和 answer 片段；继续等待、收窄问题或改走候选论文过滤"


def test_done_event_marks_stream_complete(tmp_path):
    log_path = tmp_path / "complete.sse"
    log_path.write_text(
        "\n".join(
            [
                "event: references",
                'data: {"references":[{"title":"Paper A"}]}',
                "",
                "event: delta",
                'data: {"content":"final answer"}',
                "",
                "event: done",
                'data: {"ok":true}',
                "",
            ]
        ),
        encoding="utf-8",
    )

    events = run_stream(log_path)

    assert_timestamped(events)
    assert events[-1]["event"] == "stream_final"
    assert events[-1]["answer_status"] == "complete"
    assert events[-1]["has_done"] is True


def test_max_events_interrupts_stream_but_keeps_final_summary(tmp_path):
    log_path = tmp_path / "long.sse"
    log_path.write_text(
        "\n".join(
            [
                "event: references",
                'data: {"references":[{"title":"Paper A"}]}',
                "",
                "event: delta",
                'data: {"content":"first chunk"}',
                "",
                "event: delta",
                'data: {"content":"second chunk"}',
                "",
            ]
        ),
        encoding="utf-8",
    )

    events = run_stream(log_path, "--max-events", "2")

    assert_timestamped(events)
    assert [event["event"] for event in events] == [
        "stream_started",
        "references_ready",
        "answer_delta",
        "stream_interrupted",
        "stream_final",
    ]
    assert events[-2]["reason"] == "max_events"
    assert events[-1]["interrupted"]["reason"] == "max_events"
    assert events[-1]["answer_status"] == "incomplete"
    assert events[-1]["user_action"] == "先展示已返回 references 和 answer 片段；继续等待、收窄问题或改走候选论文过滤"


def test_missing_endpoint_url_reports_interface_unavailable_without_hanging():
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--endpoint-url",
            "",
            "--prompt",
            "test question",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    events = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]

    assert_timestamped(events)
    assert completed.returncode == 2
    assert [event["event"] for event in events] == [
        "stream_started",
        "interface_unavailable",
        "stream_final",
    ]
    assert events[-1]["answer_status"] == "interface_unavailable"
    assert "https://giiisp.com/#/mcp/authenticate" in events[-1]["user_action"]
