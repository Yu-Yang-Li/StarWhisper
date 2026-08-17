import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "dry_run_paper_search.py"


EXPECTED = {
    "arxiv": ("/first/paper/searchArxiv", {"key": "test query", "pageNum": 1, "pageSize": 10}),
    "arxiv-abstract": ("/first/paper/searchArxivByAbstract", {"key": "test query", "pageNum": 1, "pageSize": 10}),
    "arxiv-no": ("/first/paper/searchArxivByArxivNo1", {"key": "test query", "pageNum": 1, "pageSize": 10}),
    "arxiv-title": ("/first/paper/searchArxivByTitle", {"key": "test query", "pageNum": 1, "pageSize": 10}),
    "oa": ("/first/oaPaper/searchArticlesByQuery1", {"titleAndAbs": ["test query"]}),
}


def run_dry(mode, output_format="request"):
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--mode",
            mode,
            "--query",
            "test query",
            "--format",
            output_format,
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(completed.stdout)


def test_request_modes_build_expected_post_body_without_network():
    for mode, (path, body) in EXPECTED.items():
        payload = run_dry(mode)
        assert payload["method"] == "POST"
        assert payload["url"] == "https://giiisp.com" + path
        assert payload["body"] == body
        assert "no request was sent" in payload["safety"]


def test_example_formats_have_expected_contract_keys():
    for output_format in ["normalized-example", "fallback-example", "end-to-end-example"]:
        payload = run_dry("arxiv-title", output_format)
        assert "safety" in payload
        assert "no request was sent" in payload["safety"]

    assert "normalized_results" in run_dry("arxiv-title", "normalized-example")
    assert "fallback_sources" in run_dry("arxiv-title", "fallback-example")
    assert "output_table" in run_dry("arxiv-title", "end-to-end-example")
