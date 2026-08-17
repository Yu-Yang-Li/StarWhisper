import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "select_figure_variant.py"


def write_manifest(path, run_id, status, image_path=None, machine_status=None, blocker=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "giiisp_figure_manifest_v1",
        "run_id": run_id,
        "figure_id": run_id,
        "status": status,
        "output": {"image_path": image_path},
        "quality": {
            "machine_check_summary": {"status": machine_status},
            "quality_review_axes": [
                {"name": "content_accuracy", "score": 8.0},
                {"name": "layout_quality", "score": 7.0},
            ],
        },
        "blocker": blocker,
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")


def write_semantic_manifest(path, run_id, ready, action, missing=None, forbidden=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "giiisp_figure_manifest_v1",
        "run_id": run_id,
        "figure_id": run_id,
        "status": "completed",
        "output": {"image_path": str(path.parent / "generated_image.png")},
        "quality": {
            "machine_check_summary": {"status": "passed"},
            "semantic_review_summary": {
                "status": "completed",
                "provider": "dashscope",
                "model": "qwen3.7-plus",
                "overall_ready_to_ship": ready,
                "recommended_next_action": action,
                "missing_required_labels": missing or [],
                "forbidden_labels_seen": forbidden or [],
            },
        },
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")


def test_selects_completed_machine_passed_variant_over_blocked(tmp_path):
    blocked = tmp_path / "blocked" / "figure_manifest.json"
    completed = tmp_path / "completed" / "figure_manifest.json"
    write_manifest(
        blocked,
        "blocked",
        "blocked",
        machine_status="failed",
        blocker={"reason": "missing GIIISP_AUTH_TOKEN"},
    )
    write_manifest(
        completed,
        "completed",
        "completed",
        image_path=str(tmp_path / "completed" / "generated_image.png"),
        machine_status="passed",
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    payload = json.loads(result.stdout)

    assert payload["best"]["run_id"] == "completed"
    assert payload["best"]["score"] > payload["candidates"][1]["score"]
    assert "machine check passed" in payload["best"]["reasons"]


def test_all_blocked_candidates_are_reported_without_fake_success(tmp_path):
    first = tmp_path / "first" / "figure_manifest.json"
    second = tmp_path / "second" / "figure_manifest.json"
    write_manifest(first, "first", "blocked", machine_status="failed", blocker={"reason": "missing GIIISP_AUTH_TOKEN"})
    write_manifest(second, "second", "blocked", machine_status="failed", blocker={"reason": "reference image path does not exist"})

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    payload = json.loads(result.stdout)

    assert payload["best"]["status"] == "blocked"
    assert payload["best"]["score"] < 0
    assert len(payload["candidates"]) == 2
    assert all(candidate["status"] == "blocked" for candidate in payload["candidates"])


def test_semantic_review_ready_candidate_beats_failed_semantic_candidate(tmp_path):
    ready = tmp_path / "ready" / "figure_manifest.json"
    failed = tmp_path / "failed" / "figure_manifest.json"
    write_semantic_manifest(ready, "ready", True, "deliver")
    write_semantic_manifest(failed, "failed", False, "regenerate", missing=["关键标签"], forbidden=["INPUT"])

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    payload = json.loads(result.stdout)

    assert payload["best"]["run_id"] == "ready"
    assert "semantic review ready to ship" in payload["best"]["reasons"]
    failed_candidate = next(candidate for candidate in payload["candidates"] if candidate["run_id"] == "failed")
    assert "semantic review not ready to ship" in failed_candidate["reasons"]
    assert "missing required labels: 1" in failed_candidate["reasons"]
