import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "semantic_review_dashscope.py"


def minimal_png(width=1, height=1):
    return (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + b"\x08\x02\x00\x00\x00"
        + b"\x00\x00\x00\x00"
    )


def test_missing_dashscope_key_writes_blocked_review(tmp_path):
    run_dir = tmp_path / "smoke_001"
    run_dir.mkdir()
    (run_dir / "generated_image.png").write_bytes(minimal_png())
    env = os.environ.copy()
    env.pop("DASHSCOPE_API_KEY", None)

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--run-dir", str(run_dir)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
    )

    assert completed.returncode == 2
    review = json.loads((run_dir / "semantic_review.json").read_text(encoding="utf-8"))
    assert review["status"] == "blocked"
    assert review["blocker"]["reason"] == "missing DASHSCOPE_API_KEY"
    assert review["overall_ready_to_ship"] is None


def test_missing_image_writes_blocked_review(tmp_path):
    run_dir = tmp_path / "smoke_001"
    run_dir.mkdir()
    env = os.environ.copy()
    env["DASHSCOPE_API_KEY"] = "test-key-not-used"

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--run-dir", str(run_dir)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
    )

    assert completed.returncode == 2
    review = json.loads((run_dir / "semantic_review.json").read_text(encoding="utf-8"))
    assert review["status"] == "blocked"
    assert review["blocker"]["reason"] == "image file is missing"
