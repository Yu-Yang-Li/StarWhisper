import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_generated_image.py"


def minimal_png(width=1, height=1):
    # Valid enough for the script's header-based PNG dimension parser.
    return (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + b"\x08\x02\x00\x00\x00"
        + b"\x00\x00\x00\x00"
    )


def run_check(*args):
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.returncode, json.loads(completed.stdout)


def test_valid_png_has_machine_check_and_quality_axes(tmp_path):
    image = tmp_path / "generated.png"
    image.write_bytes(minimal_png(1024, 1024))

    code, payload = run_check("--image", str(image))

    assert code == 0
    assert payload["image_type"] == "png"
    assert payload["width"] == 1024
    assert payload["height"] == 1024
    assert payload["machine_check"]["status"] == "passed"
    assert payload["machine_check"]["format_supported"] is True
    assert [axis["name"] for axis in payload["quality_review_axes"]] == [
        "content_accuracy",
        "layout_quality",
        "text_readability",
        "aesthetic_quality",
        "artifact_severity",
    ]


def test_missing_image_fails_machine_check(tmp_path):
    code, payload = run_check("--image", str(tmp_path / "missing.png"))

    assert code == 2
    assert payload["machine_check"]["status"] == "failed"
    assert "image file is missing" in payload["machine_check"]["pixel_issues"]


def test_semantic_check_adds_review_placeholder(tmp_path):
    image = tmp_path / "generated.png"
    image.write_bytes(minimal_png())

    code, payload = run_check("--image", str(image), "--semantic-check")

    assert code == 0
    assert payload["semantic_review"]["status"] == "pending"
    assert payload["semantic_review"]["overall_ready_to_ship"] is None


def test_unsupported_binary_fails_machine_check(tmp_path):
    image = tmp_path / "generated.bin"
    image.write_bytes(b"not an image")

    code, payload = run_check("--image", str(image))

    assert code == 2
    assert payload["image_exists"] is True
    assert payload["supported_type"] is False
    assert payload["machine_check"]["status"] == "failed"
    assert "image type is not PNG, JPEG, or WebP" in payload["machine_check"]["pixel_issues"]
