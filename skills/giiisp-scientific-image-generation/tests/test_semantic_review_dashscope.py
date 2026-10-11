import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


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


def load_script(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("style_status", ["PASS", "FAIL", "UNCERTAIN", None, "malformed"])
def test_deck_repair_reviews_style_contract(tmp_path, monkeypatch, style_status):
    reviewer = load_script(SCRIPT)
    monkeypatch.delenv("GIIISP_IMAGE_SKILL_DIR", raising=False)
    repair = load_script(ROOT.parent / "visual-deck-builder/scripts/repair_visual_deck.py")
    assert repair.IMAGE_SKILL == ROOT
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key-not-sent")
    (tmp_path / "image_inputs").mkdir()
    (tmp_path / "image_inputs/01.repair.json").write_text(
        json.dumps({"prompt": "Preserve the measured light curve.", "required_labels": ["Flux"]}),
        encoding="utf-8",
    )
    contract = {"palette": {"background": "#FAFAFA"}, "icon_style": "line art"}
    (tmp_path / "visual_style_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    image = tmp_path / "slide.png"
    image.write_bytes(minimal_png())
    axes = [{"name": name, "status": "PASS"} for name in (
        "content_accuracy", "layout_quality", "text_readability", "aesthetic_quality", "artifact_severity"
    )]
    if style_status is not None:
        axes += [{"name": name, "status": style_status} for name in (
            "style_contract_compliance", "formal_ppt_feel", "template_overfit"
        )]
    if style_status == "malformed":
        axes = [{"status": "PASS"}]

    def fake_call(model, api_key, image_path, prompt, timeout):
        assert "#FAFAFA" in prompt and "line art" in prompt
        assert "style_contract_compliance" in prompt and "template_overfit" in prompt
        assert "Preserve the measured light curve." in prompt
        return {"status_code": 200, "json": {"choices": [{"message": {"content": json.dumps({
            "overall_ready_to_ship": True, "quality_review_axes": axes,
        })}}]}}

    monkeypatch.setattr(reviewer, "call_dashscope", fake_call)

    def run_review(cmd, **kwargs):
        assert Path(cmd[1]) == SCRIPT
        monkeypatch.setattr(sys, "argv", cmd[1:])
        assert reviewer.main() == (2 if style_status == "malformed" else 0)

    monkeypatch.setattr(repair, "run", run_review)
    review = repair.semantic_review(tmp_path, "01", image)
    assert review["status"] == ("blocked" if style_status == "malformed" else "completed")
    expected_ready = None if style_status == "malformed" else style_status == "PASS"
    assert review["overall_ready_to_ship"] is expected_ready
    assert review["style_contract_path"] == str(tmp_path / "visual_style_contract.json")


def test_scientific_image_review_without_style_contract(tmp_path, monkeypatch):
    reviewer = load_script(SCRIPT)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key-not-sent")
    image = tmp_path / "figure.png"
    image.write_bytes(minimal_png())
    out = tmp_path / "review.json"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--image", str(image), "--out", str(out)])

    def fake_call(model, api_key, image_path, prompt, timeout):
        assert "style_contract_compliance" not in prompt
        return {"status_code": 200, "json": {"choices": [{"message": {"content": json.dumps({
            "overall_ready_to_ship": True, "recommended_next_action": "deliver",
        })}}]}}

    monkeypatch.setattr(reviewer, "call_dashscope", fake_call)
    assert reviewer.main() == 0
    review = json.loads(out.read_text(encoding="utf-8"))
    assert review["status"] == "completed" and review["overall_ready_to_ship"] is True
    assert "style_contract_path" not in review


@pytest.mark.parametrize("contract_text", [None, "{", "{}"])
def test_missing_or_invalid_style_contract_blocks_before_api(tmp_path, monkeypatch, contract_text):
    reviewer = load_script(SCRIPT)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key-not-sent")
    image = tmp_path / "slide.png"
    image.write_bytes(minimal_png())
    contract = tmp_path / "style.json"
    if contract_text is not None:
        contract.write_text(contract_text, encoding="utf-8")
    out = tmp_path / "review.json"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--image", str(image), "--style-contract", str(contract), "--out", str(out)])

    def unexpected_call(*args):
        pytest.fail("Invalid style contracts must not reach the provider")

    monkeypatch.setattr(reviewer, "call_dashscope", unexpected_call)
    assert reviewer.main() == 2
    review = json.loads(out.read_text(encoding="utf-8"))
    assert review["status"] == "blocked"
    assert review["overall_ready_to_ship"] is None
