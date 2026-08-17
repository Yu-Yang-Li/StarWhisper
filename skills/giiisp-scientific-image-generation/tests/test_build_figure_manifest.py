import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_figure_manifest.py"


def test_manifest_recovers_figure_spec_without_request_file(tmp_path):
    run_dir = tmp_path / "smoke_001"
    run_dir.mkdir()
    figure_spec = {
        "schema": "giiisp_figure_spec_v1",
        "figure_kind": "workflow",
        "caption": "方法流程",
        "communicative_intent": "展示模型从输入到验证的关键步骤",
        "required_labels": ["输入", "推理", "验证"],
        "layout_brief": "横向三步流程",
        "reference_role": "preserve_structure",
    }
    run_input = {
        "schema": "giiisp_run_input_v1",
        "run_id": "smoke_001",
        "figure_kind": "workflow",
        "caption": "方法流程",
        "communicative_intent": "展示模型从输入到验证的关键步骤",
        "required_labels": ["输入", "推理", "验证"],
        "layout_brief": "横向三步流程",
        "reference_role": "preserve_structure",
        "figure_spec": figure_spec,
        "request_body_summary": {
            "prompt": "画一个横向三步科研流程图",
            "negativePrompt": "水印，模糊文字",
            "aspectRatio": "1:1",
            "imageSize": "1K",
            "numberOfImages": 1,
            "responseModalities": ["IMAGE", "TEXT"],
            "outputMimeType": "image/png",
        },
    }
    check = {
        "mime_type": "image/png",
        "width": 1024,
        "height": 1024,
        "machine_check": {"status": "passed"},
        "quality_review_axes": [{"name": "content_accuracy", "score": None}],
    }
    semantic_review = {
        "schema": "giiisp_semantic_review_v1",
        "provider": "dashscope",
        "model": "qwen3.7-plus",
        "status": "completed",
        "overall_ready_to_ship": True,
        "recommended_next_action": "deliver",
        "missing_required_labels": [],
        "forbidden_labels_seen": [],
    }
    (run_dir / "figure_spec.json").write_text(json.dumps(figure_spec, ensure_ascii=False), encoding="utf-8")
    (run_dir / "run_input.json").write_text(json.dumps(run_input, ensure_ascii=False), encoding="utf-8")
    (run_dir / "check.json").write_text(json.dumps(check, ensure_ascii=False), encoding="utf-8")
    (run_dir / "semantic_review.json").write_text(json.dumps(semantic_review, ensure_ascii=False), encoding="utf-8")

    subprocess.run(
        [sys.executable, str(SCRIPT), "--run-dir", str(run_dir)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    manifest = json.loads((run_dir / "figure_manifest.json").read_text(encoding="utf-8"))
    assert manifest["task"]["prompt"] == "画一个横向三步科研流程图"
    assert manifest["task"]["layout_brief"] == "横向三步流程"
    assert manifest["task"]["reference_role"] == "preserve_structure"
    assert manifest["task"]["figure_spec"] == figure_spec
    assert manifest["artifacts"]["figure_spec"] == str(run_dir / "figure_spec.json")
    assert manifest["artifacts"]["semantic_review"] == str(run_dir / "semantic_review.json")
    assert manifest["quality"]["machine_check_summary"] == {"status": "passed"}
    assert manifest["quality"]["semantic_review_summary"]["model"] == "qwen3.7-plus"
    assert manifest["quality"]["semantic_review_summary"]["overall_ready_to_ship"] is True
