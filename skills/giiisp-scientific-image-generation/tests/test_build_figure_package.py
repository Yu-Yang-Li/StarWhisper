import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_SCRIPT = ROOT / "scripts" / "build_figure_package.py"


def test_package_carries_figure_spec_fields(tmp_path):
    run_dir = tmp_path / "smoke_001"
    run_dir.mkdir()
    figure_spec = {
        "schema": "giiisp_figure_spec_v1",
        "figure_kind": "workflow",
        "caption": "方法流程",
        "communicative_intent": "展示模型从输入到验证的关键步骤",
        "required_labels": ["输入", "推理", "验证"],
        "layout_brief": "横向三步流程",
        "style_brief": "蓝绿色学术配色",
        "reference_role": "preserve_structure",
    }
    (run_dir / "figure_spec.json").write_text(json.dumps(figure_spec, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "schema": "giiisp_figure_manifest_v1",
        "run_id": "smoke_001",
        "figure_id": "smoke_001",
        "run_dir": str(run_dir),
        "status": "blocked",
        "task": {
            "caption": "方法流程",
            "figure_kind": "workflow",
            "intent": "展示模型从输入到验证的关键步骤",
            "required_labels": ["输入", "推理", "验证"],
            "layout_brief": "横向三步流程",
            "reference_role": "preserve_structure",
            "figure_spec": figure_spec,
        },
        "artifacts": {
            "figure_spec": str(run_dir / "figure_spec.json"),
            "semantic_review": str(run_dir / "semantic_review.json"),
        },
        "output": {},
        "quality": {
            "semantic_review_summary": {
                "status": "completed",
                "model": "qwen3.7-plus",
                "overall_ready_to_ship": True,
                "recommended_next_action": "deliver",
            }
        },
        "blocker": {"reason": "missing GIIISP_AUTH_TOKEN"},
    }
    (run_dir / "figure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    package_dir = tmp_path / "package"

    subprocess.run(
        [
            sys.executable,
            str(PACKAGE_SCRIPT),
            "--input",
            str(run_dir),
            "--package-dir",
            str(package_dir),
            "--title",
            "测试图包",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    plan = json.loads((package_dir / "package_plan.json").read_text(encoding="utf-8"))
    report = json.loads((package_dir / "figure_package.json").read_text(encoding="utf-8"))
    assert plan["items"][0]["layout_brief"] == "横向三步流程"
    assert plan["items"][0]["reference_role"] == "preserve_structure"
    assert plan["items"][0]["figure_spec_path"] == str(run_dir / "figure_spec.json")
    assert plan["items"][0]["semantic_review_path"] == str(run_dir / "semantic_review.json")
    assert report["items"][0]["figure_spec_path"] == str(run_dir / "figure_spec.json")
    assert report["items"][0]["semantic_review_model"] == "qwen3.7-plus"
    assert report["items"][0]["overall_ready_to_ship"] is True
