import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_scientific_image_smoke.py"


def test_no_token_input_json_writes_structured_run_contract(tmp_path):
    params = tmp_path / "params.json"
    out_dir = tmp_path / "runs"
    params.write_text(
        json.dumps(
            {
                "prompt": "画一个横向三步科研流程图：输入、推理、验证",
                "caption": "方法流程",
                "intent": "展示模型从输入到验证的关键步骤",
                "figure_spec": {
                    "figure_kind": "workflow",
                    "required_labels": ["输入", "推理", "验证"],
                    "layout_brief": "横向三步流程，每步一个卡片，箭头单向连接",
                    "style_brief": "白底、蓝绿色学术配色、留白充足",
                    "reference_role": "preserve_structure",
                    "allowed_changes": ["替换第三步标签"],
                    "disallowed_changes": ["新增步骤"],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.pop("GIIISP_AUTH_TOKEN", None)
    env.pop("SITIANAI_IMAGE_TOKEN", None)

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--input-json", str(params), "--output-dir", str(out_dir)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
    )

    assert completed.returncode == 2
    run_dirs = list(out_dir.glob("smoke_*"))
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    figure_spec = json.loads((run_dir / "figure_spec.json").read_text(encoding="utf-8"))
    run_input = json.loads((run_dir / "run_input.json").read_text(encoding="utf-8"))
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "figure_manifest.json").read_text(encoding="utf-8"))

    assert figure_spec["required_labels"] == ["输入", "推理", "验证"]
    assert figure_spec["layout_brief"] == "横向三步流程，每步一个卡片，箭头单向连接"
    assert figure_spec["reference_role"] == "preserve_structure"
    assert run_input["figure_spec"] == figure_spec
    assert run_input["token_policy"] == "read from GIIISP_AUTH_TOKEN or SITIANAI_IMAGE_TOKEN environment variable; token is not written to files"
    assert metadata["status"] == "blocked"
    assert metadata["error"] == "missing GIIISP_AUTH_TOKEN or SITIANAI_IMAGE_TOKEN"
    assert manifest["run_id"] == run_dir.name
    assert manifest["task"]["layout_brief"] == "横向三步流程，每步一个卡片，箭头单向连接"
    assert manifest["task"]["reference_role"] == "preserve_structure"
    assert manifest["task"]["figure_spec"] == figure_spec
    assert manifest["artifacts"]["figure_spec"] == str(run_dir / "figure_spec.json")


def test_missing_reference_image_still_writes_auditable_run_contract(tmp_path):
    out_dir = tmp_path / "runs"
    missing_reference = tmp_path / "missing_reference.png"
    env = os.environ.copy()
    env["GIIISP_AUTH_TOKEN"] = "test-token-not-used"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--prompt",
            "基于参考图改成横向三步流程",
            "--reference-image",
            str(missing_reference),
            "--source-run",
            str(tmp_path / "parent_run"),
            "--reference-role",
            "preserve_structure",
            "--output-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
    )

    assert completed.returncode == 2
    run_dir = next(out_dir.glob("smoke_*"))
    assert (run_dir / "run_input.json").exists()
    assert (run_dir / "metadata.json").exists()
    assert (run_dir / "figure_manifest.json").exists()
    assert (run_dir / "request_metadata.json").exists()

    blocker = json.loads((run_dir / "blocker.json").read_text(encoding="utf-8"))
    run_input = json.loads((run_dir / "run_input.json").read_text(encoding="utf-8"))
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "figure_manifest.json").read_text(encoding="utf-8"))

    assert blocker["reason"] == "reference image path does not exist"
    assert run_input["reference_image_exists"] is False
    assert run_input["continue_from"]["source_image_exists"] is False
    assert run_input["continue_from"]["reference_role"] == "preserve_structure"
    assert metadata["status"] == "blocked"
    assert metadata["error"] == "reference image path does not exist"
    assert manifest["status"] == "blocked"
    assert manifest["task"]["reference_role"] == "preserve_structure"
