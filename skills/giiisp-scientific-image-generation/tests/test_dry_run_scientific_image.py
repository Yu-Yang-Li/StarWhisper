import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "dry_run_scientific_image.py"


def run_dry(*extra):
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--prompt", "画一个四步科研流程图：问题定义、数据整理、模型生成、结果检查", *extra],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(completed.stdout)


def test_basic_dry_run_contract():
    payload = run_dry("--caption", "科研流程图", "--intent", "展示从问题到结果的流程")
    assert payload["method"] == "POST"
    assert payload["url"] == "http://images.sitianai.com/api/generate-async"
    assert payload["body"]["responseModalities"] == ["IMAGE", "TEXT"]
    assert "no image generation request was sent" in payload["safety"]
    assert payload["run_input_preview"]["caption"] == "科研流程图"
    assert payload["figure_spec_preview"]["schema"] == "giiisp_figure_spec_v1"


def test_reference_image_dry_run_redacts_base64(tmp_path):
    image = tmp_path / "ref.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    payload = run_dry("--reference-image", str(image), "--source-run", str(tmp_path))
    assert payload["body"]["imageBase64"] == "<redacted reference image base64>"
    assert payload["edit_metadata"]["reference_image_exists"] is True
    assert payload["edit_metadata"]["reference_image_mime_type"] == "image/png"


def test_input_json_populates_structured_figure_spec(tmp_path):
    params = tmp_path / "params.json"
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
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--input-json", str(params)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    payload = json.loads(completed.stdout)
    spec = payload["figure_spec_preview"]
    assert payload["body"]["prompt"] == "画一个横向三步科研流程图：输入、推理、验证"
    assert spec["required_labels"] == ["输入", "推理", "验证"]
    assert spec["layout_brief"] == "横向三步流程，每步一个卡片，箭头单向连接"
    assert spec["reference_role"] == "preserve_structure"
    assert spec["allowed_changes"] == ["替换第三步标签"]
    assert spec["disallowed_changes"] == ["新增步骤"]


def test_input_json_without_prompt_fails_fast(tmp_path):
    params = tmp_path / "params.json"
    params.write_text(json.dumps({"caption": "缺少 prompt"}, ensure_ascii=False), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--input-json", str(params)],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode != 0
    assert "--prompt is required unless --input-json provides it" in completed.stderr
