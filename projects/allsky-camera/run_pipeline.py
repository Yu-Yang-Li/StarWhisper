# -*- coding: utf-8 -*-
"""单入口pipeline：原始全天相机照片 -> 整图掩码 -> 重排输出。
运行模式：
    python run_pipeline.py --image x.jpg                     # 单张
    python run_pipeline.py --mode batch --position-name p5   # 批量
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Windows终端默认为GBK，重配置为UTF-8避免中文乱码
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


PIPELINE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PIPELINE_DIR / "configs" / "default.json"


def progress(message: str) -> None:
    print(f"[pipeline] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="单入口pipeline：原始全天相机照片->整图掩码->重排输出。"
    )
    parser.add_argument(
        "--mode",
        default="single",
        choices=["single", "batch"],
        help="运行模式：single=单张图片处理（默认），batch=批量序列处理",
    )
    parser.add_argument(
        "--image", default=None,
        help="输入原始全天相机照片路径（single模式必填）。",
    )
    parser.add_argument(
        "--output-root",
        default=str(PIPELINE_DIR / "output"),
        help="pipeline输出根目录（single模式）。",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="pipeline默认配置JSON。",
    )
    parser.add_argument(
        "--position-name",
        default=None,
        help="覆盖默认参数组名，如position_2。",
    )
    parser.add_argument(
        "--save-overlay",
        action="store_true",
        help="在掩码推理阶段额外保存overlay对比图。",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="掩码推理设备。",
    )
    parser.add_argument(
        "--count-mode",
        default="label_count",
        choices=["label_count", "model_count"],
        help="batch模式统计口径：label_count=按标签掩码计数，model_count=按模型掩码计数。",
    )
    parser.add_argument(
        "--raw-img-dir",
        default=str(PIPELINE_DIR / "test_seq_img" / "raw_img"),
        help="batch模式原始鱼眼图像目录。",
    )
    parser.add_argument(
        "--label-mask-dir",
        default=str(PIPELINE_DIR / "test_seq_img" / "raw_label_voc" / "SegmentationClassPNG"),
        help="batch模式标签鱼眼图像目录；仅 label_count 模式需要。",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_command(cmd: list[str]) -> None:
    progress("执行命令: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def find_single_replan_dir(root: Path) -> Path:
    candidates = sorted(
        [path for path in root.glob("*_mask") if path.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"未在{root}下找到任何重排输出目录")
    return candidates[0]


# ──── single模式 ────

def run_single(args: argparse.Namespace) -> None:
    image_path = Path(args.image).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()

    if not image_path.is_file():
        raise FileNotFoundError(f"未找到输入图像: {image_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"未找到配置文件: {config_path}")

    config = load_json(config_path)
    position_name = args.position_name or config["position_name"]
    conda_env = (
        config.get("conda_env")
        or config.get("skyseg_env")
        or config.get("star_env")
    )
    if not conda_env:
        raise ValueError("配置中缺少conda_env")
    mask_infer_script = PIPELINE_DIR / "Pic2mask" / "full_image_infer" / "infer_full_image.py"
    weights_path = PIPELINE_DIR / "Pic2mask" / "deeplabv3_best_loss_3_16.pth"
    replan_script = PIPELINE_DIR / "sequence_adjust_tool" / "scheme_horizon_to_image" / "replan.py"
    parameters_path = PIPELINE_DIR / "sequence_adjust_tool" / "parameters.json"
    observe_config_path = PIPELINE_DIR / "sequence_adjust_tool" / "observe_config.json"
    batch_dir = PIPELINE_DIR / "sequence_adjust_tool" / "batch_0116_output"

    run_root = output_root / image_path.stem
    input_dir = run_root / "input"
    mask_dir = run_root / "mask"
    replan_root = run_root / "replan"
    input_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    replan_root.mkdir(parents=True, exist_ok=True)

    copied_input = input_dir / image_path.name
    if copied_input.resolve() != image_path:
        copied_input.write_bytes(image_path.read_bytes())

    mask_output = mask_dir / f"{image_path.stem}.png"
    overlay_output = mask_dir / f"{image_path.stem}_overlay.png"

    progress("阶段1/2：生成整图掩码")
    mask_cmd = [
        "conda", "run", "-n", str(conda_env), "python",
        str(mask_infer_script),
        "--image", str(image_path),
        "--output", str(mask_output),
        "--weights", str(weights_path),
        "--device", args.device,
    ]
    if args.save_overlay:
        mask_cmd.extend(["--save-overlay", "--overlay-output", str(overlay_output)])
    run_command(mask_cmd)

    progress("阶段2/2：执行观测序列重排")
    replan_cmd = [
        "conda", "run", "-n", str(conda_env), "python",
        str(replan_script),
        "--mask-path", str(mask_output),
        "--output-root", str(replan_root),
        "--batch-dir", str(batch_dir),
        "--config", str(observe_config_path),
        "--parameters", str(parameters_path),
        "--position-name", position_name,
    ]
    run_command(replan_cmd)

    final_replan_dir = find_single_replan_dir(replan_root)
    scenario_path = final_replan_dir / "scenario.json"
    replanned_schedule_path = final_replan_dir / "replanned_schedule.json"
    deferred_targets_path = final_replan_dir / "deferred_targets.json"
    nina_path = final_replan_dir / "replanned_sequence.ninaTargetSet"
    sky_plot_path = final_replan_dir / "sky_replan_plot.png"

    report = {
        "input_image": str(image_path),
        "copied_input": str(copied_input),
        "mask_output": str(mask_output),
        "overlay_output": str(overlay_output) if args.save_overlay else None,
        "position_name": position_name,
        "replan_output_dir": str(final_replan_dir),
        "scenario_json": str(scenario_path),
        "replanned_schedule_json": str(replanned_schedule_path),
        "deferred_targets_json": str(deferred_targets_path),
        "nina_targetset": str(nina_path),
        "sky_replan_plot": str(sky_plot_path),
    }
    report_path = run_root / "pipeline_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    progress("pipeline完成")
    progress(f"报告文件: {report_path}")
    progress(f"重排目录: {final_replan_dir}")


# ──── batch模式 ────

def run_batch(args: argparse.Namespace) -> None:
    config = load_json(Path(args.config).expanduser().resolve())
    position_name = args.position_name or config["position_name"]
    conda_env = (
        config.get("conda_env")
        or config.get("skyseg_env")
        or config.get("star_env")
    )
    if not conda_env:
        raise ValueError("配置中缺少conda_env")

    batch_mode_script = (
        PIPELINE_DIR / "sequence_adjust_tool" / "scheme_horizon_to_image" / "batch_mode.py"
    )
    cmd = [
        "conda", "run", "-n", conda_env, "python",
        str(batch_mode_script),
        "--position-name", position_name,
        "--conda-env", conda_env,
        "--device", args.device,
        "--count-mode", args.count_mode,
        "--raw-img-dir", args.raw_img_dir,
    ]
    if args.save_overlay:
        cmd.append("--save-overlay")
    if args.count_mode == "label_count":
        cmd.extend(["--label-mask-dir", args.label_mask_dir])
    progress("执行命令: " + " ".join(cmd))
    subprocess.run(cmd, check=True)

    progress("batch模式完成")


# ──── 主入口 ────

def main() -> None:
    args = parse_args()
    if args.mode == "batch":
        run_batch(args)
    else:
        if not args.image:
            raise ValueError("single模式必须提供--image参数")
        run_single(args)


if __name__ == "__main__":
    main()
