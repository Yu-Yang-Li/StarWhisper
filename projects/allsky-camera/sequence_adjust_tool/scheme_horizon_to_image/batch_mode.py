# -*- coding: utf-8 -*-
"""batch模式：对连续的全天相机序列图片进行批量统计评估。
可独立运行：
    python batch_mode.py --position-name position_5 --conda-env cv --device auto
也可被 run_pipeline.py 通过 conda run 调用。
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

# 确保当前目录在 sys.path 中，以便直接 import replan
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import numpy as np
import replan

try:
    import psutil
except ImportError:
    psutil = None

# Windows终端默认为GBK，重配置为UTF-8避免中文乱码
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

_PIPELINE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_RAW_IMG_DIR = _PIPELINE_DIR / "test_seq_img" / "raw_img"
DEFAULT_LABEL_MASK_DIR = _PIPELINE_DIR / "test_seq_img" / "raw_label_voc" / "SegmentationClassPNG"
BATCH_OUTPUT_ROOT = _PIPELINE_DIR / "output_batch"

BEIJING_TZ = timezone(timedelta(hours=8))
DEFAULT_STATION = {
    "name": "XingLong",
    "lat": 40.393,
    "lon": 117.575,
}


def progress(message: str) -> None:
    print(f"[batch] {message}", flush=True)


def _collect_cpu_snapshot() -> dict:
    """收集当前运行环境的 CPU 信息，尽量使用标准库，若有 psutil 则补充更多字段。"""
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
    }
    if psutil is not None:
        try:
            info["physical_cpu_count"] = psutil.cpu_count(logical=False)
            info["cpu_percent_snapshot"] = psutil.cpu_percent(interval=0.1)
            freq = psutil.cpu_freq()
            if freq is not None:
                info["cpu_frequency_mhz"] = {
                    "current": freq.current,
                    "min": freq.min,
                    "max": freq.max,
                }
        except Exception:
            info["psutil_available"] = True
            info["psutil_snapshot_error"] = True
        else:
            info["psutil_available"] = True
    else:
        info["psutil_available"] = False
    return info


def _format_schedule_time(dt: datetime) -> str:
    """格式化为与 Schedule.json 一致的 UTC 时间字符串。"""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _align_slot_to_batch_utc(slot_key: str, batch_anchor_utc: datetime) -> datetime:
    """
    将 Schedule 中固定年份的槽位时间，对齐到本次批处理所在观测夜。
    这里复用 replan.py 的“保留日期、替换时分秒”思路，保证 batch 与 single 一致。
    """
    slot_utc = replan.parse_schedule_time(slot_key)
    return replan.replace_datetime_time(batch_anchor_utc, slot_utc)


# ──── 掩码推理 ────

def _build_masks_for_raw_images(
    conda_env: str,
    device: str,
    raw_img_dir: Path,
    model_mask_dir: Path,
    save_overlay: bool,
    model_overlay_dir: Path | None,
) -> dict[Path, Path]:
    """对raw_img/下所有jpg推理生成模型掩码，返回{原图Path: 掩码Path}映射"""
    mask_infer_script = _PIPELINE_DIR / "Pic2mask" / "full_image_infer" / "infer_full_image.py"
    weights_path = _PIPELINE_DIR / "Pic2mask" / "deeplabv3_best_loss_3_16.pth"
    model_mask_dir.mkdir(parents=True, exist_ok=True)
    if save_overlay and model_overlay_dir is not None:
        model_overlay_dir.mkdir(parents=True, exist_ok=True)

    result: dict[Path, Path] = {}
    raw_images = sorted(raw_img_dir.glob("*.jpg"), key=lambda p: p.stem)
    progress(f"共{len(raw_images)}张原始照片，开始推理掩码...")
    for img in raw_images:
        mask_out = model_mask_dir / f"{img.stem}.png"
        overlay_out = (
            model_overlay_dir / f"{img.stem}_overlay.png"
            if save_overlay and model_overlay_dir is not None
            else None
        )
        if mask_out.is_file():
            progress(f"  掩码已存在: {mask_out}")
        else:
            cmd = [
                "conda", "run", "-n", conda_env, "python",
                str(mask_infer_script),
                "--image", str(img),
                "--output", str(mask_out),
                "--weights", str(weights_path),
                "--device", device,
            ]
            if overlay_out is not None:
                cmd.extend(["--save-overlay", "--overlay-output", str(overlay_out)])
            subprocess.run(cmd, check=True)
        result[img] = mask_out
    progress("所有掩码推理完成")
    return result


# ──── 核心判断与分类 ────

def _check_target_on_mask(
    target: dict,
    interval_time_utc: str,
    mask: np.ndarray,
    camera_model,
    replan,
) -> tuple[bool, object]:
    """用指定掩码对目标过三道筛子（掩码像素+高度角≥30°+月距），返回(全通过, 地平点)"""
    lat = DEFAULT_STATION["lat"]
    lon = DEFAULT_STATION["lon"]
    d_moon = 15.0

    point = replan.target_altaz(target, interval_time_utc, lat, lon)
    if point.alt < 0:
        return False, point
    xy = replan.altaz_to_pixel_fast(point.az, point.alt, camera_model, image_shape=mask.shape)
    if xy is None:
        return False, point
    x, y = xy
    h, w = mask.shape[:2]
    if x < 0 or y < 0 or x >= w or y >= h:
        return False, point
    if not bool(mask[y, x]):
        return False, point
    if point.alt < 30.0:
        return False, point
    if not replan.target_observable(
        interval_time_utc, lat, lon,
        float(target["ra"]), float(target["dec"]), 0.0, d_moon,
    ):
        return False, point
    return True, point


def _classify_target(
    target: dict,
    slot_utc: str,
    masks_before: list,
    camera_model,
    replan,
) -> str:
    """根据目标在前一张最近标注掩码上的表现分类：observable / unobservable。"""
    def _check_list(mask_list, slot_time):
        for _dt, mask in mask_list:
            ok, _ = _check_target_on_mask(target, slot_time, mask, camera_model, replan)
            if ok:
                return True
        return False

    before_ok = _check_list(masks_before, slot_utc) if masks_before else False
    return "observable" if before_ok else "unobservable"


# ──── 主入口 ────

def run_batch(
    position_name: str,
    conda_env: str,
    device: str,
    count_mode: str,
    raw_img_dir: str,
    label_mask_dir: str | None,
    save_overlay: bool,
) -> None:
    """
    执行batch模式完整流程：
    1. 加载序列照片和标签掩码
    2. DeepLabV3推理模型掩码
    3. 匹配最近Schedule
    4. 对窗口内每个目标做原始序列统计 + 我方方法统计
    5. 输出batch_summary_report.json
    """
    # ── 初始化astropy离线数据 ──
    replan._configure_astropy_offline_data()
    run_started_at = datetime.now(BEIJING_TZ)
    perf_start = time.perf_counter()
    cpu_time_start = time.process_time()
    cpu_snapshot_start = _collect_cpu_snapshot()

    parameters_path = _PIPELINE_DIR / "sequence_adjust_tool" / "parameters.json"
    batch_dir = _PIPELINE_DIR / "sequence_adjust_tool" / "batch_0116_output"
    csv_path = _PIPELINE_DIR / "sequence_adjust_tool" / "50_select_bright.csv"
    raw_img_path = Path(raw_img_dir).expanduser().resolve()
    label_mask_path = (
        Path(label_mask_dir).expanduser().resolve()
        if label_mask_dir
        else None
    )

    if not raw_img_path.is_dir():
        raise FileNotFoundError(f"未找到原始鱼眼图像目录: {raw_img_path}")
    if count_mode == "label_count":
        if label_mask_path is None:
            raise ValueError("label_count 模式必须提供标签鱼眼图像目录")
        if not label_mask_path.is_dir():
            raise FileNotFoundError(f"未找到标签鱼眼图像目录: {label_mask_path}")

    camera_model = replan.load_camera_model(parameters_path, position_name)
    csv_targets = replan.load_csv_targets(csv_path)
    progress(f"CSV亮星库加载完成，共{len(csv_targets)}个候选目标")
    progress(f"统计计数口径: {count_mode}")
    progress(f"原始鱼眼图像目录: {raw_img_path}")
    if label_mask_path is not None:
        progress(f"标签鱼眼图像目录: {label_mask_path}")

    # ── Step 1: 加载原始照片 ──
    raw_images = sorted(raw_img_path.glob("*.jpg"), key=lambda p: p.stem)
    if not raw_images:
        raise FileNotFoundError(f"未在{raw_img_path}下找到原始照片")
    progress(f"找到{len(raw_images)}张原始照片")

    # ── Step 2: 解析时间并准备本次batch输出目录 ──
    first_photo_dt_beijing = replan.parse_mask_time_beijing(raw_images[0])
    last_photo_dt_beijing = replan.parse_mask_time_beijing(raw_images[-1])
    first_photo_dt_utc = first_photo_dt_beijing.astimezone(timezone(timedelta(0)))
    last_photo_dt_utc = last_photo_dt_beijing.astimezone(timezone(timedelta(0)))
    date_range_str = (
        f"{first_photo_dt_beijing.strftime('%Y%m%d')}"
        f"_{last_photo_dt_beijing.strftime('%Y%m%d')}"
    )
    batch_output_dir = BATCH_OUTPUT_ROOT / date_range_str
    model_mask_dir = batch_output_dir / "model_masks"
    model_overlay_dir = batch_output_dir / "model_overlays"
    batch_output_dir.mkdir(parents=True, exist_ok=True)
    progress(f"时间窗口（北京）: {first_photo_dt_beijing.isoformat()} ~ {last_photo_dt_beijing.isoformat()}")
    progress(f"时间窗口（UTC）: {first_photo_dt_utc.isoformat()} ~ {last_photo_dt_utc.isoformat()}")

    # ── Step 3: 推理模型掩码 ──
    model_mask_map = _build_masks_for_raw_images(
        conda_env,
        device,
        raw_img_path,
        model_mask_dir,
        save_overlay,
        model_overlay_dir,
    )

    # ── Step 4: 匹配Schedule ──
    schedule_path = replan.nearest_schedule_path(batch_dir, first_photo_dt_beijing)
    progress(f"匹配计划文件: {schedule_path}")
    schedule = replan.load_schedule(schedule_path)

    # ── Step 5: 按与 single 模式一致的逻辑筛选窗口内槽位 ──
    # 先把 Schedule 的时分秒对齐到本次批处理所在观测夜，再按真实批次时间窗口截取。
    window_slots: list[tuple[str, dict, datetime]] = []
    for slot_key, slot_val in schedule.items():
        aligned_slot_utc = _align_slot_to_batch_utc(slot_key, first_photo_dt_utc)
        if first_photo_dt_utc <= aligned_slot_utc <= last_photo_dt_utc:
            window_slots.append((slot_key, slot_val, aligned_slot_utc))

    selection_mode = "strict_window"
    if not window_slots:
        # 若严格窗口没有命中，再退回到与 replan.py 相同的“最近槽位索引”逻辑，
        # 至少输出一份报告，方便定位是时间窗问题还是后续统计问题。
        start_idx = replan.closest_schedule_index(schedule, first_photo_dt_utc)
        end_idx = replan.closest_schedule_index(schedule, last_photo_dt_utc)
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx
        keys = list(schedule.keys())
        for idx in range(start_idx, end_idx + 1):
            slot_key = keys[idx]
            window_slots.append(
                (slot_key, schedule[slot_key], _align_slot_to_batch_utc(slot_key, first_photo_dt_utc))
            )
        selection_mode = "closest_index_fallback"

    window_slots.sort(key=lambda x: x[2])
    progress(f"槽位筛选方式: {selection_mode}")
    progress(f"时间窗口内共{len(window_slots)}个观测槽位")
    if window_slots:
        progress(
            "窗口首槽位: "
            f"{window_slots[0][0]} -> {_format_schedule_time(window_slots[0][2])}"
        )
        progress(
            "窗口末槽位: "
            f"{window_slots[-1][0]} -> {_format_schedule_time(window_slots[-1][2])}"
        )
    else:
        progress("时间窗口内仍未命中任何观测槽位，将输出空统计报告")

    # ── 构建按时间排序的标签/模型掩码列表（均以北京时间为键） ──
    label_sorted: list[tuple[datetime, np.ndarray]] = []
    model_sorted: list[tuple[datetime, np.ndarray]] = []
    for img in raw_images:
        dt = replan.parse_mask_time_beijing(img)
        model_path = model_mask_map[img]
        model_sorted.append((dt, replan.load_binary_mask(model_path)))
        if count_mode == "label_count":
            assert label_mask_path is not None
            label_path = label_mask_path / f"{img.stem}.png"
            if not label_path.is_file():
                raise FileNotFoundError(
                    f"label_count 模式缺少标签掩码: {label_path}"
                )
            label_sorted.append((dt, replan.load_binary_mask(label_path)))

    # ── Step 6: 逐目标统计 ──
    stats_original = {"observable": 0, "unobservable": 0}
    stats_our = {"observable": 0, "unobservable": 0, "replaced": 0}
    detail_log: list[dict] = []
    used_csv_names: set[str] = set()

    for slot_idx, (slot_key, slot_val, aligned_slot_utc) in enumerate(window_slots):
        target = slot_val.get("target")
        if not target:
            continue

        slot_utc_str = _format_schedule_time(aligned_slot_utc)
        slot_bjt = aligned_slot_utc.astimezone(BEIJING_TZ)

        # 找到槽位时间前最近的一张标签/模型掩码
        all_before = [(dt, mask) for dt, mask in label_sorted if dt <= slot_bjt]
        label_before = [all_before[-1]] if all_before else []
        model_before_list = [(dt, mask) for dt, mask in model_sorted if dt <= slot_bjt]
        model_before = [model_before_list[-1]] if model_before_list else []
        count_before = label_before if count_mode == "label_count" else model_before

        # a) 原始序列统计：按 count_mode 选择标签掩码或模型掩码
        orig_class = _classify_target(target, slot_utc_str, count_before,
                                      camera_model, replan)
        stats_original[orig_class] += 1

        # b) 我方方法统计
        model_before_mask = model_before_list[-1][1] if model_before_list else None

        model_before_available = model_before_mask is not None
        if model_before_mask is not None:
            model_ok, _ = _check_target_on_mask(target, slot_utc_str, model_before_mask,
                                                camera_model, replan)
        else:
            model_ok = None

        final_target = target
        replaced_name = None
        if model_ok is False:
            # 从CSV亮星库寻找替代品
            picked = None
            for csv_t in csv_targets:
                if csv_t["objname"] in used_csv_names:
                    continue
                csv_ok, _ = _check_target_on_mask(csv_t, slot_utc_str, model_before_mask,
                                                  camera_model, replan)
                if csv_ok:
                    picked = csv_t
                    break
            if picked is not None:
                used_csv_names.add(picked["objname"])
                final_target = picked
                replaced_name = picked["objname"]
                stats_our["replaced"] += 1

        # 对最终目标也按 count_mode 选择标签掩码或模型掩码分类
        our_class = _classify_target(final_target, slot_utc_str, count_before,
                                     camera_model, replan)
        stats_our[our_class] += 1

        detail_log.append({
            "slot_utc_original": slot_key,
            "slot_utc_aligned": slot_utc_str,
            "slot_beijing_aligned": slot_bjt.isoformat(),
            "original_target": target.get("objname", ""),
            "original_classification": orig_class,
            "our_target": final_target.get("objname", ""),
            "our_classification": our_class,
            "count_mode": count_mode,
            "model_before_available": model_before_available,
            "replaced": replaced_name is not None,
            "replacement_name": replaced_name,
        })

        if (slot_idx + 1) % max(1, len(window_slots) // 10) == 0:
            progress(f"统计进度: {slot_idx + 1}/{len(window_slots)}")

    progress(f"统计完成: 原始序列({stats_original}), 我方方法({stats_our})")

    # ── Step 7: 输出报告 ──
    run_finished_at = datetime.now(BEIJING_TZ)
    elapsed_seconds = time.perf_counter() - perf_start
    process_cpu_time_seconds = time.process_time() - cpu_time_start
    cpu_snapshot_end = _collect_cpu_snapshot()

    summary = {
        "mode": "batch",
        "run_statistics": {
            "started_at_beijing": run_started_at.isoformat(),
            "finished_at_beijing": run_finished_at.isoformat(),
            "elapsed_seconds": elapsed_seconds,
            "process_cpu_time_seconds": process_cpu_time_seconds,
            "cpu": {
                "start": cpu_snapshot_start,
                "end": cpu_snapshot_end,
            },
        },
        "time_window": {
            "start_beijing": first_photo_dt_beijing.isoformat(),
            "end_beijing": last_photo_dt_beijing.isoformat(),
            "start_utc": first_photo_dt_utc.isoformat(),
            "end_utc": last_photo_dt_utc.isoformat(),
        },
        "photo_count": len(raw_images),
        "matched_schedule": str(schedule_path),
        "position_name": position_name,
        "count_mode": count_mode,
        "save_overlay": save_overlay,
        "model_mask_dir": str(model_mask_dir),
        "model_overlay_dir": str(model_overlay_dir) if save_overlay else None,
        "slot_selection_mode": selection_mode,
        "window_slot_count": len(window_slots),
        "total_targets_in_window": len(detail_log),
        "original_sequence": stats_original,
        "our_method": {
            "observable": stats_our["observable"],
            "unobservable": stats_our["unobservable"],
            "replaced_count": stats_our["replaced"],
        },
        "detail_log": detail_log,
    }
    report_path = batch_output_dir / "batch_summary_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    progress(f"batch汇总报告输出: {report_path}")
    progress("batch模式完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="batch模式批量统计评估")
    parser.add_argument("--position-name", required=True, help="标定参数组名，如position_5")
    parser.add_argument("--conda-env", default="cv", help="conda环境名（仅文档用途）")
    parser.add_argument("--device", default="auto", help="掩码推理设备")
    parser.add_argument(
        "--save-overlay",
        action="store_true",
        help="推理模型掩码时额外保存 overlay 到 output_batch/<日期范围>/model_overlays",
    )
    parser.add_argument(
        "--count-mode",
        default="label_count",
        choices=["label_count", "model_count"],
        help="统计计数口径：label_count=按标签掩码计数，model_count=按模型掩码计数",
    )
    parser.add_argument(
        "--raw-img-dir",
        default=str(DEFAULT_RAW_IMG_DIR),
        help="原始鱼眼图像目录（batch 模式输入）",
    )
    parser.add_argument(
        "--label-mask-dir",
        default=str(DEFAULT_LABEL_MASK_DIR),
        help="标签鱼眼图像目录（仅 label_count 模式需要）",
    )
    args = parser.parse_args()
    run_batch(
        position_name=args.position_name,
        conda_env=args.conda_env,
        device=args.device,
        count_mode=args.count_mode,
        raw_img_dir=args.raw_img_dir,
        label_mask_dir=args.label_mask_dir if args.count_mode == "label_count" else None,
        save_overlay=args.save_overlay,
    )
