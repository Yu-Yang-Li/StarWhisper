#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一推理耗时基准：新划分实验 + 特征提取 + 端到端总耗时。

对比原则:
  - varlen/1121 主对比: 3–30 部署场景，手工特征含 LS 特征提取 + 推理
  - 50obs 预训练: 中间阶段，单独列出，不与 varlen 主对比混排
  - E2E/RNN: 原始时序推理（离线 prepare_e2e 预处理，不计入特征提取列）

用法（推荐）:
  cd /root/shared-nvme

  # 部署性能（默认）：仅 GPU 上测 PyTorch；XGB/LS 在 CPU。约 20–40 分钟（全量 test）
  conda run -n astro_classifier python train_models/benchmark_inference_time.py

  # 快速试跑
  conda run -n astro_classifier python train_models/benchmark_inference_time.py \\
      --max-samples 10000 --feature-samples 200

  # 论文用：同一 PyTorch 模型在 CPU 与 GPU 各跑一遍（耗时长，可能 1–3 小时）
  conda run -n astro_classifier python train_models/benchmark_inference_time.py --device both

  # 跳过旧路径 legacy 模型
  conda run -n astro_classifier python train_models/benchmark_inference_time.py --skip-legacy

LS 特征提取前提:
  - 特征 CSV 含 file_path 列
  - 原始光变 CSV 在 /root/shared-nvme/train2/... 或 train4/...（与 e2e 预处理相同数据源）
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from xgboost import XGBClassifier

from dl_common import NumpyTensorDataset, TrueVarLenDataset
from experiment_registry import (
    EXPERIMENTS,
    BASE,
    TRAIN_MODELS,
    Experiment,
    _checkpoint_mb,
    collect_metrics_table,
)
from split_utils import load_split_feature_bundle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

BATCH_SIZE = 256
DEVICE_CPU = torch.device("cpu")
DEVICE_GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HANDCRAFTED_GROUPS = {"手工特征"}

USAGE_EPILOG = """
示例:
  %(prog)s                              # 默认 --device gpu（部署向基准）
  %(prog)s --device gpu --skip-legacy   # 同上，显式跳过 legacy
  %(prog)s --max-samples 10000          # 快速试跑
  %(prog)s --device both                # CPU+GPU 对照
  %(prog)s --report-only                # 从已有 CSV 重生成 Markdown

说明:
  --device gpu   PyTorch 在 GPU 推理；XGBoost / LS 特征提取仍在 CPU（与部署一致）
  --device both  额外在 CPU 上再跑 PyTorch；CPU 侧默认 1000 样本外推（见 --cpu-infer-samples）
  --device cpu   仅 CPU；大模型默认 1000 样本外推，避免全量 test 极慢
  --cpu-infer-samples  CPU PyTorch 推理子集大小（默认 1000，线性外推至全 test）
  --report-only  仅更新 inference_time_benchmark.md，沿用 inference_time_benchmark.csv
"""


def _per_10k(total_sec: float, n: int) -> float:
    return total_sec / max(n, 1) * 10000


def resolve_infer_plan(
    n_full: int,
    max_samples: Optional[int],
    device_label: str,
    cpu_infer_samples: int,
    is_pytorch: bool,
) -> tuple[int, int, str]:
    """返回 (报告用 n_full, 实际运行 n, 外推备注)。"""
    if max_samples is not None:
        n_run = min(max_samples, n_full)
        note = f"子集 n={n_run}" if n_run < n_full else ""
        return n_full, n_run, note
    if is_pytorch and device_label == "cpu" and cpu_infer_samples < n_full:
        return n_full, min(cpu_infer_samples, n_full), f"CPU {cpu_infer_samples}样本外推"
    return n_full, n_full, ""


def scale_infer_result(result: dict, n_report: int, note: str) -> dict:
    n_run = int(result["n"])
    if n_run <= 0 or n_run >= n_report or not note:
        result["speed_note"] = note
        return result
    scale = n_report / n_run
    seconds = float(result["seconds"]) * scale
    scaled = {
        **result,
        "n": n_report,
        "seconds": seconds,
        "per_10k": _per_10k(seconds, n_report),
        "speed_note": note,
    }
    return scaled


def _compare_group(exp: Optional[Experiment]) -> str:
    if exp is None:
        return "其他"
    return "50obs预训练" if exp.pool == "50obs" else "varlen主对比"


def _model_category(exp: Optional[Experiment]) -> str:
    if exp is None:
        return "其他"
    if exp.group in HANDCRAFTED_GROUPS:
        return "手工特征"
    return "原始时序"


def _build_row(
    *,
    exp: Optional[Experiment],
    exp_id: str,
    name: str,
    category: str,
    compare_group: str,
    device: str,
    n: int,
    infer_sec: float,
    feat_sec: float = 0.0,
    status: str = "OK",
    pool: str = "",
    stage: str = "",
    peak_mem_mb: float = 0.0,
    weight_mb: Optional[float] = None,
    speed_note: str = "",
) -> dict:
    total = feat_sec + infer_sec
    status_out = status if not speed_note else f"{status} ({speed_note})"
    return {
        "对比组": compare_group,
        "类别": category,
        "exp_id": exp_id,
        "模型": name,
        "数据池": pool or (exp.pool if exp else ""),
        "阶段": stage or (exp.stage if exp else ""),
        "device": device,
        "n_test": n,
        "特征提取_秒": round(feat_sec, 4),
        "推理_秒": round(infer_sec, 4),
        "总耗时_秒": round(total, 4),
        "每万样本_秒": round(_per_10k(total, n), 4),
        "吞吐量_样本每秒": round(n / max(total, 1e-9), 2),
        "峰值显存_MB": round(peak_mem_mb, 2) if peak_mem_mb > 0 else 0.0,
        "权重_MB": weight_mb,
        "测速备注": speed_note or "",
        "状态": status_out,
    }


def benchmark_xgboost(model_dir: Path, pool: str, max_samples: Optional[int]) -> dict:
    model_path = model_dir / f"{model_dir.name}_best.json"
    if not model_path.is_file():
        model_path = model_dir / "best_model.json"
    if not model_path.is_file():
        raise FileNotFoundError(f"缺少 XGB 模型: {model_dir}")

    bundle = load_split_feature_bundle(pool=pool)
    X = bundle.X_test
    n = len(X)
    if max_samples and n > max_samples:
        X = X[:max_samples]
        n = max_samples

    model = XGBClassifier()
    model.load_model(str(model_path))
    t0 = time.perf_counter()
    _ = model.predict(X)
    elapsed = time.perf_counter() - t0
    return {"n": n, "seconds": elapsed, "per_10k": _per_10k(elapsed, n)}


def _load_feature_transformer(model_dir: Path, device: torch.device) -> tuple:
    from finetune_transformer_1117_50obs import TransformerClassifier  # noqa: WPS433

    feat_cols = joblib.load(model_dir / "feature_columns.pkl")
    scaler = joblib.load(model_dir / "scaler.pkl")
    le = joblib.load(model_dir / "label_encoder.pkl")
    ckpt = torch.load(model_dir / "best_model.pth", map_location=device, weights_only=False)
    model = TransformerClassifier(
        input_dim=len(feat_cols),
        num_classes=len(le.classes_),
    ).to(device)
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    return model, scaler, feat_cols


def benchmark_feature_transformer(
    model_dir: Path, pool: str, device: torch.device, max_samples: Optional[int]
) -> dict:
    bundle = load_split_feature_bundle(pool=pool)
    model, scaler, feat_cols = _load_feature_transformer(model_dir, device)

    sub = bundle.test_df
    X = sub[feat_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0).values.astype(np.float32)
    X = np.clip(X, -5.0, 5.0)
    X_scaled = scaler.transform(X)

    n = len(X_scaled)
    if max_samples and n > max_samples:
        X_scaled = X_scaled[:max_samples]
        n = max_samples

    tensor = torch.from_numpy(X_scaled.astype(np.float32))

    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            batch = tensor[i : i + BATCH_SIZE].to(device)
            _ = model(batch)
    elapsed = time.perf_counter() - t0
    peak_mb = 0.0
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    return {"n": n, "seconds": elapsed, "per_10k": _per_10k(elapsed, n), "peak_mem_mb": peak_mb}


def _bench_torch_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_mask: bool = False,
) -> tuple[float, float]:
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                xb, _, mask = batch
                xb, mask = xb.to(device), mask.to(device)
                if use_mask and hasattr(model, "forward_with_mask"):
                    _ = model.forward_with_mask(xb, mask)
                else:
                    _ = model(xb)
            else:
                xb, _ = batch
                _ = model(xb.to(device))
    elapsed = time.perf_counter() - t0
    peak_mb = 0.0
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    return elapsed, peak_mb


def benchmark_e2e_small_50obs(device: torch.device, max_samples: Optional[int]) -> dict:
    from split_utils import load_e2e_arrays, load_e2e_label_encoder
    from train_transformer_e2e_50obs import TimeSeriesTransformer

    X, y = load_e2e_arrays("test")
    le = load_e2e_label_encoder()
    if max_samples:
        X, y = X[:max_samples], y[:max_samples]
    loader = DataLoader(NumpyTensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=False)
    model_dir = TRAIN_MODELS / "transformer_e2e_model_50obs"
    model = TimeSeriesTransformer(num_classes=len(le.classes_)).to(device)
    model.load_state_dict(torch.load(model_dir / "best_model.pth", map_location=device))
    elapsed, peak_mb = _bench_torch_loader(model, loader, device)
    return {"n": len(y), "seconds": elapsed, "per_10k": _per_10k(elapsed, len(y)), "peak_mem_mb": peak_mb}


def benchmark_e2e_varlen(
    model_dir: Path,
    model_factory: Callable[[int], nn.Module],
    device: torch.device,
    max_samples: Optional[int],
    use_mask: bool = True,
) -> dict:
    varlen = BASE / "data/e2e_varlen"
    with open(varlen / "test_data.pkl", "rb") as f:
        data = pickle.load(f)
    labels = np.load(varlen / "test_labels.npy")
    le = joblib.load(varlen / "label_encoder.pkl")
    if max_samples:
        data, labels = data[:max_samples], labels[:max_samples]

    loader = DataLoader(
        TrueVarLenDataset(data, labels, max_len=50),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    model = model_factory(num_classes=len(le.classes_)).to(device)
    model.load_state_dict(torch.load(model_dir / "best_model.pth", map_location=device))
    elapsed, peak_mb = _bench_torch_loader(model, loader, device, use_mask=use_mask)
    return {
        "n": len(labels),
        "seconds": elapsed,
        "per_10k": _per_10k(elapsed, len(labels)),
        "peak_mem_mb": peak_mb,
    }


def estimate_feature_times(
    test_df: pd.DataFrame, n_full: int, feature_sample_size: int
) -> dict[str, float]:
    """在 varlen test 子集上测特征提取，再按 n_full 线性外推。"""
    import benchmark_inference_time as leg

    sample_df = test_df.head(min(feature_sample_size, len(test_df)))
    out = {"all57": 0.0, "ls": 0.0}

    try:
        t_all, n_all = leg.benchmark_feature_extraction_all(sample_df, sample_size=len(sample_df))
        if n_all > 0:
            out["all57"] = t_all / n_all * n_full
    except Exception as e:
        logger.warning("全部57特征提取估算跳过: %s", e)

    try:
        t_ls, n_ls = leg.benchmark_feature_extraction_ls_only(sample_df, sample_size=len(sample_df))
        if n_ls > 0:
            out["ls"] = t_ls / n_ls * n_full
    except Exception as e:
        logger.warning("LS 特征提取估算跳过: %s", e)

    return out


def run_feature_extraction_rows(
    test_df: pd.DataFrame, n_full: int, feature_sample_size: int
) -> list[dict]:
    import benchmark_inference_time as leg

    rows = []
    sample_df = test_df.head(min(feature_sample_size, len(test_df)))
    fs = len(sample_df)

    try:
        t_all, n_all = leg.benchmark_feature_extraction_all(sample_df, sample_size=fs)
        feat_est = t_all / max(n_all, 1) * n_full
        rows.append(
            _build_row(
                exp=None,
                exp_id="feat_all_57",
                name="全部57特征提取",
                category="特征提取",
                compare_group="varlen主对比",
                device="cpu",
                n=n_full,
                infer_sec=0.0,
                feat_sec=feat_est,
                status="OK (模拟，按子集外推)",
                pool="varlen",
                stage="特征工程",
            )
        )
    except Exception as e:
        logger.warning("全部57特征提取跳过: %s", e)

    try:
        t_ls, n_ls = leg.benchmark_feature_extraction_ls_only(sample_df, sample_size=fs)
        feat_est = t_ls / max(n_ls, 1) * n_full if n_ls else 0.0
        rows.append(
            _build_row(
                exp=None,
                exp_id="feat_ls",
                name="Lomb-Scargle特征提取",
                category="特征提取",
                compare_group="varlen主对比",
                device="cpu",
                n=n_full if n_ls else n_ls,
                infer_sec=0.0,
                feat_sec=feat_est,
                status="OK" if n_ls else "无有效CSV",
                pool="varlen",
                stage="特征工程",
            )
        )
    except Exception as e:
        logger.warning("LS 特征提取跳过: %s", e)

    return rows


def run_original_four_comparisons(
    max_samples: Optional[int], feature_sample_size: int = 10000
) -> pd.DataFrame:
    """新划分 varlen test 上的旧模型路径基准（若存在）。"""
    import benchmark_inference_time as leg

    bundle = load_split_feature_bundle(pool="varlen")
    test_df = bundle.test_df
    n_full = len(test_df)
    if max_samples and n_full > max_samples:
        test_df = test_df.head(max_samples)
        n_infer = max_samples
    else:
        n_infer = n_full

    feat_times = estimate_feature_times(test_df, n_infer, feature_sample_size)
    rows = []

    for dev_name, dev in [("cpu", leg.DEVICE_CPU), ("gpu", leg.DEVICE_GPU)]:
        if dev_name == "gpu" and not torch.cuda.is_available():
            continue
        try:
            model, scaler, _, feat_cols = leg.load_transformer_model()
            elapsed = leg.benchmark_transformer_prediction(
                model, scaler, feat_cols, test_df, dev, dev_name.upper()
            )
            rows.append(
                _build_row(
                    exp=None,
                    exp_id="legacy_tf_1117",
                    name="Transformer 1117 (旧路径)",
                    category="手工特征",
                    compare_group="varlen主对比",
                    device=dev_name,
                    n=n_infer,
                    infer_sec=elapsed,
                    feat_sec=feat_times["ls"],
                    status="OK",
                    pool="varlen",
                    stage="旧模型",
                )
            )
        except Exception as e:
            logger.warning("旧 Transformer 1117 (%s) 跳过: %s", dev_name, e)

    try:
        xgb_model, feat_cols = leg.load_xgboost_model()
        elapsed = leg.benchmark_xgboost_prediction(xgb_model, feat_cols, test_df)
        rows.append(
            _build_row(
                exp=None,
                exp_id="legacy_xgb",
                name="XGBoost 7class (旧路径)",
                category="手工特征",
                compare_group="varlen主对比",
                device="cpu",
                n=n_infer,
                infer_sec=elapsed,
                feat_sec=feat_times["ls"],
                status="OK",
                pool="varlen",
                stage="旧模型",
            )
        )
    except Exception as e:
        logger.warning("旧 XGBoost 跳过: %s", e)

    return pd.DataFrame(rows)


def run_benchmarks(
    device_name: str,
    max_samples: Optional[int],
    feature_sample_size: int,
    cpu_infer_samples: int = 1000,
) -> pd.DataFrame:
    torch_device = DEVICE_GPU if device_name == "gpu" and torch.cuda.is_available() else DEVICE_CPU
    device_label = "gpu" if torch_device.type == "cuda" else "cpu"

    varlen_bundle = load_split_feature_bundle(pool="varlen")
    pool_n_full: dict[str, int] = {"varlen": len(varlen_bundle.test_df)}
    for pool in ("1121", "50obs"):
        pool_n_full[pool] = len(load_split_feature_bundle(pool=pool).test_df)

    n_varlen = min(max_samples, pool_n_full["varlen"]) if max_samples else pool_n_full["varlen"]

    feat_by_pool: dict[str, dict[str, float]] = {}
    for pool in ("varlen", "1121"):
        n_pool = min(max_samples, pool_n_full[pool]) if max_samples else pool_n_full[pool]
        feat_by_pool[pool] = estimate_feature_times(
            load_split_feature_bundle(pool=pool).test_df, n_pool, feature_sample_size
        )

    rows: list[dict] = []
    rows.extend(run_feature_extraction_rows(varlen_bundle.test_df, n_varlen, feature_sample_size))

    jobs: list[tuple[str, Callable[[Optional[int]], dict], str]] = [
        (
            "xgb_1117",
            lambda n: benchmark_xgboost(TRAIN_MODELS / "xgboost_optuna_1117", "varlen", n),
            "cpu",
        ),
        (
            "xgb_1121",
            lambda n: benchmark_xgboost(TRAIN_MODELS / "xgboost_optuna_1121", "1121", n),
            "cpu",
        ),
        (
            "tf_feat_50obs",
            lambda n: benchmark_feature_transformer(
                TRAIN_MODELS / "transformer_classifier_model_1117_50obs",
                "50obs",
                torch_device,
                n,
            ),
            device_label,
        ),
        (
            "tf_feat_finetune",
            lambda n: benchmark_feature_transformer(
                TRAIN_MODELS / "transformer_classifier_model_1117_50obs_finetuned",
                "varlen",
                torch_device,
                n,
            ),
            device_label,
        ),
        (
            "tf_feat_scratch",
            lambda n: benchmark_feature_transformer(
                TRAIN_MODELS / "transformer_classifier_model_1117",
                "varlen",
                torch_device,
                n,
            ),
            device_label,
        ),
        ("e2e_tf_small_50", lambda n: benchmark_e2e_small_50obs(torch_device, n), device_label),
    ]

    def e2e_small_ft(n: Optional[int]):
        from train_transformer_e2e_50obs import TimeSeriesTransformer

        return benchmark_e2e_varlen(
            TRAIN_MODELS / "transformer_e2e_model_finetuned_varlen",
            TimeSeriesTransformer,
            torch_device,
            n,
        )

    def e2e_matched_50(n: Optional[int]):
        from split_utils import load_e2e_arrays, load_e2e_label_encoder
        from transformer_e2e_matched import MatchedTimeSeriesTransformer

        X, y = load_e2e_arrays("test")
        le = load_e2e_label_encoder()
        if n:
            X, y = X[:n], y[:n]
        loader = DataLoader(NumpyTensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=False)
        model_dir = TRAIN_MODELS / "transformer_e2e_model_50obs_matched"
        if not (model_dir / "best_model.pth").is_file():
            raise FileNotFoundError("缺少 matched 50obs 权重")
        model = MatchedTimeSeriesTransformer(num_classes=len(le.classes_)).to(torch_device)
        model.load_state_dict(torch.load(model_dir / "best_model.pth", map_location=torch_device))
        elapsed, peak_mb = _bench_torch_loader(model, loader, torch_device)
        return {"n": len(y), "seconds": elapsed, "per_10k": _per_10k(elapsed, len(y)), "peak_mem_mb": peak_mb}

    def e2e_matched_ft(n: Optional[int]):
        from transformer_e2e_matched import MatchedTimeSeriesTransformer

        return benchmark_e2e_varlen(
            TRAIN_MODELS / "transformer_e2e_model_finetuned_varlen_matched",
            MatchedTimeSeriesTransformer,
            torch_device,
            n,
        )

    def e2e_matched_scratch(n: Optional[int]):
        from transformer_e2e_matched import MatchedTimeSeriesTransformer

        return benchmark_e2e_varlen(
            TRAIN_MODELS / "transformer_e2e_model_varlen_matched_scratch",
            MatchedTimeSeriesTransformer,
            torch_device,
            n,
        )

    def rnn(n: Optional[int]):
        from train_rnn_classifier_e2e import LSTMClassifier

        return benchmark_e2e_varlen(
            TRAIN_MODELS / "rnn_e2e_model_varlen",
            LSTMClassifier,
            torch_device,
            n,
        )

    jobs.extend(
        [
            ("e2e_tf_small_ft", e2e_small_ft, device_label),
            ("e2e_tf_matched_50", e2e_matched_50, device_label),
            ("e2e_tf_matched_ft", e2e_matched_ft, device_label),
            ("e2e_tf_matched_scratch", e2e_matched_scratch, device_label),
            ("rnn_e2e_varlen", rnn, device_label),
        ]
    )

    exp_map = {e.exp_id: e for e in EXPERIMENTS}
    for exp_id, fn, job_device in jobs:
        exp = exp_map.get(exp_id)
        name = exp.name if exp else exp_id
        category = _model_category(exp)
        compare_group = _compare_group(exp)
        pool_key = exp.pool if exp else "varlen"
        n_full = pool_n_full.get(pool_key, pool_n_full["varlen"])
        is_pytorch = not exp_id.startswith("xgb_")
        n_report, n_run, speed_note = resolve_infer_plan(
            n_full, max_samples, device_label, cpu_infer_samples, is_pytorch
        )
        feat_sec = 0.0
        if category == "手工特征" and compare_group == "varlen主对比":
            feat_sec = feat_by_pool.get(pool_key, feat_by_pool["varlen"])["ls"]
        weight_mb = _checkpoint_mb(exp.model_dir) if exp else None

        try:
            r = fn(n_run)
            r = scale_infer_result(r, n_report, speed_note)
            rows.append(
                _build_row(
                    exp=exp,
                    exp_id=exp_id,
                    name=name,
                    category=category,
                    compare_group=compare_group,
                    device=job_device,
                    n=r["n"],
                    infer_sec=r["seconds"],
                    feat_sec=feat_sec,
                    peak_mem_mb=float(r.get("peak_mem_mb", 0.0) or 0.0),
                    weight_mb=weight_mb,
                    speed_note=r.get("speed_note", speed_note),
                )
            )
            logger.info(
                "%s (%s): infer=%.2fs feat=%.2fs total=%.2fs%s",
                name,
                job_device,
                r["seconds"],
                feat_sec,
                r["seconds"] + feat_sec,
                f" [{speed_note}]" if speed_note else "",
            )
        except Exception as e:
            logger.warning("%s 跳过: %s", name, e)
            rows.append(
                _build_row(
                    exp=exp,
                    exp_id=exp_id,
                    name=name,
                    category=category,
                    compare_group=compare_group,
                    device=job_device,
                    n=0,
                    infer_sec=0.0,
                    feat_sec=0.0,
                    status=f"跳过: {e}",
                )
            )

    return pd.DataFrame(rows)


def _df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "(无数据)"
    try:
        return df.to_markdown(index=False)
    except ImportError:
        cols = list(df.columns)
        header = "| " + " | ".join(str(c) for c in cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        rows = [
            "| " + " | ".join(str(row[c]) for c in cols) + " |"
            for _, row in df.iterrows()
        ]
        return "\n".join([header, sep, *rows])


def _df_section(title: str, df: pd.DataFrame) -> list[str]:
    return [f"## {title}", "", _df_to_markdown(df), ""]


def _ok_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["状态"].astype(str).str.startswith("OK")].copy()


def _deploy_device_breakdown(row: pd.Series) -> tuple[str, str]:
    """主表生产部署口径：各阶段实际硬件。"""
    category = str(row.get("类别", ""))
    exp_id = str(row.get("exp_id", ""))
    infer_dev = str(row.get("device", ""))
    if category == "原始时序":
        return "—", f"{infer_dev}（神经网络）"
    if exp_id.startswith("xgb_"):
        return "cpu（Lomb–Scargle）", "cpu（XGBoost）"
    return "cpu（Lomb–Scargle）", f"{infer_dev}（Transformer 分类头）"


def _build_deploy_device_section(deploy: pd.DataFrame) -> list[str]:
    if deploy.empty:
        return []
    rows = []
    for _, r in deploy.iterrows():
        feat_dev, infer_dev = _deploy_device_breakdown(r)
        rows.append(
            {
                "模型": r["模型"],
                "特征提取": feat_dev,
                "模型推理": infer_dev,
                "主表 vs 附录": "主表（生产部署）"
                if infer_dev.startswith("gpu")
                or str(r.get("exp_id", "")).startswith("xgb_")
                else "主表",
            }
        )
    return [
        "## 各模型测速设备（主表 = 生产部署，附录 = CPU 对照）",
        "",
        "| 路径 | 特征提取 | 模型推理 | 放入 |",
        "| --- | --- | --- | --- |",
        *[
            f"| {row['模型']} | {row['特征提取']} | {row['模型推理']} | {row['主表 vs 附录']} |"
            for row in rows
        ],
        "",
        "- **主表**：各模型最合理的上线组合（神经网络用 GPU 全量 test；LS / XGB 用 CPU）",
        "- **附录 Table S1**：仅 PyTorch 在 CPU 上 1000 样本外推，用于无 GPU 环境对照，**不含 XGB**（XGB 本身即 CPU 全量实测）",
        "",
    ]


def select_deploy_rows(df: pd.DataFrame) -> pd.DataFrame:
    """主文 Table：生产部署向（E2E/TF 推理=gpu；LS/XGB=cpu，每个 exp_id 一行）。"""
    ok = _ok_rows(df)
    deploy = ok[
        (ok["对比组"] == "varlen主对比")
        & (ok["类别"].isin(["手工特征", "原始时序"]))
    ].copy()
    if deploy.empty:
        return deploy

    picked: list[pd.Series] = []
    for exp_id, grp in deploy.groupby("exp_id", sort=False):
        gpu = grp[grp["device"] == "gpu"]
        if not gpu.empty:
            picked.append(gpu.iloc[0])
            continue
        cpu = grp[grp["device"] == "cpu"]
        if not cpu.empty:
            picked.append(cpu.iloc[0])
    out = pd.DataFrame(picked)
    return out.sort_values("每万样本_秒")


def select_cpu_appendix_rows(df: pd.DataFrame) -> pd.DataFrame:
    """附录 Table：CPU 上 PyTorch 1000 样本外推（及 50obs 预训练 CPU 外推）。"""
    ok = _ok_rows(df)
    note = ok["测速备注"].astype(str)
    status = ok["状态"].astype(str)
    mask = note.str.contains("CPU", na=False) | status.str.contains("CPU 1000", na=False)
    out = ok[mask & (ok["device"] == "cpu")].copy()
    return out.sort_values(["对比组", "每万样本_秒"])


def select_feature_rows(df: pd.DataFrame) -> pd.DataFrame:
    """特征提取行：each 模式只保留一套（LS 仅在 CPU 上测）。"""
    feat = _ok_rows(df)
    feat = feat[feat["类别"] == "特征提取"].copy()
    if feat.empty:
        return feat
    return feat.drop_duplicates(subset=["exp_id"], keep="first")


def build_resources_table(df: pd.DataFrame) -> pd.DataFrame:
    """第三张表：精度 + 训练成本 + GPU 部署推理/显存。"""
    metrics = pd.DataFrame(collect_metrics_table(include_resources=True))
    deploy = select_deploy_rows(df)
    if deploy.empty:
        return metrics

    infer = deploy[
        [
            "exp_id",
            "每万样本_秒",
            "特征提取_秒",
            "推理_秒",
            "峰值显存_MB",
            "权重_MB",
        ]
    ].rename(
        columns={
            "每万样本_秒": "GPU部署_每万样本_秒",
            "特征提取_秒": "LS特征提取_秒",
            "推理_秒": "GPU推理_秒",
            "峰值显存_MB": "GPU峰值显存_MB",
            "权重_MB": "checkpoint_MB",
        }
    )
    merged = metrics.merge(infer, on="exp_id", how="left")
    cols = [
        "exp_id",
        "模型",
        "实验组",
        "阶段",
        "宏平均F1",
        "准确率",
        "权重文件_MB",
        "参数量_M",
        "训练墙钟_小时",
        "计划epoch",
        "最佳epoch",
        "GPU部署_每万样本_秒",
        "LS特征提取_秒",
        "GPU推理_秒",
        "GPU峰值显存_MB",
        "训练备注",
    ]
    cols = [c for c in cols if c in merged.columns]
    main_ids = set(deploy["exp_id"].tolist())
    out = merged[merged["exp_id"].isin(main_ids)].copy()
    return out.sort_values("宏平均F1", ascending=False)[cols]


def _build_ranking_section(deploy: pd.DataFrame, title: str) -> list[str]:
    if deploy.empty:
        return [f"## {title}", "", "(无数据)", ""]

    baseline_row = deploy.loc[deploy["每万样本_秒"].idxmax()]
    baseline_name = str(baseline_row["模型"])
    baseline_sec = float(baseline_row["每万样本_秒"])

    ranked = deploy.sort_values("每万样本_秒")
    summary_rows = []
    for _, row in ranked.iterrows():
        sec = float(row["每万样本_秒"])
        speedup = baseline_sec / sec if sec > 0 else float("inf")
        summary_rows.append(
            {
                "模型": row["模型"],
                "路径": row["类别"],
                "每万样本总耗时(s)": f"{sec:.4f}",
                f"相对{baseline_name}": f"{speedup:.2f}x",
                "GPU峰值显存(MB)": f"{float(row.get('峰值显存_MB', 0) or 0):.1f}",
            }
        )
    lines = [
        f"## {title}",
        "",
        f"- **排序基准（最慢）**: {baseline_name}（{baseline_sec:.4f} s/万样本）",
        "",
        _df_to_markdown(pd.DataFrame(summary_rows)),
        "",
    ]
    return lines


def _paper_table_cols(df: pd.DataFrame, extra: Optional[list[str]] = None) -> pd.DataFrame:
    cols = [
        "exp_id",
        "模型",
        "类别",
        "阶段",
        "device",
        "n_test",
        "特征提取_秒",
        "推理_秒",
        "总耗时_秒",
        "每万样本_秒",
        "峰值显存_MB",
        "权重_MB",
        "测速备注",
    ]
    if extra:
        cols = extra + [c for c in cols if c not in extra]
    existing = [c for c in cols if c in df.columns]
    return df[existing]


def _build_summary_section(df: pd.DataFrame) -> list[str]:
    deploy = select_deploy_rows(df)
    feat = select_feature_rows(df)

    lines = ["## 结论摘要（主文 GPU 部署向）", ""]
    if deploy.empty:
        lines.extend(["(无 varlen 部署模型 GPU 数据)", ""])
        return lines

    lines.extend(_build_ranking_section(deploy, "varlen 部署模型耗时排名")[2:])  # skip duplicate header

    if not feat.empty:
        feat_rows = []
        for _, row in feat.iterrows():
            feat_rows.append(
                {
                    "任务": row["模型"],
                    "每万样本耗时(s)": f"{float(row['每万样本_秒']):.4f}",
                    "备注": "模拟" if "模拟" in str(row["状态"]) else "实测外推",
                }
            )
        lines.extend(
            [
                "### 特征提取（varlen，单独测速）",
                "",
                _df_to_markdown(pd.DataFrame(feat_rows)),
                "",
            ]
        )
    return lines


def write_report(df: pd.DataFrame, out_md: Path, out_csv: Path, *, write_csv: bool = True) -> None:
    out_dir = out_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    if write_csv:
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    deploy_df = select_deploy_rows(df)
    cpu_df = select_cpu_appendix_rows(df)
    feat_df = select_feature_rows(df)
    resources_df = build_resources_table(df)
    pretrain_gpu = _ok_rows(df)
    pretrain_gpu = pretrain_gpu[
        (pretrain_gpu["对比组"] == "50obs预训练") & (pretrain_gpu["device"] == "gpu")
    ].copy()

    deploy_cols = _paper_table_cols(deploy_df)
    cpu_cols = _paper_table_cols(cpu_df)
    resources_cols = resources_df  # already curated

    # --- 主文 GPU 部署表 ---
    deploy_md = out_dir / "inference_time_benchmark_deploy.md"
    deploy_csv = out_dir / "inference_time_benchmark_deploy.csv"
    deploy_lines = [
        "# Table 1 — 生产部署推理耗时（主文）",
        "",
        f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 口径",
        "",
        "- **适用**: 论文主文 Table 1 / 正文部署性能（混合 CPU+GPU，非「全 GPU」）",
        "- **E2E / RNN**: 仅 GPU 神经网络推理，全量 test（n≈257742）；不含离线 `prepare_e2e`",
        "- **Transformer 特征路径**: CPU 做 Lomb–Scargle 特征 + GPU 做 Transformer 分类头",
        "- **XGBoost 路径**: CPU 做 Lomb–Scargle 特征 + CPU 做 XGBoost 预测（无 GPU 实现）",
        "- **LS 特征**: CPU 上 1000 样本子集实测，线性外推至全 test",
        "",
    ]
    deploy_lines.extend(_build_deploy_device_section(deploy_df))
    deploy_lines.extend(_build_ranking_section(deploy_df, "排名摘要"))
    if not feat_df.empty:
        deploy_lines.extend(
            _df_section("特征提取（单独）", _paper_table_cols(feat_df))
        )
    deploy_lines.extend(_df_section("varlen 部署模型（主表）", deploy_cols))
    deploy_lines.extend(
        [
            "## 英文图注参考 (Figure/Table caption)",
            "",
            "Inference latency on the varlen test split (75/10/15 hold-out) under a "
            "production-style mixed CPU/GPU setup: end-to-end neural models use GPU "
            "forward passes; Lomb–Scargle features and XGBoost use CPU; "
            "transformer-on-features runs LS on CPU and the classifier on GPU. "
            "End-to-end models exclude offline light-curve preprocessing.",
            "",
        ]
    )
    deploy_md.write_text("\n".join(deploy_lines), encoding="utf-8")
    deploy_df.to_csv(deploy_csv, index=False, encoding="utf-8-sig")

    # --- 附录 CPU 外推表 ---
    appendix_md = out_dir / "inference_time_benchmark_cpu_appendix.md"
    appendix_csv = out_dir / "inference_time_benchmark_cpu_appendix.csv"
    appendix_lines = [
        "# Table S1 — CPU 推理外推（附录）",
        "",
        f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 口径",
        "",
        "- **适用**: 论文附录；**不**代表生产部署配置",
        "- **PyTorch**: CPU 上仅测 **1000** 个 test 样本，再线性外推至全 test",
        "- 仅用于与 GPU 或不同硬件环境对照，勿与主文 Table 1 直接混排排名",
        "",
        _df_to_markdown(cpu_cols) if not cpu_df.empty else "(无 CPU 外推数据；请使用 `--device both` 或 `--device cpu` 运行 benchmark)",
        "",
        "## 英文附录说明参考",
        "",
        "CPU timings for PyTorch models were measured on 1,000 test samples "
        "and linearly extrapolated to the full test set. "
        "These results are for hardware comparison only.",
        "",
    ]
    appendix_md.write_text("\n".join(appendix_lines), encoding="utf-8")
    cpu_df.to_csv(appendix_csv, index=False, encoding="utf-8-sig")

    # --- 资源与精度综合表 ---
    resources_md = out_dir / "inference_time_benchmark_resources.md"
    resources_csv = out_dir / "inference_time_benchmark_resources.csv"
    resources_lines = [
        "# Table 2 — 精度、训练成本与 GPU 部署资源（主文/附录）",
        "",
        f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 口径",
        "",
        "- **精度**: 统一 75/10/15 test 集 Macro-F1 / Accuracy",
        "- **训练墙钟_小时**: 从各实验 `training.log` 首尾时间戳估算",
        "- **GPU 推理列**: 与 Table 1 同源（`inference_time_benchmark_deploy.csv`）",
        "",
        _df_to_markdown(resources_cols) if not resources_df.empty else "(无数据)",
        "",
    ]
    resources_md.write_text("\n".join(resources_lines), encoding="utf-8")
    resources_df.to_csv(resources_csv, index=False, encoding="utf-8-sig")

    # --- 总索引 + 完整归档 ---
    lines = [
        "# 推理与特征提取耗时基准（索引）",
        "",
        f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 论文用表（已拆分）",
        "",
        "| 文件 | 用途 |",
        "| --- | --- |",
        "| `inference_time_benchmark_deploy.md` | **主文 Table 1** — GPU 部署 |",
        "| `inference_time_benchmark_cpu_appendix.md` | **附录 Table S1** — CPU 1000 样本外推 |",
        "| `inference_time_benchmark_resources.md` | **Table 2** — 精度 + 训练成本 + GPU 显存/耗时 |",
        "",
        "## 对比口径",
        "",
        "- **varlen 主对比**: `data/split/varlen` hold-out 15% test（n≈257742）",
        "- **50obs 预训练**: 中间 checkpoint，见 deploy 表或完整 CSV",
        "- **手工特征总耗时** = Lomb-Scargle（子集外推）+ 模型推理",
        "- **原始时序总耗时** = GPU 模型推理（`prepare_e2e` 离线，未计入）",
        "",
    ]
    lines.extend(_build_summary_section(df))
    if not pretrain_gpu.empty:
        lines.extend(
            _df_section(
                "50obs 预训练（GPU，仅供参考）",
                _paper_table_cols(pretrain_gpu),
            )
        )
    lines.extend(_df_section("完整原始结果（归档）", df))

    lines.extend(
        [
            "## 说明",
            "",
            "- 修改报告样式后可用 `--report-only` 从 `inference_time_benchmark.csv` 重生成",
            "- 推荐 benchmark 命令: `--device gpu --skip-legacy`（主文）；`--device both` 可同时得到 CPU 附录",
            "",
        ]
    )
    out_md.write_text("\n".join(lines), encoding="utf-8")
    logger.info(
        "已保存 %s, %s, %s, %s 及索引 %s",
        deploy_csv,
        appendix_csv,
        resources_csv,
        out_csv,
        out_md,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="统一推理与特征提取耗时基准（默认 --device gpu，部署向）",
        epilog=USAGE_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--max-samples", type=int, default=None, help="限制推理 test 样本数")
    parser.add_argument(
        "--feature-samples",
        type=int,
        default=1000,
        help="LS/57特征测速用的 test 子集大小，再外推至全 test",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu", "both"],
        default="gpu",
        help="PyTorch 推理设备；默认 gpu（部署）。both=CPU+GPU 各跑一遍（慢）",
    )
    parser.add_argument(
        "--cpu-infer-samples",
        type=int,
        default=1000,
        help="CPU 上 PyTorch 推理子集大小，再线性外推至全 test（默认 1000）",
    )
    parser.add_argument("--skip-legacy", action="store_true", help="跳过旧路径手工特征模型")
    parser.add_argument("--legacy-only", action="store_true", help="仅跑旧路径手工特征模型")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="从 results/inference_time_benchmark.csv 重生成 Markdown，不重新测速",
    )
    args = parser.parse_args()

    out_dir = TRAIN_MODELS / "results"
    out_csv = out_dir / "inference_time_benchmark.csv"
    out_md = out_dir / "inference_time_benchmark.md"

    if args.report_only:
        if not out_csv.exists():
            logger.error("未找到 %s，请先完整运行一次 benchmark", out_csv)
            sys.exit(1)
        df = pd.read_csv(out_csv)
        write_report(df, out_md, out_csv, write_csv=False)
        main_show = df[df["对比组"] == "varlen主对比"] if not df.empty else df
        print(main_show.to_string(index=False))
        return

    if args.device == "both":
        logger.warning(
            "--device both 会在 CPU 上对 PyTorch 再跑一遍（默认 %d 样本外推）。"
            "部署向请用 --device gpu --skip-legacy。",
            args.cpu_infer_samples,
        )

    frames = []

    if not args.legacy_only:
        if args.device in ("cpu", "both"):
            frames.append(
                run_benchmarks("cpu", args.max_samples, args.feature_samples, args.cpu_infer_samples)
            )
        if args.device in ("gpu", "both") and torch.cuda.is_available():
            frames.append(
                run_benchmarks("gpu", args.max_samples, args.feature_samples, args.cpu_infer_samples)
            )
        elif args.device == "gpu":
            logger.warning("GPU 不可用，跳过 GPU 测试")

    if not args.skip_legacy:
        frames.append(run_original_four_comparisons(args.max_samples, args.feature_samples))

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    write_report(df, out_dir / "inference_time_benchmark.md", out_dir / "inference_time_benchmark.csv")

    main_show = df[df["对比组"] == "varlen主对比"] if not df.empty else df
    print(main_show.to_string(index=False))


if __name__ == "__main__":
    main()
