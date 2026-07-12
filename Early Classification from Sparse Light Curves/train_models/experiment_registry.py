#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一实验清单：划分、指标路径、推理基准配置。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

BASE = Path("/root/shared-nvme")
TRAIN_MODELS = BASE / "train_models"
SPLIT_ROOT = BASE / "data/split"


@dataclass
class Experiment:
    exp_id: str
    name: str
    group: str  # 手工特征 / E2E轻量 / E2E同量级 / RNN
    stage: str  # 50obs预训练 / 3-30微调 / 端到端
    pool: str  # varlen | 50obs | 1121
    input_desc: str
    model_dir: Path
    metrics_candidates: List[Path] = field(default_factory=list)
    split_ratio: str = "75/10/15"
    random_state: int = 42
    notes: str = ""

    @property
    def split_dir(self) -> Path:
        return SPLIT_ROOT / self.pool


EXPERIMENTS: List[Experiment] = [
    Experiment(
        "xgb_1117",
        "XGBoost 1117",
        "手工特征",
        "3-30变长",
        "varlen",
        "train2 手工特征",
        TRAIN_MODELS / "xgboost_optuna_1117",
        [
            TRAIN_MODELS / "xgboost_optuna_1117/results/test_metrics.json",
        ],
    ),
    Experiment(
        "xgb_1121",
        "XGBoost 1121",
        "手工特征",
        "3-30变长",
        "1121",
        "train2_1121 手工特征",
        TRAIN_MODELS / "xgboost_optuna_1121",
        [
            TRAIN_MODELS / "xgboost_optuna_1121/results/test_metrics.json",
        ],
    ),
    Experiment(
        "tf_feat_50obs",
        "Transformer 特征 50obs",
        "手工特征",
        "50点预训练",
        "50obs",
        "train4 手工特征, ~250M",
        TRAIN_MODELS / "transformer_classifier_model_1117_50obs",
        [
            TRAIN_MODELS / "transformer_classifier_model_1117_50obs/results/test_metrics.json",
            TRAIN_MODELS / "transformer_classifier_model_1117_50obs/results/metrics.txt",
        ],
    ),
    Experiment(
        "tf_feat_finetune",
        "Transformer 特征 3-30微调",
        "手工特征",
        "3-30微调",
        "varlen",
        "train2 手工特征, ~250M",
        TRAIN_MODELS / "transformer_classifier_model_1117_50obs_finetuned",
        [
            TRAIN_MODELS / "transformer_classifier_model_1117_50obs_finetuned/results/test_metrics.json",
            TRAIN_MODELS / "transformer_classifier_model_1117_50obs_finetuned/results/metrics.txt",
        ],
    ),
    Experiment(
        "tf_feat_scratch",
        "Transformer 特征 3-30从头",
        "手工特征",
        "3-30从头",
        "varlen",
        "train2 手工特征, ~250M, 无50obs预训练",
        TRAIN_MODELS / "transformer_classifier_model_1117",
        [
            TRAIN_MODELS / "transformer_classifier_model_1117/results/test_metrics.json",
            TRAIN_MODELS / "transformer_classifier_model_1117/results/metrics.txt",
        ],
    ),
    Experiment(
        "e2e_tf_small_50",
        "E2E Transformer 轻量 50obs",
        "E2E轻量",
        "50点预训练",
        "50obs",
        "原始时序 (3,50), ~11M",
        TRAIN_MODELS / "transformer_e2e_model_50obs",
        [TRAIN_MODELS / "transformer_e2e_model_50obs/results/test_metrics.json"],
    ),
    Experiment(
        "e2e_tf_small_ft",
        "E2E Transformer 轻量 3-30微调",
        "E2E轻量",
        "3-30微调",
        "varlen",
        "原始时序 3-30, ~11M",
        TRAIN_MODELS / "transformer_e2e_model_finetuned_varlen",
        [TRAIN_MODELS / "transformer_e2e_model_finetuned_varlen/results/test_metrics.json"],
    ),
    Experiment(
        "e2e_tf_matched_50",
        "E2E Transformer 同量级 50obs",
        "E2E同量级",
        "50点预训练",
        "50obs",
        "原始时序 (3,50), ~227M",
        TRAIN_MODELS / "transformer_e2e_model_50obs_matched",
        [TRAIN_MODELS / "transformer_e2e_model_50obs_matched/results/test_metrics.json"],
    ),
    Experiment(
        "e2e_tf_matched_ft",
        "E2E Transformer 同量级 3-30微调",
        "E2E同量级",
        "3-30微调",
        "varlen",
        "原始时序 3-30, ~227M",
        TRAIN_MODELS / "transformer_e2e_model_finetuned_varlen_matched",
        [TRAIN_MODELS / "transformer_e2e_model_finetuned_varlen_matched/results/test_metrics.json"],
    ),
    Experiment(
        "e2e_tf_matched_scratch",
        "E2E Transformer 同量级 3-30从头",
        "E2E同量级",
        "3-30从头",
        "varlen",
        "原始时序 3-30, ~227M, 无50obs预训练",
        TRAIN_MODELS / "transformer_e2e_model_varlen_matched_scratch",
        [
            TRAIN_MODELS
            / "transformer_e2e_model_varlen_matched_scratch/results/test_metrics.json"
        ],
    ),
    Experiment(
        "rnn_e2e_varlen",
        "RNN LSTM E2E",
        "RNN",
        "3-30端到端",
        "varlen",
        "原始时序 3-30, ~0.2M",
        TRAIN_MODELS / "rnn_e2e_model_varlen",
        [TRAIN_MODELS / "rnn_e2e_model_varlen/results/test_metrics.json"],
    ),
]


def _parse_metrics_txt(text: str) -> dict:
    acc, macro_f1, n_test = 0.0, 0.0, None
    for line in text.splitlines():
        if line.startswith("Accuracy:"):
            acc = float(line.split(":", 1)[1].strip())
        if line.startswith("Macro-F1:"):
            macro_f1 = float(line.split(":", 1)[1].strip())
        if line.strip().startswith("macro avg"):
            parts = line.split()
            if len(parts) >= 4:
                try:
                    macro_f1 = float(parts[3])
                except ValueError:
                    pass
        m = re.search(r"\baccuracy\b\s+([\d.]+)\s+(\d+)", line)
        if m:
            acc = float(m.group(1))
            n_test = int(m.group(2))
    return {"accuracy": acc, "macro_f1": macro_f1, "n_test": n_test}


def _checkpoint_mb(model_dir: Path) -> Optional[float]:
    candidates = [
        model_dir / "best_model.pth",
        model_dir / "best_model.json",
        model_dir / f"{model_dir.name}_best.json",
    ]
    for path in candidates:
        if path.is_file():
            return round(path.stat().st_size / (1024 * 1024), 2)
    return None


def _parse_config_text(text: str) -> dict:
    out: dict = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip()
        val = val.strip()
        if key == "N_EPOCHS":
            try:
                out["train_epochs_planned"] = int(val)
            except ValueError:
                pass
        if key.startswith("best_epoch"):
            m = re.search(r"best_epoch\s*=\s*(\d+)", line)
            if m:
                out["best_epoch"] = int(m.group(1))
        if key == "MODEL_CONFIG":
            out["model_config"] = val
    m = re.search(r"best_epoch\s*=\s*(\d+)", text)
    if m and "best_epoch" not in out:
        out["best_epoch"] = int(m.group(1))
    return out


def _parse_training_log(model_dir: Path) -> dict:
    log_path = model_dir / "training.log"
    if not log_path.is_file():
        return {}
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines:
        return {}
    out: dict = {}
    ts_pattern = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+")
    timestamps: list[datetime] = []
    for line in lines:
        m = ts_pattern.match(line)
        if m:
            try:
                timestamps.append(datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
            except ValueError:
                pass
        pm = re.search(r"模型参数量:\s*([\d,]+)\s*\(([\d.]+)M\)", line)
        if pm:
            out["params_m"] = float(pm.group(2))
    if len(timestamps) >= 2:
        hours = (timestamps[-1] - timestamps[0]).total_seconds() / 3600
        out["train_wall_hours"] = round(max(hours, 0.0), 2)
    return out


def _training_history_epochs(model_dir: Path) -> Optional[int]:
    for rel in ("results/training_history.csv", "training_history.csv"):
        path = model_dir / rel
        if not path.is_file():
            continue
        try:
            import pandas as pd

            hist = pd.read_csv(path)
            if "epoch" in hist.columns:
                return int(hist["epoch"].max())
        except Exception:
            pass
    return None


def collect_resource_profile(exp: Experiment) -> dict:
    """训练成本与模型占用（来自已有 config/log/checkpoint，无需重训）。"""
    model_dir = exp.model_dir
    profile: dict = {
        "权重文件_MB": _checkpoint_mb(model_dir),
        "参数量_M": None,
        "计划epoch": None,
        "实际epoch": None,
        "最佳epoch": None,
        "训练墙钟_小时": None,
        "训练备注": "",
    }

    for cfg_name in ("training_config.txt", "finetuning_config.txt"):
        cfg = model_dir / cfg_name
        if cfg.is_file():
            parsed = _parse_config_text(cfg.read_text(encoding="utf-8"))
            profile["计划epoch"] = parsed.get("train_epochs_planned")
            profile["最佳epoch"] = parsed.get("best_epoch")
            break

    log_info = _parse_training_log(model_dir)
    if log_info.get("params_m") is not None:
        profile["参数量_M"] = log_info["params_m"]
    if log_info.get("train_wall_hours") is not None:
        profile["训练墙钟_小时"] = log_info["train_wall_hours"]

    actual = _training_history_epochs(model_dir)
    if actual is not None:
        profile["实际epoch"] = actual

    if exp.exp_id.startswith("xgb_"):
        profile["训练备注"] = "Optuna 超参搜索（见 best_params.txt）"
        if profile["权重文件_MB"] is None:
            for p in model_dir.glob("*_best.json"):
                profile["权重文件_MB"] = round(p.stat().st_size / (1024 * 1024), 2)
                break

    if profile["参数量_M"] is None and profile["权重文件_MB"]:
        # 粗估：fp32 参数 ≈ 权重体积；含 optimizer 状态时 pth 更大
        profile["参数量_M"] = round(profile["权重文件_MB"] * 0.25, 1)

    return profile


def read_experiment_metrics(exp: Experiment) -> dict:
    """读取 test 集 Accuracy / Macro-F1；文件缺失则返回空指标。"""
    for path in exp.metrics_candidates:
        if not path.is_file():
            continue
        if path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            return {
                "accuracy": float(data.get("accuracy", 0)),
                "macro_f1": float(data.get("macro_f1", 0)),
                "n_test": data.get("n_test"),
                "metrics_file": str(path),
            }
        if path.suffix == ".txt":
            parsed = _parse_metrics_txt(path.read_text(encoding="utf-8"))
            parsed["metrics_file"] = str(path)
            return parsed
    return {"accuracy": None, "macro_f1": None, "n_test": None, "metrics_file": None}


def collect_metrics_table(include_resources: bool = True) -> List[dict]:
    rows = []
    for exp in EXPERIMENTS:
        m = read_experiment_metrics(exp)
        row = {
            "exp_id": exp.exp_id,
            "模型": exp.name,
            "实验组": exp.group,
            "阶段": exp.stage,
            "数据池": exp.pool,
            "输入": exp.input_desc,
            "划分": exp.split_ratio,
            "random_state": exp.random_state,
            "准确率": m.get("accuracy"),
            "宏平均F1": m.get("macro_f1"),
            "n_test": m.get("n_test"),
            "指标文件": m.get("metrics_file"),
            "模型目录": str(exp.model_dir),
            "状态": "OK" if m.get("accuracy") is not None else "缺失",
        }
        if include_resources:
            row.update(collect_resource_profile(exp))
        rows.append(row)
    return rows
