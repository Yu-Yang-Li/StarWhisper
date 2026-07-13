#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""深度学习训练通用工具：早停、评估、作图。"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class NumpyTensorDataset(Dataset):
    """固定长度数据集（用于预训练 50 点）"""
    def __init__(self, X: np.ndarray, y: np.ndarray, channels_first: bool = True):
        self.X = X
        self.y = y
        self.channels_first = channels_first

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = self.X[idx]
        if self.channels_first:
            x = np.transpose(x, (1, 0))  # (C,L) -> (L,C)
        t = torch.from_numpy(np.ascontiguousarray(x)).float()
        return t, torch.tensor(self.y[idx], dtype=torch.long)


class VarLenCropDataset(Dataset):
    """从固定 50 点序列随机截取 3~30 点（用于微调阶段，模拟变长）"""
    def __init__(self, X: np.ndarray, y: np.ndarray, min_len: int = 3, max_len: int = 30, seed: int = 42):
        self.X = X
        self.y = y
        self.min_len = min_len
        self.max_len = max_len
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = self.X[idx]  # (3, 50)
        L = int(self.rng.integers(self.min_len, self.max_len + 1))
        start = int(self.rng.integers(0, 50 - L + 1))
        patch = x[:, start:start + L]
        pad = np.zeros((3, 50), dtype=np.float32)
        pad[:, :L] = patch
        t = torch.from_numpy(np.transpose(pad, (1, 0))).float()
        mask = torch.zeros(50, dtype=torch.bool)
        mask[:L] = True
        return t, torch.tensor(self.y[idx], dtype=torch.long), mask


class TrueVarLenDataset(Dataset):
    """真实的变长序列数据集（从 pickle 加载，用于真正变长数据）"""
    def __init__(self, data_list, labels, max_len: int = 50):
        self.data_list = data_list  # list of (3, L) arrays
        self.labels = labels
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data_list[idx]  # (3, L)
        L = x.shape[1]
        label = self.labels[idx]

        pad_len = self.max_len - L
        if pad_len > 0:
            padded = np.pad(x, ((0, 0), (0, pad_len)), mode='constant', constant_values=0)
            mask = np.array([1] * L + [0] * pad_len, dtype=bool)
        else:
            padded = x
            mask = np.array([1] * L, dtype=bool)

        padded = np.transpose(padded, (1, 0))  # (L, C)
        return (
            torch.FloatTensor(padded),
            torch.tensor(label, dtype=torch.long),
            torch.BoolTensor(mask)
        )


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_mask: bool = False,
) -> Tuple[float, np.ndarray, np.ndarray, float]:
    model.eval()
    preds, trues = [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    n = 0
    for batch in loader:
        if len(batch) == 3:
            xb, yb, mask = batch
            mask = mask.to(device)
        else:
            xb, yb = batch
            mask = None
        xb, yb = xb.to(device), yb.to(device)
        if mask is not None and hasattr(model, "forward_with_mask"):
            logits = model.forward_with_mask(xb, mask)
        else:
            logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)
        preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        trues.append(yb.cpu().numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)
    acc = accuracy_score(y_true, y_pred)
    return acc, y_true, y_pred, total_loss / max(n, 1)


@torch.no_grad()
def evaluate_classifier_safe(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_mask: bool = False,
) -> Tuple[float, np.ndarray, np.ndarray, float]:
    """评估；CUDA OOM 时自动回退 CPU（与 1117 脚本一致）。"""
    try:
        return evaluate_classifier(model, loader, device, use_mask=use_mask)
    except RuntimeError as e:
        err = str(e).lower()
        if "cuda" not in err and "out of memory" not in err:
            raise
        logger.warning("评估 CUDA 错误，回退 CPU: %s", e)
        cpu = torch.device("cpu")
        model_cpu = model.to(cpu)
        try:
            return evaluate_classifier(model_cpu, loader, cpu, use_mask=use_mask)
        finally:
            model.to(device)


@dataclass
class EarlyStoppingTracker:
    patience: int
    min_delta: float = 0.0001
    best_acc: float = 0.0
    best_epoch: int = 0
    patience_counter: int = 0

    def update(self, val_acc: float, epoch: int) -> bool:
        """返回 True 表示验证集有改善（应保存 best）。"""
        if val_acc > self.best_acc + self.min_delta:
            self.best_acc = val_acc
            self.best_epoch = epoch
            self.patience_counter = 0
            return True
        self.patience_counter += 1
        return False

    def should_stop(self) -> bool:
        return self.patience_counter >= self.patience


def setup_dual_logging(model_dir: Path, log_filename: str) -> None:
    """文件 + 终端双通道日志。"""
    model_dir.mkdir(parents=True, exist_ok=True)
    log_path = model_dir / log_filename
    root = logging.getLogger()
    if any(
        isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(log_path.resolve())
        for h in root.handlers
    ):
        return
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.handlers.clear()
    root.setLevel(logging.INFO)
    root.addHandler(fh)
    root.addHandler(sh)


def log_epoch(
    epoch: int,
    n_epochs: int,
    train_loss: float,
    val_acc: float,
    lr: float,
    stop: EarlyStoppingTracker,
    *,
    improved: bool,
    phase: str = "Train",
) -> None:
    if improved:
        logger.info(
            "%s Epoch %02d/%d loss=%.4f val_acc=%.2f%% lr=%.2e * (best, saved)",
            phase,
            epoch,
            n_epochs,
            train_loss,
            val_acc * 100,
            lr,
        )
    else:
        logger.info(
            "%s Epoch %02d/%d loss=%.4f val_acc=%.2f%% lr=%.2e "
            "(best: %.2f%% @ ep %d, patience: %d/%d)",
            phase,
            epoch,
            n_epochs,
            train_loss,
            val_acc * 100,
            lr,
            stop.best_acc * 100,
            stop.best_epoch,
            stop.patience_counter,
            stop.patience,
        )


def save_lr_history(lr_history: Sequence[float], out_dir: Path) -> None:
    if not lr_history:
        return
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    lr_df = pd.DataFrame({"epoch": range(1, len(lr_history) + 1), "learning_rate": list(lr_history)})
    lr_df.to_csv(out_dir / "learning_rate_history.csv", index=False)
    plt.figure(figsize=(10, 6))
    plt.plot(lr_df["epoch"], lr_df["learning_rate"], linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Learning Rate", fontsize=12)
    plt.title("Learning Rate Schedule", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(out_dir / "learning_rate_schedule.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("学习率曲线: %s", out_dir / "learning_rate_schedule.png")


def save_config_file(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("配置已保存: %s", path)


def extract_band_codes_from_varlen(data_list: Sequence) -> np.ndarray:
    """从 (3,L) 序列第 3 通道读取 band_code（与 e2e 编码一致）。"""
    codes = [int(np.round(float(np.asarray(d)[2, 0]))) for d in data_list]
    return np.asarray(codes, dtype=np.int32)


def plot_band_code_vs_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    band_codes: np.ndarray,
    out_dir: Path,
    *,
    title: str = "Finetuned: band_code vs accuracy",
    csv_name: str = "band_code_vs_accuracy.csv",
    png_name: str = "band_code_vs_accuracy.png",
    dpi: int = 300,
):
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    df_plot = pd.DataFrame(
        {"band_code": band_codes, "correct": (y_true == y_pred).astype(np.float64)}
    )
    g = (
        df_plot.groupby("band_code", as_index=False)["correct"]
        .mean()
        .rename(columns={"correct": "accuracy"})
        .sort_values("band_code")
    )
    g.to_csv(out_dir / csv_name, index=False)
    plt.figure(figsize=(10, 6))
    plt.plot(g["band_code"], g["accuracy"], marker="o")
    plt.xlabel("band_code")
    plt.ylabel("accuracy")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / png_name, dpi=dpi, bbox_inches="tight")
    plt.close()
    logger.info("已保存 %s / %s", out_dir / csv_name, out_dir / png_name)
    return g


def finalize_test_outputs(
    model: nn.Module,
    model_dir: Path,
    results_dir: Path,
    test_loader: DataLoader,
    device: torch.device,
    classes: List[str],
    history: Dict[str, List[float]],
    label_encoder,
    config_lines: Sequence[str],
    *,
    config_name: str = "training_config.txt",
    use_mask: bool = False,
    test_lengths: Optional[np.ndarray] = None,
    band_codes: Optional[np.ndarray] = None,
    lr_history: Optional[Sequence[float]] = None,
    save_last: bool = True,
    num_points_title: str = "Finetuned: num_points vs accuracy",
    band_title: str = "Finetuned: band_code vs accuracy",
) -> Dict[str, float]:
    """测试集评估 + 与 1117 对齐的产物（指标、曲线、num_points、band_code、last 模型）。"""
    import joblib

    if save_last:
        torch.save(model.state_dict(), model_dir / "last_model.pth")

    test_acc, y_true, y_pred, _ = evaluate_classifier_safe(
        model, test_loader, device, use_mask=use_mask
    )
    metrics = save_eval_results(y_true, y_pred, classes, results_dir, prefix="test_")
    (results_dir / "metrics.txt").write_text(
        (results_dir / "test_metrics.txt").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    save_json_metrics(results_dir / "test_metrics.json", metrics)
    try:
        save_training_curves(history, results_dir)
    except Exception as e:
        logger.warning("训练曲线保存失败: %s", e)
    if lr_history:
        try:
            save_lr_history(lr_history, results_dir)
        except Exception as e:
            logger.warning("学习率曲线保存失败: %s", e)
    if test_lengths is not None:
        try:
            plot_num_points_vs_accuracy(
                y_true, y_pred, test_lengths, results_dir, title=num_points_title
            )
        except Exception as e:
            logger.warning("num_points 图保存失败: %s", e)
    if band_codes is not None:
        try:
            plot_band_code_vs_accuracy(
                y_true, y_pred, band_codes, results_dir, title=band_title
            )
        except Exception as e:
            logger.warning("band_code 图保存失败: %s", e)
    joblib.dump(label_encoder, model_dir / "label_encoder.pkl")
    lines = list(config_lines) + [
        f"test_acc={metrics['accuracy']:.4f}",
        f"test_macro_f1={metrics['macro_f1']:.4f}",
    ]
    save_config_file(model_dir / config_name, lines)
    logger.info("测试集准确率: %.4f, 输出目录: %s", test_acc, model_dir)
    return metrics


def save_training_curves(history: Dict[str, List[float]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = {k: v for k, v in history.items()}
    import pandas as pd

    pd.DataFrame(df).to_csv(out_dir / "training_history.csv", index=False)
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curves.png", dpi=200)
    plt.close()


def save_eval_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str],
    out_dir: Path,
    prefix: str = "",
) -> Dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, target_names=classes, digits=4)
    (out_dir / f"{prefix}metrics.txt").write_text(
        f"Accuracy: {acc:.4f}\nMacro-F1: {macro_f1:.4f}\n\n{report}",
        encoding="utf-8",
    )
    metrics_dict = {"accuracy": acc, "macro_f1": macro_f1}
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        try:
            import seaborn as sns

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=classes,
                yticklabels=classes,
            )
        except ImportError:
            plt.imshow(cm, interpolation="nearest", cmap="Blues")
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45, ha="right")
            plt.yticks(tick_marks, classes)
            thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(
                        j,
                        i,
                        format(cm[i, j], "d"),
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > thresh else "black",
                    )
        plt.title(f"Confusion Matrix {prefix}")
        plt.tight_layout()
        plt.savefig(out_dir / f"{prefix}confusion_matrix.png", dpi=200)
        plt.close()
    except Exception as e:
        logger.warning("混淆矩阵绘图失败（指标文本已保存）: %s", e)
    return metrics_dict


def save_json_metrics(path: Path, metrics: dict) -> None:
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def plot_num_points_vs_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_points: np.ndarray,
    out_dir: Path,
    *,
    title: str = "num_points vs accuracy",
    csv_name: str = "num_points_vs_accuracy.csv",
    png_name: str = "num_points_vs_accuracy.png",
    dpi: int = 300,
):
    """按观测点数统计测试集准确率并保存 CSV/PNG。"""
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    num_points = np.asarray(num_points).astype(int)
    df_plot = pd.DataFrame(
        {"num_points": num_points, "correct": (y_true == y_pred).astype(np.float64)}
    )
    g = (
        df_plot.groupby("num_points", as_index=False)["correct"]
        .mean()
        .rename(columns={"correct": "accuracy"})
        .sort_values("num_points")
    )
    g.to_csv(out_dir / csv_name, index=False)

    acc_min = float(g["accuracy"].min())
    acc_max = float(g["accuracy"].max())
    y_lo = 0.5 if acc_min >= 0.45 else max(0.0, acc_min - 0.05)
    y_hi = min(1.0, acc_max + 0.05)
    if y_hi - y_lo < 0.15:
        y_hi = min(1.0, y_lo + 0.15)

    plt.figure(figsize=(10, 6))
    plt.plot(g["num_points"], g["accuracy"], marker="o")
    plt.xlabel("num_points")
    plt.ylabel("accuracy")
    plt.title(title)
    plt.ylim(y_lo, y_hi)
    step = 0.1 if y_lo >= 0.4 else 0.05
    plt.yticks(np.arange(np.ceil(y_lo / step) * step, y_hi + step * 0.5, step))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / png_name, dpi=dpi, bbox_inches="tight")
    plt.close()
    logger.info("已保存 %s / %s", out_dir / csv_name, out_dir / png_name)
    return g
