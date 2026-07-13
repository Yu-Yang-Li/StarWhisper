#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为已训练的 XGBoost Optuna 1121 模型生成完整可视化

数据：原有 train2/test2 1121 特征 CSV + data/split/1121/ 中 15% 测试集索引
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_utils import FEATURE_POOLS, load_split_feature_bundle  # noqa: E402

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path("/root/shared-nvme")
POOL = "1121"
MODEL_DIR = BASE_DIR / "train_models/xgboost_reduced"
RESULTS_DIR = MODEL_DIR / "results"
MODEL_FILE = MODEL_DIR / "xgboost_reduced_best.json"
SPLIT_DIR = BASE_DIR / "data/split" / POOL

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(MODEL_DIR / "plot_results.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def plot_confusion_matrix(y_true, y_pred, classes, out_path: Path) -> None:
    labels = np.arange(len(classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix (1121, 15% test split)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("混淆矩阵已保存: %s", out_path)


def plot_feature_importance(
    model: XGBClassifier,
    feature_names: List[str],
    out_png: Path,
    out_csv: Path,
) -> None:
    importances = model.feature_importances_
    mask = np.array([name != "band_code" for name in feature_names])
    filtered_names = [n for n, m in zip(feature_names, mask) if m]
    filtered_importances = importances[mask]
    total = filtered_importances.sum()
    vals = filtered_importances / total if total > 0 else filtered_importances

    order = np.argsort(vals)[::-1]
    names = [filtered_names[i] for i in order]
    vals = vals[order]

    pd.DataFrame({"feature": names, "importance": vals}).to_csv(out_csv, index=False)

    top_n = min(30, len(names))
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), vals[:top_n][::-1])
    plt.yticks(range(top_n), names[:top_n][::-1])
    plt.xlabel("Importance (Normalized)")
    plt.title(f"Feature Importance Top {top_n} (1121)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("特征重要性已保存: %s", out_png)


def plot_accuracy_by_band(df: pd.DataFrame, out_csv: Path, out_png: Path) -> None:
    if "band_code" not in df.columns:
        logger.warning("无 band_code 列，跳过")
        return
    df = df.copy()
    df["correct"] = (df["true_label"] == df["pred_label"]).astype(int)
    band_acc = (
        df.groupby("band_code")["correct"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "accuracy"})
    )
    band_acc.to_csv(out_csv, index=False)
    plt.figure(figsize=(10, 6))
    plt.bar(band_acc["band_code"], band_acc["accuracy"], alpha=0.7)
    plt.xlabel("Band Code")
    plt.ylabel("Accuracy")
    plt.title("Accuracy by Band Code (1121)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def plot_accuracy_by_numpoints(df: pd.DataFrame, out_csv: Path, out_png: Path) -> None:
    if "num_points" not in df.columns:
        logger.warning("无 num_points 列，跳过")
        return
    df = df.copy()
    df["correct"] = (df["true_label"] == df["pred_label"]).astype(int)
    g = (
        df.groupby("num_points")["correct"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "accuracy"})
    )
    g.to_csv(out_csv, index=False)
    plt.figure(figsize=(12, 6))
    plt.plot(g["num_points"], g["accuracy"], marker="o", linewidth=1.5, markersize=3, alpha=0.6)
    plt.xlabel("Number of Points")
    plt.ylabel("Accuracy")
    plt.title("Accuracy by Number of Points (1121)")
    plt.grid(alpha=0.3)
    plt.ylim(0.5, 1.0)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def print_classification_metrics(y_true, y_pred, classes) -> None:
    acc = accuracy_score(y_true, y_pred)
    labels = np.arange(len(classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    logger.info("总体准确率: %.2f%%", acc * 100)
    for i, cls in enumerate(classes):
        logger.info(
            "%s P=%.4f R=%.4f F1=%.4f n=%d",
            cls,
            precision[i],
            recall[i],
            f1[i],
            support[i],
        )
    logger.info(
        "\n%s",
        classification_report(
            y_true, y_pred, labels=labels, target_names=classes, zero_division=0
        ),
    )


def main() -> None:
    logger.info("=" * 80)
    logger.info("XGBoost Optuna 1121 结果可视化（统一 75/10/15 划分）")
    logger.info("模型: %s", MODEL_FILE)
    logger.info("划分: %s", SPLIT_DIR)
    cfg = FEATURE_POOLS[POOL]
    logger.info("特征 CSV: %s + %s", cfg["train_csv"].name, cfg["test_csv"].name)

    if not MODEL_FILE.is_file():
        logger.error("模型不存在: %s", MODEL_FILE)
        sys.exit(1)

    bundle = load_split_feature_bundle(pool=POOL)
    logger.info(
        "测试集样本: %d (15%% hold-out), 特征数: %d",
        len(bundle.X_test),
        len(bundle.feature_cols),
    )

    model = XGBClassifier()
    model.load_model(str(MODEL_FILE))

    y_pred = model.predict(bundle.X_test)
    acc = accuracy_score(bundle.y_test, y_pred)
    logger.info("测试集准确率: %.2f%%", acc * 100)

    print_classification_metrics(bundle.y_test, y_pred, bundle.classes)

    pred_labels = bundle.label_encoder.inverse_transform(y_pred)
    true_labels = bundle.label_encoder.inverse_transform(bundle.y_test)
    proba = model.predict_proba(bundle.X_test)

    predictions_df = bundle.test_df[["file_path", "category"]].copy()
    predictions_df["predicted_category"] = pred_labels
    predictions_df["true_label"] = true_labels
    predictions_df["pred_label"] = pred_labels
    for i, cls in enumerate(bundle.classes):
        predictions_df[f"prob_{cls}"] = proba[:, i]
    if "band_code" in bundle.test_df.columns:
        predictions_df["band_code"] = bundle.test_df["band_code"].values
    if "num_points" in bundle.test_df.columns:
        predictions_df["num_points"] = bundle.test_df["num_points"].values

    predictions_df.to_csv(
        RESULTS_DIR / "test_predictions_with_probabilities.csv", index=False
    )

    plot_confusion_matrix(
        bundle.y_test,
        y_pred,
        bundle.classes,
        RESULTS_DIR / "less_feature_confusion_matrix.png",
    )
    plot_feature_importance(
        model,
        bundle.feature_cols,
        RESULTS_DIR / "less_feature_feature_importance.png",
        RESULTS_DIR / "less_feature_feature_importance.csv",
    )
    if "band_code" in predictions_df.columns:
        plot_accuracy_by_band(
            predictions_df,
            RESULTS_DIR / "less_feature_band_code_vs_accuracy.csv",
            RESULTS_DIR / "less_feature_band_code_vs_accuracy.png",
        )
    if "num_points" in predictions_df.columns:
        plot_accuracy_by_numpoints(
            predictions_df,
            RESULTS_DIR / "less_feature_num_points_vs_accuracy.csv",
            RESULTS_DIR / "less_feature_num_points_vs_accuracy.png",
        )

    import json

    from sklearn.metrics import f1_score

    json.dump(
        {
            "accuracy": float(acc),
            "macro_f1": float(
                f1_score(bundle.y_test, y_pred, average="macro", zero_division=0)
            ),
            "pool": POOL,
            "split_dir": str(SPLIT_DIR),
            "n_test": len(bundle.X_test),
        },
        open(RESULTS_DIR / "test_metrics.json", "w", encoding="utf-8"),
        indent=2,
    )

    logger.info("完成，结果目录: %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
