#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost 分类器（1117 特征集版本，使用 Optuna 超参数优化）

数据：
- 特征文件: features/train2_1117_*_balanced.csv + test2_1117_*_balanced.csv
- 划分索引: data/split/varlen/ （75/10/15，15% 测试集不参与训练）

输出目录：/root/shared-nvme/train_models/xgboost_optuna_1117
生成：
- 最佳模型: xgboost_optuna_1117_best.json
- feature_importance.png/.csv
- confusion_matrix.png
- band_code_vs_accuracy.png/.csv（若存在band_code列）
- num_points_vs_accuracy.png/.csv
- test_predictions_with_probabilities.csv
- optuna_study.db（Optuna 研究数据库）
- training.log
"""

import logging
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path("/root/shared-nvme")
from split_utils import load_split_feature_bundle  # noqa: E402

MODEL_DIR = BASE_DIR / "train_models/xgboost_optuna_1117"
RESULTS_DIR = MODEL_DIR / "results"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(MODEL_DIR / "training.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Optuna 优化参数
N_TRIALS = 50  # Optuna 试验次数（减少以加快速度）
N_FOLDS = 3  # 交叉验证折数（减少以加快速度）
RANDOM_STATE = 42
OPTUNA_STUDY_NAME = "xgboost_1117_optuna"
# 数据采样（如果数据量太大，可以采样一部分进行快速测试）
USE_SAMPLING = False  # 设为 True 可以只使用部分数据
SAMPLE_SIZE = 100000  # 采样数量（如果 USE_SAMPLING=True）


def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """将指定列转换为数值类型"""
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_and_preprocess(path: Path):
    """加载并预处理数据"""
    df = pd.read_csv(path)
    if "category" not in df.columns:
        raise ValueError("缺少分类列 category")

    # 选择数值特征列：排除 file_path 和 category
    drop_cols = {"file_path", "category"}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # 数值化并清洗
    df = coerce_numeric(df, feature_cols)
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols)

    X = df[feature_cols].values
    y = df["category"].values

    return df, X, y, feature_cols


def objective(trial, X_train, y_train):
    """Optuna 目标函数：使用交叉验证评估模型性能"""
    logger.info(f"\n[Trial {trial.number}] 开始试验...")

    # 超参数搜索空间
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 10.0, log=True),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "tree_method": "auto",
        "missing": np.nan,
        "verbosity": 0,
    }

    logger.info(
        f"[Trial {trial.number}] 参数: n_estimators={params['n_estimators']}, "
        f"max_depth={params['max_depth']}, lr={params['learning_rate']:.4f}"
    )

    # 交叉验证
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        logger.info(f"[Trial {trial.number}] Fold {fold_idx + 1}/{N_FOLDS} 开始...")
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model = XGBClassifier(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        cv_scores.append(acc)
        logger.info(
            f"[Trial {trial.number}] Fold {fold_idx + 1}/{N_FOLDS} 完成, "
            f"准确率: {acc:.4f}"
        )

    mean_acc = np.mean(cv_scores)
    logger.info(f"[Trial {trial.number}] 完成! 平均准确率: {mean_acc:.4f}")
    return mean_acc


def plot_confusion_matrix(y_true, y_pred, classes, out_path: Path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
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
    plt.title("Confusion Matrix - XGBoost (Optuna Optimized, 1117)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_importance(
    model: XGBClassifier,
    feature_names: List[str],
    out_png: Path,
    out_csv: Path,
):
    """绘制特征重要性"""
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    names = [feature_names[i] for i in order]
    vals = importances[order]

    # 保存 CSV
    importance_df = pd.DataFrame({"feature": names, "importance": vals})
    importance_df.to_csv(out_csv, index=False)

    # 绘制图表（只显示前30个）
    top_n = min(30, len(names))
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), vals[:top_n][::-1])
    plt.yticks(range(top_n), names[:top_n][::-1])
    plt.xlabel("Importance")
    plt.title(f"Feature Importance (Top {top_n}) - XGBoost (Optuna)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"特征重要性已保存: {out_csv}")


def plot_accuracy_by_band(df_true_pred: pd.DataFrame, out_csv: Path, out_png: Path):
    """绘制 band_code vs accuracy"""
    if "band_code" not in df_true_pred.columns:
        logger.warning("数据中无 band_code 列，跳过 band_code vs accuracy 分析")
        return

    df_true_pred["correct"] = (
        df_true_pred["true_label"] == df_true_pred["pred_label"]
    ).astype(int)
    band_acc = (
        df_true_pred.groupby("band_code")["correct"]
        .agg(["mean", "count"])
        .reset_index()
    )
    band_acc.columns = ["band_code", "accuracy", "count"]
    band_acc = band_acc.sort_values("band_code")

    # 保存 CSV
    band_acc.to_csv(out_csv, index=False)
    logger.info(f"band_code vs accuracy 已保存: {out_csv}")

    # 绘制图表
    plt.figure(figsize=(10, 6))
    plt.bar(band_acc["band_code"], band_acc["accuracy"], alpha=0.7)
    plt.xlabel("Band Code")
    plt.ylabel("Accuracy")
    plt.title("Accuracy by Band Code")
    plt.xticks(band_acc["band_code"])
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def plot_accuracy_by_numpoints(
    df_true_pred: pd.DataFrame, out_csv: Path, out_png: Path
):
    """绘制 num_points vs accuracy"""
    if "num_points" not in df_true_pred.columns:
        logger.warning("数据中无 num_points 列，跳过 num_points vs accuracy 分析")
        return

    df_true_pred["correct"] = (
        df_true_pred["true_label"] == df_true_pred["pred_label"]
    ).astype(int)

    # 将 num_points 分组（每10个点一组）
    df_true_pred["num_points_bin"] = (df_true_pred["num_points"] // 10) * 10

    numpoints_acc = (
        df_true_pred.groupby("num_points_bin")["correct"]
        .agg(["mean", "count"])
        .reset_index()
    )
    numpoints_acc.columns = ["num_points", "accuracy", "count"]
    numpoints_acc = numpoints_acc[numpoints_acc["count"] >= 10]  # 至少10个样本
    numpoints_acc = numpoints_acc.sort_values("num_points")

    # 保存 CSV
    numpoints_acc.to_csv(out_csv, index=False)
    logger.info(f"num_points vs accuracy 已保存: {out_csv}")

    # 绘制图表
    plt.figure(figsize=(12, 6))
    plt.plot(
        numpoints_acc["num_points"],
        numpoints_acc["accuracy"],
        marker="o",
        linewidth=2,
        markersize=6,
    )
    plt.xlabel("Number of Points (binned)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy by Number of Points")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

"""
def print_classification_metrics(y_true, y_pred, classes):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )

    logger.info("\n" + "=" * 80)
    logger.info("分类指标汇总")
    logger.info("=" * 80)
    logger.info(f"总体准确率: {accuracy * 100:.2f}%")
    logger.info("\n各类别指标:")
    logger.info(
        f"{'类别':<15} {'精确率':<10} {'召回率':<10} " f"{'F1分数':<10} {'样本数':<10}"
    )
    logger.info("-" * 60)

    for i, cls in enumerate(classes):
        logger.info(
            f"{cls:<15} {precision[i]:<10.4f} {recall[i]:<10.4f} "
            f"{f1[i]:<10.4f} {support[i]:<10}"
        )

    logger.info("\n详细分类报告:")
    logger.info(classification_report(y_true, y_pred, labels=classes, zero_division=0))
"""

def print_classification_metrics(y_true, y_pred, classes):
    """打印分类指标"""
    accuracy = accuracy_score(y_true, y_pred)
    
    # 修复：把 labels=classes 改成 labels=list(range(len(classes)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(classes))), zero_division=0
    )

    logger.info("\n" + "=" * 80)
    logger.info("分类指标汇总")
    logger.info("=" * 80)
    logger.info(f"总体准确率: {accuracy * 100:.2f}%")
    logger.info("\n各类别指标:")
    logger.info(f"{'类别':<15} {'精确率':<10} {'召回率':<10} {'F1分数':<10} {'样本数':<10}")
    logger.info("-" * 60)

    for i, cls in enumerate(classes):
        logger.info(f"{cls:<15} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1[i]:<10.4f} {support[i]:<10}")

    logger.info("\n详细分类报告:")
    logger.info(classification_report(y_true, y_pred, labels=list(range(len(classes))), target_names=classes, zero_division=0))

def main():
    logger.info("=" * 80)
    logger.info("XGBoost 分类器训练（Optuna 超参数优化，1117特征集）")
    logger.info("=" * 80)
    logger.info("数据: 统一划分 + features_1117")
    logger.info(f"Optuna 试验次数: {N_TRIALS}")
    logger.info(f"交叉验证折数: {N_FOLDS}")

    bundle = load_split_feature_bundle(pool="varlen")
    X_train, y_train_encoded = bundle.X_train, bundle.y_train
    X_val, y_val_encoded = bundle.X_val, bundle.y_val
    X_test, y_test_encoded = bundle.X_test, bundle.y_test
    train_df, test_df = bundle.train_df, bundle.test_df
    label_encoder = bundle.label_encoder
    feature_cols = bundle.feature_cols
    classes = bundle.classes
    logger.info(f"训练/验证/测试: {len(X_train)}/{len(X_val)}/{len(X_test)}, 特征数={len(feature_cols)}")

    logger.info(f"\n类别数量: {len(classes)}")
    logger.info(f"类别列表: {list(classes)}")

    # 数据采样（可选，用于快速测试）
    if USE_SAMPLING and len(X_train) > SAMPLE_SIZE:
        logger.info(f"\n使用数据采样: {SAMPLE_SIZE} 个样本（原始: {len(X_train)}）")
        from sklearn.model_selection import train_test_split

        X_train, _, y_train_encoded, _ = train_test_split(
            X_train,
            y_train_encoded,
            train_size=SAMPLE_SIZE,
            stratify=y_train_encoded,
            random_state=RANDOM_STATE,
        )
        logger.info(f"采样后训练集大小: {len(X_train)}")

    # Optuna 超参数优化
    logger.info("\n" + "=" * 80)
    logger.info("开始 Optuna 超参数优化...")
    logger.info("=" * 80)

    study = optuna.create_study(
        direction="maximize",
        study_name=OPTUNA_STUDY_NAME,
        storage=f"sqlite:///{MODEL_DIR / 'optuna_study.db'}",
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(trial, X_train, y_train_encoded),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    logger.info("\n" + "=" * 80)
    logger.info("Optuna 优化完成")
    logger.info("=" * 80)
    logger.info(f"最佳试验: {study.best_trial.number}")
    logger.info(f"最佳准确率: {study.best_value * 100:.2f}%")
    logger.info("\n最佳超参数:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")

    # 使用最佳参数训练最终模型
    logger.info("\n" + "=" * 80)
    logger.info("使用最佳参数训练最终模型...")
    logger.info("=" * 80)

    best_params = study.best_params.copy()
    best_params.update(
        {
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "tree_method": "auto",
            "missing": np.nan,
            "verbosity": 1,
        }
    )

    final_model = XGBClassifier(**best_params)
    final_model.fit(
        X_train,
        y_train_encoded,
        eval_set=[(X_val, y_val_encoded)],
        verbose=True,
    )

    # 保存模型
    model_file = MODEL_DIR / "xgboost_optuna_1117_best.json"
    final_model.save_model(str(model_file))
    logger.info(f"模型已保存: {model_file}")

    # 保存最佳参数
    best_params_file = MODEL_DIR / "best_params.txt"
    with open(best_params_file, "w", encoding="utf-8") as f:
        f.write("最佳超参数:\n")
        f.write("=" * 80 + "\n")
        for key, value in study.best_params.items():
            f.write(f"{key}: {value}\n")
        f.write(f"\n最佳交叉验证准确率: {study.best_value * 100:.2f}%\n")
    logger.info(f"最佳参数已保存: {best_params_file}")

    # 评估模型
    logger.info("\n" + "=" * 80)
    logger.info("评估模型性能...")
    logger.info("=" * 80)

    # 训练集预测
    train_pred = final_model.predict(X_train)
    train_acc = accuracy_score(y_train_encoded, train_pred)
    logger.info(f"训练集准确率: {train_acc * 100:.2f}%")

    # 测试集预测
    test_pred = final_model.predict(X_test)
    test_pred_labels = label_encoder.inverse_transform(test_pred)
    test_acc = accuracy_score(y_test_encoded, test_pred)
    from sklearn.metrics import f1_score
    import json

    macro_f1 = f1_score(y_test_encoded, test_pred, average="macro", zero_division=0)
    json.dump(
        {"accuracy": float(test_acc), "macro_f1": float(macro_f1)},
        open(RESULTS_DIR / "test_metrics.json", "w", encoding="utf-8"),
        indent=2,
    )
    logger.info(f"测试集准确率: {test_acc * 100:.2f}%")

    # 测试集概率
    test_proba = final_model.predict_proba(X_test)

    # 打印分类指标
    print_classification_metrics(y_test_encoded, test_pred, classes)

    # 保存预测结果
    predictions_df = test_df[["file_path", "category"]].copy()
    predictions_df["predicted_category"] = test_pred_labels
    predictions_df["true_label"] = y_test
    predictions_df["pred_label"] = test_pred_labels

    # 添加各类别概率
    for i, cls in enumerate(classes):
        predictions_df[f"prob_{cls}"] = test_proba[:, i]

    predictions_file = RESULTS_DIR / "test_predictions_with_probabilities.csv"
    predictions_df.to_csv(predictions_file, index=False)
    logger.info(f"预测结果已保存: {predictions_file}")

    # 绘制图表
    logger.info("\n生成可视化图表...")

    # 混淆矩阵
    plot_confusion_matrix(
        y_test_encoded,
        test_pred,
        classes,
        RESULTS_DIR / "confusion_matrix.png",
    )

    # 特征重要性
    plot_feature_importance(
        final_model,
        feature_cols,
        RESULTS_DIR / "feature_importance.png",
        RESULTS_DIR / "feature_importance.csv",
    )

    # band_code vs accuracy
    if "band_code" in test_df.columns:
        predictions_df["band_code"] = test_df["band_code"].values
        plot_accuracy_by_band(
            predictions_df,
            RESULTS_DIR / "band_code_vs_accuracy.csv",
            RESULTS_DIR / "band_code_vs_accuracy.png",
        )

    # num_points vs accuracy
    if "num_points" in test_df.columns:
        predictions_df["num_points"] = test_df["num_points"].values
        plot_accuracy_by_numpoints(
            predictions_df,
            RESULTS_DIR / "num_points_vs_accuracy.csv",
            RESULTS_DIR / "num_points_vs_accuracy.png",
        )

    logger.info("\n" + "=" * 80)
    logger.info("训练完成！")
    logger.info("=" * 80)
    logger.info(f"模型文件: {model_file}")
    logger.info(f"结果目录: {RESULTS_DIR}")
    logger.info(f"Optuna 研究数据库: {MODEL_DIR / 'optuna_study.db'}")


if __name__ == "__main__":
    main()
