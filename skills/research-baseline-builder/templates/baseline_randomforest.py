"""
RandomForest 表格数据基线模板
=================================
适用场景：
    - 表格/结构化数据（pandas DataFrame）的分类或回归任务
    - 快速搭建可跑通的强基线，作为后续模型对比的标杆
    - 需要特征重要性以支持可解释性需求
    - 特征工程未完善、特征量纲不一致、存在缺失值时的稳健基线

数据接口：
    X : pandas.DataFrame, shape (n_samples, n_features)
    y : pandas.Series 或 numpy.ndarray, shape (n_samples,)
    分类任务：y 为离散标签；回归任务：y 为连续值

依赖：scikit-learn, pandas, numpy, joblib
用法：
    python baseline_randomforest.py                 # 用 sklearn 自带 breast_cancer 跑演示
    python baseline_randomforest.py --task regression  # 用 california_housing 跑回归演示
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score, train_test_split


# ---------- 0. 配置（修改这里即可） ----------
CONFIG = {
    "task": "classification",          # "classification" 或 "regression"
    "test_size": 0.2,
    "random_state": 42,
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_split": 2,
    "n_jobs": -1,
    "cv": 5,
    "model_save_path": Path("./rf_model.joblib"),
    "metrics_path": Path("./metrics.json"),
    "summary_path": Path("./baseline_summary.json"),
    "log_path": Path("./train_log.txt"),
}


# ---------- 1. 数据加载（示例用 sklearn 数据集，替换成你自己的数据即可） ----------
def load_demo_data(task: str):
    """加载演示数据集。替换此函数以接入自己的数据。"""
    if task == "classification":
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="target")
        target_names = data.target_names
    else:
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="MedHouseVal")
        target_names = None
    return X, y, target_names


def load_csv_data(csv_path: Path, target: str):
    data = pd.read_csv(csv_path)
    if target not in data.columns:
        raise ValueError(f"target column not found: {target}")
    y = data[target]
    X = data.drop(columns=[target])
    X = pd.get_dummies(X, drop_first=False)
    target_names = [str(item) for item in sorted(y.dropna().unique())] if y.nunique() <= 20 else None
    return X, y, target_names


def emit(message: str, log_lines: list[str]) -> None:
    print(message)
    log_lines.append(message)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------- 2. 模型构建 ----------
def build_model(task: str, cfg: dict):
    """根据任务类型构建 RandomForest 模型。"""
    common = dict(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        min_samples_split=cfg["min_samples_split"],
        n_jobs=cfg["n_jobs"],
        random_state=cfg["random_state"],
    )
    if task == "classification":
        return RandomForestClassifier(**common)
    return RandomForestRegressor(**common)


# ---------- 3. 训练 & 评估 ----------
def train_and_evaluate(X: pd.DataFrame, y, cfg: dict, target_names=None, data_mode="demo", source=None):
    log_lines = []
    task = cfg["task"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["test_size"],
        random_state=cfg["random_state"],
        stratify=y if task == "classification" else None,
    )
    emit(f"[数据模式] {data_mode}：{'当前使用示例/合成数据，不代表用户真实数据结果。' if data_mode == 'demo' else '当前使用用户提供的数据文件。'}", log_lines)
    emit(f"[数据] train={X_train.shape}, test={X_test.shape}", log_lines)

    model = build_model(task, cfg)

    # 交叉验证
    scoring = "accuracy" if task == "classification" else "r2"
    cv_scores = cross_val_score(model, X_train, y_train, cv=cfg["cv"],
                                scoring=scoring, n_jobs=cfg["n_jobs"])
    emit(f"[CV] {cfg['cv']}-fold {scoring}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}", log_lines)

    # 全量训练
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 评估
    if task == "classification":
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        report_text = classification_report(y_test, y_pred, target_names=target_names)
        emit(f"[Test] accuracy={acc:.4f}", log_lines)
        emit(report_text, log_lines)
        metrics = {"test_accuracy": float(acc), "classification_report": report}
    else:
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        emit(f"[Test] RMSE={rmse:.4f}, R2={r2:.4f}", log_lines)
        metrics = {"test_rmse": float(rmse), "test_r2": float(r2)}

    # 特征重要性
    importances = pd.Series(model.feature_importances_, index=X.columns)
    top = importances.sort_values(ascending=False).head(15)
    emit("[Top 特征重要性]", log_lines)
    emit(top.to_string(), log_lines)

    # 保存模型
    cfg["model_save_path"].parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, cfg["model_save_path"])
    emit(f"[保存] 模型已保存到 {cfg['model_save_path']}", log_lines)
    summary = {
        "schema": "research_baseline_summary_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_mode": data_mode,
        "data_source": str(source) if source else None,
        "model": "RandomForest",
        "task": task,
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "cv": {"folds": cfg["cv"], "scoring": scoring, "mean": float(cv_scores.mean()), "std": float(cv_scores.std())},
        "metrics": metrics,
        "top_features": [{"field": str(k), "importance": float(v)} for k, v in top.items()],
        "model_path": str(cfg["model_save_path"]),
        "warning": "当前为 demo 数据，不代表用户真实数据结果。" if data_mode == "demo" else None,
    }
    write_json(cfg["metrics_path"], metrics)
    write_json(cfg["summary_path"], summary)
    emit(f"[记录] metrics={cfg['metrics_path']} summary={cfg['summary_path']} log={cfg['log_path']}", log_lines)
    cfg["log_path"].write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return model, summary


# ---------- 4. 入口 ----------
def main():
    parser = argparse.ArgumentParser(description="RandomForest 表格基线")
    parser.add_argument("--task", choices=["classification", "regression"],
                        default=CONFIG["task"])
    parser.add_argument("--csv", type=Path, help="可选：用户 CSV 数据路径。")
    parser.add_argument("--target", help="CSV 中的目标列名。")
    args = parser.parse_args()
    cfg = {**CONFIG, "task": args.task}

    if args.csv:
        if not args.target:
            parser.error("--csv requires --target")
        X, y, target_names = load_csv_data(args.csv, args.target)
        train_and_evaluate(X, y, cfg, target_names=target_names, data_mode="user_csv", source=args.csv)
    else:
        X, y, target_names = load_demo_data(cfg["task"])
        train_and_evaluate(X, y, cfg, target_names=target_names, data_mode="demo")


if __name__ == "__main__":
    main()
