"""
XGBoost + Optuna 表格数据强基线（含超参搜索）
================================================
适用场景：表格/结构化数据强基线；中大规模数据；需要超参搜索拿到可汇报结果；类别不平衡稳健
数据接口：X: pandas.DataFrame (n_samples, n_features); y: pandas.Series (n_samples,)
依赖：xgboost>=2.0, optuna, scikit-learn, pandas, numpy, joblib
用法：
    python baseline_xgboost_optuna.py                 # 分类 demo（breast_cancer）
    python baseline_xgboost_optuna.py --task regression --n_trials 30
"""
from __future__ import annotations
import argparse, warnings
import json
from datetime import datetime
from pathlib import Path
import joblib, numpy as np, optuna, pandas as pd, xgboost as xgb
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------- 0. 配置 ----------
CONFIG = dict(task="classification", test_size=0.2, random_state=42,
              n_trials=50, cv=5, timeout=600,
              model_save_path=Path("./xgb_optuna_model.joblib"),
              metrics_path=Path("./metrics.json"),
              summary_path=Path("./baseline_summary.json"),
              log_path=Path("./train_log.txt"))


# ---------- 1. 数据加载 ----------
def load_demo(task):
    if task == "classification":
        from sklearn.datasets import load_breast_cancer
        d = load_breast_cancer()
        return pd.DataFrame(d.data, columns=d.feature_names), pd.Series(d.target)
    from sklearn.datasets import fetch_california_housing
    d = fetch_california_housing()
    return pd.DataFrame(d.data, columns=d.feature_names), pd.Series(d.target)


def load_csv(csv_path: Path, target: str):
    data = pd.read_csv(csv_path)
    if target not in data.columns:
        raise ValueError(f"target column not found: {target}")
    y = data[target]
    X = pd.get_dummies(data.drop(columns=[target]), drop_first=False)
    return X, y


def emit(message: str, log_lines: list[str]) -> None:
    print(message)
    log_lines.append(message)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------- 2. Optuna objective ----------
def make_objective(X, y, cfg):
    is_clf = cfg["task"] == "classification"
    n_cls = len(np.unique(y)) if is_clf else 1
    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 1000, step=50),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            gamma=trial.suggest_float("gamma", 0.0, 5.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            random_state=cfg["random_state"], tree_method="hist", verbosity=0,
        )
        if is_clf:
            params.update(objective="binary:logistic" if n_cls==2 else "multi:softmax")
            if n_cls > 2: params["num_class"] = n_cls
            Model, scoring = xgb.XGBClassifier, "accuracy"
            kf = StratifiedKFold(n_splits=cfg["cv"], shuffle=True, random_state=cfg["random_state"])
        else:
            Model, scoring = xgb.XGBRegressor, "r2"
            kf = KFold(n_splits=cfg["cv"], shuffle=True, random_state=cfg["random_state"])
        return cross_val_score(Model(**params), X, y, cv=kf, scoring=scoring, n_jobs=1).mean()
    return objective


# ---------- 3. 训练 & 评估 ----------
def train_eval(X, y, cfg, data_mode="demo", source=None):
    log_lines = []
    is_clf = cfg["task"] == "classification"
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg["test_size"], random_state=cfg["random_state"],
        stratify=y if is_clf else None)
    emit(f"[数据模式] {data_mode}：{'当前使用示例数据，不代表用户真实数据结果。' if data_mode == 'demo' else '当前使用用户提供的数据文件。'}", log_lines)
    emit(f"[数据] train={X_tr.shape} test={X_te.shape}", log_lines)
    study = optuna.create_study(direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg["random_state"]))
    study.optimize(make_objective(X_tr, y_tr, cfg), n_trials=cfg["n_trials"], timeout=cfg["timeout"])
    emit(f"[Optuna] best_cv={study.best_value:.4f}", log_lines)
    emit(f"[Optuna] best_params={study.best_params}", log_lines)

    bp = {**study.best_params, "random_state": cfg["random_state"],
          "tree_method":"hist", "verbosity":0}
    n_cls = len(np.unique(y_tr)) if is_clf else 1
    if is_clf:
        bp.update(objective="binary:logistic" if n_cls==2 else "multi:softmax")
        if n_cls > 2: bp["num_class"] = n_cls
        model = xgb.XGBClassifier(**bp)
    else:
        model = xgb.XGBRegressor(**bp)
    model.fit(X_tr, y_tr)
    yp = model.predict(X_te)
    if is_clf:
        acc = accuracy_score(y_te, yp)
        emit(f"[Test] accuracy={acc:.4f}", log_lines)
        metrics = {"test_accuracy": float(acc)}
    else:
        rmse = np.sqrt(mean_squared_error(y_te, yp))
        r2 = r2_score(y_te, yp)
        emit(f"[Test] RMSE={rmse:.4f} R2={r2:.4f}", log_lines)
        metrics = {"test_rmse": float(rmse), "test_r2": float(r2)}
    cfg["model_save_path"].parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "best_params": bp}, cfg["model_save_path"])
    emit(f"[保存] {cfg['model_save_path']}", log_lines)
    summary = {
        "schema": "research_baseline_summary_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_mode": data_mode,
        "data_source": str(source) if source else None,
        "model": "XGBoost+Optuna",
        "task": cfg["task"],
        "train_shape": list(X_tr.shape),
        "test_shape": list(X_te.shape),
        "optuna": {"best_cv": float(study.best_value), "best_params": study.best_params, "n_trials": len(study.trials)},
        "metrics": metrics,
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
    ap = argparse.ArgumentParser(description="XGBoost+Optuna 表格强基线")
    ap.add_argument("--task", choices=["classification","regression"], default=CONFIG["task"])
    ap.add_argument("--n_trials", type=int, default=CONFIG["n_trials"])
    ap.add_argument("--csv", type=Path, help="可选：用户 CSV 数据路径。")
    ap.add_argument("--target", help="CSV 中的目标列名。")
    args = ap.parse_args()
    cfg = {**CONFIG, "task": args.task, "n_trials": args.n_trials}
    if args.csv:
        if not args.target:
            ap.error("--csv requires --target")
        X, y = load_csv(args.csv, args.target)
        train_eval(X, y, cfg, data_mode="user_csv", source=args.csv)
    else:
        X, y = load_demo(cfg["task"])
        train_eval(X, y, cfg, data_mode="demo")

if __name__ == "__main__":
    main()
