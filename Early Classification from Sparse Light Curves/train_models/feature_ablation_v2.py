#!/usr/bin/env python3
"""
Rigorous feature ablation study with consistent conditions.
All experiments use identical subsampling and hyperparameters.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_utils import load_split_feature_bundle

OUT_DIR = Path("/root/shared-nvme/train_models/results/feature_ablation_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Fixed hyperparameters for fair comparison
FIXED_PARAMS = {
    "n_estimators": 500,
    "max_depth": 10,
    "learning_rate": 0.2,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 4,
    "random_state": 42,
    "objective": "multi:softprob",
    "num_class": 7,
    "tree_method": "hist",
    "n_jobs": -1,
}

# Feature categories from 1117 model
LS_FEATURES = [
    "ls_power_to_median", "ls_entropy", "ls_harmonic_ratio",
    "ls_significant_peaks", "ls_peak_width", "ls_max_power",
    "ls_median_power", "ls_power_variance", "ls_dominant_period", "ls_fap",
]
IS_MISSING_FEATURES = [
    "ls_entropy_is_missing", "ls_max_power_is_missing",
    "ls_median_power_is_missing", "ls_power_variance_is_missing",
    "ls_fap_is_missing", "skewness_is_missing", "kurtosis_is_missing",
]
IS_PERIODIC = ["is_periodic"]

# Top 15 from 1117 importance (including 2 LS features)
TOP15_FROM_1117 = [
    "ndethist", "w_std_mag", "excess_variance", "is_periodic",
    "ls_dominant_period", "std_mag", "chi2_dof", "ls_peak_width",
    "autocorr_lag1", "skewness", "ls_fap", "num_points", "max_mag",
    "min_mag", "percent_amplitude",
]


def run_experiment(X_tr, y_tr, X_te, y_te, feature_names, exp_name):
    """Run XGBoost with fixed params and return results."""
    logger.info(f"  {exp_name}: {len(feature_names)} features")
    model = xgb.XGBClassifier(**FIXED_PARAMS)
    model.fit(X_tr, y_tr, verbose=False)
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
    _, _, f1_cls, _ = precision_recall_fscore_support(y_te, y_pred, zero_division=0)
    result = {
        "experiment": exp_name,
        "n_features": len(feature_names),
        "features": feature_names,
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(f1), 4),
    }
    with open(OUT_DIR / f"{exp_name}.json", "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"    Acc={acc:.4f}  F1={f1:.4f}")
    return result


def analyze_feature_correlation(bundle):
    """Analyze correlation between LS features and other features."""
    fi_1117 = pd.read_csv("/root/shared-nvme/train_models/xgboost_optuna_1117/results/feature_importance.csv")
    fi_1121 = pd.read_csv("/root/shared-nvme/train_models/xgboost_optuna_1121/results/less_feature_feature_importance.csv")

    logger.info("\n=== Feature Importance Comparison ===")
    logger.info("1117 top 10:")
    for _, row in fi_1117.head(10).iterrows():
        logger.info(f"  {row['feature']:30s} {row['importance']:.6f}")
    logger.info("\n1121 top 10:")
    for _, row in fi_1121.head(10).iterrows():
        logger.info(f"  {row['feature']:30s} {row['importance']:.6f}")

    # Check correlation between LS features and top non-LS features
    ls_feats = [f for f in bundle.feature_cols if f in LS_FEATURES]
    top_stat_feats = [f for f in ["ndethist", "w_std_mag", "excess_variance", "std_mag",
                                    "chi2_dof", "autocorr_lag1", "skewness"] if f in bundle.feature_cols]

    if ls_feats and top_stat_feats:
        ls_idx = [bundle.feature_cols.index(f) for f in ls_feats]
        stat_idx = [bundle.feature_cols.index(f) for f in top_stat_feats]
        X = bundle.X_test
        corr_matrix = np.corrcoef(X[:, ls_idx].T, X[:, stat_idx].T)
        n_ls = len(ls_feats)
        logger.info("\n=== Correlation: LS features vs Top statistical features ===")
        for i, ls_f in enumerate(ls_feats):
            for j, st_f in enumerate(top_stat_feats):
                c = corr_matrix[i, n_ls + j]
                if abs(c) > 0.1:
                    logger.info(f"  {ls_f:30s} <-> {st_f:20s}: r={c:.4f}")

    return fi_1117, fi_1121


def main():
    logger.info("=" * 60)
    logger.info("Feature Ablation Study v2 (Consistent Conditions)")
    logger.info("=" * 60)

    bundle = load_split_feature_bundle(pool="varlen")
    all_cols = bundle.feature_cols
    logger.info(f"Total features: {len(all_cols)}, test samples: {len(bundle.X_test)}")

    # Subsample training data for speed
    rng = np.random.RandomState(42)
    sub_idx = rng.choice(len(bundle.X_train), size=50000, replace=False)
    X_tr, y_tr = bundle.X_train[sub_idx], bundle.y_train[sub_idx]
    X_te, y_te = bundle.X_test, bundle.y_test

    results = []

    # Exp 1: All 57 features (1117 baseline)
    all_mask = np.ones(len(all_cols), dtype=bool)
    r = run_experiment(X_tr, y_tr, X_te, y_te, all_cols, "exp1_all_57")
    results.append(r)

    # Exp 2: 39 features without LS/is_missing/periodic (1121)
    remove_set = set(LS_FEATURES + IS_MISSING_FEATURES + IS_PERIODIC)
    no_ls_mask = np.array([f not in remove_set for f in all_cols])
    no_ls_cols = [f for f, m in zip(all_cols, no_ls_mask) if m]
    r = run_experiment(X_tr[:, no_ls_mask], y_tr, X_te[:, no_ls_mask], y_te,
                       no_ls_cols, "exp2_39_no_ls")
    results.append(r)

    # Exp 3: Top 15 from 1117 importance (includes 2 LS features)
    top15_mask = np.array([f in TOP15_FROM_1117 for f in all_cols])
    top15_cols = [f for f, m in zip(all_cols, top15_mask) if m]
    r = run_experiment(X_tr[:, top15_mask], y_tr, X_te[:, top15_mask], y_te,
                       top15_cols, "exp3_top15_from_1117")
    results.append(r)

    # Exp 4: Top 15 WITHOUT LS features + 39 non-LS features
    # This tests: keep the 2 LS features from top-15, but also keep all 39 non-LS
    top15_no_ls = [f for f in TOP15_FROM_1117 if f not in LS_FEATURES]
    combined_41 = list(set(top15_no_ls + no_ls_cols))
    combined_41_mask = np.array([f in combined_41 for f in all_cols])
    combined_41_cols = [f for f, m in zip(all_cols, combined_41_mask) if m]
    r = run_experiment(X_tr[:, combined_41_mask], y_tr, X_te[:, combined_41_mask], y_te,
                       combined_41_cols, "exp4_top15_no_ls_plus_39")
    results.append(r)

    # Exp 5: Only LS features
    ls_mask = np.array([f in LS_FEATURES for f in all_cols])
    ls_cols = [f for f, m in zip(all_cols, ls_mask) if m]
    if ls_mask.sum() > 0:
        r = run_experiment(X_tr[:, ls_mask], y_tr, X_te[:, ls_mask], y_te,
                           ls_cols, "exp5_ls_only")
        results.append(r)

    # Exp 6: Only is_missing
    miss_mask = np.array([f in IS_MISSING_FEATURES for f in all_cols])
    miss_cols = [f for f, m in zip(all_cols, miss_mask) if m]
    if miss_mask.sum() > 0:
        r = run_experiment(X_tr[:, miss_mask], y_tr, X_te[:, miss_mask], y_te,
                           miss_cols, "exp6_missing_only")
        results.append(r)

    # Exp 7: 39 non-LS features + 2 best LS features (ls_dominant_period, ls_peak_width)
    best_2_ls = ["ls_dominant_period", "ls_peak_width"]
    combo_41 = no_ls_cols + best_2_ls
    combo_41_mask = np.array([f in combo_41 for f in all_cols])
    combo_41_cols = [f for f, m in zip(all_cols, combo_41_mask) if m]
    r = run_experiment(X_tr[:, combo_41_mask], y_tr, X_te[:, combo_41_mask], y_te,
                       combo_41_cols, "exp7_39_plus_2_best_ls")
    results.append(r)

    # Exp 8: is_periodic only
    periodic_mask = np.array([f in IS_PERIODIC for f in all_cols])
    periodic_cols = [f for f, m in zip(all_cols, periodic_mask) if m]
    if periodic_mask.sum() > 0:
        r = run_experiment(X_tr[:, periodic_mask], y_tr, X_te[:, periodic_mask], y_te,
                           periodic_cols, "exp8_is_periodic_only")
        results.append(r)

    # Analyze feature correlations
    analyze_feature_correlation(bundle)

    # Summary
    summary = pd.DataFrame(results)
    summary.to_csv(OUT_DIR / "ablation_summary_v2.csv", index=False)
    logger.info("\n=== Summary ===")
    logger.info(summary[["experiment", "n_features", "accuracy", "macro_f1"]].to_string(index=False))


if __name__ == "__main__":
    import sys
    main()
