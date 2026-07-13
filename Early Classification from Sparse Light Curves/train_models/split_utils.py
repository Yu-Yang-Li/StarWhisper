#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从「原有特征 CSV + data/split 索引」加载 train/val/test。
不改变特征文件内容，只在行索引上重新划分。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)

BASE_DIR = Path("/root/shared-nvme")
SPLIT_ROOT = BASE_DIR / "data/split"

FEATURE_POOLS = {
    "50obs": {
        "train_csv": BASE_DIR / "features/train4_1117_20251119_204347_balanced.csv",
        "test_csv": BASE_DIR / "features/test4_1117_20251119_204347_balanced.csv",
    },
    "varlen": {
        "train_csv": BASE_DIR / "features/train2_1117_20251117_235357_balanced.csv",
        "test_csv": BASE_DIR / "features/test2_1117_20251117_235357_balanced.csv",
    },
    "1121": {
        "train_csv": BASE_DIR / "features/train2_1121_20251121_121804_balanced.csv",
        "test_csv": BASE_DIR / "features/test2_1121_20251121_121804_balanced.csv",
    },
}


@dataclass
class SplitBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    label_encoder: LabelEncoder
    classes: List[str]
    scaler: StandardScaler
    feature_cols: List[str]
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame


def _load_pool_dataframe(pool: str) -> pd.DataFrame:
    cfg = FEATURE_POOLS[pool]
    train_df = pd.read_csv(cfg["train_csv"], low_memory=False)
    test_df = pd.read_csv(cfg["test_csv"], low_memory=False)
    df = pd.concat([train_df, test_df], ignore_index=True)
    if "file_path" in df.columns:
        df = df.drop_duplicates(subset=["file_path"], keep="first")
    return df.reset_index(drop=True)


def _load_indices(pool: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = SPLIT_ROOT / pool
    return (
        np.load(d / "train_indices.npy"),
        np.load(d / "val_indices.npy"),
        np.load(d / "test_indices.npy"),
    )


def load_split_feature_bundle(
    pool: str = "varlen",
    drop_cols: Optional[Set[str]] = None,
    clip_value: float = 5.0,
) -> SplitBundle:
    """
    pool: 'varlen' (train2+test2) | '50obs' (train4+test4) | '1121' (train4_1121+test4_1121)
    """
    if pool not in FEATURE_POOLS:
        raise ValueError(f"未知 pool: {pool}")
    if drop_cols is None:
        drop_cols = {"file_path", "category", "raw_category", "_src", "index"}

    df = _load_pool_dataframe(pool)
    idx_train, idx_val, idx_test = _load_indices(pool)

    exclude = set(drop_cols)
    feature_cols = [
        c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]

    def pack(indices: np.ndarray):
        sub = df.iloc[indices].copy()
        X = (
            sub[feature_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .values.astype(np.float32)
        )
        return X, sub["category"].values, sub

    X_train, y_train_raw, train_df = pack(idx_train)
    X_val, y_val_raw, val_df = pack(idx_val)
    X_test, y_test_raw, test_df = pack(idx_test)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_val = le.transform(y_val_raw)
    y_test = le.transform(y_test_raw)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    if clip_value is not None:
        X_train = np.clip(X_train, -clip_value, clip_value)
        X_val = np.clip(X_val, -clip_value, clip_value)
        X_test = np.clip(X_test, -clip_value, clip_value)

    X_train = np.nan_to_num(X_train)
    X_val = np.nan_to_num(X_val)
    X_test = np.nan_to_num(X_test)

    return SplitBundle(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        label_encoder=le,
        classes=list(le.classes_),
        scaler=scaler,
        feature_cols=feature_cols,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )


def to_transformer_train_bundle(sb: SplitBundle):
    """训练循环里 X_test 指向验证集；最终评估用 sb.X_test。"""

    @dataclass
    class DataBundle:
        train_df: pd.DataFrame
        test_df: pd.DataFrame
        X_train: np.ndarray
        y_train: np.ndarray
        X_test: np.ndarray
        y_test: np.ndarray
        label_encoder: LabelEncoder
        classes: List[str]
        scaler: StandardScaler
        feature_cols: List[str]

    bundle = DataBundle(
        train_df=sb.train_df,
        test_df=sb.val_df,
        X_train=sb.X_train,
        y_train=sb.y_train,
        X_test=sb.X_val,
        y_test=sb.y_val,
        label_encoder=sb.label_encoder,
        classes=sb.classes,
        scaler=sb.scaler,
        feature_cols=sb.feature_cols,
    )
    return bundle, sb


# 端到端数据目录（与 prepare_e2e_data.py / prepare_e2e_varlen.py 输出一致）
E2E_DATA_DIRS = {
    "50obs": BASE_DIR / "data/e2e",
    "varlen": BASE_DIR / "data/e2e_varlen",
}


def load_e2e_arrays(split: str, pool: str = "50obs"):
    """加载固定形状 (N, 3, 50) 的 numpy 数组（pool=50obs）。"""
    d = E2E_DATA_DIRS.get(pool, BASE_DIR / "data/e2e")
    data_path = d / f"{split}_data.npy"
    labels_path = d / f"{split}_labels.npy"
    if not data_path.is_file():
        raise FileNotFoundError(
            f"缺少 {data_path}，请先运行: python3 data/prepare_e2e_data.py"
        )
    return np.load(data_path), np.load(labels_path)


def load_e2e_label_encoder(pool: str = "50obs"):
    import joblib

    d = E2E_DATA_DIRS.get(pool, BASE_DIR / "data/e2e")
    path = d / "label_encoder.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"缺少 {path}")
    return joblib.load(path)
