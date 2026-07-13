#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一数据划分（不生成新特征，不改变原有特征 CSV 文件）

在「特征提取阶段已合并的 train+test 池」上重新分层划分：
  75% 训练 / 10% 验证 / 15% 测试（random_state=42）

三个独立池（互不混用）：
  - 50obs : train4_1117_balanced + test4_1117_balanced（固定 50 点）
  - varlen: train2_1117_balanced + test2_1117_balanced（3–30 点）
  - 1121  : train2_1121_balanced + test2_1121_balanced（1121 特征版本，3-30点）

输出：
  data/split/50obs/{train,val,test}_indices.npy, manifest.csv, split_statistics.txt
  data/split/varlen/...
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
SPLIT_ROOT = SCRIPT_DIR / "split"

RANDOM_STATE = 42
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.75, 0.10, 0.15

# 与现有四个模型一致的特征文件（只读，不修改）
POOLS = {
    "50obs": {
        "train_csv": BASE_DIR / "features/train4_1117_20251119_204347_balanced.csv",
        "test_csv": BASE_DIR / "features/test4_1117_20251119_204347_balanced.csv",
        "desc": "train4+test4, 50 points",
    },
    "varlen": {
        "train_csv": BASE_DIR / "features/train2_1117_20251117_235357_balanced.csv",
        "test_csv": BASE_DIR / "features/test2_1117_20251117_235357_balanced.csv",
        "desc": "train2+test2, 3-30 points",
    },
    "1121": {
        "train_csv": BASE_DIR / "features/train2_1121_20251121_121804_balanced.csv",
        "test_csv": BASE_DIR / "features/test2_1121_20251121_121804_balanced.csv",
        "desc": "train2+test2, 1121 features",
    },
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(SCRIPT_DIR / "prepare_data_split.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_pool_dataframe(train_csv: Path, test_csv: Path) -> pd.DataFrame:
    if not train_csv.is_file() or not test_csv.is_file():
        raise FileNotFoundError(f"缺少特征文件: {train_csv} 或 {test_csv}")
    train_df = pd.read_csv(train_csv, low_memory=False)
    test_df = pd.read_csv(test_csv, low_memory=False)
    train_df["_src"] = "train_csv"
    test_df["_src"] = "test_csv"
    df = pd.concat([train_df, test_df], ignore_index=True)
    if "file_path" in df.columns:
        df = df.drop_duplicates(subset=["file_path"], keep="first")
    df = df.reset_index(drop=True)
    df["index"] = np.arange(len(df), dtype=np.int64)
    return df


def stratified_three_way_split(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(y))
    idx_remain, idx_test = train_test_split(
        indices, test_size=TEST_RATIO, random_state=RANDOM_STATE, stratify=y
    )
    val_frac = VAL_RATIO / (1.0 - TEST_RATIO)
    idx_train, idx_val = train_test_split(
        idx_remain,
        test_size=val_frac,
        random_state=RANDOM_STATE,
        stratify=y[idx_remain],
    )
    return (
        np.sort(idx_train.astype(np.int64)),
        np.sort(idx_val.astype(np.int64)),
        np.sort(idx_test.astype(np.int64)),
    )


def write_pool_stats(
    pool_name: str,
    df: pd.DataFrame,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"Pool: {pool_name}",
        f"Total: {len(df)}",
        f"Train/Val/Test: {len(idx_train)}/{len(idx_val)}/{len(idx_test)}",
        "",
    ]
    for name, idx in [("train", idx_train), ("val", idx_val), ("test", idx_test)]:
        lines.append(f"--- {name} ---")
        vc = df.iloc[idx]["category"].value_counts().sort_index()
        for cat, cnt in vc.items():
            lines.append(f"  {cat}: {cnt} ({cnt/len(idx)*100:.2f}%)")
        lines.append("")
    (out_dir / "split_statistics.txt").write_text("\n".join(lines), encoding="utf-8")

    cats = sorted(df["category"].unique())
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(cats))
    w = 0.25
    for i, (sn, idx) in enumerate([("train", idx_train), ("val", idx_val), ("test", idx_test)]):
        counts = [int((df.iloc[idx]["category"] == c).sum()) for c in cats]
        ax.bar(x + (i - 1) * w, counts, w, label=sn)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=30, ha="right")
    ax.legend()
    ax.set_title(f"Split {pool_name} (merged categories)")
    plt.tight_layout()
    plt.savefig(out_dir / "split_distribution.png", dpi=200)
    plt.close()


def process_pool(pool_name: str, cfg: dict) -> None:
    logger.info("处理池 [%s] %s", pool_name, cfg["desc"])
    df = load_pool_dataframe(cfg["train_csv"], cfg["test_csv"])
    y = df["category"].values
    idx_train, idx_val, idx_test = stratified_three_way_split(y)

    out_dir = SPLIT_ROOT / pool_name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "train_indices.npy", idx_train)
    np.save(out_dir / "val_indices.npy", idx_val)
    np.save(out_dir / "test_indices.npy", idx_test)
    df.to_csv(out_dir / "manifest.csv", index=False)
    write_pool_stats(pool_name, df, idx_train, idx_val, idx_test, out_dir)
    logger.info(
        "[%s] 完成 train=%d val=%d test=%d -> %s",
        pool_name,
        len(idx_train),
        len(idx_val),
        len(idx_test),
        out_dir,
    )


def main() -> None:
    logger.info("基于原有特征 CSV 构建 75/10/15 划分")
    for pool_name, cfg in POOLS.items():
        process_pool(pool_name, cfg)
    logger.info("全部完成: %s", SPLIT_ROOT)


if __name__ == "__main__":
    main()
