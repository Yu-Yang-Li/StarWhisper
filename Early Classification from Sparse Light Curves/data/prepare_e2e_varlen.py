#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
端到端数据预处理（变长 3–30 点，用于微调）

- 数据池: train2 特征 CSV 对应样本（data/split/varlen/）
- 输出: data/e2e_varlen/{train,val,test}_data.pkl + *_labels.npy + *_lengths.npy
- 每条样本形状: (3, L)，L ∈ [3, 30]
"""

from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from e2e_utils import encode_band_scalar, resolve_csv_path  # noqa: E402

BASE_DIR = SCRIPT_DIR.parent
POOL = "varlen"
SPLIT_DIR = SCRIPT_DIR / "split" / POOL
OUT_DIR = SCRIPT_DIR / "e2e_varlen"
MIN_LEN, MAX_LEN = 3, 30

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_varlen_curve(path: Path) -> tuple[np.ndarray | None, int]:
    try:
        df = pd.read_csv(path)
        if "mjd" not in df.columns or "mag" not in df.columns:
            return None, 4
        mjd = pd.to_numeric(df["mjd"], errors="coerce").to_numpy()
        mag = pd.to_numeric(df["mag"], errors="coerce").to_numpy()
        valid = ~(np.isnan(mjd) | np.isnan(mag))
        mjd, mag = mjd[valid], mag[valid]
        L = len(mag)
        if L < MIN_LEN or L > MAX_LEN:
            return None, 4
        order = np.argsort(mjd)
        mjd, mag = mjd[order], mag[order]
        time = (mjd - mjd[0]).astype(np.float32)
        band = encode_band_scalar(df, path)
        band_arr = np.full(L, band, dtype=np.float32)
        data = np.stack([time, mag.astype(np.float32), band_arr], axis=0)
        return data, band
    except Exception as e:
        logger.debug("跳过 %s: %s", path, e)
        return None, 4


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = SPLIT_DIR / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"请先运行 prepare_data_split.py: {manifest_path}")

    manifest = pd.read_csv(manifest_path, low_memory=False)
    if "index" not in manifest.columns:
        manifest["index"] = np.arange(len(manifest), dtype=np.int64)

    train_idx = np.load(SPLIT_DIR / "train_indices.npy")
    val_idx = np.load(SPLIT_DIR / "val_indices.npy")
    test_idx = np.load(SPLIT_DIR / "test_indices.npy")

    all_data: list = []
    all_labels: list = []
    all_lengths: list = []
    valid_indices: list = []

    logger.info("构建变长时序 manifest=%d", len(manifest))

    for _, row in tqdm(manifest.iterrows(), total=len(manifest)):
        path = resolve_csv_path(str(row["file_path"]), BASE_DIR)
        if path is None:
            continue
        data, _ = load_varlen_curve(path)
        if data is None:
            continue
        all_data.append(data)
        all_labels.append(row["category"])
        all_lengths.append(data.shape[1])
        valid_indices.append(int(row["index"]))

    if not all_data:
        raise RuntimeError("无有效变长样本，请检查 train2 原始 CSV")

    le = LabelEncoder()
    y_enc = le.fit_transform(np.array(all_labels))
    idx_to_pos = {int(i): p for p, i in enumerate(valid_indices)}
    logger.info(
        "有效 %d / %d，长度 %d-%d",
        len(all_data),
        len(manifest),
        min(all_lengths),
        max(all_lengths),
    )

    def gather(split_idx: np.ndarray):
        pos = [idx_to_pos[int(i)] for i in split_idx if int(i) in idx_to_pos]
        if not pos:
            raise RuntimeError("划分索引无匹配样本")
        data_list = [all_data[p] for p in pos]
        labels = y_enc[pos]
        lengths = np.array([all_lengths[p] for p in pos], dtype=np.int32)
        return data_list, labels, lengths

    splits = {}
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        data_list, labels, lengths = gather(idx)
        with open(OUT_DIR / f"{name}_data.pkl", "wb") as f:
            pickle.dump(data_list, f)
        np.save(OUT_DIR / f"{name}_labels.npy", labels)
        np.save(OUT_DIR / f"{name}_lengths.npy", lengths)
        splits[name] = len(data_list)
        logger.info("%s: %d", name, len(data_list))

    import joblib

    joblib.dump(le, OUT_DIR / "label_encoder.pkl")

    with open(OUT_DIR / "e2e_varlen_statistics.txt", "w", encoding="utf-8") as f:
        f.write(f"pool={POOL}\nvalid={len(all_data)}\n")
        f.write(f"length_range=[{min(all_lengths)}, {max(all_lengths)}]\n")
        f.write(f"classes={list(le.classes_)}\n")
        for name, n in splits.items():
            f.write(f"{name}={n}\n")

    logger.info("完成: %s", OUT_DIR)


if __name__ == "__main__":
    main()
