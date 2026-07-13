#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
端到端数据预处理：原始光变曲线 -> (N, 3, 50) 数组

通道: [归一化时间, mag, band编码]
按 data/split/50obs/ 的 train/val/test 索引划分，输出到 data/e2e/
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
POOL = "50obs"
SPLIT_DIR = SCRIPT_DIR / "split" / POOL
OUT_DIR = SCRIPT_DIR / "e2e"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

SEQ_LEN = 50
FILTERCODE_MAP = {"zg": 1, "zr": 2, "zz": 3, "g": 1, "r": 2, "z": 3}


def resolve_csv_path(file_path: str) -> Path | None:
    fp = str(file_path).replace("\\", "/").lstrip("/")
    candidates = [
        BASE_DIR / fp,
        BASE_DIR / "train4" / Path(fp).name,
    ]
    parts = fp.split("/")
    if len(parts) >= 3 and parts[0] == "train4":
        candidates.insert(1, BASE_DIR / "train4" / parts[1] / parts[-1])
    for p in candidates:
        if p.is_file():
            return p
    return None


def encode_filtercode(df: pd.DataFrame, path: Path) -> np.ndarray:
    for col in ("filtercode", "filter", "band"):
        if col in df.columns and df[col].notna().any():
            val = str(df[col].dropna().iloc[0]).lower().strip()
            for key, code in FILTERCODE_MAP.items():
                if key in val:
                    return np.full(SEQ_LEN, code, dtype=np.float32)
    if "fid" in df.columns and df["fid"].notna().any():
        fid = int(pd.to_numeric(df["fid"].iloc[0], errors="coerce"))
        if fid in (1, 2, 3):
            return np.full(SEQ_LEN, fid, dtype=np.float32)
    name = path.name.lower()
    for key, code in FILTERCODE_MAP.items():
        if f"_{key}" in name:
            return np.full(SEQ_LEN, code, dtype=np.float32)
    return np.full(SEQ_LEN, 4, dtype=np.float32)


def load_curve_matrix(path: Path) -> np.ndarray | None:
    try:
        df = pd.read_csv(path)
        if "mjd" not in df.columns or "mag" not in df.columns:
            return None
        mjd = pd.to_numeric(df["mjd"], errors="coerce").to_numpy()
        mag = pd.to_numeric(df["mag"], errors="coerce").to_numpy()
        if len(mjd) != SEQ_LEN or len(mag) != SEQ_LEN:
            return None
        if np.isnan(mag).any() or np.isnan(mjd).any():
            return None
        order = np.argsort(mjd)
        mjd, mag = mjd[order], mag[order]
        time = (mjd - mjd[0]).astype(np.float32)
        band = encode_filtercode(df, path)
        return np.stack([time, mag.astype(np.float32), band], axis=0)
    except Exception as e:
        logger.debug("跳过 %s: %s", path, e)
        return None


def main() -> None:
    manifest_path = SPLIT_DIR / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"请先运行 prepare_data_split.py，缺少 {manifest_path}")

    manifest = pd.read_csv(manifest_path, low_memory=False)
    if "index" not in manifest.columns:
        manifest["index"] = np.arange(len(manifest), dtype=np.int64)

    train_idx = np.load(SPLIT_DIR / "train_indices.npy")
    val_idx = np.load(SPLIT_DIR / "val_indices.npy")
    test_idx = np.load(SPLIT_DIR / "test_indices.npy")

    X_list, y_list, valid_indices = [], [], []
    logger.info("从 manifest 构建 50 点时序矩阵，共 %d 条…", len(manifest))

    for _, row in tqdm(manifest.iterrows(), total=len(manifest)):
        fp = resolve_csv_path(str(row["file_path"]))
        if fp is None:
            continue
        mat = load_curve_matrix(fp)
        if mat is None:
            continue
        X_list.append(mat)
        y_list.append(row["category"])
        valid_indices.append(int(row["index"]))

    if not X_list:
        raise RuntimeError("无有效 50 点样本，请检查 train4 原始 CSV 路径")

    X_all = np.stack(X_list, axis=0).astype(np.float32)
    y_all = np.array(y_list)
    valid_indices = np.array(valid_indices, dtype=np.int64)
    logger.info("有效 50 点样本: %d / %d", len(X_all), len(manifest))

    le = LabelEncoder()
    y_enc = le.fit_transform(y_all)
    idx_to_pos = {int(idx): pos for pos, idx in enumerate(valid_indices)}

    def gather(split_idx: np.ndarray):
        positions = [idx_to_pos[int(i)] for i in split_idx if int(i) in idx_to_pos]
        if not positions:
            raise RuntimeError("划分索引与有效样本无交集")
        return X_all[positions], y_enc[positions]

    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        Xs, ys = gather(idx)
        np.save(OUT_DIR / f"{name}_data.npy", Xs)
        np.save(OUT_DIR / f"{name}_labels.npy", ys)
        logger.info("%s: %d 样本 -> %s", name, len(Xs), OUT_DIR / f"{name}_data.npy")

    import joblib

    joblib.dump(le, OUT_DIR / "label_encoder.pkl")

    stats = OUT_DIR / "e2e_statistics.txt"
    with open(stats, "w", encoding="utf-8") as f:
        f.write(f"pool={POOL}\n")
        f.write(f"total_valid_50pt={len(X_all)}\n")
        f.write(f"classes={list(le.classes_)}\n")
        for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            _, ys = gather(idx)
            f.write(f"\n{name} n={len(ys)}\n")
            for c in le.classes_:
                f.write(f"  {c}: {(ys == le.transform([c])[0]).sum()}\n")

    logger.info("E2E 数据已写入 %s", OUT_DIR)


if __name__ == "__main__":
    main()
