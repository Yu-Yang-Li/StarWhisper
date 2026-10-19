#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为统一 manifest 构建手工特征表（1117 / 1121）。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
SPLIT_DIR = SCRIPT_DIR / "split"
FEATURE_DIR = SCRIPT_DIR / "features"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from category_config import merge_category  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

_EXTRACTOR_CACHE: Dict[str, Any] = {}


def _get_extractor(set_name: str):
    if set_name in _EXTRACTOR_CACHE:
        return _EXTRACTOR_CACHE[set_name]
    path = (
        BASE_DIR / "features/extract_features_full.py"
        if set_name == "1117"
        else BASE_DIR / "features/extract_features_reduced.py"
    )
    spec = importlib.util.spec_from_file_location(f"ext_{set_name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _EXTRACTOR_CACHE[set_name] = mod
    return mod


def _resolve_csv(file_path: str, raw_category: str) -> Path:
    p = BASE_DIR / str(file_path).replace("\\", "/")
    if p.is_file():
        return p
    p2 = BASE_DIR / "train4" / raw_category / Path(file_path).name
    if p2.is_file():
        return p2
    raise FileNotFoundError(file_path)


def _extract_worker(task: tuple) -> Optional[Dict[str, Any]]:
    idx, file_path, raw_category, set_name = task
    try:
        mod = _get_extractor(set_name)
        path = _resolve_csv(file_path, raw_category)
        feat = mod.extract_features_from_file(path, merge_category(raw_category))
        if feat is None:
            return None
        feat["index"] = idx
        feat["raw_category"] = raw_category
        feat["category"] = merge_category(raw_category)
        return feat
    except Exception:
        return None


def _load_merged_csv_features(set_name: str) -> pd.DataFrame:
    pat = "*1117*" if set_name == "1117" else "*1121*"
    files = sorted((BASE_DIR / "features").glob(f"{pat}balanced.csv"))
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_csv(f, low_memory=False) for f in files]
    feat = pd.concat(dfs, ignore_index=True).drop_duplicates("file_path", keep="first")
    logger.info("从 %d 个 CSV 合并 %d 条特征", len(files), len(feat))
    return feat


def build_features(set_name: str, workers: int = 8, extract_missing: bool = True) -> None:
    manifest = pd.read_csv(SPLIT_DIR / "sample_manifest.csv")
    csv_feat = _load_merged_csv_features(set_name)

    if not csv_feat.empty:
        df = manifest.merge(csv_feat, on="file_path", how="left", suffixes=("", "_csv"))
        # 合并后 category 以 manifest 为准
        if "category_csv" in df.columns:
            df.drop(columns=["category_csv"], inplace=True, errors="ignore")
    else:
        df = manifest.copy()

    missing_idx = df.index[df.get("mean_mag", pd.Series(np.nan, index=df.index)).isna()].tolist()
    logger.info("CSV 未覆盖样本: %d", len(missing_idx))

    if extract_missing and missing_idx:
        tasks = [
            (int(manifest.loc[i, "index"]), manifest.loc[i, "file_path"], manifest.loc[i, "raw_category"], set_name)
            for i in missing_idx
        ]
        rows = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_extract_worker, t) for t in tasks]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="extract"):
                r = fut.result()
                if r:
                    rows.append(r)
        if rows:
            ext = pd.DataFrame(rows).set_index("index")
            df = df.set_index("index", drop=False)
            for col in ext.columns:
                if col in ("index", "raw_category", "category"):
                    continue
                df.loc[ext.index, col] = ext[col].values
            df = df.reset_index(drop=True)

    out = df[df["mean_mag"].notna()].copy()
    logger.info("有效特征: %d / %d", len(out), len(manifest))

    out_path = FEATURE_DIR / f"features_{set_name}.parquet"
    try:
        out.to_parquet(out_path, index=False)
    except Exception:
        out_path = FEATURE_DIR / f"features_{set_name}.csv"
        out.to_csv(out_path, index=False)

    meta = {"file_path", "category", "raw_category", "index", "num_points", "band_code"}
    feat_cols = [c for c in out.columns if c not in meta and pd.api.types.is_numeric_dtype(out[c])]
    (FEATURE_DIR / f"feature_columns_{set_name}.json").write_text(
        json.dumps(feat_cols, indent=2), encoding="utf-8"
    )
    logger.info("保存 %s (%d 特征)", out_path, len(feat_cols))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--set", choices=["1117", "1121"], default="1117")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--no-extract", action="store_true")
    args = p.parse_args()
    build_features(args.set, workers=args.workers, extract_missing=not args.no_extract)


if __name__ == "__main__":
    main()
