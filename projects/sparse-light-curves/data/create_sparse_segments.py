#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从原始 ZTF 光变 CSV 切分稀疏片段。

论文数据池与默认输出目录：
  --pool 50obs  -> train4/   固定 50 点（50obs 预训练池）
  --pool varlen -> train2/   3–30 点变长（varlen 主实验池）

输入：--source-dir 下按 raw category 命名的子文件夹（见 data/category_config.py）。
输出：<output_dir>/<raw_category>/*.csv，mjd 归零。
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from category_config import RAW_CATEGORIES  # noqa: E402

POOL_PRESETS = {
    "50obs": {"min_length": 50, "max_length": 50, "output_dir": "train4"},
    "varlen": {"min_length": 3, "max_length": 30, "output_dir": "train2"},
}

GAP_SPLIT_DAYS = 30.0
CEP_OBJECT_LIMIT = 108
DEFAULT_OBJECT_LIMIT = 54


def _setup_logging(log_path: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def _iter_blocks_by_filter(
    df: pd.DataFrame, min_length: int, gap_days: float
) -> Iterator[Tuple[str, pd.DataFrame, int, int]]:
    for filtercode, group in df.groupby("filtercode"):
        group = group.sort_values("mjd").reset_index(drop=True)
        if len(group) < min_length:
            continue
        mjd = group["mjd"].values
        split_points = np.where(np.diff(mjd) > gap_days)[0]
        starts = [0] + (split_points + 1).tolist()
        ends = split_points.tolist() + [len(group) - 1]
        for s, e in zip(starts, ends):
            block = group.iloc[s : e + 1][["mjd", "mag"]].reset_index(drop=True)
            if len(block) >= min_length:
                yield str(filtercode), block, s, e


def _load_csv_safe(csv_file: Path) -> Optional[pd.DataFrame]:
    common_kwargs = {
        "usecols": ["filtercode", "mjd", "mag"],
        "dtype": {"mjd": "float64", "mag": "float32", "filtercode": "category"},
        "on_bad_lines": "skip",
        "low_memory": True,
    }
    for engine_name in ("c", "python"):
        try:
            df = pd.read_csv(csv_file, engine=engine_name, **common_kwargs)
            if not {"filtercode", "mjd", "mag"}.issubset(df.columns):
                return None
            return df
        except Exception as exc:
            logging.getLogger(__name__).error(
                "读取 %s 失败（engine=%s）: %s", csv_file, engine_name, exc
            )
    return None


def _create_output_dirs(output_dir: Path, categories: List[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for category in categories:
        (output_dir / category).mkdir(parents=True, exist_ok=True)


def process_category(
    source_dir: Path,
    output_dir: Path,
    category: str,
    min_length: int,
    max_length: int,
    gap_days: float,
    max_samples_per_category: Optional[int],
) -> Dict:
    category_dir = source_dir / category
    if not category_dir.exists():
        logging.getLogger(__name__).warning("类别目录不存在: %s", category_dir)
        return {"processed": 0, "failed": 0, "limited": False, "saved_counts": {}}

    csv_files = list(category_dir.glob("*.csv"))
    if not csv_files:
        logging.getLogger(__name__).warning("类别 %s 无 CSV", category)
        return {"processed": 0, "failed": 0, "limited": False, "saved_counts": {}}

    logger = logging.getLogger(__name__)
    logger.info("处理 %s：%d 个文件", category, len(csv_files))

    lengths = list(range(min_length, max_length + 1))
    saved_counts = {L: 0 for L in lengths}
    total_saved = 0
    processed = 0
    failed = 0
    out_dir = output_dir / category
    out_dir.mkdir(parents=True, exist_ok=True)
    per_object_limit = CEP_OBJECT_LIMIT if category in {"CEP", "CEPII"} else DEFAULT_OBJECT_LIMIT

    for idx, csv_file in enumerate(csv_files, 1):
        if max_samples_per_category is not None and total_saved >= max_samples_per_category:
            break
        df = _load_csv_safe(csv_file)
        if df is None:
            failed += 1
            continue

        obj_name = csv_file.stem
        obj_saved = 0
        for filtercode, block, s, e in _iter_blocks_by_filter(df, min_length, gap_days):
            if max_samples_per_category is not None and total_saved >= max_samples_per_category:
                break
            if obj_saved >= per_object_limit:
                break

            n = len(block)
            max_l = min(max_length, n)
            for L in range(min_length, max_l + 1):
                if max_samples_per_category is not None and total_saved >= max_samples_per_category:
                    break
                if obj_saved >= per_object_limit:
                    break
                min_count = min(saved_counts.values()) if saved_counts else 0
                if saved_counts[L] > min_count:
                    continue
                for i in range(0, n - L + 1):
                    if obj_saved >= per_object_limit:
                        break
                    min_count = min(saved_counts.values()) if saved_counts else 0
                    if saved_counts[L] > min_count:
                        break
                    seg = block.iloc[i : i + L].copy()
                    seg["mjd"] = seg["mjd"] - seg["mjd"].iloc[0]
                    filename = f"{obj_name}_L{L}_{s}-{e}_{i}-{i + L}_{filtercode}.csv"
                    seg.to_csv(out_dir / filename, index=False)
                    saved_counts[L] += 1
                    total_saved += 1
                    obj_saved += 1
                    if max_samples_per_category is not None and total_saved >= max_samples_per_category:
                        break

        processed += 1
        if idx % 200 == 0:
            logger.info("%s: 进度 %d/%d，已保存 %d", category, idx, len(csv_files), total_saved)

    limited = max_samples_per_category is not None and total_saved >= max_samples_per_category
    logger.info(
        "%s 完成：成功 %d，失败 %d，保存片段 %d",
        category,
        processed,
        failed,
        sum(saved_counts.values()),
    )
    return {
        "processed": processed,
        "failed": failed,
        "limited": limited,
        "saved_counts": saved_counts,
    }


def build_stats_pivot(per_category_saved_counts: Dict[str, Dict[int, int]]) -> pd.DataFrame:
    records = []
    for category, saved_counts in per_category_saved_counts.items():
        for length, count in saved_counts.items():
            records.append({"category": category, "length": length, "count": count})
    stats_df = pd.DataFrame(records)
    if stats_df.empty:
        return pd.DataFrame()
    return stats_df.pivot_table(
        index="category", columns="length", values="count", fill_value=0
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="从原始 ZTF CSV 切分 train2/train4 稀疏片段")
    p.add_argument(
        "--source-dir",
        type=Path,
        default=BASE_DIR,
        help="含各类别子文件夹的原始数据根目录（默认：仓库根目录）",
    )
    p.add_argument(
        "--pool",
        choices=list(POOL_PRESETS),
        default="50obs",
        help="50obs -> train4/ 固定 50 点；varlen -> train2/ 3-30 点",
    )
    p.add_argument("--output-dir", type=Path, default=None, help="输出目录（默认随 pool 自动选择）")
    p.add_argument("--min-length", type=int, default=None)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--gap-days", type=float, default=GAP_SPLIT_DAYS)
    p.add_argument("--max-samples-per-category", type=int, default=None)
    p.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="要处理的原始类别文件夹名（默认 category_config.RAW_CATEGORIES）",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    preset = POOL_PRESETS[args.pool]
    min_length = args.min_length if args.min_length is not None else preset["min_length"]
    max_length = args.max_length if args.max_length is not None else preset["max_length"]
    output_dir = args.output_dir or (BASE_DIR / preset["output_dir"])
    source_dir = args.source_dir.resolve()
    categories = args.categories or RAW_CATEGORIES

    log_path = SCRIPT_DIR / f"create_sparse_segments_{args.pool}.log"
    _setup_logging(log_path)
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("切分稀疏片段 pool=%s", args.pool)
    logger.info("source_dir : %s", source_dir)
    logger.info("output_dir : %s", output_dir)
    logger.info("length     : %d – %d", min_length, max_length)
    logger.info("gap_days   : %.1f", args.gap_days)
    logger.info("=" * 80)

    _create_output_dirs(output_dir, categories)

    all_stats: Dict[str, Dict] = {}
    per_category_saved_counts: Dict[str, Dict[int, int]] = {}
    for category in categories:
        logger.info("-" * 80)
        stats = process_category(
            source_dir=source_dir,
            output_dir=output_dir,
            category=category,
            min_length=min_length,
            max_length=max_length,
            gap_days=args.gap_days,
            max_samples_per_category=args.max_samples_per_category,
        )
        all_stats[category] = stats
        per_category_saved_counts[category] = stats.get("saved_counts", {})

    stats_pivot = build_stats_pivot(per_category_saved_counts)
    if not stats_pivot.empty:
        stats_path = SCRIPT_DIR / "split" / f"segment_length_stats_{args.pool}.csv"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_pivot.to_csv(stats_path)
        logger.info("长度分布统计: %s", stats_path)
        logger.info("\n%s", stats_pivot.to_string())

    per_cat_saved = stats_pivot.sum(axis=1).to_dict() if not stats_pivot.empty else {}
    for category in categories:
        stats = all_stats.get(category, {"processed": 0, "failed": 0, "limited": False})
        logger.info(
            "%s - 文件 %d, 片段 %d, 失败 %d%s",
            category,
            stats["processed"],
            int(per_cat_saved.get(category, 0)),
            stats["failed"],
            " [已限制]" if stats.get("limited") else "",
        )


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()
