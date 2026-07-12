#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
天体类别合并配置
原始文件夹类别 -> 合并后的训练标签
"""

from __future__ import annotations

from typing import Dict

# 原始细分类别目录名
RAW_CATEGORIES = [
    "BYDra",
    "CEP",
    "CEPII",
    "CV",
    "DSCT",
    "EA",
    "EW",
    "Mira",
    "RR",
    "RRc",
    "RSCVN",
    "SR",
    "SN",
]

# 细分类 -> 合并类（用于分层划分与模型训练标签）
MERGE_MAPPING: Dict[str, str] = {
    "RR": "RR",
    "RRc": "RR",
    "EA": "Eclipsing",
    "EW": "Eclipsing",
    "Mira": "LPV",
    "SR": "LPV",
    "BYDra": "Active",
    "RSCVN": "Active",
    "CV": "Cataclysmic",
    "CEP": "Pulsating",
    "CEPII": "Pulsating",
    "DSCT": "Pulsating",
    "SN": "SN",
}

MERGED_CATEGORIES = sorted(set(MERGE_MAPPING.values()))


def merge_category(raw_category: str) -> str:
    """将原始目录类别映射为合并后的类别名。"""
    key = str(raw_category).strip()
    if key not in MERGE_MAPPING:
        raise KeyError(
            f"未知原始类别 '{key}'，请在 category_config.MERGE_MAPPING 中补充映射"
        )
    return MERGE_MAPPING[key]
