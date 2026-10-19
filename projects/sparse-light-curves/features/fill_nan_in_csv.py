#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
填充CSV文件中的NaN值并覆盖原文件

处理文件：
- E:/ZTF_variables/features/train2_1117_20251117_235357_balanced.csv
- E:/ZTF_variables/features/test2_1117_20251117_235357_balanced.csv
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

BASE_DIR = Path("E:/ZTF_variables/features")
FILES = [
    BASE_DIR / "train2_1117_20251117_235357_balanced.csv",
    BASE_DIR / "test2_1117_20251117_235357_balanced.csv",
]


def fill_nan_in_file(file_path: Path):
    """填充CSV文件中的NaN值并覆盖原文件"""
    logger.info("\n处理文件: {}".format(file_path))

    if not file_path.exists():
        logger.error("文件不存在: {}".format(file_path))
        return False

    # 读取CSV
    logger.info("读取CSV文件...")
    df = pd.read_csv(file_path)
    logger.info("原始数据形状: {}".format(df.shape))

    # 分离数值列和字符串列
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    string_cols = df.select_dtypes(include=[object]).columns

    # 检查NaN数量
    nan_counts = df[numeric_cols].isna().sum()
    total_nans = nan_counts.sum()

    if total_nans > 0:
        logger.warning("发现 {} 个NaN值:".format(total_nans))
        for col, count in nan_counts[nan_counts > 0].items():
            logger.warning("  {}: {} 个NaN".format(col, count))

        # 填充数值列的NaN为0
        df[numeric_cols] = df[numeric_cols].fillna(0.0)
        logger.info("✅ 所有数值列的NaN已填充为0")
    else:
        logger.info("✅ 未发现NaN值")

    # 检查字符串列的NaN
    string_nan_counts = df[string_cols].isna().sum()
    if string_nan_counts.sum() > 0:
        logger.warning("发现字符串列的NaN:")
        for col, count in string_nan_counts[string_nan_counts > 0].items():
            logger.warning("  {}: {} 个NaN".format(col, count))
        # 字符串列填充为空字符串
        df[string_cols] = df[string_cols].fillna("")
        logger.info("✅ 所有字符串列的NaN已填充为空字符串")

    # 检查inf值
    inf_counts = {}
    for col in numeric_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            inf_counts[col] = inf_count

    if inf_counts:
        logger.warning("发现 {} 个inf值:".format(sum(inf_counts.values())))
        for col, count in inf_counts.items():
            logger.warning("  {}: {} 个inf".format(col, count))
        # 将inf替换为0
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], 0.0)
        logger.info("✅ 所有inf值已替换为0")

    # 保存覆盖原文件
    logger.info("保存文件...")
    df.to_csv(file_path, index=False, encoding="utf-8-sig")
    logger.info("✅ 文件已更新: {}".format(file_path))
    logger.info("最终数据形状: {}".format(df.shape))

    return True


def main():
    logger.info("=" * 80)
    logger.info("开始填充CSV文件中的NaN值")
    logger.info("=" * 80)

    success_count = 0
    for file_path in FILES:
        if fill_nan_in_file(file_path):
            success_count += 1

    logger.info("\n" + "=" * 80)
    logger.info("处理完成: {}/{} 个文件".format(success_count, len(FILES)))
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
