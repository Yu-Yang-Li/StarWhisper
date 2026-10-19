#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集平衡处理脚本
1. 根据映射合并类别
2. 限制最大类别样本数不超过最小类别的4倍
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# 配置参数
BASE_DIR = Path("E:/ZTF_variables")
TRAIN_FILE = BASE_DIR / r"features\train2_1105_20251105_184349.csv"
TEST_FILE = BASE_DIR / r"features\test2_1105_20251105_184349.csv"

# 输出文件
TRAIN_OUTPUT = BASE_DIR / r"features\train2_1105_20251105_184349_balanced.csv"
TEST_OUTPUT = BASE_DIR / r"features\test2_1105_20251105_184349_balanced.csv"

# 定义合并映射
MERGE_MAPPING = {
    "RR": "RR",  # 保留原名
    "RRc": "RR",  # 合并到 RR
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
}

# 最大样本比例（从2倍改为4倍）
MAX_RATIO = 4.0


def merge_categories(df):
    """
    根据映射合并类别

    Parameters:
    -----------
    df : DataFrame
        输入数据框

    Returns:
    --------
    DataFrame
        合并后的数据框
    """
    df_copy = df.copy()

    # 统计合并前的数量
    logger.info("\n合并前的类别分布:")
    original_counts = df_copy["category"].value_counts().sort_index()
    for cat, count in original_counts.items():
        logger.info(f"  {cat:12s}: {count:6d}")

    # 应用映射
    df_copy["category"] = df_copy["category"].map(MERGE_MAPPING)

    # 统计合并后的数量
    logger.info("\n合并后的类别分布:")
    merged_counts = df_copy["category"].value_counts().sort_index()
    for cat, count in merged_counts.items():
        logger.info(f"  {cat:12s}: {count:6d}")

    return df_copy


def balance_dataset(df, max_ratio=4.0):
    """
    平衡数据集，使最大类别样本数不超过最小类别的指定倍数

    Parameters:
    -----------
    df : DataFrame
        输入数据框
    max_ratio : float
        最大类别与最小类别的最大比例

    Returns:
    --------
    DataFrame
        平衡后的数据框
    """
    # 统计每个类别的样本数
    category_counts = df["category"].value_counts().sort_values()

    logger.info("\n平衡前各类别样本数:")
    for cat, count in category_counts.items():
        logger.info(f"  {cat:12s}: {count:6d}")

    # 找到最小类别的样本数
    min_count = category_counts.min()
    max_allowed = int(min_count * max_ratio)

    logger.info(f"\n最小类别样本数: {min_count}")
    logger.info(f"最大允许样本数: {max_allowed} ({max_ratio}倍)")

    # 对每个类别进行处理
    balanced_dfs = []

    for category in category_counts.index:
        cat_df = df[df["category"] == category]
        current_count = len(cat_df)

        if current_count > max_allowed:
            # 随机采样到最大允许数量
            cat_df_sampled = cat_df.sample(n=max_allowed, random_state=42)
            removed = current_count - max_allowed
            logger.info(
                f"  {category:12s}: {current_count:6d} -> "
                f"{max_allowed:6d} (删除 {removed})"
            )
            balanced_dfs.append(cat_df_sampled)
        else:
            logger.info(f"  {category:12s}: {current_count:6d} (保持不变)")
            balanced_dfs.append(cat_df)

    # 合并所有类别
    result_df = pd.concat(balanced_dfs, ignore_index=True)

    # 打乱顺序
    result_df = result_df.sample(frac=1, random_state=42)
    result_df = result_df.reset_index(drop=True)

    logger.info("\n平衡后各类别样本数:")
    final_counts = result_df["category"].value_counts().sort_values()
    for cat, count in final_counts.items():
        logger.info(f"  {cat:12s}: {count:6d}")

    return result_df


def process_file(input_file, output_file, file_type):
    """
    处理单个文件

    Parameters:
    -----------
    input_file : Path
        输入文件路径
    output_file : Path
        输出文件路径
    file_type : str
        文件类型（用于日志）
    """
    logger.info("=" * 80)
    logger.info(f"处理{file_type}...")
    logger.info(f"输入文件: {input_file}")

    # 读取数据
    df = pd.read_csv(input_file)
    logger.info(f"原始样本数: {len(df)}")

    # 1. 合并类别
    logger.info("\n步骤1: 根据映射合并类别")
    df = merge_categories(df)

    # 2. 平衡数据集（最大比例改为4倍）
    logger.info("\n步骤2: 平衡数据集")
    df = balance_dataset(df, max_ratio=MAX_RATIO)

    # 3. 保存结果
    logger.info(f"\n保存到: {output_file}")
    df.to_csv(output_file, index=False)
    logger.info(f"最终样本数: {len(df)}")
    logger.info("=" * 80 + "\n")


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("数据集平衡处理（5类分类）")
    logger.info("=" * 80)
    logger.info(f"最大样本比例: {MAX_RATIO}倍")
    logger.info("=" * 80 + "\n")

    # 处理训练集
    process_file(TRAIN_FILE, TRAIN_OUTPUT, "训练集")

    # 处理测试集
    process_file(TEST_FILE, TEST_OUTPUT, "测试集")

    # 输出总结
    logger.info("\n" + "=" * 80)
    logger.info("处理完成！")
    logger.info("=" * 80)
    logger.info(f"训练集输出: {TRAIN_OUTPUT}")
    logger.info(f"测试集输出: {TEST_OUTPUT}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
