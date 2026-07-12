#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为 train4 和 train2 生成 75/10/15 分层划分索引
"""

import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

BASE_DIR = Path.cwd()
SPLIT_DIR = BASE_DIR / "data/split"
RANDOM_STATE = 42
TEST_RATIO = 0.15
VAL_RATIO = 0.10

def generate_split(manifest_name: str, output_dir_name: str):
    """为单个 manifest 生成划分索引"""
    manifest_path = SPLIT_DIR / manifest_name
    if not manifest_path.exists():
        print(f"警告: {manifest_path} 不存在，跳过")
        return
    
    manifest = pd.read_csv(manifest_path)
    y = manifest['category'].values
    
    indices = np.arange(len(manifest))
    
    # 先分出测试集 (15%)
    idx_remain, idx_test = train_test_split(
        indices, test_size=TEST_RATIO, random_state=RANDOM_STATE, stratify=y
    )
    
    # 从剩余中分验证集 (10% of total = 10/85 of remain)
    val_frac = VAL_RATIO / (1.0 - TEST_RATIO)
    idx_train, idx_val = train_test_split(
        idx_remain, test_size=val_frac, random_state=RANDOM_STATE, stratify=y[idx_remain]
    )
    
    # 保存
    out_dir = SPLIT_DIR / output_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(out_dir / "train_indices.npy", np.sort(idx_train))
    np.save(out_dir / "val_indices.npy", np.sort(idx_val))
    np.save(out_dir / "test_indices.npy", np.sort(idx_test))
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    
    print(f"{output_dir_name}:")
    print(f"  总样本: {len(manifest)}")
    print(f"  训练集: {len(idx_train)} ({len(idx_train)/len(manifest)*100:.1f}%)")
    print(f"  验证集: {len(idx_val)} ({len(idx_val)/len(manifest)*100:.1f}%)")
    print(f"  测试集: {len(idx_test)} ({len(idx_test)/len(manifest)*100:.1f}%)")

# 生成划分
generate_split("manifest_train4.csv", "train4")
generate_split("manifest_train2.csv", "train2")

print("\n完成！划分文件保存在 data/split/train4/ 和 data/split/train2/")