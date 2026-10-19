#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 train4 和 train2 的 manifest 文件
- train4: 50 点固定长度数据
- train2: 3-30 点变长数据
"""

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
SPLIT_DIR = SCRIPT_DIR / "split"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from category_config import merge_category

def generate_manifest(data_dir_name: str, output_name: str):
    """生成单个数据集的 manifest"""
    data_dir = BASE_DIR / data_dir_name
    if not data_dir.exists():
        print(f"警告: {data_dir} 不存在，跳过")
        return None
    
    rows = []
    for cat_dir in data_dir.iterdir():
        if not cat_dir.is_dir():
            continue
        raw_category = cat_dir.name
        try:
            category = merge_category(raw_category)
        except KeyError:
            print(f"跳过未知类别: {raw_category}")
            continue
        
        for csv_file in cat_dir.glob('*.csv'):
            rows.append({
                'file_path': str(csv_file.relative_to(BASE_DIR)),
                'raw_category': raw_category,
                'category': category
            })
    
    manifest = pd.DataFrame(rows)
    manifest.insert(0, 'index', range(len(manifest)))
    output_path = SPLIT_DIR / output_name
    manifest.to_csv(output_path, index=False)
    print(f"{data_dir_name}: 生成 {len(manifest)} 个样本 -> {output_path}")
    print(f"  类别分布:\n{manifest['category'].value_counts()}")
    return manifest

# 生成两个 manifest
generate_manifest("train4", "manifest_train4.csv")
print("\n" + "="*50 + "\n")
generate_manifest("train2", "manifest_train2.csv")

print("\n完成！")