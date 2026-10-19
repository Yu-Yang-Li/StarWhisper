"""
生成测试集观测点分布图
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_distribution(csv_path, save_dir, title, color='#2E86AB'):
    """绘制观测点分布柱状图"""
    
    # 读取数据
    df = pd.read_csv(csv_path)
    counts = df['num_points'].value_counts().sort_index()
    total = len(df)
    
    # 创建保存目录
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 绘图
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = counts.index.tolist()
    y = counts.values.tolist()
    
    # 绘制柱状图
    bars = ax.bar(x, y, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # 设置标签和标题
    ax.set_xlabel('Number of Points', fontsize=12)
    ax.set_ylabel('Sample Count', fontsize=12)
    ax.set_title(f'{title}\n(Total: {total} samples)', fontsize=14)
    
    # 设置 x 轴刻度
    ax.set_xticks(range(3, 31, 3))
    ax.set_xlim(2, 31)
    
    # 添加数值标签
    max_y = max(y) if y else 1
    for bar, val in zip(bars, y):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_y*0.01,
                    f'{val}', ha='center', va='bottom', fontsize=8)
    
    # 添加网格
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # 保存
    save_path = save_dir / 'num_points_distribution.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 已保存: {save_path}")
    return save_path

# ==================== 完整特征实验 (1117) ====================
csv_1117 = 'features/test2_1117_20251117_235357_balanced.csv'
save_dir_1117 = 'train_models/xgboost_full/results'

print("生成完整特征实验分布图...")
plot_distribution(
    csv_path=csv_1117,
    save_dir=save_dir_1117,
    title='Test Set Distribution (1117 Features)',
    color='#2E86AB'
)

# ==================== 消融实验 (1121) ====================
csv_1121 = 'features/test4_1121_20251121_121804_balanced.csv'
save_dir_1121 = 'train_models/xgboost_reduced/results'

print("生成消融实验分布图...")
plot_distribution(
    csv_path=csv_1121,
    save_dir=save_dir_1121,
    title='Test Set Distribution (1121 Features - No Lomb-Scargle)',
    color='#2E86AB'
)

print("\n完成！图片保存在:")
print("  - train_models/xgboost_full/results/num_points_distribution.png")
print("  - train_models/xgboost_reduced/results/num_points_distribution.png")