"""
Optuna 最佳模型评估 - 生成三个单独的柱状图（精确率、召回率、F1）
"""

import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_single_bar_chart(values, classes, title, ylabel, save_path, color='steelblue'):
    """绘制单个柱状图"""
    plt.figure(figsize=(10, 6))
    
    # 创建柱状图
    bars = plt.bar(range(len(classes)), values, color=color, edgecolor='black', linewidth=1)
    
    # 设置刻度
    plt.xticks(range(len(classes)), classes, rotation=45, ha='right', fontsize=10)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.ylim(0, 1.05)
    
    # 添加数值标签
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 已保存: {save_path}")

def main():
    # 配置
    model_path = 'train_models/xgboost_full/xgboost_full_best.json'
    test_file = 'features/test2_1117_20251117_235357_balanced.csv'
    results_dir = Path('train_models/xgboost_full/results')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("生成精确率、召回率、F1 柱状图")
    print("=" * 60)
    
    # 加载模型
    print(f"加载模型: {model_path}")
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    # 加载数据
    print(f"加载数据: {test_file}")
    df = pd.read_csv(test_file)
    
    # 提取标签和特征
    y = df['category']
    exclude_cols = ['file_path', 'category', 'target']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols]
    
    # 编码标签
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    classes = le.classes_
    
    # 预测
    print("开始预测...")
    y_pred = model.predict(X)
    
    # 计算指标
    precision, recall, f1, support = precision_recall_fscore_support(
        y_encoded, y_pred, average=None, zero_division=0
    )
    
    # 计算准确率
    accuracy = accuracy_score(y_encoded, y_pred)
    print(f"\n测试集准确率: {accuracy * 100:.2f}%")
    
    # 打印详细数值
    print("\n各类别指标:")
    print("-" * 60)
    print(f"{'类别':<15} {'精确率':>10} {'召回率':>10} {'F1分数':>10} {'样本数':>10}")
    print("-" * 60)
    for i, cls in enumerate(classes):
        print(f"{cls:<15} {precision[i]:>10.4f} {recall[i]:>10.4f} {f1[i]:>10.4f} {support[i]:>10}")
    print("-" * 60)
    
    # 生成三个单独的柱状图
    print("\n生成柱状图...")
    
    # 1. 精确率柱状图
    plot_single_bar_chart(
        precision, classes, 
        'Precision by Class - XGBoost Optuna 1117', 
        'Precision', 
        results_dir / 'precision_bar_chart.png',
        color='steelblue'
    )
    
    # 2. 召回率柱状图
    plot_single_bar_chart(
        recall, classes, 
        'Recall by Class - XGBoost Optuna 1117', 
        'Recall', 
        results_dir / 'recall_bar_chart.png',
        color='seagreen'
    )
    
    # 3. F1 分数柱状图
    plot_single_bar_chart(
        f1, classes, 
        'F1-Score by Class - XGBoost Optuna 1117', 
        'F1-Score', 
        results_dir / 'f1_bar_chart.png',
        color='coral'
    )
    
    print(f"\n✓ 所有图片已保存到: {results_dir}")
    print("  - precision_bar_chart.png (精确率)")
    print("  - recall_bar_chart.png (召回率)")
    print("  - f1_bar_chart.png (F1分数)")

if __name__ == "__main__":
    main()