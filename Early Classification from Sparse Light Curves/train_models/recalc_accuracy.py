"""
完整重新计算两个模型的准确率
- 完整特征模型 (1117): 使用 test2_1117 测试集
- 消融实验模型 (1121): 使用 test4_1121 测试集
"""

import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np

print("=" * 60)
print("重新计算完整特征模型 (1117)")
print("=" * 60)

# 1. 完整特征模型
model_full = xgb.XGBClassifier()
model_full.load_model('train_models/xgboost_optuna_1117/xgboost_optuna_1117_best.json')
print("✓ 模型加载成功")

df_full = pd.read_csv('features/test2_1117_20251117_235357_balanced.csv')
print(f"✓ 测试集加载成功: {len(df_full)} 个样本")

# 分离特征和标签
exclude_cols = ['file_path', 'category', 'target']
feature_cols = [col for col in df_full.columns if col not in exclude_cols]
X_full = df_full[feature_cols]
y_full = df_full['category']

print(f"特征数量: {len(feature_cols)}")
print(f"标签分布:\n{y_full.value_counts()}")

# 编码标签
le_full = LabelEncoder()
y_full_enc = le_full.fit_transform(y_full)

# 预测
y_pred_full = model_full.predict(X_full)
acc_full = accuracy_score(y_full_enc, y_pred_full)

print(f"\n测试集准确率: {acc_full * 100:.4f}%")
print(f"正确预测: {(y_pred_full == y_full_enc).sum()} / {len(y_full_enc)}")

print("\n" + "=" * 60)
print("重新计算消融实验模型 (1121)")
print("=" * 60)

# 2. 消融实验模型
model_ablation = xgb.XGBClassifier()
model_ablation.load_model('train_models/xgboost_optuna_1121/xgboost_optuna_1121_best.json')
print("✓ 模型加载成功")

df_ablation = pd.read_csv('features/test4_1121_20251121_121804_balanced.csv')
print(f"✓ 测试集加载成功: {len(df_ablation)} 个样本")

# 分离特征和标签
feature_cols_ablation = [col for col in df_ablation.columns if col not in exclude_cols]
X_ablation = df_ablation[feature_cols_ablation]
y_ablation = df_ablation['category']

print(f"特征数量: {len(feature_cols_ablation)}")
print(f"标签分布:\n{y_ablation.value_counts()}")

# 编码标签
le_ablation = LabelEncoder()
y_ablation_enc = le_ablation.fit_transform(y_ablation)

# 预测
y_pred_ablation = model_ablation.predict(X_ablation)
acc_ablation = accuracy_score(y_ablation_enc, y_pred_ablation)

print(f"\n测试集准确率: {acc_ablation * 100:.4f}%")
print(f"正确预测: {(y_pred_ablation == y_ablation_enc).sum()} / {len(y_ablation_enc)}")

print("\n" + "=" * 60)
print("对比结果")
print("=" * 60)
print(f"完整特征模型 (1117): {acc_full * 100:.4f}%")
print(f"消融实验模型 (1121): {acc_ablation * 100:.4f}%")
print(f"差值: {(acc_ablation - acc_full) * 100:+.4f}%")

if acc_ablation > acc_full:
    print("\n⚠️ 消融实验准确率更高！")
    print("可能原因: 测试集不同 / 特征不同 / Optuna 次数不同")
else:
    print("\n✓ 完整特征准确率更高，周期特征重要")