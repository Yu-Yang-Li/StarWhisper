# light_curve_classifier

ZTF 光变曲线多分类：手工特征（XGB / Transformer）与端到端（E2E Transformer / RNN）统一实验与对比。

## 仓库内容

- `data/`：划分、E2E 预处理、手工特征构建脚本；`split/{varlen,50obs,1121}/` 含 **索引 npy** 与统计
- `features/`：**特征提取 Python 脚本**（入库）；大型特征 CSV 需本地自备或从 Zenodo 获取
- `train_models/`：训练、微调、评估与 benchmark 脚本；`results/` 为汇总图表与指标表
- `paper_output/figures/`：论文用图的 PDF/PNG 及出图脚本
- 各实验目录下 `results/test_metrics.json`、`metrics.txt`（**不含** 模型权重）

## 本地需自备（未入库）

| 路径 | 说明 |
|------|------|
| `train2/`、`train4/` | 原始光变 CSV |
| `features/*.csv` | 手工特征表（可由下方脚本从原始 CSV 提取） |
| `data/split/*/manifest.csv` | 运行 `python data/prepare_data_split.py` 生成 |
| `data/e2e/`、`data/e2e_varlen/` | 运行 `prepare_e2e_data.py` / `prepare_e2e_varlen.py` |
| 各 `train_models/*/best_model.pth` 等 | 训练得到的权重 |

## 数据处理

```bash
cd /path/to/shared-nvme

# 0. 从原始 ZTF CSV 切分稀疏片段（需自备各类别文件夹，见 data/category_config.py）
python data/create_sparse_segments.py --pool 50obs    # -> train4/，50 点
python data/create_sparse_segments.py --pool varlen   # -> train2/，3–30 点

# 1. 生成划分 manifest
python data/generate_manifests.py
python data/prepare_data_split.py

# 2. 从 train2/train4 原始片段 CSV 提取手工特征
python features/extract_featrures_train4.py   # 1117 特征集 -> features/*.csv
python features/extract_features_1121.py      # 1121 特征集（可选）
# 或增量补全：python data/build_handcrafted_features.py --set 1117

# 3. E2E 预处理数组
python data/prepare_e2e_data.py          # 50 点固定长度
python data/prepare_e2e_varlen.py        # 3–30 点变长
```

`features/` 目录说明：

- **入库**：`extract_featrures_train4.py`（1117）、`extract_features_1121.py`、`balance_dataset.py` 等
- **不入库**：`*.csv` 特征表（体积大，建议 Zenodo）


## 训练与指标汇总

```bash
# XGBoost（Optuna 调参）
python train_models/train_xgb_optuna_1117.py
python train_models/train_xgb_optuna_1121.py

# 汇总各模型 Accuracy / Macro-F1，生成 comparison.png
python train_models/compare_models.py
```

`compare_models.py` 从各实验目录的 `test_metrics.json` 读取指标，实验清单见 `train_models/experiment_registry.py`。

## 推理性能 benchmark
| 脚本 | 作用 |
|------|------|
| `benchmark_inference_time.py` | 入口；测量 XGB / Transformer / E2E 推理与 LS 特征提取耗时 |
| `benchmark_inference_unified.py` | 统一 benchmark 实现（由 time 脚本调用） |
| `compare_models.py` | 汇总分类指标并可选叠加推理时间 |

```bash
# 部署向 GPU benchmark（需已训练权重；跳过 legacy 旧模型）
conda run -n astro_classifier python train_models/benchmark_inference_time.py \
    --device gpu --skip-legacy

# 快速试跑（子集样本）
conda run -n astro_classifier python train_models/benchmark_inference_time.py \
    --device gpu --skip-legacy --max-samples 10000 --feature-samples 200
```

## 划分

统一 **75/10/15**，`random_state=42`，分池：`varlen`（3–30 点）、`50obs`、`1121`。
