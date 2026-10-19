# Table 1 — 生产部署推理耗时（主文）

生成时间: 2026-06-16 21:38:55

## 口径

- **适用**: 论文主文 Table 1 / 正文部署性能（混合 CPU+GPU，非「全 GPU」）
- **E2E / RNN**: 仅 GPU 神经网络推理，全量 test（n≈257742）；不含离线 `prepare_e2e`
- **Transformer 特征路径**: CPU 做 Lomb–Scargle 特征 + GPU 做 Transformer 分类头
- **XGBoost 路径**: CPU 做 Lomb–Scargle 特征 + CPU 做 XGBoost 预测（无 GPU 实现）
- **LS 特征**: CPU 上 1000 样本子集实测，线性外推至全 test

## 各模型测速设备（主表 = 生产部署，附录 = CPU 对照）

| 路径 | 特征提取 | 模型推理 | 放入 |
| --- | --- | --- | --- |
| RNN LSTM E2E | — | gpu（神经网络） | 主表（生产部署） |
| E2E Transformer 轻量 3-30微调 | — | gpu（神经网络） | 主表（生产部署） |
| E2E Transformer 同量级 3-30从头 | — | gpu（神经网络） | 主表（生产部署） |
| E2E Transformer 同量级 3-30微调 | — | gpu（神经网络） | 主表（生产部署） |
| XGBoost 1121 | cpu（Lomb–Scargle） | cpu（XGBoost） | 主表（生产部署） |
| XGBoost 1117 | cpu（Lomb–Scargle） | cpu（XGBoost） | 主表（生产部署） |
| Transformer 特征 3-30从头 | cpu（Lomb–Scargle） | gpu（Transformer 分类头） | 主表（生产部署） |
| Transformer 特征 3-30微调 | cpu（Lomb–Scargle） | gpu（Transformer 分类头） | 主表（生产部署） |

- **主表**：各模型最合理的上线组合（神经网络用 GPU 全量 test；LS / XGB 用 CPU）
- **附录 Table S1**：仅 PyTorch 在 CPU 上 1000 样本外推，用于无 GPU 环境对照，**不含 XGB**（XGB 本身即 CPU 全量实测）

## 排名摘要

- **排序基准（最慢）**: Transformer 特征 3-30微调（38.8641 s/万样本）

| 模型 | 路径 | 每万样本总耗时(s) | 相对Transformer 特征 3-30微调 | GPU峰值显存(MB) |
| --- | --- | --- | --- | --- |
| RNN LSTM E2E | 原始时序 | 0.6790 | 57.24x | 24.4 |
| E2E Transformer 轻量 3-30微调 | 原始时序 | 0.7041 | 55.20x | 220.3 |
| E2E Transformer 同量级 3-30从头 | 原始时序 | 2.8817 | 13.49x | 1053.5 |
| E2E Transformer 同量级 3-30微调 | 原始时序 | 2.8913 | 13.44x | 1053.5 |
| XGBoost 1121 | 手工特征 | 34.5768 | 1.12x | 0.0 |
| XGBoost 1117 | 手工特征 | 37.2692 | 1.04x | 0.0 |
| Transformer 特征 3-30从头 | 手工特征 | 38.8636 | 1.00x | 1743.4 |
| Transformer 特征 3-30微调 | 手工特征 | 38.8641 | 1.00x | 1743.4 |

## 特征提取（单独）

| exp_id | 模型 | 类别 | 阶段 | device | n_test | 特征提取_秒 | 推理_秒 | 总耗时_秒 | 每万样本_秒 | 峰值显存_MB | 权重_MB | 测速备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| feat_all_57 | 全部57特征提取 | 特征提取 | 特征工程 | cpu | 257742 | 8.1983 | 0.0 | 8.1983 | 0.3181 | 0.0 | nan | nan |
| feat_ls | Lomb-Scargle特征提取 | 特征提取 | 特征工程 | cpu | 257742 | 828.5289 | 0.0 | 828.5289 | 32.1457 | 0.0 | nan | nan |

## varlen 部署模型（主表）

| exp_id | 模型 | 类别 | 阶段 | device | n_test | 特征提取_秒 | 推理_秒 | 总耗时_秒 | 每万样本_秒 | 峰值显存_MB | 权重_MB | 测速备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rnn_e2e_varlen | RNN LSTM E2E | 原始时序 | 3-30端到端 | gpu | 257742 | 0.0 | 17.5 | 17.5 | 0.679 | 24.37 | 0.77 | nan |
| e2e_tf_small_ft | E2E Transformer 轻量 3-30微调 | 原始时序 | 3-30微调 | gpu | 257742 | 0.0 | 18.147 | 18.147 | 0.7041 | 220.3 | 40.73 | nan |
| e2e_tf_matched_scratch | E2E Transformer 同量级 3-30从头 | 原始时序 | 3-30从头 | gpu | 257742 | 0.0 | 74.2729 | 74.2729 | 2.8817 | 1053.48 | 867.72 | nan |
| e2e_tf_matched_ft | E2E Transformer 同量级 3-30微调 | 原始时序 | 3-30微调 | gpu | 257742 | 0.0 | 74.5216 | 74.5216 | 2.8913 | 1053.48 | 867.72 | nan |
| xgb_1121 | XGBoost 1121 | 手工特征 | 3-30变长 | cpu | 257742 | 839.7722 | 51.4179 | 891.1901 | 34.5768 | 0.0 | 407.93 | nan |
| xgb_1117 | XGBoost 1117 | 手工特征 | 3-30变长 | cpu | 257742 | 898.169 | 62.4139 | 960.5829 | 37.2692 | 0.0 | 489.76 | nan |
| tf_feat_scratch | Transformer 特征 3-30从头 | 手工特征 | 3-30从头 | gpu | 257742 | 994.0764 | 7.6015 | 1001.6779 | 38.8636 | 1743.44 | 867.74 | nan |
| tf_feat_finetune | Transformer 特征 3-30微调 | 手工特征 | 3-30微调 | gpu | 257742 | 994.0764 | 7.6148 | 1001.6913 | 38.8641 | 1743.44 | 867.74 | nan |

## 英文图注参考 (Figure/Table caption)

Inference latency on the varlen test split (75/10/15 hold-out) under a production-style mixed CPU/GPU setup: end-to-end neural models use GPU forward passes; Lomb–Scargle features and XGBoost use CPU; transformer-on-features runs LS on CPU and the classifier on GPU. End-to-end models exclude offline light-curve preprocessing.
