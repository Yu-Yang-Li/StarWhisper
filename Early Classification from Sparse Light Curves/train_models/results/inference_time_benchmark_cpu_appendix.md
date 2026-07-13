# Table S1 — CPU 推理外推（附录）

生成时间: 2026-06-16 21:38:55

## 口径

- **适用**: 论文附录；**不**代表生产部署配置
- **PyTorch**: CPU 上仅测 **1000** 个 test 样本，再线性外推至全 test
- 仅用于与 GPU 或不同硬件环境对照，勿与主文 Table 1 直接混排排名

| exp_id | 模型 | 类别 | 阶段 | device | n_test | 特征提取_秒 | 推理_秒 | 总耗时_秒 | 每万样本_秒 | 峰值显存_MB | 权重_MB | 测速备注 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tf_feat_50obs | Transformer 特征 50obs | 手工特征 | 50点预训练 | cpu | 244938 | 0.0 | 1410.8046 | 1410.8046 | 57.5984 | 0.0 | 867.74 | CPU 1000样本外推 |
| e2e_tf_small_50 | E2E Transformer 轻量 50obs | 原始时序 | 50点预训练 | cpu | 244938 | 0.0 | 3578.4436 | 3578.4436 | 146.0959 | 0.0 | 40.73 | CPU 1000样本外推 |
| e2e_tf_matched_50 | E2E Transformer 同量级 50obs | 原始时序 | 50点预训练 | cpu | 244938 | 0.0 | 37206.7651 | 37206.7651 | 1519.0279 | 0.0 | 867.72 | CPU 1000样本外推 |
| rnn_e2e_varlen | RNN LSTM E2E | 原始时序 | 3-30端到端 | cpu | 257742 | 0.0 | 255.3586 | 255.3586 | 9.9075 | 0.0 | 0.77 | CPU 1000样本外推 |
| tf_feat_scratch | Transformer 特征 3-30从头 | 手工特征 | 3-30从头 | cpu | 257742 | 898.169 | 1606.0558 | 2504.2248 | 97.1601 | 0.0 | 867.74 | CPU 1000样本外推 |
| tf_feat_finetune | Transformer 特征 3-30微调 | 手工特征 | 3-30微调 | cpu | 257742 | 898.169 | 1638.764 | 2536.933 | 98.4292 | 0.0 | 867.74 | CPU 1000样本外推 |
| e2e_tf_small_ft | E2E Transformer 轻量 3-30微调 | 原始时序 | 3-30微调 | cpu | 257742 | 0.0 | 3942.1586 | 3942.1586 | 152.9498 | 0.0 | 40.73 | CPU 1000样本外推 |
| e2e_tf_matched_scratch | E2E Transformer 同量级 3-30从头 | 原始时序 | 3-30从头 | cpu | 257742 | 0.0 | 17299.7682 | 17299.7682 | 671.2049 | 0.0 | 867.72 | CPU 1000样本外推 |
| e2e_tf_matched_ft | E2E Transformer 同量级 3-30微调 | 原始时序 | 3-30微调 | cpu | 257742 | 0.0 | 17913.9048 | 17913.9048 | 695.0324 | 0.0 | 867.72 | CPU 1000样本外推 |

## 英文附录说明参考

CPU timings for PyTorch models were measured on 1,000 test samples and linearly extrapolated to the full test set. These results are for hardware comparison only.
