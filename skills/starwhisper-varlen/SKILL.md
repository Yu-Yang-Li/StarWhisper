---
name: starwhisper-varlen
description: Compare the published sparse ZTF/ATLAS early-classification benchmark and lint a light-curve table against the varlen 3-30 / 7-class contract. Use for varlen, sparse light curves, XGBoost vs Transformer vs LSTM, seven merged variability classes, or Early Classification from Sparse Light Curves. Does not train; never treats a test-set score as an explosion time.
license: Apache-2.0
---

# 稀疏光变 varlen 基准

把已经公开的 11 个配置成绩单变成**可横比、可拒比**的判定，并把别人的光变表先对照合同检查一遍。

> 数字由 `scripts/eval_varlen.py` 读取（stdlib）。脚本不训练、不下载权重、不改划分。
> 主设定：3–30 个观测点、7 个合并类、75/10/15、`random_state = 42`。代码在 [`Early Classification from Sparse Light Curves/`](../../Early%20Classification%20from%20Sparse%20Light%20Curves/README.md)。

默认读技能里冻结的 `references/published_metrics.csv`（从各配置 `test_metrics.json` / `metrics.txt` 抄来）。**只在同一 `pool` 里比。** `50obs` 的准确率不是 varlen 3–30 的结果。

## 何时使用

- "这 11 个配置谁最好""XGBoost 和同量级 Transformer 差多少"
- "50 点预训练那个 99% 能不能当早期分类成绩写"
- "我这批光变点够不够 3 个、标签对不对得上那 7 类"
- "BYDra / EA / RRc 合并成哪一类"

## 工作流

1. **`contract`**：先念主设定。3–30、7 类、划分种子。没念完不要比数字。
2. **`table --pool varlen`**：只看 varlen 池。不要把 50obs 行掺进来排序。
3. **`compare --a --b`**：两个 `exp_id` 必须同池，否则标 `NOT COMPARABLE` 并退出说明，不要把差值写成方法增益。
4. **`check`**：用户自己的 CSV 先过合同，再谈训练或投稿。

## 已核对的 varlen 排序（按 macro-F1）

| exp_id | macro-F1 | acc | 说明 |
| --- | ---: | ---: | --- |
| `e2e_tf_matched_ft` | 0.9497 | 0.9458 | 同量级 Transformer，50obs 预训练再微调 |
| `xgb_1117` | 0.9231 | 0.9134 | 57 维手工特征 XGBoost |
| `tf_feat_scratch` | 0.8925 | 0.8799 | 手工特征 Transformer 从头训 |
| `rnn_e2e_varlen` | 0.8639 | 0.8467 | LSTM 基线 |
| `tf_feat_finetune` | 0.8428 | 0.8224 | 手工特征 Transformer 微调 |
| `e2e_tf_small_ft` | 0.7921 | 0.7638 | 轻量 ~11M 微调 |
| `e2e_tf_matched_scratch` | 0.5522 | 0.5128 | 同量级从头训，**稳定失败** |

`xgb_1121` 是另一个数据池（无 Lomb-Scargle 的 39 维），不要和 `xgb_1117` 直接当同一设定。`e2e_tf_matched_50` 的 0.9987 是 50 点设定，**不能**写成早期分类主结果。

## 脚本用法

```powershell
python scripts/eval_varlen.py contract
python scripts/eval_varlen.py table --pool varlen
python scripts/eval_varlen.py best --pool varlen
python scripts/eval_varlen.py compare --a e2e_tf_matched_ft --b xgb_1117
python scripts/eval_varlen.py labels --raw BYDra,EA,RRc,SN
python scripts/eval_varlen.py check --csv lightcurves.csv --nobs-col n_obs --label-col label
```

`compare` 跨池、`labels` 遇到未知类、`check` 有 error 时退出码为 1。

用户表至少要有观测点数和标签两列。标签可以是 7 个合并类，也可以是原始目录名（`BYDra`、`EA`、`RRc` 等）。

## 报告纪律

- 先说设定再报数字。varlen 和 50obs 分开展示。
- `e2e_tf_matched_scratch` 的 0.51 要照报，不要从对比里拿掉。
- 测试集指标不是爆发时刻，也不是发现。
- 不要拿 StarWhisper LC 的 Kepler/K2 约 90% 顶替本基准。
- 权重在 https://huggingface.co/castor0705/sparse-lc-early-classification ；本技能不下载、不推理。
