---
name: research-baseline-builder
description: "把天文研究问题整理成明确的输入、输出、样本单位、数据划分和简单基线，适用于光变、光谱、图像和观测日志的首次实验。"
---

# 建立研究基线

把研究问题整理成一次可检查的实验：输入什么、输出什么、一条样本代表什么，以及怎样判断结果是否回答了原问题。

## 先确认数据

先写清研究目标、样本单位、输入字段、目标变量、分组或时间信息与成功判据。天文对象、时间系统和数据划分的约定见 [astronomy.md](astronomy.md)。

缺少数据时，先给数据说明、预期表结构和实验安排；不编造字段或训练结果。需要判断数据任务与框架时，分别阅读 [问题与数据](references/problem-to-data-routing.md) 和 [框架选择](references/framework-selection.md)。

## 选择一个简单基线

| 数据 | 先做什么 | 模板参数 |
| --- | --- | --- |
| 光变特征表、观测日志 | 简单规则或 RandomForest | `rf` |
| 时序预测 | 历史均值或最后值，再考虑 GRU | `gru` |
| 图像分类 | 预训练特征与简单分类器，再考虑微调 | `efficientnet` |
| 只整理问题与数据 | 生成说明，不复制模型 | `none` |

XGBoost 调参模板为 `xgb_optuna`。完整依赖、关键词选择规则与运行例子见 [模型模板](references/model-templates.md)。应显式选择数据类型；“天文”关键词会匹配图像模板，不能据此判断光变表该用什么模型。

## 创建工作目录

以下命令在本技能目录中运行：

```bash
python scripts/init_research_baseline_workspace.py "光变特征分类" --template rf --root ./lightcurve-baseline
cd ./lightcurve-baseline/光变特征分类
python scripts/baseline_randomforest.py --help
```

`--root` 是父目录，脚本仍会创建课题子目录，并打印实际路径。运行复制出的脚本时使用该工作目录。带 `--run-baseline` 的工作流可做一次演示运行；自带或合成数据的结果只用于检查环境。

## 检查实验是否成立

按源身份、观测场、夜次或时间分组，避免相关样本落入不同划分。不把对象 ID、文件名或结果产生之后的变量当预测特征；类别不平衡时不只报告准确率。

记录预处理、数据划分、随机种子、指标、误差与未解决的问题。完整步骤见 [实验流程](references/baseline-workflow.md)。结果出来后，按 [目标核查](references/goal-check.md) 检查它是否回答了原科学问题；相关性或预测表现不能直接支持因果结论。

## 会得到什么

工作目录包含问题定义、数据表结构、探索与预处理计划、基线和训练评估计划、报告模板。实际运行后再报告 `metrics.json`、`baseline_summary.json` 和 `train_log.txt`；`routing_decision.json` 与 `workflow_status.json` 记录选择和执行状态。说明实际使用了哪些数据、执行了哪些步骤，以及更复杂建模前还缺什么。
