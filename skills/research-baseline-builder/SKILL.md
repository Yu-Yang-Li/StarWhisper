---
name: research-baseline-builder
description: Translate astronomy questions into data contracts over light curves, spectra, FITS, alert streams, or telescope decision logs. Use when the user needs inputs, outputs, sample units, splits, leakage checks, and a non-LLM baseline before modeling.
---

# Research Baseline Builder

## StarWhisper astronomy overlay

This copy is adapted for astronomy research and telescope-agent work.
**Read [`astronomy.md`](astronomy.md) before following generic biomedical / clinical defaults in the rest of this file.**

Default literature route: NASA ADS, then arXiv `astro-ph.*`, then the original skill's search backend if credentials exist.
Do not claim a real hardware observing loop, a discovery, or a referee-ready result unless the user supplied that evidence.


> **ℹ️ Skill 形态说明**：本 skill **不是空骨架**，而是"模板生成器 + LLM 引导 + 可运行代码模板"工作流：
> - `scripts/init_research_baseline_workspace.py` 会在指定 root 下创建一整套结构化 Markdown 模板（problem_definition / eda_plan / preprocess_plan / baseline_plan / train_eval_plan / baseline_report）+ `data_schema.csv`（UTF-8-BOM）+ `figures/` `scripts/` 子目录。
> - 通过 `--template` 参数可把 `templates/` 下的**可运行基线代码**复制到 `scripts/` 下，直接 `python scripts/baseline_xxx.py` 就能在自己的（或 sklearn demo）数据上跑起来。
> - `scripts/run_research_baseline_workflow.py` 是可选薄编排层：调用 init、输出中文进度、可选运行一次复制出的 baseline，并回写 `workflow_status.json`。
> - 初始化会生成 `routing_decision.json` 和 `workflow_status.json`；baseline 模板会生成 `metrics.json`、`baseline_summary.json`、`train_log.txt`，明确区分 `demo` 和 `user_csv`。
> - `references/` 提供 3 份方法学指南（框架选型 / 目标核查 / 问题→数据路由）。
> - 模板内的具体研究内容由 Codex/LLM 结合用户课题对话式填充。直接 `init` 只会得到空模板 + 决策表，这是设计如此，不是 bug。
> - 默认 root 为 `./research-baseline/<slugified-topic>/`（相对路径，会在 CWD 下创建中文目录）；可通过 `--root` 指定任意绝对路径。**不再依赖任何硬编码中文路径（如 `03-AI笔记`）**。

## Overview

Use this skill to help a researcher turn a scientific question into a data problem with explicit inputs, outputs, and a baseline SOP. The goal is a defensible first experiment, not a model leaderboard.

Assume the user is a scientist or domain researcher. Do not over-explain their field. Help them make the data contract, baseline path, and evaluation boundary explicit.

The working shape is:

```text
Goal + Data + Data description
-> data-task recommendation
-> framework selection
-> visualization / preprocessing / baseline execution (with runnable template)
-> interpretation report
-> check against the original scientific goal
```

## User-Visible Progress Updates

Emit concise progress updates at workflow transitions while translating the scientific question into a data workflow. Use the user's language for user-facing updates; for Chinese requests, write progress in Chinese. Prefix each update with current local time or elapsed time, then include one useful payload. Do not change the actual init script, copied templates, baseline ladder, evaluation rules, or user data just to create these messages.

Chinese progress update shape:

```text
[21:31 | 数据契约完成]
当前样本单位是单个患者时点，输入为转录组+临床表，输出为 30 天响应标签；主要风险是标签时间窗和训练/验证划分需确认。
```

Use this event-driven pattern:

- After request parsing: timestamp, scientific question, inferred input, output, sample unit, label/outcome, and missing fields that change the task.
- After task routing: timestamp, selected data-task family and first credible baseline candidate, such as RF, XGBoost+Optuna, GRU, or EfficientNet.
- After workspace generation: timestamp, output root, generated templates/scripts, and which SOP file is being filled.
- Before running any baseline: timestamp and whether it is demo data, a fast sanity baseline, or the user's real data; do not imply real-data results from demo runs.
- After baseline or dry run: timestamp, command/result summary, metric boundary, and whether leakage/split risks appeared.
- On long waits while filling SOP or running code: send one keepalive only when there has been no visible workflow transition for a while; include completed files, pending files, data availability, and current risk.
- On blocker: timestamp, missing data, missing dependency, unclear label, or causal-identification gap, plus the exact artifact still produced.
- On completion: timestamp, data contract, recommended framework, generated files/scripts, split/leakage risks, and what must be confirmed before stronger modeling.

## Core Rule

Do not start with models. First ask what goes in, what should come out, and what one sample means.

Always separate:

- scientific question: what the researcher wants to know;
- input: raw data, features, conditions, interventions, time points, images, text, spectra, tables, or sequences;
- output: label, measurement, effect, ranking, cluster, forecast, mechanism claim, or report;
- sample unit: what one row/sample/image/event/patient/material/paper represents;
- success criterion: what result would answer the scientific question;
- baseline: the simplest credible way to solve that data problem.

## Collaboration Style

- Treat domain claims as hypotheses to operationalize, not as material to lecture back.
- Ask only for missing information that changes the data task, split, metric, or baseline.
- Use the scientist's terms when they are clear; translate them into data roles beside the original wording.
- Be direct about unidentifiable causal claims, missing labels, leakage, weak ground truth, and small-sample limits.
- Avoid beginner tutorials unless the user asks. Give a research workflow, not a course note.

## 基线模型选择决策表

做完 input-output 合同后，按下表挑第一个能跑的基线。**原则：先跑快基线（5-10分钟拿到数），再跑强基线**。

| 问题类型 | 数据形态 | 快速基线 (5min 可跑) | 强基线 | 代码模板 |
|---------|---------|---------|--------|---------|
| 表格/结构化 (分类/回归) | DataFrame `(n_samples, n_features)` | **RandomForest**（可解释、无需归一化、抗噪声） | **XGBoost + Optuna**（表格 SOTA、超参搜索） | `baseline_randomforest.py` → `baseline_xgboost_optuna.py` |
| 时序/序列预测 | 1D/多维 array `(T,)` / `(T, F)` | 历史均值 / 季节性 Naive | **GRU** (PyTorch) | `baseline_gru.py` |
| 图像/视觉分类 | 图片文件夹 `train/<class>/xxx.jpg` | 预训练特征 + 线性头 | **EfficientNet-B0** 迁移学习（冻结→微调） | `baseline_efficientnet.py` |

决策关键词：
- 包含 `image/图像/视觉/显微/天文/影像` → 默认 `efficientnet`
- 包含 `tabular/表格/结构化` → 默认 `xgb_optuna`
- 包含 `time series/timeseries/时序/序列/forecast` → 默认 `gru`
- 其他 → 默认 `rf`（表格通用基线）

## 代码模板

`templates/` 目录下提供 4 个**可独立运行**的基线脚本，顶部 docstring 说明适用场景/数据接口/依赖，参数集中在 CONFIG 区，用 `pathlib.Path`，中文注释，均含 `if __name__ == "__main__":` 演示块（直接 `python baseline_xxx.py` 就会用 sklearn 自带/合成数据跑通）。表格和时序模板支持 CSV 输入；所有模板都会写结构化指标和训练日志。

| 模板文件 | 模型 | 框架 | 依赖 | 适用场景 |
|---------|------|------|------|---------|
| `baseline_randomforest.py` | RandomForest 分类/回归 | scikit-learn | `scikit-learn, pandas, numpy, joblib` | 表格数据快速基线、可解释性、特征重要性 |
| `baseline_xgboost_optuna.py` | XGBoost + Optuna 超参搜索 | xgboost + optuna | `xgboost>=2.0, optuna, scikit-learn, pandas, numpy, joblib` | 表格 SOTA 基线、中大规模、需要调参 |
| `baseline_gru.py` | GRU 时序预测 | PyTorch | `torch, numpy, pandas, scikit-learn` | 单/多变量时序、传感器/金融/气象/实验数据 |
| `baseline_efficientnet.py` | EfficientNet-B0 迁移学习 | PyTorch + torchvision | `torch, torchvision, pillow` | 图像分类、科学图像（显微/天文/医学）、小样本 |

> 依赖可按需安装；没装 torch 的话 rf / xgb_optuna 两个表格模板依然可用，互不依赖。

### 使用示例

```bash
# 1. 最简：自动根据 topic 关键词选模板，在 ./research-baseline/<topic>/ 下生成骨架
python scripts/init_research_baseline_workspace.py "显微图像细胞分类"
# -> 自动检测到"图像"关键词，复制 templates/baseline_efficientnet.py 到 scripts/

# 2. 显式指定模板（覆盖自动识别）
python scripts/init_research_baseline_workspace.py "用户流失预测" --template xgb_optuna
# -> 复制 baseline_xgboost_optuna.py 到 scripts/

# 3. 只要骨架和决策表，不要复制模板代码
python scripts/init_research_baseline_workspace.py "某新材料性能预测" --template none

# 4. 指定输出根目录
python scripts/init_research_baseline_workspace.py "股票收益预测" --template gru --root ./my_project

# 5. 进入生成目录，直接跑模板（以自带 demo 数据验证环境）
cd ./research-baseline/用户流失预测
python scripts/baseline_xgboost_optuna.py                # 跑 demo (breast_cancer) 验证环境
python scripts/baseline_xgboost_optuna.py --help         # 查看参数

# 6. 用户 CSV 数据：直接指定目标列，结果会写 metrics.json / baseline_summary.json / train_log.txt
python scripts/baseline_randomforest.py --csv ./data.csv --target label
python scripts/baseline_xgboost_optuna.py --csv ./data.csv --target label --n_trials 20
python scripts/baseline_gru.py --csv ./series.csv --value-column signal --epochs 10

# 7. 可选：一键串联初始化 + 快速 baseline（仍会标明 demo 或 user_csv）
python scripts/run_research_baseline_workflow.py "结构化表格风险预测" --run-baseline
```

## Workflow

1. Restate the scientific question in one sentence.
2. Clarify the input-output contract:
   - research goal;
   - input data;
   - data description or field meaning;
   - expected output;
   - sample unit;
   - label/outcome/effect;
   - available features;
   - grouping/time/batch/source fields;
   - missing fields and assumptions.
3. Read `references/problem-to-data-routing.md` and choose the data-task family.
4. Read `references/framework-selection.md` and recommend the lightest sufficient framework（对照上方决策表）。
5. Run `scripts/init_research_baseline_workspace.py <topic> [--template xxx]` 创建工作空间（自动复制匹配的代码模板），或用 `scripts/run_research_baseline_workflow.py <topic> --run-baseline` 做初始化 + 一次快速验证。
6. Write the SOP outputs in order:
   - `problem_definition.md`
   - `data_schema.csv`
   - `eda_plan.md`
   - `preprocess_plan.md`
   - `baseline_plan.md`（已追加"推荐基线模型"章节和决策表）
   - `train_eval_plan.md`
   - `baseline_report.md`
7. 把 `scripts/baseline_xxx.py` 里的 `load_demo_data()` 替换为真实数据加载，先跑 fast baseline 验证数据/评估 pipeline，再上强基线。
8. After results exist, read `references/goal-check.md` and check whether the output actually answers the original scientific goal. If not, decompose the task, revise the data question, or stop with the missing evidence.

If no data file is available, do not invent columns or write runnable training code. Produce the input-output contract, expected schema, baseline SOP, and the checks needed once data is provided.

## Baseline Ladder

After the input-output contract is clear, prefer this order:

1. sanity baseline: majority class, mean/median, last value, random, simple rule;
2. interpretable baseline: linear/logistic regression, Cox model, ARIMA, TF-IDF + linear model, simple statistical test;
3. strong classical baseline: **RandomForest**（→XGBoost+Optuna）, SVM, mixed effects, propensity/matching, difference-in-differences;
4. neural baseline only when modality/scale justifies it: **GRU** for sequence, **EfficientNet** for images.

If the scientific question is causal, do not turn it into plain prediction without warning. Ask for intervention/exposure, outcome, confounders, timing, and identification assumptions.

## Standard SOP

Use this spine after the scientific question has been translated:

1. **Data visualization**: label distribution, missingness, feature distributions, group/time/batch balance, target leakage checks.
2. **Data preprocessing**: cleaning, units, outliers, missing values, normalization, encoding, duplicate handling, train/val/test split.
3. **Model building**: baseline ladder（先 rf/naive，再 xgb_optuna/gru/efficientnet）, feature set, assumptions, implementation package.
4. **Model training**: split protocol, seeds, cross-validation, hyperparameter boundary (Optuna for XGBoost), logging.
5. **Model evaluation**: primary metric, secondary metrics, uncertainty, subgroup performance, error slices, calibration when relevant.
6. **Interpretation**: translate figures, metrics, and errors back to the scientific question.

## Guardrails

- Do not recommend random split when samples share patient, material, paper, lab, batch, site, time window, or source identity.
- Do not use post-outcome variables as features.
- Do not use row IDs, object IDs, filenames, source IDs, or database keys as predictive features unless the scientific question explicitly justifies them.
- Do not optimize only accuracy under imbalance.
- Do not call correlation an effect.
- Do not hide unavailable labels, small sample size, weak ground truth, or annotation noise.
- Do not overbuild. A clear baseline beats an impressive but uncheckable model.

## Output Contract

Report:

- scientific question;
- input-output contract;
- recommended data task and framework;
- data-task family and why;
- dataset schema and missing fields;
- split rule and leakage risks;
- baseline ladder（含选用的代码模板路径）;
- preprocessing and visualization checklist;
- training/evaluation plan;
- generated files or scripts, including `routing_decision.json`, `workflow_status.json`, and baseline result files when present;
- what must be confirmed before stronger modeling.
