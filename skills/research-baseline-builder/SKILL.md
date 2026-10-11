---
name: research-baseline-builder
description: Translate astronomy questions into data contracts over light curves, spectra, FITS, alert streams, or telescope decision logs. Use when the user needs inputs, outputs, sample units, splits, leakage checks, and a non-LLM baseline before modeling.
---

# Research Baseline Builder

## 天文用法

先阅读 [astronomy.md](astronomy.md)，其中说明资料来源、数据约定和本技能的天文用法。需要查文献时优先使用 NASA ADS，其次是 arXiv `astro-ph.*`，保留实际来源。

本技能不发送望远镜控制指令。合成实验、分类分数和演示候选应按其实际性质报告，不能写成新发现。


这个技能把研究问题整理成数据说明、分析计划和可运行的基线。初始化脚本会生成 Markdown 模板、`data_schema.csv`、`figures/` 和 `scripts/`；具体研究内容需要根据课题和实际数据填写。

通过 `--template` 选择一个模型模板，脚本会将它复制到新工作目录。`routing_decision.json` 和 `workflow_status.json` 记录选择与执行状态；运行模型后会写出 `metrics.json`、`baseline_summary.json` 和 `train_log.txt`，标明使用的是演示数据还是用户数据。

默认输出到当前目录下的 `research-baseline/<课题名>/`。`--root` 指定父目录，脚本仍会在其下创建课题子目录，并在最后一行打印实际路径。[references/](references/) 中有问题与数据对应、框架选择和目标核查的说明。

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

Use the user's language for brief updates when the data task is clarified, the workspace is created, or a run finishes. State what is known, what was produced, and which uncertainty affects the next step.

Before running a baseline, say whether it uses demo data or the user's data. After the run, report the actual result and remaining split or leakage concerns. If data or dependencies are missing, identify them and link to the files already produced. Include a timestamp only when it is useful, using the actual time.

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

明确输入、输出和样本单位后，按数据类型选择一个简单基线。运行时间取决于数据规模和环境；先检查数据与评估方法，再决定是否增加模型复杂度。

| 问题类型 | 数据形态 | 初始基线 | 后续候选 | 代码模板 |
|---------|---------|---------|--------|---------|
| 表格/结构化 (分类/回归) | DataFrame `(n_samples, n_features)` | **RandomForest** | **XGBoost + Optuna**（需限定搜索范围） | `baseline_randomforest.py` → `baseline_xgboost_optuna.py` |
| 时序/序列预测 | 1D/多维 array `(T,)` / `(T, F)` | 历史均值 / 季节性 Naive | **GRU** (PyTorch) | `baseline_gru.py` |
| 图像/视觉分类 | 图片文件夹 `train/<class>/xxx.jpg` | 预训练特征 + 线性头 | **EfficientNet-B0** 迁移学习（冻结→微调） | `baseline_efficientnet.py` |

未指定 `--template` 时，脚本按关键词选择：
- 包含 `image/图像/视觉/显微/天文/影像` → 默认 `efficientnet`
- 包含 `tabular/表格/结构化` → 默认 `xgb_optuna`
- 包含 `time series/timeseries/时序/序列/forecast` → 默认 `gru`
- 其他 → 默认 `rf`（表格通用基线）

关键词选择只是提示，不能判断数据的实际形态。例如“天文”也会匹配图像模板；处理光变特征表或观测日志时，应显式指定 `--template rf`、`xgb_optuna` 或 `gru`。

## 代码模板

`templates/` 中有 4 个可独立运行的脚本，文件开头说明数据格式与依赖。表格和时序模板支持 CSV，图像模板读取分类目录。没有用户数据时可用自带或合成数据检查环境；这些结果不能作为课题的实验结果。

| 模板文件 | 模型 | 框架 | 依赖 | 适用场景 |
|---------|------|------|------|---------|
| `baseline_randomforest.py` | RandomForest 分类/回归 | scikit-learn | `scikit-learn, pandas, numpy, joblib` | 表格数据快速基线、可解释性、特征重要性 |
| `baseline_xgboost_optuna.py` | XGBoost + Optuna 超参搜索 | xgboost + optuna | `xgboost>=2.0, optuna, scikit-learn, pandas, numpy, joblib` | 表格分类或回归，需要调参时使用 |
| `baseline_gru.py` | GRU 时序预测 | PyTorch | `torch, numpy, pandas, scikit-learn` | 单/多变量时序、传感器/金融/气象/实验数据 |
| `baseline_efficientnet.py` | EfficientNet-B0 迁移学习 | PyTorch + torchvision | `torch, torchvision, pillow` | 图像分类、科学图像（显微/天文/医学）、小样本 |

> 依赖可按需安装；没装 torch 的话 rf / xgb_optuna 两个表格模板依然可用，互不依赖。

### 使用示例

以下初始化命令在本技能目录中运行，每次只复制所选的一个模型模板：

```bash
python scripts/init_research_baseline_workspace.py "光变特征分类" --template rf --root ./lightcurve-baseline
python scripts/init_research_baseline_workspace.py "全天相机图像分类" --template efficientnet
python scripts/init_research_baseline_workspace.py "巡天天气时序预测" --template gru
python scripts/init_research_baseline_workspace.py "超新星候选筛选" --template none
```

以第一个工作目录为例，进入后运行复制出的 RandomForest 模板：

```bash
cd ./lightcurve-baseline/光变特征分类
python scripts/baseline_randomforest.py --help
python scripts/baseline_randomforest.py  # 自带 demo，仅检查环境
python scripts/baseline_randomforest.py --csv ./data.csv --target label
```

`data.csv` 和 `label` 要替换成实际文件与目标列。结果写到该工作目录下。选择了 XGBoost 或 GRU 模板的目录可分别运行：

```bash
python scripts/baseline_xgboost_optuna.py --csv ./data.csv --target label --n_trials 20
python scripts/baseline_gru.py --csv ./series.csv --value-column signal --epochs 10
```

也可回到本技能目录，由工作流脚本完成初始化和一次演示运行：

```bash
python scripts/run_research_baseline_workflow.py "光变特征分类" --template rf --root ./lightcurve-demo --run-baseline
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
7. 在生成的工作目录中，通过 CSV 或图像目录参数读取实际数据；需要适配新的数据格式时，再修改相应加载函数。先检查基线的数据划分和评估结果，再考虑更复杂的模型。
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
