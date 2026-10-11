# 基线实验与结果解释

[返回技能说明](../SKILL.md)

按本次任务阅读相关部分。本文件中的命令从技能目录执行，文件名与参数按实际输入替换。早期试运行与历史记录不表示外部服务当前可用。

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
