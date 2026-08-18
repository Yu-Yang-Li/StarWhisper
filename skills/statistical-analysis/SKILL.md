---
name: statistical-analysis
description: Run confirmatory statistics on light-curve, survey, or agent-evaluation tables after the analysis plan is locked. Use for tests, effect sizes, confidence intervals, selection effects, small-count intervals, and multiple testing across candidates.
license: MIT
---

# 统计分析 Statistical Analysis

## StarWhisper astronomy overlay

Read [`astronomy.md`](astronomy.md) first. That file sets the astronomy defaults for this copy.

Literature: NASA ADS, then arXiv `astro-ph.*`. Do not invent papers.
A synthetic Explore run, a classifier score, or a demo candidate is not a discovery.
This skill does not send telescope commands.


实验设计管"采集前"，这个技能管"采集后"：把数据变成可报告、可复核的统计结论。核心纪律是
**按预注册的分析计划做验证性分析**——计划里写了什么检验就跑什么检验，临时起意的探索要标注
exploratory，不能事后当成验证性结果报告。

> 所有数值由 `scripts/run_stat_tests.py` 计算（封装 scipy/statsmodels），模型不徒手估计
> 统计量。每个检验同时返回效应量、95% 置信区间、前提检查结果和一句证据边界提示。

## 何时使用

- "两组/多组差异显著吗""处理前后有变化吗"
- "帮我跑 t 检验 / ANOVA / 卡方 / 相关分析"
- "这批 p 值要不要做多重比较校正"
- "数据不满足正态性怎么办，用什么检验"
- "结果怎么写进论文（效应量、置信区间、p 值报告）"

## 工作流

1. **先找分析计划**：有预注册或实验设计阶段的分析计划就照着执行；没有的话，先和用户确认
   主要结局、比较组和检验方法，再动手，并在报告里注明"计划为事后确定"。
2. **描述统计打底**：`describe` 看每组 n、均值、SD、中位数、IQR 和缺失，异常值和数据录入
   问题在这一步暴露。
3. **前提检查**：检验输出自带 Shapiro 正态性（逐组）和 Brown-Forsythe 方差齐性；不满足时
   按决策树换非参数方法，而不是硬套。
4. **执行检验**：用下面的决策树选命令；p 值、效应量、CI 一次算齐。
5. **多重比较校正**：同一族的所有检验（包括不显著的）一起进 `correct`；只报显著结果是
   p-hacking。
6. **写结论**：按 `templates/analysis_report.md` 报告；每条结论带效应量与 CI，非显著结果
   照报，探索性分析单独一节并明确标注。

## 检验选择决策树

```
比较什么？
├─ 两独立组的连续结局
│   ├─ 近似正态（或每组 n≥30） → ttest（默认 Welch，不假定等方差）
│   └─ 偏态/序数/小样本        → mannwhitney
├─ 同一对象前后两次测量        → paired；偏态时用 Wilcoxon（报告里注明）
├─ 三组及以上连续结局
│   ├─ 近似正态、方差齐        → anova（显著后 --posthoc 做 Tukey）
│   └─ 不满足                  → kruskal
├─ 两个分类变量是否相关        → chi2（2x2 且期望频数小时看 fisher_exact 输出）
└─ 两个连续变量的关系          → corr（线性用 pearson，单调/有离群用 spearman）
```

## 脚本用法

```powershell
python scripts/run_stat_tests.py describe    --csv data.csv --value strength --group process
python scripts/run_stat_tests.py ttest       --csv data.csv --value strength --group process
python scripts/run_stat_tests.py paired      --csv data.csv --before pre_score --after post_score
python scripts/run_stat_tests.py anova       --csv data.csv --value yield --group temperature --posthoc
python scripts/run_stat_tests.py chi2        --csv data.csv --row exposure --col outcome
python scripts/run_stat_tests.py corr        --csv data.csv --x dose --y response --method spearman
python scripts/run_stat_tests.py correct     --pvalues 0.012,0.034,0.20,0.41 --method holm
```

输出是结构化 JSON：统计量、p 值、效应量（Cohen's d/Hedges g、η²、ε²、Cramér's V、
rank-biserial r）、95% CI、前提检查明细，以及一句 `interpretation_boundary` 提示怎么把
结论说得不过头。改脚本后运行 `python tests/regression.py` 应全绿。

## 报告数字的规矩

- p 值报到三位小数（p<0.001 写 `p<0.001`），**永远同时报效应量和 CI**，不报裸 p。
- 显著 ≠ 重要：解释时对照最小实际意义效应（SESOI），不用"极其显著"这类修辞。
- 非显著 ≠ 无差异：只能写"未检测到显著差异"，功效不足时要说明。
- 相关 ≠ 因果：观察性数据的相关只能写"相关、提示、可能有关"。
- 亚组分析、换检验方法重跑、临时加的比较，全部标 exploratory，进多重比较校正。

## 质量红线

- **不做 p-hacking**：不因为结果不显著就换检验、删数据、切亚组；换方法必须给统计学理由
  并在报告中披露。
- **报告全部检验**：跑过的每个检验都出现在报告里，校正基于全族 p 值。
- **缺失与剔除透明**：describe 阶段报缺失量；任何数据剔除写明规则与数量。
- **不越界**：组学管线（scanpy/Seurat/DESeq）、混合效应模型、因果推断方法超出本技能脚本
  范围时明说，不硬算。

## 与其它技能的边界

- 采集前的设计、随机化、样本量与功效 →「实验设计」（本技能执行它锁定的分析计划）。
- 把问题拆成数据任务、建 baseline、机器学习建模 →「数据处理」。
- 结果写进论文的表达与措辞 →「学术写作」「文本润色」。
