---
name: experiment-design
description: Design observing campaigns, injection-recovery tests, and agent evaluations before collecting or replaying data. Use for night/field blocking, seeded synthetic nights, shadow-mode hardware runs, sample size, and pre-registered thresholds. Do not default to clinical RCT language.
license: MIT
---

# 实验设计 Experiment Design

## StarWhisper astronomy overlay

This copy is adapted for astronomy research and telescope-agent work.
**Read [`astronomy.md`](astronomy.md) before following generic biomedical / clinical defaults in the rest of this file.**

Default literature route: NASA ADS, then arXiv `astro-ph.*`, then the original skill's search backend if credentials exist.
Do not claim a real hardware observing loop, a discovery, or a referee-ready result unless the user supplied that evidence.


设计决定了数据能回答什么问题——**混杂或伪重复的设计，事后再高级的分析也救不回来**。本技能覆盖
Fisher 三原则、设计类型选择、随机化/区组、样本量与功效、以及采集前就锁定的分析计划。跨学科通用：
湿实验、临床、社科实证、计算/机器学习实验都适用。

> 本技能整合了实验设计与统计功效分析的通行方法（Fisher 随机化/重复/区组原则、
> 经典 DOE 与响应面设计、Cohen 效应量与 a-priori 功效分析），并将样本量、随机化、
> DOE 布局等确定性计算下沉为可复现脚本。

## 脚本（确定性工具，采集前直接生成方案）

- `scripts/power.py` — 样本量/功效计算（依赖 statsmodels，见 `requirements.txt`）
- `scripts/randomization.py` — seeded 随机化方案：simple/block/stratified/cluster（仅标准库）
- `scripts/doe_designs.py` — DOE 矩阵：全析因/两水平析因/拉丁超立方，运行顺序随机化（仅标准库）
- `templates/preregistration.md` — 采集前预注册 / 分析计划模板（对应工作流第 7 步）
- `tests/regression.py` — 脚本回归测试（数值与 statsmodels 对拍 + 边界/编码/输出路径），改脚本后运行 `python tests/regression.py` 应全绿

## 何时使用

- "帮我设计一个验证 X 的实验/研究方案""这些样本/被试怎么分组"
- "该用什么设计——对照？随机？区组？析因？交叉？整群？"
- "样本量要多大 / 帮我算功效 / effect size 怎么定"
- "怎么避免混杂 / 批次效应 / 伪重复"
- "规划机器学习消融实验 / 基准对比 / 实验矩阵"
- "定一份采集前的统计分析计划 / 预注册"

## Fisher 三原则（每个好设计的地基）

- **随机化 Randomization**：随机分配处理，让已知/未知混杂在期望上均衡——这是把"比较"变成"因果"的关键。
- **重复 Replication**：在**正确的层级**上独立重复，才能估计变异。最常见致命错误是**伪重复**：
  3 只小鼠各测 100 个细胞，对施加在小鼠上的处理而言 n=3（小鼠），不是 n=300（细胞）。
- **区组/局部控制 Blocking**：把相似单元按批次/日期/site 分组、组内随机化，把这部分噪声从误差项里移走。

## 工作流（先设计，后采集）

1. **陈述问题、单元、响应**：什么被随机化？测什么？真正独立重复在哪个层级？——这决定一切。
2. **列出干扰因素**（批次、日期、site、操作者、板位），逐一计划区组/分层/随机化。
3. **选设计类型**（用下面的决策树）。
4. **在正确层级定重复数**，并用样本量/功效模块算 n。
5. **生成随机化/DOE 布局**（用 `scripts/randomization.py`、`scripts/doe_designs.py`，seeded 可归档复现）。
6. **随机化运行/处理顺序**与板位/批次位置，防时间漂移与边缘效应。
7. **锁定分析计划并预注册**，让分析是验证性的而非事后自由发挥（用 `templates/preregistration.md`）。
8. **让分析匹配设计**：区组/分层/整群/嵌套必须进入模型（交给数据分析技能）。

## 设计类型决策树

```
你要学到什么？
├─ 比较少数预设条件 (A vs B vs C)？
│   ├─ 单元独立，有已知干扰因素(日/批/site)？ → 完全随机 或 随机区组设计
│   ├─ 每个单元可依次接受所有条件(可洗脱)？   → 交叉/重复测量设计(功效高，防残留效应)
│   └─ 只能随机化群体而非个体(学校/诊所)？    → 整群随机设计(在群体层级分析，防伪重复)
├─ 筛选很多因素(5+)找出关键的少数？          → 部分析因 / Plackett-Burman 筛选设计
├─ 量化少数因素的主效应+交互作用？            → 全 2^k 析因设计
├─ 找到使响应最优的设置(有曲率)？             → 响应面设计: 中心复合 / Box-Behnken
└─ 在连续空间探索仿真/计算模型？              → 空间填充设计: 拉丁超立方
```

选定后用脚本生成可复现布局（seeded、运行顺序随机化、可导出 CSV）：

```bash
python scripts/randomization.py block --n 60 --arms treatment,control --seed 42
python scripts/randomization.py stratified --strata siteA:30,siteB:30 --arms drug,placebo --ratio 2,1 --seed 42
python scripts/doe_designs.py full2 --factor temp:20,60 --factor conc:1,10 --factor pH:6,8 --seed 42
python scripts/doe_designs.py lhs --factor temp:20,60 --factor conc:1,10 --n 8 --seed 42
```

## 样本量与统计功效（可直接执行）

功效分析四参数——效应量、显著性 α、功效(1-β)、样本量 N——固定其三求第四。默认 α=0.05、power=0.80。
**只做 a priori（采集前）功效分析；事后功效分析是循环论证、无意义的。**

本技能附带脚本 `scripts/power.py`（封装 statsmodels），直接算：

```bash
python scripts/power.py ttest --effect 0.5 --alpha 0.05 --power 0.80      # 两样本 t 检验每组 N
python scripts/power.py anova --effect 0.25 --alpha 0.05 --power 0.80 --k 4
python scripts/power.py power --test ttest --effect 0.5 --n 50            # 反解：给定 N 的功效
```

效应量参考（不要默认套"medium"，应基于文献/预实验/最小实际意义效应 SESOI）：

| 检验 | 小 | 中 | 大 |
|---|---|---|---|
| Cohen's d (t) | 0.2 | 0.5 | 0.8 |
| Cohen's f (ANOVA) | 0.10 | 0.25 | 0.40 |
| r (相关) | 0.1 | 0.3 | 0.5 |

报告模板：`基于[文献/预实验/meta]的预期效应 [d/f=X]，α=.05，power=.80，[检验]所需样本量为[N]；
考虑约[X]%脱落，实际招募[最终N]。`

## 计算/机器学习实验（消融与基准）

- **单变量隔离**：每次消融只改一个因素，其余(seed/data split/epochs/硬件)固定并记录。
- **实验矩阵**：因素少(≤3)用全析因；因素多先跑单因素消融再组合优胜者；太贵用拉丁方抽代表组合。
- **资源估算**：总运行数=各因素水平数之积；GPU 小时=运行数×单次时长；超预算就提示优先级。
- **采集前定分析**：主/次指标、显著性检验(配对 t / bootstrap CI)、失败运行处理、可视化。

## 毁掉研究的结构性错误（分析救不回，只能靠设计）

1. **伪重复**：把同一单元的重复测量当独立重复。
2. **干扰因素混杂**：处理组周一跑、对照组周二跑 → 处理与"日期"混杂。
3. **随机化缺失/破坏**：便利分配让混杂溜进来。
4. **无恰当对照**：缺并行对照/载体对照/盲法，分不清处理效应与时间/安慰剂/操作效应。
5. **批次效应误当生物学**：组学里尤甚，跨批次随机/区组处理，别让批次与条件对齐。
6. **板边缘/位置效应**：别把对照全放第一列。
7. **部分析因忽略混叠**：低分辨率设计把主效应与交互混叠，先看 alias 结构再下结论。
8. **无曲率却做优化**：两水平析因测不出曲面最优，用响应面。

## 质量红线

- **只做采集前设计**；数据清洗、统计运行、组学/回归执行属数据分析技能。
- **样本量必须有依据**：效应量来自文献/预实验/SESOI，不拍脑袋。
- **显式标注偏倚与可行性/伦理限制**。
- **默认建议预注册**，降低事后自由度。

## 与其它技能的边界

- 把想法拆成数据任务/输入输出/baseline（偏 ML 工程）→「数据处理(baseline)」。
- 采集后统计执行（检验/效应量/CI/多重比较校正）→「统计分析」（skills/statistical-analysis）；组学专业管线（scanpy/Seurat/DESeq）另找专门工具。
- 生成研究假设/机制线索 →「假设生成」（本技能承接其后"如何验证"）。
