---
name: starwhisper-snclock
description: 读取 SN Clock 超新星爆发年龄预测表，按年龄、置信标签和数据留存情况筛选候选。用于 TNS 源排序、两天内筛选和预测表核查；不做光谱分类或新源发现。
license: Apache-2.0
---

# SN Clock 候选筛选

从爆发年龄预测表中筛选年轻超新星候选，并说明预测时间、数据来源和估计的不确定性。

计算使用 `scripts/screen_snclock.py`，只依赖 Python 标准库，不访问网络。报告中应保留表的覆盖范围、输入数据留存情况和区间定义。

默认读取仓库里的 [`projects/snclock/`](../../projects/snclock/README.md)。安装到其他位置后，需设置 `STARWHISPER_ROOT` 或用 `--csv` 指定表格；自备表格需使用相同列名。

`rank` 和 `screen` 按发现时的年龄与 tier 筛选。`screen --asof` 只增加更新后的年龄字段，不重新筛选 tier；查询指定时刻仍处于年龄窗口内的源，应使用 `window`。

## 何时使用

- "这批源里哪些还在爆发两天内""挑几个今晚值得跟的"
- "按年轻程度排个序""只要保守口径也算年轻的"
- "这个候选表能不能直接写进申请书 / 发 AstroNote"
- "这张表覆盖了多久""这些预测是什么时候生成的，过期没有"

## 工作流

1. **先 `describe`**：看清行数、发现日期跨度、两档 tier 的比例、置信标签分布。跨度和 `scope_note_cn` 声明的筛选窗口往往不一致，这一步就要发现。
2. **`audit` 检查数据留存**：查看输入快照、预测是否过期，以及 tier 标志和布尔列是否一致。引用缺少输入快照的行时，需说明无法逐行回放。
3. **`rank` 或 `screen` 筛选候选**：`rank` 按 q50 从小到大；`screen` 按 tier / 年龄 / 置信 / 红移过滤。
4. **`window` 更新年龄**：`age_now = 发现时年龄 + 已过去的时间`。筛选当前候选时，应计算到指定时刻，不能直接沿用发现时的年龄。
5. **按 `templates/observing_shortlist.md` 写结论**：先写口径和边界，再写清单，不要倒过来。

## 选哪一档

```
要多保守？
├─ 对外发提案、写 AstroNote        → --tier strict   （q84 也 ≤ 2 天，本表只有 2 个）
├─ 内部排观测、愿意承担假阳性       → --tier q50      （q50 ≤ 2 天，q84 越界）
└─ 全量看分布                      → --tier any（默认）

需要限制数据来源或置信标签时：
├─ 只要输入快照留存的              → --exclude-weak-provenance
├─ 只要有宿主红移的                → --require-redshift
└─ 只要模型自评不低的              → --min-confidence MEDIUM
```

`--tier strict` 与 `--exclude-weak-provenance` 同时使用时，结果可能为空。报告应保留筛选条件，不为得到候选而放宽条件。

## 脚本用法

下列路径相对于本技能目录。`--csv` 和 `--json` 放在子命令前。带 `--asof` 的示例用于回看历史预测；筛选当前候选时需换用相应时刻和最新数据。

```powershell
python scripts/screen_snclock.py describe
python scripts/screen_snclock.py audit --asof 2026-08-16T04:10:00Z --stale-after-days 7
python scripts/screen_snclock.py rank   --top 10 --min-confidence MEDIUM
python scripts/screen_snclock.py screen --tier strict --require-redshift --asof 2026-08-16T04:00:00Z
python scripts/screen_snclock.py window --within-days 2 --asof 2026-08-16T04:00:00Z --conservative
python scripts/screen_snclock.py --csv path\to\other_table.csv --json rank
```

加 `--json` 输出结构化结果，`must_state` 字段就是必须转述的边界。

## 结果说明

- **年龄是估计，不是观测量。** 写"H3 q50 约 1.0 天（q16–q84 0.16–1.88）"，不写"爆发于 1 天前"。
- **区间口径要带上。** fold dispersion 得到的区间，不是后验置信区间。
- **不做分类。** 年龄估计推不出 Ia / II / Ibc，那要光谱。
- **不写"发现"。** 表里的源都已经在 TNS 上。
- **覆盖范围照抄。** 声明的筛选窗口和资产实际覆盖不一致时，两个都写出来。
- **筛空就是结论。** 不要为了凑数放宽 tier 或改口径。

## 相关

抓取、强制测光、可见性和三日筛选工作流见 [SitianClaw](https://github.com/Yu-Yang-Li/SitianClaw) 的 `snc-*` 技能。观测配置与曝光预算检查见 [`starwhisper-night-plan`](../starwhisper-night-plan/SKILL.md)。
