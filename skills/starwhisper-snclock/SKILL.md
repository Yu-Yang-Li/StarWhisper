---
name: starwhisper-snclock
description: Screen SN Clock explosion-age predictions into a defensible young-supernova shortlist. Use for 超新星时钟, explosion epoch, young SN candidates, H3 age quantiles, within-2-days selection, TNS candidate ranking, or reading snclock prediction tables. Computes ages, applies tiers, and audits provenance; never claims a classification or a discovery.
license: Apache-2.0
---

# SN Clock 候选筛选

把一张爆发年龄预测表变成**可以拿去申请观测时间的候选清单**，并且每一条都带得住追问。

> 所有数字由 `scripts/screen_snclock.py` 计算（stdlib，无网络）。模型不徒手估年龄、不徒手推时间差。
> 每次输出都会附上覆盖范围、输入快照留存情况和区间口径三条边界。

默认读仓库里的 [`snclock/`](../../snclock/README.md)。别的表用 `--csv` 指过去，列名相同即可。

## 何时使用

- "这批源里哪些还在爆发两天内""挑几个今晚值得跟的"
- "按年轻程度排个序""只要保守口径也算年轻的"
- "这个候选表能不能直接写进申请书 / 发 AstroNote"
- "这张表覆盖了多久""这些预测是什么时候生成的，过期没有"

## 工作流

1. **先 `describe`**：看清行数、发现日期跨度、两档 tier 的比例、置信标签分布。跨度和 `scope_note_cn` 声明的筛选窗口往往不一致，这一步就要发现。
2. **`audit` 定证据强度**：哪些行的输入快照没落盘、哪些预测已经过期、tier 标志和布尔列是否自洽。证据弱的行照样可以用，但引用时必须说明。
3. **`rank` 或 `screen` 出清单**：`rank` 按 q50 从小到大；`screen` 按 tier / 年龄 / 置信 / 红移过滤。
4. **`window` 算"现在还年不年轻"**：年龄会随时间涨，`age_now = 发现时年龄 + 已过去的时间`。昨天的清单今天就不成立了。
5. **按 `templates/observing_shortlist.md` 写结论**：先写口径和边界，再写清单，不要倒过来。

## 选哪一档

```
要多保守？
├─ 对外发提案、写 AstroNote        → --tier strict   （q84 也 ≤ 2 天，本表只有 2 个）
├─ 内部排观测、愿意承担假阳性       → --tier q50      （q50 ≤ 2 天，q84 越界）
└─ 全量看分布                      → --tier any（默认）

还要更硬的证据？
├─ 只要输入快照留存的              → --exclude-weak-provenance
├─ 只要有宿主红移的                → --require-redshift
└─ 只要模型自评不低的              → --min-confidence MEDIUM
```

`--tier strict` 叠 `--exclude-weak-provenance` 常常直接筛空。筛空就报筛空，这是正确结果，不要放宽条件凑数。

## 脚本用法

```powershell
python scripts/screen_snclock.py describe
python scripts/screen_snclock.py audit --asof 2026-08-16T04:10:00Z --stale-after-days 7
python scripts/screen_snclock.py rank   --top 10 --min-confidence MEDIUM
python scripts/screen_snclock.py screen --tier strict --require-redshift --asof 2026-08-16T04:00:00Z
python scripts/screen_snclock.py window --within-days 2 --asof 2026-08-16T04:00:00Z --conservative
python scripts/screen_snclock.py rank   --csv path\to\other_table.csv --json
```

加 `--json` 输出结构化结果，`must_state` 字段就是必须转述的边界。

## 报告纪律

- **年龄是估计，不是观测量。** 写"H3 q50 约 1.0 天（q16–q84 0.16–1.88）"，不写"爆发于 1 天前"。
- **区间口径要带上。** fold dispersion 得到的区间，不是后验置信区间。
- **不做分类。** 年龄估计推不出 Ia / II / Ibc，那要光谱。
- **不写"发现"。** 表里的源都已经在 TNS 上。
- **覆盖范围照抄。** 声明的筛选窗口和资产实际覆盖不一致时，两个都写出来。
- **筛空就是结论。** 不要为了凑数放宽 tier 或改口径。

## 相关

完整的抓取、强制测光、可见性和三日筛选工作流在 [SitianClaw](https://github.com/Yu-Yang-Li/SitianClaw) 的 `snc-*` 技能，不要在这里重写。本仓库的真假源样机是 `GOTTA_Prototype/`。排夜次可见性和曝光预算用 `starwhisper-night-plan`。
