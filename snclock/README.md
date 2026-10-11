# SN Clock 预测表

本目录保存虚拟司天早期科学应用的一次历史预测：估计 TNS 暂现源的**爆发后年龄**，查看哪些源在发现时可能仍处于爆发两天内。

配套技能：[`skills/starwhisper-snclock`](../skills/starwhisper-snclock/SKILL.md)。

| 文件 | 内容 |
| --- | --- |
| `snclock_recent_6m_within2d_sources_20260816.csv` | 22 个源，发现日期 2026-07-02 至 2026-08-15，模型 `h3_canonical` |

## 字段与筛选

每行记录一个源在**发现时刻**的年龄估计，以三个分位表示：`h3_age_q16_days` / `q50` / `q84`，单位为天。筛选分为两档：

| `selection_tier` | 判据 | 本表数量 |
| --- | --- | ---: |
| `strict_q84_within_2d` | 连 q84 都 ≤ 2 天，保守口径也算年轻 | 2 |
| `q50_within_2d_only` | q50 ≤ 2 天，但 q84 越界 | 20 |

置信标签 `model_confidence_label` 为 LOW 17 个、MEDIUM 5 个，没有 HIGH。`host_redshift` 全部有值。

## 数据局限

**一、覆盖范围不等于半年全量。** 表里 `scope_note_cn` 写明：筛选窗口是 2026-02-16 至 2026-08-16，但生产预测资产实际只覆盖发现日期 2026-06-24 至 2026-08-15。这张 within-2d 子集的实际跨度是 07-02 至 08-15。不能拿它当"近半年全部 TNS 年轻源"。

**二、多数行的输入快照没有落盘。** 22 行里有 18 行 `input_mode_record` 是 `not_persisted_in_historical_snapshot`，即当时的输入没有留存，无法逐行回放。只有 4 行是 `tns_plus_forced_nightly`。引用具体某个源时要带上这一点。

**三、区间是 fold dispersion 口径。** 表里 `prediction_warning` 写明：H3 区间来自 fold dispersion，而 canonical H3 本身是点估计集成。所以这个区间不是严格意义上的后验置信区间。

## 适用范围

- 年龄估计用于排序；Ia / II / Ibc 等分类仍需光谱等证据。
- 这些源已收录于 TNS，表格不构成新源发现。
- 表格不提供 broker 警报或观测指令；安排观测时需另行检查可见性和设备条件。
- 这份结果与 NGSS 运行记录、[Explore 合成实验](../explore/README.md) 分开使用。

## 使用

```powershell
python ..\skills\starwhisper-snclock\scripts\screen_snclock.py describe
python ..\skills\starwhisper-snclock\scripts\screen_snclock.py rank --top 10
python ..\skills\starwhisper-snclock\scripts\screen_snclock.py audit
python ..\skills\starwhisper-snclock\scripts\screen_snclock.py window --within-days 2 --conservative
```

`rank` 和 `screen` 使用发现时的年龄；`window` 才按当前或 `--asof` 指定时刻重新检查年龄窗口。历史表中的年轻候选不能直接作为今晚仍年轻的候选。

完整工作流见 [SitianClaw](https://github.com/Yu-Yang-Li/SitianClaw) 的 `snc-*` 技能。本目录保存预测表，读表工具见 [`starwhisper-snclock`](../skills/starwhisper-snclock/SKILL.md)。
