# SN Clock 预测表

虚拟司天早期科学应用的一次生产输出：对 TNS 上的暂现源估计**爆发后年龄**，并筛出发现时可能仍在爆发两天内的源。

配套技能：[`skills/starwhisper-snclock`](../skills/starwhisper-snclock/SKILL.md)。

| 文件 | 内容 |
| --- | --- |
| `snclock_recent_6m_within2d_sources_20260816.csv` | 22 个源，发现日期 2026-07-02 至 2026-08-15，模型 `h3_canonical` |

## 这张表是什么

每行是一个源在**发现时刻**的年龄估计，给三个分位：`h3_age_q16_days` / `q50` / `q84`，单位是天。筛选分两档：

| `selection_tier` | 判据 | 本表数量 |
| --- | --- | ---: |
| `strict_q84_within_2d` | 连 q84 都 ≤ 2 天，保守口径也算年轻 | 2 |
| `q50_within_2d_only` | q50 ≤ 2 天，但 q84 越界 | 20 |

置信标签 `model_confidence_label` 为 LOW 17 个、MEDIUM 5 个，没有 HIGH。`host_redshift` 全部有值。

## 用之前必须知道的三件事

**一、覆盖范围不等于半年全量。** 表里 `scope_note_cn` 写明：筛选窗口是 2026-02-16 至 2026-08-16，但生产预测资产实际只覆盖发现日期 2026-06-24 至 2026-08-15。这张 within-2d 子集的实际跨度是 07-02 至 08-15。不能拿它当"近半年全部 TNS 年轻源"。

**二、多数行的输入快照没有落盘。** 22 行里有 18 行 `input_mode_record` 是 `not_persisted_in_historical_snapshot`，即当时的输入没有留存，无法逐行回放。只有 4 行是 `tns_plus_forced_nightly`。引用具体某个源时要带上这一点。

**三、区间是 fold dispersion 口径。** 表里 `prediction_warning` 写明：H3 区间来自 fold dispersion，而 canonical H3 本身是点估计集成。所以这个区间不是严格意义上的后验置信区间。

## 这张表不是什么

- 不是光谱分类。年龄估计说不了 Ia / II / Ibc。
- 不是发现。这些源都已经在 TNS 上，本表只做年龄排序。
- 不是 broker 警报，也不是观测指令。要接观测得自己排可见性和设备。
- 不是 NGSS 的运行记录，也和 [`../explore/`](../explore/README.md) 的合成实验无关。

## 怎么读

```powershell
python ..\skills\starwhisper-snclock\scripts\screen_snclock.py describe
python ..\skills\starwhisper-snclock\scripts\screen_snclock.py rank --top 10
python ..\skills\starwhisper-snclock\scripts\screen_snclock.py audit
```

可运行的完整工作流在 [SitianClaw](https://github.com/Yu-Yang-Li/SitianClaw)（`snc-*` 系列）。本目录只放已经产出的表和读表工具。
