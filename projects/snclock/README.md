# SN Clock 预测表

本目录保存虚拟司天早期科学应用的一次历史预测，用于估计 TNS 暂现源的爆发后年龄。附带的 22 个源可以直接离线排序和筛选，不需要模型权重或额外观测数据。

[项目目录](../README.md) · [研究文档](../../docs/research.md) · [读表工具说明](../../skills/starwhisper-snclock/SKILL.md)

## 直接读表

需要 Python 3.10 或更新版本，只依赖标准库。从本目录（`projects/snclock/`）运行：

```bash
python ../../skills/starwhisper-snclock/scripts/screen_snclock.py describe
python ../../skills/starwhisper-snclock/scripts/screen_snclock.py rank --top 10
python ../../skills/starwhisper-snclock/scripts/screen_snclock.py audit
```

`describe` 概览数据，`rank` 按发现时的年龄排序，`audit` 检查输入留存和预测范围。它们读取下表，不会重新预测年龄。

## 表格里是什么

| 文件 | 内容 |
| --- | --- |
| [`snclock_recent_6m_within2d_sources_20260816.csv`](snclock_recent_6m_within2d_sources_20260816.csv) | 22 个源，发现日期 2026-07-02 至 2026-08-15，模型 `h3_canonical` |

每行记录一个源在发现时刻的年龄估计，以三个分位表示：`h3_age_q16_days` / `q50` / `q84`，单位为天。筛选分为两档：

| `selection_tier` | 判据 | 本表数量 |
| --- | --- | ---: |
| `strict_q84_within_2d` | 连 q84 都 ≤ 2 天，保守口径也算年轻 | 2 |
| `q50_within_2d_only` | q50 ≤ 2 天，但 q84 越界 | 20 |

置信标签 `model_confidence_label` 为 LOW 17 个、MEDIUM 5 个，没有 HIGH。`host_redshift` 全部有值。

## 数据局限

文件名中的“近半年”是筛选窗口，不代表半年全量数据。`scope_note_cn` 记录的窗口为 2026-02-16 至 2026-08-16，实际预测只覆盖发现日期 2026-06-24 至 2026-08-15；本表的 within-2d 子集则从 07-02 到 08-15。

22 行里有 18 行的 `input_mode_record` 为 `not_persisted_in_historical_snapshot`，当时的输入没有留存，无法逐行回放；另外 4 行为 `tns_plus_forced_nightly`。引用单个源时，需要说明其输入留存情况。

`prediction_warning` 说明，H3 区间来自 fold dispersion，而 canonical H3 本身是点估计集成。这些区间不应当作严格意义上的后验置信区间。

## 换用当前数据

`rank` 和 `screen` 使用发现时的年龄；`window` 才按当前或 `--asof` 指定时刻重新检查年龄窗口。例如，从本目录运行：

```bash
python ../../skills/starwhisper-snclock/scripts/screen_snclock.py window --within-days 2 --conservative
```

对这份历史表，当前窗口返回空结果是正常的：发现时年轻的源，今天未必仍然年轻。筛选当前候选体需要最新预测表，可将 `--csv` 放在子命令之前：

```bash
python ../../skills/starwhisper-snclock/scripts/screen_snclock.py --csv /path/to/current_predictions.csv rank --top 10
```

新表需遵循[读表工具的字段说明](../../skills/starwhisper-snclock/SKILL.md)。完整预测和数据获取流程在 [SitianClaw](https://github.com/Yu-Yang-Li/SitianClaw) 的 `snc-*` 技能中维护，本目录未附带其模型与完整输入。

## 怎样使用结果

- 年龄估计用于排序；Ia / II / Ibc 等分类仍需光谱等证据。
- 这些源已收录于 TNS，表格不构成新源发现。
- 表格不提供 broker 警报或观测指令；安排观测时需另行检查可见性和设备条件。
- 这份历史预测与 [NGSS](../ngss/README.md) 运行记录、[Explore 合成实验](../explore/README.md)分别使用。
