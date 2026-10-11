# 年轻超新星候选清单

**表**：`<csv 文件名>`　**口径时刻**：`<asof UTC>`　**模型**：`<model_id>`

## 筛选条件

<记录实际命令。筛选指定时刻的年轻候选时，使用 `window --tier strict --require-redshift --within-days 2 --conservative --asof ...`；`screen` 的 tier 仍按发现时年龄判断。>

从 `<n_input>` 行中选出 `<n_selected>` 行。

## 数据局限

- 覆盖范围：`<scope_note_cn 原文>`
- 输入快照：所选行中 `<weak>` 行未留存输入（`not_persisted_in_historical_snapshot`），无法逐行回放
- 区间口径：H3 区间基于 fold dispersion，canonical H3 本身是点估计集成，不是后验置信区间
- 年龄为模型估计，非光谱分类，非新发现

## 清单

| 源 | TNS | 发现 (UTC) | 发现时 q16 / q50 / q84 (天) | tier | 置信 | z | 指定时刻 q50 (天) |
| --- | --- | --- | --- | --- | --- | ---: | ---: |
| | | | | | | | |

## 结论

<说明这批候选的用途、估计的不确定性，以及安排观测前仍需核对的条件。>

<结果为空时，保留筛选条件并说明原因，不为凑候选而放宽标准。>
