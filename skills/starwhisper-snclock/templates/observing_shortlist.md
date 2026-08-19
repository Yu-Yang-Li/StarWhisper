# 年轻超新星候选清单

**表**：`<csv 文件名>`　**口径时刻**：`<asof UTC>`　**模型**：`<model_id>`

## 筛选条件

<照抄实际用的命令行，例如 `screen --tier strict --require-redshift --asof ...`>

从 `<n_input>` 行中选出 `<n_selected>` 行。

## 边界（先读这段）

- 覆盖范围：`<scope_note_cn 原文>`
- 输入快照：`<weak>/<n_selected>` 行的输入未落盘（`not_persisted_in_historical_snapshot`），无法逐行回放
- 区间口径：H3 区间基于 fold dispersion，canonical H3 本身是点估计集成，不是后验置信区间
- 年龄为模型估计，非光谱分类，非新发现

## 清单

| 源 | TNS | 发现 (UTC) | q16 / q50 / q84 (天) | tier | 置信 | z | 当前年龄 q50 |
| --- | --- | --- | --- | --- | --- | ---: | ---: |
| | | | | | | | |

## 结论

<一句话说明这批源适合做什么观测，以及不能据此声称什么。>

<如果筛空：写明筛空，并说明是哪个条件卡掉的，不要放宽条件重筛。>
