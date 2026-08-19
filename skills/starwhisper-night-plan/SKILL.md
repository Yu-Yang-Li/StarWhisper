---
name: starwhisper-night-plan
description: Validate an NGSS observing configuration, compute the night exposure budget, and lint a target list before anyone touches hardware. Use for observe_config.json, StarWhisper Telescope, NGSS, NINA, 夜计划, moon distance constraints, exposure count/time/wait, how many targets fit in a night, or the NGSS endpoint contract. Never sends a command.
license: Apache-2.0
---

# 夜计划检查

在有人碰硬件之前，把配置、时间预算和目标表的问题找出来。

> 所有数字由 `scripts/plan_night.py` 计算（stdlib）。脚本不开任何 socket：没有 HTTP、MQTT、FTP、NINA。
> 这是 [StarWhisper Telescope](https://doi.org/10.1038/s44172-025-00520-4) 论文对应的 [`NGSS/`](../../NGSS/README.md) 栈，不是 Explore 合成环境，也不是虚拟司天。

配置按 `--config` → 仓库 `NGSS/observe_config.json` → 内置样例的顺序解析，输出会写明用的是哪一个。

## 何时使用

- "这份 observe_config 有没有问题""d_moon 设 15 度够不够"
- "一夜能排多少个目标""3×120 秒够不够用"
- "这批目标表能不能直接交上去"
- "NGSS 有哪些接口，哪些会真的动望远镜"

## 工作流

1. **`check-config`**：字段缺失、取值越界、以及配置本身不自洽的地方。有 error 就先修，不要往下排。
2. **`budget`**：算单目标耗时和一夜容量。这一步定下"能排几个"，后面所有讨论都以它为准。
3. **`lint-targets`**：目标表列名、重复、RA/Dec 越界，以及数量是否超容量。
4. **`endpoints`**：要动真栈时，先看清哪些路由是 read、哪些 mutate、哪些直接落到硬件。

## 时间预算怎么算

```
单目标 = 滤镜数 × (曝光张数 × 曝光时长 + (张数-1) × 间隔) + 转向开销
一夜容量 = 各时间窗小时数之和 ÷ 单目标
```

默认 `--slew-seconds 60`。这是**几何容量**：不含高度角、月亮规避、天气、转向路径。真实可排数一定更少，所以容量只能当上界用，不能当承诺。

## 脚本用法

```powershell
python scripts/plan_night.py check-config
python scripts/plan_night.py budget --slew-seconds 90
python scripts/plan_night.py lint-targets --targets targets.csv
python scripts/plan_night.py endpoints
python scripts/plan_night.py check-config --config path\to\observe_config.json --json
```

目标表需要一列目标名（`name` / `objname` / `target` / `source_name`）和一对坐标（`ra` / `dec` 及其常见变体）。`check-config` 和 `lint-targets` 有 error 时退出码为 1，可以直接进 CI。

## 接手真实栈之前

`endpoints` 把路由分三类。`/ftp_transfer` 和 `/manipulate_nina/{action}` 会落到硬件，本技能任何情况下都不调用它们。

要真跑，前置条件缺一不可，缺了就直说缺了，不要绕过：NINA、`FMoraes.NINA.SitesPlugin.dll`、x-opstep、FTP/MQTT 通道、望远镜连接。启动命令是从 `NGSS/` 执行 `uvicorn src.app.app2:app --reload`。

安全联锁优先于任何 agent 动作。不要从一台不是那个栈的笔记本上发指令。

## 报告纪律

- 本地跑通导入不等于上了天。别把 `check-config` 通过写成"夜计划已验证"。
- 容量是上界，不是排班结果。
- 不要拿 Explore 的四策略表当 NGSS 生产指标。
- `Pachong.py` 是抓 TNS 公开页的历史脚本，不是稳定 API。

## 相关

候选源筛选用 `starwhisper-snclock`；策略对比的预注册门槛用 `starwhisper-explore`。
