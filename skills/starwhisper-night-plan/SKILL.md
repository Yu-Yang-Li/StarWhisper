---
name: starwhisper-night-plan
description: 检查 NGSS 观测配置与目标表，计算曝光时间预算，并查看接口说明。用于 observe_config.json、夜间观测计划和 NINA 接口核查；不连接设备或发送指令。
license: Apache-2.0
---

# 夜计划检查

检查观测配置、曝光时间预算和目标表，供安排夜间观测时参考。

计算使用 `scripts/plan_night.py`，只依赖 Python 标准库，不建立 HTTP、MQTT、FTP 或 NINA 连接。配置来自 [StarWhisper Telescope](https://doi.org/10.1038/s44172-025-00520-4) 对应的 [`projects/ngss/`](../../projects/ngss/README.md)，与 Explore 合成实验和虚拟司天分开使用。

优先使用 `--config` 指定的配置，否则读取仓库的 `projects/ngss/observe_config.json`（旧仓库为 `NGSS/observe_config.json`）。找不到仓库配置时读取内置样例，输出会写明来源。支持 NGSS 的平铺字段和技能样例的分组字段。

## 何时使用

- "这份 observe_config 有没有问题"、"d_moon 设 15 度够不够"
- "一夜能排多少个目标"、"3×120 秒够不够用"
- "这批目标表能不能直接交上去"
- "NGSS 有哪些接口，哪些会真的动望远镜"

## 工作流

1. **`check-config`**：字段缺失、取值越界、以及配置本身不自洽的地方。有 error 就先修，不要往下排。
2. **`budget`**：估算单目标耗时和一夜容量，供检查目标表数量时参考。
3. **`lint-targets`**：目标表列名、重复、RA/Dec 越界，以及数量是否超容量。
4. **`endpoints`**：查看只读接口、修改状态的接口和设备控制接口，分别标记为 read、mutate、hardware。

## 时间预算怎么算

```
NGSS 配置：单目标秒数 = 滤镜数 × 曝光张数 × 曝光秒数 + WaitTime × 60
分组样例：单目标秒数 = 滤镜数 × (曝光张数 × 曝光秒数 + (张数-1) × 间隔分钟 × 60) + 额外开销秒数
一夜容量 = 各时间窗小时数之和 × 3600 ÷ 单目标秒数，向下取整
```

NGSS 的 `WaitTime` 是每个目标结束后的指向与读出冗余，按观测代码逐目标计算。分组样例使用 `exposure.wait` 表示曝光间隔，额外开销默认为 60 秒；`--slew-seconds` 可覆盖两种格式的逐目标额外开销。预算不含高度角、月亮规避、天气和转向路径约束，实际耗时需结合观测日志核对。

## 脚本用法

下列路径相对于本技能目录。`--config`、`--slew-seconds` 和 `--json` 放在子命令前；`targets.csv` 和配置路径需替换为实际文件。

```powershell
python scripts/plan_night.py check-config
python scripts/plan_night.py --slew-seconds 90 budget
python scripts/plan_night.py lint-targets --targets targets.csv
python scripts/plan_night.py endpoints
python scripts/plan_night.py --config path\to\observe_config.json --json check-config
```

目标表需要一列目标名（`name` / `objname` / `target` / `source_name`）和一对坐标（`ra` / `dec` 及其常见变体）。`check-config` 和 `lint-targets` 有 error 时退出码为 1，可以直接进 CI。

## 对应的观测服务

`endpoints` 把路由分三类。`/ftp_transfer` 和 `/manipulate_nina/{action}` 会落到硬件，本技能任何情况下都不调用它们。

NGSS 服务依赖 NINA、`FMoraes.NINA.SitesPlugin.dll`、x-opstep、FTP/MQTT 通道和望远镜连接。服务启动命令为在 `projects/ngss/` 中执行 `uvicorn src.app.app2:app --reload`。本技能只检查文件，不启动这些服务。

安全联锁优先于任何 agent 动作。设备指令只能由已配置并获得授权的台站环境执行。

## 结果说明

- `check-config` 通过只说明配置检查通过，不能写成“夜间观测已验证”。
- 容量取决于公式中的时间假设，不能直接作为实际排班结果。
- 不要拿 Explore 的四策略表当 NGSS 生产指标。
- `Pachong.py` 是抓 TNS 公开页的历史脚本，不是稳定 API。

## 相关

候选源筛选见 [`starwhisper-snclock`](../starwhisper-snclock/SKILL.md)；观测策略比较见 [`starwhisper-explore`](../starwhisper-explore/SKILL.md)。
