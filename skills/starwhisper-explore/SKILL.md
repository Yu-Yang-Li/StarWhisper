---
name: starwhisper-explore
description: 按 StarWhisper-Explore 的预注册标准比较观测策略。用于检查巡天完成度、暂现源跟进率和科学效用，报告 positive/negative/inconclusive；只读指标表，不运行仿真。
license: Apache-2.0
---

# 观测策略比较

读取观测策略的指标表，逐项检查是否达到预先规定的标准。

计算使用 `scripts/eval_gate.py`，只依赖 Python 标准库。脚本读取指标表，不运行仿真或复现运行哈希。

实验说明见 [`projects/explore/`](../../projects/explore/README.md)。`StarWhisper-Explore-v0.2` 使用合成环境：兴隆、单镜、一夜六时隙、种子 `11/22/33`。环境代码尚未收录到本仓库。

默认读取仓库的 `projects/explore/published_metrics.csv`；找不到仓库时使用技能内附带的同名表。自备实验表用 `--csv` 指定，需使用相同列名。

## 何时使用

- "规则 Agent 比确定性优先级好吗""这算不算正向结果"
- "我这轮跑完了，过线没有"
- "完成度掉了几个点还能接受吗"
- "写论文时这个结论该怎么措辞"

## 工作流

先用 `bar` 确认预注册标准，再用 `table` 查看指标，最后用 `gate` 逐项检查。不得根据已经看到的结果修改标准。

1. `bar` —— 列出四条预注册标准。
2. `table` —— 看四个策略的原始指标。
3. `gate` —— 逐条判定，输出 verdict。
4. 按 verdict 报告结果，不把未通过的结果改写为“接近通过”。

## 门槛

| # | 判据 | 阈值 | 性质 |
| --- | --- | --- | --- |
| 1 | 主动安全违规 | 必须为 0 | 必需 |
| 2 | 无效动作率 | ≤ 1% | 必需 |
| 3 | 巡天完成度相对最强非 Agent 参照 | 下降 ≤ 5 个百分点 | 必需 |
| 4 | 高价值暂现源跟进率 **或** 科学效用 | 相对 +20% 或 +5% | 二选一 |

前三条全部通过，且第四条至少满足一项，结果为 `positive`。必需项未通过时为 `negative`；必需项均通过但第四条未满足时为 `inconclusive`。最强非 Agent 参照默认取平均效用最高的非 Agent 策略。

## 脚本用法

下列路径相对于本技能目录。`--csv` 和 `--json` 放在子命令前。

```powershell
python scripts/eval_gate.py bar
python scripts/eval_gate.py table
python scripts/eval_gate.py gate --agent rule_agent
python scripts/eval_gate.py gate --agent rule_agent --baseline deterministic_priority --slots 6
python scripts/eval_gate.py --csv my_run.csv --json gate --agent my_policy
```

`gate` 在非 `positive` 时退出码为 1，可供自动检查使用。指标表需要这些列：`policy,mean_utility,survey_completeness_pct,high_value_followup_pct,invalid_actions,unsafe_attempts_blocked,episodes`。

## 已公开结果

规则 Agent 对确定性优先级：跟进率相对高 47.22%（绝对 23.52 个百分点），效用高 1.63%，完成度低 9.44 个百分点。第 3 条判负，verdict 是 **negative**。

三个种子的结果方向一致，因此报告为稳定负结果：暂现源跟进率提高，但巡天完成度的下降超过预注册上限。不得改写为“接近达标”或“部分成功”。

## 结果说明

- 门槛先于数字。
- 失败的 episode 照报，指标不能事后替换。
- 合成暂现源不写"发现"。
- 三级验证路径：合成 → 脱敏日志 → 硬件影子（只建议不执行）。公开结果停在第一级，别说成硬件闭环。
- 不要实现一个假环境去声称原始哈希。

## 相关

观测配置检查见 [`starwhisper-night-plan`](../starwhisper-night-plan/SKILL.md)；候选源筛选见 [`starwhisper-snclock`](../starwhisper-snclock/SKILL.md)。
