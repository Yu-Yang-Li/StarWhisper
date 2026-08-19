---
name: starwhisper-explore
description: Evaluate an observing-policy comparison against the StarWhisper-Explore pre-registered bar and return pass/fail per criterion. Use for 决策边界, survey completeness vs transient follow-up, policy A/B comparison, stable negative result, GOAI Explore, or judging whether an agent policy actually cleared its threshold. Reads a metrics table; never simulates nights.
license: Apache-2.0
---

# 决策边界门槛判定

判断一个观测策略**到底有没有过预注册的线**，而不是挑好看的数字讲故事。

> 判定由 `scripts/eval_gate.py` 计算（stdlib）。脚本只读指标表，不跑仿真、不复现运行哈希。
> 规格见 [`explore/`](../../explore/README.md)。`StarWhisper-Explore-v0.2` 是合成环境：兴隆、单镜、一夜六时隙、种子 `11/22/33`，**环境代码不在本仓库**。

默认读仓库的 `explore/published_metrics.csv`。自己的实验用 `--csv` 指过去，列名相同即可复用同一套判定。

## 何时使用

- "规则 Agent 比确定性优先级好吗""这算不算正向结果"
- "我这轮跑完了，过线没有"
- "完成度掉了几个点还能接受吗"
- "写论文时这个结论该怎么措辞"

## 工作流

**顺序不能反。** 先 `bar` 把门槛说出来，再 `table` 看数字，最后 `gate` 判定。看完数字再定门槛就是 p-hacking。

1. `bar` —— 念出预注册门槛，四条一条不漏。
2. `table` —— 看四个策略的原始指标。
3. `gate` —— 逐条判定，输出 verdict。
4. 按 verdict 措辞，不要自己润色成"接近通过"。

## 门槛

| # | 判据 | 阈值 | 性质 |
| --- | --- | --- | --- |
| 1 | 主动安全违规 | 必须为 0 | 必需 |
| 2 | 无效动作率 | ≤ 1% | 必需 |
| 3 | 巡天完成度相对最强非 Agent 参照 | 下降 ≤ 5 个百分点 | 必需 |
| 4 | 高价值暂现源跟进率 **或** 科学效用 | 相对 +20% 或 +5% | 二选一 |

前三条全过、且第四条至少中一个，才是 `positive`。必需项挂了就是 `negative`；必需项都过但第四条两个都没中，是 `inconclusive`。最强非 Agent 参照默认取平均效用最高的非 Agent 策略。

## 脚本用法

```powershell
python scripts/eval_gate.py bar
python scripts/eval_gate.py table
python scripts/eval_gate.py gate --agent rule_agent
python scripts/eval_gate.py gate --agent rule_agent --baseline deterministic_priority --slots 6
python scripts/eval_gate.py gate --csv my_run.csv --agent my_policy --json
```

`gate` 在非 `positive` 时退出码为 1，可以直接当门禁用。指标表需要这些列：`policy,mean_utility,survey_completeness_pct,high_value_followup_pct,invalid_actions,unsafe_attempts_blocked,episodes`。

## 已公开结果

规则 Agent 对确定性优先级：跟进率相对高 47.22%（绝对 23.52 个百分点），效用高 1.63%，完成度低 9.44 个百分点。第 3 条判负，verdict 是 **negative**。

这是**稳定负结果**：三个种子方向一致。短期响应换来了过量的巡天欠账。这个结论本身有价值，不要写成"接近达标"或"部分成功"。

## 报告纪律

- 门槛先于数字。
- 失败的 episode 照报，指标不能事后替换。
- 合成暂现源不写"发现"。
- 三级验证路径：合成 → 脱敏日志 → 硬件影子（只建议不执行）。公开结果停在第一级，别说成硬件闭环。
- 不要实现一个假环境去声称原始哈希。

## 相关

真实观测栈的夜计划检查用 `starwhisper-night-plan`；候选源筛选用 `starwhisper-snclock`。
