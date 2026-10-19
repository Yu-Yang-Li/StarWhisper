# StarWhisper-Explore-v0.2

Telescope 用于巡天自动观测，Explore 比较观测策略的收益、巡天完成度损失和安全暂停条件。

本目录保存合成环境的实验设定和一次已核对的指标表。环境代码尚未收录到仓库；现有材料可用于重新计算策略比较，不能用于复现环境或声称已经验证真实观测。

## 实验设定

- 版本：`StarWhisper-Explore-v0.2`
- 台站：兴隆；单镜；一夜六个有序时隙
- 扰动：临时目标、短时天气、设备状态，由种子生成，各策略共用同一份剧本
- Agent 看不到未来扰动，也不能改台站坐标、物理可见性或安全阈值
- 真实硬件安全联锁优先于任何 Agent 动作
- 这不是光学传播仿真，也不是真实天气动力学

动作集合：`observe_plan`、`follow_transient`、`defer_transient`、`safe_pause`、`recover_and_replan`。

## 预注册标准

正向结果要求三个种子均满足：

- 没有主动安全违规
- 无效动作率 ≤ 1%
- 巡天完成度相对最强非 Agent 参照下降不超过 5 个百分点
- 并且：高价值暂现源跟进率相对提高至少 20%，或综合科学效用提高至少 5%

比较对象预先定为：无干预、随机、确定性优先级、规则 Agent。

## 已公开结果

90 episode / 策略，种子 `11/22/33`。数字见 [`published_metrics.csv`](published_metrics.csv)。从本目录逐条判定：

```powershell
python ..\..\skills\starwhisper-explore\scripts\eval_gate.py bar
python ..\..\skills\starwhisper-explore\scripts\eval_gate.py gate --agent rule_agent
```

自备实验表需使用相同列名，并将 `--csv` 放在子命令前：

```powershell
python ..\..\skills\starwhisper-explore\scripts\eval_gate.py --csv my_run.csv gate --agent my_policy
```

规则 Agent 相对确定性优先级：跟进率高 23.52 个百分点，效用大约高 1.6%，巡天完成度低 9.44 个百分点，超过预注册的 5 个百分点上限。三个种子的结果方向一致，已记录的复跑哈希一致。因此报告为稳定负结果。

图：

- [策略权衡](https://yu-yang-li.github.io/StarWhisper/assets/goai-metrics-source.png)
- [三级验证路径](https://yu-yang-li.github.io/StarWhisper/assets/goai-verification-source.png)

当前公开结果停在验证路径的第一级（合成环境）。脱敏日志回放和真实硬件影子运行还没有作为本仓库材料发布。
