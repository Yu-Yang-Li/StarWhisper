---
name: starwhisper-index
description: Route a StarWhisper question to the skill that can actually run it, or to the reference asset when no skill exists. Use when the user asks what StarWhisper is, which folder to open, which skill to install, or how the LLM / light-curve / Telescope / Explore / SN Clock lines relate.
license: Apache-2.0
---

# StarWhisper 路由

先问一句：这个问题是要**跑**什么，还是要**读**什么。

```powershell
python scripts/route.py --query "这批候选还在两天内吗" --json
```

路由结果分两类。命中 `skills` 就去跑；只命中 `assets` 说明这条线在本仓库只有材料、没有可跑流程，读完如实说，不要假装跑过。

## 会跑的技能

| 技能 | 做什么 |
| --- | --- |
| `starwhisper-snclock` | 把 SN Clock 年龄预测筛成年轻超新星候选清单，并审计证据强度 |
| `starwhisper-explore` | 按预注册门槛逐条判定策略对比，给 positive / negative / inconclusive |
| `starwhisper-night-plan` | 校验 observe_config、算一夜容量、lint 目标表 |
| `giiisp-paper-search-apis` | 先 NASA ADS 再 arXiv `astro-ph`，无 token 时 dry-run |

其余 13 个天文科研技能（假设、实验设计、统计、写作、审稿、绘图、PPT）见 [`skills/README.md`](../README.md)。

## 只能读的材料

语言模型、Kepler/K2 光变、脉冲星、全天相机、稀疏光变、低信噪光谱、GOTTA 样机、SitianClaw 工作流——位置和边界都在 [`references/asset-map.md`](references/asset-map.md)。

## 跨线纪律

一句话里不要同时混用已发表论文指标、Explore 合成表和硬件指令，这三者的证据强度完全不同。

- 论文指标：有同行评议，但只在论文设定下成立
- Explore 表：合成环境，环境代码不在仓库
- 硬件：需要真实栈接通，安全联锁优先于 agent

## 安装

```powershell
powershell -File ..\install_native.ps1
```

装到别处后，把本仓库路径设成 `STARWHISPER_ROOT`，脚本才找得到 `snclock/`、`explore/`、`NGSS/`。
