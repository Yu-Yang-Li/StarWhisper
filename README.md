# 星语 / StarWhisper

[![GitHub Repo stars](https://img.shields.io/github/stars/Yu-Yang-Li/StarWhisper?style=social)](https://github.com/Yu-Yang-Li/StarWhisper/stargazers)
[![License](https://img.shields.io/github/license/Yu-Yang-Li/StarWhisper)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-Communications%20Engineering-0B1B33)](https://doi.org/10.1038/s44172-025-00520-4)
[![GitHub last commit](https://img.shields.io/github/last-commit/Yu-Yang-Li/StarWhisper)](https://github.com/Yu-Yang-Li/StarWhisper/commits/main)

<p align="center">
  中文 &nbsp;|&nbsp; <a href="README_EN.md">English</a>
</p>

<div align="center">

![StarWhisper](https://yu-yang-li.github.io/StarWhisper/assets/starwhisper-hero.jpg)

</div>

StarWhisper 是面向天文学的开源工作，由国家天文台、之江实验室等单位支持。2023 年从天文语言模型做起，随后做到 Kepler / K2 光变分类、脉冲星候选，以及接到近邻星系巡天（NGSS）上的观测 agent。2026 年的主线是虚拟司天（SN Clock）。最新正式论文是 2025 年 11 月的 [StarWhisper Telescope](https://doi.org/10.1038/s44172-025-00520-4)。

这个仓库现在以**技能包**的形式交付：17 个可以装进 Codex 或 Cursor 的天文科研技能，加上它们操作的数据和规格。往下先是能跑什么，再是这些工作的时间线。

---

## 技能包

```powershell
git clone https://github.com/Yu-Yang-Li/StarWhisper.git
cd StarWhisper
powershell -File .\skills\install.ps1          # Linux / macOS: ./skills/install.sh
```

4 个本线技能只用标准库，不装依赖、不联网、不碰硬件，克隆下来就能跑：

| 我要做什么 | 跑这个 |
| --- | --- |
| 从 SN Clock 表里挑最年轻的超新星候选 | `screen_snclock.py rank --top 10` |
| 判断某个源现在还在不在爆发窗口内 | `screen_snclock.py window --within-days 2` |
| 查这批预测的证据有多硬 | `screen_snclock.py audit` |
| 判断一个观测策略有没有过预注册的线 | `eval_gate.py gate --agent rule_agent` |
| 算一夜能排多少个目标 | `plan_night.py budget` |
| 检查目标表能不能交 | `plan_night.py lint-targets --targets t.csv` |
| 看稀疏光变 11 个配置谁最好 | `eval_varlen.py best --pool varlen` |
| 检查光变表是否符合 3–30 / 7 类合同 | `eval_varlen.py check --csv t.csv` |

| 技能 | 全部子命令 |
| --- | --- |
| [`starwhisper-snclock`](skills/starwhisper-snclock/SKILL.md) | `describe` `rank` `screen` `window` `audit` |
| [`starwhisper-explore`](skills/starwhisper-explore/SKILL.md) | `bar` `table` `gate` |
| [`starwhisper-night-plan`](skills/starwhisper-night-plan/SKILL.md) | `check-config` `budget` `lint-targets` `endpoints` |
| [`starwhisper-varlen`](skills/starwhisper-varlen/SKILL.md) | `contract` `table` `best` `compare` `labels` `check` |

另外 13 个是从他山改编的科研技能：文献检索、假设生成、实验设计、统计分析、写作、审稿、绘图、PPT。清单和环境变量见 [`skills/README.md`](skills/README.md)。

技能包的硬规矩：没有密钥就 dry-run，不编文献；夜计划只做检查，任何情况下不调用 `/manipulate_nina` 和 `/ftp_transfer`；筛空、判负、缺数据都照报，不放宽条件凑数。

## 数据与规格

技能操作的是这两份已公开材料，不是凭空生成的：

| 目录 | 内容 | 边界 |
| --- | --- | --- |
| [`snclock/`](snclock/README.md) | 22 个 TNS 源的爆发年龄预测（q16/q50/q84） | 覆盖范围小于声明窗口，多数行输入未留存 |
| [`explore/`](explore/README.md) | `StarWhisper-Explore-v0.2` 规格和四策略已核对表 | 合成环境，环境代码尚未入库 |
| [`skills/starwhisper-varlen/references/`](skills/starwhisper-varlen/SKILL.md) | 稀疏光变 11 个配置的已发表成绩单 | 只在同一 pool 里比；测试集指标不是爆发时刻 |

| 还要找什么 | 去哪 |
| --- | --- |
| 观测 agent 代码 | [`NGSS/`](NGSS)，依赖 NINA 等外部服务 |
| 光变分类测试 | [`StarWhisper_LC/`](StarWhisper_LC)，不是完整训练复现 |
| 稀疏光变训练代码 | [`Early Classification from Sparse Light Curves/`](Early%20Classification%20from%20Sparse%20Light%20Curves)，权重在 Hugging Face |
| 虚拟司天可运行系统 | [地图](https://yu-yang-li.github.io/StarWhisper/virtual-gotta-map.html)、[SitianClaw](https://github.com/Yu-Yang-Li/SitianClaw)，不在本仓库根目录 |

---

## 时间线

```mermaid
flowchart LR
  A["2023 语言模型"] --> B["2024–25 光变 / 脉冲星"]
  B --> C["2025 Telescope / NGSS"]
  C --> D["2026 虚拟司天"]
  S["天文科研技能"] -.-> A
  S -.-> B
  S -.-> C
  S -.-> D
```

| 年 | 做了什么 | 代码 / 权重 |
| --- | --- | --- |
| 2023 | 仓库建立，天文语言模型 | [`LLM_Data/`](LLM_Data)，[StarWhisper3](https://www.modelscope.cn/models/AstroYuYang/StarWhisper3) |
| 2024 | 光变分类、脉冲星、Telescope 预印本 | [`StarWhisper_LC/`](StarWhisper_LC)，[Pulsar 代码](https://github.com/ACMISLab/StarWhisper-Pulsar) |
| 2025 | LC、Telescope 发表 | [`NGSS/`](NGSS) |
| 2026 | 虚拟司天、Explore、技能包 | [SitianClaw](https://github.com/Yu-Yang-Li/SitianClaw)，[`skills/`](skills/README.md) |

---

## 2023–2024 · 语言模型

StarWhisper 3 做天文问答和写代码。训练数据在 [`LLM_Data/`](LLM_Data)，权重在魔搭 [AstroYuYang/StarWhisper3](https://www.modelscope.cn/models/AstroYuYang/StarWhisper3)。4.0 仍在整理科普和科研文本，权重打算放到魔搭。

---

## 2024–2025 · 光变曲线和脉冲星

[StarWhisper LC](https://spj.science.org/doi/10.34133/icomputing.0110) 用 Kepler / K2 光变做变星分类，主要看造父变星、RR Lyrae、食双星等。文里除了 Conv1D–BiLSTM 和 Swin Transformer，还有一组少做手工特征的 LLM / 多模态 / 音频模型，准确率大约 90%。2024 年 4 月预印本，2025 年 2 月 26 日发表于 *Intelligent Computing*。测试代码在 [`StarWhisper_LC/`](StarWhisper_LC)。

<div align="center">

![StarWhisper LC](https://yu-yang-li.github.io/StarWhisper/assets/starwhisper-lc.png)

</div>

2024 年 12 月，[StarWhisper-Pulsar](https://openreview.net/pdf?id=8SKgWpZiDL) 在 NeurIPS 2024 FM4Science workshop 报告，用多模态大模型做脉冲星候选分类。代码在 [ACMISLab/StarWhisper-Pulsar](https://github.com/ACMISLab/StarWhisper-Pulsar)。

---

## 2025 · StarWhisper Telescope

[论文](https://doi.org/10.1038/s44172-025-00520-4) 2024 年 12 月预印本，2025 年 11 月 6 日发表于 *Communications Engineering*。观测自动化 agent 接到近邻星系巡天（NGSS）上，大约 10 台业余级望远镜。代码在 [`NGSS/`](NGSS)。

夜前计划很少能原样执行。临时目标会插进来，天气会关掉窗口，跟踪、调焦、相机或穹顶也可能出问题。系统要反复决定：继续、插入、延后、安全暂停，还是恢复后重规划。

<div align="center">

![Observing loop](https://yu-yang-li.github.io/StarWhisper/assets/goai-observe-loop-source.png)

</div>

<p align="center"><sub>既定计划、三类扰动、观测智能体、受约束行动与反馈。这是 Telescope 与 Explore 共用的判断结构，不是光学仿真，也不是真实硬件截图。</sub></p>

<div align="center">

![StarWhisper Telescope](https://yu-yang-li.github.io/StarWhisper/assets/starwhisper-telescope.png)

</div>

---

## 2026 · 虚拟司天

2026 年才把虚拟司天做成可运行的系统（SN Clock）。早期科学应用是超新星时钟：估计爆发时刻、筛年轻超新星候选。公开说明见 [Virtual-GOTTA 地图](https://yu-yang-li.github.io/StarWhisper/virtual-gotta-map.html)。同一套工作流的可安装技能在 [SitianClaw](https://github.com/Yu-Yang-Li/SitianClaw)，真假源样机在 [`GOTTA_Prototype/`](GOTTA_Prototype)。

[`snclock/`](snclock/README.md) 放了一次生产输出：22 个 TNS 源在发现时刻的爆发后年龄估计，给 q16 / q50 / q84 三个分位，按是否落在两天内分两档。其中只有 2 个源连保守口径 q84 都在两天内。这张表的覆盖范围小于声明的筛选窗口，多数行的输入快照也没有留存，具体在数据卡里写清楚了。读表用 [`starwhisper-snclock`](skills/starwhisper-snclock/SKILL.md)：

```powershell
python .\skills\starwhisper-snclock\scripts\screen_snclock.py rank --top 10
python .\skills\starwhisper-snclock\scripts\screen_snclock.py audit
```

年龄估计不是光谱分类，表里的源也都已经在 TNS 上，不构成新发现。

司天计划在国内多个台站放 54 台 1 米级大视场望远镜，大约每 30 分钟扫 1 万平方度、三色。StarWhisper 是“司天大脑”的一条候选路径。

<div align="center">

![司天](https://yu-yang-li.github.io/StarWhisper/assets/sitian.png)

</div>

同属 2026 年、但独立于虚拟司天主线的还有：

- 兴隆全天相机 [`AllSky-Camera-XL/`](AllSky-Camera-XL)：从原图排到新的观测序列
- 葵花卫星云量临近预报：私有仓库，用 2022–2025 年历史夜次，2026 年 8 月主线已收口，还不是业务预报
- 稀疏 ZTF / ATLAS 光变早期分类：[`Early Classification from Sparse Light Curves/`](Early%20Classification%20from%20Sparse%20Light%20Curves)。读已核对成绩单用 [`starwhisper-varlen`](skills/starwhisper-varlen/SKILL.md)，不要把 50 点预训练的准确率写成 3–30 点主结果
- 低信噪比恒星光谱：[`Low-SNR-Stellar-Spectra-as-Language/`](Low-SNR-Stellar-Spectra-as-Language)，独立仓库 [Jared-web03](https://github.com/Jared-web03/Low-SNR-Stellar-Spectra-as-Language)

---

## 2026 · Explore

Telescope 已经能在巡天里自动观测。Explore 看的是这个判断什么时候靠得住、会牺牲什么、什么时候必须停。

规格、动作集合和已核对表在 [`explore/`](explore/README.md)。`StarWhisper-Explore-v0.2` 是合成环境：兴隆、单镜、一夜六个时隙。Agent 看不到未来扰动，也不能改安全阈值。这不是光学仿真，没有接到真实望远镜上，**环境代码也还没有放进本仓库**。

预先比较无干预、随机、确定性优先级和规则 Agent。算正向结果，三个种子都要过：没有主动安全违规，无效动作率 ≤ 1%，巡天完成度掉不超过 5 个百分点，并且高价值暂现源跟进率相对提高至少 20%，或科学效用提高至少 5%。

90 episode / 策略：

| 策略 | 平均科学效用 | 巡天完成度 | 高价值暂现源跟进率 | 无效动作 | 被拦截的危险尝试 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 无干预 | 3.9343 | 77.41% | 0.00% | 0 | 122 |
| 随机 | 2.4027 | 30.19% | 30.37% | 18 | 72 |
| 确定性优先级 | 4.3289 | 61.11% | 49.81% | 0 | 0 |
| 规则 Agent | 4.3996 | 51.67% | 73.33% | 0 | 0 |

<div align="center">

![策略权衡](https://yu-yang-li.github.io/StarWhisper/assets/goai-metrics-source.png)

</div>

规则 Agent 相对确定性优先级，跟进率高 23.52 个百分点，效用大约高 1.6%，巡天完成度低 9.44 个百分点，过了预注册的 5 个百分点线。三个种子方向一样，复跑哈希一致。这是稳定的负结果：短期响应换来了过量的巡天欠账。

判定可以自己跑一遍，也可以拿去判自己的实验表：

```powershell
python .\skills\starwhisper-explore\scripts\eval_gate.py bar
python .\skills\starwhisper-explore\scripts\eval_gate.py gate --agent rule_agent
```

<div align="center">

![Verification path](https://yu-yang-li.github.io/StarWhisper/assets/goai-verification-source.png)

</div>

<p align="center"><sub>先合成环境可复现，再脱敏日志校准，最后才是真实硬件影子运行（只建议、不执行）。当前公开结果停在第一级。</sub></p>

---

## 引用

```BibTeX
@article{wang2025starwhisper,
  title={StarWhisper Telescope: an AI framework for automating end-to-end astronomical observations},
  author={Wang, Cunshi and Zhang, Yu and Li, Yuyang and Hu, Xinjie and Mao, Yiming and Chen, Xunhao and Du, Pengliang and Wang, Rui and Wu, Ying and Yang, Hang and Li, Yansong and Wang, Beichuan and Mu, Haiyang and Chen, Xiaohan and He, Shunxuan and Mo, Hao and Zhang, Liyue and Du, Lin and Zhao, Yunning and Tian, Jianfeng and Ge, Liang and Mao, Yongna and Li, Shengming and Wang, Zheng and Lu, Xiaomeng and Zou, Jinhang and Huang, Yang and Sun, Ningchen and Zheng, Jie and He, Min and Bai, Yu and Jin, Junjie and Wu, Hong and Liu, Jifeng},
  journal={Communications Engineering},
  volume={4},
  pages={184},
  year={2025},
  doi={10.1038/s44172-025-00520-4},
  url={https://doi.org/10.1038/s44172-025-00520-4}
}
```

GitHub 也认根目录的 [`CITATION.cff`](CITATION.cff)。源代码 Apache-2.0。`skills/` 里改编自他山的部分见 [`NOTICE.md`](skills/NOTICE.md)，按 MIT。权重按各自许可证。

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Yu-Yang-Li/StarWhisper&type=Date)](https://star-history.com/#Yu-Yang-Li/StarWhisper&Date)
