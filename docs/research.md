# StarWhisper 研究文档

[返回首页](../README.md) · 中文 · [English](research-en.md) · [技能安装](../skills/README.md)

这里记录集成库中各模块的研究背景、时间线、论文原图、公开结果和复现范围。模块入口见[首页目录](../README.md#集成内容)。下列命令均从仓库根目录执行。

## 时间线

| 年 | 做了什么 | 代码 / 权重 |
| --- | --- | --- |
| 2023 | 仓库建立，天文语言模型 | [`LLM_Data/`](../LLM_Data)，[StarWhisper3](https://www.modelscope.cn/models/AstroYuYang/StarWhisper3) |
| 2024 | 光变分类、脉冲星、Telescope 预印本 | [`StarWhisper_LC/`](../StarWhisper_LC)，[Pulsar 代码](https://github.com/ACMISLab/StarWhisper-Pulsar) |
| 2025 | LC、Telescope 发表 | [`NGSS/`](../NGSS) |
| 2026 | 虚拟司天、Explore、技能包 | [SitianClaw](https://github.com/Yu-Yang-Li/SitianClaw)，[`skills/`](../skills/README.md) |

---

## 2023–2024 · 语言模型

StarWhisper 3 做天文问答和写代码。训练数据在 [`LLM_Data/`](../LLM_Data)，权重在魔搭 [AstroYuYang/StarWhisper3](https://www.modelscope.cn/models/AstroYuYang/StarWhisper3)。4.0 仍在整理科普和科研文本，权重打算放到魔搭。

---

## 2024–2025 · 光变曲线和脉冲星

[StarWhisper LC](https://spj.science.org/doi/10.34133/icomputing.0110) 用 Kepler / K2 光变做变星分类，主要看造父变星、RR Lyrae、食双星等。文里除了 Conv1D–BiLSTM 和 Swin Transformer，还有一组少做手工特征的 LLM / 多模态 / 音频模型，准确率大约 90%。2024 年 4 月预印本，2025 年 2 月 26 日发表于 *Intelligent Computing*。测试代码在 [`StarWhisper_LC/`](../StarWhisper_LC)。

![StarWhisper LC](assets/starwhisper-lc.png)

2024 年 12 月，[StarWhisper-Pulsar](https://openreview.net/pdf?id=8SKgWpZiDL) 在 NeurIPS 2024 FM4Science workshop 报告，用多模态大模型做脉冲星候选分类。代码在 [ACMISLab/StarWhisper-Pulsar](https://github.com/ACMISLab/StarWhisper-Pulsar)。

---

## 2025 · StarWhisper Telescope

[论文](https://doi.org/10.1038/s44172-025-00520-4) 2024 年 12 月预印本，2025 年 11 月 6 日发表于 *Communications Engineering*。观测自动化 agent 接到近邻星系巡天（NGSS）上，大约 10 台业余级望远镜。代码在 [`NGSS/`](../NGSS)。

夜前计划很少能原样执行。临时目标会插进来，天气会关掉窗口，跟踪、调焦、相机或穹顶也可能出问题。系统要反复决定：继续、插入、延后、安全暂停，还是恢复后重规划。

![Observing loop](assets/goai-observe-loop-source.png)

<p align="center"><sub>既定计划、三类扰动、观测智能体、受约束行动与反馈。这是 Telescope 与 Explore 共用的判断结构，不是光学仿真，也不是真实硬件截图。</sub></p>

![StarWhisper Telescope](assets/starwhisper-telescope.png)

---

## 2026 · 虚拟司天

2026 年才把虚拟司天做成可运行的系统（SN Clock）。早期科学应用是超新星时钟：估计爆发时刻、筛年轻超新星候选。公开说明见 [Virtual-GOTTA 地图](https://yu-yang-li.github.io/StarWhisper/virtual-gotta-map.html)。同一套工作流的可安装技能在 [SitianClaw](https://github.com/Yu-Yang-Li/SitianClaw)，真假源样机在 [`GOTTA_Prototype/`](../GOTTA_Prototype)。

[`snclock/`](../snclock/README.md) 放了一次生产输出：22 个 TNS 源在发现时刻的爆发后年龄估计，给 q16 / q50 / q84 三个分位，按是否落在两天内分两档。其中只有 2 个源连保守口径 q84 都在两天内。这张表的覆盖范围小于声明的筛选窗口，多数行的输入快照也没有留存，具体在数据卡里写清楚了。读表用 [`starwhisper-snclock`](../skills/starwhisper-snclock/SKILL.md)：

```powershell
python .\skills\starwhisper-snclock\scripts\screen_snclock.py rank --top 10
python .\skills\starwhisper-snclock\scripts\screen_snclock.py audit
```

年龄估计不是光谱分类，表里的源也都已经在 TNS 上，不构成新发现。

司天计划在国内多个台站放 54 台 1 米级大视场望远镜，大约每 30 分钟扫 1 万平方度、三色。StarWhisper 是“司天大脑”的一条候选路径。

![司天](assets/sitian.png)

同属 2026 年、但独立于虚拟司天主线的还有：

- 兴隆全天相机 [`AllSky-Camera-XL/`](../AllSky-Camera-XL)：从原图排到新的观测序列
- 葵花卫星云量临近预报：私有仓库，用 2022–2025 年历史夜次，2026 年 8 月主线已收口，还不是业务预报
- 稀疏 ZTF / ATLAS 光变早期分类：[`Early Classification from Sparse Light Curves/`](../Early%20Classification%20from%20Sparse%20Light%20Curves)。读已核对成绩单用 [`starwhisper-varlen`](../skills/starwhisper-varlen/SKILL.md)，不要把 50 点预训练的准确率写成 3–30 点主结果
- 低信噪比恒星光谱：[`Low-SNR-Stellar-Spectra-as-Language/`](../Low-SNR-Stellar-Spectra-as-Language)，独立仓库 [Jared-web03](https://github.com/Jared-web03/Low-SNR-Stellar-Spectra-as-Language)

---

## 2026 · Explore

Telescope 已经能在巡天里自动观测。Explore 看的是这个判断什么时候靠得住、会牺牲什么、什么时候必须停。

规格、动作集合和已核对表在 [`explore/`](../explore/README.md)。`StarWhisper-Explore-v0.2` 是合成环境：兴隆、单镜、一夜六个时隙。Agent 看不到未来扰动，也不能改安全阈值。这不是光学仿真，没有接到真实望远镜上，**环境代码也还没有放进本仓库**。

预先比较无干预、随机、确定性优先级和规则 Agent。算正向结果，三个种子都要过：没有主动安全违规，无效动作率 ≤ 1%，巡天完成度掉不超过 5 个百分点，并且高价值暂现源跟进率相对提高至少 20%，或科学效用提高至少 5%。

90 episode / 策略：

| 策略 | 平均科学效用 | 巡天完成度 | 高价值暂现源跟进率 | 无效动作 | 被拦截的危险尝试 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 无干预 | 3.9343 | 77.41% | 0.00% | 0 | 122 |
| 随机 | 2.4027 | 30.19% | 30.37% | 18 | 72 |
| 确定性优先级 | 4.3289 | 61.11% | 49.81% | 0 | 0 |
| 规则 Agent | 4.3996 | 51.67% | 73.33% | 0 | 0 |

![策略权衡](assets/goai-metrics-source.png)

规则 Agent 相对确定性优先级，跟进率高 23.52 个百分点，效用大约高 1.6%，巡天完成度低 9.44 个百分点，过了预注册的 5 个百分点线。三个种子方向一样，复跑哈希一致。这是稳定的负结果：短期响应换来了过量的巡天欠账。

判定可以自己跑一遍，也可以拿去判自己的实验表：

```powershell
python .\skills\starwhisper-explore\scripts\eval_gate.py bar
python .\skills\starwhisper-explore\scripts\eval_gate.py gate --agent rule_agent
```

![Verification path](assets/goai-verification-source.png)

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

GitHub 也认根目录的 [`CITATION.cff`](../CITATION.cff)。源代码 Apache-2.0。`skills/` 里改编自他山的部分见 [`NOTICE.md`](../skills/NOTICE.md)，按 MIT。权重按各自许可证。
