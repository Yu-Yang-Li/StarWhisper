# 星语 / StarWhisper

[![GitHub Repo stars](https://img.shields.io/github/stars/Yu-Yang-Li/StarWhisper?style=social)](https://github.com/Yu-Yang-Li/StarWhisper/stargazers)
[![License](https://img.shields.io/github/license/Yu-Yang-Li/StarWhisper)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-Communications%20Engineering-0B1B33)](https://doi.org/10.1038/s44172-025-00520-4)
[![GitHub last commit](https://img.shields.io/github/last-commit/Yu-Yang-Li/StarWhisper)](https://github.com/Yu-Yang-Li/StarWhisper/commits/main)

<p align="center">
  中文 &nbsp;|&nbsp; <a href="README_EN.md">English</a>
</p>

<div align="center">

![StarWhisper: AI astrophysicist workflow](https://yu-yang-li.github.io/StarWhisper/assets/starwhisper-hero.jpg)

</div>

**StarWhisper** 从天文学语言模型做起，再接到光变曲线、脉冲星候选、真实望远镜观测，再到 2026 年的虚拟司天（SN Clock）和可回放决策边界。项目由国家天文台、之江实验室等单位支持。最新正式论文是 2025 年 11 月的 [StarWhisper Telescope](https://doi.org/10.1038/s44172-025-00520-4)。

| 现在从哪进 | 对应阶段 |
| --- | --- |
| [`LLM_Data/`](LLM_Data) · [ModelScope AstroYuYang](https://www.modelscope.cn/models/AstroYuYang/StarWhisper3) | 2023–2024 语言模型 |
| [`StarWhisper_LC/`](StarWhisper_LC) · [Pulsar 代码](https://github.com/ACMISLab/StarWhisper-Pulsar) | 2024–2025 时序与多模态 |
| [`NGSS/`](NGSS) · [Telescope 论文](https://doi.org/10.1038/s44172-025-00520-4) | 2025 观测自动化 |
| [Virtual-GOTTA](https://yu-yang-li.github.io/StarWhisper/virtual-gotta-map.html) · [SitianClaw](https://github.com/Yu-Yang-Li/SitianClaw) | 2026 虚拟司天 / SN Clock |
| [`AllSky-Camera-XL/`](AllSky-Camera-XL) · [`Low-SNR-Stellar-Spectra-as-Language/`](Low-SNR-Stellar-Spectra-as-Language) | 2026 全天相机与光谱 |
| 下文 Explore · [`skills/`](skills/README.md) | 2026 决策边界与科研技能 |

Yu-Yang-Li 账号下其它天文仓库（本页是总入口）：

| 仓库 | 可见性 | 年份 | 做什么 |
| --- | --- | --- | --- |
| 本仓库 `StarWhisper` | 公开 | 2023– | 模型、Telescope、路线图与技能 |
| [ACMISLab/StarWhisper-Pulsar](https://github.com/ACMISLab/StarWhisper-Pulsar) | 公开 | 2024 | 脉冲星候选分类代码与实验 |
| [AstroYuYang/StarWhisper3](https://www.modelscope.cn/models/AstroYuYang/StarWhisper3) | 公开 | 2023– | 星语 3 权重（魔搭） |
| `tns_project` | 私有 | 2025.07 | TNS / ATLAS / ZTF 暂现源监测 |
| [SitianClaw](https://github.com/Yu-Yang-Li/SitianClaw) | 公开 | 2026.03 | 把 SN Clock 工作流装成可安装技能 |
| [Jared-web03/Low-SNR-Stellar-Spectra-as-Language](https://github.com/Jared-web03/Low-SNR-Stellar-Spectra-as-Language) | 公开 | 2026.04 | 低信噪比恒星光谱；本仓有副本 |
| snclock | 见该仓库 | 2026 | 虚拟司天 / SN Clock 主代码 |
| `xinglong-cloud-nowcasting-research` | 私有 | 2026.07–08 | 兴隆全天相机 × 葵花 AHI 临近预报，主线已收口 |

---

## 时间线

```mermaid
timeline
    title StarWhisper
    2023 : 仓库建立 : 天文学 LLM
    2024 : LC 预印本 : Pulsar NeurIPS workshop : Telescope 预印本
    2025 : LC 正式发表 : Telescope 正式发表
    2026 : 虚拟司天 / SN Clock : 兴隆全天相机 : Explore : 科研技能
```

| 时间 | 阶段 | 公开产物 |
| --- | --- | --- |
| 2023.07 | 仓库建立 | GitHub `Yu-Yang-Li/StarWhisper` |
| 2023–2024 | 天文学语言模型 | `LLM_Data`；权重 [AstroYuYang/StarWhisper3](https://www.modelscope.cn/models/AstroYuYang/StarWhisper3) |
| 2024.04 | 光变曲线 | [arXiv:2404.10757](https://arxiv.org/abs/2404.10757)；代码 `StarWhisper_LC` |
| 2024.12 | 脉冲星候选 | [报告](https://openreview.net/pdf?id=8SKgWpZiDL)；代码 [ACMISLab/StarWhisper-Pulsar](https://github.com/ACMISLab/StarWhisper-Pulsar) |
| 2024.12 | 观测自动化预印本 | [arXiv:2412.06412](https://arxiv.org/abs/2412.06412) |
| 2025.02 | 光变曲线正式发表 | [Intelligent Computing](https://spj.science.org/doi/10.34133/icomputing.0110) |
| 2025.07 | TNS 监测 | 私有仓库 `tns_project`（TNS / ATLAS / ZTF） |
| 2025.11 | 望远镜 agent 正式发表 | [Communications Engineering 4, 184](https://doi.org/10.1038/s44172-025-00520-4)；代码 `NGSS` |
| 2026.03 | SN Clock 技能包 | [SitianClaw](https://github.com/Yu-Yang-Li/SitianClaw) |
| 2026.04 | 低信噪比光谱 | [`Low-SNR-Stellar-Spectra-as-Language/`](Low-SNR-Stellar-Spectra-as-Language) |
| 2026.06 | 全天相机重规划 · GOTTA 样机 | [`AllSky-Camera-XL/`](AllSky-Camera-XL)、[`GOTTA_Prototype/`](GOTTA_Prototype) |
| 2026.07 | 稀疏光变早期分类 | [`Early Classification from Sparse Light Curves/`](Early%20Classification%20from%20Sparse%20Light%20Curves) |
| 2026.07–08 | 兴隆云量临近预报 | 私有仓库 `xinglong-cloud-nowcasting-research`（主线已收口） |
| 2026 | 虚拟司天 / SN Clock | snclock 仓库；公开地图 [Virtual-GOTTA](https://yu-yang-li.github.io/StarWhisper/virtual-gotta-map.html) |
| 2026 | 决策边界 | StarWhisper-Explore-v0.2（合成环境，稳定负结果） |
| 2026.08 | 科研技能 | [`skills/`](skills/README.md) |

<div align="center">

![StarWhisper architecture: models, telescope agent, Virtual-GOTTA, research skills](https://yu-yang-li.github.io/StarWhisper/assets/starwhisper-architecture.jpg)

</div>

<p align="center"><sub>现在的工作面：模型、望远镜 agent、Virtual-GOTTA 与科研技能叠在同一条线上，不是四套互不相干的 demo。</sub></p>

---

## 2023–2024 · 天文学语言模型

先做能回答天文问题、写代码、读观测知识的领域模型。StarWhisper 3 的训练数据在 `LLM_Data`，权重在魔搭 [AstroYuYang/StarWhisper3](https://www.modelscope.cn/models/AstroYuYang/StarWhisper3)。4.0 继续清洗科普与科研数据。

这一阶段解决的是“模型懂不懂天文”，还没有接到望远镜控制。

---

## 2024–2025 · 光变曲线与脉冲星

把模型从问答推到时序和多模态数据。

**2024 年 4 月**，[StarWhisper LC](https://arxiv.org/abs/2404.10757) 预印本上线；**2025 年 2 月 26 日**正式发表于 *Intelligent Computing*。用 Kepler / K2 光变曲线做变星分类，并给出一组少做手工特征的 LLM / 多模态 / 音频模型。测试代码在仓库里。

<div align="center">

![StarWhisper LC](https://yu-yang-li.github.io/StarWhisper/assets/starwhisper-lc.png)

</div>

**2024 年 12 月**，[StarWhisper-Pulsar](https://openreview.net/pdf?id=8SKgWpZiDL) 在 NeurIPS 2024 FM4Science workshop 报告：用多模态大模型做脉冲星候选分类。代码与实验在 [ACMISLab/StarWhisper-Pulsar](https://github.com/ACMISLab/StarWhisper-Pulsar)，不在本仓根目录。

---

## 2025 · StarWhisper Telescope

**2024 年 12 月**预印本，**2025 年 11 月 6 日**发表于 *Communications Engineering*。[论文](https://doi.org/10.1038/s44172-025-00520-4)给出端到端观测自动化 agent，并在近邻星系巡天（NGSS）网络落地。代码在 [`NGSS/`](NGSS)。

夜前计划几乎从不会原样执行。临时科学目标会插入，短时天气会关掉窗口，跟踪、调焦、相机或穹顶也可能突然异常。系统必须连续决定：继续、插入、延后、安全暂停，还是恢复后重规划。

<div align="center">

![StarWhisper Telescope](https://yu-yang-li.github.io/StarWhisper/assets/starwhisper-telescope.png)

</div>

<div align="center">

![Scheduled plan, three disturbances, observation agent, constrained actions and feedback](https://yu-yang-li.github.io/StarWhisper/assets/starwhisper-observe-loop.jpg)

</div>

<p align="center"><sub>图 1. 既定计划、三类扰动、观测智能体与受约束行动。</sub></p>

<div align="center">

![StarWhisper demonstration](https://yu-yang-li.github.io/StarWhisper/assets/demo-1.png)

</div>
<div align="center">

![StarWhisper telescope agent interface](https://yu-yang-li.github.io/StarWhisper/assets/demo-2.png)

</div>

同一年 7 月，账号下还有私有仓库 `tns_project`：从 TNS 拉暂现源，接 ATLAS / ZTF 测光。这是后来 SN Clock 的数据入口，不是 Telescope 论文的一部分。

这一阶段证明的是：agent 可以接到真实巡天流程。它还没有回答“判断在什么条件下值得信任”。

---

## 2026 · 虚拟司天（SN Clock）

2026 年才把司天做成可运行的虚拟系统。主代码在 **snclock**（SN Clock）：编排警报、台站状态、观测计划和数据回传，早期科学应用是超新星时钟。本仓库里的公开说明是 [Virtual-GOTTA 路线图](https://yu-yang-li.github.io/StarWhisper/virtual-gotta-map.html)。可安装的工作流技能在 [SitianClaw](https://github.com/Yu-Yang-Li/SitianClaw)（2026 年 3 月）。真假源样机在 [`GOTTA_Prototype/`](GOTTA_Prototype)。

**司天工程**计划在国内多个台站部署 54 台 1 米级大视场望远镜，约每 30 分钟完成 1 万平方度的三色巡天。StarWhisper 是“司天大脑”的一条候选路径。2025 年 Telescope 论文证明的是真实巡天里的观测 agent；虚拟司天 / SN Clock 是次年的工作，不要写成同一年已经做完。

<div align="center">

![SiTian / Sitian survey concept](https://yu-yang-li.github.io/StarWhisper/assets/sitian.png)

</div>

---

## 2026 · 全天相机、光谱与稀疏光变

同一年里，账号下还有几条不走 Telescope 论文主线、但同属天文的工作：

| 时间 | 入口 | 做什么 |
| --- | --- | --- |
| 2026.04 | [`Low-SNR-Stellar-Spectra-as-Language/`](Low-SNR-Stellar-Spectra-as-Language) | 把低信噪比恒星光谱当语言来建模；独立仓库 [Jared-web03/…](https://github.com/Jared-web03/Low-SNR-Stellar-Spectra-as-Language) |
| 2026.06 | [`AllSky-Camera-XL/`](AllSky-Camera-XL) | 兴隆全天相机：原图 → 掩码 → 重排观测序列 |
| 2026.07 | [`Early Classification from Sparse Light Curves/`](Early%20Classification%20from%20Sparse%20Light%20Curves) | 稀疏 ZTF/ATLAS 光变的早期分类基准 |
| 2026.07–08 | `xinglong-cloud-nowcasting-research`（私有） | 兴隆全天相机 × 葵花 AHI 临近预报。主线 2026-08-17 收口：在 2022–2025 历史夜次上，稳定增强未来 180/360 分钟的相机预测；新夜 prospective 复制被观测站数据源挡住，不要写成已经上线业务预报 |

---

## 2026 · 决策边界（StarWhisper-Explore）

真实栈已经能跑自动观测。Explore 问的是下一句：**在什么条件下这个判断值得信任，会稳定牺牲什么，何时必须拒绝。**

环境版本 `StarWhisper-Explore-v0.2`。固定台站 XingLong，单望远镜，一夜六个时隙；候选目标、暂现源到达、天气和设备扰动由种子生成，策略之间共享同一剧本。Agent 不得读取未来扰动，也不得改安全阈值。真实硬件联锁的优先级高于任何建议动作。

这是**合成决策闭环**，用来研究策略，不是光学传播仿真，也不是真实硬件闭环。公开材料不包含望远镜凭据、FTP/MQTT 地址和未脱敏图像。

比较对象预先固定为四类：无干预、随机、确定性优先级、规则 Agent。预注册的正向门槛要求：三个种子均无主动安全违规，无效动作率 ≤ 1%，巡天完成度下降不超过 5 个百分点，同时高价值暂现源跟进率相对提高至少 20%，或综合科学效用提高至少 5%。

当前 90 episode / 策略的合成结果：

| 策略 | 平均科学效用 | 巡天完成度 | 高价值暂现源跟进率 | 无效动作 | 被联锁拦截的危险尝试 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 无干预 | 3.9343 | 77.41% | 0.00% | 0 | 122 |
| 随机 | 2.4027 | 30.19% | 30.37% | 18 | 72 |
| 确定性优先级 | 4.3289 | 61.11% | 49.81% | 0 | 0 |
| 规则 Agent | 4.3996 | 51.67% | 73.33% | 0 | 0 |

<div align="center">

![Pre-registered trade-off: rule agent raises follow-up but misses the survey-completeness floor](https://yu-yang-li.github.io/StarWhisper/assets/goai-metrics-source.png)

</div>

<p align="center"><sub>图 2. 预注册门槛下的策略权衡：规则 Agent 跟进率最高，但落在巡天完成度允许区左侧。</sub></p>

规则 Agent 相对确定性优先级把跟进率提高了 23.52 个百分点，科学效用只提高约 1.6%，但巡天完成度下降 9.44 个百分点，超过预注册的 5 个百分点容忍线。这是**稳定负结果**：当前边际价值规则过度偏向短期响应。三个种子方向一致，双跑输出哈希一致。它不能写成“Agent 已经赢了”。

<div align="center">

![Synthetic environment, de-identified log replay, hardware shadow mode](https://yu-yang-li.github.io/StarWhisper/assets/starwhisper-verification.jpg)

</div>

<p align="center"><sub>图 3. 先证明可复现，再校准现实，最后才进入只建议、不执行的硬件影子运行。</sub></p>

下一步：补上 AstroQ / TJO 风格的约束调度 baseline，用脱敏日志校准扰动频率，再把同一决策接口接到观测时序模型、短时气象和控制模型。影子运行仍然是建议，不是放权。

---

## 2026 · 天文科研技能

2026 年 8 月，从 [tashan-research-skills](https://github.com/TashanGKD/tashan-research-skills) 选出 13 个技能做天文适配后放进 [`skills/`](skills/README.md)。默认值改成 NASA ADS、arXiv `astro-ph`、AAS 引用、光变/光谱/FITS 数据合同，以及“合成 / 回放 / 硬件”三条边界。

<div align="center">

![Astronomy research skills matrix](https://yu-yang-li.github.io/StarWhisper/assets/starwhisper-skills-matrix.jpg)

</div>

| 类 | 技能 |
| --- | --- |
| 文献证据 | 论文检索 · 深度研究 · 论文审查 |
| 研究构思 | 假设生成 · 数据处理 · 实验设计 · 统计分析 |
| 成果表达 | 文本润色 · 学术写作 · 科研绘图 · PPT |
| 协作沉淀 | 引用合规 · 科研画像 |

```powershell
git clone https://github.com/Yu-Yang-Li/StarWhisper.git
Copy-Item -Recurse .\StarWhisper\skills\giiisp-paper-search-apis "$env:USERPROFILE\.codex\skills\giiisp-paper-search-apis"
```

没有 ADS / Giiisp 密钥时走 dry-run 或本地回退，不伪造检索命中。技能不能对望远镜下发指令。

---

## 使用边界

| 可以说 | 不可以说 |
| --- | --- |
| 2025 年论文描述的观测自动化框架已在 NGSS 落地 | 本仓库已经接管真实望远镜的安全联锁 |
| Explore-v0.2 的合成夜次可复现、哈希一致 | 规则 Agent 已经通过正向发现门槛 |
| 技能能辅助 ADS 检索、写作和引用检查 | 技能输出等于已发表结果或已发现暂现源 |
| 影子运行可以给出建议 | 建议已被硬件执行 |

源代码：**Apache-2.0**。`skills/` 中改编自他山科研技能库的部分见 [`skills/NOTICE.md`](skills/NOTICE.md)，遵循 MIT。基础模型权重遵循各自许可证。

---

## 引用

如果这项工作对你有帮助，请引用 Telescope 正式论文：

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

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Yu-Yang-Li/StarWhisper&type=Date)](https://star-history.com/#Yu-Yang-Li/StarWhisper&Date)
