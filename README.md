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

StarWhisper 是天文模型和观测方面的开源工作，由国家天文台、之江实验室等单位支持。2023 年从语言模型做起，后来做到光变曲线、脉冲星候选和真实望远镜观测。2026 年的主线是虚拟司天（SN Clock）。

目前最新的正式论文是 2025 年 11 月的 [StarWhisper Telescope](https://doi.org/10.1038/s44172-025-00520-4)。

| 年 | 做了什么 | 代码 / 权重 |
| --- | --- | --- |
| 2023 | 仓库建立，天文语言模型 | [`LLM_Data/`](LLM_Data)，[StarWhisper3](https://www.modelscope.cn/models/AstroYuYang/StarWhisper3) |
| 2024 | LC 预印本；Pulsar；Telescope 预印本 | [`StarWhisper_LC/`](StarWhisper_LC)，[Pulsar 代码](https://github.com/ACMISLab/StarWhisper-Pulsar) |
| 2025 | LC、Telescope 发表 | [`NGSS/`](NGSS) |
| 2026 | 虚拟司天、兴隆全天相机、Explore、科研技能 | snclock，[SitianClaw](https://github.com/Yu-Yang-Li/SitianClaw)，[`skills/`](skills/README.md) |

---

## 2023–2024 · 语言模型

先做能回答天文问题、写代码的领域模型。StarWhisper 3 的训练数据在 [`LLM_Data/`](LLM_Data)，权重在魔搭 [AstroYuYang/StarWhisper3](https://www.modelscope.cn/models/AstroYuYang/StarWhisper3)。4.0 还在继续清数据。

---

## 2024–2025 · 光变曲线和脉冲星

[StarWhisper LC](https://arxiv.org/abs/2404.10757) 2024 年 4 月预印本，2025 年 2 月 26 日发表于 *Intelligent Computing*。用 Kepler / K2 光变做变星分类。测试代码在 [`StarWhisper_LC/`](StarWhisper_LC)。

<div align="center">

![StarWhisper LC](https://yu-yang-li.github.io/StarWhisper/assets/starwhisper-lc.png)

</div>

同年 12 月，[StarWhisper-Pulsar](https://openreview.net/pdf?id=8SKgWpZiDL) 在 NeurIPS 2024 FM4Science workshop 报告，做脉冲星候选分类。代码在 [ACMISLab/StarWhisper-Pulsar](https://github.com/ACMISLab/StarWhisper-Pulsar)。

---

## 2025 · StarWhisper Telescope

[论文](https://doi.org/10.1038/s44172-025-00520-4) 2024 年 12 月预印本，2025 年 11 月 6 日发表于 *Communications Engineering*。这是接到近邻星系巡天（NGSS）上的观测自动化 agent。代码在 [`NGSS/`](NGSS)。

夜前计划很少能原样执行。临时目标会插进来，天气会关掉窗口，跟踪、调焦、相机或穹顶也可能出问题。系统要反复决定：继续、插入、延后、安全暂停，还是恢复后重规划。

<div align="center">

![StarWhisper Telescope](https://yu-yang-li.github.io/StarWhisper/assets/starwhisper-telescope.png)

</div>

<div align="center">

![观测环](https://yu-yang-li.github.io/StarWhisper/assets/starwhisper-observe-loop.jpg)

</div>

<div align="center">

![演示](https://yu-yang-li.github.io/StarWhisper/assets/demo-1.png)

</div>

---

## 2026 · 虚拟司天

虚拟司天的代码在 snclock（SN Clock）仓库，不是 2025 年 Telescope 论文里已经做完的部分。公开说明见 [Virtual-GOTTA 地图](https://yu-yang-li.github.io/StarWhisper/virtual-gotta-map.html)。工作流技能在 [SitianClaw](https://github.com/Yu-Yang-Li/SitianClaw)，真假源样机在 [`GOTTA_Prototype/`](GOTTA_Prototype)。

司天计划在国内多个台站放 54 台 1 米级大视场望远镜，大约每 30 分钟扫 1 万平方度、三色。StarWhisper 是“司天大脑”的一条候选路径。

<div align="center">

![司天](https://yu-yang-li.github.io/StarWhisper/assets/sitian.png)

</div>

同一年里还有：

- [`Low-SNR-Stellar-Spectra-as-Language/`](Low-SNR-Stellar-Spectra-as-Language)：低信噪比恒星光谱，独立仓库在 [Jared-web03](https://github.com/Jared-web03/Low-SNR-Stellar-Spectra-as-Language)
- [`AllSky-Camera-XL/`](AllSky-Camera-XL)：兴隆全天相机，从原图到重排观测序列
- [`Early Classification from Sparse Light Curves/`](Early%20Classification%20from%20Sparse%20Light%20Curves)：稀疏 ZTF / ATLAS 光变的早期分类

兴隆全天相机和葵花卫星的云量临近预报在私有仓库，2026 年 8 月主线已经收口，用的是 2022–2025 的历史夜次，还不是业务预报。

---

## 2026 · Explore

Telescope 已经能跑自动观测。Explore 看的是：这个判断什么时候靠得住，会牺牲什么，什么时候必须拒绝。

`StarWhisper-Explore-v0.2` 是合成环境，固定兴隆、单镜、一夜六个时隙。候选目标、暂现源、天气和设备扰动由种子生成，各策略共用同一份剧本。Agent 看不到未来扰动，也不能改安全阈值。这不是光学仿真，也不是真机闭环。

预先固定比较无干预、随机、确定性优先级和规则 Agent。正向结果要求三个种子都没有主动安全违规，无效动作率 ≤ 1%，巡天完成度下降不超过 5 个百分点，同时高价值暂现源跟进率相对提高至少 20%，或科学效用提高至少 5%。

90 episode / 策略的结果：

| 策略 | 平均科学效用 | 巡天完成度 | 高价值暂现源跟进率 | 无效动作 | 被拦截的危险尝试 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 无干预 | 3.9343 | 77.41% | 0.00% | 0 | 122 |
| 随机 | 2.4027 | 30.19% | 30.37% | 18 | 72 |
| 确定性优先级 | 4.3289 | 61.11% | 49.81% | 0 | 0 |
| 规则 Agent | 4.3996 | 51.67% | 73.33% | 0 | 0 |

<div align="center">

![策略权衡](https://yu-yang-li.github.io/StarWhisper/assets/goai-metrics-source.png)

</div>

规则 Agent 相对确定性优先级，跟进率高 23.52 个百分点，效用大约高 1.6%，巡天完成度低 9.44 个百分点，超过预注册的 5 个百分点线。三个种子方向一致，复跑哈希一致。这是稳定的负结果，不是 Agent 已经赢了。

---

## 2026 · 科研技能

8 月从 [tashan-research-skills](https://github.com/TashanGKD/tashan-research-skills) 改了 13 个技能，放在 [`skills/`](skills/README.md)。文献默认走 NASA ADS 和 arXiv `astro-ph`。没有密钥就 dry-run，不编检索结果，也不给望远镜下指令。

```powershell
Copy-Item -Recurse .\skills\giiisp-paper-search-apis "$env:USERPROFILE\.codex\skills\giiisp-paper-search-apis"
```

源代码是 Apache-2.0。`skills/` 里改编自他山的部分见 [`NOTICE.md`](skills/NOTICE.md)，按 MIT。权重按各自许可证。

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

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Yu-Yang-Li/StarWhisper&type=Date)](https://star-history.com/#Yu-Yang-Li/StarWhisper&Date)
