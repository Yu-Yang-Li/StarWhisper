# 星语 / StarWhisper

中文 · [English](README_EN.md) · [研究文档](docs/research.md) · [技能包](skills/README.md)

![山地天文台与夜空的概念插画](docs/assets/starwhisper-cover.png)

StarWhisper 是面向天文学的开源研究，由国家天文台、之江实验室等单位支持。从 2023 年的天文语言模型出发，逐步开展光变曲线与脉冲星候选分类、望远镜自主观测，以及虚拟司天的超新星时钟研究。

这个仓库汇集各条研究线的代码、公开数据、论文材料和天文科研技能。2026 年的主线是虚拟司天（SN Clock）；观测智能体工作见已发表的 [StarWhisper Telescope](https://doi.org/10.1038/s44172-025-00520-4)。

[![License](https://img.shields.io/badge/License-Apache--2.0-172b4d)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-Communications%20Engineering-172b4d)](https://doi.org/10.1038/s44172-025-00520-4)
[![GitHub Repo stars](https://img.shields.io/github/stars/Yu-Yang-Li/StarWhisper?style=social)](https://github.com/Yu-Yang-Li/StarWhisper/stargazers)

## 研究方向

| 方向 | 研究内容 | 入口 |
| --- | --- | --- |
| 天文语言模型 | 天文问答、科研文本与代码 | [训练数据](LLM_Data/README.md) · [StarWhisper 3 权重](https://www.modelscope.cn/models/AstroYuYang/StarWhisper3) |
| 光变与脉冲星 | Kepler / K2 变星分类、脉冲星候选分类 | [StarWhisper LC](StarWhisper_LC/README.md) · [Pulsar](https://github.com/ACMISLab/StarWhisper-Pulsar) |
| 观测智能体 | 近邻星系巡天中的计划、观测与异常处理 | [NGSS](NGSS/README.md) · [论文](https://doi.org/10.1038/s44172-025-00520-4) |
| 虚拟司天 | 超新星爆发时刻估计与年轻候选筛选 | [SN Clock 数据](snclock/README.md) · [SitianClaw](https://github.com/Yu-Yang-Li/SitianClaw) · [项目地图](https://yu-yang-li.github.io/StarWhisper/virtual-gotta-map.html) |

相关研究：[稀疏光变早期分类](Early%20Classification%20from%20Sparse%20Light%20Curves/README.md)、[兴隆全天相机](AllSky-Camera-XL/README.md)、[低信噪比恒星光谱](Low-SNR-Stellar-Spectra-as-Language/README.md)。观测策略的合成实验见 [Explore](explore/README.md)。

各项工作的时间线、论文原图、公开结果及复现范围见[研究文档](docs/research.md)。

## 使用入口

如果想先试一个例子，可以离线读取仓库附带的 SN Clock 历史预测表：

```bash
git clone https://github.com/Yu-Yang-Li/StarWhisper.git
cd StarWhisper
python skills/starwhisper-snclock/scripts/screen_snclock.py rank --top 10
```

这个例子只需 Python 3.10 或更新版本和标准库。表中的 22 个源是一次历史输出，覆盖范围和输入留存情况见[数据说明](snclock/README.md)；当前时刻的候选筛选应使用最新数据。

| 想做什么 | 从哪里开始 |
| --- | --- |
| 在 Codex / Cursor 中使用天文科研技能 | [技能安装与清单](skills/README.md)：4 个本线技能、13 个天文适配科研技能 |
| 复现稀疏光变分类 | [训练流程与模型权重](Early%20Classification%20from%20Sparse%20Light%20Curves/README.md) |
| 配置观测智能体 | [NGSS 环境与接口](NGSS/README.md)，需对应的观测服务和设备连接 |
| 查看虚拟司天工作流 | [SitianClaw](https://github.com/Yu-Yang-Li/SitianClaw) |

## 论文与引用

- [StarWhisper Telescope: an AI framework for automating end-to-end astronomical observations](https://doi.org/10.1038/s44172-025-00520-4) — *Communications Engineering*, 2025。
- [StarWhisper LC](https://spj.science.org/doi/10.34133/icomputing.0110) — *Intelligent Computing*, 2025。
- [StarWhisper-Pulsar](https://openreview.net/pdf?id=8SKgWpZiDL) — NeurIPS 2024 FM4Science workshop。

引用格式见 [CITATION.cff](CITATION.cff) 和[研究文档中的 BibTeX](docs/research.md#引用)。使用具体研究线时，也请引用相应论文。

欢迎通过 [Issue](https://github.com/Yu-Yang-Li/StarWhisper/issues) 或 Pull Request 反馈问题、补充复现说明；提交前请阅读[贡献说明](CONTRIBUTING.md)。源代码采用 [Apache-2.0](LICENSE)，改编科研技能的来源与 MIT 许可见 [NOTICE](skills/NOTICE.md)，模型权重遵循各自许可证。