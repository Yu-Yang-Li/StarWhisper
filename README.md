# 星语 / StarWhisper

中文 · [English](README_EN.md) · [研究文档](docs/research.md) · [技能包](skills/README.md)

![山地天文台与夜空的概念插画](docs/assets/starwhisper-cover.png)

StarWhisper 是面向天文学的开源 AI 集成库，集中组织天文语言模型、数据分析工具、观测智能体和科研技能，提供代码、数据、模型资源及使用文档的统一入口。

各模块可按任务选用，环境和运行步骤见各自的 README；模型权重与部分完整工作流在外部项目维护，入口一并列在下表。相关工作由国家天文台、之江实验室等单位支持。

[![License](https://img.shields.io/badge/License-Apache--2.0-172b4d)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-Communications%20Engineering-172b4d)](https://doi.org/10.1038/s44172-025-00520-4)
[![GitHub Repo stars](https://img.shields.io/github/stars/Yu-Yang-Li/StarWhisper?style=social)](https://github.com/Yu-Yang-Li/StarWhisper/stargazers)

## 集成内容

| 模块 | 用途 | 代码与资源入口 |
| --- | --- | --- |
| 天文语言模型 | 天文问答、科研文本与代码 | [训练数据](LLM_Data/README.md) · [StarWhisper 3 权重](https://www.modelscope.cn/models/AstroYuYang/StarWhisper3) |
| 光变与脉冲星 | Kepler / K2 变星分类、脉冲星候选分类 | [StarWhisper LC](StarWhisper_LC/README.md) · [Pulsar](https://github.com/ACMISLab/StarWhisper-Pulsar) |
| 稀疏光变 | ZTF / ATLAS 稀疏光变早期分类 | [训练、评估与权重](Early%20Classification%20from%20Sparse%20Light%20Curves/README.md) |
| 恒星光谱 | 低信噪比恒星光谱分析 | [代码与说明](Low-SNR-Stellar-Spectra-as-Language/README.md) |
| 观测智能体 | 近邻星系巡天中的计划、观测与异常处理 | [NGSS](NGSS/README.md) · [论文](https://doi.org/10.1038/s44172-025-00520-4) |
| 全天相机 | 从天空图像生成掩码、调整观测序列 | [兴隆全天相机](AllSky-Camera-XL/README.md) |
| 虚拟司天 | 超新星爆发时刻估计与年轻候选筛选 | [SN Clock 数据](snclock/README.md) · [SitianClaw](https://github.com/Yu-Yang-Li/SitianClaw) · [项目地图](https://yu-yang-li.github.io/StarWhisper/virtual-gotta-map.html) |
| 观测策略评估 | 合成环境中的策略比较与结果判定 | [Explore 规格与结果](explore/README.md) |
| 科研技能 | 文献检索、研究设计、统计、写作、绘图与汇报 | [17 个技能的安装与清单](skills/README.md) |

各模块的研究背景、论文原图、公开结果及复现范围见[研究文档](docs/research.md)。

## 使用入口

先从上表选择模块，按对应文档配置环境和数据。科研技能可安装到 Codex / Cursor，步骤见[技能包文档](skills/README.md)。如果想先试一个无需额外依赖的例子，可以离线读取库内的 SN Clock 历史预测表：

```bash
git clone https://github.com/Yu-Yang-Li/StarWhisper.git
cd StarWhisper
python skills/starwhisper-snclock/scripts/screen_snclock.py rank --top 10
```

这个例子只需 Python 3.10 或更新版本和标准库。表中的 22 个源是一次历史输出，覆盖范围和输入留存情况见[数据说明](snclock/README.md)；当前时刻的候选筛选应使用最新数据。

## 论文与引用

- [StarWhisper Telescope: an AI framework for automating end-to-end astronomical observations](https://doi.org/10.1038/s44172-025-00520-4) — *Communications Engineering*, 2025。
- [StarWhisper LC](https://spj.science.org/doi/10.34133/icomputing.0110) — *Intelligent Computing*, 2025。
- [StarWhisper-Pulsar](https://openreview.net/pdf?id=8SKgWpZiDL) — NeurIPS 2024 FM4Science workshop。

引用格式见 [CITATION.cff](CITATION.cff) 和[研究文档中的 BibTeX](docs/research.md#引用)。使用具体研究线时，也请引用相应论文。

欢迎通过 [Issue](https://github.com/Yu-Yang-Li/StarWhisper/issues) 或 Pull Request 反馈问题、补充复现说明；提交前请阅读[贡献说明](CONTRIBUTING.md)。源代码采用 [Apache-2.0](LICENSE)，改编科研技能的来源与 MIT 许可见 [NOTICE](skills/NOTICE.md)，模型权重遵循各自许可证。
