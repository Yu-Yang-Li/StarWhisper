# 星语 / StarWhisper

中文 · [English](README_EN.md) · [研究文档](docs/research.md) · [技能包](skills/README.md) · [仓库结构](docs/architecture.md)

![山地天文台与夜空的概念插画](docs/assets/starwhisper-cover.png)

StarWhisper（星语）收录了天文语言模型、光变曲线与光谱分析、望远镜自主观测等工作的代码、数据和模型。这里也有一些可在 Codex / Cursor 中使用的科研技能。相关工作由国家天文台、之江实验室等单位支持。

[![License](https://img.shields.io/badge/License-Apache--2.0-172b4d)](LICENSE)
[![GitHub Repo stars](https://img.shields.io/github/stars/Yu-Yang-Li/StarWhisper?style=social)](https://github.com/Yu-Yang-Li/StarWhisper/stargazers)

## 项目

| 项目 | 内容 | 代码与文档 |
| --- | --- | --- |
| 天文语言模型 | 天文问答、科研文本与代码 | [训练数据](LLM_Data/README.md) · [StarWhisper 3 权重](https://www.modelscope.cn/models/AstroYuYang/StarWhisper3) |
| 光变与脉冲星 | Kepler / K2 变星分类、脉冲星候选分类 | [StarWhisper LC](StarWhisper_LC/README.md) · [Pulsar](https://github.com/ACMISLab/StarWhisper-Pulsar) |
| 稀疏光变 | ZTF / ATLAS 稀疏光变早期分类 | [训练、评估与权重](Early%20Classification%20from%20Sparse%20Light%20Curves/README.md) |
| 恒星光谱 | 低信噪比恒星光谱分析 | [代码与说明](Low-SNR-Stellar-Spectra-as-Language/README.md) |
| StarWhisper Telescope | 近邻星系巡天中的计划、观测与异常处理 | [NGSS](NGSS/README.md) · [论文](https://doi.org/10.1038/s44172-025-00520-4) |
| 全天相机 | 从天空图像生成掩码、调整观测序列 | [兴隆全天相机](AllSky-Camera-XL/README.md) |
| 虚拟司天 | 超新星爆发时刻估计与年轻候选筛选 | [SN Clock 数据](snclock/README.md) · [SitianClaw](https://github.com/Yu-Yang-Li/SitianClaw) · [项目地图](https://yu-yang-li.github.io/StarWhisper/virtual-gotta-map.html) |
| StarWhisper Explore | 合成环境下的观测策略实验 | [实验说明与结果](explore/README.md) |
| 科研技能 | 文献检索、研究设计、统计、写作、绘图与汇报 | [技能清单与安装方法](skills/README.md) |

论文图表、实验结果和复现说明见[研究文档](docs/research.md)。

## 使用

各项目的安装和运行方法见对应目录的 README。部分代码和模型放在独立仓库，链接已列在上方。[科研技能](skills/README.md)可安装到 Codex / Cursor 中使用。

例如，下面的命令读取仓库附带的 SN Clock 历史预测表：

```bash
git clone https://github.com/Yu-Yang-Li/StarWhisper.git
cd StarWhisper
python skills/starwhisper-snclock/scripts/screen_snclock.py rank --top 10
```

需要 Python 3.10 或更新版本，不需要安装其他依赖。表里是 22 个源的一次历史预测，来源和局限见[数据说明](snclock/README.md)。筛选当前候选体时，应换用最新数据。

## 论文与引用

- [StarWhisper Telescope: an AI framework for automating end-to-end astronomical observations](https://doi.org/10.1038/s44172-025-00520-4) — *Communications Engineering*, 2025。
- [StarWhisper LC](https://spj.science.org/doi/10.34133/icomputing.0110) — *Intelligent Computing*, 2025。
- [StarWhisper-Pulsar](https://openreview.net/pdf?id=8SKgWpZiDL) — NeurIPS 2024 FM4Science workshop。

引用格式见 [CITATION.cff](CITATION.cff) 和 [BibTeX](docs/research.md#引用)。使用某个项目的代码或数据时，也请引用相应论文。

问题和复现记录可以提交到 [Issue](https://github.com/Yu-Yang-Li/StarWhisper/issues)。也欢迎通过 Pull Request 改进代码或文档，提交前请阅读[贡献说明](CONTRIBUTING.md)。

代码采用 [Apache-2.0](LICENSE)。改编的科研技能保留原有 MIT 许可和来源标注，见 [NOTICE](skills/NOTICE.md)；模型权重按各自许可证使用。
