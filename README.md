# 星语 / StarWhisper

中文 · [English](README_EN.md) · [研究文档](docs/research.md) · [技能包](skills/README.md) · [仓库结构](docs/architecture.md)

![山地天文台与夜空的概念插画](docs/assets/starwhisper-cover.png)

StarWhisper（星语）收录了天文语言模型、光变曲线与光谱分析、望远镜自主观测等工作的代码、数据和模型。这里也有一些可在 Codex / Cursor 中使用的科研技能。相关工作由国家天文台、之江实验室等单位支持。

[![License](https://img.shields.io/badge/License-Apache--2.0-172b4d)](LICENSE)
[![GitHub Repo stars](https://img.shields.io/github/stars/Yu-Yang-Li/StarWhisper?style=social)](https://github.com/Yu-Yang-Li/StarWhisper/stargazers)

## 项目

| 研究方向 | 项目入口 |
| --- | --- |
| 天文语言模型 | [训练数据](projects/language-models/README.md) · [StarWhisper 3 权重](https://www.modelscope.cn/models/AstroYuYang/StarWhisper3) |
| 光变与暂现源 | [Kepler / K2 分类](projects/light-curves/README.md) · [稀疏光变](projects/sparse-light-curves/README.md) · [SN Clock](projects/snclock/README.md) · [Pulsar](https://github.com/ACMISLab/StarWhisper-Pulsar) |
| 恒星光谱 | [低信噪比光谱分析](projects/stellar-spectra/README.md) |
| 观测与调度 | [NGSS](projects/ngss/README.md) · [全天相机](projects/allsky-camera/README.md) · [Explore](projects/explore/README.md) · [SitianClaw](https://github.com/Yu-Yang-Li/SitianClaw) |

各项目的内容和运行条件见[项目目录](projects/README.md)。文献检索、研究设计、统计、写作与绘图工具见[科研技能](skills/README.md)。

论文图表、实验结果和复现说明见[研究文档](docs/research.md)。

## 使用

先在[项目目录](projects/README.md)查看各项工作已经提供的材料和运行条件。SN Clock、Explore、稀疏光变指标和 NGSS 配置有现成的本地读取命令；重新训练模型需要对应的数据与环境，完整观测流程还需准备台站系统。[科研技能](skills/README.md)也可按任务安装到 Codex / Cursor。

例如，下面的命令读取仓库附带的 SN Clock 历史预测表：

```bash
git clone https://github.com/Yu-Yang-Li/StarWhisper.git
cd StarWhisper
python skills/starwhisper-snclock/scripts/screen_snclock.py rank --top 10
```

需要 Python 3.10 或更新版本，不需要安装其他依赖或先安装技能。表里是 22 个源的一次历史预测，来源和局限见[数据说明](projects/snclock/README.md)。筛选当前候选体时，应换用最新数据。其他离线示例见[项目目录](projects/README.md#先试一条命令)。

## 论文与引用

- [StarWhisper Telescope: an AI framework for automating end-to-end astronomical observations](https://doi.org/10.1038/s44172-025-00520-4) — *Communications Engineering*, 2025。
- [StarWhisper LC](https://spj.science.org/doi/10.34133/icomputing.0110) — *Intelligent Computing*, 2025。
- [StarWhisper-Pulsar](https://openreview.net/pdf?id=8SKgWpZiDL) — NeurIPS 2024 FM4Science workshop。

引用格式见 [CITATION.cff](CITATION.cff) 和 [BibTeX](docs/research.md#引用)。使用某个项目的代码或数据时，也请引用相应论文。

问题和复现记录可以提交到 [Issue](https://github.com/Yu-Yang-Li/StarWhisper/issues)。也欢迎通过 Pull Request 改进代码或文档，提交前请阅读[贡献说明](CONTRIBUTING.md)。

代码采用 [Apache-2.0](LICENSE)。改编的科研技能保留原有 MIT 许可和来源标注，见 [NOTICE](skills/NOTICE.md)；模型权重按各自许可证使用。
