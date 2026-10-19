# 研究项目

[项目首页](../README.md) · [研究文档](../docs/research.md) · [科研技能](../skills/README.md)

每个目录保存一项研究的代码或数据。先读项目说明，再进入该目录准备环境；数据范围、运行条件和复现限制由各项目分别说明。

| 项目 | 目录 | 可以从哪里开始 |
| --- | --- | --- |
| 天文语言模型 | [language-models](language-models/README.md) | 查看 StarWhisper 3 的中英文问答数据；模型权重通过外部链接获取 |
| Kepler / K2 光变分类 | [light-curves](light-curves/README.md) | 阅读测试代码和论文相关结果 |
| 稀疏光变早期分类 | [sparse-light-curves](sparse-light-curves/README.md) | 准备 ZTF / ATLAS 数据，运行预处理、训练与评估 |
| 低信噪比恒星光谱 | [stellar-spectra](stellar-spectra/README.md) | 准备光谱和词表，运行预训练与微调 |
| StarWhisper Telescope | [ngss](ngss/README.md) | 核对台站环境、配置和观测服务 |
| 兴隆全天相机 | [allsky-camera](allsky-camera/README.md) | 准备图像、模型和标定参数，生成掩码及观测序列 |
| SN Clock | [snclock](snclock/README.md) | 读取 22 个源的一次历史预测；可用离线技能筛选 |
| StarWhisper Explore | [explore](explore/README.md) | 重算合成实验的指标比较；环境代码尚未收录 |

脉冲星分类在独立的 [StarWhisper-Pulsar](https://github.com/ACMISLab/StarWhisper-Pulsar) 仓库，虚拟司天相关技能在 [SitianClaw](https://github.com/Yu-Yang-Li/SitianClaw) 维护。

旧目录名称与新位置见[目录迁移](../docs/architecture.md#目录迁移)。
