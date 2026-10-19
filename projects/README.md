# 研究项目

[项目首页](../README.md) · [研究文档](../docs/research.md) · [科研技能](../skills/README.md)

这里的 8 个项目开放的材料有所不同：有些附带可直接读取的结果表，有些提供训练代码，需要另外准备数据；观测项目还涉及相机标定或台站系统。先按下面的表选择，再读项目自己的说明。

| 项目 | 下载后可以做什么 | 进一步运行需要什么 |
| --- | --- | --- |
| [天文语言模型](language-models/README.md) | 阅读中英文天文、物理问答 JSON | 本目录只保存数据；使用模型需从外部获取权重，并按模型说明准备推理环境 |
| [Kepler / K2 光变分类](light-curves/README.md) | 阅读论文预测结果和对照模型脚本 | 原始光变、图像或合并后的序列表，以及部分预训练权重；未提供完整复现包 |
| [稀疏光变早期分类](sparse-light-curves/README.md) | 用 Python 查看 11 个配置的已发表指标 | 重新训练需要光变片段、匹配的特征 CSV 和项目依赖；模型权重另行下载 |
| [低信噪比恒星光谱](stellar-spectra/README.md) | 阅读模型、数据处理与训练代码，查看固定词表 | 原始或模拟光谱、生成的训练 CSV、训练环境；微调还需前一阶段权重 |
| [StarWhisper Telescope / NGSS](ngss/README.md) | 用 Python 检查附带配置、估算曝光预算 | 完整服务还缺计算模块、台站与星表数据，并需配置 NINA、通信和数据处理系统 |
| [兴隆全天相机](allsky-camera/README.md) | 查看样例照片、分割权重、计划和标定文件 | 安装 Conda 与推理依赖后运行样例；自己的相机需要重新核对标定与计划 |
| [SN Clock](snclock/README.md) | 用 Python 排序、筛选 22 个源的历史预测 | 查看附带表格无需额外数据；当前候选筛选需最新预测表，完整流程在外部仓库 |
| [StarWhisper Explore](explore/README.md) | 用 Python 重算合成实验的策略比较 | 查看指标无需额外数据；仿真环境代码尚未收录，不能从本目录重跑实验 |

## 先试一条命令

从仓库根目录运行下面的命令，需要 Python 3.10 或更新版本，无需安装其他依赖，也不必先安装技能包。

```bash
python skills/starwhisper-snclock/scripts/screen_snclock.py rank --top 10
python skills/starwhisper-explore/scripts/eval_gate.py table
python skills/starwhisper-varlen/scripts/eval_varlen.py table --pool varlen
python skills/starwhisper-night-plan/scripts/plan_night.py check-config
```

它们分别读取历史预测、合成实验指标、稀疏光变分类结果和观测配置，不会启动训练或连接设备。各项目 README 中的命令则从该项目目录运行；涉及占位路径时，先换成自己的文件。

脉冲星分类在独立的 [StarWhisper-Pulsar](https://github.com/ACMISLab/StarWhisper-Pulsar) 仓库，虚拟司天相关技能在 [SitianClaw](https://github.com/Yu-Yang-Li/SitianClaw) 维护。

旧目录名称与新位置见[目录迁移](../docs/architecture.md#目录迁移)。
