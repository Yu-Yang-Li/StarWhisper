# 仓库结构与依赖

[项目首页](../README.md) · 中文 · [English](architecture-en.md) · [研究说明](research.md) · [科研技能](../skills/README.md)

StarWhisper 按研究方向保存代码、数据和实验材料。各项目有自己的运行环境；科研技能提供检索、分析或阅读这些材料的方法。使用前先选项目或任务，再查看对应说明。

## 从哪里开始

| 要做的事 | 先读什么 | 接着做什么 |
| --- | --- | --- |
| 了解论文、结果与适用范围 | [研究说明](research.md) | 查对应项目和原论文 |
| 复现或修改某个研究项目 | 该项目的 README，见下表 | 准备其数据、模型与运行环境 |
| 在 Codex / Cursor 中使用技能 | [技能目录](../skills/README.md) | 按需安装，再读相应 `SKILL.md` |
| 检查仓库附带表格或配置 | 下方离线工具关系表 | 先运行读取命令，确认实际输入文件 |
| 修改台站服务或观测流程 | [NGSS](../NGSS/README.md) 或 [全天相机](../AllSky-Camera-XL/README.md) | 核对外部系统、设备权限和项目自己的检查方法 |

## 目录职责

| 目录 | 保存什么 | 运行与数据关系 |
| --- | --- | --- |
| [LLM_Data](../LLM_Data/README.md) | 天文语言模型的训练文本说明与材料 | 模型权重通过项目列出的外部地址获取 |
| [StarWhisper_LC](../StarWhisper_LC/README.md) | Kepler / K2 光变分类测试代码 | 与稀疏光变基准分别使用 |
| [Early Classification from Sparse Light Curves](../Early%20Classification%20from%20Sparse%20Light%20Curves/README.md) | 稀疏光变训练、评估、权重与结果 | 数据池、标签与比较条件由该项目定义 |
| [Low-SNR-Stellar-Spectra-as-Language](../Low-SNR-Stellar-Spectra-as-Language/README.md) | 恒星光谱训练与分析 | 自有词表、数据处理与模型依赖 |
| [NGSS](../NGSS/README.md) | 巡天配置、计划、服务与观测操作 | 依赖 NINA、FTP、MQTT 等台站系统；部分接口会改变外部状态 |
| [AllSky-Camera-XL](../AllSky-Camera-XL/README.md) | 天空图像推断、掩码与观测序列调整 | 使用图像模型和校准数据；生成序列文件不表示已执行观测 |
| [GOTTA_Prototype](../GOTTA_Prototype) | 真假源分类样机材料 | 样机结果按其数据范围解释 |
| [snclock](../snclock/README.md) | 一次历史爆发年龄预测表及字段说明 | 供离线读取与候选筛选；不是实时数据服务 |
| [explore](../explore/README.md) | 合成观测策略的实验设定与指标 | 可重算指标比较，环境代码尚未收录 |
| [skills](../skills/README.md) | 17 个可安装技能及配套工具 | 安装说明见 [INSTALL.md](../skills/INSTALL.md)；各技能按任务读取参考文件 |
| [skill](../skill/README.md) | 旧技能入口的迁移说明 | 当前安装脚本只扫描 `skills/` |
| [docs](index.html) 与 [example](../example/README.md) | 研究说明、图表、素材来源与旧展示材料 | 文档站点从 `docs/index.html` 开始 |

脉冲星代码在独立的 [StarWhisper-Pulsar](https://github.com/ACMISLab/StarWhisper-Pulsar) 仓库，虚拟司天相关工作流在 [SitianClaw](https://github.com/Yu-Yang-Li/SitianClaw) 维护。外部项目以各自的安装和运行说明为准。

## 离线工具怎样读取项目材料

四个 `starwhisper-*` 技能只读取 CSV / JSON，不导入训练项目，也不访问网络或控制设备。它们需要 Python 3.10 或更新版本，使用标准库。

| 工具 | 默认读取 | 显式输入或找不到仓库时 |
| --- | --- | --- |
| [SN Clock](../skills/starwhisper-snclock/SKILL.md) | `snclock/snclock_*.csv` 中按文件名排序的最后一张表 | 用 `--csv` 指定；缺少表时退出并说明原因 |
| [Explore](../skills/starwhisper-explore/SKILL.md) | `explore/published_metrics.csv` | 用 `--csv` 指定；找不到仓库时读技能附带指标表 |
| [夜计划](../skills/starwhisper-night-plan/SKILL.md) | `NGSS/observe_config.json` | 用 `--config` 指定；缺少该文件时读示例配置并标注来源 |
| [稀疏光变比较](../skills/starwhisper-varlen/SKILL.md) | 技能附带的结果表与 `contract.json` | 按既定数据池比较；`check --csv` 检查用户输入的点数和标签 |

安装技能不会复制整个科研项目。需要访问仓库文件时设置 `STARWHISPER_ROOT`；也可在命令中指定输入文件。夜计划的预算是固定假设下的估算，与 NGSS 的实际计划和硬件执行分别核对。

## 科研技能怎样组织

```text
skills/<技能名>/
├── SKILL.md          适用任务、开始方法、结果与必要限制
├── astronomy.md      天文用法；改编技能按需阅读
├── references/       数据约定、接口、较长流程与检查细则
├── scripts/          已有的计算、解析、检索或打包工具
├── templates/        输入或报告模板
└── agents/           已提供的界面名称与调用设置
```

不是每个技能都有这些目录。入口说明只保留开始工作必需的信息，详细流程按任务加载；命令从对应技能目录运行，生成的工作目录按各技能说明处理。

`visual-deck-builder` 的脚本使用相邻 `giiisp-scientific-image-generation` 的生成与审查脚本。它默认按同级目录查找，也可用 `GIIISP_IMAGE_SKILL_DIR` 指定位置。两个技能应一同安装；图片服务与视觉审查的密钥分别配置。本地 PPTX 结构检查不能代替内容和图像审查。

## 修改放在哪里

- 科研代码、项目自己的配置与数据处理仍放在对应项目目录，依赖按该项目说明安装。
- 科学结果、数据范围和复现限制写在项目 README 与研究说明；来源、数字和比较条件一起维护。
- 技能的简短用法写在 `SKILL.md`，特定任务需要的内容放在 `references/`，并从入口给出链接。
- 技能安装、仓库数据定位和外部服务设置统一说明在 [安装文档](../skills/INSTALL.md)。
- 生成文件写到各技能的工作目录，密钥、私人配置与未公开资料不提交到仓库。

修改命令、路径或输入格式时，同时检查对应文档和测试。跨项目变化先核对调用者和数据约定；无需为文档整理统一训练环境或移动已有数据。提交方法见 [贡献说明](../CONTRIBUTING.md)。
