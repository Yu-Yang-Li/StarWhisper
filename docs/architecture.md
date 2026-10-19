# 仓库结构与依赖

[项目首页](../README.md) · 中文 · [English](architecture-en.md) · [研究说明](research.md) · [科研技能](../skills/README.md)

StarWhisper 按研究方向保存代码、数据和实验材料。各项目有自己的运行环境；科研技能提供检索、分析或阅读这些材料的方法。使用前先选项目或任务，再查看对应说明。

## 从哪里开始

| 要做的事 | 先读什么 | 接着做什么 |
| --- | --- | --- |
| 了解论文、结果与适用范围 | [研究说明](research.md) | 查对应项目和原论文 |
| 复现或修改某个研究项目 | [项目目录](../projects/README.md)中的 README | 准备其数据、模型与运行环境 |
| 在 Codex / Cursor 中使用技能 | [技能目录](../skills/README.md) | 按需安装，再读相应 `SKILL.md` |
| 检查仓库附带表格或配置 | 下方离线工具关系表 | 先运行读取命令，确认实际输入文件 |
| 修改台站服务或观测流程 | [NGSS](../projects/ngss/README.md) 或 [全天相机](../projects/allsky-camera/README.md) | 核对外部系统、设备权限和项目自己的检查方法 |

## 目录职责

```text
StarWhisper/
├── projects/   研究项目，各自保存代码、数据与运行说明
├── skills/     可安装的科研技能、脚本与参考文件
└── docs/       研究说明、图表与文档站
```

| 目录 | 从哪里开始 | 内容放置原则 |
| --- | --- | --- |
| [projects](../projects/README.md) | 选择研究方向，再读项目 README | 代码、模型、数据和结果随所属项目保存；环境分别安装 |
| [skills](../skills/README.md) | 按任务选择技能 | 17 个技能仍按原名称安装，安装设置见 [INSTALL.md](../skills/INSTALL.md) |
| [docs](index.html) | 阅读[研究说明](research.md) | 保存跨项目说明、论文图表和素材来源 |

`projects/` 下有 8 个项目目录，使用小写、连字符命名。项目内部的数据布局和运行方法见各自 README；NGSS 的外部服务地址与台站部署路径仍需按实际环境配置。

脉冲星代码在独立的 [StarWhisper-Pulsar](https://github.com/ACMISLab/StarWhisper-Pulsar) 仓库，虚拟司天相关工作流在 [SitianClaw](https://github.com/Yu-Yang-Li/SitianClaw) 维护。外部项目以各自的安装和运行说明为准。

## 离线工具怎样读取项目材料

四个 `starwhisper-*` 技能只读取 CSV / JSON，不导入训练项目，也不访问网络或控制设备。它们需要 Python 3.10 或更新版本，使用标准库。

| 工具 | 默认读取 | 显式输入或找不到仓库时 |
| --- | --- | --- |
| [SN Clock](../skills/starwhisper-snclock/SKILL.md) | `projects/snclock/snclock_*.csv` 中按文件名排序的最后一张表 | 用 `--csv` 指定；缺少表时退出并说明原因 |
| [Explore](../skills/starwhisper-explore/SKILL.md) | `projects/explore/published_metrics.csv` | 用 `--csv` 指定；找不到仓库时读技能附带指标表 |
| [夜计划](../skills/starwhisper-night-plan/SKILL.md) | `projects/ngss/observe_config.json` | 用 `--config` 指定；缺少该文件时读示例配置并标注来源 |
| [稀疏光变比较](../skills/starwhisper-varlen/SKILL.md) | 技能附带的结果表与 `contract.json` | 按既定数据池比较；`check --csv` 检查用户输入的点数和标签 |

安装技能不会复制整个科研项目。需要访问仓库文件时将 `STARWHISPER_ROOT` 设为仓库根目录；工具兼容旧版的顶层项目路径；也可在命令中指定输入文件。夜计划的预算是固定假设下的估算，与 NGSS 的实际计划和硬件执行分别核对。

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

修改命令、路径或输入格式时，同时检查对应文档和测试。跨项目变化先核对调用者和数据约定；移动目录时同时更新文档链接与离线工具的数据定位。提交方法见 [贡献说明](../CONTRIBUTING.md)。

## 目录迁移

2026 年 10 月整理后的路径如下。项目内部的相对路径继续沿用；从仓库根目录执行的命令需使用新位置。

| 原位置 | 当前位置 |
| --- | --- |
| `LLM_Data/` | [projects/language-models/](../projects/language-models/) |
| `StarWhisper_LC/` | [projects/light-curves/](../projects/light-curves/) |
| `Early Classification from Sparse Light Curves/` | [projects/sparse-light-curves/](../projects/sparse-light-curves/) |
| `Low-SNR-Stellar-Spectra-as-Language/` | [projects/stellar-spectra/](../projects/stellar-spectra/) |
| `NGSS/` | [projects/ngss/](../projects/ngss/) |
| `AllSky-Camera-XL/` | [projects/allsky-camera/](../projects/allsky-camera/) |
| `snclock/` | [projects/snclock/](../projects/snclock/) |
| `explore/` | [projects/explore/](../projects/explore/) |

旧展示图片、旧技能入口、NGSS 历史日志与编译缓存，以及 `GOTTA_Prototype` 占位文件已从当前分支移除。需要追溯时，可查看[清理前的文件快照](https://github.com/Yu-Yang-Li/StarWhisper/tree/9496f3683a6138e9133510cd45f8bacd6588045b/archive)。
