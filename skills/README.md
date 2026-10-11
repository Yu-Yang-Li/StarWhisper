# 天文科研技能

这里有 17 个技能：4 个用于读取星语的数据和检查观测配置，13 个改编自他山科研技能。每个目录中的 `SKILL.md` 说明适用任务和使用方法；部分技能附有计算、检索或生成文件的脚本。

[项目首页](../README.md) · [研究文档](../docs/research.md) · [来源与许可](NOTICE.md)

## 安装

以下命令从仓库根目录运行。Windows 使用 PowerShell：

```powershell
powershell -File .\skills\install.ps1 -List
powershell -File .\skills\install.ps1 -DryRun
powershell -File .\skills\install.ps1
```

Linux / macOS 使用 Bash：

```bash
bash ./skills/install.sh --list
bash ./skills/install.sh --dry-run
bash ./skills/install.sh
```

默认安装全部技能到 `~/.codex/skills` 和 `~/.cursor/skills`。安装时会替换同名技能；有本地修改的目录请先备份。也可以只安装其中一组：

```powershell
powershell -File .\skills\install.ps1 -Set native -Target codex
```

Bash 对应参数为 `--set native --target codex`。`native` 选择 4 个星语技能，`research` 选择 13 个改编技能，`all` 选择全部；安装位置可选 `codex`、`cursor` 或 `both`。

安装到其他位置后，将 `STARWHISPER_ROOT` 指向本仓库，供脚本读取 `snclock/`、`explore/` 和 `NGSS/`。从仓库根目录设置当前 PowerShell 会话：

```powershell
$env:STARWHISPER_ROOT = (Get-Location).Path
setx STARWHISPER_ROOT "$env:STARWHISPER_ROOT"
```

第一行对当前会话生效，`setx` 保存到之后新开的终端。Linux / macOS 使用 `export STARWHISPER_ROOT="$PWD"`，需要长期保留时可写入自己的 shell 配置。

4 个星语技能需要 Python 3.10 或更新版本，只依赖标准库。其他技能的依赖按需安装，见 [experiment-design](experiment-design/requirements.txt)、[statistical-analysis](statistical-analysis/requirements.txt)、[thesis-audit-reviewer](thesis-audit-reviewer/requirements.txt)、[visual-deck-builder](visual-deck-builder/scripts/requirements.txt) 和 [papercheck](papercheck/assets/paperchecker-rules/requirements.txt)。基线模型模板的依赖在各脚本开头列出。

## 星语数据与观测配置

这 4 个技能提供 18 个子命令，均在本地读取文件，不访问网络或控制设备。详细参数见各自的 `SKILL.md`。

| 技能 | 做什么 | 子命令 |
| --- | --- | --- |
| [starwhisper-snclock](starwhisper-snclock/SKILL.md) | 读取 [SN Clock](../snclock/README.md) 预测表，排序和筛选年轻候选，检查数据留存与预测时间 | `describe`、`rank`、`screen`、`window`、`audit` |
| [starwhisper-explore](starwhisper-explore/SKILL.md) | 按预注册标准检查 [Explore](../explore/README.md) 的观测策略指标，也可读取自备指标表 | `bar`、`table`、`gate` |
| [starwhisper-night-plan](starwhisper-night-plan/SKILL.md) | 检查 NGSS 配置、曝光时间预算、目标表与接口类型 | `check-config`、`budget`、`lint-targets`、`endpoints` |
| [starwhisper-varlen](starwhisper-varlen/SKILL.md) | 比较 11 个稀疏光变配置的公开结果，检查点数和标签；只在同一数据设定下比较 | `contract`、`table`、`best`、`compare`、`labels`、`check` |

从仓库根目录可以先运行这些例子：

```powershell
python .\skills\starwhisper-snclock\scripts\screen_snclock.py rank --top 10
python .\skills\starwhisper-explore\scripts\eval_gate.py bar
python .\skills\starwhisper-night-plan\scripts\plan_night.py budget
python .\skills\starwhisper-varlen\scripts\eval_varlen.py contract
python .\skills\starwhisper-varlen\scripts\eval_varlen.py best --pool varlen
```

检查观测策略前先查看 `bar`；比较稀疏光变结果前先查看 `contract`。`gate` 未得到 `positive`、配置或目标表有错误、跨数据池比较或标签检查失败时，相关命令以退出码 1 结束。

离线测试需要 `pytest`：

```bash
python -m pytest skills/starwhisper-snclock/tests skills/starwhisper-explore/tests skills/starwhisper-night-plan/tests skills/starwhisper-varlen/tests -q
```

## 其他科研技能

以下 13 个技能来自 [tashan-research-skills](https://github.com/TashanGKD/tashan-research-skills)，资料来源、例子和写作要求已按天文研究调整。使用时先阅读各目录的 `astronomy.md`，再按 `SKILL.md` 执行。

| 任务 | 技能 | 说明 |
| --- | --- | --- |
| 查文献 | [giiisp-paper-search-apis](giiisp-paper-search-apis/SKILL.md) | 优先查询 NASA ADS 和 arXiv `astro-ph`，Giiisp 作补充；保留 bibcode / arXiv id |
| 做文献调研 | [sci-employee-deep-research](sci-employee-deep-research/SKILL.md) | 整理暂现源、巡天和望远镜相关证据，区分观测、仿真与决策日志 |
| 检查引用 | [papercheck](papercheck/SKILL.md) | 核对正文引用、参考文献及其对论述的支持程度 |
| 审论文或学位论文 | [thesis-audit-reviewer](thesis-audit-reviewer/SKILL.md) | 检查方法、引用、坐标、测光与时间系统，以及结论是否超出证据 |
| 提出研究假设 | [scispark](scispark/SKILL.md) | 从文献和已有结果提出可检验假设，保留来源并标出推测部分 |
| 建立基线 | [research-baseline-builder](research-baseline-builder/SKILL.md) | 明确样本、输入、输出和数据划分，选择与数据形态相符的简单基线 |
| 设计实验 | [experiment-design](experiment-design/SKILL.md) | 安排注入回收、夜次与观测场分组、策略对照和影子运行 |
| 统计分析 | [statistical-analysis](statistical-analysis/SKILL.md) | 按预先确定的分析计划计算检验、效应量和区间，检查选择效应与多重检验 |
| 修改中文稿 | [scientific-humanization](scientific-humanization/SKILL.md) | 改生硬句子和模板措辞，保留研究事实、术语和证据强度 |
| 学术写作 | [academic-writing](academic-writing/SKILL.md) | 撰写或修改论文、审稿意见、回复与天文项目申请 |
| 画示意图 | [giiisp-scientific-image-generation](giiisp-scientific-image-generation/SKILL.md) | 绘制流程和系统示意图；有实际刻度的光变或光谱图仍需由数据绘制 |
| 做汇报 | [visual-deck-builder](visual-deck-builder/SKILL.md) | 根据论文和巡天材料制作汇报，注明已发表结果与合成实验的区别 |
| 记录研究偏好 | [cognitive-profile](cognitive-profile/SKILL.md) | 按用户要求记录研究方向、常用星表和写作偏好，不保存账号或未公开目标表 |

论文检索可先查看请求计划，不发送网络请求：

```powershell
python .\skills\giiisp-paper-search-apis\scripts\ads_first_search.py --query "StarWhisper Telescope" --dry-run
```

## 数据和外部服务

| 变量 | 用途 | 未设置时 |
| --- | --- | --- |
| `STARWHISPER_ROOT` | 定位仓库数据与 NGSS 配置 | 从脚本位置向上查找；SN Clock 找不到数据时需指定 `--csv`，Explore 使用附带指标表，夜计划使用示例配置，varlen 使用附带结果表 |
| `ADS_API_TOKEN` | NASA ADS 文献检索 | `--dry-run` 只查看请求；实际检索可回退到 arXiv `astro-ph` |
| `GIIISP_AUTH_TOKEN` | Giiisp 补充检索、科研绘图与 PPT 接口 | 可构造请求，不能声称检索或生成已经完成 |
| `MINERU_API_TOKEN` | 论文 PDF 解析 | 使用可用的本地解析方法，并说明缺失内容 |
| `DASHSCOPE_API_KEY` | 部分汇报生成与视觉审查步骤 | 按技能说明使用可用方法；未执行的生成或审查步骤需明确标注 |

外部服务的密钥、依赖和可用性需分别确认。PPT 自动修订依赖相邻的科研绘图技能，但当前审查脚本不支持修订流程传入的 `--style-contract`，因此这条自动流程尚不能直接运行。使用其他可用工具生成与逐页审查时，应记录实际执行的步骤；本地格式检查通过不代表图像生成或视觉审查已经完成。

## 其他项目资料

以下内容保留代码或研究材料，未另做成此包中的安装技能：

| 项目 | 资料 | 说明 |
| --- | --- | --- |
| 天文语言模型 | [训练数据](../LLM_Data/README.md)、[StarWhisper3](https://www.modelscope.cn/models/AstroYuYang/StarWhisper3) | 问答文本与模型资源；本仓库尚未收录 4.0 权重 |
| 光变分类 | [StarWhisper LC](../StarWhisper_LC/README.md) | Kepler/K2 测试代码，与稀疏 ZTF/ATLAS 基准分开使用 |
| 脉冲星 | [StarWhisper-Pulsar](https://github.com/ACMISLab/StarWhisper-Pulsar) | 代码在独立仓库；候选分类不等于确认星表 |
| 全天相机 | [AllSky-Camera-XL](../AllSky-Camera-XL/README.md) | 生成观测序列文件，不表示序列已执行 |
| 恒星光谱 | [Low-SNR-Stellar-Spectra-as-Language](../Low-SNR-Stellar-Spectra-as-Language/README.md) | 完整 tokenized 数据集仍待发布 |
| 真假源分类样机 | [GOTTA_Prototype](../GOTTA_Prototype) | 样机评分不等于 broker 警报 |
| 虚拟司天工作流 | [SitianClaw](https://github.com/Yu-Yang-Li/SitianClaw) | `snc-*` 技能在该仓库维护 |

## 结果说明与许可

- 这些技能不替代 NGSS，也不向设备下发指令。`starwhisper-night-plan` 不调用 `/manipulate_nina` 或 `/ftp_transfer`。
- 报告中分别说明合成实验、脱敏日志回放和真实观测；年龄估计不等于光谱分类，候选不等于发现。
- 空结果、未达到预注册标准和缺少数据都应如实报告，不为得到正向结论而放宽条件。

4 个 `starwhisper-*` 技能采用 Apache-2.0；他山改编部分采用 MIT，作者与许可证见 [NOTICE.md](NOTICE.md)。
