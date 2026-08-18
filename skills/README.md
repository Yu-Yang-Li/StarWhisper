# 技能包

两套技能放在一起，安装方式相同：复制 `skills/<name>/` 到 Codex 或 Cursor 的 skills 目录。

1. **星语本线** `starwhisper-*`：把仓库里已经公开的 LLM、光变、Telescope、Explore、全天相机、稀疏光变、低信噪光谱和虚拟司天收成可路由的技能。
2. **天文科研（他山改编）**：文献、假设、数据合同、写作。默认 NASA ADS / arXiv `astro-ph`。

先装 [`starwhisper-index`](starwhisper-index/SKILL.md)。问“StarWhisper 是什么 / 开哪个目录 / 装哪个技能”时用它。

```powershell
Copy-Item -Recurse .\skills\starwhisper-index "$env:USERPROFILE\.codex\skills\starwhisper-index"
Copy-Item -Recurse .\skills\starwhisper-index "$env:USERPROFILE\.cursor\skills\starwhisper-index"
```

来源与许可证见 [`NOTICE.md`](NOTICE.md)。本线技能跟主仓库一样是 Apache-2.0；他山改编部分是 MIT。

## 星语本线

| 技能 | 对应过去内容 | 做什么 |
| --- | --- | --- |
| 总路由 | 整条时间线 | [`starwhisper-index`](starwhisper-index/SKILL.md) |
| 语言模型 | `LLM_Data/`、StarWhisper 3 | [`starwhisper-llm`](starwhisper-llm/SKILL.md) |
| 光变分类 | `StarWhisper_LC/` | [`starwhisper-lc`](starwhisper-lc/SKILL.md) |
| 脉冲星 | ACMISLab/StarWhisper-Pulsar | [`starwhisper-pulsar`](starwhisper-pulsar/SKILL.md) |
| 观测 agent | `NGSS/`、Telescope 论文 | [`starwhisper-telescope`](starwhisper-telescope/SKILL.md) |
| 决策边界 | `explore/` | [`starwhisper-explore`](starwhisper-explore/SKILL.md) |
| 全天相机 | `AllSky-Camera-XL/` | [`starwhisper-allsky`](starwhisper-allsky/SKILL.md) |
| 稀疏光变 | `Early Classification from Sparse Light Curves/` | [`starwhisper-sparse-lc`](starwhisper-sparse-lc/SKILL.md) |
| 低信噪光谱 | `Low-SNR-Stellar-Spectra-as-Language/` | [`starwhisper-lowsnr-spectra`](starwhisper-lowsnr-spectra/SKILL.md) |
| 虚拟司天 | SitianClaw、GOTTA | [`starwhisper-sitian`](starwhisper-sitian/SKILL.md) |

本线技能默认读代码和说明。Telescope / All-sky 只有在用户明确说本地栈已接通时才涉及 NINA 或流水线；否则只解释，不下令。虚拟司天主技能在 [SitianClaw](https://github.com/Yu-Yang-Li/SitianClaw)，这里只做路由。

## 天文科研（他山改编）

从 [tashan-research-skills](https://github.com/TashanGKD/tashan-research-skills) 选出 13 个技能，按天文学改了默认值。每个技能先读同目录 `astronomy.md`。论文检索可跑：

```powershell
python .\skills\giiisp-paper-search-apis\scripts\ads_first_search.py --query "StarWhisper Telescope" --dry-run
```

<div align="center">

![StarWhisper astronomy research skills](https://yu-yang-li.github.io/StarWhisper/assets/starwhisper-skills-matrix.jpg)

</div>

### 文献证据

| 技能 | 路径 | 天文适配后做什么 |
| --- | --- | --- |
| 论文检索 | [`giiisp-paper-search-apis`](giiisp-paper-search-apis/SKILL.md) | 先查 NASA ADS 和 arXiv `astro-ph`，再回退 Giiisp OA；输出 bibcode / arXiv id，不编造文献。 |
| 深度研究 | [`sci-employee-deep-research`](sci-employee-deep-research/SKILL.md) | 围绕暂现源、巡天和望远镜智能体拆关键词、找证据，并标出“观测 / 仿真 / 决策日志”的证据边界。 |
| 论文审查 | [`thesis-audit-reviewer`](thesis-audit-reviewer/SKILL.md) | 审查坐标框架、测光系统、时间系统、选择函数，以及把合成候选写成“发现”的越界表述。 |

### 研究构思

| 技能 | 路径 | 天文适配后做什么 |
| --- | --- | --- |
| 假设生成 | [`scispark`](scispark/SKILL.md) | 生成可检验的时域天文或观测策略假设，每条假设绑定 ADS/arXiv 记录或标成 speculative。 |
| 数据处理 | [`research-baseline-builder`](research-baseline-builder/SKILL.md) | 把问题写成光变曲线 / 光谱 / FITS / 警报流 / 决策时隙的数据合同，先定样本单位再谈模型。 |
| 实验设计 | [`experiment-design`](experiment-design/SKILL.md) | 采集或回放前锁定注入回收、夜次区组、策略对照和影子运行，而不是套临床 RCT。 |
| 统计分析 | [`statistical-analysis`](statistical-analysis/SKILL.md) | 按预注册计划做验证性统计，并检查选择效应、小样本计数区间和候选多重检验。 |

### 成果表达

| 技能 | 路径 | 天文适配后做什么 |
| --- | --- | --- |
| 文本润色 | [`scientific-humanization`](scientific-humanization/SKILL.md) | 改中文天文稿的腔调，但不改 bibcode、滤镜、MJD 和发现用语的证据强度。 |
| 学术写作 | [`academic-writing`](academic-writing/SKILL.md) | 覆盖 AAS/MNRAS/A&A 论文、审稿回复和 NSFC 天文申请的写—审—改—投。 |
| 科研绘图 | [`giiisp-scientific-image-generation`](giiisp-scientific-image-generation/SKILL.md) | 画观测流程和系统示意图；带真实刻度的光变/光谱图应回数据重绘，不让模型伪造坐标轴。 |
| PPT 制作 | [`visual-deck-builder`](visual-deck-builder/SKILL.md) | 把论文、巡天和望远镜智能体材料做成可汇报文稿，并分开已发表结果与合成实验结果。 |

### 协作沉淀

| 技能 | 路径 | 天文适配后做什么 |
| --- | --- | --- |
| 引用合规 | [`papercheck`](papercheck/SKILL.md) | 核正文引用、参考文献和 ADS 可解析性；英文天文稿默认 AAS 格式。 |
| 科研画像 | [`cognitive-profile`](cognitive-profile/SKILL.md) | 记录子领域、常用星表和表述边界。不存望远镜账号、FTP 或未公开目标表。 |

## 环境变量

| 变量 | 用在 | 没有时 |
| --- | --- | --- |
| `ADS_API_TOKEN` | 论文检索（NASA ADS） | dry-run，或改走 arXiv `astro-ph` |
| `GIIISP_AUTH_TOKEN` | 论文检索补充、科研绘图、PPT | 构造请求，不假装已经查到 |
| `MINERU_API_TOKEN` | 论文审查 PDF 解析 | 本地回退 |

## 使用边界

- 科研技能不能替代 `NGSS`，也不能对硬件下发指令。
- 本线 Telescope / All-sky 技能在未接通真实栈时只读代码。
- 合成环境、脱敏日志和真实硬件必须分开写。
- 没有密钥就 dry-run，不编文献。
