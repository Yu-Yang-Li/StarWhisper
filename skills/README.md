# 技能包

两套技能放在一起，都是**做事的**，不是介绍性的：每个技能有决策规则、可跑的脚本和回归测试。没有密钥就 dry-run，没有数据就报缺，不编文献，不下硬件指令。

```powershell
powershell -File .\skills\install_native.ps1
python .\skills\starwhisper-snclock\scripts\screen_snclock.py rank --top 10
```

装到 `~/.codex/skills` 或 `~/.cursor/skills` 之后，把本仓库路径设成 `STARWHISPER_ROOT`，脚本才找得到 `snclock/`、`explore/`、`NGSS/`。

许可证见 [`NOTICE.md`](NOTICE.md)：本线 Apache-2.0，他山改编部分 MIT。

## 星语本线

| 技能 | 做什么 | 主命令 |
| --- | --- | --- |
| [`starwhisper-snclock`](starwhisper-snclock/SKILL.md) | 把爆发年龄预测筛成年轻超新星候选清单，并审计证据强度 | `screen_snclock.py rank --top 10` |
| [`starwhisper-explore`](starwhisper-explore/SKILL.md) | 按预注册门槛逐条判定策略对比 | `eval_gate.py gate --agent rule_agent` |
| [`starwhisper-night-plan`](starwhisper-night-plan/SKILL.md) | 校验 observe_config、算一夜容量、lint 目标表 | `plan_night.py budget` |

几个例子：

```powershell
python .\skills\starwhisper-snclock\scripts\screen_snclock.py screen --tier strict --require-redshift
python .\skills\starwhisper-explore\scripts\eval_gate.py gate --agent rule_agent
python .\skills\starwhisper-night-plan\scripts\plan_night.py lint-targets --targets targets.csv
```

`eval_gate gate` 判定非正向、`plan_night` 有 error 时退出码为 1，可以直接进 CI。

下面这些线在本仓库是材料，没有技能。缺了就说缺了，不要假装跑过。

| 线 | 位置 | 边界 |
| --- | --- | --- |
| 语言模型 | `LLM_Data/`，[StarWhisper3](https://www.modelscope.cn/models/AstroYuYang/StarWhisper3) | 清洗后的问答文本，不是观测日志；4.0 未发布 |
| 光变分类 | `StarWhisper_LC/` | 测试代码，不是完整训练复现 |
| 脉冲星 | [ACMISLab/StarWhisper-Pulsar](https://github.com/ACMISLab/StarWhisper-Pulsar) | 未 vendored；候选分类不是确认星表 |
| 全天相机 | `AllSky-Camera-XL/` | 产出的是序列文件，不是已执行的夜次 |
| 稀疏光变 | `Early Classification from Sparse Light Curves/` | 测试集指标不是爆发时刻 |
| 低信噪光谱 | `Low-SNR-Stellar-Spectra-as-Language/` | 完整 tokenized 数据集仍是 coming soon |
| GOTTA 样机 | `GOTTA_Prototype/` | 样机分数不是 broker 警报 |
| 虚拟司天工作流 | [SitianClaw](https://github.com/Yu-Yang-Li/SitianClaw) | `snc-*` 装在那边跑，不要在这里重写 |

## 天文科研（他山改编）

从 [tashan-research-skills](https://github.com/TashanGKD/tashan-research-skills) 选出 13 个，按天文学改了默认值。每个技能先读同目录 `astronomy.md`。

```powershell
python .\skills\giiisp-paper-search-apis\scripts\ads_first_search.py --query "StarWhisper Telescope" --dry-run
```

<div align="center">

![StarWhisper astronomy research skills](https://yu-yang-li.github.io/StarWhisper/assets/starwhisper-skills-matrix.jpg)

</div>

### 文献证据

| 技能 | 天文适配后做什么 |
| --- | --- |
| [`giiisp-paper-search-apis`](giiisp-paper-search-apis/SKILL.md) | 先查 NASA ADS 和 arXiv `astro-ph`，再回退 Giiisp OA；输出 bibcode / arXiv id，不编造文献 |
| [`sci-employee-deep-research`](sci-employee-deep-research/SKILL.md) | 围绕暂现源、巡天和望远镜智能体找证据，标出"观测 / 仿真 / 决策日志"的边界 |
| [`thesis-audit-reviewer`](thesis-audit-reviewer/SKILL.md) | 审坐标框架、测光系统、时间系统、选择函数，以及把合成候选写成"发现"的越界表述 |

### 研究构思

| 技能 | 天文适配后做什么 |
| --- | --- |
| [`scispark`](scispark/SKILL.md) | 生成可检验的时域天文或观测策略假设，每条绑定 ADS/arXiv 记录或标成 speculative |
| [`research-baseline-builder`](research-baseline-builder/SKILL.md) | 把问题写成光变 / 光谱 / FITS / 警报流 / 决策时隙的数据合同，先定样本单位再谈模型 |
| [`experiment-design`](experiment-design/SKILL.md) | 锁定注入回收、夜次区组、策略对照和影子运行，而不是套临床 RCT |
| [`statistical-analysis`](statistical-analysis/SKILL.md) | 按预注册计划做验证性统计，检查选择效应、小样本计数区间和候选多重检验 |

### 成果表达

| 技能 | 天文适配后做什么 |
| --- | --- |
| [`scientific-humanization`](scientific-humanization/SKILL.md) | 改中文天文稿的腔调，不改 bibcode、滤镜、MJD 和发现用语的证据强度 |
| [`academic-writing`](academic-writing/SKILL.md) | 覆盖 AAS/MNRAS/A&A 论文、审稿回复和 NSFC 天文申请的写—审—改—投 |
| [`giiisp-scientific-image-generation`](giiisp-scientific-image-generation/SKILL.md) | 画流程和系统示意图；带真实刻度的光变/光谱图回数据重绘，不让模型伪造坐标轴 |
| [`visual-deck-builder`](visual-deck-builder/SKILL.md) | 把论文和巡天材料做成可汇报文稿，分开已发表结果与合成实验结果 |

### 协作沉淀

| 技能 | 天文适配后做什么 |
| --- | --- |
| [`papercheck`](papercheck/SKILL.md) | 核正文引用、参考文献和 ADS 可解析性；英文天文稿默认 AAS 格式 |
| [`cognitive-profile`](cognitive-profile/SKILL.md) | 记录子领域、常用星表和表述边界；不存望远镜账号、FTP 或未公开目标表 |

## 环境变量

| 变量 | 用在 | 没有时 |
| --- | --- | --- |
| `STARWHISPER_ROOT` | 本线技能定位仓库数据 | 从脚本位置向上找；找不到就用内置样例并标注 |
| `ADS_API_TOKEN` | 论文检索（NASA ADS） | dry-run，或改走 arXiv `astro-ph` |
| `GIIISP_AUTH_TOKEN` | 论文检索补充、科研绘图、PPT | 构造请求，不假装已经查到 |
| `MINERU_API_TOKEN` | 论文审查 PDF 解析 | 本地回退 |

## 使用边界

- 科研技能不能替代 `NGSS`，也不能对硬件下发指令。
- `starwhisper-night-plan` 只做检查，任何情况下不调用 `/manipulate_nina` 和 `/ftp_transfer`。
- 合成环境、脱敏日志和真实硬件必须分开写。
- 年龄估计不是光谱分类，候选不是发现。
- 筛空、判负、缺数据都是结论，照报，不要放宽条件凑数。
