# 天文科研技能包

从 [tashan-research-skills](https://github.com/TashanGKD/tashan-research-skills) 选出 13 个科研技能，按天文学默认值做了强化：文献走 NASA ADS / arXiv `astro-ph`，数据合同按光变曲线、光谱、FITS 和观测决策日志来写，统计补上选择效应，写作默认 AAS / MNRAS 而不是临床 RCT。

每个技能仍以 `SKILL.md` 为入口。先读同目录的 `astronomy.md`，再执行原来的脚本和模板。

安装示例（Codex / Cursor / Claude Code 均可）：

```powershell
Copy-Item -Recurse .\skills\giiisp-paper-search-apis "$env:USERPROFILE\.codex\skills\giiisp-paper-search-apis"
```

来源与许可证见 [`NOTICE.md`](NOTICE.md)。上游仓库是 MIT；StarWhisper 主仓库代码仍是 Apache-2.0。

<div align="center">

![StarWhisper astronomy research skills](https://cdn.jsdelivr.net/gh/Yu-Yang-Li/StarWhisper@main/docs/assets/starwhisper-skills-matrix.jpg)

</div>

## 文献证据

| 技能 | 路径 | 天文适配后做什么 |
| --- | --- | --- |
| 论文检索 | [`giiisp-paper-search-apis`](giiisp-paper-search-apis/SKILL.md) | 先查 NASA ADS 和 arXiv `astro-ph`，再回退 Giiisp OA；输出 bibcode / arXiv id，不编造文献。 |
| 深度研究 | [`sci-employee-deep-research`](sci-employee-deep-research/SKILL.md) | 围绕暂现源、巡天和望远镜智能体拆关键词、找证据，并标出“观测 / 仿真 / 决策日志”的证据边界。 |
| 论文审查 | [`thesis-audit-reviewer`](thesis-audit-reviewer/SKILL.md) | 审查坐标框架、测光系统、时间系统、选择函数，以及把合成候选写成“发现”的越界表述。 |

## 研究构思

| 技能 | 路径 | 天文适配后做什么 |
| --- | --- | --- |
| 假设生成 | [`scispark`](scispark/SKILL.md) | 生成可检验的时域天文或观测策略假设，每条假设绑定 ADS/arXiv 记录或标成 speculative。 |
| 数据处理 | [`research-baseline-builder`](research-baseline-builder/SKILL.md) | 把问题写成光变曲线 / 光谱 / FITS / 警报流 / 决策时隙的数据合同，先定样本单位再谈模型。 |
| 实验设计 | [`experiment-design`](experiment-design/SKILL.md) | 采集或回放前锁定注入回收、夜次区组、策略对照和影子运行，而不是套临床 RCT。 |
| 统计分析 | [`statistical-analysis`](statistical-analysis/SKILL.md) | 按预注册计划做验证性统计，并检查选择效应、小样本计数区间和候选多重检验。 |

## 成果表达

| 技能 | 路径 | 天文适配后做什么 |
| --- | --- | --- |
| 文本润色 | [`scientific-humanization`](scientific-humanization/SKILL.md) | 改中文天文稿的腔调，但不改 bibcode、滤镜、MJD 和发现用语的证据强度。 |
| 学术写作 | [`academic-writing`](academic-writing/SKILL.md) | 覆盖 AAS/MNRAS/A&A 论文、审稿回复和 NSFC 天文申请的写—审—改—投。 |
| 科研绘图 | [`giiisp-scientific-image-generation`](giiisp-scientific-image-generation/SKILL.md) | 画观测流程和系统示意图；带真实刻度的光变/光谱图应回数据重绘，不让模型伪造坐标轴。 |
| PPT 制作 | [`visual-deck-builder`](visual-deck-builder/SKILL.md) | 把论文、巡天和望远镜智能体材料做成可汇报文稿，并分开已发表结果与合成实验结果。 |

## 协作沉淀

| 技能 | 路径 | 天文适配后做什么 |
| --- | --- | --- |
| 引用合规 | [`papercheck`](papercheck/SKILL.md) | 核正文引用、参考文献和 ADS 可解析性；英文天文稿默认 AAS 格式。 |
| 科研画像 | [`cognitive-profile`](cognitive-profile/SKILL.md) | 记录子领域、常用星表和表述边界。不存望远镜账号、FTP 或未公开目标表。 |

## 使用边界

- 这些技能辅助文献、设计、写作和检查，**不能替代** `NGSS` 里的真实观测代码，也不能对硬件下发指令。
- 没有 ADS / Giiisp / MinerU 密钥时，技能应 dry-run 或走本地回退，而不是伪造检索结果。
- 合成环境、脱敏日志和真实硬件必须分开写。
