# 天文使用说明

本技能沿用他山科研技能的脚本、模板和证据要求。处理天文问题时，使用下列资料来源和学科约定；具体步骤见 [SKILL.md](SKILL.md)。

## 资料来源

需要查证时，优先使用对应的学科资料：

| 内容 | 优先查阅 | 补充资料 |
| --- | --- | --- |
| 天文期刊论文 | NASA ADS，保留 bibcode | arXiv `astro-ph.*` |
| 预印本与方法 | arXiv `astro-ph.*` | ADS 中的 arXiv 记录 |
| 天体信息 | SIMBAD / NED | 用户提供的星表 |
| 星表与数据表 | VizieR / CDS | 论文附表 |
| 暂现源 | TNS / ATels / GCN | 用户提供的警报 |
| 台站与观测背景 | StarWhisper Telescope / NGSS 说明 | 台站公开资料 |

## 分析与表述

- 分别说明合成环境、脱敏日志回放和真实观测；安全联锁优先于智能体动作。
- 拟合、分类分数或仿真不能单独证明物理机制；合成暂现源和演示候选不能写成新发现。
- 保留数值、单位、滤镜、MJD/UTC、坐标和 bibcode。
- 按天体、夜次、观测场或实际数据单位设计分析，不直接套用临床 RCT、CONSORT 或湿实验的样本量默认值。
- 英文天文稿按目标期刊的 AAS / MNRAS / A&A / PASP 格式处理；中文申请书或学位论文需要 GB/T 7714 时，说明所用标准。

## 检索顺序

1. 将问题整理为天体、事件、方法和巡天关键词，保留 ZTF、TNS、NGSS、SiTian、GOTO、ATLAS 等名称。
2. 先运行 `scripts/ads_first_search.py`。需要查看请求而不联网时加 `--dry-run`；查询 ADS 时保留实际返回的 bibcode、题名和年份。
3. ADS 无可用结果或凭据时，同一脚本可回退到 arXiv `astro-ph.*`。预印本命中不能据此声称论文已经过同行评审。
4. Giiisp OA / arXiv 接口作为补充，按其鉴权要求使用 `GIIISP_AUTH_TOKEN`。
5. 返回简短结果表，列出 bibcode 或 arXiv id、题名、年份、发表信息、相关性和核验状态。

不编造 bibcode。ADS 不可用时说明原因；没有命中时保留空结果。
