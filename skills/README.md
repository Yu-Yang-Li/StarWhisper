# 天文科研技能

这里有 17 个技能：4 个用于读取星语的数据和检查观测配置，13 个改编自他山科研技能。先按任务选择，再读对应的 `SKILL.md`；较长流程、接口和检查细则放在各技能的参考文件中。

[项目首页](../README.md) · [安装与环境](INSTALL.md) · [仓库结构](../docs/architecture.md) · [来源与许可](NOTICE.md)

## 先选任务

| 任务 | 技能 | 会得到什么 |
| --- | --- | --- |
| 查文献 | [天文文献检索](giiisp-paper-search-apis/SKILL.md) | 候选论文、实际来源、论文编号与核验状态 |
| 比较研究方法、整理证据 | [天文文献调研](sci-employee-deep-research/SKILL.md) | 有引用的报告、分歧与待核验问题 |
| 提出研究假设 | [Scispark](scispark/SKILL.md) | 可检验的假设、文献依据与检验方法 |
| 做第一次实验 | [建立研究基线](research-baseline-builder/SKILL.md) | 数据说明、划分方法、简单模型与实验计划 |
| 安排观测或对照实验 | [实验设计](experiment-design/SKILL.md) | 分组、随机化、样本量与预先确定的判据 |
| 分析数据 | [统计分析](statistical-analysis/SKILL.md) | 检验、效应量、区间与方法限制 |
| 撰写论文、回复或申请书 | [学术写作](academic-writing/SKILL.md) | 与已有研究证据一致的稿件 |
| 修改生硬的中文 | [科研文本修改](scientific-humanization/SKILL.md) | 保留事实与作者表达的修改稿 |
| 检查引用 | [PaperCheck](papercheck/SKILL.md) | 引用对应关系、支持程度与证据限制 |
| 审查论文全文 | [论文与学位论文审查](thesis-audit-reviewer/SKILL.md) | 可定位、有依据的批注或审核报告 |
| 绘制或修改示意图 | [科研示意图](giiisp-scientific-image-generation/SKILL.md) | 图片、版本记录与检查结果 |
| 制作或修改汇报 | [天文汇报与 PPT](visual-deck-builder/SKILL.md) | PPTX、页面预览与逐页审查记录 |
| 记录长期研究偏好 | [认知画像](cognitive-profile/SKILL.md) | 按用户要求保存的偏好与来源 |

上述技能来自 [tashan-research-skills](https://github.com/TashanGKD/tashan-research-skills)，资料来源、例子和写作要求已按天文研究调整。各技能的 `astronomy.md` 说明相应的天文约定。

## 读取星语数据与观测配置

这 4 个原生技能有 18 个子命令，只在本地读取文件，不访问网络或控制设备。

| 技能 | 做什么 | 子命令 |
| --- | --- | --- |
| [SN Clock](starwhisper-snclock/SKILL.md) | 排序和筛选历史年龄预测，检查数据留存与预测时间 | `describe`、`rank`、`screen`、`window`、`audit` |
| [Explore](starwhisper-explore/SKILL.md) | 按预注册标准比较合成环境的观测策略指标 | `bar`、`table`、`gate` |
| [夜计划](starwhisper-night-plan/SKILL.md) | 检查 NGSS 配置、时间预算、目标表与接口类型 | `check-config`、`budget`、`lint-targets`、`endpoints` |
| [稀疏光变比较](starwhisper-varlen/SKILL.md) | 在同一数据设定下比较结果，检查点数与标签 | `contract`、`table`、`best`、`compare`、`labels`、`check` |

从仓库根目录可以先运行：

```bash
python skills/starwhisper-snclock/scripts/screen_snclock.py rank --top 10
python skills/starwhisper-explore/scripts/eval_gate.py bar
python skills/starwhisper-night-plan/scripts/plan_night.py budget
python skills/starwhisper-varlen/scripts/eval_varlen.py contract
```

需要 Python 3.10 或更新版本，只依赖标准库。安装后的数据定位、示例配置与依赖见 [安装文档](INSTALL.md)；回归测试见 [贡献说明](../CONTRIBUTING.md)。

## 怎样阅读技能说明

`SKILL.md` 先回答任务、开始方法与结果；`astronomy.md` 给出天文约定；`references/` 只在遇到相应流程或问题时阅读。`scripts/` 是实际工具，`templates/` 是输入或报告模板。除明确要求进入生成工作目录的命令外，各技能命令从自身目录运行。

安装脚本、分组选择和 Codex / Cursor 路径统一说明在 [INSTALL.md](INSTALL.md)。科研训练代码与模型资料仍在各项目目录，见 [仓库结构](../docs/architecture.md)。

## 解释结果与许可

候选筛选不等于新发现，分类分数不等于确认分类；合成实验、历史预测和真实观测分别报告。空结果、未达到判据或缺少数据时如实说明。夜计划工具不向设备下发指令，硬件安全规则与台站权限以实际观测系统为准。

4 个 `starwhisper-*` 技能采用 Apache-2.0；改编部分采用 MIT，保留原作者与许可证，见 [NOTICE.md](NOTICE.md)。
