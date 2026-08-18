---
name: academic-writing
description: Write, review, and submit astronomy manuscripts for AAS/MNRAS/A&A/PASP and Chinese NSFC astronomy proposals. Use for papers, referee reports, rebuttals, cover letters, and grant text. Do not retrieve literature or audit citation authenticity.
license: MIT
---

# 学术写作 Academic Writing

## StarWhisper astronomy overlay

Read [`astronomy.md`](astronomy.md) first. That file sets the astronomy defaults for this copy.

Literature: NASA ADS, then arXiv `astro-ph.*`. Do not invent papers.
A synthetic Explore run, a classifier score, or a demo candidate is not a discovery.
This skill does not send telescope commands.


把一份研究成果从草稿推进到"可投、可评、可辩、可资助"。覆盖论文写作、同行评审、回稿修订、
基金申请、投稿材料五条主线。它管**表达与协作**，不管**发现**（文献检索/综述综合走文献技能），
也不管**引用真实性与格式合规**（走引用合规技能）。

> 本技能整合了学术出版通行的写作、评审、修订、基金与投稿最佳实践，遵循 IMRaD、
> CONSORT/STROBE/PRISMA/ARRIVE 报告规范与主要资助机构（NSF/NIH/DOE/DARPA）的申报要求。

## 资源（按需加载）

- `templates/peer_review_report.md` — 审稿报告模板（模式 2）
- `templates/editorial_decision.md` — 编辑决定信 + 修订路线图（模式 2）
- `templates/response_to_reviewers.md` — R→A→C 审稿回复模板（模式 3）
- `templates/nih_specific_aims.md` — NIH Specific Aims 模板（模式 4）
- `templates/cover_letter.md` — 投稿 cover letter 模板（模式 5）
- `references/peer_review_checklist.md` — 同行评审 7 阶段系统清单细则
- `references/grant_agencies.md` — NSF/NIH/DOE/DARPA 要点细则

## 何时使用

- "帮我写/改这篇论文（或引言/方法/讨论）""按 Nature/NeurIPS 风格重构这一节"
- "审一下这篇论文 / 生成审稿意见""从方法、统计、设计角度挑问题"
- "帮我回复审稿人 / 排修订优先级 / 写 rebuttal"
- "写一份 NSF / NIH / 面上项目 申请书""把想法拆成 aims / significance / 可行性 / 预算"
- "生成摘要 / 标题 / cover letter / 会议摘要""按目标期刊格式适配""帮我选目标期刊"
- "核对标题/摘要/方法/结果之间数字、术语、结论是否前后一致"

---

## 模式 1 · 论文写作（manuscript drafting）

按 IMRaD 组织，先定"论文故事"再改句子。核心纪律：

- **一段一义**：段落第一句就说明本段要做什么；名词自足，术语先定义再复用。
- **句间有关系**：因果 / 对比 / 递进 / 细化，任意两句要连得上。
- **逆向大纲自检**：写完一节，倒推出论点→各段主题句→证据点，映射不上的段落删改。
- **视觉即内容**：teaser/pipeline 图、少墨表格、统一排版当正文对待。
- **claim–evidence 硬约束**：Abstract/Introduction 每个主张都要能对到实验证据。

**分节撰写**：各节按需加载章节指南（引言/摘要/相关工作/方法/实验/结论）；方法节要能被独立复现
（材料、参数、软件版本、统计计划齐全）。

**输出契约**（改写/起草章节时返回）：
1. 紧凑小标题大纲(3–7 点)；2. 标注段落角色的改写段落(opening/challenge/method/advantage/evidence/limitation)；
3. 五维自审清单(贡献/清晰/实验强度/评估完整/方法合理)；4. `Claim: … | Evidence: … | Status: supported/needs evidence` 映射表。

## 模式 2 · 同行评审（peer review）

一条评审流水线，把"多视角审稿人"与"系统化评审清单"合成三个阶段——**清单是每位审稿人评审时的
检查维度，面板是让不同审稿人从不同角度用这份清单**，二者不是二选一。

**阶段 0 · 配置审稿视角**
先识别论文领域与方法类型，据此配置 5 个互不重叠的审稿视角：主编（期刊契合/原创性/整体质量）、
方法学审稿人（研究设计/统计效力/可复现）、领域审稿人（文献覆盖/理论框架/领域贡献）、
视角审稿人（跨学科关联/实际影响）、Devil's Advocate（核心论点挑战/逻辑漏洞/最强反驳）。

**阶段 1 · 各视角独立评审（共用 7 维检查清单）**
每位审稿人对论文走一遍系统清单，只在自己的视角上给出深度意见：
初评（中心问题/主要发现/是否契合期刊/有无致命缺陷）→ 逐节评审（摘要/引言/方法/结果/讨论）→
方法与统计严谨性 → 可复现与透明 → 图表质量与完整性 → 伦理 → 写作质量。
方法阶段必查：样本量与功效、随机化与盲法、纳排标准、软件/版本、多重比较校正；
按学科核对报告规范 **CONSORT / STROBE / PRISMA / ARRIVE**。完整清单细则见
`references/peer_review_checklist.md`（按需加载）；每位审稿人用 `templates/peer_review_report.md`。

**阶段 2 · 编辑综合与决定**
综合 5 份报告，标出共识（多数认可/多数提出）与分歧（编辑仲裁并给理由），产出结构化决定信 +
分优先级的修订路线图。用 `templates/editorial_decision.md`。

**铁律（防常见失败）**：
① 5 个视角独立评审，不互相参照，避免"假多样性"；
② 综合者不得编造意见，每条须溯源到具体审稿报告；
③ Devil's Advocate 判定 CRITICAL 时，决定不能是 Accept；
④ **只读约束**：评审只产报告，绝不直接改稿；
⑤ 每条批评须含"错在哪、在何处、怎么改"，禁泛泛套话与谄媚打分。

**输出**：总评（推荐 accept/minor/major/reject + 各 2–3 条优缺点）→ Major comments（编号：问题+为何+解法+是否必须）→ Minor comments（定位到节/段/图）。

## 模式 3 · 回稿与修订（response & revision）

- 把审稿意见拆成可执行修订清单并分级(major/minor)、排优先级；
- 逐条起草 point-by-point 回复，标注每处改动落在正文哪一节；
- 用 R&R 可追溯矩阵(Reviewer 意见 | 作者回应 | 已核验?)防"敷衍式全部已改"。
- 模板：`templates/response_to_reviewers.md`（R→A→C 格式 + 自查矩阵）。

## 模式 4 · 基金 / 项目申请（proposal writing）

按资助机构套路组织（覆盖 NSF/NIH/DOE/DARPA）：

- **NIH Specific Aims(1 页)**：知识缺口与意义→长期目标与当前目标→中心假设→2–4 个 aims(动词开头、独立又互补)→预期产出与影响→payoff 段。用 `templates/nih_specific_aims.md`。
- **通用要素**：意义/创新、研究设计与方法、初步数据/可行性、时间线与里程碑(Gantt)、团队与分工、预算与逐项论证。
- **常见失败**：意义不清、创新增量当变革、方法细节不足判不了可行性、预算与活动/时间线不匹配、超页数。
- **各机构差异**（NSF 双权重 Broader Impacts / DOE cost sharing / DARPA 阶段里程碑）见 `references/grant_agencies.md`。

## 模式 5 · 投稿材料与全文一致性

- 生成投稿件：cover letter（用 `templates/cover_letter.md`）、结构化摘要、会议摘要(压缩全文)、标题/摘要优化；按目标期刊格式适配、期刊匹配。
- **全文一致性核验**：标题/摘要/方法/结果/图表间的数字、术语、结论是否自洽。

> **引用相关一律委托「引用合规」技能**：正文—参考文献 parity、DOI 有效性、引用风格、GB/T 7714 / 期刊格式、
> claim 是否有引用支撑等，都由专门的引用合规 skill 负责，本技能不重复实现。写作/投稿时如需引用核验，转交该技能。

---

## 质量红线（去糟粕）

- **不编造数据/审稿意见**：审稿点必须可核验、可溯源；核验类只报事实差异。引用真实性/合规交「引用合规」技能。
- **只做协作产出，不做文献发现**：literature-review / systematic-review / 综述综合转文献模块。
- **保事实、保术语、保数字**：写作/改写不得改动事实结论、专业术语与数值。
- **审稿只读、要具体**：不改稿；每条批评须含"错在哪、在何处、怎么改"，禁泛泛套话与谄媚打分。

## 与其它技能的边界

- 找文献 / 读文献 / 综述综合 → 文献检索 / 深度研究技能。
- 严格参考文献格式合规(GB/T 7714 等) → 「引用合规」技能。
- 研究该怎么设计(对照/样本量/统计计划) → 「实验设计」技能。
- 图像化机制配图 / PPT / 讲解视频 → 成果表达模块对应技能。
