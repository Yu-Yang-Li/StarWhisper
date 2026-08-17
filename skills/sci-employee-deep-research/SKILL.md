---
name: sci-employee-deep-research
description: 调用或模拟 Deep Research 流程，把论文检索、关键词拆解、候选证据和最终 answer 组织成可核验研究报告。适用于赛题讲解、开题调研、方案依据整理。Astronomy overlay: build evidence-bounded reviews for transients, surveys, telescope agents, and time-domain methods using ADS plus arXiv astro-ph.
---

# 深度研究员工

## StarWhisper astronomy overlay

This copy is adapted for astronomy research and telescope-agent work.
**Read [`astronomy.md`](astronomy.md) before following generic biomedical / clinical defaults in the rest of this file.**

Default literature route: NASA ADS, then arXiv `astro-ph.*`, then the original skill's search backend if credentials exist.
Do not claim a real hardware observing loop, a discovery, or a referee-ready result unless the user supplied that evidence.


## Use When

用户已经有研究问题，需要形成一份有结构、有引用、有证据边界的研究报告时使用。

不要直接替代论文检索。先用 `paper-search` 找候选或确认检索口径，再进入 Deep Research。

## 输入

- 研究问题。
- 已有候选论文或允许调用的检索接口。
- 页码、每页数量、是否返回原始检索结果。
- 需要的输出形态：报告、PPT 依据、赛题讲法、实操路线。

## 接口

Deep Research 接口：

```text
POST http://123.56.218.60:18000/api/research/ask
```

请求体核心字段：

```json
{
  "prompt": "研究主题或问题",
  "model": "qwen-deep-research",
  "keyword_model": "qwen-plus",
  "page_num": 1,
  "page_size": 5,
  "endpoint_names": [
    "searchArticlesByQuery1",
    "searchArxivByTitle",
    "searchArxivByAbstract",
    "searchArxivByArxivNo1",
    "searchArxiv"
  ],
  "include_raw": false
}
```

响应是 `text/event-stream`。必须持续消费事件并拼接最终报告，不能在后端等完整 `done` 后一次性返回。

## 实时进度转发

Deep Research 接口本身会中途返回 SSE。调用层必须把这些事件实时透出给用户，至少展示阶段、关键词、检索汇总、references 和 answer 生成片段。

推荐使用 `scripts/stream_deep_research.py` 消费 SSE 并输出 JSONL：

```powershell
python scripts/stream_deep_research.py `
  --prompt "large language models for scientific discovery survey"
```

每一行都是一个进度事件，可由服务端转成 SSE 或 WebSocket 消息：

| JSONL 事件 | 来源 SSE | 用户侧含义 |
|---|---|---|
| `stream_started` | 本地包装器 | 请求已开始 |
| `phase` | `phase` | 进入关键词规划、私有检索、写作等阶段 |
| `keywords_ready` | `keywords` | 关键词已生成 |
| `private_search_hit` | `private_search_hit` | 某个检索接口有命中或失败 |
| `private_search_summary` | `private_search_summary` | 检索总命中、失败接口已汇总 |
| `references_ready` | `references` | 候选 references 已可提前展示 |
| `answer_delta` | `delta` | answer 正在生成，可逐步展示 |
| `done` | `done` | 服务端完成 |
| `usage` | `usage` | token/用量信息 |
| `stream_final` | 本地包装器 | 本次流结束，给出 `answer_status` |

如果 30-45 秒内已收到 `references_ready` 但没收到 `done`，不要丢弃结果。UI 应展示 references，并把状态标为 `incomplete` 或“answer 生成中/未完成”。
做可用性探针或首屏体验测试时，可加 `--max-events 8`，让脚本在已有阶段/检索命中后优雅输出 `stream_interrupted` 和 `stream_final`，而不是等最终长答案。

如果通过 Nginx 或其他网关转发 SSE，必须关闭响应缓冲：

```nginx
proxy_buffering off;
```

后端响应头建议包含：

```text
Content-Type: text/event-stream
Cache-Control: no-cache
X-Accel-Buffering: no
```

## 已实测状态

2026-06-10 测试结果：

- `GET /health` 返回 HTTP 200，`ok: true`。
- 服务端模型列表包含 `qwen-deep-research` 和 `qwen-deep-research-2025-12-15`。
- `POST /api/research/ask` 能返回 SSE。
- 已观察到阶段：`KeywordPlanning`、`PrivateSearch`、`references`、`answer`、`done`。
- 测试主题“中文学术文本人味化生成与改写的评测方法”没有检索到 references，最终 answer 是澄清问题。
- 第二次测试主题“large language models for scientific discovery survey”返回 `totalResults=82`，解析到 50 条 references，并开始生成 answer；但 45 秒限制内未等到 `done`，标为 `references 已返回，answer 未完成`。
- 第三次测试主题“agentic scientific discovery deterministic data access benchmark”返回 `totalResults=31`，解析到 31 条 references，并开始生成 answer；35 秒限制内未等到 `done`，标为 `incomplete`。更聚焦的问题减少了混杂，但摘要检索仍会引入弱相关论文。
- 已新增 `scripts/parse_deep_research_sse.py`，可从 SSE 文本日志提取关键词、检索命中、references 数、answer 状态和前 10 条 references。
- 已新增 `scripts/stream_deep_research.py`，可实时转发 SSE 事件为 JSONL，避免调用层等最终 answer。

硬规则：

> 如果 `references` 为空或 `private_search_summary.totalResults=0`，不能把结果标为“研究报告”。只能标为“问题澄清”“检索失败记录”或“无证据初稿”。

> 如果 references 已返回但没有 `done`，不能丢弃结果。标为“answer 未完成”，保留 references、已生成片段和下一步继续等待/缩短问题/收窄候选。

## 工作流

1. 先写研究 brief：问题、边界、需要回答的 3-5 个子问题。
2. 优先读取 `paper-search` 给出的候选论文交接合同；没有候选时，先收窄关键词再调用接口。
3. 调用接口或构造 dry-run 请求体。
4. 记录 SSE 阶段：
   - `KeywordPlanning`
   - `PrivateSearch`
   - `ResearchPlanning`
   - `WebResearch`
   - `answer`
   - `KeepAlive`
5. 从事件中整理：
   - 关键词
   - 检索来源
   - 候选论文
   - 报告正文
   - 引用或证据片段
6. 输出研究报告，不只输出模型正文。
7. 给复核清单：哪些引用已核验，哪些仍需人工看全文。

## 与论文检索员工的串联

Deep Research 接收的理想输入不是一句宽泛问题，而是 `paper-search` 交来的候选包：

```json
{
  "research_question": "需要回答的问题",
  "candidate_papers": [],
  "exclude": [],
  "questions_to_answer": []
}
```

输出时必须补回：

```json
{
  "references_count": 0,
  "answer_status": "complete/incomplete/clarification/no-evidence",
  "evidence_used": [],
  "claims_need_fulltext_check": [],
  "retrieval_limits": []
}
```

## 报告结构

```text
研究问题：
边界：
关键词：
证据来源：
主要结论：
分论点：
引用与证据：
争议或不足：
可实操路线：
还需要核验：
```

## 失败处理

- 健康检查失败：记录服务不可用，不伪造报告。
- SSE 断流：保留已收到阶段、事件数和最后事件。
- 没有引用：报告只能标为初稿，不能当可提交证据。
- 候选论文不足：退回 `paper-search` 换词或扩源。
- `searchArticlesByQuery1` 返回 400：记录 endpoint、关键词和错误；不要重试刷接口，改用更短关键词或 arXiv/OpenAlex 回退。
- answer 只返回澄清问题：把它作为需求澄清，不要包装成调研结论。
- references 很多但混杂：退回 `paper-search` 做候选过滤，不要让 answer 引用不贴题论文。
- answer 超时或没有 `done`：保留 references 和已生成片段，标为 `incomplete`，不要当最终报告。
- 前端或托管层收到 `answer_status=incomplete` 时，先展示已返回 references 和 answer 片段，并把用户动作设为继续等待、收窄问题或回到候选论文过滤；不要让用户长时间只看到加载中。

## 最小实测记录格式

```text
健康检查：
请求主题：
请求字段：
观察到的 SSE 阶段：
关键词：
检索命中数：
失败 endpoint：
references 数量：
answer 类型：研究报告 / 澄清问题 / 无证据初稿
下一步：
```

## SSE 解析脚本

保存 SSE 日志后运行：

```powershell
python outputs\ai_scientist_open_source_research\deep_dive\scientific-digital-employee-skill-pack\deep-research\scripts\parse_deep_research_sse.py `
  outputs\ai_scientist_open_source_research\deep_dive\api_tests\round4_deep_research_sse.txt `
  --out outputs\ai_scientist_open_source_research\deep_dive\api_tests\round4_deep_research_summary.json
```

实时转发或回放 SSE 日志：

```powershell
python scripts/stream_deep_research.py --from-log outputs\api_tests\round4_deep_research_sse.txt
```

输出字段：

- `phases`
- `keywords`
- `private_search.totalResults`
- `private_search.failures`
- `references_count`
- `first_references`
- `answer_status`
- `answer_chars`
- `answer_preview`

进度事件脚本测试：

```powershell
python -m pytest tests/test_stream_deep_research.py
```

`answer_status` 判定：

| 状态 | 含义 |
|---|---|
| `complete` | 有 references 且收到 `done` |
| `incomplete` | 有 references 和 answer 片段，但未收到 `done` |
| `references_only` | 有 references，但 answer 未开始 |
| `clarification_or_no_evidence` | 有 answer 但无 references |
| `no_evidence` | `totalResults=0` |

## 讲解抓手

现场讲时强调：

- Deep Research 的价值不是长文本，而是把“检索、证据、报告、待核验问题”串起来。
- 可以把 SSE 阶段作为透明过程展示，让听众看到系统不是黑箱直接写答案。
- 最终答辩要讲清楚：哪些结论来自论文元数据，哪些需要全文复核。
- Deep Research 也会失败；失败时最重要的是留下关键词、检索来源、空结果和下一步换词策略。
- 宽泛问题会带来混杂 references。明天实操时先由论文检索员工筛候选，再让 Deep Research 组织证据。
- 聚焦问题能减少混杂，但不能完全替代候选过滤；标题强相关和摘要弱相关要分开看。
