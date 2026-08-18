---
name: giiisp-paper-search-apis
description: Search astronomy papers via NASA ADS and arXiv astro-ph, with optional Giiisp OA as a supplement. Use for ADS, bibcode, ApJ/MNRAS/A&A papers, transients, surveys, or StarWhisper literature. Dry-run without ADS_API_TOKEN; never invent papers.
---

# Giiisp Paper Search APIs

## StarWhisper astronomy overlay

This copy is adapted for astronomy research and telescope-agent work.
**Read [`astronomy.md`](astronomy.md) before following generic biomedical / clinical defaults in the rest of this file.**

Default literature route: NASA ADS, then arXiv `astro-ph.*`, then the original skill's search backend if credentials exist.
Do not claim a real hardware observing loop, a discovery, or a referee-ready result unless the user supplied that evidence.

天文检索先跑这个脚本，不要一上来调 Giiisp：

```powershell
python skills/giiisp-paper-search-apis/scripts/ads_first_search.py --query "StarWhisper Telescope NGSS" --dry-run
```

有 `ADS_API_TOKEN` 时去掉 `--dry-run` 会查 NASA ADS；没有 token 则查 arXiv `astro-ph`。空结果就写空，不要补一篇“看起来像”的论文。Giiisp 只作补充。

## 接口依据

本 skill 的接口清单来自稳定目录中的 `reference_giiisp_search.md`。原文件由用户提供，已从临时路径复制到稳定工作目录；后续自动化和测试只引用稳定副本。

## 先判断任务

天文问题先走 `scripts/ads_first_search.py`。下表只在 ADS/arXiv 不够、需要 Giiisp OA 补充时使用。

| 用户要做什么 | 首选接口 |
|---|---|
| 天文论文、bibcode、ApJ/MNRAS | `scripts/ads_first_search.py`（ADS → arXiv `astro-ph`） |
| 按主题查开放论文 | `/first/oaPaper/searchArticlesByQuery1` |
| 按方法描述查 arXiv | `/first/paper/searchArxivByAbstract` |
| 已有 arXiv 编号 | `/first/paper/searchArxivByArxivNo1` |
| 同时给题名、作者、摘要词 | `/first/paper/searchArxiv` |
| 查重或核对题名 | `/first/paper/searchArxivByTitle` |

基址为 `https://giiisp.com`。所有接口使用 `POST` 和 JSON body，不把业务参数放在 URL query。

## 调用前检查

1. 明确用户要查的主题、时间范围、领域和语言。
2. 把用户原话整理成 1-3 组检索词，不要替用户编论文名。
3. 如果接口需要登录态或鉴权，但当前没有凭据，只构造请求体和 curl 示例，不真实调用；提醒用户到 `https://giiisp.com/#/mcp/authenticate` 申请或刷新 Giiisp MCP 认证，并设置 `GIIISP_AUTH_TOKEN`。
4. 如果返回 HTML 登录页、非 JSON、401 或 403，先报告鉴权问题和认证页，再给开放源回退方案。

## 请求体规则

| 接口 | 必填字段 | 说明 |
|---|---|---|
| `/first/oaPaper/searchArticlesByQuery1` | `titleAndAbs` | 数组，放标题/摘要关键词 |
| `/first/paper/searchArxivByAbstract` | `key`, `pageNum`, `pageSize` | 用摘要语义或方法词检索 |
| `/first/paper/searchArxivByArxivNo1` | `key`, `pageNum`, `pageSize` | `key` 放 arXiv 编号 |
| `/first/paper/searchArxiv` | `key`, `pageNum`, `pageSize` | 默认 arXiv 多字段入口 |
| `/first/paper/searchArxivByTitle` | `key`, `pageNum`, `pageSize` | 题名定位 |

分页默认从 `pageNum=1`、`pageSize=10` 开始。只在用户需要扩展结果时继续翻页。

## 进度事件输出

普通论文搜索接口是同步 JSON 接口，不要假装它能逐条流式返回。响应时间长时，由调用层把“扩展检索计划”拆成多个可观察步骤，并在每一步发进度事件。

推荐使用 `scripts/progressive_paper_search.py` 生成 JSONL 事件，一行一个事件，可直接转成 SSE：

```powershell
python scripts/progressive_paper_search.py `
  --query "large language model scientific discovery" `
  --mode arxiv-title `
  --expand arxiv-abstract,oa `
  --max-pages 2 `
  --dry-run
```

事件顺序：

| 事件 | 含义 |
|---|---|
| `search_started` | 搜索开始，包含主检索模式、扩展模式和计划请求数 |
| `auth_status` | 报告是否检测到 `GIIISP_AUTH_TOKEN`；无 token 时给出 `https://giiisp.com/#/mcp/authenticate` 和用户动作 |
| `request_prepared` | 某个接口/页码的请求体已生成，可展示“正在查 title / abstract / OA” |
| `response_received` | 真实调用模式下某个请求已返回，包含 HTTP 状态、content type 和命中数估计 |
| `search_complete` | 所有计划请求已完成或 dry-run 计划已输出 |

如果要真实调用，去掉 `--dry-run`。真实调用仍是“一次接口一次响应”，但调用层可以在每个请求前后展示进度，避免用户长时间看不到任何反馈。
如果 `response_received.response` 带 `auth_url` 或 `user_action`，前端应直接提示用户去认证页刷新 token、收窄检索词或稍后重试，而不是让用户等待一个无反馈的长请求。

## 输出格式

每次检索输出一张短表：

| 字段 | 要求 |
|---|---|
| 检索目标 | 用户原始问题的简写 |
| 实际检索词 | 写出真实传入的词 |
| 接口 | 写完整路径 |
| 论文 | 题名、作者、年份、来源 |
| 链接 | arXiv、DOI、开放论文页或来源页 |
| 摘要依据 | 只摘核心命中点 |
| 状态 | 已核验 / 待核验 / 接口受限 |
| 下一步 | 翻页、换词、交叉核验或停止 |

不要编造引用量、期刊分区、全文内容或不存在的 DOI。

最小端到端示例见 `examples/end_to_end_example.json`：它展示“用户问题 -> 路由 -> dry-run 请求 -> 输出表”的完整结构。该示例只用于格式对照，不代表真实命中，也不会调用需要鉴权的接口。

## 失败响应处理

只要响应不能被确认为可用的 Giiisp JSON 结果，就不要把它当作论文命中。先说明接口状态，再给开放源回退结果或回退计划。

| 响应现象 | 判断 | 处理 |
|---|---|---|
| HTTP `401` / `403` | 登录态、权限或鉴权缺失 | 标记 `接口受限`，不要重试刷接口；改用开放源回退 |
| HTTP `429` | 频率限制 | 标记 `接口受限`，说明需要降频或稍后再试；不要并发补打 |
| `Content-Type` 是 `text/html` | 可能返回登录页、验证码页或错误页 | 不解析为论文；摘录页面标题或前 120 字作为失败摘要 |
| JSON 里有 `code`、`success`、`message`，但无结果数组 | 业务失败或空结果 | 保留原始错误字段，状态写 `接口受限` 或 `待核验` |
| 网络超时 / DNS / TLS 错误 | 当前环境无法访问 | 标记 `接口受限`，输出开放源回退 |

失败响应记录示例：

```json
{
  "query": "large language model scientific discovery",
  "source_api": "/first/paper/searchArxivByTitle",
  "request_body": {
    "key": "large language model scientific discovery",
    "pageNum": 1,
    "pageSize": 10
  },
  "failure": {
    "http_status": 403,
    "content_type": "application/json",
    "message": "Authentication required or session expired.",
    "raw_excerpt": "{\"code\":403,\"message\":\"login required\"}"
  },
  "verification_status": "接口受限",
  "next_step": "改用 arXiv、OpenAlex、Semantic Scholar、Crossref 等开放源交叉检索。"
}
```

## 返回字段归一化

接口返回字段可能随入口不同而变化。整理结果时先保留原始命中，再归一化成统一字段；缺失值用 `null` 或 `待核验`，不要用推测值补齐。

| 归一化字段 | 来源字段候选 | 要求 |
|---|---|---|
| `title` | `title`, `paperTitle`, `articleTitle`, `name` | 保留原题名，去掉首尾空白 |
| `authors` | `authors`, `author`, `authorList`, `creator` | 输出作者名数组；只有字符串时按接口原分隔符谨慎拆分 |
| `year` | `year`, `publishYear`, `published`, `created` | 只填可直接解析的四位年份 |
| `venue` | `venue`, `journal`, `source`, `publication` | 期刊、会议或来源站点 |
| `abstract` | `abstract`, `summary`, `abs`, `description` | 保留摘要文本；引用时只摘核心依据 |
| `doi` | `doi`, `DOI` | 统一小写前缀之外的 DOI 字符串，不编造 |
| `arxiv_id` | `arxivNo`, `arxivId`, `eprint`, `id` | 只填明确 arXiv 编号 |
| `url` | `url`, `link`, `paperUrl`, `htmlUrl` | 优先开放论文页，其次来源页 |
| `pdf_url` | `pdfUrl`, `pdf`, `fullTextUrl` | 只填明确 PDF 链接 |
| `source_api` | 当前接口路径 | 记录命中的 Giiisp 接口路径 |
| `match_reason` | 检索词 + 摘要/标题命中 | 用一句话说明为什么纳入候选 |
| `verification_status` | 人工核验状态 | `已核验` / `待核验` / `接口受限` / `非 Giiisp 结果` |

推荐内部结构：

```json
{
  "query": "large language model scientific discovery",
  "source_api": "/first/paper/searchArxivByTitle",
  "normalized_results": [
    {
      "title": "Example Paper Title",
      "authors": ["First Author", "Second Author"],
      "year": 2025,
      "venue": "arXiv",
      "abstract": "Short abstract text from the source response.",
      "doi": null,
      "arxiv_id": "2501.01234",
      "url": "https://arxiv.org/abs/2501.01234",
      "pdf_url": "https://arxiv.org/pdf/2501.01234",
      "match_reason": "Title contains the requested method phrase.",
      "verification_status": "待核验"
    }
  ]
}
```

## 引用审计输出模板

用于检查报告、PPT、论文草稿中的引用是否可追溯。不要只给“看起来相关”的结论；必须把原文主张、候选论文和核验动作放在同一行。

| 原文主张 | 引用/占位符 | 检索词 | 候选论文 | 证据字段 | 链接 | 状态 | 处理意见 |
|---|---|---|---|---|---|---|---|
| 文中需要支撑的具体句子 | `[?]`、DOI、arXiv 或作者年份 | 实际传入接口的词 | 题名、作者、年份、来源 | 标题/摘要/DOI/编号命中点 | DOI、arXiv 或来源页 | 已核验 / 待核验 / 不支持 / 接口受限 | 保留、替换、补充引用、删除主张或改写 |

审计结论按这四类收束：

- `已核验`：题名、作者/年份和 DOI/arXiv/来源页能互相对上，摘要或全文元数据支持原文主张。
- `待核验`：元数据看起来匹配，但缺少 DOI、全文页、作者年份或关键摘要字段。
- `不支持`：候选论文与原文主张不一致，或只能支持更弱的说法。
- `接口受限`：Giiisp 登录态、非 JSON、401/403 或网络限制导致无法核验；给开放源回退建议。

## 回退

Giiisp 接口不可用时，按任务选择开放源：

- arXiv：预印本定位和编号核验。
- OpenAlex：开放论文元数据、机构和引用关系。
- Semantic Scholar：主题扩展和相似论文。
- Crossref：DOI 与出版元数据。
- PubMed / Europe PMC：医学与生命科学。

回退结果必须标明“非 Giiisp 结果”。

开放源回退输出格式：

```json
{
  "query": "large language model scientific discovery",
  "giiisp_status": "接口受限",
  "fallback_reason": "Giiisp returned 403 or a non-JSON login page.",
  "fallback_sources": ["arXiv", "OpenAlex", "Semantic Scholar", "Crossref"],
  "normalized_results": [
    {
      "title": "Example Open Source Paper Title",
      "authors": ["First Author", "Second Author"],
      "year": 2025,
      "venue": "arXiv",
      "abstract": "Short abstract text from an open source metadata provider.",
      "doi": null,
      "arxiv_id": "2501.01234",
      "url": "https://arxiv.org/abs/2501.01234",
      "pdf_url": "https://arxiv.org/pdf/2501.01234",
      "source_api": "arXiv API or arXiv web page",
      "match_reason": "Open source title or abstract contains the requested phrase.",
      "verification_status": "非 Giiisp 结果"
    }
  ],
  "next_step": "用 DOI、arXiv 编号或出版页做二次核验后再引用。"
}
```

## 验证

优先做 dry-run：

```powershell
python scripts/dry_run_paper_search.py --mode arxiv-title --query "large language model scientific discovery"
```

dry-run 只输出请求 URL、headers 摘要和 JSON body，不发起真实请求。

模拟归一化结果示例：

```powershell
python scripts/dry_run_paper_search.py --mode arxiv-title --query "large language model scientific discovery" --format normalized-example
```

模拟 Giiisp 不可用时的开放源回退结果示例：

```powershell
python scripts/dry_run_paper_search.py --mode arxiv-title --query "large language model scientific discovery" --format fallback-example
```

最小端到端示例：

```powershell
python scripts/dry_run_paper_search.py --mode arxiv-title --query "large language model scientific discovery" --format end-to-end-example
```

进度事件脚本测试：

```powershell
python -m pytest tests/test_progressive_paper_search.py
```
