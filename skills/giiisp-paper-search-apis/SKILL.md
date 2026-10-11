---
name: giiisp-paper-search-apis
description: "检索天文论文或核对题名、作者和论文编号；优先 NASA ADS 与 arXiv astro-ph，按需使用 Giiisp 补充，保留实际来源和核验状态。"
---

# 天文文献检索

按主题查论文，或核对已有题名、作者、DOI、bibcode 和 arXiv 编号。先查 NASA ADS，再查 arXiv `astro-ph`；需要补充开放论文时使用 Giiisp。

## 怎么开始

先明确研究问题、时间范围和已有论文。天文来源与编号约定见 [astronomy.md](astronomy.md)。以下命令在本技能目录中运行：

```bash
python scripts/ads_first_search.py --query "StarWhisper Telescope NGSS" --dry-run
```

`--dry-run` 只查看请求计划。有 `ADS_API_TOKEN` 时，去掉它会查 ADS；没有 token 则回退到 arXiv `astro-ph`。记录实际使用的来源，不把回退结果写成 ADS 或 Giiisp 的结果。

## 检索与核验

1. 把研究问题整理为 1–3 组检索词，保留实际查询内容。
2. 先核对元数据：题名、作者、年份、论文编号和链接。
3. 需要解释结论时，再阅读摘要或全文，并区分两种证据。
4. 去重后记录纳入理由；空结果如实报告，不补造论文。

题名相似不等于引用已核验，摘要也不能代替全文中的方法与结果。密钥只从环境读取，不写进日志、请求示例或报告。

## 需要 Giiisp 时

读取 [接口与结果处理](references/api-and-results.md)，其中保留请求字段、分页、鉴权、进度事件和结果归一化格式。请求示例见 [examples/](examples/)，本地检查方法见 [ACCEPTANCE.md](ACCEPTANCE.md)。

没有 `GIIISP_AUTH_TOKEN` 时只准备请求；401、403、HTML 登录页或非 JSON 响应均需说明实际失败原因，再选择开放来源。不要把示例和 dry-run 当成真实检索结果。

## 结果怎么交代

返回一张候选表，包含论文题名、作者、年份、来源、链接、相关理由和核验状态。另说明检索词、范围、空结果或服务限制，以及哪些论述仍需全文核对。
