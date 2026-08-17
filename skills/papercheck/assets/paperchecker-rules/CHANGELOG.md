# 变更日志

本文档记录了PaperChecker项目的所有显著变化。

## [未发布]

### 新增功能
- 初始版本发布
- 引用合规性检查功能
- 支持Word和PDF文档格式
- AI驱动的引文分析
- 完整的API端点
- 新增 `citation_standard` 标准模式开关（`legacy` / `ucas`），并贯通 CLI、API、service、contract。
- 新增 UCAS P0 严格规则校验：正文著者-出版年规则与文后排序规则。
- 新增 `.doc` 输入链路（`textutil` 优先，`libreoffice/soffice` 回退），支持无损转 `.docx` 后分析。
- 新增 UCAS P1 注意事项规则集（`UCAS_NOTE_*`）：机构作者全称、作者人数截断、期刊最小字段、阿拉伯数字、非公元纪年括注、标点等校验。
- 新增 UCAS P2 分层输出：`rule_strength`（强规则/启发式）与 `confidence_tier`（high/medium/low）及分层计数统计。
- 新增作者-年份严格匹配二轮复核（`ucas` 模式）：防止“仅同年份”误匹配到错误参考文献。

### 变更
- 从TashanChecker项目分叉而来
- 完善报告文献 `[R]` 校验：新增出版项、报告编号、页码定位检查，并扩展到在线载体 `[R/OL]` 的发布时间检查。
- 文献类型校验统一按基础类型处理（如 `[R/OL]` 复用 `[R]` 基础规则），并保留电子载体 URL/访问日期校验。
- 收紧 PDF 参考文献抽取兜底策略：全文模式改为“参考文献区/文末窗口”扫描，并新增正文长句过滤规则，降低“正文被误识别为参考文献”概率。

### 修复
- 修复 AI 引文补抽误报：新增正文证据门控，拦截半截作者/机构片段（如 `ger (2019)`、`家能源局（2020）`）进入匹配链路。
- 修复 `et al` 误分类：合法 `Surname et al` 不再直接归为抽取噪声，同时保留 `Koganetal.` 这类粘连噪声识别。
- 修复参考文献粘连拆分遗漏：支持 `An X.` 后紧跟标题（无空格）场景，恢复 `An (2024)` 与文后条目映射。
- 修复 `tests/test_api.py` 对 `localhost:8000` 的硬依赖，改为进程内 ASGI 测试，避免环境端口导致的假失败。
- 修复多处历史测试函数 `return 非 None` 的写法，统一为 `assert/skip/fail`，清理 `PytestReturnNotNoneWarning`。
- 修复数字制无显式编号场景的隐式序号错位：参考文献提取新增“槽位保留”模式，避免去重/过滤导致 `[n]` 映射错位（黄国强样本 `0.0% -> 100.0%`）。
- 修复参考文献拆分误伤：补齐英文作者前导片段、中文作者前导片段、`[D]` 后来源后缀、`W H C.` 首字母作者新条目识别。
- 修复英文姓氏中 `and` 子串误拆分（如 `Brandenburger` 被切成 `Br and enburger`）。
- 修复 UCAS 期刊最小字段误报：`YYYY(issue): pages`（年份后无逗号）现按合规处理。
- 修复 `unmatched_classification` 可读性：补充 `label` 字段（`true_missing/ambiguous/extraction_noise`）。

### 测试
- 新增 `[R]` / `[R/OL]` 规则单元测试（缺失与完整样例各 2 组），覆盖错误码：
  `REF_R_MISSING_PUBLISHER_INFO`、`REF_R_MISSING_REPORT_NO`、`REF_R_MISSING_PAGE_LOCATOR`、`REF_R_OL_MISSING_PUBLISH_DATE`。
- 新增 PDF 参考文献抽取防回归测试：覆盖“正文作者-年份长句误入参考文献”的两个典型场景。
- 新增 UCAS 正文/排序规则单元测试：`tests/test_ucas_author_year_rules.py`。
- 新增 UCAS 注意事项规则单元测试：`tests/test_ucas_reference_notes_rules.py`。
- 新增 `.doc` 转换链路测试（工具缺失/主路径/回退路径）：`tests/test_doc_converter.py`。
- 新增 UCAS 合成样本 smoke：`tests/smoke_cases_manifest_ucas.json`。
- 新增 UCAS 违规样本 smoke：`tests/examples/ucas_reference_notes_smoke.docx`（含 `min_reference_format_issue_count` 门禁）。
- 新增 P2 分层门禁测试：`tests/test_smoke_quality_gate.py` 增加强规则/启发式/置信度阈值覆盖。
- 新增严格匹配场景测试：`tests/test_author_year_strict_matching.py`，覆盖“缺文献不硬匹配 + 补齐后匹配恢复”。
- 回归验证：`pytest -q`（71 passed）+ 三样本 smoke（legacy）+ UCAS smoke（含 P2 分层门禁）全通过。
- 本轮回归验证：`pytest -q`（73 passed）+ legacy/ucas smoke 全通过。
- 本轮新增防回归测试：AI 引文门控（Word/PDF）、`et al` 噪声边界、`An X.` 粘连拆分；目标回归 `51 passed`。
- 本轮全量回归：`pytest -q`（修复前 `122 passed, 1 failed`；修复后 `123 passed`）。
- 本轮 smoke：`audit_runs/2026-04-20_step24_16_smoke_legacy` 与 `audit_runs/2026-04-20_step24_16_smoke_ucas` 均 `SMOKE_ALL_PASSED=True`。
- 本轮 warnings 治理回归：`pytest -q`（`121 passed, 2 skipped`），新增 `pytest.ini` 过滤第三方噪声 warning。
- 新增 `scripts/pytest_clean.sh` 作为 CI 清洁入口，精确屏蔽启动阶段 `urllib3` OpenSSL 环境噪声。
- 新增与更新回归用例：数字制槽位保留、作者前导合并、`W H C.` 拆分、`Brandenburger` 防误拆、`YYYY(issue)` 期刊格式合规校验。
- 本轮定向回归：`pytest -q tests/test_word_reference_extraction_guards.py tests/test_reference_mapper_matching_guards.py tests/test_citation_checker_gbt_rules.py tests/test_author_year_strict_matching.py tests/test_ucas_reference_notes_rules.py` -> `41 passed`。

### 移除
- 无
