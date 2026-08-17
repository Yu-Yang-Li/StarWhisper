# 输入输出策略

本 Skill 默认先把单篇论文做精。批量审查只能建立在单篇流程稳定、完整、可验证的基础上，不能为了吞吐量牺牲单篇审查质量。

## 默认使用场景

默认场景是用户提交可进行第三方解析的普通论文材料，并要求依据默认自检清单和问题汇总进行正式审查。

默认可使用第三方解析服务，例如 MinerU online VLM，但仍需遵守：

- 不在命令、报告或仓库文件中写入 token。
- 若用户明确说明论文保密、未发表、涉密、不得外传，立即切换到本地解析或先确认处理方式。
- 若上传平台、学校或机构有特别限制，以用户说明为准。

如果 MinerU online VLM 不可用或不应上传，运行 `scripts/pdf_local_fallback_extract.py`。该脚本不需要 key，先用 `pymupdf4llm` 保留 Markdown 式结构，再用 `PyMuPDF/fitz` 分页文本兜底。对本地文本抽取无法覆盖的页面或对象，记录为 blocked/residual risk，或改用更强解析方式。

## 输入类型与默认输出

| 用户上传类型 | 默认输出 | 说明 |
|---|---|---|
| PDF | PDF 审查报告，同时保留 Markdown 工作稿和工作产物 | PDF 报告用于正式交付；工作产物用于追溯。 |
| DOC / DOCX | 带批注的 DOCX，同时保留问题库、矩阵和必要的审查报告 | DOCX 批注是主交付，外部报告是辅助交付。 |
| PDF + DOCX | 先确认哪个是主审版本 | 不混审不同版本。 |
| 多篇论文 | 默认逐篇建立独立工作区 | 不共享问题库，不混用页码和证据。 |

## PDF 审查输出契约

PDF 输入默认交付：

```text
outputs/<paper_id>/
├── 审查报告.pdf
├── 审查报告.md
├── issue_database.md
├── review_matrix.csv
├── object_ledger.csv
├── method_profile.md
├── verifiable_claims.csv
├── external_factcheck.md
└── coverage_report.md
```

PDF 报告必须包含：

- 总体审核结论；
- 必须修改问题汇总；
- 逐处批注式审核意见；
- 分项审核意见；
- 外部事实核验表；
- 可验证声明、方法前提与核心结果复核表；
- 建议修改顺序；
- 复核清单；
- 覆盖与剩余风险。

## DOC / DOCX 审查输出契约

DOC / DOCX 输入默认交付：

```text
outputs/<paper_id>/
├── 批注版论文.docx
├── 批注清单.md
├── issue_database.md
├── review_matrix.csv
├── method_profile.md
├── verifiable_claims.csv
├── external_factcheck.md
└── coverage_report.md
```

批注版 DOCX 要求：

- 批注应落在原文对应位置附近。
- 每条批注使用正式审查语言。
- 批注内容应包含问题、依据和修改要求。
- 对无法直接落点的总体问题，可在批注清单和报告中呈现。
- 若 Documents 工具不可用，应明确说明无法生成批注版 DOCX，并交付可追溯批注清单作为替代，不能伪称已写入批注。

## 单篇优先原则

单篇完整审查必须具备：

1. 源文件版本登记。
2. 解析方式登记。
3. 审查依据登记。
4. 对象账本或无法建立账本的说明。
5. 方法画像。
6. 可验证声明账本。
7. 审阅矩阵。
8. 问题库。
9. 事实核验表。
10. 覆盖与剩余风险。
11. 最终交付物。
12. 验证结果。

## 未来批量原则

批量审查时：

- 每篇论文独立 `paper_id`、独立工作区、独立 issue database。
- 可并发解析和初筛。
- 可并发生成单篇审查草稿。
- 最终汇总必须单线程生成。
- 不得把多篇论文的页码、问题、证据、批注混入同一工作产物。
- 批量模式只增加调度层，不削弱单篇完成门禁。
