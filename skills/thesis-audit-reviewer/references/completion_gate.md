# 完成门禁

Use this gate before claiming that a formal thesis audit is complete.

## Hard Gates

Do not claim completion if any hard gate fails.

1. **Source Gate**
   - Source file path and version are recorded.
   - Page count or parsing scope is recorded.
   - Checklist/reference documents are recorded when provided.
   - If no custom checklist/problem summary is provided, use of `default_review_sources.md` is recorded.
   - Input/output mode is recorded: PDF -> PDF report, DOC/DOCX -> commented DOCX.

2. **Parsing Gate**
   - PDF/DOCX parsing method is recorded.
   - Low-quality extraction, missing pages, garbled formula/table regions, and scanned pages are listed.
   - High-risk formulas, tables, references, and numbering issues were checked against the PDF image or marked blocked.
   - For DOC/DOCX inputs, `scripts/docx_integrity_scan.py` has been run or an equivalent inspection is recorded.
   - If OOXML Math, images, text boxes, comments, revisions, footnotes, or field codes exist, the report states how they were made visible or why they remain blocked.

3. **Coverage Gate**
   - Object ledger exists or the report explains why it could not be built.
   - Method profile exists or the report explains why no substantive method is present.
   - Verifiable claim ledger exists or the report explains why it could not be built.
   - Review matrix has no blank status.
   - `pass` rows identify the checked scope.
   - `pass` rows for facts, numbers, dates, methods, formulas, tables, and references identify the completed verification action.
   - `blocked` rows have residual-risk explanations.

4. **Method and Fact Gate**
   - All externally checkable current/company/policy/standard/product facts have sources or residual status.
   - All repeated numbers, dates, amounts, percentages, and sample counts were internally cross-checked or marked unresolved.
   - All method-dependent premises were checked or marked unresolved.
   - All author-computed core results were assigned a feasible verification level: internal consistency, formula substitution, public-data minimal recalculation, author-source-required, or blocked.
   - `needs_author_source` is used only after internal crosscheck, public-source lookup, and minimal-recalculation feasibility have been considered.

5. **Evidence Gate**
   - Every formal finding has location, excerpt, basis, modification requirement, priority, and certainty.
   - No report finding appears only in prose without issue/matrix/fact/reference trace.
   - Unsupported leads are not presented as confirmed problems.

6. **Factcheck Gate**
   - User-requested external verification has been performed.
   - Current/company/policy/standard/product facts have source links or are marked `needs_author_source`.
   - Official or primary sources are preferred where available.

7. **Report Gate**
   - Report uses the standard template.
   - Report has a must-fix summary and detailed comment section.
   - Report has a verifiable-claim, method-premise, and external fact verification table.
   - Report has a coverage and residual-risk section.
   - Report language is formal audit/comment language, not process narration.
   - For PDF input, the final PDF report path is recorded.
   - For DOC/DOCX input, the commented DOCX path is recorded, or the inability to create DOCX comments is explicitly stated with a traceable substitute.

8. **Validation Gate**
   - Run `scripts/validate_audit_report.py --report <report.md> --strict`.
   - Fix failures, or state exactly which validation failures remain and why they are acceptable/blocked.

9. **Documentation Gate**
   - If working inside a project, update existing progress/findings docs with output paths, verification performed, and unresolved risks.

10. **Single-Paper Gate Before Batch Gate**
   - For batch work, every paper has its own source manifest, matrix, issue database, coverage report, and final output.
   - Batch summary is not a substitute for per-paper completion.

## Soft Gates

Soft gates should be satisfied for high-stakes delivery, but may be listed as residual risk when time or source quality prevents completion.

- References have field-level audit.
- All formulas are indexed.
- All tables/figures are indexed.
- All strong claims are in an argument-chain record.
- All verifiable claims are in a claim ledger.
- Core computed results have at least Level 0 internal consistency checks.
- All policy suggestions are reverse-checked against prior chapters.
- Report is rendered to PDF when user asked for a PDF deliverable.

## Final Reply Checklist

The final reply should state:

- output report path(s)
- whether validation passed
- what was verified
- what could not be verified
- any remaining risk that the author or user must handle

Do not overstate completeness. Use exact wording such as:

- "已完成并通过报告结构验证。"
- "已完成审阅报告，但仍有 X 项因原文/数据缺失标记为需作者补充材料。"
- "未能完成交付门禁，因为……"
