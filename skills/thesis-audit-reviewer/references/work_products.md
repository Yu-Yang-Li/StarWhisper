# 工作产物契约

## 推荐目录

```text
subprojects/thesis_audit_<date>/
├── inputs/
├── work/
│   └── <paper_id>/
│       ├── 00_source_manifest.md
│       ├── 01_page_index.md
│       ├── 02_structure_map.md
│       ├── 03_method_profile.md
│       ├── 04_object_ledger.csv
│       ├── 05_verifiable_claims.csv
│       ├── 06_review_matrix.csv
│       ├── 07_issue_database.md
│       ├── 08_external_factcheck.md
│       ├── 09_reference_audit.csv
│       ├── 10_coverage_report.md
│       └── 11_final_review_report.md
├── outputs/
│   └── <paper_id>/
│       ├── 审查报告.md
│       ├── 审查报告.pdf
│       ├── 批注版论文.docx
│       └── 批注清单.md
└── logs/
```

Use `scripts/init_audit_workspace.py` to create this scaffold.

## 00_source_manifest.md

Required content:

- paper_id
- title
- source file absolute path
- file size
- SHA256
- page count if known
- parse method
- review basis: default_review_sources.md or custom checklist/problem summary paths
- input type and output mode
- checklist/reference documents
- audit date
- residual input risks

For DOC/DOCX inputs, attach or link the output of `scripts/docx_integrity_scan.py`.

## 03_method_profile.md

The method profile identifies which specialized audit obligations should trigger. It is not a conclusion; it is a routing layer.

| field | required | meaning |
|---|---:|---|
| method_id | yes | stable id, e.g. METHOD-001 |
| method_name | yes | method named or implied by the paper |
| location | yes | chapter/page/section |
| role | yes | background/auxiliary/core_evidence/core_conclusion_source |
| input_data | yes | data, sample, material, or corpus used |
| premises | yes | dates, windows, sample units, variables, assumptions |
| outputs | yes | computed or inferred results |
| supports_main_claim | yes | yes/no |
| public_or_internal_recheck | yes | public/internal/author_source/blocked |
| required_actions | yes | internal_crosscheck/external_factcheck/method_premise_check/minimal_recalculation/author_source_required |
| status | yes | pending/pass/issue/blocked |

## 04_object_ledger.csv

Object ledger is the audit denominator.

| field | required | meaning |
|---|---:|---|
| paper_id | yes | paper identifier |
| object_ledger_id | yes | stable object id |
| object_type | yes | page/section/paragraph/table/figure/formula/reference/concept/method/verifiable_claim/data_fact/strong_claim/policy_suggestion |
| pdf_page | yes | PDF physical page |
| thesis_page | no | printed page number if visible |
| page_region | no | top/middle/bottom, table row, coordinates, etc. |
| section | yes | section or document area |
| object_id | no | figure/table/formula/reference number |
| object_title | no | caption or title |
| original_excerpt | yes | short excerpt or visual description |
| extraction_confidence | yes | high/medium/low |
| needs_visual_check | yes | yes/no |
| notes | no | remarks |

Suggested id prefixes: `PAGE-*`, `SEC-*`, `PAR-*`, `TAB-*`, `FIG-*`, `FOR-*`, `REF-*`, `METHOD-*`, `CON-*`, `FACT-*`, `CLAIM-*`, `POL-*`.

## 05_verifiable_claims.csv

Use for every candidate number, amount, percentage, date, sample size, method window, cross reference, data-source claim, and author-computed result.

| field | required | meaning |
|---|---:|---|
| claim_id | yes | stable id, e.g. CLAIM-001 |
| paper_id | yes | paper identifier |
| location | yes | page/section/paragraph/table |
| claim_type | yes | number/money/percent/date/sample/window/cross_reference/data_source/computed_result/strong_claim |
| claim_text | yes | original claim or necessary excerpt |
| value | no | extracted value |
| unit | no | unit or target object |
| subject | no | entity or topic |
| method_id | no | linked method if relevant |
| verification_action | yes | internal_crosscheck/external_factcheck/method_premise_check/minimal_recalculation/author_source_required/blocked |
| status | yes | pending/pass/issue/needs_factcheck/needs_internal_crosscheck/needs_method_premise_check/needs_recalculation/needs_author_source/blocked |
| verification_notes | no | source, recalculation note, or reason |
| issue_id | if issue | linked issue |

## 06_review_matrix.csv

The review matrix proves coverage.

| field | required | meaning |
|---|---:|---|
| matrix_id | yes | stable row id |
| source_doc | yes | self-check list/problem summary/school norm/general norm |
| source_rule_id | yes | rule id or local generated id |
| source_requirement | yes | requirement text |
| object_ledger_id | yes | object or range checked |
| pdf_page | yes | page or page range |
| section | yes | chapter/document area |
| check_item | yes | what was checked |
| status | yes | pass/issue/needs_factcheck/needs_internal_crosscheck/needs_method_premise_check/needs_recalculation/needs_author_source/not_applicable/blocked |
| evidence_excerpt | yes unless not_applicable | short original evidence |
| issue_id | if issue | link to issue database |
| notes | no | audit note |

`pass` means the required verification action was completed. If a claim was only read or registered, use a `needs_*` status instead.

## 07_issue_database.md

Each formal issue must use this shape:

```text
### ISSUE-000: title

- severity: 严重 / 重要 / 一般 / 建议
- priority: 必须修改 / 建议修改 / 格式修正
- certainty: 确认问题 / 疑似问题 / 需作者补充材料 / 无法判断
- matrix_id:
- object_ledger_id:
- claim_id:
- method_id:
- location:
- original_excerpt:
- problem:
- basis:
- modification_requirement:
- factcheck_status:
```

Do not promote leads without `location`, `original_excerpt`, and `basis` into the formal issue database.

## 08_external_factcheck.md

Use for externally checkable claims and author-computed results that were externally or publicly rechecked.

| fact_id | claim_id | method_id | location | paper_claim | fact_type | verification_status | source | source_date | checked_date | result | required_action |
|---|---|---|---|---|---|---|---|---|---|---|---|

Fact types: timeline, number, product, policy, company, standard, market, literature, data_source, computed_result, method_premise, internal_consistency.

## 09_reference_audit.csv

Use for references and citation consistency.

| ref_id | pdf_page | original_entry | type_mark | cited_in_text | function | author | year | title | source | volume_issue | pages | doi_url | access_date | coverage_years | matches_method_or_data_source | issue |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

Reference function values: theory, method, fact, data, background, decoration.

## 10_coverage_report.md

The coverage report must answer:

- How many pages/sections/tables/figures/formulas/references were identified?
- How many methods and verifiable claims were identified?
- Which checklist categories were checked?
- Which rows are `pass`, `issue`, `needs_factcheck`, `needs_internal_crosscheck`, `needs_method_premise_check`, `needs_recalculation`, `needs_author_source`, `not_applicable`, and `blocked`?
- Which high-risk items were checked against the PDF image?
- Which core results were minimally recalculated, internally cross-checked, or marked author-source-required?
- What remains uncertain and why?

## Final Report Traceability

Every final report finding must include at least one of:

- `matrix_id`
- `issue_id`
- `object_ledger_id`
- `fact_id`
- `ref_id`
- `claim_id`
- `method_id`

If the final report cannot be traced back to a work product, remove the finding or add the missing work record first.
