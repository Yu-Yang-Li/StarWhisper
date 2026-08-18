---
name: thesis-audit-reviewer
description: Audit astronomy theses and papers for ADS-valid citations, coordinate/time/photometric-system consistency, survey selection effects, and over-claimed discoveries. Use for checklist-based, evidence-grounded comments on PDF/DOCX manuscripts.
---

# Thesis Audit Reviewer

## StarWhisper astronomy overlay

This copy is adapted for astronomy research and telescope-agent work.
**Read [`astronomy.md`](astronomy.md) before following generic biomedical / clinical defaults in the rest of this file.**

Default literature route: NASA ADS, then arXiv `astro-ph.*`, then the original skill's search backend if credentials exist.
Do not claim a real hardware observing loop, a discovery, or a referee-ready result unless the user supplied that evidence.


## Purpose

Produce standardized, evidence-grounded thesis audit reports through a repeatable audit workflow. This skill is an audit operating system, not a one-shot prompt for writing comments.

Use this skill for Chinese or bilingual thesis review tasks involving PDF/DOCX papers, self-check lists, problem summaries, external facts, formulas, tables, references, or supervisor-style comments.

Default review basis: use `references/default_review_sources.md`, derived from the user's `学位论文自检清单.docx` and `学位论文问题汇总.docx`, unless the user provides a newer checklist or explicitly asks for a different standard.

Default delivery mode: PDF input produces a PDF audit report; DOC/DOCX input produces a commented DOCX. Preserve Markdown, issue database, review matrix, factcheck, and coverage artifacts for traceability.

## Non-Negotiable Standard

Every finding needs:

- precise location: PDF physical page, printed page/chapter if available, object type if available
- original excerpt: short, necessary quote only
- audit comment: what is wrong, in reviewer language
- evidence basis: checklist, paper context, external source, PDF image, or academic norm
- modification requirement: concrete author action
- priority: 必须修改 / 建议修改 / 格式修正
- certainty: 确认问题 / 疑似问题 / 需作者补充材料 / 无法判断

Never deliver only a summary when the user asks for a formal audit. Write in 批注和审核语言, for example: "该处……" "应补充……" "建议作者……" "不宜……".

If a conclusion lacks location, excerpt, and basis, keep it in the work notes as a lead; do not put it in the formal report.

## Operating Model

Adopt the mature-skill pattern from prior document-audit work:

- Treat the report as a product generated from work artifacts, not as the audit itself.
- Build the audit denominator first: pages, sections, figures, tables, formulas, references, data facts, strong claims, and policy suggestions.
- Use deterministic tools for parsing, splitting, scaffolding, rendering, and report validation.
- Use AI judgment for academic reasoning, evidence chain evaluation, overclaim detection, and reviewer-language drafting.
- Record pass/blocked/unknown states, not only discovered issues.
- Add a fact-audit layer before normative review: identify methods, verifiable claims, method premises, internally repeated numbers, and author-computed core results.
- Finish only after the completion gate passes or after residual risks are explicitly listed.

## Workflow

1. **Confirm Scope**
   - Identify source paper(s), checklist files, reference problem files, and requested output format.
   - If no alternative checklist/problem summary is supplied, use `references/default_review_sources.md` as the audit basis.
   - Determine input/output mode using `references/input_output_policy.md`: PDF -> PDF report; DOC/DOCX -> commented DOCX.
   - Prioritize one paper deeply. For multiple papers, create independent per-paper workspaces and do not weaken single-paper gates.
   - If the user asks to use external fact verification, browse and cite sources.
   - If the user asks for a Skill/process discussion, do not run the full audit.
   - If the user wants DOCX comments or redlines, use the Documents skill and keep this skill as the audit logic.

2. **Create Project Work Area**
   - Use a subfolder under the current project, e.g. `subprojects/thesis_audit_<date>/`.
   - Keep intermediate artifacts: source manifest, parsed pages, issue database, final reports.
   - Do not store API tokens or private credentials in files.
   - Prefer `scripts/init_audit_workspace.py` to create the standard directory and empty ledgers.

3. **Parse Paper**
   - Default scenario is ordinary thesis material that may be parsed by third-party tools; for PDF, prefer MinerU online VLM parsing when available.
   - If the user marks the paper confidential, unpublished, restricted, or not uploadable, do not use online parsing; switch to local parsing or ask for permission.
   - Use `scripts/mineru_vlm_extract.py` to upload the full PDF and download results when MinerU is available and upload is allowed.
   - If MinerU is unavailable, unconfigured, timed out, or upload is not allowed, run `scripts/pdf_local_fallback_extract.py` first. It tries local `pymupdf4llm` Markdown extraction, then local `PyMuPDF/fitz` page text extraction.
   - Use `scripts/split_mineru_vlm_pages.py` to create page-level object files after MinerU parsing.
   - Always verify high-risk findings against the original PDF image or rendered page; parsed text is evidence support, not final truth.
   - For DOCX, use the Documents skill or direct OOXML inspection when comments/redlines are requested.
   - For DOC/DOCX, run `scripts/docx_integrity_scan.py` and read `references/docx_integrity_gate.md`; do not rely on `python-docx paragraph.text` or long command-output previews as full-text coverage.

4. **Build Audit Denominator**
   - Create source manifest, page index, structure map, object ledger, fact ledger, and reference ledger before drafting findings.
   - Index every page, section, table, figure, formula, reference entry, data source, method, verifiable claim, strong conclusion, and policy recommendation that is visible enough to audit.
   - Mark low-quality extraction, ambiguous page objects, and missing pages as `blocked`; do not silently skip them.

5. **Build Method and Claim Ledgers**
   - Read `references/method_and_fact_audit.md`.
   - Identify every substantive research method and create a method profile: role, data, assumptions, parameters, core outputs, and whether results support main conclusions.
   - Run `scripts/scan_verifiable_claims.py` on extracted text where available, then let AI clean and complete the ledger.
   - Register all numbers, percentages, money amounts, dates, sample sizes, method windows, cross references, data-source claims, and author-computed core results.
   - Decide each claim's verification action: internal crosscheck, external factcheck, method premise check, minimal recalculation, author source required, or blocked.

6. **Build Review Coverage**
   - Read checklist/reference docs and map them to review categories.
   - Use the taxonomy in `references/issue_taxonomy.md`.
   - For each checked area, record either findings or checked-no-issue notes in the work files when coverage matters.
   - Use statuses: `pass`, `issue`, `needs_factcheck`, `needs_internal_crosscheck`, `needs_method_premise_check`, `needs_recalculation`, `needs_author_source`, `not_applicable`, `blocked`.
   - Do not use `pass` unless the required verification action for that object has actually been completed.

7. **Audit High-Risk Areas First**
   - Abstract and English abstract.
   - Table of contents, figure/table lists, numbering, cross references.
   - Research question, title concept alignment, chapter progression.
   - Literature review coverage and critique.
   - Research design, interviews, samples, coding, variables, formulas, models.
   - Method premises, data sources, calculation consistency, reproducibility, and author-computed core results.
   - Conclusions, policy suggestions, overclaims.
   - References, duplicate entries, missing fields, source suitability.

8. **Verify Facts and Reproducibility**
   - Browse when facts may be current, contested, or externally checkable.
   - Prefer primary/official sources for company facts, policy facts, dates, products, and standards.
   - Separate “verified”, “partly verified”, and “not verified” in the report.
   - Record facts as you read. Do not reconstruct the fact list from memory at the end.
   - For author-computed core results, perform the strongest feasible check: internal table consistency, formula substitution, public-data minimal recalculation, or author-source-required.
   - Do not mark a result as `needs_author_source` until internal crosscheck, public-source lookup, and minimal recalculation feasibility have been considered.

9. **Build Issue Database**
   - Promote only evidence-backed findings into the issue database.
   - Deduplicate, classify, and assign severity before writing the final report.
   - Keep unresolved leads, blocked evidence, and author-source-needed items separate from confirmed problems.

10. **Write the Report**
   - Use `references/standard_report_template.md`.
   - Use `references/comment_language.md` for tone and wording.
   - Generate Markdown first. If requested or useful, use `scripts/render_md_report_pdf.py` to create a PDF copy.

11. **Run Completion Gate**
   - Read `references/completion_gate.md`.
   - Run `scripts/validate_audit_report.py` on the Markdown report before claiming completion.
   - If validation fails, fix the report or explicitly state which gate is blocked and why.

12. **Document the Work**
   - If inside a project, update existing progress/findings/docs with factual notes: what was parsed, what was verified, output paths, pitfalls.
   - Do not create unrelated README or personal notes inside the Skill.

## When to Load References

- Load `references/audit_operating_protocol.md` before planning or executing a full audit.
- Load `references/default_review_sources.md` before building the review matrix unless the user supplied a replacement checklist.
- Load `references/input_output_policy.md` before choosing the final deliverable.
- Load `references/docx_integrity_gate.md` for DOC/DOCX inputs or when formulas/images may be lost.
- Load `references/method_and_fact_audit.md` before building method profiles, claim ledgers, fact checks, or reproducibility checks.
- Load `references/work_products.md` when creating workspace files, ledgers, matrices, or issue databases.
- Load `references/completion_gate.md` before final delivery.
- Load `references/standard_report_template.md` before writing the final report.
- Load `references/comment_language.md` when drafting comments.
- Load `references/issue_taxonomy.md` when planning coverage or classifying findings.
- Load `references/evidence_rules.md` when deciding whether a statement is sufficiently supported.

## Script Quick Use

```bash
python scripts/doctor.py
python scripts/init_audit_workspace.py --out subprojects/thesis_audit_YYYY_MM_DD --paper-id paper01 --title "论文题名" --source-file paper.pdf
# DOCX input example:
# python scripts/init_audit_workspace.py --out subprojects/thesis_audit_YYYY_MM_DD --paper-id paper01 --title "论文题名" --source-file paper.docx
python scripts/docx_integrity_scan.py --docx paper.docx --markdown-output work/paper01/docx_integrity.md --json-output work/paper01/docx_integrity.json
python scripts/scan_verifiable_claims.py --input work/paper01/extracted_text.md --output work/paper01/05_verifiable_claims.csv
python scripts/mineru_vlm_extract.py --file paper.pdf --out work/mineru_vlm --model-version vlm
python scripts/split_mineru_vlm_pages.py --content-list work/mineru_vlm/<extract_dir>/content_list_v2.json --out work/vlm_pages
python scripts/pdf_local_fallback_extract.py --file paper.pdf --out work/pdf_fallback
python scripts/render_md_report_pdf.py --input outputs/paper01/审查报告.md --output outputs/paper01/审查报告.pdf
python scripts/validate_audit_report.py --report outputs/paper01/审查报告.md --strict
```

The MinerU script reads the token from `MINERU_API_TOKEN`, hidden TTY input, or stdin. If the user has no token, point them to `https://mineru.net/apiManage/token` to register and create one. Never write the token into a command, report, or repository file. The local PDF fallback script does not need a token; treat its output as lower confidence than MinerU for scanned PDFs, formulas, complex tables, multi-column layout, headers/footers, and split reference lists.

## Completion Definition

For a formal thesis audit, completion requires:

- source and parsing status recorded
- default or user-supplied review basis recorded
- input/output mode recorded: PDF report or commented DOCX
- DOCX integrity gate completed when input is DOC/DOCX, especially formula/image visibility
- object denominator created or residual gaps listed
- method profile and verifiable claim ledger created or explicitly blocked
- checklist coverage recorded, including pass and blocked rows
- all formal findings traceable to issue database entries
- external facts, internal numeric consistency, method premises, and feasible recalculations checked or marked with the correct residual status
- high-risk formulas, tables, references, and numbering checked against PDF image when needed
- final report follows the standard template and passes validation
- project progress/findings documentation updated when working inside a project

For batch work, completion means every single paper satisfies the single-paper gate first, then the batch summary is generated.
