---
name: papercheck
description: Check astronomy citations against NASA ADS/arXiv, AAS/MNRAS reference practice, and whether in-text claims are actually supported. Use for .docx/.pdf citation audits, bibcode resolvability, and context-support mismatches. Do not invent references.
---

# PaperCheck

## StarWhisper astronomy overlay

Read [`astronomy.md`](astronomy.md) first. That file sets the astronomy defaults for this copy.

Literature: NASA ADS, then arXiv `astro-ph.*`. Do not invent papers.
A synthetic Explore run, a classifier score, or a demo candidate is not a discovery.
This skill does not send telescope commands.


## Overview

Use this skill to run citation audits without any external model API key. Local code extracts references, citation markers, format/matching evidence, and citation contexts; the current mounted Codex model performs the semantic review.

Bundled runtime:

- No-key evidence extractor: `scripts/extract_citation_evidence.py`
- Rules engine source: `assets/paperchecker-rules`

Optional override:

- `PAPERCHECK_RULES_REPO`: use a different TaShan-PaperChecker rules repo.

Do not package or print private configs, uploaded papers, generated reports, dependency folders, caches, or API keys. The normal workflow does not need provider keys.

## Workflow

1. Run `scripts/check_papercheck_env.py` before the first run or before packaging.
2. Extract citation evidence:

```powershell
python .\scripts\extract_citation_evidence.py "<path-to-paper.docx>" --out "<path-to-evidence.json>"
```

For user-visible progress, prefer the JSONL wrapper:

```powershell
python .\scripts\progressive_papercheck.py "<path-to-paper.docx-or-pdf>" --mode subjective --out "<path-to-evidence.json>" --report "<path-to-report.md>"
```

Modes:

- `quick`: structural evidence extraction plus bundled UCAS/GB/T rules checks; skips semantic review.
- `subjective`: default; rules checks plus a current-model review queue for citation support.
- `full`: subjective mode plus explicit source-verification limits for supplied source PDFs, DOI/OA evidence, or other verified source content.

It emits `mode_selected`, `env_check_started`, `env_check_complete`, `parser_selected`, `evidence_extract_started` or `pdf_extract_started`, `evidence_ready`, `rules_check_started`, `rules_check_complete`, `model_review_ready` or `semantic_review_skipped`, optional `source_verification_ready`, `report_ready`, and `papercheck_complete`. For PDF, it also reports whether MinerU is configured, whether local `pymupdf4llm` / `PyMuPDF` fallback is available, and how to fix missing keys or dependencies.

3. Read the evidence JSON and let the current Codex model judge high-value citations. For each judgment, cite the extracted reference entry and context. Mark weak or ambiguous cases as `待人工确认`; do not invent paper content beyond the provided evidence.
4. For format and matching rules, run the bundled rules server from `assets/paperchecker-rules` and call `/api/v2/analysis/report`, or inspect its JSON report if already produced.
5. If using a web UI, start the bundled rules version. Prefer CLI/evidence files when the user only needs a report.

Read `references/runtime.md` when you need install commands, API routes, failure handling, or the rules/evidence split.

## Output Contract

When reporting results, include:

- command used
- input `.docx` path and generated report/evidence path
- whether the result came from evidence extraction, rules engine, or current-model review
- high-risk findings: unmatched citations, unused references, weak citation-context support, format-rule issues
- evidence limits: treat findings as audit assistance, not academic/legal certification

Do not say a citation is authentic just because a title appears in a reference list. Only make stronger claims when the supplied reference entry, citation context, and any provided source paper/PDF support them.

## Guardrails

- Do not ask for provider API keys for the normal workflow; the mounted Codex model is the semantic reviewer.
- Evidence extraction input is `.docx`; convert other formats first or ask for a `.docx`.
- If a citation has no matching reference entry, report it as a structural citation/reference error first. Do not convert it into a semantic "not related" judgment.
- If only the reference title and local citation context are available, avoid claims about the cited paper's full content. Mark broad-title or claim-scope mismatches as weak support or `待人工确认` unless the source paper/PDF is also provided.
- When the reference title and citation context directly match the same method, sensor type, task, limitation, or review scope, it is acceptable to mark the citation as topically supported from the available evidence. Still avoid saying the cited paper truly proves the claim unless source content is provided.
- For range citations such as `[1-3]`, judge each numbered reference against the shared claim. Mark partial support when one reference supports only a method, sensor type, dataset, or application slice rather than the whole sentence.
- Bundle only cleaned rules source. Do not bundle dependency folders, uploaded papers, generated reports, caches, or local configs.
- If dependencies are missing, report what can still be checked and what was not run.
