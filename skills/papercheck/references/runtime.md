# PaperCheck Runtime Notes

## Architecture

- `scripts/extract_citation_evidence.py` is the default no-key path. It extracts citation/reference/context evidence from `.docx`.
- The mounted Codex model reviews the extracted evidence and writes the semantic citation-support judgment.
- `assets/paperchecker-rules` is the bundled TaShan-PaperChecker rules engine for GB/T 7714-2015, UCAS-style format checks, and citation/reference matching.

The old PaperCheck AI project needed a provider API key because it was not itself a Codex skill. This skill does not need that pattern: it uses local extraction plus the current model that is already running the skill.

## Evidence Extraction

```powershell
python ..\scripts\progressive_papercheck.py "<path-to-paper.docx-or-pdf>" --mode subjective --out "<path-to-evidence.json>" --report "<path-to-report.md>"
```

Use modes to match the two upstream project styles:

- `quick` maps to the rule-engine workflow: local extraction, UCAS/GB/T rules, missing citations, unused references, and format issues.
- `subjective` maps to the PaperCheck AI workflow without a provider key: it prepares citation/context/reference evidence for the mounted Codex model to judge support.
- `full` keeps the same no-key model review path and adds source-verification caveats. Only claim source-content support when the user supplies source PDFs or verified retrieval evidence.

The evidence JSON contains:

- citation markers and expanded ranges
- numbered references
- missing citations and unused references
- local context around each citation
- `needs_model_review` markers for citations where Codex should judge support

The Markdown report contains runtime status, MinerU/fallback status, evidence summary, rules summary, semantic review queue, and audit limits.

Review rule: judge only from extracted context, reference entry, and any user-supplied paper/PDF. If the actual cited paper content is unavailable, mark that limitation instead of overclaiming.

## Rules Engine

Bundled source: `assets/paperchecker-rules`

Install:

```powershell
cd .\assets\paperchecker-rules
pip install -r requirements.txt
```

Start:

```powershell
python run_server.py
```

Default local URL after startup is usually `http://127.0.0.1:8002`. Confirm with:

```powershell
Invoke-RestMethod http://127.0.0.1:8002/api/health
```

Upload check:

```powershell
curl.exe -s -X POST -F "file=@<path-to-paper.docx>" -F "author_format=full" -F "citation_standard=ucas" "http://127.0.0.1:8002/api/v2/analysis/report" -o report.json
```

Expected success fields: `contract_version`, `run.status=succeeded`, `summary.match_rate`, and issue groups under `issues`.

## PDF Extraction And MinerU

For PDF uploads, the rules engine tries this order:

1. MinerU API converts the PDF into Markdown with layout-aware extraction.
2. If MinerU is missing, expired, or unavailable, PaperCheck first tries local `pymupdf4llm.to_markdown()` to preserve Markdown-like structure.
3. If `pymupdf4llm` is unavailable, fails, or returns unusable text, PaperCheck falls back to local `PyMuPDF/fitz` and calls `page.get_text(sort=True)` page by page.

When `scripts/check_papercheck_env.py` reports `pdf_extraction.configured=false`, tell the user:

- MinerU is not configured.
- They can set `MINERU_API_KEY` or fill `assets/paperchecker-rules/config/config.json` under `mineru_config.api_key`.
- Until then, PaperCheck will continue with the local `pymupdf4llm` / `PyMuPDF` fallback if those dependencies are installed.
- The fallback is the best no-key built-in option currently packaged here for text-layer PDFs, especially when `pymupdf4llm` can preserve headings, lists, and some table structure. It is still not the optimal production parser for scanned PDFs, complex multi-column layout, tables, formulas, headers/footers, or references split across pages. MinerU or another OCR/layout parser is preferred for higher-confidence PDF audits.

## Known Limits

- Evidence extraction only supports `.docx` directly.
- Rules-service upload supports `.docx`, `.doc`, and `.pdf` according to the upstream rules project, but extraction quality depends on document structure.
- The current model can judge support from extracted context and reference text, but it cannot verify the actual cited paper's full content unless the paper/PDF is provided.

## Packaging Boundary

The skill package should contain `SKILL.md`, `agents/`, `scripts/`, `references/`, and cleaned `assets/paperchecker-rules`. It must not contain:

- provider API keys
- `node_modules`
- uploaded or sample papers
- generated markdown/JSON reports
- cache directories
- local config values
