# arXiv Integration

Use arXiv as the Scispark search route because it is open, stable, and does not require login.

## Query Planning

Turn the user request into 1-3 query groups:

| Group | Purpose | Example |
|---|---|---|
| domain | field background | `retrieval augmented generation scientific literature` |
| mechanism | causal or MoA evidence | `context aware retrieval augmented generation` |
| method | feasible methods and benchmarks | `scientific paper question answering benchmark` |

Prefer precise terms over broad buzzwords. If the first query returns low relevance, refine before writing stage conclusions.

## Script

Run:

```powershell
python C:\Users\16571\.codex\skills\scispark\scripts\search_arxiv.py "retrieval augmented generation scientific literature" --max-results 20 --out arxiv_results.json
```

The script outputs normalized records compatible with `literature.csv`:

- `title`
- `authors`
- `year`
- `venue`
- `arxiv_id`
- `url`
- `pdf_url`
- `abstract`
- `source_api`
- `query`
- `verification_status`
- `match_reason`

## Literature CSV

Use this header:

```csv
id,title,authors,year,venue,doi,arxiv_id,url,pdf_url,source_api,query,stage,usage,verification_status,notes
```

`source_api` should be `arXiv API`. `verification_status` may be:

- `已核验`: metadata is complete and title/abstract supports use.
- `待核验`: metadata exists, but relevance or intended use is not yet strong enough.
- `不支持`: the paper does not support the intended claim.

## Threshold Handling

- 50+ relevant records: deep analysis.
- 30+ relevant records: standard workflow.
- 15+ relevant records: proceed with limitation note.
- below 15: broaden or refine queries before final claims.

Do not chase count if the user only asks for quick mode or target stage 1-3.

## When arXiv Is Not Enough

If arXiv returns too few or off-topic records, do not pretend the evidence is sufficient. Try one narrower query and one broader query. If the result is still below threshold, stop before strong final claims and report the limitation.
