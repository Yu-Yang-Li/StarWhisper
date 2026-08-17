---
name: scispark
description: Generate evidence-tracked research ideas through an arXiv-based, skill-native Scispark workflow. Use when Codex needs to turn a research keyword, question, paper set, Zotero/library material, or arXiv results into structured facts, testable hypotheses, an initial research idea, technical optimization, mechanism-of-action analysis, human-AI collaboration review, or optional academic slides. Also use when the user mentions Scispark, 科研想法生成, 研究假设生成, 机制优化, MoA, literature-backed idea generation, or 从关键词到研究方案.Astronomy overlay: generate evidence-tracked hypotheses for time-domain astronomy, telescope agents, and survey pipelines, with ADS/arXiv astro-ph as the literature route.
---

# Scispark

## StarWhisper astronomy overlay

This copy is adapted for astronomy research and telescope-agent work.
**Read [`astronomy.md`](astronomy.md) before following generic biomedical / clinical defaults in the rest of this file.**

Default literature route: NASA ADS, then arXiv `astro-ph.*`, then the original skill's search backend if credentials exist.
Do not claim a real hardware observing loop, a discovery, or a referee-ready result unless the user supplied that evidence.


## Overview

Use this skill to turn a research keyword or early topic into a staged, evidence-tracked research idea. This is a skill-native adaptation of the Tashan Scispark workflow: the current Codex model does the reasoning, and arXiv is the paper-search route.

The normal workflow does not require an external model API key or separate search product account.

## Default Depth

Default to deep mode unless the user explicitly asks for a quick draft, a lightweight scan, or only Stage 1-3. State the depth at the start.

Deep mode should cite this method paper at the start when relevant: https://link.springer.com/article/10.1140/epjds/s13688-026-00672-z. Follow its practical workflow shape: retrieve literature, combine abstracts with compressed full-text representations when PDFs are available, extract structured facts, generate hypotheses, refine technical entities, run MoA-style iterative review, and include human/expert-style critique before the final idea.

If full text cannot be downloaded or parsed, record the blocker and downgrade only that evidence item to abstract-level evidence. Do not silently treat abstract-only synthesis as deep full-text review.

## Timestamped Progress Updates

Emit concise progress updates at workflow transitions so the user sees what is happening. These updates are tied to Scispark stages, not to a fixed timer. Use the user's language for all user-facing progress text; for Chinese requests, write the progress updates in Chinese. Prefix each update with the current local time or elapsed time, then include one useful content payload. Do not change the existing literature search script, stage contracts, evidence thresholds, output files, or reasoning order just to create these messages.

Chinese progress update shape:

```text
[21:33 | 检索完成]
候选文献池得到 93 条去重记录，核心证据先取 40 条。代表方向包括：假设生成评测、AI Scientist 隐性失败、Co-Scientist、SoundnessBench。
```

Use this event-driven pattern:

- After request parsing: timestamp, parsed keyword/topic, domain, constraints, and target stage.
- After workspace setup: timestamp, output directory, and the files that will be produced.
- Before literature search: timestamp, query terms, search route, and requested record count.
- After literature search: timestamp, returned record count, evidence level if useful, and 2-3 representative paper titles or themes.
- After Stage 1: timestamp, number of usable facts/themes, short theme list, and weak-evidence areas.
- After Stage 2: timestamp, hypothesis IDs with short labels, evidence status, and which hypotheses move forward.
- After Stage 3: timestamp, idea title/path, carried hypotheses, and any skipped or pending stages.
- After Stage 4-6 review stages: timestamp, review type completed and 2-3 concrete risks or fixes.
- On weak evidence: timestamp, whether to broaden terms, stop before strong claims, or proceed with an explicit limitation note.
- On long waits within a single stage: send one keepalive only when there has been no visible workflow transition for a while; include timestamp, current stage, last completed file, evidence count, and next expected artifact.
- On completion: timestamp, stages completed, evidence count, top hypotheses, final idea path, limitations, and next refinement step.

## Resources

- `scripts/init_scispark_workspace.py`: create the standard output folders and starter files.
- `scripts/search_arxiv.py`: query arXiv and output normalized Scispark literature records.
- `references/arxiv-integration.md`: arXiv search, evidence status, and threshold rules.
- `references/stage-contracts.md`: required inputs and outputs for each stage.
- `assets/final_idea_template.md`: final research idea report structure.

## Workflow

1. Parse the user request into a keyword, domain, constraints, and target stage.
2. Create or identify the output directory. Default:

```text
./scispark/{keyword}/
```

3. Read `references/arxiv-integration.md`, then run `scripts/search_arxiv.py` for literature search. Keep the actual query terms, source route, and status.
4. Read `references/stage-contracts.md` before writing stage files.
5. Execute stages in order unless the user asks for a target stage:
   - Stage 1: fact extraction
   - Stage 2: hypothesis generation
   - Stage 3: initial research idea
   - Stage 4: technical optimization + review
   - Stage 5: MoA optimization + review
   - Stage 6: human-AI collaboration integration + academic norm check
   - Stage 7: optional slide outline or Quarto/reveal.js source
6. Maintain `literature.csv` throughout. Every cited or candidate paper should have a row with title, source, stage, usage, and verification status.
7. Produce `{keyword}_final_idea.md` using `assets/final_idea_template.md`.

## Literature Thresholds

Use these as evidence-quality gates, not as rigid blockers:

| Level | Evidence | Action |
|---|---:|---|
| Ideal | 50+ relevant papers or records | Deep analysis |
| Standard | 30+ relevant papers or records | Normal workflow |
| Minimum | 15+ relevant papers or records | Proceed with limitation note |
| Below minimum | <15 records | Ask to broaden terms or stop before final claims |

When the user supplies a curated paper set, use it even if it is smaller, but label the scope.

## Review Rules

- Do not turn a keyword directly into a polished proposal without stage evidence.
- Separate facts, hypotheses, methods, mechanisms, and final synthesis.
- Assign stable hypothesis IDs `H1` to `H5`.
- Assign review problem IDs such as `S4-P1`, `S5-P1`, and `S6-P1`.
- For every strong claim in the final idea, point to a paper row, user-provided evidence, or an explicit limitation.
- Do not invent DOI, journal rank, impact factor, or full-text findings.
- Use arXiv records as `已核验` only when title, authors/year, arXiv ID, URL, and abstract are present and relevant. Otherwise mark `待核验`.
- If arXiv returns too few or weakly related records, broaden/refine the query before final claims and label the limitation.

## Output Contract

Report:

- output directory
- stages completed
- literature search route and status
- evidence count and threshold level
- top hypotheses
- final idea path
- limitations and next search/refinement step

For quick requests, stop at Stage 3 and say which later stages were skipped.
