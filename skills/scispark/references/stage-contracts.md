# Scispark Stage Contracts

## Stage 1: Fact Extraction

Inputs:

- keyword and domain constraints
- arXiv search results or user-provided papers
- user-provided papers or summaries

Output: `01_fact_extraction.md`

Must include:

- search log: query, route/source, count, status
- structured fact table
- each fact linked to `Lxxx`
- evidence gaps and threshold level
- optional expert overview in `experts/01_domain_overview.md`

## Stage 2: Hypothesis Generation

Input: `01_fact_extraction.md`

Output: `02_hypothesis.md`

Must include:

- 3-5 hypotheses with IDs `H1` to `H5`
- hypothesis statement
- reasoning basis
- falsifiable prediction
- key evidence rows
- evaluation matrix: verifiability, novelty, feasibility, impact, priority
- suggestion for combining hypotheses in Stage 3

## Stage 3: Initial Idea

Inputs:

- `02_hypothesis.md`
- 5-8 additional method/domain papers when needed

Output: `03_initial_idea.md`

Must include:

- title
- 200-300 word abstract
- 3-5 research aims
- method overview
- expected results
- innovation points
- hypothesis integration table

For quick-mode requests, Stage 3 can be the stopping point.

## Stage 4: Technical Optimization + Review

Input: `03_initial_idea.md`

Output: `04_technical_optimization.md`

Must include:

- review of current strengths and weaknesses
- problem IDs `S4-P1`, `S4-P2`, ...
- method papers or protocol evidence
- method-to-hypothesis mapping
- feasibility risks and fallback methods
- optional `experts/04_methodology_expert.md`

## Stage 5: MoA Optimization + Review

Input: `04_technical_optimization.md`

Output: `05_moa_optimization.md`

Evaluate five dimensions:

- molecular or event trigger
- pathway or process chain
- causal relationship
- temporal dynamics
- spatial or context specificity

Must include:

- mechanism gaps
- problem IDs `S5-P1`, `S5-P2`, ...
- mechanism-to-hypothesis mapping
- evidence strength and missing experiments
- optional `experts/05_mechanism_expert.md`

## Stage 6: Human-AI Collaboration Integration

Input: `05_moa_optimization.md`

Outputs:

- `06_human_ai_collaboration.md`
- `{keyword}_final_idea.md`

Must include:

- final review
- cross-stage issue table
- academic norm check
- final hypothesis-to-design mapping
- limitations
- next experiments or search plan

Use `assets/final_idea_template.md` for the final report.

## Stage 7: Slides

Input: `{keyword}_final_idea.md`

Output folder: `slides/`

Minimum output:

- slide outline
- speaker notes
- references list

Only create Quarto/reveal.js files when the user asks for actual slide source or the environment has Quarto available.
