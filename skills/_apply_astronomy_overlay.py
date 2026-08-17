# One-shot adapter used while polishing the public GitHub. Safe to keep.
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INSERT = (ROOT / "_overlay_insert.md").read_text(encoding="utf-8").strip() + "\n\n"

COMMON = """# Astronomy overlay

This skill is the StarWhisper astronomy adaptation of the upstream Tashan research skill.
Keep the original workflow, scripts, and evidence rules. Change the **defaults**.

## Default sources

| Need | First route | Fallback |
| --- | --- | --- |
| Refereed astronomy papers | NASA ADS (`ads_search`, bibcode) | arXiv `astro-ph.*` |
| Preprints / methods | arXiv `astro-ph.HE/CO/GA/IM/SR/EP/IM` | ADS `arxiv:` identifier |
| Object metadata | SIMBAD / NED | user-supplied catalog |
| Catalogs / tables | VizieR / CDS | paper supplementary |
| Transients | TNS / ATels / GCN | user-supplied alert |
| Observatory context | StarWhisper Telescope / NGSS notes | public site pages |

Do not treat PubMed-style biomedical search, GB/T 7714-only citation, RCT/clinical CONSORT, or wet-lab n as the default.

## Claim boundary

- Distinguish **synthetic environment**, **de-identified log replay**, and **real hardware**.
- Hardware safety interlocks outrank any agent action.
- A fitted model, a light-curve classifier score, or a simulation is not a physical mechanism.
- Do not write "discovered" for a synthetic transient or a demo candidate.
- Keep numbers, units, filters, MJD/UTC, coordinates, and bibcodes unchanged.

## Writing venues

English astronomy papers default to AAS / MNRAS / A&A / PASP citation practice.
Chinese grant or thesis text may still need GB/T 7714; say which standard is in force.
"""

SKILLS = {
    "giiisp-paper-search-apis": {
        "desc_extra": " For astronomy, prefer NASA ADS and arXiv astro-ph before Giiisp OA search. Use when the user asks for ADS, bibcode, ApJ/MNRAS papers, transient literature, or StarWhisper-related papers.",
        "body": """
## Astronomy search order

1. Parse the question into object / event / method / survey terms. Keep catalog names (ZTF, TNS, NGSS, SiTian, GOTO, ATLAS) as given.
2. Query NASA ADS first when a refereed paper is needed. Record bibcode, title, year, refereed flag.
3. Query arXiv `astro-ph.*` for methods and recent preprints.
4. Use the original Giiisp OA / arXiv routes only as a supplement, or when `GIIISP_AUTH_TOKEN` is available.
5. Return a short table: bibcode or arXiv id, title, year, venue, why it is relevant, evidence status (ADS hit / preprint / not verified).

Never invent a bibcode. If ADS is unavailable, say so and fall back to arXiv, then stop claiming refereed status.
""",
    },
    "sci-employee-deep-research": {
        "desc_extra": " Astronomy overlay: build evidence-bounded reviews for transients, surveys, telescope agents, and time-domain methods using ADS plus arXiv astro-ph.",
        "body": """
## Astronomy research report

A StarWhisper deep-research report must separate:

- what the literature actually measured;
- what is inferred from simulation or agent logs;
- what remains an open observing-system question.

Keyword split should include both science terms (early supernova, kilonova, AGN flare) and system terms (queue scheduling, ToO, dome safety, seeing, airmass).
If the original Deep Research HTTP endpoint is unreachable, do the same staged report locally with ADS/arXiv hits. Do not pretend a remote research API succeeded.
""",
    },
    "thesis-audit-reviewer": {
        "desc_extra": " Astronomy overlay: audit theses and papers for ADS-valid citations, coordinate/unit consistency, survey selection effects, and over-claimed discoveries.",
        "body": """
## Astronomy audit extras

In addition to the generic thesis checklist, flag:

- coordinates without epoch/frame (ICRS/FK5/B1950);
- magnitudes without filter/system;
- times without MJD/UTC/TT;
- "discovery" language on unconfirmed or synthetic candidates;
- missing selection-function / injection-recovery discussion for classification papers;
- references that cannot be resolved to a bibcode or arXiv id.

For StarWhisper Telescope / NGSS chapters, require an explicit hardware vs simulation boundary.
""",
    },
    "scispark": {
        "desc_extra": " Astronomy overlay: generate evidence-tracked hypotheses for time-domain astronomy, telescope agents, and survey pipelines, with ADS/arXiv astro-ph as the literature route.",
        "body": """
## Astronomy hypothesis rules

Prefer hypotheses that can be tested with:

- public light curves / spectra / catalogs; or
- a replayable observing-decision environment (fixed seeds, logged actions); or
- an ablation on a published classifier, not a new unfalsifiable "AI brain".

Mechanism-of-action language from biomedicine does not map cleanly to astrophysics.
Replace "MoA" with **physical process / observational selection / decision policy**.
Every hypothesis row must cite at least one ADS or arXiv record, or be marked `speculative`.
""",
    },
    "research-baseline-builder": {
        "desc_extra": " Astronomy overlay: turn questions into data contracts over light curves, spectra, FITS images, alert streams, or telescope decision logs.",
        "body": """
## Astronomy data contract

Name these fields before any model:

| Field | Astronomy default |
| --- | --- |
| Sample unit | object-night, alert, spectrum, image cutout, or decision slot — pick one |
| Input | light curve, spectrum, FITS, alert packet, weather/device log |
| Output | class, redshift, flux, ranking, or action (`observe`/`follow`/`defer`/`pause`) |
| Split | by object / night / field, never random-row split of time series |
| Leakage | future photometry, later spectroscopic labels, duplicate detections |
| Baseline | magnitude cut, template matching, or a published LC classifier — not an LLM first |

Do not start from EfficientNet/GRU unless the data really are images or sequences.
For StarWhisper-Explore decision logs, the sample unit is one slot in one synthetic night.
""",
    },
    "experiment-design": {
        "desc_extra": " Astronomy overlay: design observing campaigns, injection-recovery tests, and agent A/B evaluations before collecting or replaying data. Prefer survey/observational designs over clinical RCT language.",
        "body": """
## Astronomy design types

Map the generic DOE tree onto observing work:

- **Injection-recovery** for classifier completeness/purity.
- **Night/field blocking** instead of clinical site blocking.
- **Seeded synthetic campaigns** for agent policies (no-intervention / random / priority / agent).
- **Shadow-mode hardware runs**: agent suggests, interlock executes.

Randomizing human patients is the wrong default. Randomize nights, fields, seeds, or policy assignment.
Pre-register success and failure thresholds, including the allowed survey-completeness loss.
""",
    },
    "statistical-analysis": {
        "desc_extra": " Astronomy overlay: confirmatory stats for light-curve, survey, and agent-evaluation tables, including selection effects and multiple-testing across candidates.",
        "body": """
## Astronomy statistics extras

After the generic tests, check whether the claim also needs:

- Poisson / Gehrels small-n intervals for counts;
- survival / censoring for truncated light curves;
- look-elsewhere or trial-factor notes for many candidates;
- Malmquist / Eddington / Malmquist-like selection comments;
- bootstrap by object, not by photometric point, when points are correlated.

Do not upgrade a p-value on synthetic nights into a statement about real telescope performance.
""",
    },
    "scientific-humanization": {
        "desc_extra": " Astronomy overlay: humanize Chinese astronomy papers, proposals, and talks while keeping bibcodes, filters, MJD, and discovery language conservative.",
        "body": """
## Astronomy voice

Allowed concrete terms: 暂现源, 测光, 光谱证认, 巡天完成度, 安全联锁, 影子运行.
Forbidden inflated terms unless the evidence is explicit: 发现新物理, 全面智能观测, 无人值守已完成.

Keep filter names, magnitudes, redshifts, and bibcodes. Downgrade "证明超新星机制" to what the data actually constrain.
""",
    },
    "academic-writing": {
        "desc_extra": " Astronomy overlay: write, review, and submit astronomy manuscripts for AAS/MNRAS/A&A/PASP and Chinese NSFC astronomy proposals.",
        "body": """
## Astronomy writing venues

- Journal defaults: ApJ/AJ, MNRAS, A&A, PASP, Communications Engineering for the telescope-agent paper.
- Methods must state software, catalog versions, and whether results are from hardware, replay, or simulation.
- Related-work should cite robot telescopes / queue schedulers (pt5m, TJO/ISROCS, AstroQ) without claiming StarWhisper already solved their problem.
- Grant text for NSFC 天文: scientific question, instrument/data, verification metric, risk. Not "AI 平台生态".
""",
    },
    "giiisp-scientific-image-generation": {
        "desc_extra": " Astronomy overlay: generate paper figures for survey workflows, telescope decision loops, light-curve panels, and spectral sequences. Forbid fake data ticks and fake English labels.",
        "body": """
## Astronomy figure rules

- If the figure encodes data, the labels must match the paper: filters, MJD, flux units.
- Do not invent axis ticks, spectra, or sky images that look like real observations.
- Observing-loop figures may be schematic. Light-curve and spectrum figures should be regenerated from data, not from an image model, unless the user only wants a layout draft.
- Prefer the StarWhisper navy/teal/gold palette for system diagrams.
""",
    },
    "visual-deck-builder": {
        "desc_extra": " Astronomy overlay: build StarWhisper / time-domain / telescope-agent decks with conservative claims and readable observatory diagrams.",
        "body": """
## Astronomy deck rules

- Cover: project name, one scientific question, venue or date. No "AI 赋能宇宙".
- Keep a claims slide that separates published telescope results, open models, and synthetic Explore experiments.
- Do not put unpublished credentials, FTP hosts, or mount IPs on slides.
- Reuse `docs/assets/` diagrams when presenting StarWhisper itself.
""",
    },
    "papercheck": {
        "desc_extra": " Astronomy overlay: check citations against NASA ADS/arXiv, AAS reference style, and whether in-text claims are actually supported.",
        "body": """
## Astronomy citation check

- Resolve each reference to a bibcode or arXiv id when possible.
- English astronomy papers: AAS / ADS-export style is the default, not GB/T 7714.
- Flag "Wang et al. 2025 StarWhisper Telescope" if the DOI `10.1038/s44172-025-00520-4` is missing.
- A citation that exists but does not support the sentence is still a failure.
""",
    },
    "cognitive-profile": {
        "desc_extra": " Astronomy overlay: remember a researcher's subfield, preferred catalogs, claim conservatism, and telescope-safety boundaries for long-running StarWhisper assistance.",
        "body": """
## Astronomy persona fields

When the user is doing astronomy work, also record if stated:

- subfield (time-domain, stars, galaxies, cosmology, instrumentation);
- preferred catalogs and telescopes;
- whether they want conservative discovery language;
- whether hardware commands are allowed at all (default: no).

Do not store observatory passwords, FTP accounts, or unpublished target lists.
""",
    },
}

DESC_MARK = " StarWhisper astronomy overlay:"


def patch_skill_md(path: Path, extra: str) -> None:
    text = path.read_text(encoding="utf-8")
    if "## StarWhisper astronomy overlay" in text:
        return
    extra = extra.strip()
    # append trigger terms to YAML description
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        for i, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                break
            if line.startswith("description:"):
                if line.strip() == "description:":
                    # folded/block; append after the block by adding extra to first non-empty later is hard.
                    # Put extra into a new YAML comment-free suffix on the last description line before ---.
                    pass
                elif DESC_MARK not in line:
                    # quoted or plain single-line description
                    raw = line[len("description:") :].strip()
                    if raw.startswith(">-"):
                        pass
                    elif raw.startswith(">") or raw.startswith("|"):
                        pass
                    else:
                        quote = ""
                        if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
                            quote = raw[0]
                            raw = raw[1:-1]
                        new = raw.rstrip() + extra
                        if quote:
                            new = quote + new.replace(quote, "") + quote
                        lines[i] = "description: " + new
                break
    text = "\n".join(lines)
    if not text.endswith("\n"):
        text += "\n"
    # insert overlay after first markdown heading
    out = []
    inserted = False
    for line in text.splitlines(True):
        out.append(line)
        if not inserted and line.startswith("# "):
            out.append("\n" + INSERT)
            inserted = True
    path.write_text("".join(out), encoding="utf-8")


def main() -> None:
    for name, spec in SKILLS.items():
        skill_dir = ROOT / name
        (skill_dir / "astronomy.md").write_text(COMMON + spec["body"].lstrip() + "\n", encoding="utf-8")
        patch_skill_md(skill_dir / "SKILL.md", spec["desc_extra"])
        print("adapted", name)


if __name__ == "__main__":
    main()
