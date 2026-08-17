# Astronomy overlay

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
## Astronomy hypothesis rules

Prefer hypotheses that can be tested with:

- public light curves / spectra / catalogs; or
- a replayable observing-decision environment (fixed seeds, logged actions); or
- an ablation on a published classifier, not a new unfalsifiable "AI brain".

Mechanism-of-action language from biomedicine does not map cleanly to astrophysics.
Replace "MoA" with **physical process / observational selection / decision policy**.
Every hypothesis row must cite at least one ADS or arXiv record, or be marked `speculative`.

