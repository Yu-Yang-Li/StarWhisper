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

