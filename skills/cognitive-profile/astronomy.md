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
## Astronomy persona fields

When the user is doing astronomy work, also record if stated:

- subfield (time-domain, stars, galaxies, cosmology, instrumentation);
- preferred catalogs and telescopes;
- whether they want conservative discovery language;
- whether hardware commands are allowed at all (default: no).

Do not store observatory passwords, FTP accounts, or unpublished target lists.

