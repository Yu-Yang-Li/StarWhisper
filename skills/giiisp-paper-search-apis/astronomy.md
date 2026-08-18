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
## Astronomy search order

1. Parse the question into object / event / method / survey terms. Keep catalog names (ZTF, TNS, NGSS, SiTian, GOTO, ATLAS) as given.
2. Run `scripts/ads_first_search.py` first (`--dry-run` if there is no token). Query NASA ADS when a refereed paper is needed. Record bibcode, title, year, refereed flag.
3. Query arXiv `astro-ph.*` for methods and recent preprints. The same script does this when ADS is unavailable.
4. Use the original Giiisp OA / arXiv routes only as a supplement, or when `GIIISP_AUTH_TOKEN` is available.
5. Return a short table: bibcode or arXiv id, title, year, venue, why it is relevant, evidence status (ADS hit / preprint / not verified).

Never invent a bibcode. If ADS is unavailable, say so and fall back to arXiv, then stop claiming refereed status.

