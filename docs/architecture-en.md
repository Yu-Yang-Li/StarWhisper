# Repository structure and dependencies

[Project home](../README_EN.md) · [中文](architecture.md) · English · [Research notes](research-en.md) · [Research skills](../skills/README.md)

StarWhisper keeps code, data, and experiment material in directories for each research topic. Projects use their own runtime environments. Research skills provide ways to search, analyse, or read these materials. Choose the project or task first, then follow its instructions.

## Where to start

| Task | Start with | Next step |
| --- | --- | --- |
| Understand papers, results, and scope | [Research notes](research-en.md) | Read the project and original paper |
| Reproduce or modify a research project | Its README, listed below | Prepare its data, models, and environment |
| Use skills in Codex / Cursor | [Skill directory](../skills/README.md) | Install the needed set and read its `SKILL.md` |
| Inspect bundled tables or configuration | The offline tool table below | Confirm which input file the command reads |
| Modify station services or observing workflows | [NGSS](../NGSS/README.md) or [all-sky camera](../AllSky-Camera-XL/README.md) | Check external systems, permissions, and project validation |

## Directory responsibilities

| Directory | Material | Runtime and data relationship |
| --- | --- | --- |
| [LLM_Data](../LLM_Data/README.md) | Astronomy language-model training text and notes | Model weights use the external locations listed by the project |
| [StarWhisper_LC](../StarWhisper_LC/README.md) | Kepler / K2 classification test code | Separate from the sparse light-curve benchmark |
| [Early Classification from Sparse Light Curves](../Early%20Classification%20from%20Sparse%20Light%20Curves/README.md) | Sparse light-curve training, evaluation, weights, and results | The project defines its data pools, labels, and comparison protocol |
| [Low-SNR-Stellar-Spectra-as-Language](../Low-SNR-Stellar-Spectra-as-Language/README.md) | Stellar-spectrum training and analysis | Own vocabulary, preprocessing, and model dependencies |
| [NGSS](../NGSS/README.md) | Survey configuration, planning, services, and observing operations | Uses NINA, FTP, MQTT, and station systems; some routes change external state |
| [AllSky-Camera-XL](../AllSky-Camera-XL/README.md) | Image inference, masks, and observing-sequence adjustment | Uses image models and calibration data; a generated sequence is not an executed observation |
| [GOTTA_Prototype](../GOTTA_Prototype) | Real/bogus classification prototype material | Interpret results within the prototype's data scope |
| [snclock](../snclock/README.md) | A historical explosion-age prediction table and field notes | Offline inspection and screening, rather than a live data service |
| [explore](../explore/README.md) | Synthetic observing-policy setup and metrics | Metric comparisons can be recalculated; environment code is not included |
| [skills](../skills/README.md) | 17 installable skills and supporting tools | See [INSTALL.md](../skills/INSTALL.md); load references for the current task |
| [skill](../skill/README.md) | Migration note for the old skill location | Current installers scan `skills/` |
| [docs](index.html) and [example](../example/README.md) | Research notes, figures, provenance, and older examples | The documentation site starts at `docs/index.html` |

Pulsar code is maintained in [StarWhisper-Pulsar](https://github.com/ACMISLab/StarWhisper-Pulsar), and related Virtual Sitian workflows in [SitianClaw](https://github.com/Yu-Yang-Li/SitianClaw). Follow those repositories' own setup instructions.

## How offline tools read project material

The four `starwhisper-*` skills read CSV / JSON. They do not import training projects, make network calls, or control devices. They require Python 3.10 or newer and use the standard library.

| Tool | Default input | Explicit input or fallback |
| --- | --- | --- |
| [SN Clock](../skills/starwhisper-snclock/SKILL.md) | Last filename-sorted `snclock/snclock_*.csv` | Use `--csv`; missing data produces an error |
| [Explore](../skills/starwhisper-explore/SKILL.md) | `explore/published_metrics.csv` | Use `--csv`; without the repository, use the skill's bundled metrics |
| [Night planning](../skills/starwhisper-night-plan/SKILL.md) | `NGSS/observe_config.json` | Use `--config`; missing configuration uses a labelled bundled example |
| [Sparse light-curve comparison](../skills/starwhisper-varlen/SKILL.md) | Bundled results and `contract.json` | Compare within the defined data pool; `check --csv` checks user point counts and labels |

Installing skills does not copy entire research projects. Set `STARWHISPER_ROOT` when repository data is needed, or specify inputs in the command. The night budget is an estimate under fixed assumptions; check it separately against actual NGSS plans and hardware execution.

## Skill organisation

```text
skills/<skill-name>/
├── SKILL.md          Task, starting point, output, and essential constraints
├── astronomy.md      Astronomy-specific guidance for adapted skills
├── references/       Data conventions, interfaces, longer workflows, and checks
├── scripts/          Existing computation, parsing, search, or packaging tools
├── templates/        Input or report templates
└── agents/           Supplied UI metadata and invocation settings
```

Not every skill needs every directory. Keep starting instructions in the entrypoint and load details for the current task. Run commands from the skill directory; use generated workspaces as described by that skill.

The deck scripts use generation and review scripts from the neighboring `giiisp-scientific-image-generation` skill. The sibling location is the default; `GIIISP_IMAGE_SKILL_DIR` can override it. Install both skills and configure image-service and visual-review credentials separately. Local PPTX structure checks do not replace content and image review.

## Where changes belong

- Keep scientific code, project configuration, and data processing in their project directory, with that project's dependencies.
- Record results, data scope, and reproduction limits in the project README and research notes, alongside sources and comparison conditions.
- Put short skill instructions in `SKILL.md`; put task-specific detail in linked `references/` files.
- Maintain installation, repository data location, and external-service setup in the [installation guide](../skills/INSTALL.md).
- Write generated files to the skill's work directory. Do not commit credentials, private configuration, or unpublished material.

When changing commands, paths, or input formats, check the associated documentation and tests. Review callers and data conventions before cross-project changes. Documentation maintenance does not require a shared training environment or moving existing data. See [Contributing](../CONTRIBUTING.md).
