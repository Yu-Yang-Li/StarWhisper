# Repository structure and dependencies

[Project home](../README_EN.md) · [中文](architecture.md) · English · [Research notes](research-en.md) · [Research skills](../skills/README.md)

StarWhisper keeps code, data, and experiment material in directories for each research topic. Projects use their own runtime environments. Research skills provide ways to search, analyse, or read these materials. Choose the project or task first, then follow its instructions.

## Where to start

| Task | Start with | Next step |
| --- | --- | --- |
| Understand papers, results, and scope | [Research notes](research-en.md) | Read the project and original paper |
| Reproduce or modify a research project | Its README in the [project index](../projects/README.md) | Prepare its data, models, and environment |
| Use skills in Codex / Cursor | [Skill directory](../skills/README.md) | Install the needed set and read its `SKILL.md` |
| Inspect bundled tables or configuration | The offline tool table below | Confirm which input file the command reads |
| Modify station services or observing workflows | [NGSS](../projects/ngss/README.md) or [all-sky camera](../projects/allsky-camera/README.md) | Check external systems, permissions, and project validation |

## Directory responsibilities

```text
StarWhisper/
├── projects/   Research projects, with their code, data, and setup notes
├── skills/     Installable research skills, scripts, and references
└── docs/       Research notes, figures, and the documentation site
```

| Directory | Start here | What belongs here |
| --- | --- | --- |
| [projects](../projects/README.md) | Choose a research topic and read its README | Keep code, models, data, and results with their project; install environments separately |
| [skills](../skills/README.md) | Choose by task | The 17 skill names remain the same; see [INSTALL.md](../skills/INSTALL.md) for setup |
| [docs](index.html) | Read the [research notes](research-en.md) | Cross-project documentation, paper figures, and asset provenance |

The eight project directories use lowercase names with hyphens. Each project README describes its internal data layout and working directory. Configure NGSS service URLs and station deployment paths for the actual environment.

Pulsar code is maintained in [StarWhisper-Pulsar](https://github.com/ACMISLab/StarWhisper-Pulsar), and related Virtual Sitian workflows in [SitianClaw](https://github.com/Yu-Yang-Li/SitianClaw). Follow those repositories' own setup instructions.

## How offline tools read project material

The four `starwhisper-*` skills read CSV / JSON. They do not import training projects, make network calls, or control devices. They require Python 3.10 or newer and use the standard library.

| Tool | Default input | Explicit input or fallback |
| --- | --- | --- |
| [SN Clock](../skills/starwhisper-snclock/SKILL.md) | Last filename-sorted `projects/snclock/snclock_*.csv` | Use `--csv`; missing data produces an error |
| [Explore](../skills/starwhisper-explore/SKILL.md) | `projects/explore/published_metrics.csv` | Use `--csv`; without the repository, use the skill's bundled metrics |
| [Night planning](../skills/starwhisper-night-plan/SKILL.md) | `projects/ngss/observe_config.json` | Use `--config`; missing configuration uses a labelled bundled example |
| [Sparse light-curve comparison](../skills/starwhisper-varlen/SKILL.md) | Bundled results and `contract.json` | Compare within the defined data pool; `check --csv` checks user point counts and labels |

Installing skills does not copy entire research projects. Set `STARWHISPER_ROOT` to the repository root when data is needed; the tools also support the old top-level project paths, or specify inputs in the command. The night budget is an estimate under fixed assumptions; check it separately against actual NGSS plans and hardware execution.

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

When changing commands, paths, or input formats, check the associated documentation and tests. Review callers and data conventions before cross-project changes. Directory moves must update documentation links and offline input discovery together. See [Contributing](../CONTRIBUTING.md).

## Directory migration

The October 2026 layout uses the paths below. Relative paths inside projects stay the same; commands started from the repository root must use the new location.

| Previous location | Current location |
| --- | --- |
| `LLM_Data/` | [projects/language-models/](../projects/language-models/) |
| `StarWhisper_LC/` | [projects/light-curves/](../projects/light-curves/) |
| `Early Classification from Sparse Light Curves/` | [projects/sparse-light-curves/](../projects/sparse-light-curves/) |
| `Low-SNR-Stellar-Spectra-as-Language/` | [projects/stellar-spectra/](../projects/stellar-spectra/) |
| `NGSS/` | [projects/ngss/](../projects/ngss/) |
| `AllSky-Camera-XL/` | [projects/allsky-camera/](../projects/allsky-camera/) |
| `snclock/` | [projects/snclock/](../projects/snclock/) |
| `explore/` | [projects/explore/](../projects/explore/) |

Earlier showcase images, the retired skill entrypoint, NGSS historical logs and compiled caches, and the `GOTTA_Prototype` placeholders have been removed from the current branch. They remain available in the [snapshot before cleanup](https://github.com/Yu-Yang-Li/StarWhisper/tree/9496f3683a6138e9133510cd45f8bacd6588045b/archive).
