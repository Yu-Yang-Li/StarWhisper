# StarWhisper

[中文](README.md) · English · [Research notes](docs/research-en.md) · [Skill pack](skills/README.md)

![Concept illustration of a mountain observatory under the night sky](docs/assets/starwhisper-cover.png)

StarWhisper is an open-source repository integrating astronomy AI language models, data-analysis tools, observing agents, and research skills. It provides a single entry point for code, data, model resources, and usage documentation.

Choose modules according to your task and follow their individual READMEs for environment setup and execution. Model weights and some complete workflows are maintained in external projects, linked in the catalog below. The work is supported by NAOC, Zhejiang Lab, and collaborators.

[![License](https://img.shields.io/badge/License-Apache--2.0-172b4d)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-Communications%20Engineering-172b4d)](https://doi.org/10.1038/s44172-025-00520-4)
[![GitHub Repo stars](https://img.shields.io/github/stars/Yu-Yang-Li/StarWhisper?style=social)](https://github.com/Yu-Yang-Li/StarWhisper/stargazers)

## Modules and resources

| Module | Purpose | Code and resources |
| --- | --- | --- |
| Astronomy language models | Astronomy questions, scientific text, and code | [Training data](LLM_Data/README.md) · [StarWhisper 3 weights](https://www.modelscope.cn/models/AstroYuYang/StarWhisper3) |
| Light curves and pulsars | Kepler / K2 variable-star classification and pulsar-candidate classification | [StarWhisper LC](StarWhisper_LC/README.md) · [Pulsar](https://github.com/ACMISLab/StarWhisper-Pulsar) |
| Sparse light curves | Early classification of sparse ZTF / ATLAS light curves | [Training, evaluation, and weights](Early%20Classification%20from%20Sparse%20Light%20Curves/README.md) |
| Stellar spectra | Low-SNR stellar-spectra analysis | [Code and documentation](Low-SNR-Stellar-Spectra-as-Language/README.md) |
| Observing agents | Planning, observing, and exception handling on the Nearby Galaxy Supernovae Survey | [NGSS](NGSS/README.md) · [Paper](https://doi.org/10.1038/s44172-025-00520-4) |
| All-sky camera | Image masking and observing-sequence adjustment | [Xinglong all-sky camera](AllSky-Camera-XL/README.md) |
| Virtual Sitian | Supernova explosion-epoch estimates and young-candidate screening | [SN Clock data](snclock/README.md) · [SitianClaw](https://github.com/Yu-Yang-Li/SitianClaw) · [Project map](https://yu-yang-li.github.io/StarWhisper/virtual-gotta-map.html) |
| Observing-policy evaluation | Policy comparisons and result gates in a synthetic environment | [Explore specification and results](explore/README.md) |
| Research skills | Literature search, research design, statistics, writing, figures, and presentations | [Installation and catalog of 17 skills](skills/README.md) |

The [research notes](docs/research-en.md) collect each module's background, original figures, public results, and reproduction scope.

## Getting started

Select a module from the catalog and follow its documentation to set up the environment and data. Research skills can be installed in Codex / Cursor using the [skill-pack guide](skills/README.md). For an example with no extra dependencies, read the included historical SN Clock prediction table entirely offline:

```bash
git clone https://github.com/Yu-Yang-Li/StarWhisper.git
cd StarWhisper
python skills/starwhisper-snclock/scripts/screen_snclock.py rank --top 10
```

This example needs Python 3.10 or newer and the standard library. The 22 sources come from a historical output; see the [data card](snclock/README.md) for coverage and input retention. Screening candidates at the current time requires current data.

## Papers and citation

- [StarWhisper Telescope: an AI framework for automating end-to-end astronomical observations](https://doi.org/10.1038/s44172-025-00520-4) — *Communications Engineering*, 2025.
- [StarWhisper LC](https://spj.science.org/doi/10.34133/icomputing.0110) — *Intelligent Computing*, 2025.
- [StarWhisper-Pulsar](https://openreview.net/pdf?id=8SKgWpZiDL) — NeurIPS 2024 FM4Science workshop.

See [CITATION.cff](CITATION.cff) and the [BibTeX entry](docs/research-en.md#citation). Please also cite the relevant paper when using a specific research project.

Questions and reproduction notes are welcome through [issues](https://github.com/Yu-Yang-Li/StarWhisper/issues) and pull requests; see the [contribution guide](CONTRIBUTING.md). Source code is [Apache-2.0](LICENSE); adapted research skills retain their MIT license and attribution in [NOTICE](skills/NOTICE.md). Model weights follow their own licenses.
