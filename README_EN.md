# StarWhisper

[中文](README.md) · English · [Research notes](docs/research-en.md) · [Skill pack](skills/README.md)

![Concept illustration of a mountain observatory under the night sky](docs/assets/starwhisper-cover.png)

StarWhisper is open astronomy research supported by NAOC, Zhejiang Lab, and collaborators. It began with astronomy language models in 2023 and expanded to light-curve and pulsar-candidate classification, autonomous observing, and the supernova-clock work within Virtual Sitian.

This repository brings together code, public data, paper materials, and astronomy research skills from these projects. The 2026 main line is Virtual Sitian (SN Clock); the observing-agent work is described in the published [StarWhisper Telescope](https://doi.org/10.1038/s44172-025-00520-4) paper.

[![License](https://img.shields.io/badge/License-Apache--2.0-172b4d)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-Communications%20Engineering-172b4d)](https://doi.org/10.1038/s44172-025-00520-4)
[![GitHub Repo stars](https://img.shields.io/github/stars/Yu-Yang-Li/StarWhisper?style=social)](https://github.com/Yu-Yang-Li/StarWhisper/stargazers)

## Research directions

| Direction | Research | Entry points |
| --- | --- | --- |
| Astronomy language models | Astronomy questions, scientific text, and code | [Training data](LLM_Data/README.md) · [StarWhisper 3 weights](https://www.modelscope.cn/models/AstroYuYang/StarWhisper3) |
| Light curves and pulsars | Kepler / K2 variable-star classification and pulsar-candidate classification | [StarWhisper LC](StarWhisper_LC/README.md) · [Pulsar](https://github.com/ACMISLab/StarWhisper-Pulsar) |
| Observing agents | Planning, observing, and exception handling on the Nearby Galaxy Supernovae Survey | [NGSS](NGSS/README.md) · [Paper](https://doi.org/10.1038/s44172-025-00520-4) |
| Virtual Sitian | Supernova explosion-epoch estimates and young-candidate screening | [SN Clock data](snclock/README.md) · [SitianClaw](https://github.com/Yu-Yang-Li/SitianClaw) · [Project map](https://yu-yang-li.github.io/StarWhisper/virtual-gotta-map.html) |

Related work: [early classification from sparse light curves](Early%20Classification%20from%20Sparse%20Light%20Curves/README.md), [Xinglong all-sky camera](AllSky-Camera-XL/README.md), and [low-SNR stellar spectra](Low-SNR-Stellar-Spectra-as-Language/README.md). Synthetic observing-policy experiments are documented in [Explore](explore/README.md).

The [research notes](docs/research-en.md) collect the timeline, original figures, public results, and reproduction scope.

## Getting started

For a first example, read the historical SN Clock prediction table included in the repository, entirely offline:

```bash
git clone https://github.com/Yu-Yang-Li/StarWhisper.git
cd StarWhisper
python skills/starwhisper-snclock/scripts/screen_snclock.py rank --top 10
```

This example needs Python 3.10 or newer and the standard library. The 22 sources come from a historical output; see the [data card](snclock/README.md) for coverage and input retention. Screening candidates at the current time requires current data.

| What you want to do | Start here |
| --- | --- |
| Use astronomy research skills in Codex / Cursor | [Installation and catalog](skills/README.md): 4 native skills and 13 astronomy-adapted research skills |
| Reproduce sparse light-curve classification | [Training workflow and model weights](Early%20Classification%20from%20Sparse%20Light%20Curves/README.md) |
| Set up the observing agent | [NGSS environment and interfaces](NGSS/README.md), with the corresponding observing services and device connections |
| Explore the Virtual Sitian workflow | [SitianClaw](https://github.com/Yu-Yang-Li/SitianClaw) |

## Papers and citation

- [StarWhisper Telescope: an AI framework for automating end-to-end astronomical observations](https://doi.org/10.1038/s44172-025-00520-4) — *Communications Engineering*, 2025.
- [StarWhisper LC](https://spj.science.org/doi/10.34133/icomputing.0110) — *Intelligent Computing*, 2025.
- [StarWhisper-Pulsar](https://openreview.net/pdf?id=8SKgWpZiDL) — NeurIPS 2024 FM4Science workshop.

See [CITATION.cff](CITATION.cff) and the [BibTeX entry](docs/research-en.md#citation). Please also cite the relevant paper when using a specific research project.

Questions and reproduction notes are welcome through [issues](https://github.com/Yu-Yang-Li/StarWhisper/issues) and pull requests; see the [contribution guide](CONTRIBUTING.md). Source code is [Apache-2.0](LICENSE); adapted research skills retain their MIT license and attribution in [NOTICE](skills/NOTICE.md). Model weights follow their own licenses.