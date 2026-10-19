# StarWhisper

[中文](README.md) · English · [Research notes](docs/research-en.md) · [Skill pack](skills/README.md) · [Repository structure](docs/architecture-en.md)

![Concept illustration of a mountain observatory under the night sky](docs/assets/starwhisper-cover.png)

StarWhisper collects code, data, and models from work on astronomy language models, light curves, stellar spectra, and autonomous telescope observing. It also includes research skills for Codex / Cursor. The work is supported by NAOC, Zhejiang Lab, and collaborators.

[![License](https://img.shields.io/badge/License-Apache--2.0-172b4d)](LICENSE)
[![GitHub Repo stars](https://img.shields.io/github/stars/Yu-Yang-Li/StarWhisper?style=social)](https://github.com/Yu-Yang-Li/StarWhisper/stargazers)

## Projects

| Research area | Projects |
| --- | --- |
| Astronomy language models | [Training data](projects/language-models/README.md) · [StarWhisper 3 weights](https://www.modelscope.cn/models/AstroYuYang/StarWhisper3) |
| Light curves and transients | [Kepler / K2 classification](projects/light-curves/README.md) · [Sparse light curves](projects/sparse-light-curves/README.md) · [SN Clock](projects/snclock/README.md) · [Pulsar](https://github.com/ACMISLab/StarWhisper-Pulsar) |
| Stellar spectra | [Low-SNR spectral analysis](projects/stellar-spectra/README.md) |
| Observing and scheduling | [NGSS](projects/ngss/README.md) · [All-sky camera](projects/allsky-camera/README.md) · [Explore](projects/explore/README.md) · [SitianClaw](https://github.com/Yu-Yang-Li/SitianClaw) |

See the [project index](projects/README.md) for contents and setup, and [research skills](skills/README.md) for literature, experiment design, statistics, writing, and figures.

Paper figures, experimental results, and reproduction notes are in the [research notes](docs/research-en.md).

## Getting started

Each project's README explains installation and usage. Some code and models are kept in separate repositories, linked above. The [research skills](skills/README.md) can be installed in Codex / Cursor.

For example, these commands read the historical SN Clock prediction table included in the repository:

```bash
git clone https://github.com/Yu-Yang-Li/StarWhisper.git
cd StarWhisper
python skills/starwhisper-snclock/scripts/screen_snclock.py rank --top 10
```

Python 3.10 or newer is required, with no additional dependencies. The table contains historical predictions for 22 sources; see the [data notes](projects/snclock/README.md) for their origin and limitations. Use current data when screening current candidates.

## Papers and citation

- [StarWhisper Telescope: an AI framework for automating end-to-end astronomical observations](https://doi.org/10.1038/s44172-025-00520-4) — *Communications Engineering*, 2025.
- [StarWhisper LC](https://spj.science.org/doi/10.34133/icomputing.0110) — *Intelligent Computing*, 2025.
- [StarWhisper-Pulsar](https://openreview.net/pdf?id=8SKgWpZiDL) — NeurIPS 2024 FM4Science workshop.

See [CITATION.cff](CITATION.cff) and [BibTeX](docs/research-en.md#citation). When using a project's code or data, please cite its paper as well.

Report bugs or reproduction problems through [issues](https://github.com/Yu-Yang-Li/StarWhisper/issues). Contributions to code and documentation are welcome; please read the [contribution guide](CONTRIBUTING.md) before opening a pull request.

Code is licensed under [Apache-2.0](LICENSE). Adapted research skills retain their original MIT license and attribution; see [NOTICE](skills/NOTICE.md). Model weights follow their own licenses.
