# StarWhisper

[![GitHub Repo stars](https://img.shields.io/github/stars/Yu-Yang-Li/StarWhisper?style=social)](https://github.com/Yu-Yang-Li/StarWhisper/stargazers)
[![License](https://img.shields.io/github/license/Yu-Yang-Li/StarWhisper)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-Communications%20Engineering-0B1B33)](https://doi.org/10.1038/s44172-025-00520-4)
[![GitHub last commit](https://img.shields.io/github/last-commit/Yu-Yang-Li/StarWhisper)](https://github.com/Yu-Yang-Li/StarWhisper/commits/main)

<p align="center">
  <a href="README.md">中文</a> &nbsp;|&nbsp; English
</p>

<div align="center">

![StarWhisper](https://yu-yang-li.github.io/StarWhisper/assets/starwhisper-hero.jpg)

</div>

StarWhisper is open work on astronomy models and observing, supported by NAOC, Zhejiang Lab, and collaborators. It started in 2023 as a language model, then moved through light curves, pulsar candidates, and live telescope observing. The 2026 line is Virtual Sitian (SN Clock).

The latest peer-reviewed paper is [StarWhisper Telescope](https://doi.org/10.1038/s44172-025-00520-4) (November 2025).

| Year | What | Code / weights |
| --- | --- | --- |
| 2023 | repo created; astronomy LLM | [`LLM_Data/`](LLM_Data), [StarWhisper3](https://www.modelscope.cn/models/AstroYuYang/StarWhisper3) |
| 2024 | LC preprint; Pulsar; Telescope preprint | [`StarWhisper_LC/`](StarWhisper_LC), [Pulsar code](https://github.com/ACMISLab/StarWhisper-Pulsar) |
| 2025 | LC and Telescope published | [`NGSS/`](NGSS) |
| 2026 | Virtual Sitian, Xinglong all-sky camera, Explore, research skills | snclock, [SitianClaw](https://github.com/Yu-Yang-Li/SitianClaw), [`skills/`](skills/README.md) |

---

## 2023–2024 · Language models

The first piece was a domain model that can answer astronomy questions and write code. StarWhisper 3 training data is in [`LLM_Data/`](LLM_Data); weights are on ModelScope at [AstroYuYang/StarWhisper3](https://www.modelscope.cn/models/AstroYuYang/StarWhisper3). Version 4.0 is still cleaning data.

---

## 2024–2025 · Light curves and pulsars

[StarWhisper LC](https://arxiv.org/abs/2404.10757) went up as a preprint in April 2024 and was published in *Intelligent Computing* on 26 February 2025. Variable-star classification on Kepler / K2. Test code: [`StarWhisper_LC/`](StarWhisper_LC).

<div align="center">

![StarWhisper LC](https://yu-yang-li.github.io/StarWhisper/assets/starwhisper-lc.png)

</div>

In December 2024, [StarWhisper-Pulsar](https://openreview.net/pdf?id=8SKgWpZiDL) was presented at the NeurIPS 2024 FM4Science workshop. Code: [ACMISLab/StarWhisper-Pulsar](https://github.com/ACMISLab/StarWhisper-Pulsar).

---

## 2025 · StarWhisper Telescope

The [paper](https://doi.org/10.1038/s44172-025-00520-4) was a December 2024 preprint and appeared in *Communications Engineering* on 6 November 2025. It is an observing-automation agent on the Nearby Galaxy Supernovae Survey. Code: [`NGSS/`](NGSS).

A night plan rarely survives the night. Targets of opportunity arrive, weather closes windows, and tracking, focus, cameras, or the dome can fail. The system has to choose, repeatedly: continue, insert, defer, pause safely, or recover and replan.

<div align="center">

![StarWhisper Telescope](https://yu-yang-li.github.io/StarWhisper/assets/starwhisper-telescope.png)

</div>

<div align="center">

![Observing loop](https://yu-yang-li.github.io/StarWhisper/assets/starwhisper-observe-loop.jpg)

</div>

<div align="center">

![Demo](https://yu-yang-li.github.io/StarWhisper/assets/demo-1.png)

</div>

---

## 2026 · Virtual Sitian

The Virtual Sitian code is in the snclock (SN Clock) repo. That is 2026 work, not something already finished in the 2025 Telescope paper. Public sketch: [Virtual-GOTTA map](https://yu-yang-li.github.io/StarWhisper/virtual-gotta-map.html). Workflow skills: [SitianClaw](https://github.com/Yu-Yang-Li/SitianClaw). Real/bogus prototype: [`GOTTA_Prototype/`](GOTTA_Prototype).

Sitian plans 54 one-meter-class wide-field telescopes at Chinese sites, covering about 10,000 square degrees in three colors every 30 minutes. StarWhisper is one candidate path for a “Sitian brain”.

<div align="center">

![Sitian](https://yu-yang-li.github.io/StarWhisper/assets/sitian.png)

</div>

Also in 2026:

- [`Low-SNR-Stellar-Spectra-as-Language/`](Low-SNR-Stellar-Spectra-as-Language): low-SNR stellar spectra; sibling repo at [Jared-web03](https://github.com/Jared-web03/Low-SNR-Stellar-Spectra-as-Language)
- [`AllSky-Camera-XL/`](AllSky-Camera-XL): Xinglong all-sky camera, from image to a replanned sequence
- [`Early Classification from Sparse Light Curves/`](Early%20Classification%20from%20Sparse%20Light%20Curves): early classification on sparse ZTF / ATLAS light curves

Xinglong all-sky camera × Himawari cloud nowcasting is in a private repo. The main line closed in August 2026 on 2022–2025 archived nights. It is not an operational forecast.

---

## 2026 · Explore

Telescope can already run automated observing. Explore asks when that judgement is trustworthy, what it gives up, and when it must refuse.

`StarWhisper-Explore-v0.2` is a synthetic environment: Xinglong, one telescope, six slots per night. Targets, transients, weather, and device faults are generated from seeds and shared across policies. The agent cannot see future disturbances or edit safety thresholds. This is not optical-propagation physics and not a live hardware loop.

The comparators were fixed in advance: no intervention, random, deterministic priority, and a rule agent. A positive finding needs, on all three seeds, no active safety violations, invalid-action rate ≤ 1%, survey-completeness drop ≤ 5 percentage points, and either ≥ 20% relative gain in high-value transient follow-up or ≥ 5% gain in scientific utility.

Result, 90 episodes / policy:

| Policy | Mean utility | Survey completeness | High-value follow-up | Invalid actions | Unsafe attempts blocked |
| --- | ---: | ---: | ---: | ---: | ---: |
| No intervention | 3.9343 | 77.41% | 0.00% | 0 | 122 |
| Random | 2.4027 | 30.19% | 30.37% | 18 | 72 |
| Deterministic priority | 4.3289 | 61.11% | 49.81% | 0 | 0 |
| Rule agent | 4.3996 | 51.67% | 73.33% | 0 | 0 |

<div align="center">

![Policy trade-off](https://yu-yang-li.github.io/StarWhisper/assets/goai-metrics-source.png)

</div>

Against deterministic priority, the rule agent raises follow-up by 23.52 percentage points and utility by about 1.6%, while completeness falls 9.44 points — past the 5-point line. Same direction on three seeds; a second run matched hashes. That is a stable negative result, not a win.

---

## 2026 · Research skills

In August, 13 skills from [tashan-research-skills](https://github.com/TashanGKD/tashan-research-skills) were adapted for astronomy and put in [`skills/`](skills/README.md). Literature defaults to NASA ADS and arXiv `astro-ph`. With no token they dry-run; they do not invent hits or command a telescope.

```powershell
Copy-Item -Recurse .\skills\giiisp-paper-search-apis "$env:USERPROFILE\.codex\skills\giiisp-paper-search-apis"
```

Source code is Apache-2.0. The adapted skills under [`NOTICE.md`](skills/NOTICE.md) stay MIT. Model weights follow their own licenses.

---

## Citation

```BibTeX
@article{wang2025starwhisper,
  title={StarWhisper Telescope: an AI framework for automating end-to-end astronomical observations},
  author={Wang, Cunshi and Zhang, Yu and Li, Yuyang and Hu, Xinjie and Mao, Yiming and Chen, Xunhao and Du, Pengliang and Wang, Rui and Wu, Ying and Yang, Hang and Li, Yansong and Wang, Beichuan and Mu, Haiyang and Chen, Xiaohan and He, Shunxuan and Mo, Hao and Zhang, Liyue and Du, Lin and Zhao, Yunning and Tian, Jianfeng and Ge, Liang and Mao, Yongna and Li, Shengming and Wang, Zheng and Lu, Xiaomeng and Zou, Jinhang and Huang, Yang and Sun, Ningchen and Zheng, Jie and He, Min and Bai, Yu and Jin, Junjie and Wu, Hong and Liu, Jifeng},
  journal={Communications Engineering},
  volume={4},
  pages={184},
  year={2025},
  doi={10.1038/s44172-025-00520-4},
  url={https://doi.org/10.1038/s44172-025-00520-4}
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Yu-Yang-Li/StarWhisper&type=Date)](https://star-history.com/#Yu-Yang-Li/StarWhisper&Date)
