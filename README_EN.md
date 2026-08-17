# StarWhisper

[![GitHub Repo stars](https://img.shields.io/github/stars/Yu-Yang-Li/StarWhisper?style=social)](https://github.com/Yu-Yang-Li/StarWhisper/stargazers)
[![License](https://img.shields.io/github/license/Yu-Yang-Li/StarWhisper)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-Communications%20Engineering-0B1B33)](https://doi.org/10.1038/s44172-025-00520-4)
[![GitHub last commit](https://img.shields.io/github/last-commit/Yu-Yang-Li/StarWhisper)](https://github.com/Yu-Yang-Li/StarWhisper/commits/main)

<p align="center">
  <a href="README.md">中文</a> &nbsp;|&nbsp; English
</p>

<div align="center">

![StarWhisper: AI astrophysicist workflow](https://yu-yang-li.github.io/StarWhisper/assets/starwhisper-hero.jpg)

</div>

**StarWhisper** started as an astronomy language model, then moved through light curves and pulsar candidates into live telescope observing, and now into replayable decision-boundary experiments. Supported by NAOC, Zhejiang Lab, and collaborators. The latest peer-reviewed paper is [StarWhisper Telescope](https://doi.org/10.1038/s44172-025-00520-4) (November 2025).

| Start here | Era |
| --- | --- |
| [`LLM_Data/`](LLM_Data) | 2023–2024 language models |
| [`StarWhisper_LC/`](StarWhisper_LC) · [Pulsar report](https://openreview.net/pdf?id=8SKgWpZiDL) | 2024–2025 time-series and multimodal |
| [`NGSS/`](NGSS) · [Telescope paper](https://doi.org/10.1038/s44172-025-00520-4) | 2025 observing automation |
| [Virtual-GOTTA map](https://yu-yang-li.github.io/StarWhisper/virtual-gotta-map.html) | 2025– embodied telescopes / Sitian |
| Explore below · [`skills/`](skills/README.md) | 2026 decision boundaries and research skills |

---

## Timeline

```mermaid
timeline
    title StarWhisper
    2023 : repository created : astronomy LLM
    2024 : LC preprint : Pulsar NeurIPS workshop : Telescope preprint
    2025 : LC published : Telescope published : Virtual-GOTTA
    2026 : Explore synthetic decision boundary : astronomy research skills
```

| When | Stage | What is public |
| --- | --- | --- |
| 2023.07 | Repository | GitHub `Yu-Yang-Li/StarWhisper` |
| 2023–2024 | Astronomy LLMs | `LLM_Data` (StarWhisper 3 training data); 4.0 weights planned for ModelScope |
| 2024.04 | Light curves | [arXiv:2404.10757](https://arxiv.org/abs/2404.10757) |
| 2024.12 | Pulsar candidates | NeurIPS 2024 FM4Science: [StarWhisper-Pulsar](https://openreview.net/pdf?id=8SKgWpZiDL) |
| 2024.12 | Observing-automation preprint | [arXiv:2412.06412](https://arxiv.org/abs/2412.06412) |
| 2025.02 | Light curves published | [Intelligent Computing](https://spj.science.org/doi/10.34133/icomputing.0110) |
| 2025.11 | Telescope agent published | [Communications Engineering 4, 184](https://doi.org/10.1038/s44172-025-00520-4); code in `NGSS` |
| 2025– | Embodied telescopes / Sitian | [Virtual-GOTTA map](https://yu-yang-li.github.io/StarWhisper/virtual-gotta-map.html) |
| 2026 | Decision boundary | StarWhisper-Explore-v0.2 (synthetic; stable negative result) |
| 2026.08 | Research skills | [`skills/`](skills/README.md), astronomy defaults (ADS / astro-ph / AAS) |

<div align="center">

![StarWhisper architecture: models, telescope agent, Virtual-GOTTA, research skills](https://yu-yang-li.github.io/StarWhisper/assets/starwhisper-architecture.jpg)

</div>

<p align="center"><sub>The current surface: models, the telescope agent, Virtual-GOTTA, and research skills on one line — not four disconnected demos.</sub></p>

---

## 2023–2024 · Astronomy language models

The first problem was whether a model could answer astronomy questions, write code, and use observing knowledge. StarWhisper 3 training data is in `LLM_Data`. Version 4.0 continues that data work; weights are planned for ModelScope.

This stage is “does the model know astronomy”. It does not yet talk to a telescope.

---

## 2024–2025 · Light curves and pulsars

The next step was time-series and multimodal data, not only chat.

**April 2024**: [StarWhisper LC](https://arxiv.org/abs/2404.10757) preprint. **26 February 2025**: published in *Intelligent Computing*. Variable-star classification on Kepler / K2 light curves, including LLM / multimodal / audio variants that need less hand-built features. Test code is in the repo.

<div align="center">

![StarWhisper LC](https://yu-yang-li.github.io/StarWhisper/assets/starwhisper-lc.png)

</div>

**December 2024**: [StarWhisper-Pulsar](https://openreview.net/pdf?id=8SKgWpZiDL) at the NeurIPS 2024 FM4Science workshop — pulsar-candidate classification with multimodal large models.

---

## 2025 · StarWhisper Telescope

Preprint in **December 2024**; published **6 November 2025** in *Communications Engineering*. The [paper](https://doi.org/10.1038/s44172-025-00520-4) describes an end-to-end observing-automation agent on the Nearby Galaxy Supernovae Survey. Code: [`NGSS/`](NGSS).

A night plan rarely survives contact with the sky. Targets of opportunity arrive, weather closes windows, and tracking, focus, cameras, or the dome can fail. The system has to choose, repeatedly: continue, insert, defer, pause safely, or recover and replan.

<div align="center">

![StarWhisper Telescope](https://yu-yang-li.github.io/StarWhisper/assets/starwhisper-telescope.png)

</div>

<div align="center">

![Scheduled plan, three disturbances, observation agent, constrained actions and feedback](https://yu-yang-li.github.io/StarWhisper/assets/starwhisper-observe-loop.jpg)

</div>

<p align="center"><sub>Figure 1. Scheduled plan, three disturbances, observation agent, and constrained actions.</sub></p>

<div align="center">

![StarWhisper demonstration](https://yu-yang-li.github.io/StarWhisper/assets/demo-1.png)

</div>
<div align="center">

![StarWhisper telescope agent interface](https://yu-yang-li.github.io/StarWhisper/assets/demo-2.png)

</div>

This stage shows that an agent can sit on a real survey workflow. It does not yet say when that judgement is trustworthy.

---

## 2025– · Virtual-GOTTA and Sitian

After Telescope, the line is an embodied-intelligence telescope: the model as orchestrator for alerts, station state, plans, data return, and real-time response. See the [Virtual-GOTTA map](https://yu-yang-li.github.io/StarWhisper/virtual-gotta-map.html).

**Sitian** plans 54 one-meter-class wide-field telescopes across Chinese sites, covering about 10,000 square degrees in three colors every 30 minutes. StarWhisper is one candidate path for a “Sitian brain”, not a slide-only platform.

<div align="center">

![Sitian survey concept](https://yu-yang-li.github.io/StarWhisper/assets/sitian.png)

</div>

---

## 2026 · Decision boundary (StarWhisper-Explore)

The published stack can already run automated observing. Explore asks the next question: **when is that judgement trustworthy, what does it stably sacrifice, and when must it refuse.**

Environment `StarWhisper-Explore-v0.2`. Fixed site XingLong, one telescope, six slots per night. Targets, transient arrivals, weather, and device faults are generated from seeds and shared across policies. The agent cannot read future disturbances or edit safety thresholds. Hardware interlocks outrank any suggested action.

This is a **synthetic decision loop**. It is not optical-propagation physics and not a live hardware loop. Credentials, FTP/MQTT endpoints, and raw images stay private.

Four pre-registered comparators: no-intervention, random, deterministic priority, and a rule agent. A positive finding requires, on all three seeds: no active safety violations, invalid-action rate ≤ 1%, survey-completeness drop ≤ 5 percentage points, and either ≥ 20% relative gain in high-value transient follow-up or ≥ 5% gain in scientific utility.

Current synthetic result (90 episodes / policy):

| Policy | Mean utility | Survey completeness | High-value follow-up | Invalid actions | Unsafe attempts blocked |
| --- | ---: | ---: | ---: | ---: | ---: |
| No intervention | 3.9343 | 77.41% | 0.00% | 0 | 122 |
| Random | 2.4027 | 30.19% | 30.37% | 18 | 72 |
| Deterministic priority | 4.3289 | 61.11% | 49.81% | 0 | 0 |
| Rule agent | 4.3996 | 51.67% | 73.33% | 0 | 0 |

<div align="center">

![Pre-registered trade-off: rule agent raises follow-up but misses the survey-completeness floor](https://yu-yang-li.github.io/StarWhisper/assets/goai-metrics-source.png)

</div>

<p align="center"><sub>Figure 2. Policy trade-off under the pre-registered completeness floor. The rule agent has the highest follow-up and sits left of the allowed zone.</sub></p>

The rule agent raises follow-up by 23.52 percentage points over deterministic priority, with only ~1.6% more utility, but completeness falls 9.44 points — past the 5-point tolerance. That is a **stable negative result**: the current marginal-value rule overweights short-term response. The direction is the same on three seeds; a second run matched output hashes. It is not a win.

<div align="center">

![Synthetic environment, de-identified log replay, hardware shadow mode](https://yu-yang-li.github.io/StarWhisper/assets/starwhisper-verification.jpg)

</div>

<p align="center"><sub>Figure 3. Reproduce first, calibrate against reality, then enter hardware shadow mode (suggest only).</sub></p>

Next: an AstroQ / TJO-style constrained scheduler, de-identified logs to calibrate disturbance rates, then the same decision interface in front of timing, weather, and control models. Shadow mode remains advice, not authority.

---

## 2026 · Astronomy research skills

In August 2026, thirteen skills from [tashan-research-skills](https://github.com/TashanGKD/tashan-research-skills) were adapted for astronomy and placed in [`skills/`](skills/README.md). Defaults are NASA ADS, arXiv `astro-ph`, AAS citations, light-curve / spectrum / FITS contracts, and a hard split between synthetic, replay, and hardware.

<div align="center">

![Astronomy research skills matrix](https://yu-yang-li.github.io/StarWhisper/assets/starwhisper-skills-matrix.jpg)

</div>

| Group | Skills |
| --- | --- |
| Literature | paper search, deep research, thesis audit |
| Research design | hypothesis generation, baseline builder, experiment design, statistics |
| Communication | humanization, academic writing, scientific figures, decks |
| Collaboration | citation check, research persona |

```powershell
git clone https://github.com/Yu-Yang-Li/StarWhisper.git
Copy-Item -Recurse .\StarWhisper\skills\giiisp-paper-search-apis "$env:USERPROFILE\.codex\skills\giiisp-paper-search-apis"
```

With no ADS / Giiisp token the skill must dry-run or fall back locally. Skills never command a telescope.

---

## Boundaries

| Fair to say | Not fair to say |
| --- | --- |
| The 2025 observing-automation frame is implemented in NGSS | This repo already owns live hardware interlocks |
| Explore-v0.2 synthetic nights are replayable and hash-stable | The rule agent passed the positive-finding bar |
| Skills can help with ADS search, writing, and citation checks | Skill output is a published result or a discovery |
| Shadow mode may suggest | Suggestions were executed on hardware |

Source code: **Apache-2.0**. Astronomy-adapted skills under [`skills/`](skills/NOTICE.md) keep the upstream MIT license. Base-model weights follow their own licenses.

---

## Citation

If this work is useful, cite the Telescope paper:

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
