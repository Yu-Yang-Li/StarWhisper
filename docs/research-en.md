# StarWhisper research notes

[Repository home](../README_EN.md) · [中文](research.md) · English · [Skill installation](../skills/README.md)

These notes contain paper figures, experimental results, and details about the available data and code. For project links, see the [home page](../README_EN.md#projects). Run the commands below from the repository root.

## Timeline

| Year | What | Code / weights |
| --- | --- | --- |
| 2023 | repo created; astronomy LLM | [`LLM_Data/`](../LLM_Data), [StarWhisper3](https://www.modelscope.cn/models/AstroYuYang/StarWhisper3) |
| 2024 | light-curve classification, pulsars, Telescope preprint | [`StarWhisper_LC/`](../StarWhisper_LC), [Pulsar code](https://github.com/ACMISLab/StarWhisper-Pulsar) |
| 2025 | LC and Telescope published | [`NGSS/`](../NGSS) |
| 2026 | Virtual Sitian, Explore, the skill pack | [SitianClaw](https://github.com/Yu-Yang-Li/SitianClaw), [`skills/`](../skills/README.md) |

---

## 2023–2024 · Language models

StarWhisper 3 supports astronomy questions and code generation. Training data is in [`LLM_Data/`](../LLM_Data); weights are on ModelScope at [AstroYuYang/StarWhisper3](https://www.modelscope.cn/models/AstroYuYang/StarWhisper3). Popular-science and research text for version 4.0 is still being prepared, with weights planned for ModelScope.

---

## 2024–2025 · Light curves and pulsars

[StarWhisper LC](https://spj.science.org/doi/10.34133/icomputing.0110) classifies variable stars on Kepler / K2 light curves, mainly Cepheids, RR Lyrae, and eclipsing binaries. Besides Conv1D–BiLSTM and Swin Transformer, the paper includes an LLM / multimodal / audio series that needs little hand-built feature work, at about 90% accuracy. Preprint April 2024; published in *Intelligent Computing* on 26 February 2025. Test code: [`StarWhisper_LC/`](../StarWhisper_LC).

![StarWhisper LC](assets/starwhisper-lc.png)

In December 2024, [StarWhisper-Pulsar](https://openreview.net/pdf?id=8SKgWpZiDL) was presented at the NeurIPS 2024 FM4Science workshop: pulsar-candidate classification with multimodal large models. Code: [ACMISLab/StarWhisper-Pulsar](https://github.com/ACMISLab/StarWhisper-Pulsar).

---

## 2025 · StarWhisper Telescope

The [paper](https://doi.org/10.1038/s44172-025-00520-4) was a December 2024 preprint and appeared in *Communications Engineering* on 6 November 2025. The observing-automation agent runs on NGSS, a network of about 10 amateur-level telescopes. Code: [`NGSS/`](../NGSS).

Targets of opportunity, weather changes, or problems with tracking, focus, cameras, or the dome can alter an observing plan. The system then chooses whether to continue, insert a target, defer an observation, pause safely, or replan after recovery.

![Observing loop](assets/goai-observe-loop-source.png)

<p align="center"><sub>Scheduled plan, three disturbances, observing agent, constrained actions and feedback. This is the decision structure shared by Telescope and Explore. It is not an optical simulation and not a hardware screenshot.</sub></p>

![StarWhisper Telescope](assets/starwhisper-telescope.png)

---

## 2026 · Virtual Sitian

Virtual Sitian became a runnable SN Clock system in 2026. Its early science application estimates supernova explosion epochs and selects young-SN candidates. See the [Virtual-GOTTA map](https://yu-yang-li.github.io/StarWhisper/virtual-gotta-map.html) for an overview and [SitianClaw](https://github.com/Yu-Yang-Li/SitianClaw) for installable skills. The real/bogus classification prototype is in [`GOTTA_Prototype/`](../GOTTA_Prototype).

[`snclock/`](../snclock/README.md) contains one historical prediction output: post-explosion ages for 22 TNS sources at their discovery time, given as q16 / q50 / q84 quantiles and split into two tiers by whether the source falls within two days. Only 2 sources are within two days at the conservative q84. The table covers less than the declared screening window, and most rows lack retained input snapshots; the data notes explain these limitations. Read it with [`starwhisper-snclock`](../skills/starwhisper-snclock/SKILL.md):

```powershell
python .\skills\starwhisper-snclock\scripts\screen_snclock.py rank --top 10
python .\skills\starwhisper-snclock\scripts\screen_snclock.py audit
```

An age estimate is not a spectroscopic classification, and every source in the table is already on TNS, so nothing here is a new discovery.

Sitian plans 54 one-meter-class wide-field telescopes at Chinese sites, covering about 10,000 square degrees in three colors every 30 minutes. StarWhisper is one candidate approach to Sitian's data analysis and observing automation.

![Sitian](assets/sitian.png)

Other work from 2026 includes:

- Xinglong all-sky camera [`AllSky-Camera-XL/`](../AllSky-Camera-XL): from a full-sky image to a replanned sequence
- Cloud nowcasting with Himawari: a private repository using 2022–2025 archived nights; this phase ended in August 2026 and has not become an operational forecast
- Early classification on sparse ZTF / ATLAS light curves: [`Early Classification from Sparse Light Curves/`](../Early%20Classification%20from%20Sparse%20Light%20Curves). Read the checked results with [`starwhisper-varlen`](../skills/starwhisper-varlen/SKILL.md); 50-obs pretrain accuracy cannot be reported as the 3–30-obs main result
- Low-SNR stellar spectra: [`Low-SNR-Stellar-Spectra-as-Language/`](../Low-SNR-Stellar-Spectra-as-Language) ([Jared-web03](https://github.com/Jared-web03/Low-SNR-Stellar-Spectra-as-Language))

---

## 2026 · Explore

Telescope supports automated survey observations. Explore compares the benefits of observing policies, their effects on survey completeness, and the conditions that require a safe pause.

The experimental setup, action set, and metrics table are in [`explore/`](../explore/README.md). `StarWhisper-Explore-v0.2` uses a synthetic environment: Xinglong, one telescope, six slots per night. The agent cannot see future disturbances or change safety thresholds. The experiment does not simulate optical propagation or connect to live telescopes, and the environment code is not included in this repository.

The comparators were fixed in advance: no intervention, random, deterministic priority, and a rule agent. A positive finding needs all three seeds to pass: no active safety violations, invalid-action rate ≤ 1%, survey-completeness drop ≤ 5 percentage points, and either ≥ 20% relative gain in high-value transient follow-up or ≥ 5% gain in scientific utility.

90 episodes / policy:

| Policy | Mean utility | Survey completeness | High-value follow-up | Invalid actions | Unsafe attempts blocked |
| --- | ---: | ---: | ---: | ---: | ---: |
| No intervention | 3.9343 | 77.41% | 0.00% | 0 | 122 |
| Random | 2.4027 | 30.19% | 30.37% | 18 | 72 |
| Deterministic priority | 4.3289 | 61.11% | 49.81% | 0 | 0 |
| Rule agent | 4.3996 | 51.67% | 73.33% | 0 | 0 |

![Policy trade-off](assets/goai-metrics-source.png)

Against deterministic priority, the rule agent raises follow-up by 23.52 percentage points and utility by about 1.6%, while completeness falls 9.44 points, exceeding the allowed 5-point drop. All three seeds show the same direction, and recorded repeat runs matched hashes. This is a stable negative result: higher transient follow-up comes with a loss of survey completeness beyond the permitted range.

These commands recalculate the verdict; use `--csv` to supply your own metrics table:

```powershell
python .\skills\starwhisper-explore\scripts\eval_gate.py bar
python .\skills\starwhisper-explore\scripts\eval_gate.py gate --agent rule_agent
```

![Verification path](assets/goai-verification-source.png)

<p align="center"><sub>Reproduce in a synthetic environment first, calibrate on de-identified logs, then hardware shadow mode (suggest only). The public result stops at stage one.</sub></p>

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

Citation information is also available in [`CITATION.cff`](../CITATION.cff). Code is licensed under Apache-2.0. Adapted research skills retain the MIT license; see [`NOTICE.md`](../skills/NOTICE.md). Model weights follow their own licenses.
