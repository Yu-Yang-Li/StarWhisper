# StarWhisper 5.0

[![GitHub Repo stars](https://img.shields.io/github/stars/Yu-Yang-Li/StarWhisper?style=social)](https://github.com/Yu-Yang-Li/StarWhisper/stargazers)
[![GitHub Code License](https://img.shields.io/github/license/Yu-Yang-Li/StarWhisper)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/Yu-Yang-Li/StarWhisper)](https://github.com/Yu-Yang-Li/StarWhisper/commits/main)

<p align="center">
  <a href="README.md">中文</a> &nbsp;|&nbsp; English
</p>

**StarWhisper** is an open-source astronomy and AI4S model-and-agent project. Supported by NAOC, ZheJiang Lab, and collaborators, it has evolved from astronomical language, time-series, and multimodal models into **StarWhisper Telescope** and **Virtual-GOTTA**: a workflow where large models connect to real observing tasks, station information, telescope status, and real-time scientific response.

This README has been reorganized around the presentation *AI-driven science education and research*: from AI for scientific research, to embodied-intelligence telescopes, to a global transient-source telescope data network and early supernova candidate alerts.

- [Virtual-GOTTA interactive roadmap](https://yu-yang-li.github.io/StarWhisper/virtual-gotta-map.html)
- [PDF / PPT source notes](https://yu-yang-li.github.io/StarWhisper/virtual-gotta-source.html)
- [Latest published paper: StarWhisper Telescope](https://doi.org/10.1038/s44172-025-00520-4), *Communications Engineering* 4, 184 (2025)

---

## Project Positioning

| Direction | Current role | Repository content |
| --- | --- | --- |
| Astronomy large language models | Domain models for science communication, research QA, coding, and observing knowledge | `LLM_Data`, training data, model notes |
| Time-series and multimodal models | Light-curve classification, pulsar identification, and astronomical image understanding | StarWhisper LC, StarWhisper Pulsar, examples |
| StarWhisper Telescope | Agent framework for end-to-end astronomical-observation automation | `NGSS` nearby-galaxy survey code |
| Virtual-GOTTA | Embodied-intelligence transformation for scientific telescopes and observing workflows | `docs/virtual-gotta-map.html`, roadmap and source notes |

StarWhisper is not only a single model demo. Its goal is to connect astronomical knowledge, data processing, observing plans, telescope control, and real-time scientific judgement into one extensible agent workflow.

---

## Virtual-GOTTA: AI-driven Virtual Sitian Project

The Virtual-GOTTA presentation pushes StarWhisper Telescope toward an embodied-intelligence telescope system. Large models act as the interaction and orchestration layer connecting alerts, station information, telescope status, observing plans, data return, and real-time response.

Core workflow:

1. **Alert and science-target intake**: transient sources, early supernova candidates, and nearby-galaxy survey tasks.
2. **Station and telescope state awareness**: site information, device availability, weather, and observing windows.
3. **Planning and execution loop**: model decisions are translated into executable observing plans and then linked back to returned data and status.
4. **Real-time response and candidate filtering**: support early alerts for candidates within one day of explosion and a global transient-source telescope data network.
5. **Science education and research platform**: combine AI research assistants, virtual scientists, and real telescope tasks into a reusable AI4S case study.

[Open the Virtual-GOTTA interactive roadmap](https://yu-yang-li.github.io/StarWhisper/virtual-gotta-map.html)

---

## Open Modules and Papers

1. **StarWhisper 4.0 data and training enhancements**  
   Refined astronomical physics, coding, and agent capabilities through cleaned scientific and popular-science datasets. The StarWhisper 3 training data is open-sourced under `LLM_Data`; StarWhisper 4.0 weights will be released on ModelScope.

2. **[StarWhisper Pulsar](https://openreview.net/pdf?id=8SKgWpZiDL)**  
   A technical report on a SOTA multimodal large model for pulsar identification.

3. **[StarWhisper LC](https://spj.science.org/doi/epdf/10.34133/icomputing.0110)**  
   A light-curve classification method based on transfer learning and large models. Test code related to the paper is uploaded.

   <div align="center"><img src="example/StarWhisper LC.png" width="680"/></div>

4. **[StarWhisper Telescope](https://doi.org/10.1038/s44172-025-00520-4)**  
   Published in *Communications Engineering* 4, 184 (2025). The paper introduces an AI agent framework for end-to-end astronomical-observation automation and demonstrates it in the Near-Neighbor Galaxy Survey System. Code is open-sourced in the `NGSS` directory.

   <div align="center"><img src="example/Starwhisper Telescope.png" width="680"/></div>

5. **Virtual-GOTTA / StarWhisper 5.0+**  
   Based on the presentation *AI-driven science education and research*, StarWhisper 5.0+ targets embodied intelligence for scientific telescopes by integrating large models, station networks, telescope status, real-time observing, and scientific alerts.

---

## Demonstration

<div align="center"><img src="example/图片1.png" width="680"/></div>
<div align="center"><img src="example/图片2.png" width="680"/></div>

---

## Sitian Project

**Sitian** is a major time-domain astronomical infrastructure proposed by Chinese astronomers. Phase I plans to deploy 54 wide-field 1-meter telescopes across multiple observing sites in China, forming a multi-band monitoring network that can complete high-precision three-color surveys of about 10,000 square degrees every 30 minutes.

Sitian aims to discover new celestial objects and phenomena including extreme energy bursts, gravitational-wave electromagnetic counterparts, exoplanets, and solar-system bodies. StarWhisper explores how large models, agents, and astronomical tools can become the AI core of this observing network.

<div align="center"><img src="example/sitian.png" width="680"/></div>

---

## License

- Source code: **Apache-2.0 License**.
- Qwen Chat model weights and other base models follow their respective licenses.

---

## Roadmap

### Large Language Models: Science Communication and Research Assistance

- Optimize the ratio between general and domain-specific data during SFT to mitigate catastrophic forgetting.
- Improve performance through reinforcement learning with human feedback.
- Build an astronomical knowledge graph to reduce hallucinations.
- Strengthen summarization, code generation, observing-task understanding, and paper-assistance capabilities.

### Multimodal Models: Research Tools

- Release more multimodal fine-tuning weights.
- Explore astronomical image generation, recognition, and quality-control tasks.
- Link evidence from light curves, images, spectra, and text.

### Observation Agents: Sitian Brain

- Boost coding and tool-use capabilities in astronomy.
- Validate human-machine interaction agents on MiniSiTian / Sitian prototypes.
- Integrate professional tools such as ASTROLABE and CASA.
- Validate StarWhisper as a candidate technical path for the Sitian Brain.

---

## Citation

If this work is useful to you, please cite the latest published paper:

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

![Star History Chart](https://star-history.dera.page/svg?repos=Yu-Yang-Li/StarWhisper&type=Date)
