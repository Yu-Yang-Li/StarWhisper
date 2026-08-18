---
name: starwhisper-index
description: Route StarWhisper questions to the matching line, folder, paper, and skill. Use when the user asks what StarWhisper is, which folder to open, how LLM/LC/Telescope/Sitian/Explore relate, or which skill to install.
---

# StarWhisper index

Pick one line. Do not mix a published paper, a synthetic Explore table, and hardware control in the same claim.

Run this first:

```powershell
python skills/starwhisper-index/scripts/route.py --query "NGSS 夜计划" --json
```

Then open the matching skill below. Catalog: `catalog.json`.

| Line | When | Open | Skill |
| --- | --- | --- | --- |
| LLM | 问答模型、训练数据、StarWhisper 3/4 | `LLM_Data/` | `starwhisper-llm` |
| LC | Kepler/K2 光变分类 | `StarWhisper_LC/` | `starwhisper-lc` |
| Pulsar | 脉冲星候选 | ACMISLab/StarWhisper-Pulsar | `starwhisper-pulsar` |
| Telescope | NGSS 观测 agent、NINA、夜计划 | `NGSS/` | `starwhisper-telescope` |
| Explore | 决策边界、四策略表、稳定负结果 | `explore/` | `starwhisper-explore` |
| All-sky | 兴隆全天相机 → 重规划 | `AllSky-Camera-XL/` | `starwhisper-allsky` |
| Sparse LC | 稀疏 ZTF/ATLAS 早期分类 | `Early Classification from Sparse Light Curves/` | `starwhisper-sparse-lc` |
| Spectra | 低信噪比恒星光谱 | `Low-SNR-Stellar-Spectra-as-Language/` | `starwhisper-lowsnr-spectra` |
| GOTTA prototype | 真假源原型 | `GOTTA_Prototype/` | `starwhisper-gotta` |
| Sitian | 虚拟司天、超新星时钟 | [SitianClaw](https://github.com/Yu-Yang-Li/SitianClaw) | `starwhisper-sitian` |
| Research writing | ADS、假设、审稿、润色 | `skills/` 他山改编目录 | 对应科研技能 |

Literature and writing skills do not replace NGSS. Native observing skills do not invent papers.

Install every native skill:

```powershell
powershell -File skills/install_native.ps1
```

Set `STARWHISPER_ROOT` if the skills are copied out of this checkout.
