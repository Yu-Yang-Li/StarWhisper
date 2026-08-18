---
name: starwhisper-lc
description: Inventory StarWhisper LC Kepler/K2 variable-star classification test code. Use when the user asks about light curves, Cepheids, RR Lyrae, eclipsing binaries, Conv1D-BiLSTM, Swin Transformer, or the Intelligent Computing 2025 paper.
---

# StarWhisper LC

Paper: https://spj.science.org/doi/10.34133/icomputing.0110  
Code: [`StarWhisper_LC/`](../../StarWhisper_LC/README.md)

## Run

```powershell
python skills/starwhisper-lc/scripts/inventory_code.py
```

Keep the paper's scope: Kepler/K2, mainly Cepheids, RR Lyrae, eclipsing binaries; reported accuracy about 90%. This folder is test code, not a full training reproduction. If `Code/` is absent, say so.

## Do not

- Call a class label a physical mechanism or a new discovery.
- Point this skill at NGSS night plans or Explore decision slots.
- Invent missing training scripts.

Related: `starwhisper-sparse-lc` for ZTF/ATLAS sparse early classification; `starwhisper-pulsar` for pulsar candidates.
