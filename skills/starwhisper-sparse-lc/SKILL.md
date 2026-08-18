---
name: starwhisper-sparse-lc
description: Inspect the sparse ZTF/ATLAS early-classification benchmark and list the published reproduction steps. Use when the user asks about varlen 3-30 observations, seven variability classes, XGBoost/Transformer/LSTM comparison, or Early Classification from Sparse Light Curves.
---

# Sparse light-curve early classification

Folder: [`Early Classification from Sparse Light Curves/`](../../Early%20Classification%20from%20Sparse%20Light%20Curves/README.md)  
Weights: https://huggingface.co/castor0705/sparse-lc-early-classification

```powershell
python skills/starwhisper-sparse-lc/scripts/inspect_pipeline.py
```

Main **varlen** setting: 3–30 observations per segment, seven merged classes, 75/10/15 split, `random_state = 42`.

Print the bundled steps; only run training scripts if that folder is actually present and the user asked to reproduce. Do not quote LC Kepler/K2 90% accuracy as this benchmark's number.

A test-set metric is not an explosion-time or a discovery.
