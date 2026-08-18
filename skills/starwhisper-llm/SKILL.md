---
name: starwhisper-llm
description: Inventory StarWhisper 3/4 astronomy language-model data and point to weights. Use when the user asks about LLM_Data, StarWhisper3, ModelScope weights, astronomy Q&A fine-tuning, or StarWhisper 4.0 text cleaning.
---

# StarWhisper LLM

## Where

- Data: [`LLM_Data/`](../../LLM_Data/README.md)
- Weights: [AstroYuYang/StarWhisper3](https://www.modelscope.cn/models/AstroYuYang/StarWhisper3)
- 4.0 weights are not in this repo yet

## Run

```powershell
python skills/starwhisper-llm/scripts/inventory_data.py
```

Cite files that `present` is true. Sparse clones may only have the README. Do not invent a 4.0 checkpoint path.

If the user wants inference, point to ModelScope. `LLM_Data/` is cleaned Q&A text, not telescope logs.

## Do not

- Call the LLM an observing agent.
- Mix StarWhisper LC classifier accuracy into the LLM section.
- Claim 4.0 is released.
