---
name: starwhisper-llm
description: Work with StarWhisper 3/4 astronomy language-model data and weights. Use when the user asks about LLM_Data, StarWhisper3, ModelScope weights, astronomy Q&A fine-tuning, or StarWhisper 4.0 text cleaning.
---

# StarWhisper LLM

## Where

- Data: [`LLM_Data/`](../../LLM_Data/README.md)
- Weights: [AstroYuYang/StarWhisper3](https://www.modelscope.cn/models/AstroYuYang/StarWhisper3)
- 4.0 weights are not in this repo yet

## Do

1. Treat `astro_cn_*.json`, `Astro_en.json`, and `Physic.json` as cleaned Q&A text, not telescope logs.
2. Cite the files that actually exist. Do not invent a 4.0 checkpoint path.
3. If the user wants inference, point to ModelScope and say the local `LLM_Data/` folder is training text only.

## Do not

- Call the LLM an observing agent.
- Mix StarWhisper LC classifier accuracy into the LLM section.
- Claim 4.0 is released.
