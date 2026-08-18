---
name: starwhisper-explore
description: Explain StarWhisper-Explore-v0.2 decision-boundary results. Use when the user asks about survey completeness vs transient follow-up, the four policies, stable negative result, Xinglong six-slot synthetic nights, or GOAI Explore.
---

# StarWhisper Explore

Spec and table: [`explore/`](../../explore/README.md)

Synthetic only: Xinglong, one telescope, six slots, seeds `11/22/33`. The environment code is not in this repository. Do not wire this skill to NINA.

## Do

1. Read `explore/README.md` and `explore/published_metrics.csv`.
2. Repeat the pre-registered bar before the numbers.
3. State the published result as a **stable negative**: rule agent raises high-value follow-up 23.52 points vs deterministic priority, utility about +1.6%, survey completeness −9.44 points (past the 5-point line).
4. Keep the three verification stages: synthetic → de-identified logs → hardware shadow (suggest only). Public result stops at stage one.

## Do not

- Call this a hardware campaign.
- Drop failed episodes or switch metrics after seeing the table.
- Write "discovered" for a synthetic transient.
