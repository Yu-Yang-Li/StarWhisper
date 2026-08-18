---
name: starwhisper-explore
description: Report StarWhisper-Explore-v0.2 decision-boundary results from the published table. Use when the user asks about survey completeness vs transient follow-up, the four policies, stable negative result, Xinglong six-slot synthetic nights, or GOAI Explore.
---

# StarWhisper Explore

Spec: [`explore/`](../../explore/README.md)

Synthetic only: Xinglong, one telescope, six slots, seeds `11/22/33`. The environment code is not in this repository. Do not wire this skill to NINA.

## Run

```powershell
python skills/starwhisper-explore/scripts/report_metrics.py --json
```

The script reads `explore/published_metrics.csv` when the checkout is present, otherwise the bundled copy in `references/`. It does not simulate nights or reproduce SHA-256 hashes.

## Then

1. Repeat the pre-registered bar before the numbers (no safety violation; invalid actions ≤ 1%; completeness drop ≤ 5 pp vs strongest non-agent baseline; plus follow-up +20% or utility +5%).
2. State the published result as a **stable negative**: rule agent raises high-value follow-up 23.52 points vs deterministic priority, utility about +1.6%, survey completeness −9.44 points (past the 5-point line).
3. Keep the three verification stages: synthetic → de-identified logs → hardware shadow (suggest only). Public result stops at stage one.

## Do not

- Call this a hardware campaign.
- Drop failed episodes or switch metrics after seeing the table.
- Write "discovered" for a synthetic transient.
- Implement a fake environment that claims the original hashes.
