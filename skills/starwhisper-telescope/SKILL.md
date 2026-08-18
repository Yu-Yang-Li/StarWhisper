---
name: starwhisper-telescope
description: Inspect and explain the StarWhisper Telescope / NGSS observing-agent code. Use when the user asks about NGSS, NINA, night plans, observe_config.json, TNS helpers, or the Communications Engineering 2025 paper. Does not command hardware unless the user explicitly has a live stack.
---

# StarWhisper Telescope

Paper: https://doi.org/10.1038/s44172-025-00520-4  
Code: [`NGSS/`](../../NGSS/README.md)

This is the published observing-agent stack on Nearby Galaxy Supernovae Survey (about 10 amateur-level telescopes). It is not Explore, and it is not Virtual Sitian.

## Default (no hardware)

1. Read `NGSS/README.md` and `observe_config.json`.
2. Explain plan → review → load into NINA → optional target inject.
3. From `NGSS/`:

```bash
uvicorn src.app.app2:app --reload
```

4. List missing prerequisites instead of faking them: NINA, `FMoraes.NINA.SitesPlugin.dll`, x-opstep, FTP, telescope connection.

## Hardware

Only if the user says the live stack is up and asks to run it. Safety interlocks outrank the agent. Do not send UDP/MQTT/NINA commands from a laptop that is not that stack.

## Do not

- Treat a local import check as a night on sky.
- Use Explore's four-strategy table as NGSS production metrics.
- Call TNS helper `Pachong.py` a stable API; it is a legacy public-page helper.

Related: `starwhisper-explore` for decision-boundary research; `starwhisper-allsky` for all-sky photo replan.
