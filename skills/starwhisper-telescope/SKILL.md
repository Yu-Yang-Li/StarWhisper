---
name: starwhisper-telescope
description: Inspect the StarWhisper Telescope / NGSS observing-agent stack on disk. Use when the user asks about NGSS, NINA, night plans, observe_config.json, TNS helpers, or the Communications Engineering 2025 paper. Does not command hardware unless the user explicitly has a live stack.
---

# StarWhisper Telescope

Paper: https://doi.org/10.1038/s44172-025-00520-4  
Code: [`NGSS/`](../../NGSS/README.md)

This is the published observing-agent stack on Nearby Galaxy Supernovae Survey. It is not Explore, and it is not Virtual Sitian.

## Default (inspect)

```powershell
python skills/starwhisper-telescope/scripts/inspect_stack.py --json
```

The script prints `observe_config`, FastAPI routes, and which paths are hardware. It never starts uvicorn or sends MQTT/NINA/FTP.

If `NGSS/src/app/app2.py` is missing (sparse clone), it uses bundled `references/api.json`. Live files override the bundle when present.

Start command, only after inspect, from `NGSS/`:

```bash
uvicorn src.app.app2:app --reload
```

Missing prerequisites to list instead of faking: NINA, `FMoraes.NINA.SitesPlugin.dll`, x-opstep, FTP/MQTT, telescope connection.

## Hardware

Only if the user says the live stack is up and asks to run it. Safety interlocks outrank the agent. Do not call `/manipulate_nina/{action}` or `/ftp_transfer` from a laptop that is not that stack.

## Do not

- Treat a local import check as a night on sky.
- Use Explore's four-strategy table as NGSS production metrics.
- Call TNS helper `Pachong.py` a stable API; it is a legacy public-page helper.

Related: `starwhisper-explore` for decision-boundary research; `starwhisper-allsky` for all-sky photo replan.
