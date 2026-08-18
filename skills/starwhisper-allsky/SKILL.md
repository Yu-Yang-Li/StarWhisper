---
name: starwhisper-allsky
description: Inspect or run the Xinglong all-sky camera photo-to-replan pipeline. Use when the user has a full-sky image and wants a mask, deferred targets, or a NINA target set, or asks about AllSky-Camera-XL.
---

# StarWhisper all-sky replan

Code: [`AllSky-Camera-XL/`](../../AllSky-Camera-XL/README.md)  
Packaged skill: `AllSky-Camera-XL/skill/photo-to-replan/SKILL.md`

## Default (inspect)

```powershell
python skills/starwhisper-allsky/scripts/inspect_pipeline.py
```

## Run the pipeline

Only with a raw all-sky jpg whose name contains Beijing time (`YYYY_MM_DD_HH_MM_SS.jpg`):

```powershell
python skills/starwhisper-allsky/scripts/inspect_pipeline.py --image /abs/path/to/YYYY_MM_DD_HH_MM_SS.jpg
```

Then read `output/<image_stem>/pipeline_report.json`. Pipeline success means the report, scenario, schedule, and `.ninaTargetSet` exist. Whether future slots were filled is a separate check in `deferred_targets.json`.

This writes a replanned sequence file. It does not move a telescope.

## Do not

- Feed a non-all-sky science frame into this pipeline.
- Treat the NINA target set as an executed night.
- Merge this with Explore synthetic weather/device draws.
