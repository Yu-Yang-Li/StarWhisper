---
name: starwhisper-sitian
description: Route Virtual Sitian / SN Clock / GOTTA work to SitianClaw skills. Use when the user asks about 虚拟司天, supernova clock, explosion epoch, young SN candidates, forced photometry, or GOTTA.
---

# Virtual Sitian

The runnable 2026 system is not at this repo root.

```powershell
python skills/starwhisper-sitian/scripts/route.py --query "超新星时钟" --json
```

- Map: https://yu-yang-li.github.io/StarWhisper/virtual-gotta-map.html
- Skills and scripts: https://github.com/Yu-Yang-Li/SitianClaw
- Real/bogus prototype in this repo: `GOTTA_Prototype/` (`starwhisper-gotta`)

Install and run the printed `snc-*` skills in SitianClaw. Do not reimplement them in StarWhisper.

A clock estimate is not a spectroscopic classification. A GOTTA prototype score is not a broker alert.
