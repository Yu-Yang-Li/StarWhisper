"""Inspect NGSS on disk, or print the bundled contract. Never HTTP, NINA, FTP, or MQTT."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

SKILL = Path(__file__).resolve().parents[1]
API = json.loads((SKILL / "references" / "api.json").read_text(encoding="utf-8"))
EXAMPLE = json.loads((SKILL / "references" / "observe_config.example.json").read_text(encoding="utf-8"))


def repo_root() -> Path | None:
    env = os.environ.get("STARWHISPER_ROOT")
    if env:
        p = Path(env)
        if (p / "skills" / "starwhisper-index" / "SKILL.md").exists():
            return p
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "skills" / "starwhisper-index" / "SKILL.md").exists() and (parent / "README.md").exists() and (
            (parent / "explore").is_dir() or (parent / "NGSS").is_dir()
        ):
            return parent
    return None


def live_routes(app_py: Path) -> list[dict]:
    text = app_py.read_text(encoding="utf-8")
    found = re.findall(r'@app\.(get|post|put|delete)\("([^"]+)"', text)
    by_path = {row["path"]: row for row in API["routes"]}
    out = []
    for method, path in found:
        known = by_path.get(path, {})
        out.append({"method": method.upper(), "path": path, "class": known.get("class", "unknown")})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect StarWhisper Telescope / NGSS without commanding hardware.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    root = repo_root()
    ngss = (root / "NGSS") if root else None
    live_config = ngss / "observe_config.json" if ngss else None
    app_py = ngss / "src" / "app" / "app2.py" if ngss else None
    payload = {
        "mode": "inspect",
        "repo_root": str(root) if root else None,
        "ngss_present": bool(ngss and ngss.is_dir()),
        "observe_config": json.loads(live_config.read_text(encoding="utf-8")) if live_config and live_config.exists() else EXAMPLE,
        "observe_config_source": "NGSS/observe_config.json" if live_config and live_config.exists() else "bundled example",
        "start": API["start"],
        "cwd": API["cwd"],
        "routes": live_routes(app_py) if app_py and app_py.exists() else API["routes"],
        "routes_source": "NGSS/src/app/app2.py" if app_py and app_py.exists() else "bundled api.json",
        "hardware_blocked": [row["path"] for row in API["routes"] if row["class"] == "hardware"],
        "safety": "inspect only; this script does not start uvicorn, MQTT, FTP, or NINA",
    }
    if args.json:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        print()
        return 0
    print("mode:", payload["mode"])
    print("observe_config_source:", payload["observe_config_source"])
    print("observe_config:", json.dumps(payload["observe_config"], ensure_ascii=False))
    print("start:", f"cd {payload['cwd']} && {payload['start']}")
    for row in payload["routes"]:
        print(f"{row.get('method', 'GET'):6} {row['path']:40} {row.get('class', '')}")
    print("hardware_blocked:", ", ".join(payload["hardware_blocked"]))
    print(payload["safety"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
