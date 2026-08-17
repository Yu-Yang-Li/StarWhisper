from pathlib import Path
import re

for p in sorted(Path("skills").glob("*/SKILL.md")):
    text = p.read_text(encoding="utf-8")
    m = re.match(r"---\n(.*?)\n---", text, re.S)
    if not m:
        print("NO FM", p)
        continue
    fm = m.group(1)
    desc = ""
    lines = fm.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("description:"):
            rest = lines[i][len("description:") :].strip()
            if rest in {"|", ">", ">-"}:
                i += 1
                buf = []
                while i < len(lines) and (lines[i].startswith(" ") or lines[i].startswith("\t")):
                    buf.append(lines[i].strip())
                    i += 1
                desc = " ".join(buf)
            else:
                desc = rest.strip().strip('"').strip("'")
            break
        i += 1
    flag = " OVER" if len(desc) > 1024 else ""
    print(f"{len(desc):4d}{flag}  {p.parent.name}")
    print(
        "     astronomy.md",
        (p.parent / "astronomy.md").exists(),
        "overlay",
        "## StarWhisper astronomy overlay" in text,
    )
