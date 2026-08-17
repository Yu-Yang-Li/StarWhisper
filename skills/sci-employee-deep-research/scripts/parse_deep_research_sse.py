#!/usr/bin/env python3
"""Parse Deep Research SSE logs into a compact evidence summary."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def iter_events(text: str):
    event = None
    data_parts = []
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if not line:
            if event or data_parts:
                data = "\n".join(data_parts)
                yield event, data
            event = None
            data_parts = []
            continue
        if line.startswith("event:"):
            event = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data_parts.append(line.split(":", 1)[1].strip())
    if event or data_parts:
        yield event, "\n".join(data_parts)


def load_json(data: str):
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None


def parse_sse(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    phases = []
    keywords = []
    references = []
    private_search_summary = None
    failures = []
    answer_chunks = []
    done = None
    usage = None

    for event, data in iter_events(text):
        payload = load_json(data)
        if event == "phase" and isinstance(payload, dict):
            phases.append(payload)
        elif event == "keywords" and isinstance(payload, dict):
            keywords = payload.get("keywords") or []
        elif event == "private_search_hit" and isinstance(payload, dict):
            if payload.get("ok") is False or payload.get("error"):
                failures.append(payload)
        elif event == "private_search_summary" and isinstance(payload, dict):
            private_search_summary = payload
            failures.extend(payload.get("failures") or [])
        elif event == "references" and isinstance(payload, dict):
            references = payload.get("references") or []
        elif event == "delta" and isinstance(payload, dict):
            answer_chunks.append(payload.get("content") or "")
        elif event == "done" and isinstance(payload, dict):
            done = payload
        elif event == "usage" and isinstance(payload, dict):
            usage = payload.get("usage") or payload

    answer_text = "".join(answer_chunks)
    total_results = None
    if private_search_summary:
        total_results = private_search_summary.get("totalResults")

    if done and done.get("ok") and references:
        answer_status = "complete"
    elif done and done.get("answer") and not references:
        answer_status = "clarification_or_no_evidence"
    elif references and answer_text:
        answer_status = "incomplete"
    elif references:
        answer_status = "references_only"
    elif total_results == 0:
        answer_status = "no_evidence"
    else:
        answer_status = "unknown"

    return {
        "path": str(path),
        "events": {
            "phase_count": len(phases),
            "has_done": done is not None,
            "has_usage": usage is not None,
        },
        "phases": phases,
        "keywords": keywords,
        "private_search": {
            "totalResults": total_results,
            "failure_count": len(failures),
            "failures": failures[:10],
        },
        "references_count": len(references),
        "first_references": [
            {
                "title": ref.get("title"),
                "url": ref.get("url"),
                "sourceApi": ref.get("sourceApi"),
                "sourceType": ref.get("sourceType"),
            }
            for ref in references[:10]
        ],
        "answer_status": answer_status,
        "answer_chars": len(answer_text or (done or {}).get("answer", "")),
        "answer_preview": re.sub(r"\s+", " ", (answer_text or (done or {}).get("answer", ""))[:500]).strip(),
        "usage": usage,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sse_log", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    summary = parse_sse(args.sse_log)
    output = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
