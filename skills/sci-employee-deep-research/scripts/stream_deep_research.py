#!/usr/bin/env python3
"""Forward Deep Research SSE events as progress JSONL.

Each stdout line is a standalone JSON object. A web backend can forward these
lines as Server-Sent Events without waiting for the final answer.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator


DEFAULT_ENDPOINT = "http://123.56.218.60:18000/api/research/ask"
AUTH_URL = "https://giiisp.com/#/mcp/authenticate"
INTERFACE_UNAVAILABLE_ACTION = (
    "Deep Research 接口不可用或未配置；请检查服务地址/网络/后端状态。"
    f" 如果失败来自 Giiisp 认证或 key 过期，请到 {AUTH_URL} 申请或刷新认证后重试。"
)

STARTED_AT = time.monotonic()


def emit(event: dict[str, Any]) -> None:
    event.setdefault("ts", datetime.now().astimezone().isoformat(timespec="milliseconds"))
    event.setdefault("elapsed_ms", int((time.monotonic() - STARTED_AT) * 1000))
    print(json.dumps(event, ensure_ascii=False, separators=(",", ":")), flush=True)


def load_json(data: str) -> Any:
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None


def iter_sse_lines(lines: Iterable[str]) -> Iterator[tuple[str | None, str]]:
    event = None
    data_parts: list[str] = []
    for raw in lines:
        line = raw.rstrip("\r\n")
        if not line:
            if event or data_parts:
                yield event, "\n".join(data_parts)
            event = None
            data_parts = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data_parts.append(line.split(":", 1)[1].strip())
    if event or data_parts:
        yield event, "\n".join(data_parts)


def first_references(references: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    return [
        {
            "title": ref.get("title"),
            "url": ref.get("url"),
            "sourceApi": ref.get("sourceApi"),
            "sourceType": ref.get("sourceType"),
        }
        for ref in references[:limit]
    ]


def answer_status(references: list[dict[str, Any]], answer_text: str, done: dict[str, Any] | None, total_results: int | None) -> str:
    if done and done.get("ok") and references:
        return "complete"
    if done and done.get("answer") and not references:
        return "clarification_or_no_evidence"
    if references and answer_text:
        return "incomplete"
    if references:
        return "references_only"
    if total_results == 0:
        return "no_evidence"
    if answer_text:
        return "clarification_or_no_evidence"
    return "unknown"


def user_action_for_status(status: str, failures: list[Any]) -> str:
    if status == "interface_unavailable":
        return INTERFACE_UNAVAILABLE_ACTION
    if status == "incomplete":
        return "先展示已返回 references 和 answer 片段；继续等待、收窄问题或改走候选论文过滤"
    if status == "references_only":
        return "先展示 references；继续等待 answer 或把候选论文交给报告整理"
    if status in {"no_evidence", "clarification_or_no_evidence"}:
        return "不要包装成研究报告；收窄问题、换关键词或先走论文检索"
    if failures:
        return "保留失败 endpoint 和关键词；必要时提示用户完成 Giiisp 认证或改用开放源回退"
    return ""


def infer_health_url(endpoint_url: str) -> str:
    parsed = urllib.parse.urlparse(endpoint_url)
    if not parsed.scheme or not parsed.netloc:
        return ""
    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, "/health", "", "", ""))


def interface_unavailable_event(reason: str, **details: Any) -> dict[str, Any]:
    return {
        "event": "interface_unavailable",
        "reason": reason,
        "user_action": INTERFACE_UNAVAILABLE_ACTION,
        **details,
    }


def check_health(health_url: str, timeout: float) -> tuple[bool, dict[str, Any]]:
    if not health_url:
        return False, interface_unavailable_event("missing_health_url")
    request = urllib.request.Request(health_url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", response.getcode())
            body = response.read(500).decode("utf-8", errors="replace")
            if 200 <= status < 300:
                return True, {"event": "interface_status", "ok": True, "health_url": health_url, "http_status": status}
            return False, interface_unavailable_event(
                "health_check_non_2xx",
                health_url=health_url,
                http_status=status,
                raw_excerpt=body,
            )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        return False, interface_unavailable_event(
            "health_check_http_error",
            health_url=health_url,
            http_status=exc.code,
            raw_excerpt=body,
        )
    except OSError as exc:
        return False, interface_unavailable_event(
            "health_check_failed",
            health_url=health_url,
            error=type(exc).__name__,
            message=str(exc),
        )


def forward_events(events: Iterable[tuple[str | None, str]], source: str, max_events: int | None = None) -> int:
    started_at = time.time()
    phases: list[dict[str, Any]] = []
    keywords: list[str] = []
    references: list[dict[str, Any]] = []
    failures: list[Any] = []
    answer_chunks: list[str] = []
    private_search_summary: dict[str, Any] | None = None
    done: dict[str, Any] | None = None
    usage: Any = None
    raw_event_count = 0
    interrupted: dict[str, Any] | None = None
    interface_error: dict[str, Any] | None = None

    emit({"event": "stream_started", "source": source})
    for raw_event, data in events:
        raw_event_count += 1
        payload = load_json(data)
        event = raw_event or "message"

        if event == "phase" and isinstance(payload, dict):
            phases.append(payload)
            emit({"event": "phase", "phase": payload.get("phase") or payload.get("name"), "payload": payload})
        elif event == "keywords" and isinstance(payload, dict):
            keywords = payload.get("keywords") or []
            emit({"event": "keywords_ready", "keywords": keywords, "payload": payload})
        elif event == "private_search_hit" and isinstance(payload, dict):
            if payload.get("ok") is False or payload.get("error"):
                failures.append(payload)
            emit({"event": "private_search_hit", "payload": payload})
        elif event == "private_search_summary" and isinstance(payload, dict):
            private_search_summary = payload
            failures.extend(payload.get("failures") or [])
            emit(
                {
                    "event": "private_search_summary",
                    "totalResults": payload.get("totalResults"),
                    "failure_count": len(payload.get("failures") or []),
                    "payload": payload,
                }
            )
        elif event == "references" and isinstance(payload, dict):
            references = payload.get("references") or []
            emit(
                {
                    "event": "references_ready",
                    "references_count": len(references),
                    "first_references": first_references(references),
                }
            )
        elif event == "delta" and isinstance(payload, dict):
            content = payload.get("content") or ""
            answer_chunks.append(content)
            emit({"event": "answer_delta", "content": content, "answer_chars": sum(len(chunk) for chunk in answer_chunks)})
        elif event == "done" and isinstance(payload, dict):
            done = payload
            emit({"event": "done", "ok": payload.get("ok"), "payload": payload})
        elif event == "usage" and isinstance(payload, dict):
            usage = payload.get("usage") or payload
            emit({"event": "usage", "usage": usage})
        elif event in {"stream_error", "interface_unavailable"} and isinstance(payload, dict):
            interface_error = payload
            emit({"event": "interface_unavailable", **payload})
        else:
            emit({"event": "raw_event", "raw_event": event, "payload": payload if payload is not None else data[:500]})

        if max_events is not None and raw_event_count >= max_events:
            interrupted = {
                "reason": "max_events",
                "max_events": max_events,
                "message": "Stopped after the requested number of SSE events; preserve partial evidence for the UI.",
            }
            emit({"event": "stream_interrupted", **interrupted})
            break

    answer_text = "".join(answer_chunks)
    total_results = private_search_summary.get("totalResults") if private_search_summary else None
    status = (
        "interface_unavailable"
        if interface_error and not references and not answer_text and done is None
        else answer_status(references, answer_text, done, total_results)
    )
    user_action = user_action_for_status(status, failures)
    if interrupted and not user_action:
        user_action = "已输出阶段进度；继续等待更多事件，或收窄问题后重试"
    emit(
        {
            "event": "stream_final",
            "source": source,
            "raw_event_count": raw_event_count,
            "phase_count": len(phases),
            "keywords": keywords,
            "private_search": {
                "totalResults": total_results,
                "failure_count": len(failures),
                "failures": failures[:10],
            },
            "references_count": len(references),
            "answer_status": status,
            "user_action": user_action,
            "answer_chars": len(answer_text or (done or {}).get("answer", "")),
            "has_done": done is not None,
            "has_usage": usage is not None,
            "interrupted": interrupted,
            "interface_error": interface_error,
            "elapsed_ms": int((time.time() - started_at) * 1000),
        }
    )
    return 2 if status == "interface_unavailable" else 0


def lines_from_log(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        yield from handle


def lines_from_http(endpoint: str, body: dict[str, Any], timeout: float) -> Iterator[str]:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                yield raw.decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body_excerpt = exc.read().decode("utf-8", errors="replace")[:500]
        payload = interface_unavailable_event(
            "stream_http_error",
            http_status=exc.code,
            content_type=exc.headers.get("Content-Type", ""),
            raw_excerpt=body_excerpt,
        )
        yield "event: interface_unavailable\n"
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    except OSError as exc:
        payload = interface_unavailable_event("stream_request_failed", error=type(exc).__name__, message=str(exc))
        yield "event: interface_unavailable\n"
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def request_body(args: argparse.Namespace) -> dict[str, Any]:
    endpoints = [item.strip() for item in args.endpoint_names.split(",") if item.strip()]
    return {
        "prompt": args.prompt,
        "model": args.model,
        "keyword_model": args.keyword_model,
        "page_num": args.page_num,
        "page_size": args.page_size,
        "endpoint_names": endpoints,
        "include_raw": args.include_raw,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Forward Deep Research SSE as JSONL progress events.")
    parser.add_argument("--from-log", type=Path, help="Replay an existing SSE log instead of calling the live endpoint.")
    parser.add_argument("--endpoint-url", default=DEFAULT_ENDPOINT)
    parser.add_argument("--prompt")
    parser.add_argument("--model", default="qwen-deep-research")
    parser.add_argument("--keyword-model", default="qwen-plus")
    parser.add_argument("--page-num", type=int, default=1)
    parser.add_argument("--page-size", type=int, default=5)
    parser.add_argument(
        "--endpoint-names",
        default="searchArticlesByQuery1,searchArxivByTitle,searchArxivByAbstract,searchArxivByArxivNo1,searchArxiv",
    )
    parser.add_argument("--include-raw", action="store_true")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--health-url", help="Override the Deep Research health-check URL.")
    parser.add_argument("--health-timeout", type=float, default=10)
    parser.add_argument("--skip-health-check", action="store_true")
    parser.add_argument("--max-events", type=int, help="Stop after N SSE events and emit a partial final summary.")
    args = parser.parse_args()

    if args.from_log:
        return forward_events(iter_sse_lines(lines_from_log(args.from_log)), str(args.from_log), args.max_events)
    if not args.prompt:
        raise SystemExit("--prompt is required unless --from-log is used")
    if not args.endpoint_url:
        missing_event = interface_unavailable_event("missing_endpoint_url")
        return forward_events(
            iter_sse_lines(
                [
                    "event: interface_unavailable\n",
                    f"data: {json.dumps(missing_event, ensure_ascii=False)}\n\n",
                ]
            ),
            "<missing endpoint>",
            args.max_events,
        )
    if not args.skip_health_check:
        ok, health_event = check_health(args.health_url or infer_health_url(args.endpoint_url), args.health_timeout)
        emit(health_event)
        if not ok:
            return forward_events(
                iter_sse_lines(
                    [
                        "event: interface_unavailable\n",
                        f"data: {json.dumps(health_event, ensure_ascii=False)}\n\n",
                    ]
                ),
                args.endpoint_url,
                args.max_events,
            )
    return forward_events(iter_sse_lines(lines_from_http(args.endpoint_url, request_body(args), args.timeout)), args.endpoint_url, args.max_events)


if __name__ == "__main__":
    raise SystemExit(main())
