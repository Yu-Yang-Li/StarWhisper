#!/usr/bin/env python3
"""Emit progress events for Giiisp paper search expansion.

The script writes JSONL to stdout so callers can forward each line as SSE.
Dry-run mode is the safe default for tests and integration planning.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any


BASE_URL = "https://giiisp.com"
GIIISP_AUTH_URL = "https://giiisp.com/#/mcp/authenticate"
GIIISP_AUTH_ACTION = "申请或刷新 Giiisp MCP 认证后设置 GIIISP_AUTH_TOKEN"

MODES = {
    "oa": ("/first/oaPaper/searchArticlesByQuery1", lambda q, p, s: {"titleAndAbs": [q]}),
    "arxiv-abstract": ("/first/paper/searchArxivByAbstract", lambda q, p, s: {"key": q, "pageNum": p, "pageSize": s}),
    "arxiv-no": ("/first/paper/searchArxivByArxivNo1", lambda q, p, s: {"key": q, "pageNum": p, "pageSize": s}),
    "arxiv": ("/first/paper/searchArxiv", lambda q, p, s: {"key": q, "pageNum": p, "pageSize": s}),
    "arxiv-title": ("/first/paper/searchArxivByTitle", lambda q, p, s: {"key": q, "pageNum": p, "pageSize": s}),
}

STARTED_AT = time.monotonic()


def emit(event: dict[str, Any]) -> None:
    event.setdefault("ts", datetime.now().astimezone().isoformat(timespec="milliseconds"))
    event.setdefault("elapsed_ms", int((time.monotonic() - STARTED_AT) * 1000))
    print(json.dumps(event, ensure_ascii=False, separators=(",", ":")), flush=True)


def parse_expand(value: str) -> list[str]:
    if not value.strip():
        return []
    modes = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [mode for mode in modes if mode not in MODES]
    if invalid:
        raise SystemExit(f"invalid expand mode(s): {', '.join(invalid)}")
    return modes


def build_request(mode: str, query: str, page_num: int, page_size: int) -> dict[str, Any]:
    path, builder = MODES[mode]
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    token = os.environ.get("GIIISP_AUTH_TOKEN", "").strip()
    if token:
        headers["Authorization"] = "Bearer " + token
    return {
        "method": "POST",
        "url": BASE_URL + path,
        "headers": headers,
        "body": builder(query, page_num, page_size),
    }


def redact_request(request_payload: dict[str, Any]) -> dict[str, Any]:
    safe = dict(request_payload)
    headers = dict(safe.get("headers") or {})
    if "Authorization" in headers:
        headers["Authorization"] = "Bearer <redacted GIIISP_AUTH_TOKEN>"
    safe["headers"] = headers
    return safe


def auth_status_event() -> dict[str, Any]:
    has_token = bool(os.environ.get("GIIISP_AUTH_TOKEN", "").strip())
    return {
        "event": "auth_status",
        "provider": "giiisp",
        "ok": has_token,
        "env_var": "GIIISP_AUTH_TOKEN",
        "auth_url": GIIISP_AUTH_URL,
        "user_action": "" if has_token else GIIISP_AUTH_ACTION,
    }


def infer_result_count(payload: Any) -> int | None:
    if isinstance(payload, list):
        return len(payload)
    if not isinstance(payload, dict):
        return None
    for key in ("total", "totalResults", "totalCount", "count"):
        value = payload.get(key)
        if isinstance(value, int):
            return value
    for key in ("data", "results", "records", "list", "items", "papers"):
        value = payload.get(key)
        if isinstance(value, list):
            return len(value)
        if isinstance(value, dict):
            nested = infer_result_count(value)
            if nested is not None:
                return nested
    return None


def post_json(request_payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(request_payload["body"], ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        request_payload["url"],
        data=data,
        headers=request_payload["headers"],
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            content_type = response.headers.get("Content-Type", "")
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = None
            return {
                "ok": 200 <= response.status < 300 and payload is not None,
                "http_status": response.status,
                "content_type": content_type,
                "json": payload,
                "raw_excerpt": raw[:300],
                "result_count": infer_result_count(payload),
            }
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        return {
            "ok": False,
            "http_status": exc.code,
            "content_type": exc.headers.get("Content-Type", ""),
            "json": None,
            "raw_excerpt": raw[:300],
            "result_count": None,
        }
    except OSError as exc:
        return {
            "ok": False,
            "error": type(exc).__name__,
            "message": str(exc),
            "result_count": None,
        }


def add_user_guidance(response: dict[str, Any]) -> dict[str, Any]:
    status = response.get("http_status")
    content_type = (response.get("content_type") or "").lower()
    raw_excerpt = response.get("raw_excerpt") or ""
    if status in {401, 403} or "text/html" in content_type or "login" in raw_excerpt.lower():
        response["auth_url"] = GIIISP_AUTH_URL
        response["user_action"] = GIIISP_AUTH_ACTION
        response["verification_status"] = "接口受限"
    elif response.get("error") or "timed out" in (response.get("message") or "").lower():
        response["user_action"] = "接口响应慢或网络受限；先展示已有进度，收窄检索词或稍后重试"
        response["verification_status"] = "接口受限"
    return response


def iter_routes(primary: str, expand: list[str], max_pages: int) -> list[tuple[str, int]]:
    routes = [primary, *expand]
    return [(mode, page) for mode in routes for page in range(1, max_pages + 1)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Emit JSONL progress events for paper search.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--mode", choices=sorted(MODES), required=True)
    parser.add_argument("--expand", default="", help="Comma-separated extra modes to run after the primary mode.")
    parser.add_argument("--page-size", type=int, default=10)
    parser.add_argument("--max-pages", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--dry-run", action="store_true", help="Emit planned requests without sending network calls.")
    args = parser.parse_args()

    if args.max_pages < 1:
        raise SystemExit("--max-pages must be >= 1")

    expand = parse_expand(args.expand)
    routes = iter_routes(args.mode, expand, args.max_pages)
    started_at = time.time()

    emit(
        {
            "event": "search_started",
            "query": args.query,
            "primary_mode": args.mode,
            "expanded_modes": expand,
            "planned_requests": len(routes),
            "dry_run": args.dry_run,
        }
    )
    emit(auth_status_event())

    responses = []
    for index, (mode, page_num) in enumerate(routes, start=1):
        request_payload = build_request(mode, args.query.strip(), page_num, args.page_size)
        path = request_payload["url"].removeprefix(BASE_URL)
        route = {"mode": mode, "api": path, "page_num": page_num, "page_size": args.page_size}
        emit({"event": "request_prepared", "sequence": index, "route": route, "request": redact_request(request_payload)})
        if args.dry_run:
            continue
        emit({"event": "request_started", "sequence": index, "route": route, "timeout_seconds": args.timeout})
        response = post_json(request_payload, args.timeout)
        response = add_user_guidance(response)
        responses.append({"route": route, "response": response})
        emit({"event": "response_received", "sequence": index, "route": route, "response": response})

    emit(
        {
            "event": "search_complete",
            "query": args.query,
            "planned_requests": len(routes),
            "responses": len(responses),
            "dry_run": args.dry_run,
            "elapsed_ms": int((time.time() - started_at) * 1000),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
