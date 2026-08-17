#!/usr/bin/env python3
"""Independent aesthetic-refinement pass, run once before first image generation.

Mirrors PaperBanana's Stylist agent: given a draft page brief (extra_fields)
and the deck's visual_style_contract, ask a text model to enrich the visual
details (colors, spacing, icon treatment, hierarchy) so the page brief is
closer to publication-ready, WITHOUT changing semantic content, required
labels, or exact visible text. This runs before the image model ever sees
the page, unlike the VLM Critic (semantic_review_dashscope.py) which runs
after an image already exists.

Uses the same DashScope-compatible chat endpoint already wired for VLM
review, but as a text-only call (no image attached).
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import time
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
DEFAULT_MODEL = "qwen3.7-plus"
IMAGE_SAFE_EXTRA_FIELDS = ["visual_elements", "visual_focus", "layout_notes"]

SYSTEM_PROMPT = """\
## 角色
你是一位服务于科研工具产品汇报场景的资深视觉设计师（Lead Visual Designer）。

## 任务
在图片真正生成之前，对某一页 PPT 的设计草稿（视觉元素/视觉焦点/排版布局）做一轮审美精修。
你会收到：该页标题、页面文字（必须原样出现在图片里，不可修改）、当前的设计草稿三个字段，以及全篇的风格契约（配色、字体、间距、图标风格等精确约束）。

## 铁律（不可违反）
1. 绝对不能改变语义、不能新增或删除信息、不能改变页面要表达的内容和结论。
2. 绝对不能改变"页面文字"和必须出现的关键词；你只精修"视觉元素/视觉焦点/排版布局"三个设计字段的表述。
3. 如果草稿已经具体、专业、可执行，直接保留，不要为了修改而修改。
4. 只在草稿含糊、笼统、或者与风格契约不一致时才做实质性修改，把风格契约里的具体色值、字体、间距、图标规则落实到这一页的设计描述里。
5. 不要引入风格契约之外的新配色或新字体。

## 输出
严格输出 JSON，不要 Markdown 代码块，不要解释性文字：
{
  "changed": true|false,
  "visual_elements": "精修后的视觉元素描述",
  "visual_focus": "精修后的视觉焦点描述",
  "layout_notes": "精修后的排版布局描述",
  "stylist_notes": "如果 changed=true，用一句话说明改了什么；如果 changed=false，写 'No changes needed.'"
}
"""


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def find_slide(spec: dict, slide_id: str) -> dict:
    for slide in spec.get("slides", []):
        if str(slide.get("slide_id", "")).zfill(2) == str(slide_id).zfill(2):
            return slide
    raise SystemExit(f"slide_id not found: {slide_id}")


def build_user_prompt(slide: dict, contract: dict) -> str:
    extra = slide.get("extra_fields") or {}
    draft = {key: extra.get(key, "") for key in IMAGE_SAFE_EXTRA_FIELDS}
    return (
        "页面标题：" + str(slide.get("title") or "") + "\n"
        "页面文字（不可修改，仅供你理解上下文）：" + json.dumps(slide.get("body_text") or [], ensure_ascii=False) + "\n"
        "必须出现的关键词：" + json.dumps(slide.get("must_show") or [], ensure_ascii=False) + "\n\n"
        "当前设计草稿：\n" + json.dumps(draft, ensure_ascii=False, indent=2) + "\n\n"
        "全篇风格契约：\n" + json.dumps(contract, ensure_ascii=False, indent=2) + "\n\n"
        "请按系统指令输出精修结果的 JSON。"
    )


def call_dashscope_text(model: str, api_key: str, system_prompt: str, user_prompt: str, timeout: int) -> dict:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "response_format": {"type": "json_object"},
    }
    request = Request(
        ENDPOINT,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + api_key},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            text = response.read().decode("utf-8", errors="replace")
            return {"status_code": response.status, "json": json.loads(text), "text": text}
    except HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = {"raw_text": text}
        return {"status_code": exc.code, "json": data, "text": text}
    except URLError as exc:
        # Includes SSL-layer drops (e.g. "UNEXPECTED_EOF_WHILE_READING") wrapped by urlopen;
        # classify by the underlying reason's type name so the retry wrapper can catch them too.
        reason = getattr(exc, "reason", None)
        return {
            "status_code": None,
            "json": {"error": str(exc), "type": type(reason).__name__ if reason is not None else "URLError"},
            "text": str(exc),
        }
    except (ConnectionError, http.client.HTTPException, TimeoutError, OSError) as exc:
        return {"status_code": None, "json": {"error": str(exc), "type": type(exc).__name__}, "text": str(exc)}


def is_transient_network_error(response: dict) -> bool:
    if response.get("status_code") is not None:
        return False
    error_type = ""
    if isinstance(response.get("json"), dict):
        error_type = str(response["json"].get("type") or "")
    error_text = str(response.get("text") or "")
    if error_type in {
        "RemoteDisconnected",
        "ConnectionResetError",
        "ConnectionAbortedError",
        "BrokenPipeError",
        "TimeoutError",
        "IncompleteRead",
        "SSLError",
        "SSLEOFError",
        "URLError",
    }:
        return True
    return any(marker in error_text for marker in ("SSL", "EOF occurred", "Remote end closed", "Connection reset"))


def call_dashscope_text_with_retry(model, api_key, system_prompt, user_prompt, timeout, attempts=4, base_backoff_seconds=8):
    last = None
    for attempt in range(1, max(1, attempts) + 1):
        last = call_dashscope_text(model, api_key, system_prompt, user_prompt, timeout)
        if not is_transient_network_error(last):
            return last
        if attempt < attempts:
            time.sleep(base_backoff_seconds * (2 ** (attempt - 1)))
    return last


def blocked(reason: str, slide_id: str, details: dict | None = None) -> dict:
    return {
        "schema": "visual_deck_stylist_refine_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "slide_id": slide_id,
        "status": "blocked",
        "blocker": {"reason": reason, "details": details or {}},
        "changed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("slide_spec", type=Path)
    parser.add_argument("visual_style_contract", type=Path)
    parser.add_argument("--slide-id", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    spec = read_json(args.slide_spec)
    contract = read_json(args.visual_style_contract)
    slide = find_slide(spec, args.slide_id)

    extra = slide.get("extra_fields")
    if not isinstance(extra, dict) or not any(extra.get(key) for key in IMAGE_SAFE_EXTRA_FIELDS):
        result = blocked("slide has no extra_fields draft to refine (legacy page_description-only slide)", args.slide_id)
        if args.out:
            write_json(args.out, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2

    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        result = blocked("missing DASHSCOPE_API_KEY", args.slide_id)
        if args.out:
            write_json(args.out, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2

    user_prompt = build_user_prompt(slide, contract)
    response = call_dashscope_text_with_retry(args.model, api_key, SYSTEM_PROMPT, user_prompt, args.timeout)

    result = {
        "schema": "visual_deck_stylist_refine_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "slide_id": args.slide_id,
        "model": args.model,
        "provider": "dashscope",
    }

    if not (response.get("status_code") and 200 <= response["status_code"] < 300):
        result["status"] = "blocked"
        result["blocker"] = {"reason": "dashscope request failed", "details": response}
        result["changed"] = False
        if args.out:
            write_json(args.out, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2

    try:
        content = response["json"]["choices"][0]["message"]["content"]
        text = content.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        parsed = json.loads(text)
    except Exception as exc:
        result["status"] = "blocked"
        result["blocker"] = {"reason": "failed to parse stylist response", "details": {"error": str(exc)}}
        result["changed"] = False
        if args.out:
            write_json(args.out, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2

    result["status"] = "completed"
    result["changed"] = bool(parsed.get("changed"))
    result["refined_extra_fields"] = {key: parsed.get(key, extra.get(key, "")) for key in IMAGE_SAFE_EXTRA_FIELDS}
    result["stylist_notes"] = parsed.get("stylist_notes", "")

    if args.out:
        write_json(args.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
