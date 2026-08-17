import argparse
import base64
import hashlib
import http.client
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from build_figure_manifest import build_manifest
from check_generated_image import build_check

ROOT = "http://images.sitianai.com/"
GENERATE_ENDPOINT = urljoin(ROOT, "api/generate-async")
JOB_ENDPOINT_TEMPLATE = urljoin(ROOT, "api/generate-jobs/{job_id}")
DEFAULT_NEGATIVE_PROMPT = "水印，模糊文字，错乱标签，低清晰度，广告风格"


def stable_root():
    return Path(__file__).resolve().parents[3]


def infer_mime(path):
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "application/octet-stream"


def make_run_dir(output_dir, run_kind):
    base = Path(output_dir) if output_dir else stable_root() / "scientific_image_skill_runs"
    prefix = "edit" if run_kind == "edit" else "smoke"
    run_dir = base / (prefix + "_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def file_digest(path):
    path = Path(path)
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_input_json(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    figure_spec = data.get("figure_spec") or {}
    return {
        "prompt": data.get("prompt", ""),
        "negative_prompt": data.get("negative_prompt", data.get("negativePrompt", DEFAULT_NEGATIVE_PROMPT)),
        "aspect_ratio": data.get("aspect_ratio", data.get("aspectRatio", "1:1")),
        "image_size": data.get("image_size", data.get("imageSize", "1K")),
        "number_of_images": data.get("number_of_images", data.get("numberOfImages", 1)),
        "reference_image": data.get("reference_image"),
        "source_run": data.get("source_run"),
        "source_context": data.get("source_context"),
        "caption": data.get("caption"),
        "intent": data.get("intent", data.get("communicative_intent")),
        "figure_kind": data.get("figure_kind", figure_spec.get("figure_kind", "workflow")),
        "required_labels": data.get("required_labels", figure_spec.get("required_labels", [])),
        "forbidden_labels": data.get("forbidden_labels", figure_spec.get("forbidden_labels", [])),
        "style_brief": data.get("style_brief", figure_spec.get("style_brief")),
        "feedback": data.get("feedback"),
        "reference_role": data.get("reference_role", figure_spec.get("reference_role", "")),
        "preserve_constraints": data.get("preserve_constraints", figure_spec.get("preserve_constraints", [])),
        "allowed_changes": data.get("allowed_changes", figure_spec.get("allowed_changes", [])),
        "disallowed_changes": data.get("disallowed_changes", figure_spec.get("disallowed_changes", [])),
        "layout_brief": data.get("layout_brief", figure_spec.get("layout_brief")),
    }


def merge_input_json(args):
    if not args.input_json:
        return
    auto = load_input_json(args.input_json)
    for name, value in auto.items():
        current = getattr(args, name)
        if current in (None, "", []):
            setattr(args, name, value)


def summarize_body_for_metadata(body):
    summary = dict(body)
    if "imageBase64" in summary:
        summary["imageBase64"] = "<redacted reference image base64>"
    return summary


def build_figure_spec(args):
    return {
        "schema": "giiisp_figure_spec_v1",
        "figure_kind": args.figure_kind,
        "caption": args.caption,
        "communicative_intent": args.intent or args.caption,
        "required_labels": args.required_labels,
        "forbidden_labels": args.forbidden_labels,
        "layout_brief": args.layout_brief,
        "style_brief": args.style_brief,
        "reference_role": args.reference_role,
        "preserve_constraints": args.preserve_constraints,
        "allowed_changes": args.allowed_changes,
        "disallowed_changes": args.disallowed_changes,
    }


def write_run_input(run_dir, args, body, request_metadata):
    reference_path = Path(args.reference_image) if args.reference_image else None
    source_run = Path(args.source_run) if args.source_run else None
    parent_image = reference_path if reference_path and reference_path.exists() else None
    continue_from = None
    if source_run or reference_path or args.feedback:
        continue_from = {
            "source_run_id": source_run.name if source_run else None,
            "source_run_dir": str(source_run) if source_run else None,
            "source_run_exists": source_run.exists() if source_run else None,
            "parent_run_input_path": str(source_run / "run_input.json") if source_run else None,
            "parent_metadata_path": str(source_run / "metadata.json") if source_run else None,
            "parent_manifest_path": str(source_run / "figure_manifest.json") if source_run else None,
            "source_image_path": str(parent_image) if parent_image else str(reference_path) if reference_path else None,
            "source_image_exists": reference_path.exists() if reference_path else None,
            "source_image_sha256": file_digest(reference_path) if reference_path and reference_path.exists() else None,
            "feedback": args.feedback,
            "reference_role": args.reference_role,
            "preserve_constraints": args.preserve_constraints,
            "allowed_changes": args.allowed_changes,
            "disallowed_changes": args.disallowed_changes,
            "new_prompt_delta": args.prompt.strip(),
        }
    figure_spec = build_figure_spec(args)
    write_json(run_dir / "figure_spec.json", figure_spec)
    run_input = {
        "schema": "giiisp_run_input_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_dir.name,
        "run_kind": args.run_kind,
        "provider": "giiisp-imagine",
        "endpoint": GENERATE_ENDPOINT,
        "source_context": args.source_context,
        "caption": args.caption,
        "communicative_intent": args.intent or args.caption,
        "figure_kind": args.figure_kind,
        "required_labels": args.required_labels,
        "forbidden_labels": args.forbidden_labels,
        "layout_brief": args.layout_brief,
        "style_brief": args.style_brief,
        "reference_role": args.reference_role,
        "figure_spec": figure_spec,
        "prompt": args.prompt.strip(),
        "prompt_hash": prompt_digest(args.prompt.strip()),
        "negative_prompt": args.negative_prompt.strip(),
        "aspect_ratio": args.aspect_ratio,
        "image_size": args.image_size,
        "number_of_images": max(1, min(4, args.number_of_images)),
        "output_mime_type": body.get("outputMimeType"),
        "response_modalities": body.get("responseModalities"),
        "source_run": request_metadata.get("source_run"),
        "feedback": args.feedback,
        "continue_from": continue_from,
        "reference_image": str(reference_path) if reference_path else None,
        "reference_image_exists": reference_path.exists() if reference_path else None,
        "reference_image_sha256": file_digest(reference_path) if reference_path and reference_path.exists() else None,
        "request_body_summary": summarize_body_for_metadata(body),
        "token_policy": "read from GIIISP_AUTH_TOKEN or SITIANAI_IMAGE_TOKEN environment variable; token is not written to files",
        "iterations_requested": 1,
        "vector_export_requested": False,
        "editable_export_requested": False,
    }
    write_json(run_dir / "run_input.json", run_input)
    return run_input


def write_metadata(
    run_dir,
    run_input,
    status,
    image_path=None,
    check=None,
    blocker=None,
    poll_count=0,
    job_id=None,
    started_at=None,
):
    finished_at = datetime.now()
    started = started_at or finished_at
    duration_seconds = round((finished_at - started).total_seconds(), 3)
    metadata = {
        "schema": "giiisp_run_metadata_v1",
        "created_at": run_input.get("created_at"),
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "updated_at": finished_at.isoformat(timespec="seconds"),
        "duration_seconds": duration_seconds,
        "run_id": run_dir.name,
        "status": status,
        "terminal_status": status,
        "provider": "giiisp-imagine",
        "vlm_provider": None,
        "image_provider": "giiisp-imagine",
        "model": None,
        "model_version": None,
        "endpoint": GENERATE_ENDPOINT,
        "job_id": job_id,
        "run_kind": run_input.get("run_kind"),
        "prompt_hash": run_input.get("prompt_hash"),
        "source_run": run_input.get("source_run"),
        "poll_count": poll_count,
        "iterations_requested": run_input.get("iterations_requested"),
        "iterations_completed": 1 if status == "completed" else 0,
        "last_iteration": 1 if status == "completed" else 0,
        "config_snapshot": {
            "aspect_ratio": run_input.get("aspect_ratio"),
            "image_size": run_input.get("image_size"),
            "number_of_images": run_input.get("number_of_images"),
            "output_mime_type": run_input.get("output_mime_type"),
            "endpoint": GENERATE_ENDPOINT,
        },
        "attempts": {
            "generation_requests": 1 if (run_dir / "request.json").exists() else 0,
            "poll_requests": poll_count,
            "image_downloads": 1 if image_path else 0,
        },
        "usage": {
            "estimated_cost_usd": None,
            "credits_used": None,
            "tokens_used": None,
        },
        "last_image_sha256": file_digest(image_path) if image_path else None,
        "output_paths": {
            "run_dir": str(run_dir),
            "request": str(run_dir / "request.json") if (run_dir / "request.json").exists() else None,
            "response": str(run_dir / "response.json") if (run_dir / "response.json").exists() else None,
            "poll_history": str(run_dir / "poll_history.json") if (run_dir / "poll_history.json").exists() else None,
            "image": str(image_path) if image_path else None,
            "check": str(run_dir / "check.json") if (run_dir / "check.json").exists() else None,
            "blocker": str(run_dir / "blocker.json") if (run_dir / "blocker.json").exists() else None,
            "figure_manifest": str(run_dir / "figure_manifest.json"),
        },
        "machine_check": check,
        "blocker": blocker,
        "error": blocker.get("reason") if isinstance(blocker, dict) else None,
        "errors": [blocker] if blocker else [],
        "token_policy": "token redacted; only environment variable usage was allowed",
    }
    write_json(run_dir / "metadata.json", metadata)
    return metadata


def write_figure_manifest(run_dir, source_run=None):
    manifest = build_manifest(run_dir, source_run=source_run)
    write_json(run_dir / "figure_manifest.json", manifest)
    return manifest


def read_json_response(response):
    raw = response.read()
    text = raw.decode("utf-8", errors="replace")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = {"raw_text": text}
    return response.status, data, text


def http_json(method, url, token, body=None, timeout=60):
    headers = {
        "Content-Type": "application/json",
        "Referer": ROOT,
        "Authorization": "Bearer " + token,
    }
    payload = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=payload, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            status, data, text = read_json_response(response)
            return {"status_code": status, "json": data, "text": text}
    except HTTPError as exc:
        status, data, text = read_json_response(exc)
        return {"status_code": status, "json": data, "text": text}
    except URLError as exc:
        return {"status_code": None, "json": {"error": str(exc)}, "text": str(exc)}
    except TimeoutError as exc:
        return {"status_code": None, "json": {"error": str(exc), "type": "TimeoutError"}, "text": str(exc)}
    except (http.client.HTTPException, ConnectionError, OSError) as exc:
        # e.g. IncompleteRead when a large poll payload is cut mid-transfer; let the poll loop retry
        return {"status_code": None, "json": {"error": str(exc), "type": type(exc).__name__}, "text": str(exc)}


def resolve_auth_token():
    for name in ("GIIISP_AUTH_TOKEN", "SITIANAI_IMAGE_TOKEN"):
        token = os.environ.get(name, "").strip()
        if token:
            return token, name
    return "", None


def contains_access_token_required(value):
    if isinstance(value, str):
        return "ACCESS_TOKEN_REQUIRED" in value
    if isinstance(value, dict):
        return any(contains_access_token_required(v) for v in value.values())
    if isinstance(value, list):
        return any(contains_access_token_required(v) for v in value)
    return False


def recursive_find_first(value, names):
    if isinstance(value, dict):
        for key in names:
            if key in value and value[key]:
                return value[key]
        for child in value.values():
            found = recursive_find_first(child, names)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = recursive_find_first(child, names)
            if found:
                return found
    return None


def find_job_id(data):
    found = recursive_find_first(data, {"job_id", "jobId", "id", "task_id", "taskId"})
    if isinstance(found, (str, int)):
        return str(found)
    return None


def find_status(data):
    found = recursive_find_first(data, {"status", "state"})
    return str(found).lower() if found else ""


def iter_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from iter_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_strings(child)


def find_image_candidate(data):
    for text in iter_strings(data):
        if text.startswith("data:image/"):
            return {"kind": "data_uri", "value": text}
        if text.startswith("http://") or text.startswith("https://"):
            lower = text.lower().split("?", 1)[0]
            if lower.endswith((".png", ".jpg", ".jpeg", ".webp")) or "/image" in lower:
                return {"kind": "url", "value": text}
        if len(text) > 200 and re.fullmatch(r"[A-Za-z0-9+/=\s]+", text):
            return {"kind": "base64", "value": text}
    return None


def save_image(candidate, run_dir):
    if candidate["kind"] == "url":
        url = candidate["value"]
        with urlopen(url, timeout=120) as response:
            data = response.read()
        suffix = Path(url.split("?", 1)[0]).suffix.lower() or ".png"
        path = run_dir / ("generated_image" + suffix)
        path.write_bytes(data)
        (run_dir / "image_url.txt").write_text(url, encoding="utf-8")
    elif candidate["kind"] == "data_uri":
        header, encoded = candidate["value"].split(",", 1)
        mime = header.split(";", 1)[0].replace("data:", "")
        suffix = ".png" if mime == "image/png" else ".jpg" if mime == "image/jpeg" else ".bin"
        data = base64.b64decode(encoded)
        path = run_dir / ("generated_image" + suffix)
        path.write_bytes(data)
    else:
        data = base64.b64decode(candidate["value"])
        path = run_dir / "generated_image.png"
        path.write_bytes(data)
    return path


def normalize_extension(image_path, check):
    suffix_by_type = {"png": ".png", "jpeg": ".jpg", "webp": ".webp"}
    expected = suffix_by_type.get(check.get("image_type"))
    if not expected or image_path.suffix.lower() == expected:
        return image_path
    normalized = image_path.with_suffix(expected)
    if normalized.exists():
        normalized = image_path.with_name(image_path.stem + "_checked" + expected)
    image_path.rename(normalized)
    return normalized


def write_blocker(run_dir, reason, details=None):
    blocker = {"blocked": True, "reason": reason, "details": details or {}}
    write_json(run_dir / "blocker.json", blocker)
    print("BLOCKED: " + reason)
    if "GIIISP_AUTH_TOKEN" in reason or "ACCESS_TOKEN_REQUIRED" in reason:
        print("Apply for or refresh Giiisp authentication at https://giiisp.com/#/mcp/authenticate")
    print("Run directory: " + str(run_dir))


def main():
    parser = argparse.ArgumentParser(description="Run a real Giiisp Imagine generate-async smoke test.")
    parser.add_argument("--prompt")
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--aspect-ratio", default="1:1")
    parser.add_argument("--image-size", default="1K")
    parser.add_argument("--number-of-images", type=int, default=1)
    parser.add_argument("--output-dir", help="Directory that will receive timestamped smoke run folders.")
    parser.add_argument("--run-kind", choices=["smoke", "edit"], default="smoke")
    parser.add_argument("--reference-image", help="Local image path for an edit/reference generation.")
    parser.add_argument("--source-run", help="Prior run directory for edit/reference generation.")
    parser.add_argument("--source-context", help="Original paper paragraph, method description, or user material behind the figure.")
    parser.add_argument("--caption", help="Short caption or figure title.")
    parser.add_argument("--intent", help="Scientific message this figure should communicate.")
    parser.add_argument("--figure-kind", default="workflow", help="workflow, mechanism schematic, method diagram, comparison figure, etc.")
    parser.add_argument("--required-labels", nargs="*", default=[], help="Labels that must appear in the figure.")
    parser.add_argument("--forbidden-labels", nargs="*", default=[], help="Labels or text patterns that must not appear.")
    parser.add_argument("--style-brief", help="Compact visual style brief.")
    parser.add_argument("--feedback", help="User feedback for edit/continue-style runs.")
    parser.add_argument("--reference-role", default="", choices=["", "preserve_structure", "use_elements", "refine_sketch", "edit_image"], help="How to interpret the reference image, if present.")
    parser.add_argument("--preserve-constraints", nargs="*", default=[], help="Elements that should be preserved from the parent run.")
    parser.add_argument("--allowed-changes", nargs="*", default=[], help="Specific changes allowed in an edit run.")
    parser.add_argument("--disallowed-changes", nargs="*", default=[], help="Changes that must not happen in an edit run.")
    parser.add_argument("--layout-brief", help="Structured layout constraint such as horizontal four-step flow or 2x2 grid.")
    parser.add_argument("--input-json", help="Path to JSON containing prompt fields and optional figure_spec.")
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--max-polls", type=int, default=24)
    parser.add_argument("--request-timeout", type=int, default=60, help="Initial generate-async request timeout in seconds.")
    args = parser.parse_args()
    merge_input_json(args)

    if not args.prompt:
        parser.error("--prompt is required unless --input-json provides it")

    run_dir = make_run_dir(args.output_dir, args.run_kind)
    started_at = datetime.now()
    body = {
        "prompt": args.prompt.strip(),
        "negativePrompt": args.negative_prompt.strip(),
        "aspectRatio": args.aspect_ratio,
        "imageSize": args.image_size,
        "numberOfImages": max(1, min(4, args.number_of_images)),
        "responseModalities": ["IMAGE", "TEXT"],
        "outputMimeType": "image/png",
    }
    request_metadata = {"run_kind": args.run_kind}
    if args.source_run:
        source_run = Path(args.source_run)
        request_metadata["source_run"] = str(source_run)
        request_metadata["source_run_exists"] = source_run.exists()
        (run_dir / "source_run.txt").write_text(str(source_run), encoding="utf-8")
    missing_reference = None
    if args.reference_image:
        reference_path = Path(args.reference_image)
        request_metadata["reference_image_path"] = str(reference_path)
        request_metadata["reference_image_exists"] = reference_path.exists()
        if reference_path.exists():
            body["imageBase64"] = base64.b64encode(reference_path.read_bytes()).decode("ascii")
            body["imageMimeType"] = infer_mime(reference_path)
        else:
            missing_reference = reference_path
    run_input = write_run_input(run_dir, args, body, request_metadata)

    if missing_reference:
        write_json(run_dir / "request_metadata.json", request_metadata)
        write_blocker(run_dir, "reference image path does not exist", {"reference_image": str(missing_reference)})
        blocker = json.loads((run_dir / "blocker.json").read_text(encoding="utf-8"))
        write_metadata(run_dir, run_input, "blocked", blocker=blocker, started_at=started_at)
        write_figure_manifest(run_dir, source_run=request_metadata.get("source_run"))
        return 2

    token, token_env = resolve_auth_token()
    if not token:
        write_blocker(run_dir, "missing GIIISP_AUTH_TOKEN or SITIANAI_IMAGE_TOKEN")
        blocker = json.loads((run_dir / "blocker.json").read_text(encoding="utf-8"))
        write_metadata(run_dir, run_input, "blocked", blocker=blocker, started_at=started_at)
        write_figure_manifest(run_dir, source_run=request_metadata.get("source_run"))
        return 2

    write_json(
        run_dir / "request.json",
        {
            "method": "POST",
            "url": GENERATE_ENDPOINT,
            "headers": {
                "Content-Type": "application/json",
                "Referer": ROOT,
                "Authorization": f"Bearer <redacted {token_env}>",
            },
            "body": body,
            "metadata": request_metadata,
        },
    )

    response = http_json("POST", GENERATE_ENDPOINT, token, body=body, timeout=max(1, args.request_timeout))
    write_json(run_dir / "response.json", response)
    if contains_access_token_required(response):
        write_blocker(run_dir, "ACCESS_TOKEN_REQUIRED", {"status_code": response.get("status_code")})
        blocker = json.loads((run_dir / "blocker.json").read_text(encoding="utf-8"))
        write_metadata(run_dir, run_input, "blocked", blocker=blocker, started_at=started_at)
        write_figure_manifest(run_dir, source_run=request_metadata.get("source_run"))
        return 2

    job_id = find_job_id(response["json"])
    if not job_id:
        response_error = response.get("json", {}).get("error") if isinstance(response.get("json"), dict) else None
        response_error_type = response.get("json", {}).get("type") if isinstance(response.get("json"), dict) else None
        reason = "generate-async request failed" if response.get("status_code") is None and response_error else "generate-async response did not include job_id"
        write_blocker(
            run_dir,
            reason,
            {
                "status_code": response.get("status_code"),
                "response_error": response_error,
                "response_error_type": response_error_type,
                "downstream_effect": "no job_id",
            },
        )
        blocker = json.loads((run_dir / "blocker.json").read_text(encoding="utf-8"))
        write_metadata(run_dir, run_input, "blocked", blocker=blocker, started_at=started_at)
        write_figure_manifest(run_dir, source_run=request_metadata.get("source_run"))
        return 2

    poll_history = []
    image_candidate = None
    terminal_status = ""
    for index in range(max(1, args.max_polls)):
        if index:
            time.sleep(max(0.5, args.poll_interval))
        job_url = JOB_ENDPOINT_TEMPLATE.format(job_id=job_id)
        poll = http_json("GET", job_url, token)
        poll_history.append({"poll": index + 1, "url": job_url, "result": poll})
        write_json(run_dir / "poll_history.json", poll_history)

        if contains_access_token_required(poll):
            write_blocker(run_dir, "ACCESS_TOKEN_REQUIRED while polling", {"status_code": poll.get("status_code")})
            blocker = json.loads((run_dir / "blocker.json").read_text(encoding="utf-8"))
            write_metadata(run_dir, run_input, "blocked", blocker=blocker, poll_count=len(poll_history), job_id=job_id, started_at=started_at)
            write_figure_manifest(run_dir, source_run=request_metadata.get("source_run"))
            return 2

        image_candidate = find_image_candidate(poll["json"])
        terminal_status = find_status(poll["json"])
        if image_candidate:
            break
        if terminal_status in {"failed", "failure", "error", "cancelled", "canceled"}:
            write_blocker(run_dir, "job ended without image", {"status": terminal_status})
            blocker = json.loads((run_dir / "blocker.json").read_text(encoding="utf-8"))
            write_metadata(run_dir, run_input, "blocked", blocker=blocker, poll_count=len(poll_history), job_id=job_id, started_at=started_at)
            write_figure_manifest(run_dir, source_run=request_metadata.get("source_run"))
            return 2

    if not image_candidate:
        write_blocker(run_dir, "polling completed without image", {"polls": len(poll_history), "last_status": terminal_status})
        blocker = json.loads((run_dir / "blocker.json").read_text(encoding="utf-8"))
        write_metadata(run_dir, run_input, "blocked", blocker=blocker, poll_count=len(poll_history), job_id=job_id, started_at=started_at)
        write_figure_manifest(run_dir, source_run=request_metadata.get("source_run"))
        return 2

    try:
        image_path = save_image(image_candidate, run_dir)
        check = build_check(image_path, run_dir / "blocker.json")
        image_path = normalize_extension(image_path, check)
        if str(image_path) != check.get("image_path"):
            check = build_check(image_path, run_dir / "blocker.json")
        check["image_saved"] = check["image_exists"]
        if not check["non_empty"]:
            write_json(run_dir / "check.json", check)
            write_blocker(run_dir, "downloaded image is empty")
            blocker = json.loads((run_dir / "blocker.json").read_text(encoding="utf-8"))
            write_metadata(run_dir, run_input, "blocked", image_path=image_path, check=check, blocker=blocker, poll_count=len(poll_history), job_id=job_id, started_at=started_at)
            write_figure_manifest(run_dir, source_run=request_metadata.get("source_run"))
            return 2
        if not check["supported_type"]:
            write_json(run_dir / "check.json", check)
            write_blocker(run_dir, "downloaded image type is not PNG, JPEG, or WebP")
            blocker = json.loads((run_dir / "blocker.json").read_text(encoding="utf-8"))
            write_metadata(run_dir, run_input, "blocked", image_path=image_path, check=check, blocker=blocker, poll_count=len(poll_history), job_id=job_id, started_at=started_at)
            write_figure_manifest(run_dir, source_run=request_metadata.get("source_run"))
            return 2
        write_json(run_dir / "check.json", check)
    except Exception as exc:
        write_json(run_dir / "check.json", {"image_saved": False, "failure": str(exc)})
        write_blocker(run_dir, "failed to save image", {"error": str(exc)})
        blocker = json.loads((run_dir / "blocker.json").read_text(encoding="utf-8"))
        write_metadata(run_dir, run_input, "blocked", blocker=blocker, poll_count=len(poll_history), job_id=job_id, started_at=started_at)
        write_figure_manifest(run_dir, source_run=request_metadata.get("source_run"))
        return 2

    write_metadata(run_dir, run_input, "completed", image_path=image_path, check=check, poll_count=len(poll_history), job_id=job_id, started_at=started_at)
    write_figure_manifest(run_dir, source_run=request_metadata.get("source_run"))
    print("OK: generated image smoke test completed")
    print("Run directory: " + str(run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
