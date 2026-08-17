import argparse
import json
from pathlib import Path


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_SIGNATURE = b"\xff\xd8\xff"
RIFF_SIGNATURE = b"RIFF"
WEBP_SIGNATURE = b"WEBP"


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_read_error": str(exc)}


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def png_dimensions(data):
    if len(data) >= 24 and data[:8] == PNG_SIGNATURE and data[12:16] == b"IHDR":
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    return None, None


def jpeg_dimensions(data):
    if not data.startswith(JPEG_SIGNATURE):
        return None, None
    index = 2
    while index + 9 < len(data):
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        index += 2
        if marker in {0xD8, 0xD9}:
            continue
        if index + 2 > len(data):
            break
        segment_length = int.from_bytes(data[index:index + 2], "big")
        if segment_length < 2 or index + segment_length > len(data):
            break
        if marker in {
            0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
            0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
        }:
            if segment_length >= 7:
                height = int.from_bytes(data[index + 3:index + 5], "big")
                width = int.from_bytes(data[index + 5:index + 7], "big")
                return width, height
            break
        index += segment_length
    return None, None


def webp_dimensions(data):
    if len(data) < 30 or data[:4] != RIFF_SIGNATURE or data[8:12] != WEBP_SIGNATURE:
        return None, None
    chunk = data[12:16]
    if chunk == b"VP8X" and len(data) >= 30:
        width = int.from_bytes(data[24:27], "little") + 1
        height = int.from_bytes(data[27:30], "little") + 1
        return width, height
    if chunk == b"VP8L" and len(data) >= 25:
        bits = int.from_bytes(data[21:25], "little")
        width = (bits & 0x3FFF) + 1
        height = ((bits >> 14) & 0x3FFF) + 1
        return width, height
    if chunk == b"VP8 " and len(data) >= 30:
        start = data.find(b"\x9d\x01\x2a", 20)
        if start != -1 and start + 7 <= len(data):
            width = int.from_bytes(data[start + 3:start + 5], "little") & 0x3FFF
            height = int.from_bytes(data[start + 5:start + 7], "little") & 0x3FFF
            return width, height
    return None, None


def detect_image(data):
    if data.startswith(PNG_SIGNATURE):
        width, height = png_dimensions(data)
        return "png", "image/png", width, height
    if data.startswith(JPEG_SIGNATURE):
        width, height = jpeg_dimensions(data)
        return "jpeg", "image/jpeg", width, height
    if len(data) >= 12 and data[:4] == RIFF_SIGNATURE and data[8:12] == WEBP_SIGNATURE:
        width, height = webp_dimensions(data)
        return "webp", "image/webp", width, height
    return "unknown", None, None, None


def token_blocker_reason(blocker):
    if not blocker:
        return None
    reason = str(blocker.get("reason", ""))
    if "GIIISP_AUTH_TOKEN" in reason or "ACCESS_TOKEN_REQUIRED" in reason:
        return reason
    details = json.dumps(blocker, ensure_ascii=False)
    if "ACCESS_TOKEN_REQUIRED" in details:
        return reason or "ACCESS_TOKEN_REQUIRED"
    return None


def build_check(image_path, blocker_path=None):
    image_path = Path(image_path) if image_path else None
    blocker_path = Path(blocker_path) if blocker_path else None
    blocker = read_json(blocker_path) if blocker_path and blocker_path.exists() else None
    token_reason = token_blocker_reason(blocker)

    check = {
        "image_exists": False,
        "file_size_bytes": None,
        "image_type": "missing",
        "mime_type": None,
        "width": None,
        "height": None,
        "non_empty": False,
        "supported_type": False,
        "has_token_blocker": token_reason is not None,
        "blocker_reason": blocker.get("reason") if blocker else None,
        "manual_review_required": True,
        "machine_check": {
            "status": "pending",
            "non_empty": None,
            "format_supported": None,
            "dimensions_readable": None,
            "aspect_ratio_ok": None,
            "pixel_issues": [],
        },
        "quality_review_axes": [
            {
                "name": "content_accuracy",
                "question": "Does the image represent the requested scientific message and figure kind?",
                "score": None,
                "status": "manual",
            },
            {
                "name": "layout_quality",
                "question": "Are components organized with clear reading order, spacing, and hierarchy?",
                "score": None,
                "status": "manual",
            },
            {
                "name": "text_readability",
                "question": "Are required labels present, legible, concise, and free of garbling?",
                "score": None,
                "status": "manual",
            },
            {
                "name": "aesthetic_quality",
                "question": "Does the figure look clean, academic, and free of advertising style?",
                "score": None,
                "status": "manual",
            },
            {
                "name": "artifact_severity",
                "question": "Are there watermarks, fake screenshots, broken boxes, blurry text, or random symbols?",
                "score": None,
                "status": "manual",
            },
        ],
        "notes": [
            "Manual review still required for prompt fit, required labels, watermark, and text quality."
        ],
        "manual_report_fields": [
            "run_directory",
            "source_run",
            "image_path",
            "request_summary",
            "machine_check_summary",
            "prompt_fit",
            "required_label_check",
            "text_legibility",
            "watermark_or_ad_style",
            "layout_and_hierarchy",
            "scientific_specificity",
            "prompt_quality_improvement_suggestions",
            "next_edit_prompt",
        ],
        "prompt_quality_review_required": True,
    }

    if not image_path or not image_path.exists():
        check["failure"] = "image file is missing"
        check["machine_check"]["status"] = "failed"
        check["machine_check"]["pixel_issues"].append("image file is missing")
        return check

    data = image_path.read_bytes()
    image_type, mime_type, width, height = detect_image(data)
    check.update(
        {
            "image_exists": True,
            "image_path": str(image_path),
            "file_size_bytes": len(data),
            "image_type": image_type,
            "mime_type": mime_type,
            "width": width,
            "height": height,
            "non_empty": len(data) > 0,
            "supported_type": image_type in {"png", "jpeg", "webp"},
        }
    )
    machine_check = check["machine_check"]
    machine_check["non_empty"] = check["non_empty"]
    machine_check["format_supported"] = check["supported_type"]
    machine_check["dimensions_readable"] = width is not None and height is not None
    if width and height:
        machine_check["aspect_ratio_ok"] = max(width, height) / max(min(width, height), 1) <= 3.0
    if not check["non_empty"]:
        check["failure"] = "image file is empty"
        machine_check["pixel_issues"].append("image file is empty")
    elif not check["supported_type"]:
        check["failure"] = "image type is not PNG, JPEG, or WebP"
        machine_check["pixel_issues"].append("image type is not PNG, JPEG, or WebP")
    elif width is None or height is None:
        check["warning"] = "image dimensions could not be read"
        machine_check["pixel_issues"].append("image dimensions could not be read")
    if machine_check["aspect_ratio_ok"] is False:
        machine_check["pixel_issues"].append("aspect ratio is wider/taller than 3:1")
    machine_check["status"] = "passed" if not machine_check["pixel_issues"] else "failed"
    return check


def main():
    parser = argparse.ArgumentParser(description="Check a generated scientific image file.")
    parser.add_argument("--image", required=True, help="Generated image path.")
    parser.add_argument("--blocker", help="Optional blocker.json path from the same run.")
    parser.add_argument("--out", help="Optional check.json output path.")
    parser.add_argument("--semantic-check", action="store_true", help="Add empty semantic-review placeholders for human or VLM review.")
    args = parser.parse_args()

    check = build_check(args.image, args.blocker)
    if args.semantic_check:
        check["semantic_review"] = {
            "status": "pending",
            "instructions": "Fill each quality_review_axes item with PASS / FAIL / UNCERTAIN and a short rationale.",
            "overall_ready_to_ship": None,
            "recommended_next_action": None,
        }
    if args.out:
        write_json(Path(args.out), check)
    print(json.dumps(check, ensure_ascii=False, indent=2))
    return 0 if check["image_exists"] and check["non_empty"] and check["supported_type"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
