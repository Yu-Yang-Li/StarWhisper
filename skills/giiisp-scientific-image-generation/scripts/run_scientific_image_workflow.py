import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RUN_DIR_PREFIX = "Run directory:"


def now():
    return datetime.now().isoformat(timespec="seconds")


def clock():
    return datetime.now().strftime("%H:%M")


def progress(title, message):
    print(f"[{clock()} | {title}]")
    print(message)
    print()


def read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_read_error": str(exc)}


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_subprocess(command):
    started_at = now()
    proc = subprocess.run(
        command,
        cwd=str(SCRIPT_DIR),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    return {
        "command": [str(item) for item in command],
        "started_at": started_at,
        "finished_at": now(),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def find_run_dir(stdout):
    for line in stdout.splitlines():
        if line.startswith(RUN_DIR_PREFIX):
            candidate = line[len(RUN_DIR_PREFIX) :].strip()
            if candidate:
                return Path(candidate)
    return None


def stage(name, status, started_at=None, finished_at=None, details=None):
    return {
        "name": name,
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at or now(),
        "details": details or {},
    }


def find_image(run_dir):
    for name in ["generated_image.png", "generated_image.jpg", "generated_image.jpeg", "generated_image.webp"]:
        path = run_dir / name
        if path.exists():
            return path
    return None


def ensure_response_json(run_dir):
    response_path = run_dir / "response.json"
    if response_path.exists():
        return {"normalized": False, "path": str(response_path), "reason": "response.json already exists"}
    attempts = sorted(run_dir.glob("response_attempt*.json"))
    if not attempts:
        return {"normalized": False, "path": str(response_path), "reason": "no response_attempt*.json found"}
    shutil.copyfile(attempts[0], response_path)
    return {"normalized": True, "path": str(response_path), "source": str(attempts[0])}


def semantic_status(review):
    if not isinstance(review, dict):
        return "missing"
    return review.get("status") or "unknown"


def list_axis_failures(review):
    failures = []
    axes = review.get("quality_review_axes") if isinstance(review, dict) else None
    if isinstance(axes, list):
        for axis in axes:
            if not isinstance(axis, dict):
                continue
            if str(axis.get("status", "")).upper() in {"FAIL", "UNCERTAIN"}:
                failures.append(
                    {
                        "name": axis.get("name"),
                        "status": axis.get("status"),
                        "score": axis.get("score"),
                        "rationale": axis.get("rationale"),
                    }
                )
    return failures


def list_values(value):
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if value:
        return [str(value)]
    return []


def meaningful_text(value):
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if not text or text.lower() == "no changes needed." or text.lower() == "no changes needed":
        return ""
    return text


def remove_single_value_options(argv, option_names):
    cleaned = []
    skip_next = False
    for item in argv:
        if skip_next:
            skip_next = False
            continue
        matched = False
        for option in option_names:
            if item == option:
                skip_next = True
                matched = True
                break
            if item.startswith(option + "="):
                matched = True
                break
        if not matched:
            cleaned.append(item)
    return cleaned


def set_single_value_option(argv, option, value):
    argv = remove_single_value_options(argv, {option})
    return argv + [option, str(value)]


def should_auto_repair(review):
    if not isinstance(review, dict) or review.get("status") != "completed":
        return False, "semantic review not completed"
    action = str(review.get("recommended_next_action") or "").lower()
    if action in {"edit", "regenerate"}:
        return True, f"recommended_next_action={action}"
    if review.get("overall_ready_to_ship") is False:
        return True, "overall_ready_to_ship=false"
    if list_values(review.get("missing_required_labels")):
        return True, "missing_required_labels"
    if list_values(review.get("forbidden_labels_seen")):
        return True, "forbidden_labels_seen"
    axis_failures = list_axis_failures(review)
    hard_failures = [
        item
        for item in axis_failures
        if str(item.get("name") or "") in {"content_accuracy", "text_readability", "artifact_severity"}
        and str(item.get("status") or "").upper() == "FAIL"
    ]
    if hard_failures:
        return True, "hard quality axis failure"
    return False, "review passed or only needs human judgment"


def build_auto_repair_prompt(run_dir, review):
    run_input = read_json(run_dir / "run_input.json") or {}
    figure_spec = read_json(run_dir / "figure_spec.json") or {}
    original_prompt = run_input.get("prompt") or ""
    caption = figure_spec.get("caption") or run_input.get("caption") or ""
    figure_kind = figure_spec.get("figure_kind") or run_input.get("figure_kind") or ""
    required_labels = list_values(figure_spec.get("required_labels") or run_input.get("required_labels"))
    forbidden_labels = list_values(figure_spec.get("forbidden_labels") or run_input.get("forbidden_labels"))
    missing_labels = list_values(review.get("missing_required_labels"))
    forbidden_seen = list_values(review.get("forbidden_labels_seen"))
    issues = list_values(review.get("issues"))
    axis_failures = list_axis_failures(review)
    critic_suggestions = meaningful_text(review.get("critic_suggestions"))
    revised_description = meaningful_text(review.get("revised_description"))
    next_prompt = meaningful_text(review.get("next_edit_prompt"))

    source_description = revised_description or original_prompt.strip()
    parts = ["Render an image based on the following detailed description:", source_description, ""]
    parts.extend(
        [
            "Note that do not include figure titles in the image. Diagram:",
            "请基于上一轮视觉审查结果重新生成一版科研图，只修正基础问题，不扩展新的科学内容。",
            "保持原始科学含义、图题、画幅和主体结构；不要引入未在需求中出现的新步骤、新变量或随机英文。",
        ]
    )
    if caption:
        parts.append(f"图题/主题：{caption}")
    if figure_kind:
        parts.append(f"图类型：{figure_kind}")
    if required_labels:
        parts.append("必须清晰出现这些标签，逐字保留：" + "、".join(required_labels))
    if forbidden_labels or forbidden_seen:
        combined_forbidden = forbidden_labels + [item for item in forbidden_seen if item not in forbidden_labels]
        parts.append("必须避免这些文字或元素：" + "、".join(combined_forbidden))
    if missing_labels:
        parts.append("上一版缺失的必须标签，需要补上：" + "、".join(missing_labels))
    if issues:
        parts.append("上一版需要修正的问题：" + "；".join(issues))
    if axis_failures:
        concise = []
        for item in axis_failures:
            name = item.get("name")
            status = item.get("status")
            rationale = item.get("rationale")
            concise.append(f"{name}/{status}: {rationale}")
        parts.append("质量轴问题：" + "；".join(concise))
    if critic_suggestions:
        parts.append("Critic suggestions：" + critic_suggestions)
    if next_prompt:
        parts.append("VLM 给出的修订指令：" + next_prompt)
    parts.extend(
        [
            "输出要求：白底或浅底、学术汇报风格、结构清晰、字号可读、无水印、无广告感、无伪英文、无错乱标签。",
            "这是一次自动基础修订，不要进行多轮发散优化。",
        ]
    )
    return "\n".join(part for part in parts if part is not None).strip()


def build_auto_repair_args(generation_args, source_run, repair_prompt, review):
    repair_args = remove_single_value_options(
        list(generation_args),
        {"--run-kind", "--source-run", "--reference-image", "--reference-role", "--feedback", "--prompt"},
    )
    repair_args = set_single_value_option(repair_args, "--run-kind", "edit")
    repair_args = set_single_value_option(repair_args, "--source-run", str(source_run))
    repair_args = set_single_value_option(repair_args, "--feedback", "VLM 复查发现基础问题，自动生成一版修订图。")
    repair_args = set_single_value_option(repair_args, "--prompt", repair_prompt)
    return repair_args


def summarize_blocker(blocker):
    if not isinstance(blocker, dict):
        return "未返回结构化 blocker"
    reason = blocker.get("reason") or "unknown"
    details = blocker.get("details") if isinstance(blocker.get("details"), dict) else {}
    status_code = details.get("status_code")
    response = details.get("response") if isinstance(details.get("response"), dict) else {}
    error = response.get("error") if isinstance(response.get("error"), dict) else {}
    code = error.get("code")
    message = error.get("message")
    parts = [str(reason)]
    if status_code is not None:
        parts.append(f"HTTP {status_code}")
    if code:
        parts.append(str(code))
    if message and code != message:
        compact_message = str(message).split(" For details", 1)[0].splitlines()[0][:120]
        if compact_message and compact_message not in parts:
            parts.append(compact_message)
    return "；".join(parts)


def write_agent_review_summary(run_dir, overwrite=False):
    report_path = run_dir / "manual_review.md"
    if report_path.exists() and not overwrite:
        return {"written": False, "path": str(report_path), "reason": "manual_review.md 已存在"}

    figure_spec = read_json(run_dir / "figure_spec.json") or {}
    check = read_json(run_dir / "check.json") or {}
    review = read_json(run_dir / "semantic_review.json") or {}
    image = find_image(run_dir)
    review_state = semantic_status(review)
    missing_labels = review.get("missing_required_labels") if isinstance(review, dict) else None
    forbidden_seen = review.get("forbidden_labels_seen") if isinstance(review, dict) else None
    issues = review.get("issues") if isinstance(review, dict) else None
    axis_failures = list_axis_failures(review if isinstance(review, dict) else {})
    ready = review.get("overall_ready_to_ship") if isinstance(review, dict) else None
    next_action = review.get("recommended_next_action") if isinstance(review, dict) else None
    next_prompt = review.get("next_edit_prompt") if isinstance(review, dict) else None
    critic_suggestions = review.get("critic_suggestions") if isinstance(review, dict) else None
    revised_description = review.get("revised_description") if isinstance(review, dict) else None

    if review_state != "completed":
        ready_text = "否：VLM 语义审查未完成，正式交付前仍需要人工看图确认。"
        next_action = next_action or "manual_review"
        next_prompt = next_prompt or "补充可用的 VLM 审查，或人工检查图片里的标签、结构和错字后再交付。"
    elif ready is True:
        ready_text = "是：VLM 审查认为这张图可交付。"
    elif ready is False:
        ready_text = "否：VLM 审查发现正式交付前应继续修改的问题。"
    else:
        ready_text = "不确定：VLM 审查没有给出明确交付判断。"

    lines = [
        "# 代理复查摘要",
        "",
        "本摘要基于机器检查和可用的 VLM 语义审查生成，用于交付前复查记录；它不替代最终人工看图签核。",
        "",
        "## 运行信息",
        f"- 运行目录：{run_dir}",
        f"- 图片：{image if image else '缺失'}",
        f"- 图类型：{figure_spec.get('figure_kind')}",
        f"- 图题：{figure_spec.get('caption')}",
        f"- 必须标签：{figure_spec.get('required_labels')}",
        f"- 禁止项：{figure_spec.get('forbidden_labels')}",
        "",
        "## 机器检查",
        f"- 图片存在：{check.get('image_exists')}",
        f"- 文件非空：{check.get('non_empty')}",
        f"- 类型：{check.get('image_type')} / {check.get('mime_type')}",
        f"- 尺寸：{check.get('width')} x {check.get('height')}",
        f"- 访问码阻断：{check.get('has_token_blocker')}",
        "",
        "## 语义审查",
        f"- 状态：{review_state}",
        f"- 是否可交付：{ready}",
        f"- 缺失必须标签：{missing_labels or []}",
        f"- 出现禁止项：{forbidden_seen or []}",
        f"- 问题：{issues or []}",
        f"- 失败或不确定质量轴：{axis_failures or []}",
        f"- Critic 建议：{critic_suggestions}",
        f"- Revised description：{revised_description}",
        "",
        "## 复查结论",
        f"- 交付判断：{ready_text}",
        f"- 建议动作：{next_action}",
        f"- 下一轮修改 prompt：{next_prompt}",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"written": True, "path": str(report_path)}


def build_status(
    run_dir,
    stages,
    generation_result=None,
    semantic_result=None,
    manifest_result=None,
    initial_run_dir=None,
    auto_repair=None,
):
    image = find_image(run_dir) if run_dir else None
    status = {
        "schema": "giiisp_scientific_image_workflow_v2",
        "created_at": now(),
        "run_dir": str(run_dir) if run_dir else None,
        "initial_run_dir": str(initial_run_dir) if initial_run_dir else str(run_dir) if run_dir else None,
        "final_run_dir": str(run_dir) if run_dir else None,
        "status": "completed",
        "auto_repair": auto_repair or {"attempted": False},
        "stages": stages,
        "artifacts": {
            "image": str(image) if image else None,
            "check": str(run_dir / "check.json") if run_dir and (run_dir / "check.json").exists() else None,
            "semantic_review": str(run_dir / "semantic_review.json")
            if run_dir and (run_dir / "semantic_review.json").exists()
            else None,
            "manual_review": str(run_dir / "manual_review.md") if run_dir and (run_dir / "manual_review.md").exists() else None,
            "manifest": str(run_dir / "figure_manifest.json")
            if run_dir and (run_dir / "figure_manifest.json").exists()
            else None,
        },
    }
    if generation_result and generation_result.get("returncode") != 0:
        status["status"] = "generation_blocked"
    elif semantic_result and semantic_result.get("returncode") not in (0, None):
        status["status"] = "completed_with_semantic_review_blocked"
    elif manifest_result and manifest_result.get("returncode") != 0:
        status["status"] = "manifest_rebuild_failed"
    return status


def main():
    parser = argparse.ArgumentParser(
        description="运行 Giiisp 科研绘图标准流程：生成、语义审查、代理复查摘要、manifest 重建。"
    )
    parser.add_argument("--skip-vlm", action="store_true", help="跳过 DashScope 语义审查并记录跳过状态。")
    parser.add_argument("--model", default="qwen3.7-plus", help="用于语义审查的 DashScope VLM 模型。")
    parser.add_argument("--timeout", type=int, default=120, help="DashScope 请求超时时间，单位秒。")
    parser.add_argument("--no-agent-summary", action="store_true", help="不写 manual_review.md 代理复查摘要。")
    parser.add_argument("--overwrite-agent-summary", action="store_true", help="覆盖已有 manual_review.md 摘要。")
    parser.add_argument("--no-auto-repair", action="store_true", help="关闭 VLM 后最多一次自动修订重生。")
    parser.add_argument(
        "generation_args",
        nargs=argparse.REMAINDER,
        help="传给 generate_scientific_image_smoke.py 的参数，放在 -- 后面。",
    )
    args = parser.parse_args()
    generation_args = args.generation_args
    if generation_args and generation_args[0] == "--":
        generation_args = generation_args[1:]
    if not generation_args:
        parser.error("请把生成参数放在 -- 后面，例如：-- --prompt \"...\"")

    stages = []
    auto_repair = {"attempted": False, "max_rounds": 1, "mode": "critic_reprompt_text_to_image"}
    generation_command = [sys.executable, str(SCRIPT_DIR / "generate_scientific_image_smoke.py")] + generation_args
    progress("请求提交", "生成任务开始执行；本轮会保存脱敏请求、轮询记录、图片和机器检查结果。")
    generation = run_subprocess(generation_command)
    run_dir = find_run_dir(generation.get("stdout", "")) or find_run_dir(generation.get("stderr", ""))
    stages.append(
        stage(
            "generation",
            "completed" if generation["returncode"] == 0 else "blocked",
            generation["started_at"],
            generation["finished_at"],
            {"returncode": generation["returncode"], "run_dir": str(run_dir) if run_dir else None},
        )
    )
    if generation["returncode"] != 0 and generation.get("stdout"):
        print(generation["stdout"], end="")
    if generation.get("stderr"):
        print(generation["stderr"], end="", file=sys.stderr)
    if not run_dir:
        progress("请求受阻", "生成阶段没有返回运行目录；请查看标准错误输出和 blocker 文件。")
        return generation["returncode"] or 1
    initial_run_dir = run_dir
    response_normalization = ensure_response_json(run_dir)
    stages.append(stage("response_index", "completed", details=response_normalization))
    if generation["returncode"] == 0:
        check = read_json(run_dir / "check.json") or {}
        progress(
            "图片生成",
            f"图片已返回并完成机器检查；尺寸 {check.get('width')} x {check.get('height')}，类型 {check.get('mime_type')}。接下来进行语义复查。",
        )
    else:
        progress("请求受阻", f"生成阶段受阻；运行目录：{run_dir}")

    semantic = None
    semantic_review = None
    if generation["returncode"] == 0 and not args.skip_vlm:
        semantic_command = [
            sys.executable,
            str(SCRIPT_DIR / "semantic_review_dashscope.py"),
            "--run-dir",
            str(run_dir),
            "--model",
            args.model,
            "--timeout",
            str(args.timeout),
        ]
        progress("复查开始", "开始进行 VLM 语义审查，重点看必须标签、错字、伪英文、水印和版式层级。")
        semantic = run_subprocess(semantic_command)
        stages.append(
            stage(
                "semantic_review",
                "completed" if semantic["returncode"] == 0 else "blocked",
                semantic["started_at"],
                semantic["finished_at"],
                {"returncode": semantic["returncode"], "model": args.model},
            )
        )
        if semantic.get("stderr"):
            print(semantic["stderr"], end="", file=sys.stderr)
        if semantic["returncode"] == 0:
            semantic_review = read_json(run_dir / "semantic_review.json") or {}
            progress(
                "复查结论",
                f"VLM 语义审查完成；可交付判断：{semantic_review.get('overall_ready_to_ship')}；建议动作：{semantic_review.get('recommended_next_action')}。",
            )
        else:
            semantic_review = read_json(run_dir / "semantic_review.json") or {}
            blocker = semantic_review.get("blocker") if isinstance(semantic_review, dict) else None
            progress("复查结论", f"VLM 语义审查受阻；已写入 blocked 状态，原因：{summarize_blocker(blocker)}。")
    else:
        stages.append(stage("semantic_review", "skipped", details={"reason": "skip_vlm requested or generation blocked"}))
        progress("复查结论", "VLM 语义审查已跳过；本轮不能视为完整语义闭环。")

    repair_needed, repair_reason = should_auto_repair(semantic_review)
    if generation["returncode"] == 0 and not args.skip_vlm and not args.no_auto_repair and repair_needed:
        if not args.no_agent_summary:
            initial_summary = write_agent_review_summary(run_dir, overwrite=args.overwrite_agent_summary)
            stages.append(stage("initial_agent_review_summary", "completed", details=initial_summary))
        initial_manifest_command = [sys.executable, str(SCRIPT_DIR / "build_figure_manifest.py"), "--run-dir", str(run_dir)]
        initial_manifest = run_subprocess(initial_manifest_command)
        stages.append(
            stage(
                "initial_manifest_rebuild",
                "completed" if initial_manifest["returncode"] == 0 else "blocked",
                initial_manifest["started_at"],
                initial_manifest["finished_at"],
                {"returncode": initial_manifest["returncode"]},
            )
        )
        repair_prompt = build_auto_repair_prompt(run_dir, semantic_review)
        repair_prompt_record = {
            "schema": "giiisp_auto_repair_prompt_v1",
            "created_at": now(),
            "source_run_dir": str(run_dir),
            "strategy": "PaperBanana-style Critic revised_description -> Visualizer prompt -> text-to-image regeneration",
            "critic_suggestions": semantic_review.get("critic_suggestions") if isinstance(semantic_review, dict) else None,
            "revised_description": semantic_review.get("revised_description") if isinstance(semantic_review, dict) else None,
            "next_edit_prompt": semantic_review.get("next_edit_prompt") if isinstance(semantic_review, dict) else None,
            "visualizer_prompt": repair_prompt,
            "max_auto_repair_rounds": 1,
        }
        repair_prompt_path = run_dir / "auto_repair_prompt.json"
        write_json(repair_prompt_path, repair_prompt_record)
        repair_args = build_auto_repair_args(generation_args, run_dir, repair_prompt, semantic_review)
        auto_repair.update(
            {
                "attempted": True,
                "reason": repair_reason,
                "source_run_dir": str(run_dir),
                "prompt_record": str(repair_prompt_path),
                "strategy": "VLM Critic -> revised prompt/spec -> new text-to-image generation",
            }
        )
        progress(
            "自动修订",
            "复查发现基础问题，已收敛成一版修订 prompt；将重新生成一次，完成后停止等待人工判断。",
        )
        repair_command = [sys.executable, str(SCRIPT_DIR / "generate_scientific_image_smoke.py")] + repair_args
        repair_generation = run_subprocess(repair_command)
        repair_run_dir = find_run_dir(repair_generation.get("stdout", "")) or find_run_dir(repair_generation.get("stderr", ""))
        stages.append(
            stage(
                "auto_repair_generation",
                "completed" if repair_generation["returncode"] == 0 else "blocked",
                repair_generation["started_at"],
                repair_generation["finished_at"],
                {"returncode": repair_generation["returncode"], "run_dir": str(repair_run_dir) if repair_run_dir else None},
            )
        )
        if repair_generation.get("stderr"):
            print(repair_generation["stderr"], end="", file=sys.stderr)
        if repair_run_dir:
            ensure_response_json(repair_run_dir)
            auto_repair["repair_run_dir"] = str(repair_run_dir)
        if repair_generation["returncode"] == 0 and repair_run_dir:
            check = read_json(repair_run_dir / "check.json") or {}
            progress(
                "修订生成",
                f"自动修订版已返回并完成机器检查；尺寸 {check.get('width')} x {check.get('height')}，类型 {check.get('mime_type')}。接下来做最终语义复查。",
            )
            repair_semantic_command = [
                sys.executable,
                str(SCRIPT_DIR / "semantic_review_dashscope.py"),
                "--run-dir",
                str(repair_run_dir),
                "--model",
                args.model,
                "--timeout",
                str(args.timeout),
            ]
            repair_semantic = run_subprocess(repair_semantic_command)
            stages.append(
                stage(
                    "auto_repair_semantic_review",
                    "completed" if repair_semantic["returncode"] == 0 else "blocked",
                    repair_semantic["started_at"],
                    repair_semantic["finished_at"],
                    {"returncode": repair_semantic["returncode"], "model": args.model},
                )
            )
            if repair_semantic.get("stderr"):
                print(repair_semantic["stderr"], end="", file=sys.stderr)
            repair_review = read_json(repair_run_dir / "semantic_review.json") or {}
            if repair_semantic["returncode"] == 0:
                progress(
                    "最终复查",
                    f"自动修订版复查完成；可交付判断：{repair_review.get('overall_ready_to_ship')}；建议动作：{repair_review.get('recommended_next_action')}。",
                )
            else:
                blocker = repair_review.get("blocker") if isinstance(repair_review, dict) else None
                progress("最终复查", f"自动修订版 VLM 复查受阻；原因：{summarize_blocker(blocker)}。")
            run_dir = repair_run_dir
            generation = repair_generation
            semantic = repair_semantic
            semantic_review = repair_review
            auto_repair["completed"] = repair_generation["returncode"] == 0
        else:
            auto_repair["completed"] = False
            if repair_run_dir:
                run_dir = repair_run_dir
                generation = repair_generation
            progress("自动修订", "自动修订版生成受阻；已保留首版和修订请求证据，不继续追加轮次。")
    else:
        auto_repair.update(
            {
                "skipped_reason": "disabled" if args.no_auto_repair else repair_reason,
                "source_run_dir": str(run_dir) if run_dir else None,
            }
        )
        stages.append(stage("auto_repair", "skipped", details=auto_repair))

    if not args.no_agent_summary:
        summary = write_agent_review_summary(run_dir, overwrite=args.overwrite_agent_summary)
        stages.append(stage("agent_review_summary", "completed", details=summary))
        progress("复查摘要", "代理复查摘要" + ("已写入；交付判断和下一轮修改 prompt 已记录。" if summary.get("written") else "已保留。"))
    else:
        stages.append(stage("agent_review_summary", "skipped", details={"reason": "no_agent_summary requested"}))

    manifest_command = [sys.executable, str(SCRIPT_DIR / "build_figure_manifest.py"), "--run-dir", str(run_dir)]
    manifest = run_subprocess(manifest_command)
    stages.append(
        stage(
            "manifest_rebuild",
            "completed" if manifest["returncode"] == 0 else "blocked",
            manifest["started_at"],
            manifest["finished_at"],
            {"returncode": manifest["returncode"]},
        )
    )
    if manifest.get("stderr"):
        print(manifest["stderr"], end="", file=sys.stderr)
    progress("交付索引", "本轮图片、机器检查、语义复查和修改建议" + ("已经整理完成，可用于后续追溯或继续改图。" if manifest["returncode"] == 0 else "整理失败，请查看错误输出。"))

    workflow_status = build_status(run_dir, stages, generation, semantic, manifest, initial_run_dir, auto_repair)
    write_json(run_dir / "workflow_status.json", workflow_status)
    if workflow_status["status"] == "completed":
        final_review = read_json(run_dir / "semantic_review.json") or {}
        if final_review.get("overall_ready_to_ship") is True:
            progress("流程状态", "本轮流程已完成；最终复查显示当前图片可交付。")
        elif final_review.get("overall_ready_to_ship") is False:
            progress("流程状态", "本轮流程已完成；最终复查仍建议人工确认或继续修改，本流程不再自动追加轮次。")
        else:
            progress("流程状态", "本轮流程已完成；语义复查没有给出明确交付判断，请人工确认。")
    else:
        progress("流程状态", f"本轮流程结束，状态：{workflow_status['status']}。")

    if generation["returncode"] != 0:
        return generation["returncode"]
    if manifest["returncode"] != 0:
        return manifest["returncode"]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
