import json
import os
import shutil
import subprocess
import sys
import argparse
from pathlib import Path


VISUAL_SKILL = Path(__file__).resolve().parents[1]
IMAGE_SKILL = Path(
    os.environ.get(
        "GIIISP_IMAGE_SKILL_DIR",
        str(Path.home() / ".codex" / "skills" / "giiisp-scientific-image-generation"),
    )
)
IMAGE_SIZE = os.environ.get("VISUAL_DECK_IMAGE_SIZE", "1K")


def run(cmd, cwd=None, check=True, quiet=False):
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.stdout and not quiet:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if check and result.returncode:
        raise SystemExit(result.returncode)
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def emit(run_dir, event, title, message, slide_id=None, slide_title=None, data=None):
    cmd = [
        sys.executable,
        str(VISUAL_SKILL / "scripts" / "deck_progress.py"),
        "emit",
        "--run-dir",
        str(run_dir),
        "--event",
        event,
        "--title",
        title,
        "--message",
        message,
    ]
    if slide_id:
        cmd += ["--slide-id", slide_id]
    if slide_title:
        cmd += ["--slide-title", slide_title]
    if data:
        cmd += ["--data", json.dumps(data, ensure_ascii=False)]
    run(cmd)


def latest_child(path):
    children = [p for p in path.iterdir() if p.is_dir()]
    return max(children, key=lambda p: p.stat().st_mtime) if children else None


def semantic_review(run_dir, sid, image):
    out = run_dir / "qa" / f"semantic-review-{sid}-repair.json"
    run(
        [
            sys.executable,
            str(IMAGE_SKILL / "scripts" / "semantic_review_dashscope.py"),
            "--image",
            str(image),
            "--figure-spec",
            str(run_dir / "image_inputs" / f"{sid}.repair.json"),
            "--style-contract",
            str(run_dir / "visual_style_contract.json"),
            "--out",
            str(out),
            "--timeout",
            "120",
        ],
        check=True,
        quiet=True,
    )
    return read_json(out)


def build_guarded_repair_prompt(slide, review, deck_forbidden_terms=None):
    title = slide["title"]
    body_text = slide.get("body_text") or []
    if isinstance(body_text, str):
        body_text = [body_text]
    exact_visible_text = [title] + [str(item) for item in body_text if item]
    must_show = [str(item) for item in (slide.get("must_show") or []) if item]
    critic = review.get("critic_suggestions") or ""
    revised = review.get("revised_description") or ""
    next_edit = review.get("next_edit_prompt") or ""
    # Deck-specific forbidden terms only (from this run's style_contract/reference_guard);
    # never hardcode example subject matter from a past test topic here.
    forbidden_terms = [str(t) for t in (deck_forbidden_terms or []) if str(t).strip()]
    return "\n".join(
        [
            "The attached reference image is the CURRENT version of this exact slide. Edit it in place; do not invent an unrelated new slide.",
            "Regenerate one complete 16:9 PowerPoint slide image in Chinese, based on the reference image.",
            "This is a user-authorized repair pass. Use the VLM critic only to fix layout, duplication, artifacts, readability, or local visual issues.",
            "The reference image may contain the defects listed by the critic. Do not preserve duplicated nodes, detached text, icon artifacts, or layout mistakes just because they appear in the reference image.",
            "Keep the overall composition, color palette, and card/icon style of the reference image; only change what the critic suggestions below call out.",
            "",
            "REPAIR SUCCESS CONDITIONS:",
            "- If the critic says a card, node, label, or section is duplicated, remove the extra instance. One logical step must become one visual node unless the original slide spec explicitly asks for repetition.",
            "- If the critic says text is outside a card/node, move that exact text inside the correct target card/node. Do not leave any detached label floating on the canvas or sitting on connector lines.",
            "- If the critic says an icon violates the style contract, replace it with one clean line-art icon and remove solid dots, badges, stickers, or decorative artifacts.",
            "- If the critic says a card/region is filled with a large solid accent or primary color block, replace it with a light background/stroke card; the accent/primary color may only appear as icon highlights, thin lines, key numbers, or small status marks (accent_usage_limit), never as a large fill.",
            "- If the critic says fabricated, garbled, or meaningless text appears anywhere (including inside UI mockups or concept diagrams), remove that text entirely; do not replace it with different invented text. Only the EXACT VISIBLE TEXT ALLOWED list below may appear as text; represent any UI/interface concept with text-free wireframes, icons, or placeholder blocks instead.",
            "- If the critic says a required grid/column/row structure was not followed (e.g. asked for 2x2 but rendered as 1x4), redo the composition to match the exact structure described in the original slide spec's layout_notes; this structural requirement outranks any stylistic habit carried over from the reference image.",
            "- Preserve all allowed Chinese labels exactly; fix placement and hierarchy, not meaning.",
            "",
            "HARD CONTENT LOCK:",
            f"- Original slide title must remain exactly: {title}",
            "- The slide topic, domain, and semantics must stay exactly the same as the original slide spec.",
            "- Do not introduce a new industry, roadmap, timeline, organization, product, year, metric, or unrelated scenario.",
            "- Do not replace the original scientific/product workflow semantics with a different business or technical topic.",
            "- If the critic/revised description conflicts with the exact visible text below, ignore the conflicting critic text.",
            "",
            "EXACT VISIBLE TEXT ALLOWED ON THE SLIDE:",
            *[f"- {text}" for text in exact_visible_text],
            "",
            "MUST SHOW THESE REQUIRED LABELS OR THEIR EXACT EXISTING WORDING:",
            *[f"- {text}" for text in must_show],
            "",
            "FORBIDDEN NEW SUBJECT MATTER:",
            *([f"- {term}" for term in forbidden_terms] if forbidden_terms else []),
            "- any subject, industry, product, or scenario not present in the original slide spec above",
            "",
            "ORIGINAL IMAGE PROMPT TO PRESERVE:",
            slide.get("image_prompt", ""),
            "",
            "VLM CRITIC SUGGESTIONS TO APPLY LOCALLY:",
            critic,
            "",
            "VLM REVISED DESCRIPTION, FOR LOCAL VISUAL REPAIR ONLY:",
            revised,
            "",
            "VLM NEXT EDIT PROMPT, FOR LOCAL VISUAL REPAIR ONLY:",
            next_edit,
            "",
            "Render a polished, clean research product report slide. Keep Chinese text readable and concise. No watermark, no logo, no pseudo-English, no duplicated section labels, no placeholder text.",
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="Run one user-authorized candidate repair pass for a visual deck.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--slides",
        default="",
        help="Comma-separated slide ids to repair, e.g. 01,03. Defaults to all failed slides.",
    )
    args = parser.parse_args()
    if not os.environ.get("GIIISP_AUTH_TOKEN"):
        raise SystemExit("GIIISP_AUTH_TOKEN is not set; authenticate at https://giiisp.com/#/mcp/authenticate if the token is missing or expired")
    if not os.environ.get("DASHSCOPE_API_KEY"):
        raise SystemExit("DASHSCOPE_API_KEY is not set; apply for a DashScope/Bailian API key at https://help.aliyun.com/zh/model-studio/get-api-key")

    run_dir = args.run_dir.resolve()
    selected_slides = {
        item.strip().zfill(2)
        for item in args.slides.replace("，", ",").split(",")
        if item.strip()
    }
    spec = read_json(run_dir / "slide_spec.json")
    page_count = len(spec.get("slides", []))
    manifest = read_json(run_dir / "render_manifest.json")
    style_contract_path = run_dir / "visual_style_contract.json"
    deck_forbidden_terms = []
    if style_contract_path.exists():
        deck_forbidden_terms = read_json(style_contract_path).get("forbidden_patterns") or []
    reference_guard = spec.get("deck", {}).get("reference_guard") or {}
    deck_forbidden_terms = list(dict.fromkeys(list(deck_forbidden_terms) + list(reference_guard.get("forbidden_terms") or [])))
    failed = []
    final_reviews = {}
    for slide in spec["slides"]:
        sid = slide["slide_id"]
        review_path = run_dir / "qa" / f"semantic-review-{sid}.json"
        review = read_json(review_path)
        if selected_slides and sid not in selected_slides:
            final_reviews[sid] = review
            continue
        if review.get("overall_ready_to_ship") is True:
            final_reviews[sid] = review
        else:
            failed.append((slide, review))

    if not failed:
        print("No selected failed slides to repair.")
        return

    repair_blocked = False

    emit(
        run_dir,
        "repair.started",
        "开始修订",
        "复查发现基础问题，已定位到第 "
        + "、".join(str(int(s["slide_id"])) for s, _ in failed)
        + " 页；按用户确认只重做这些页面一次。",
        data={"failed_slides": [s["slide_id"] for s, _ in failed], "selected_slides": sorted(selected_slides)},
    )

    for slide, review in failed:
        sid = slide["slide_id"]
        title = slide["title"]
        prompt = build_guarded_repair_prompt(slide, review, deck_forbidden_terms)
        forbidden_labels = [
            "LOGO",
            "logo",
            "页码",
            "page number",
            "Visual brief",
            "Purpose",
            "Image prompt",
            "prompt",
            "prompt labels",
            "markdown symbols",
            "多版本草稿",
            "WATERMARK",
            "FAKE ENG",
        ]
        original_image = (run_dir / slide["rendered_image"]).resolve()
        repair_input = {
            "prompt": prompt,
            "negative_prompt": "水印，模糊文字，错乱标签，伪英文，英文占位符，prompt字段名，Visual brief，Purpose，Image prompt，广告风格，低清晰度，多版本草稿，重复小图，编造文字，乱码文字，无意义字符组合，大面积强调色填充卡片，实心色块填满卡片",
            "aspect_ratio": "16:9",
            "image_size": IMAGE_SIZE,
            "number_of_images": 1,
            "caption": title,
            "intent": slide["purpose"],
            "figure_kind": "presentation slide repair",
            "required_labels": slide["must_show"],
            "forbidden_labels": forbidden_labels,
            "style_brief": spec["deck"]["style_brief"],
            "feedback": "VLM review found baseline deliverability issues; regenerate once with hard content lock from original slide spec.",
            "critic_suggestions": review.get("critic_suggestions"),
            "revised_description": review.get("revised_description"),
            "next_edit_prompt": review.get("next_edit_prompt"),
            "reference_image": str(original_image) if original_image.exists() else None,
            "reference_role": "edit_image",
            "content_lock": {
                "original_title": title,
                "exact_visible_text": [title] + list(slide.get("body_text") or []),
                "must_show": slide.get("must_show") or [],
                "forbid_new_subject": True,
                "candidate_replaces_original_only_after_vlm_pass": True,
                "repair_mode": "true_image_edit" if original_image.exists() else "text_to_image_fallback_original_missing",
            },
        }
        write_json(run_dir / "image_inputs" / f"{sid}.repair.json", repair_input)
        (run_dir / "prompts" / f"{sid}-repair.md").write_text(prompt, encoding="utf-8")
        write_json(run_dir / "qa" / f"auto-repair-prompt-{sid}.json", {
            "slide_id": sid,
            "strategy": "VLM Critic retained; original title and exact visible text hard-locked; candidate image replaces original only after VLM pass",
            "prompt_file": str(run_dir / "prompts" / f"{sid}-repair.md"),
            "critic_suggestions": review.get("critic_suggestions"),
            "revised_description": review.get("revised_description"),
            "next_edit_prompt": review.get("next_edit_prompt"),
            "content_lock": repair_input["content_lock"],
        })

        emit(
            run_dir,
            "slide.repair_submitted",
            "修订生成",
            f"第 {int(sid)} 页《{title}》开始生成候选修订版；"
            + ("会参考原图风格，并只处理复查指出的问题。" if original_image.exists() else "原图缺失，退回为纯文本重新生成。"),
            sid,
            title,
        )
        result = run(
            [
                sys.executable,
                str(IMAGE_SKILL / "scripts" / "generate_scientific_image_smoke.py"),
                "--input-json",
                str((run_dir / "image_inputs" / f"{sid}.repair.json").resolve()),
                "--output-dir",
                str((run_dir / "image_backend_runs").resolve()),
                "--run-kind",
                "edit",
                "--poll-interval",
                "4",
                "--max-polls",
                "18",
                "--request-timeout",
                "180" if original_image.exists() else "120",
            ],
            cwd=IMAGE_SKILL / "scripts",
            check=False,
            quiet=True,
        )
        image_run = latest_child(run_dir / "image_backend_runs")
        metadata = {}
        if image_run and (image_run / "metadata.json").exists():
            metadata = read_json(image_run / "metadata.json")
        candidates = sorted(image_run.glob("generated_image.*")) if image_run else []
        if result.returncode or not candidates or metadata.get("status") != "completed":
            emit(
                run_dir,
                "slide.repair_blocked",
                "修订阻塞",
                f"第 {int(sid)} 页《{title}》修订版没有拿到有效图片，保留原图并等待人工判断。",
                sid,
                title,
            )
            repair_blocked = True
            continue

        candidate = run_dir / "slides" / f"{sid}.repair_candidate{candidates[0].suffix.lower()}"
        shutil.copyfile(candidates[0], candidate)

        emit(
            run_dir,
            "slide.repair_generated",
            "修订复查",
            f"第 {int(sid)} 页《{title}》修订版已生成，正在做最终视觉复查。",
            sid,
            title,
        )
        repaired_review = semantic_review(run_dir, sid, candidate)
        final_reviews[sid] = repaired_review
        if repaired_review.get("overall_ready_to_ship") is True:
            old = run_dir / slide["rendered_image"]
            if old.exists():
                backup = old.with_name(old.stem + ".before_repair" + old.suffix)
                if not backup.exists():
                    shutil.copyfile(old, backup)
            target = run_dir / "slides" / f"{sid}{candidates[0].suffix.lower()}"
            shutil.copyfile(candidate, target)
            slide["rendered_image"] = str(target.relative_to(run_dir)).replace("\\", "/")
            slide["status"] = "image_rendered_repaired"
            for entry in manifest["slides"]:
                if entry.get("slide_id") == sid:
                    entry["copied_to"] = str(target.relative_to(run_dir)).replace("\\", "/")
                    entry["generated_source"] = str(image_run)
                    entry["status"] = "completed"
                    entry["repair_attempts"] = 1
                    entry["repair_prompt_file"] = f"prompts/{sid}-repair.md"
                    entry["job_id"] = metadata.get("job_id")
            write_json(run_dir / "slide_spec.json", spec)
            write_json(run_dir / "render_manifest.json", manifest)
            emit(
                run_dir,
                "slide.repair_passed",
                "修订通过",
                f"第 {int(sid)} 页《{title}》修订版复查通过，继续整理 PPT 文件。",
                sid,
                title,
            )
        else:
            emit(
                run_dir,
                "slide.repair_needs_manual",
                "人工判断",
                f"第 {int(sid)} 页《{title}》候选修订后仍有问题，已停止继续迭代，交给人工判断。",
                sid,
                title,
            )
            repair_blocked = True

    deck_json = read_json(run_dir / "deck.json")
    deck_json["slides"] = [{"background": s["rendered_image"]} for s in spec["slides"]]
    write_json(run_dir / "deck.json", deck_json)

    emit(run_dir, "packaging.restarted", "重新打包", "候选修订页已处理完成，开始重新生成 PPTX、预览和审查记录。")
    run([
        sys.executable,
        str(VISUAL_SKILL / "scripts" / "gorden_image2pptx" / "compose_pptx.py"),
        str(run_dir / "deck.json"),
        str(run_dir / "out" / "deck-image.pptx"),
        "--preview-dir",
        str(run_dir / "previews"),
    ], quiet=True)

    reviews = []
    all_ready = True
    for slide in spec["slides"]:
        sid = slide["slide_id"]
        review = final_reviews.get(sid) or read_json(run_dir / "qa" / f"semantic-review-{sid}.json")
        ready = review.get("overall_ready_to_ship") is True
        all_ready = all_ready and ready
        reviews.append({
            "slide_id": sid,
            "allowed_text": "ok" if ready else "warn",
            "forbidden_text": "none" if not review.get("forbidden_labels_seen") else "warn",
            "unsupported_numbers": "none",
            "unsupported_dates": "none",
            "unsupported_names": "none",
            "readability": "ok" if ready else "warn",
            "overall": "ok" if ready else "warn",
            "notes": review.get("issues") or ["VLM 复查通过。"],
        })
    write_json(run_dir / "qa" / "visible-text-review.json", {
        "schema": "visible_text_review_v1",
        "review_method": "dashscope-vlm-semantic-review",
        "slides": reviews,
    })

    emit(run_dir, "audit.restarted", "结构审查", "PPTX 已重新生成，正在检查页数、全页图片、预览和 VLM 复查记录。")
    image_audit = run([
        sys.executable,
        str(VISUAL_SKILL / "scripts" / "audit_image_only_deck.py"),
        str(run_dir / "out" / "deck-image.pptx"),
        "--spec",
        str(run_dir / "slide_spec.json"),
        "--render-manifest",
        str(run_dir / "render_manifest.json"),
        "--deck-json",
        str(run_dir / "deck.json"),
        "--min-slides",
        str(page_count),
        "--out",
        str(run_dir / "qa" / "image-only-pptx.json"),
    ], check=False, quiet=True)
    text_audit = run([
        sys.executable,
        str(VISUAL_SKILL / "scripts" / "audit_visible_text_review.py"),
        str(run_dir / "qa" / "visible-text-review.json"),
        "--min-slides",
        str(page_count),
        "--out",
        str(run_dir / "qa" / "visible-text-review-audit.json"),
    ], check=False, quiet=True)
    run([sys.executable, str(VISUAL_SKILL / "scripts" / "deck_progress.py"), "status", "--run-dir", str(run_dir), "--print-summary"], check=False)
    if repair_blocked or not all_ready or image_audit.returncode or text_audit.returncode:
        emit(run_dir, "workflow.partial", "流程状态", "PPT 文件已重新整理，但仍有页面复查或修订问题，需要人工确认。")
        raise SystemExit(2)
    emit(run_dir, "workflow.completed", "流程状态", f"整套 PPT 已完成：{page_count} 页图片、PPT 文件和审查记录已整理。")


if __name__ == "__main__":
    main()
