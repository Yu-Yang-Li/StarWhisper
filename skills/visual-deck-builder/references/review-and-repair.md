# PPT 页面审查与修订

[返回技能说明](../SKILL.md)

按本次任务阅读相关部分。本文件中的命令从技能目录执行，文件名与参数按实际输入替换。早期试运行与历史记录不表示外部服务当前可用。

## Core Rules

1. Treat `slide_spec.json` as the source of truth for content, evidence, rendered image paths, and optional editable text metadata.
2. Use an actual raster image generation/editing backend for each full-slide page image. Do not draw target slides with SVG, HTML, Canvas, PIL, matplotlib, PPT shapes, or screenshots and call that image generation.
3. Every slide gets a full-slide rendered image with complete composition and intended page text. This image is the default delivery source.
4. The default final PPTX is image-only: one full-slide picture per slide, with no required editable text boxes.
5. Do not judge image-only decks with the editable reconstruction gate. Use `scripts/audit_image_only_deck.py` for the primary path.
6. Layered editable reconstruction is an opt-in route. Never place editable text directly on top of the same text already burned into an image layer.
7. Keep source-grounded decks source-grounded. If the input is a topic only, allow the model to create plausible structure, but mark unsupported examples/data as assumptions or omit specific claims.
8. Build image prompts from `page_description` plus `visual_style_contract.json`, not by dumping spec field names into a provider prompt. The spec remains the source of truth; the page description is the design-facing handoff.
9. Save reproducible artifacts: `slide_spec.json`, `visual_style_contract.json`, `prompts/`, `slides/`, `deck.json`, `render_manifest.json`, PPTX, previews, and QA notes.
10. Keep the public workflow centered on this skill's own input routing, source grounding, spec planning, full-slide image QA, packaging, and release gates. Treat any vendored composer utility as an implementation detail, not as the product identity.
11. Never write API tokens into skill files, manifests, logs, prompts, or final replies. Read tokens from environment variables only.
12. Never stretch a generated slide image to fit PPTX. For a `16:9` deck, the source slide image itself must be close to `16:9`; if a backend returns `1:1` or another wrong ratio, mark the slide as failed and regenerate with a native wide backend or stop for user choice.
13. For Giiisp-backed deck generation, default to `imageSize: "1K"` unless the user explicitly asks for high-resolution output. A `2K` pass is an optional final regeneration/upscale step after the slide has passed VLM review, not the first default generation step.
14. Lock a shared style anchor after slide 1. Generate slide 1 with no reference image. Once it passes the basic image checks, set `visual_style_contract.reference_image` to slide 1's rendered image path and generate every remaining slide with that image attached as `referenceImage`/`imageBase64` (Giiisp's `--reference-image`, `reference_role: style_reference`). This is a real image-to-image call, confirmed against the live Giiisp `/api/generate-async` endpoint (undocumented for that endpoint but functionally honored; attaching an image measurably slows the initial request, so use a longer request timeout, about 150-180s, whenever a reference image is attached). The reference image only carries palette/typography/card/icon/texture language; each slide's own composition, chart type, and text still come from that slide's `page_description`. This directly targets cross-slide style drift, which was the single biggest visible-inconsistency gap found when auditing against banana-slides' own reference-image-based generation.
15. User-authorized single-slide repair must also use true image-to-image editing when the backend supports it: attach the slide's current rendered image as the reference image and instruct the model to edit it in place, not regenerate from a blank prompt. Only fall back to text-to-image if the original rendered image file is missing.
16. Build `visual_style_contract.json` with `scripts/build_style_contract.py --preset <key>` from the preset library in `scripts/style_presets.py`, not as a hand-written short style label. A style contract must carry explicit hex colors, named fonts, and spacing/accent-usage percentages; a one-line adjective description ("clean, modern, blue and white") gets reinterpreted differently by the image model on every independent per-slide call and is a known source of cross-slide drift. Run `scripts/validate_style_contract_richness.py` before release; treat a `fail` status the same as any other blocking QA failure.
17. Write each slide's design brief as `extra_fields` (`visual_elements`, `visual_focus`, `layout_notes`, `speaker_notes`) instead of a single freeform paragraph. `scripts/build_slide_image_prompt.py` only feeds `visual_elements` + `visual_focus` + `layout_notes` into the image prompt; `speaker_notes` never reaches the image model. Freeform `page_description` remains a compatibility route for older decks written before this field existed, not the default for new decks.
18. Before the first image generation call, run `scripts/stylist_refine_slide.py` once per slide as an independent aesthetic-refinement pass: it takes the draft `extra_fields` plus the style contract and returns a refined version that adds concrete visual detail without changing semantics, required labels, or exact visible text. If it is blocked (no `DASHSCOPE_API_KEY`, network failure), proceed with the unrefined draft and note the skip; it is a quality enhancement, not a hard gate that blocks generation.
19. Review the actual slide against `visual_style_contract.json`. Pass it to the bundled `../giiisp-scientific-image-generation/scripts/semantic_review_dashscope.py` with `--style-contract`; this adds `style_contract_compliance`, `formal_ppt_feel`, and `template_overfit` to the five scientific-image quality axes. All eight axes must return PASS before a candidate can be accepted. Record manual review separately when the service is unavailable.
20. Generate content with `scripts/generate_slide_spec.py` (topic -> outline -> per-page description, mirroring banana-slides' two-stage `get_outline_generation_prompt` + `get_page_description_prompt`), not by having the agent hand-write `body_text`/`extra_fields` ad hoc. Ad hoc hand-written content is a known source of two failure modes confirmed against a real generated deck: pages reduced to 3-5 isolated 2-4 character labels (banana's default density floor is 2-6 short phrases/sentences per page, `DETAIL_LEVEL_SPECS` in its `prompts.py`), and a cover page with no subtitle/presenter information (banana requires "title, subtitle, and presenter information" on page 1). Only fall back to hand-writing the spec when `DASHSCOPE_API_KEY` is unavailable, and in that case explicitly apply the density and cover rules below yourself.
21. Every slide carries both `page_role` and `layout_profile`. `page_role` is `cover`, `content`, `data`, `comparison`, `timeline`, `summary`, or `section_divider`; `layout_profile` must mirror banana-slides' 9-field template-analysis schema (`template_role`, `layout_structure`, `content_capacity`, `text_regions`, `image_regions`, `visual_density`, `style_keywords`, `color_palette`, `notes`). A deck is not a chain of infographics: with 4+ slides, at least one slide must be `page_role: content` (text-forward: bullet lists, paragraphs, two-column text — not a diagram), and no `page_role` other than `cover` may repeat 4+ times in a row. Run `scripts/validate_content_density_and_roles.py` before release and treat `fail` as blocking, `warn` as needing a human look.
22. Run `scripts/build_slide_image_prompt.py`'s built-in anti-drift guards seriously: exact_visible_text is a whitelist, not a repetition requirement (one label = one instance unless the spec explicitly asks for repetition); one logical step in a flow/decision/mechanism diagram must become exactly one visual node (never duplicate a same-named card to fill space); every piece of visible text must sit inside its owning card/node/title/legend region, never floating outside a shape or sitting on a connector line; icons must follow the style contract's `icon_style` exactly (no solid dots/badges/stickers substituting for a required line-art icon style). These were added after a real repair failure (duplicated "生成候选版" card, floating "复查通过才替换" text, solid-dot lock icon) and must not be weakened.
23. When repairing a slide with true image-to-image editing, explicitly tell the model the reference image may itself contain the defects the critic listed and must not be preserved just because they appear in the reference (`repair_visual_deck.py`'s `build_guarded_repair_prompt`). Editing "in the style of the reference" does not mean copying its mistakes.
24. Distinguish an authentication/access-code blocker (401, invalid/expired token, provider auth error codes) from a generic "no image returned" blocker. On an auth blocker, stop the whole run immediately with a message pointing to the token refresh URL; do not keep retrying other slides against a token that will fail on every subsequent call.
25. After the main generation pass, always keep the VLM/visible-text review and give page-level repair advice, but do not start a repair pass by default. Wait for the user to confirm whether to repair and which pages to repair. For a user-authorized partial repair, invoke `scripts/repair_visual_deck.py <run_dir> --slides 01,03` so only the selected failed slides are regenerated as candidates. For regression tests or explicitly authorized all-failed-slide runs, set `VISUAL_DECK_AUTO_REPAIR=1` to invoke the single repair pass. Keep internal subprocess output (raw JSON dumps, provider logs) out of the user-visible stream; only the `deck_progress.py emit` narrative messages are user-facing.

## Default Intermediate Artifacts

Every publishable deck run must keep these artifacts under one run directory:

- `input_summary.md`: user intent, source materials, audience, page count, language, style, and assumptions.
- `visual_style_contract.json`: deck-level design contract built from `scripts/build_style_contract.py` (exact hex palette, named fonts, spacing/accent-usage percentages, icon/chart/rendering rules, forbidden patterns), plus whether the style came from the preset library, a template, an extracted description, or a user instruction. Once slide 1 renders, add `reference_image` (absolute path to slide 1's rendered image) and a `reference_policy` note so every later slide's prompt and generation call pick up the same style anchor.
- `slide_spec.json`: deck route, slide list, per-slide purpose, title, exact visible text, evidence, visual brief, `page_role`, banana-style `layout_profile`, `extra_fields` (visual_elements/visual_focus/layout_notes/speaker_notes), image prompt, status, and rendered image path.
- `prompts/NN-slide.md`: the exact prompt used for each slide image.
- `prompts/NN-stylist.json`: the independent aesthetic-refinement result for that slide (`scripts/stylist_refine_slide.py`), or its blocker if skipped.
- `qa/style-contract-richness.json`: `scripts/validate_style_contract_richness.py` output.
- `qa/content-density-roles.json`: `scripts/validate_content_density_and_roles.py` output (text density floor, cover anatomy, page-role diversity).
- `slides/NN.png|jpg`: generated or accepted full-slide images.
- `render_manifest.json`: image backend, prompt file, generated source, copied slide image, status, blocker if any.
- `stream_events.jsonl`: user-visible progress events plus technical evidence in `data`.
- `deck.json`: image-only PPTX composition source.
- `out/*.pptx`: final image-only deck unless editable output was explicitly requested.
- `previews/`: rendered preview images for QA and user review.
- `qa/semantic-review-NN.json` when a VLM or multimodal reviewer is available: per-slide readiness, issues, and repair prompt.
- `qa/visible-text-review.json`: per-slide visible text, forbidden terms, unsupported numbers/dates/names, readability, and notes.
- `qa/image-only-pptx.json`: PPTX structure audit.
- `qa/visible-text-review-audit.json`: visible-text review completeness audit.
- `deck_workflow_status.json`: final workflow status built from the above artifacts.

For each completed run, call:

```powershell
python scripts/deck_progress.py status --run-dir "<run_dir>" --print-summary
```

For progress events, call:

```powershell
python scripts/deck_progress.py emit --run-dir "<run_dir>" --event slide.generated --title "页面生成" --message "第 3 页《方法路线》已生成，正在复查错字、伪英文、水印和版面拥挤。" --slide-id 03 --slide-title "方法路线"
```

## Revision Behavior

If the user asks to modify a generated deck or a single slide, create a new revision run instead of overwriting the prior run. The revision run must reference the parent run and preserve lineage in `input_summary.md`, `slide_spec.json`, and `deck_workflow_status.json`.

Default revision route:

1. Read the parent `slide_spec.json`, `render_manifest.json`, previews, QA reports, and the user feedback.
2. Scope the change to affected slide ids when possible; do not regenerate the whole deck for a one-slide edit.
3. Write updated prompt files for affected slides and record the user feedback.
4. Regenerate affected slide images.
5. Rebuild `deck.json`, PPTX, previews, visible-text review, and audits.
6. If QA finds basic issues on affected slides such as missing required text, forbidden terms, watermark, obvious typo, unsupported number, or severe layout crowding, report the issue and suggested repair first. Start a repair pass only after the user confirms, and pass the confirmed slide ids through `--slides`.
7. Keep the VLM Critic output, but the user-authorized repair prompt must hard-lock the original slide title, original exact visible text, source domain, and required labels. It must forbid new subject matter and forbid replacing the original domain semantics with another industry scenario.
8. Generate the repair image as a candidate first, using true image-to-image editing: attach the slide's current rendered image as the reference image (`reference_role: edit_image`) so the model edits the existing page instead of starting over from text alone. Only replace the original slide image after the candidate passes VLM review. If the candidate fails or drifts, preserve the original image, mark the run `partial`, and return the deck to the user for judgment.
9. Stop after one user-authorized repair pass and return the deck to the user for judgment.

User-facing revision stream example:

```text
[21:42 | 修改识别]
收到修改：只影响第 5 页《结果对比》，我会保留其他页面不重做。

[21:43 | 修改生成]
第 5 页新版图片已生成，正在复查文字、数字和版面。

[21:44 | 修订建议]
复查发现第 5 页有一个多余英文标签，已记录建议；等你确认后再重做这一页。

[21:45 | 用户确认修订]
第 5 页候选修订版复查通过，PPTX 和预览已重新打包。
```
