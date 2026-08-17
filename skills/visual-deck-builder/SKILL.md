---
name: visual-deck-builder
description: >-
  Build image-model-driven PPT decks from a user topic, source materials, papers, reports, notes, datasets, existing slides, or style references. Use when the user asks to create, redesign, package, or QA a PowerPoint deck where every slide should first be generated as a high-quality full-slide image and packaged into PPTX with manifests, previews, and evidence-backed QA. Layered editable reconstruction is optional and only used when explicitly requested.
  This is the general deck-building entry. Prefer narrower skills only when the request is specifically an academic/research PPT workflow, a prompt/method clinic, a sci-employee course/video wrapper, or a recurring Chronicle research/script loop.
  StarWhisper astronomy overlay: build time-domain / telescope-agent decks with conservative claims.
---

# Visual Deck Builder

## StarWhisper astronomy overlay

This copy is adapted for astronomy research and telescope-agent work.
**Read [`astronomy.md`](astronomy.md) before following generic biomedical / clinical defaults in the rest of this file.**

Default literature route: NASA ADS, then arXiv `astro-ph.*`, then the original skill's search backend if credentials exist.
Do not claim a real hardware observing loop, a discovery, or a referee-ready result unless the user supplied that evidence.


Create high-quality visual PPT decks from whatever the user supplies: a topic, a document, a paper, a folder of materials, a style reference, or existing slide images. The main workflow is **spec first, page-description driven, full-slide image first, then image-only PPTX packaging**:

```
input -> intent/source resolver
      -> generate_slide_spec.py: LLM outline (title + page_role + banana 9-field layout_profile per slide,
                                cover = title/subtitle/presenter only)
                                 -> LLM per-page description (density-floor body_text + layout_profile + extra_fields)
      -> visual_style_contract.json from the preset library (exact hex/fonts/percentages)
      -> independent Stylist pass refines extra_fields against the style contract (aesthetics only, no semantic change)
      -> banana-style per-slide image prompt (演讲者备注 excluded, anti-duplication/anti-floating-text guards) -> full-slide image
      -> slide 1 image locked as shared style-anchor reference for slides 2..N
      -> deck.json -> image-only PPTX
      -> preview + render manifest + QA report (content density/page-role diversity, style-contract compliance, formal-PPT-feel, template-overfit axes)
```

The deck is not a fixed "paper mode" or "topic mode". Route by the user input, but the default final slide is a complete full-slide image inside PPTX. Use layered editable reconstruction only when the user explicitly asks for editable PPT objects and accepts the extra QA loop.

## Boundary With Related PPT Skills

Treat this skill as the default entry for actual PPT/deck creation, redesign, packaging, or QA. Route away only when the user's request clearly matches a narrower asset:

- Use `research-ppt-generation` when the task is specifically an academic or research deck and needs paper/PDF-to-narrative planning, research audience structure, figure grounding, or academic template logic.
- Use `ppt-image-generation` when the user is asking for slide-image prompting strategy, image-model page briefs, route selection, or diagnosis of weak PPT image prompts, not for an end-to-end deck package.
- Use `sci-employee-ppt-making` only for the sci-employee wrapper: research training course pages, PPT-to-video,讲稿/字幕/配音闭环, or persona-specific teaching delivery.
- Use `chronicle-ai-ppt-open-source-research-loop` only for the recurring AI PPT open-source ranking/gallery loop.
- Use `chronicle-weekly-ai-deck-script` only to turn an existing image-based AI frontier deck into a clean page-ordered speaker script.

Do not create another PPT skill for a new deck style until this routing fails for at least two real tasks.

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
19. VLM review must judge `style_contract_compliance`, `formal_ppt_feel`, and `template_overfit` in addition to the original five axes. Pass `--style-contract visual_style_contract.json` to `scripts/semantic_review_dashscope.py` so the reviewer can compare the actual image against the contract's real hex/font/spacing values instead of guessing at general aesthetics.
20. Generate content with `scripts/generate_slide_spec.py` (topic -> outline -> per-page description, mirroring banana-slides' two-stage `get_outline_generation_prompt` + `get_page_description_prompt`), not by having the agent hand-write `body_text`/`extra_fields` ad hoc. Ad hoc hand-written content is a known source of two failure modes confirmed against a real generated deck: pages reduced to 3-5 isolated 2-4 character labels (banana's default density floor is 2-6 short phrases/sentences per page, `DETAIL_LEVEL_SPECS` in its `prompts.py`), and a cover page with no subtitle/presenter information (banana requires "title, subtitle, and presenter information" on page 1). Only fall back to hand-writing the spec when `DASHSCOPE_API_KEY` is unavailable, and in that case explicitly apply the density and cover rules below yourself.
21. Every slide carries both `page_role` and `layout_profile`. `page_role` is `cover`, `content`, `data`, `comparison`, `timeline`, `summary`, or `section_divider`; `layout_profile` must mirror banana-slides' 9-field template-analysis schema (`template_role`, `layout_structure`, `content_capacity`, `text_regions`, `image_regions`, `visual_density`, `style_keywords`, `color_palette`, `notes`). A deck is not a chain of infographics: with 4+ slides, at least one slide must be `page_role: content` (text-forward: bullet lists, paragraphs, two-column text — not a diagram), and no `page_role` other than `cover` may repeat 4+ times in a row. Run `scripts/validate_content_density_and_roles.py` before release and treat `fail` as blocking, `warn` as needing a human look.
22. Run `scripts/build_slide_image_prompt.py`'s built-in anti-drift guards seriously: exact_visible_text is a whitelist, not a repetition requirement (one label = one instance unless the spec explicitly asks for repetition); one logical step in a flow/decision/mechanism diagram must become exactly one visual node (never duplicate a same-named card to fill space); every piece of visible text must sit inside its owning card/node/title/legend region, never floating outside a shape or sitting on a connector line; icons must follow the style contract's `icon_style` exactly (no solid dots/badges/stickers substituting for a required line-art icon style). These were added after a real repair failure (duplicated "生成候选版" card, floating "复查通过才替换" text, solid-dot lock icon) and must not be weakened.
23. When repairing a slide with true image-to-image editing, explicitly tell the model the reference image may itself contain the defects the critic listed and must not be preserved just because they appear in the reference (`repair_visual_deck.py`'s `build_guarded_repair_prompt`). Editing "in the style of the reference" does not mean copying its mistakes.
24. Distinguish an authentication/access-code blocker (401, invalid/expired token, provider auth error codes) from a generic "no image returned" blocker. On an auth blocker, stop the whole run immediately with a message pointing to the token refresh URL; do not keep retrying other slides against a token that will fail on every subsequent call.
25. After the main generation pass, always keep the VLM/visible-text review and give page-level repair advice, but do not start a repair pass by default. Wait for the user to confirm whether to repair and which pages to repair. For a user-authorized partial repair, invoke `scripts/repair_visual_deck.py <run_dir> --slides 01,03` so only the selected failed slides are regenerated as candidates. For regression tests or explicitly authorized all-failed-slide runs, set `VISUAL_DECK_AUTO_REPAIR=1` to invoke the single repair pass. Keep internal subprocess output (raw JSON dumps, provider logs) out of the user-visible stream; only the `deck_progress.py emit` narrative messages are user-facing.

## Runtime Streaming Updates

For long deck runs, give short user-visible progress updates from real run signals only. Do not emit generic rules, promises, or filler such as "processing" without evidence. Keep the existing generation, packaging, and QA calls unchanged; this section only controls what the user sees while waiting.

Use `scripts/deck_progress.py` to record progress when a run directory exists. It writes `stream_events.jsonl` and can build `deck_workflow_status.json`; it does not replace any generation, packaging, or QA call.

Use these evidence sources when they exist: input route, page count, current slide id, `slide_spec.json` status, prompt files written, image paths in `render_manifest.json`, PPTX path, preview paths, audit status, slide counts, picture/textbox counts, issue counts, repair change counts, and blocker records.

Only emit a user-facing "page passed" update after a real visual review artifact exists for that slide, such as a VLM semantic review, a documented human review, or `qa/visible-text-review.json` entry with `overall: ok`. If an image has only returned from the provider or passed file-structure checks, say "图片已生成，正在复查" or "基础产物检查完成，等待视觉复查"; do not call it deliverable yet.

Use a 30-minute visible progress window for deck work. Emit the first update when a real route or run directory exists. For multi-slide runs, optimize the stream around completed slide-level value: when each slide image finishes, the public message reports the slide number/title, what that page contributes to the deck, whether it can be previewed, and the next slide or packaging step. Also stream useful quality-processing milestones when they are backed by real artifacts: page plan ready, prompt ready for a named slide, generated image ready for review, preview/text/number/watermark/layout review started, review passed, review failed with a human-readable reason, slide retry started, PPT packaging started, structure audit started, and audit passed or failed. Keep job ids, provider status, elapsed seconds, raw wait counts, image dimensions, and machine checks in `stream_events.jsonl.data` or QA files. A waiting heartbeat is useful only when no new slide or artifact has appeared for about 30 seconds; that heartbeat must say which slide is still waiting, how long the user has waited, and what artifact should appear next. At 5, 10, 20, and 30 minutes, summarize completed slide count, pending slide titles, package status, QA status, and preview/output paths. At 30 minutes, return one of `completed`, `partial`, or `blocked`; for partial/blocked states, list finished slides, pending slide titles, the human-readable blocker, and the next exact action. Never use the heartbeat to repeat a generic process label.

Default public stream shape for Chinese deck requests:

```text
[21:24 | 任务识别]
已识别为 8 页科研汇报稿：先整理页面结构和证据边界，再逐页生成。

[21:25 | 页面规划]
8 页结构已定：封面、问题背景、方法路线、关键结果、机制解释、对比分析、结论、讨论。

[21:26 | 单页生成]
第 3 页《方法路线》已进入生成；这一页负责把实验路径和关键节点讲清楚。

[21:27 | 页面复查]
第 3 页图片已生成，正在检查错字、伪英文、水印、数字和版面拥挤。

[21:28 | 页面通过]
第 3 页《方法路线》复查通过；现在继续生成第 4 页《关键结果》。

[21:35 | 打包开始]
全部页面图片已就绪，开始整理 PPTX、预览图和审查记录。

[21:36 | 结构审查]
PPTX 已生成，正在检查页数、全页图片、预览和可见文字风险。

[21:37 | 流程状态]
整套 PPT 已完成：8 页图片、PPT 文件和审查记录已整理。
```

Separate user-facing stream copy from technical evidence. User-facing copy should sound like a product assistant, not an implementation note. Do not put implementation labels, provider internals, package internals, or machine-check field names into public messages unless the user explicitly asks for technical/debug output. Put those terms in `stream_events.jsonl.data`, QA reports, or a technical appendix. A good user-facing first update names the deck shape: page count, page titles or roles, and what will appear first.

If an image provider returns an authentication or access blocker before a `job_id` exists, emit that blocker as the progress update. Use event names such as `auth.blocked` or `provider.blocked`, include the status code and provider error code when available, and do not invent `job_id`, poll count, generated image paths, or completion language. If another real image backend is available and used for the same deck, record a backend switch event in `render_manifest.json` and the user-visible stream, with the original blocker kept as evidence.

Good update examples:

- `已开始生成 8 页汇报稿：先出封面和问题页，完成一页就给你看一页。`
- `页面规划已完成：8 页结构已定，先生成《研究背景》。`
- `第 3 页《实验路线》已经进入生成，完成后会先给你看这一页，再继续打包。`
- `第 3 页图片已生成，正在审查是否有错字、多余数字、水印和版面拥挤。`
- `第 3 页审查通过：文字和版面可用，可以进入 PPT。`
- `第 1 页《研究背景》已生成，可以预览；现在继续生成第 2 页《核心方法》。`
- `第 2 页还在排队生成，已等待约 30 秒；目前还没有新图片，下一条会在本页完成或出现阻塞时更新。`
- `图片服务没有接受当前凭证，这页还没有开始生成；我已停在这里，避免生成假页面。`
- `第 4 页《结果对比》已完成，版面检查通过；现在继续整理最后的 PPT 文件。`
- `结构检查发现第 5 页图标和文字太挤，正在修正位置后重新合成。`
- `第 5 页拥挤问题已修正，复查通过；继续生成下一页。`
- `第 6 页预览里有文字不清晰，我会重做这一页，不把这个版本放进最终 PPT。`
- `第 7 页还没有拿到有效图片，我会停在这里，不用假页面补位。`

Avoid updates that are only process labels:

- `正在生成 PPT。`
- `正在提升质量。`
- `快完成了。`
- `正在检查。`

## Upstream Alignment Checks

When optimizing, packaging, or release-checking this skill, include an upstream consistency check in every 20-minute work loop when a local `banana-slides` checkout is available. This check is not user-facing runtime copy; it is an engineering guardrail.

Compare against upstream `banana-slides` for these contracts:

- Description fields: upstream default description fields are `视觉元素`、`视觉焦点`、`排版布局`、`演讲者备注`; image prompts should use only the image-safe fields and keep speaker notes out of image generation.
- Image prompt shape: upstream passes a page description plus style/template guidance into a full-slide PPT image prompt. This skill keeps `visual_style_contract.json` as the source file, but serializes it into the provider prompt as upstream-style `<page_style>` guidance; it must not regress to raw spec-field dumps.
- Reference image generation: upstream `generate_image` accepts a main reference image and additional references; this skill's slide-1 style anchor for slides 2..N is the Giiisp-compatible equivalent and must keep reference text/semantics from bleeding into later slides.
- Image editing: upstream `edit_image` uses the current slide image as the reference and asks the model to preserve content/style while applying the edit instruction. This skill's user-authorized repair must remain candidate-first, use `reference_role: edit_image`, and replace the original only after VLM review passes.
- Intentional deltas: this skill adds科研汇报 style presets, Giiisp auth/runtime handling, image-only PPTX QA, VLM release gates, and one user-authorized repair pass. Treat these as product-specific extensions, not upstream drift.

Record any mismatch in the run notes or release audit before calling the package publishable.

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

## Input Routing

Classify input before generating:

- `topic_only`: user gives a topic, goal, audience, or theme but no source material.
- `source_grounded`: user provides papers, PDFs, reports, notes, webpages, data, or extracted text.
- `style_reference`: user provides screenshots, templates, brand images, color references, or old PPTs.
- `existing_slide_edit`: user provides slide images or PPT/PDF pages to redesign, continue, or make editable.
- `mixed`: combine source grounding with style references and user constraints.

Read [references/input-routing.md](references/input-routing.md) when routing is unclear.

## Standard Workflow

### Script Requirements

Local packaging and QA scripts need a Python environment with:

- `Pillow`
- `python-pptx`

If imports fail, switch to an environment that has these packages or install them before running `scripts/audit_image_only_deck.py`, `scripts/compose_layered_deck.py`, `scripts/render_layered_preview.py`, `scripts/validate_visual_deck.py`, or optional editable reconstruction tests.

### 1. Resolve Requirements

Infer reasonable defaults unless the user asks for exact choices:

- audience and use case
- page count
- language
- aspect ratio
- visual style
- source citations/evidence requirements
- whether existing slide images should be converted, continued, or redesigned

If the user supplies no template or style reference, build `visual_style_contract.json` from the preset library instead of asking by default or hand-writing a short label:

```powershell
python scripts/build_style_contract.py --list
python scripts/build_style_contract.py --preset scientific_tool_report --out visual_style_contract.json
```

Pick the preset closest to the audience/topic (`scientific_tool_report` is the default for research-tool product decks; `academic_formal_report`, `modern_tech_briefing`, and `business_minimal_report` cover other common cases). If none fit, add a new preset to `scripts/style_presets.py` at the same precision level (explicit hex colors, named fonts, spacing/accent-usage percentages) rather than falling back to a vague description. A missing template image is not a blocker when a style contract exists.

If a missing detail changes the deck materially and cannot be inferred, ask one concise question.

### 2. Build Content Plan

For topic-only input, generate outline + per-page content with `scripts/generate_slide_spec.py` instead of hand-writing it:

```powershell
python scripts/generate_slide_spec.py --topic "<user topic/idea>" --audience "<audience>" --title "<deck title>" --detail-level default --reference-file "<source.md>" --out slide_spec.json
```

This runs banana-slides' own two-stage content pipeline (LLM outline with simple/part-based outline support, `page_role` assignment, and cover-page rules, then LLM per-page description with a density floor and `extra_fields`). Repeated `--reference-file` values are prepended to the outline and description prompts as banana-style `<uploaded_files>` XML, so source material is visible to the same planning chain rather than pasted ad hoc. It requires `DASHSCOPE_API_KEY`; if unavailable, fall back to hand-writing the spec and manually apply the density (step 3) and page-role (step 3) rules below.

For source-grounded input:

- extract claims, entities, methods, results, figures, and must-use terminology
- preserve exact names and numbers
- record where each important claim came from
- avoid claims not supported by the source
- feed the extracted claims into `generate_slide_spec.py --topic` as the source material, or hand-write the spec directly from the source when the content is already fully drafted

For topic-only input, `generate_slide_spec.py` already applies:

- a useful narrative from the topic and user intent
- conceptual frameworks, examples, and comparisons over fabricated numbers
- flagging any invented case studies, market data, or citations as assumptions unless verified (add this as an explicit instruction in `--topic` when it matters)

### 3. Write `slide_spec.json`

Every slide must include:

- `slide_id`
- `purpose`
- `title`
- `body_text`: uses banana-slides' `DETAIL_LEVEL_SPECS` modes. Default is `--detail-level default`: 2-6 short phrases/sentences per slide, each phrase around 15-20 characters. `concise` and `detailed` are available for explicit requests, but default deck generation stays at `default`; never reduce a default slide to 1-2 isolated 2-4 character keyword tags.
- `page_role`: `cover` | `content` | `data` | `comparison` | `timeline` | `summary` | `section_divider` (mirrors banana-slides' `template_role`). The first slide must be `cover`. A deck with 4+ slides must include at least one `content` (text-forward) slide, and no role other than `cover` may repeat 4+ times in a row
- `layout_profile`: the full banana-slides template-analysis schema equivalent (`template_role`, `layout_structure`, `content_capacity`, `text_regions`, `image_regions`, `visual_density`, `style_keywords`, `color_palette`, `notes`). This is required for new decks; `page_role` alone is only partial alignment.
- `must_show`
- `visual_brief`
- `page_description`: keep for backward compatibility, but write `extra_fields` as the primary design brief for new decks
- `extra_fields`: `visual_elements`, `visual_focus`, `layout_notes`, `speaker_notes` (see step 4). `layout_notes` must name the text region(s) and image region(s) separately (e.g. "标题区在顶部；正文文字区占左侧60%，中等容量；配图区占右侧40%，小尺寸") so the composition is not defaulted to a diagram-only layout
- `delivery_mode`: usually `image_only`
- `image_prompt`
- `rendered_image` once generated
- `editable_text` only when an optional overlay or editable reconstruction is requested
- `evidence` when source-grounded
- `status`

The cover slide (`page_role: cover`) must contain title + subtitle + presenter/team/date information in `body_text`, never a bare title alone and never a diagram (banana-slides requires "title, subtitle, and presenter information" on page 1).

Run the content-level QA gate before generating any images:

```powershell
python scripts/validate_content_density_and_roles.py slide_spec.json --out qa/content-density-roles.json
```

Treat `fail` as a blocking issue (missing cover anatomy, density floor violated, no `content`-role slide in a 4+ slide deck, or every slide using a diagram role) — fix the spec, not just the image prompt, since the image model is only ever as good as the content brief it is given.

Use the schema and examples in [references/slide-spec-contract.md](references/slide-spec-contract.md).

### 4. Generate Page Descriptions And Full-Slide Image Prompts

Create a design-facing brief for every slide before creating the image prompt: write `extra_fields` (`visual_elements`, `visual_focus`, `layout_notes`, `speaker_notes`) instead of one freeform paragraph. This mirrors banana-slides' `description_extra_fields`/`image_prompt_extra_fields` split: the design brief is structured into independently useful fields, and only the image-safe subset feeds the image prompt. `speaker_notes` is presenter-only and must never appear in the image prompt or on the rendered slide. Older decks that only have a freeform `page_description` string remain supported through a compatibility route, not the default for new decks.

#### 4a. Independent Stylist Pass

Before the first image generation call for a slide, run the aesthetic-refinement pass:

```powershell
python scripts/stylist_refine_slide.py slide_spec.json visual_style_contract.json --slide-id 01 --out prompts/01-stylist.json
```

This calls a text model with the draft `extra_fields` plus the full style contract and asks it to enrich concrete visual detail (how the contract's colors/fonts/spacing actually apply to this page) without changing semantics, required labels, or exact visible text. If `changed: true`, write `refined_extra_fields` back into the slide's `extra_fields` before building the image prompt. If the pass is `blocked` (missing `DASHSCOPE_API_KEY`, network failure), proceed with the unrefined draft and record the skip in the run's technical evidence; do not block generation on this step.

Then create self-contained prompt files for each slide:

- `prompts/NN-slide.md`: full finished slide image with complete composition, strong aesthetics, all intended page text, and no placeholders.

Each prompt must include:

- slide aspect ratio and language context
- the full `page_description`, not raw spec field labels
- the deck `visual_style_contract`, serialized into an upstream-style `<page_style>` block
- exact visible text to render, separated from design instructions
- style, color, typography, spacing, and layout-density constraints
- what must appear on the final full-slide image
- what text must be exact or should be simplified if too small
- reference image role if provided: `style_reference` (slide 1 style anchor for slides 2..N), `edit_image` (repair pass editing the current slide), `preserve_structure`, or `use_elements`

Use `scripts/build_slide_image_prompt.py` when `slide_spec.json` and `visual_style_contract.json` exist:

```powershell
python scripts/build_slide_image_prompt.py slide_spec.json visual_style_contract.json --slide-id 03 --out prompts/03-slide.md
```

Keep internal planning metadata out of provider-visible slide text. Do not paste field labels such as `visual_brief`, `purpose`, `image_prompt`, `must_show`, "Visual brief", or "Purpose" into the image prompt as visible content. Convert them into natural design instructions, state the exact allowed slide text separately, and add those internal field labels to forbidden text for review and regeneration.

Read [references/aesthetic-generation.md](references/aesthetic-generation.md) before changing the prompt shape, adding a new image backend, or diagnosing weak PPT aesthetics.

For generated decks, the rendered full-slide image is the PPT source. For existing slide images, use the user's slide image directly only when the user asks for packaging or style-preserving continuation. Show generated images with `view_image` before accepting them. A local path in a prompt is not a real image input.

For `existing_slide_edit`, each slide spec must include `source_visual_target` or the deck must include `source_visual_targets`. This proves which source page is being packaged, redesigned, or optionally reconstructed and prevents accidental reuse of an old run artifact.

For any `style_reference` route, add `reference_guard` to the deck or slide spec before generating:

- `style_reference_policy`: usually `layout_density_only`, `palette_only`, or `preserve_structure`.
- `allowed_text_source`: usually `slide_spec`.
- `forbidden_terms`: brands, years, metrics, topic words, and example labels that appear in the reference but must not migrate into this deck.

When using any public example as a reference, inherit only layout density, hierarchy, and reconstruction mechanics. Do not inherit its topic, brands, dates, numbers, or example copy.

### 5. Render Full-Slide Images With Image Model

Use the best available backend in the current environment:

- Codex `image_gen` if available in chat.
- Giiisp/SiTian Imagine when `GIIISP_AUTH_TOKEN` is set. If the user has no token, point them to `https://giiisp.com/#/mcp/authenticate` to apply for or refresh Giiisp authentication.
- Project-specific image provider if working inside an app.

Read [references/image-runtime.md](references/image-runtime.md) before calling Giiisp or another HTTP image API.

Before optional extraction, run a backend/input preflight:

```powershell
python scripts/preflight_extraction_backend.py --target visual_targets/01.png --run-root . --slide-id 01 --out qa/preflight-01.json
```

If preflight reports `blocked`, stop and record the blocker. Do not generate replacement slides or layers with local code as a substitute for image-model output.

For every generated slide image, update `render_manifest.json`:

```json
{
  "slides": [
    {
      "slide_id": "03",
      "prompt_file": "prompts/03-slide.md",
      "backend": "giiisp|codex-imagegen|project-provider",
      "generated_source": "provider run id or source image path",
      "copied_to": "slides/03.png",
      "status": "completed"
    }
  ]
}
```

If image generation is blocked, write the blocker to the manifest and stop. Do not replace a blocked slide with programmatic art.

### 6. Build `deck.json` And Compose Image-Only PPTX

Build a simple deck manifest:

```json
{
  "slide_width_in": 13.333,
  "slide_height_in": 7.5,
  "units": "fraction",
  "slides": [
    {"background": "slides/01.png"},
    {"background": "slides/02.png"}
  ]
}
```

Compose with the available image PPT composer, for example:

```powershell
python scripts/gorden_image2pptx/compose_pptx.py deck.json out/deck-image.pptx --preview-dir previews
```

The output should have exactly one full-slide picture per slide.

### 7. Optional Editable Reconstruction

Use the older background/frame/icons/text reconstruction path only when the user explicitly requests editable PPT objects. Treat it as experimental for dense Chinese real-world slides. It requires separate extraction prompts, layer manifests, visual comparison, text audits, and manual/multimodal review.

### 8. Build `layered_deck.json` For Optional Editable Runs

Layer stack per slide:

1. `background`: full-slide image with no ordinary text, no frame, no icons.
2. `frame`: full-slide transparent PNG containing structure, chart geometry, panels, separators, fills, arrows, and non-text scaffolding.
3. `icons`: positioned transparent PNG items for icons, decorations, pictorial objects, and stylized text.
4. `texts`: real PPT text boxes for ordinary editable text.

Read [references/reconstructed-editable.md](references/reconstructed-editable.md) for the layout contract. Read [references/editable-reconstruction-route.md](references/editable-reconstruction-route.md) for the visual-target reconstruction route and QA loop.

### 9. Compose Editable PPTX For Optional Editable Runs

Use the reconstruction composer for source-image extraction work:

```powershell
python scripts/gorden_image2pptx/compose_pptx.py layered_deck.json out/deck-reconstructed-editable.pptx --preview-dir previews
```

Use `scripts/compose_layered_deck.py` only when a simple internal fixture or compatibility case already follows this skill's normalized `layered_deck.json` schema:

```powershell
python scripts/compose_layered_deck.py layered_deck.json out/deck-reconstructed-editable.pptx
```

This creates an optional editable PPTX. Text is real PPT text; frame and icon layers are movable images. Do not use this path as the default unless the user asks for editability.

For reconstruction composer layouts, keep `fit_text` enabled unless there is a deliberate typography reason to disable it. Dense Chinese pages should use real CJK-capable font names in the PPTX, and the preview path must resolve a CJK font on the current OS. A preview with black CJK glyph boxes, unreadable overflow, or title/body overlap is a failed visual QA artifact even when the PPTX has editable text boxes.

### 10. Render QA Previews

Use `scripts/render_layered_preview.py` to create visual QA previews from `layered_deck.json`:

```powershell
python scripts/render_layered_preview.py layered_deck.json --out-dir previews --show-boxes
```

The preview renderer is for inspection only. Do not use preview images as PPT source layers or as a replacement for image-model-generated background, frame, or icon assets.

### 11. QA

Run image-only validation before primary delivery:

```powershell
python scripts/audit_image_only_deck.py out/deck-image.pptx --spec slide_spec.json --render-manifest render_manifest.json --deck-json deck.json --out qa/image-only-pptx.json
python scripts/audit_visible_text_review.py qa/visible-text-review.json --min-slides 1 --out qa/visible-text-review-audit.json
python scripts/validate_style_contract_richness.py visual_style_contract.json --slide-spec slide_spec.json --out qa/style-contract-richness.json
```

A `fail` from `validate_style_contract_richness.py` (missing hex colors, missing named fonts, missing `extra_fields`) is a release blocker, not a warning to note and move past.

For optional editable reconstruction, run the editable validation gates:

```powershell
python scripts/validate_visual_deck.py slide_spec.json --layers layered_deck.json --layer-manifest layer_manifest.json
```

Audit the final PPTX structure:

```powershell
python scripts/audit_pptx_editability.py out/deck-reconstructed-editable.pptx --out qa/pptx-editability.json --fail-flattened
python scripts/audit_pptx_text_against_spec.py slide_spec.json out/deck-reconstructed-editable.pptx --out qa/pptx-text-vs-spec.json
python scripts/audit_layered_layout.py layered_deck.json --out qa/layered-layout.json --fail-on-warn
python scripts/audit_visual_quality.py layered_deck.json --out qa/visual-quality.json
python scripts/gorden_image2pptx/icon_coverage_audit.py layered_deck.json qa/icon-coverage-expected.json --out qa/icon-coverage.json
python scripts/gorden_image2pptx/build_frame_residue_contract.py layered_deck.json --icon-coverage-expected qa/icon-coverage-expected.json --out qa/frame-residue-regions.json
python scripts/gorden_image2pptx/frame_residue_audit.py layered_deck.json qa/frame-residue-regions.json --out qa/frame-residue.json
python scripts/audit_visual_acceptance.py qa/visual-review.json --compare qa/visual/report.json --out qa/visual-acceptance.json --fail-on-warn
```

`build_frame_residue_contract.py` emits bbox-only skeleton regions by default. Add explicit `--color-family teal_green`, `--color-family any_saturated`, or another supported family only when the slide spec/extraction plan says those movable decorations must not remain in the frame layer.

When icon coverage or visual acceptance fails on a dense page, generate targeted next-pass prompts instead of repeating a broad extraction prompt:

```powershell
python scripts/gorden_image2pptx/build_extraction_prompt_pack.py qa/icon-coverage.json qa/icon-coverage-expected.json --out-dir prompts/coverage-next-pass --language zh
```

Open the current visual target with `view_image` before using any generated prompt. These prompt files assume the just-opened image is the edit target; local paths inside prompts are not image inputs.

If `audit_layered_layout.py` reports text overlap, icon crowding, cramped text, or edge clipping, repair the normalized layer coordinates before recomposing:

```powershell
python scripts/repair_layered_layout.py layered_deck.json --out layered_deck.repaired.json --report qa/layered-layout-repair.json
python scripts/audit_layered_layout.py layered_deck.repaired.json --out qa/layered-layout-repaired.json --fail-on-warn
python scripts/compose_layered_deck.py layered_deck.repaired.json out/deck-reconstructed-editable.pptx
```

Use the repair script only for coordinate-level cleanup of already extracted layers. It must not replace visual-target generation, image-model layer extraction, or manual visual QA.

For planning-stage specs without layer assets, use `--allow-planning`; do not use that flag for final delivery QA.

QA must check:

- `slide_spec.json` parses and every slide has required fields
- `scripts/validate_content_density_and_roles.py` passes: cover slide has subtitle/presenter content, body_text meets the density floor, every slide has the banana-style 9-field `layout_profile`, at least one `content`-role slide exists in a 4+ slide deck, and no non-cover `page_role` repeats 4+ times in a row
- `visual_style_contract.json` exists, parses, and has a real style source; a deck without a template must still have a concrete style contract built from `scripts/build_style_contract.py`, and `scripts/validate_style_contract_richness.py` passes (explicit hex colors, named fonts, no missing richness fields)
- every primary slide has `extra_fields` (or, for legacy decks, `page_description`), and `image_prompt` is derived from it rather than from raw field labels; `speaker_notes` never appears in `image_prompt` or on the rendered slide
- VLM review (`semantic_review_dashscope.py`) was called with `--style-contract` so `style_contract_compliance`, `formal_ppt_feel`, and `template_overfit` are judged against the deck's actual palette/typography/spacing values, not general aesthetics
- primary delivery slides use `delivery_mode: image_only` or clearly imply image-only delivery through `rendered_image`
- every slide has a generated or source-provided full-slide image
- every slide image has the same aspect ratio as the target PPTX within QA tolerance; a square `1024 x 1024` source image is not acceptable for a `16:9` deck, even if the PPTX structure can stretch it to full slide
- `deck.json` covers every final slide and points to readable slide images
- `render_manifest.json` records image-model provenance for every generated slide
- the final image-only PPTX passes `audit_image_only_deck.py`: one full-slide picture per slide and no required editable text boxes
- previews are manually compared with `slides/` for unreadable text, bad aesthetics, wrong aspect ratio, drift, repeated layouts, or chroma-key artifacts
- `qa/visible-text-review.json` records visible-text QA for every slide, including allowed text, forbidden terms, unsupported numbers, unsupported dates, unsupported names, readability, and notes
- `audit_visible_text_review.py` passes before release; use `--fail-on-warn` for source-grounded decks that require exact wording
- `style_reference` routes include `reference_guard`, and the visual target is checked for reference semantic bleed
- visual acceptance includes aesthetics: slide hierarchy, spacing, density, visual richness, style consistency, chart/diagram clarity, and whether the page reads like a finished PPT slide rather than a prompt dump
- visible-text and VLM review inputs must carry forbidden terms from `visual_style_contract.forbidden_patterns`, `reference_guard.forbidden_terms`, and internal prompt-field guards. Treat generic `LOGO`, page numbers, watermarks, markdown symbols, prompt labels, fake English, and internal fields as forbidden unless they are explicitly required slide text.
- `existing_slide_edit` routes declare `source_visual_target` or `source_visual_targets`
- source-grounded decks carry evidence
- no primary slide uses `programmatic`, `PIL`, `SVG`, `HTML`, `Canvas`, matplotlib, screenshots, or copied local mockups as a substitute for image-model output
- no primary slide is accepted after non-uniform resizing, squeezing, or stretching to fit the slide canvas
- token strings are absent from prompts, manifests, reports, scripts, logs, and final replies

Optional editable reconstruction QA must check:

- `layered_deck.json` covers every spec slide
- every slide has background and frame assets
- icons/decorations are positioned when needed
- dense visual-target reconstruction runs include `qa/icon-coverage-expected.json` and pass `scripts/gorden_image2pptx/icon_coverage_audit.py`, so important source regions are covered by actual movable icon/decor assets instead of only by total icon count
- dense visual-target reconstruction runs include `qa/frame-residue-regions.json` when the slide spec declares areas where movable decorations must not remain in the frame layer; generate a skeleton from icon/decor ownership evidence with `build_frame_residue_contract.py`, add explicit `forbidden_residue` checks only where the slide plan warrants them, and require `scripts/gorden_image2pptx/frame_residue_audit.py` to pass
- ordinary text is real PPT text in `texts`
- the final PPTX passes `audit_pptx_editability.py` and is not flagged as flattened image-only
- the final PPTX passes `audit_pptx_text_against_spec.py`, with no missing or extra editable text against `slide_spec.json`
- reconstruction composer previews show readable CJK text and fitted dense text; do not accept a deck whose preview only passes after ignoring renderer black boxes, text overflow, or doubled text
- the final `layered_deck.json` passes `audit_layered_layout.py --fail-on-warn`, so dense pages do not ship with overlapping text, crowded icons, cramped text boxes, or edge-clipped editable elements
- `audit_visual_quality.py` is reviewed for dense decks, especially repeated card-only layouts, missing chart/table/legend semantics, and overly mechanical text-box patterns
- `audit_visual_acceptance.py` passes against actual human or multimodal visual review notes; do not ship a deck just because structural audits pass
- image layer files exist and are readable
- `layer_manifest.json` records image-model provenance for each required visual target and layer
- no layer uses `programmatic`, `PIL`, `SVG`, `HTML`, `Canvas`, matplotlib, screenshots, or copied final-slide crops as its source
- previews are manually compared with `visual_targets/` for missing layers, doubled text, bad alignment, unreadable text, weak aesthetics, and visual drift
- if the preview renderer cannot display Chinese or other CJK text correctly, inspect PPTX text structure and use another renderer or direct PowerPoint review before marking `preview_readability` and `text_rendering` as pass

For primary release or regression checks, run the image-only package audit against a real run:

```powershell
python scripts/audit_image_only_deck.py out/deck-image.pptx --spec slide_spec.json --render-manifest render_manifest.json --deck-json deck.json --out qa/image-only-pptx.json
python scripts/audit_visible_text_review.py qa/visible-text-review.json --min-slides 1 --out qa/visible-text-review-audit.json
python scripts/validate_style_contract_richness.py visual_style_contract.json --slide-spec slide_spec.json --out qa/style-contract-richness.json
```

For optional editable reconstruction regression checks, run:

```powershell
python scripts/self_test_visual_deck.py
```

For deeper QA and originality guidance, read [references/qa-and-originality.md](references/qa-and-originality.md).

## Output Structure

Use an isolated run directory:

```
visual_deck_runs/<timestamp>_<slug>/
|-- input_summary.md
|-- slide_spec.json
|-- prompts/
|   `-- 01-slide.md
|-- slides/
|   `-- 01.png
|-- deck.json
|-- render_manifest.json
|-- qa_report.json
|-- qa/
|   `-- image-only-pptx.json
|   `-- visible-text-review.json
|   `-- visible-text-review-audit.json
|-- previews/
|   `-- slide_01.png
`-- out/
    `-- deck-image.pptx
```

Optional editable reconstruction runs may additionally include `visual_targets/`, `layers/`, `layered_deck.json`, `layer_manifest.json`, overlay previews, and `out/deck-reconstructed-editable.pptx`.

## When To Read References

- Routing or mixed inputs: [references/input-routing.md](references/input-routing.md)
- JSON schema and editable layer contract: [references/slide-spec-contract.md](references/slide-spec-contract.md)
- Banana-style text-to-image aesthetic prompt contract: [references/aesthetic-generation.md](references/aesthetic-generation.md)
- Optional editable visual-target reconstruction route: [references/editable-reconstruction-route.md](references/editable-reconstruction-route.md)
- Optional reconstructed editable PPTX path: [references/reconstructed-editable.md](references/reconstructed-editable.md)
- Giiisp/SiTian, Codex imagegen, or provider token handling: [references/image-runtime.md](references/image-runtime.md)
- QA gates and avoiding "wrapper" positioning: [references/qa-and-originality.md](references/qa-and-originality.md)
