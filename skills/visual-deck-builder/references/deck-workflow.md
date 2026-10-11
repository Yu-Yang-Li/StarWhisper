# PPT 内容计划、生成与打包

[返回技能说明](../SKILL.md)

按本次任务阅读相关部分。本文件中的命令从技能目录执行，文件名与参数按实际输入替换。早期试运行与历史记录不表示外部服务当前可用。

## Input Routing

Classify input before generating:

- `topic_only`: user gives a topic, goal, audience, or theme but no source material.
- `source_grounded`: user provides papers, PDFs, reports, notes, webpages, data, or extracted text.
- `style_reference`: user provides screenshots, templates, brand images, color references, or old PPTs.
- `existing_slide_edit`: user provides slide images or PPT/PDF pages to redesign, continue, or make editable.
- `mixed`: combine source grounding with style references and user constraints.

Read [references/input-routing.md](input-routing.md) when routing is unclear.

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

Use the schema and examples in [references/slide-spec-contract.md](slide-spec-contract.md).

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

Read [references/aesthetic-generation.md](aesthetic-generation.md) before changing the prompt shape, adding a new image backend, or diagnosing weak PPT aesthetics.

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

Read [references/image-runtime.md](image-runtime.md) before calling Giiisp or another HTTP image API.

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

Read [references/reconstructed-editable.md](reconstructed-editable.md) for the layout contract. Read [references/editable-reconstruction-route.md](editable-reconstruction-route.md) for the visual-target reconstruction route and QA loop.

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
- scientific-content and style reviews are recorded, including the actual review method; all eight axes pass when using `--style-contract`, and an unavailable or incomplete model review is recorded as requiring manual review
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

For deeper QA and originality guidance, read [references/qa-and-originality.md](qa-and-originality.md).

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
