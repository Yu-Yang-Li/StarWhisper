# Slide Spec Contract

`slide_spec.json` is the planning contract for image-model-driven PPT generation. It plans the content and full-slide image first, then records the rendered slide image used for image-only PPTX packaging. Editable reconstruction fields are optional and only used when the user explicitly requests editable PPT objects.

## Minimal Shape

```json
{
  "deck": {
    "title": "Deck title",
    "language": "zh",
    "aspect_ratio": "16:9",
    "audience": "researchers",
    "route": ["topic_only"],
    "style_brief": "clean visual report style",
    "visual_style_contract": "visual_style_contract.json",
    "reference_guard": {
      "style_reference_policy": "layout_density_only",
      "allowed_text_source": "slide_spec",
      "forbidden_terms": ["OpenAI", "GPT-4", "ChatGPT", "2023"]
    }
  },
  "slides": [
    {
      "slide_id": "01",
      "purpose": "Open the presentation with the core claim",
      "title": "Main title",
      "page_role": "cover",
      "content_density": "low",
      "layout_profile": {
        "template_role": "cover",
        "layout_structure": "centered-title-subtitle-presenter",
        "content_capacity": "low",
        "text_regions": [
          {"name": "title", "position": "center", "size": "large"},
          {"name": "subtitle", "position": "center", "size": "medium"},
          {"name": "presenter", "position": "bottom", "size": "small"}
        ],
        "image_regions": [
          {"name": "background", "position": "center", "size": "large"}
        ],
        "visual_density": "low",
        "style_keywords": ["academic", "clean"],
        "color_palette": ["#FFFFFF"],
        "notes": "Cover stays simple; no chart, flow, or mechanism diagram."
      },
      "body_text": ["Point one", "Point two"],
      "must_show": ["keyword", "metric or source term"],
      "visual_brief": "full-slide image with strong hierarchy",
      "page_description": "Design-facing page description with exact page text, layout intent, visual hierarchy, and material needs.",
      "extra_fields": {
        "visual_elements": "Concrete visual elements this page needs: icon set, chart type, diagram shape, material images.",
        "visual_focus": "The single strongest focal point of the page and why it should draw the eye first.",
        "layout_notes": "Concrete layout plan: grid, column count, card sizes, spacing, alignment.",
        "speaker_notes": "Presenter talking points for this slide. Never rendered into the image."
      },
      "delivery_mode": "image_only",
      "image_prompt": "Generate the full finished slide image with all intended text and complete composition...",
      "rendered_image": "slides/01.png",
      "editable_text": [
        {
          "text": "Main title",
          "role": "title",
          "box": {"x": 0.08, "y": 0.10, "w": 0.72, "h": 0.12},
          "style": {"font_size": 30, "bold": true, "color": "111111"}
        }
      ],
      "evidence": [],
      "status": "planned"
    }
  ]
}
```

## Required Fields

Deck:

- `title`
- `language`
- `aspect_ratio`
- `route`
- `style_brief`
- `visual_style_contract`: path or embedded object

Slide:

- `slide_id`
- `purpose`
- `title`
- `part`: optional section/part label carried through when `generate_slide_spec.py` receives or creates a banana-slides part-based outline; not visible slide text by itself
- `body_text`
- `must_show`
- `visual_brief`
- `page_description`
- `page_role`: `cover | content | data | comparison | timeline | summary | section_divider`; mirrors banana-slides' `template_role` (see `scripts/validate_content_density_and_roles.py`)
- `layout_profile`: banana-slides 9-field template-analysis schema equivalent: `template_role`, `layout_structure`, `content_capacity`, `text_regions`, `image_regions`, `visual_density`, `style_keywords`, `color_palette`, `notes`
- `content_density`: `low | medium | high`; usually copied from `layout_profile.visual_density` for template-fit logic
- `extra_fields`: structured design fields (see below); required for new decks, optional for decks written before this field existed
- `delivery_mode`: usually `image_only`
- `image_prompt`
- `rendered_image` after generation

Optional editable fields:

- `editable_strategy`: use `layered_editable` only for explicit editable reconstruction requests
- `layer_prompts`
- `editable_text`

After image generation:

- `status`: `image_rendered`, `composed`, `qa_passed`, `needs_retry`, or `blocked`

## Extra Fields (Structured Design Brief)

`extra_fields` splits the design-facing brief into four independently useful fields instead of one freeform paragraph:

- `visual_elements`: concrete visual elements this page needs (icon set, chart type, diagram shape, material images).
- `visual_focus`: the single strongest focal point of the page.
- `layout_notes`: concrete layout plan (grid, column count, card sizes, spacing, alignment).
- `speaker_notes`: presenter talking points. **Never** rendered into the image; `build_slide_image_prompt.py` deliberately excludes it from the image prompt.

`build_slide_image_prompt.py` uses `visual_elements` + `visual_focus` + `layout_notes` (in that order) as the primary design brief when `extra_fields` is present, falling back to the freeform `page_description` string when it is not (older decks, or decks written before this field existed). Write `extra_fields` for every new deck; treat freeform-only `page_description` as a legacy compatibility path, not the default.

`layout_notes` must name text regions and image regions separately, not just describe a single diagram. Compare:

- Too diagram-centric (avoid as the deck-wide default): `"横向五步流程图，五个圆角卡片节点"`.
- Text-region-aware (prefer, especially for `page_role: content`): `"标题区在顶部；正文文字区占左侧60%宽度，中等容量，两段说明文字；配图区占右侧40%宽度，小尺寸点缀图标"`.

## Page Role

`page_role` is one of `cover`, `content`, `data`, `comparison`, `timeline`, `summary`, `section_divider` — mirroring banana-slides' `template_role` field from its template-analysis schema (`get_template_analysis_prompt`, 9-field JSON: `template_role`, `layout_structure`, `content_capacity`, `text_regions`, `image_regions`, `visual_density`, `style_keywords`, `color_palette`, `notes`).

`generate_slide_spec.py` accepts both banana-slides outline shapes: a flat list of pages, or a part-based list such as `[{ "part": "Part 1", "pages": [...] }]`. The final `slide_spec.json` stays flat for packaging, but the part label is copied to each slide's optional `part` field so the section structure is not discarded.

`page_role` alone is not enough for source-level alignment. Every new slide must also carry `layout_profile`, using the exact 9-field schema above:

- `template_role`: mirrors `page_role` (`section_divider` for section divider pages).
- `layout_structure`: kebab-case layout summary, such as `title-top-two-column`, `centered-title-subtitle-presenter`, or `horizontal-timeline-five-nodes`.
- `content_capacity`: `low | medium | high`; how much text/content the layout can carry.
- `text_regions`: array of `{name, position, size}` objects; positions are `top | center | bottom | left | right`, sizes are `small | medium | large`.
- `image_regions`: same region shape as `text_regions`; can be `[]` when the page is text-only.
- `visual_density`: `low | medium | high`; used with `content_capacity` for template-fit rhythm.
- `style_keywords`: up to five English adjectives.
- `color_palette`: up to five hex colors when known; can be empty if the deck-level style contract controls colors.
- `notes`: one or two sentences of layout constraints.

`build_slide_image_prompt.py` exposes `layout_profile` and a highest-priority `layout_requirement` block to the image model. `validate_content_density_and_roles.py` fails a new spec that lacks this schema, so a deck cannot silently regress to only `page_role + layout_notes`.

Rules, enforced by `scripts/validate_content_density_and_roles.py`:

- Slide 1 must be `page_role: cover`, and its `body_text` must contain subtitle and presenter/team/date content, not just a bare title (banana-slides requires "title, subtitle, and presenter information" on the first page).
- A deck with 4+ slides must include at least one `page_role: content` slide: a text-forward page (bullet list, paragraph, two-column text), not a diagram. Do not let every slide default to `data`/`comparison`/`timeline`.
- No `page_role` other than `cover` may repeat 4+ times in a row (banana-slides explicitly avoids assigning 5 consecutive identical templates).

## Content Density

Body-text density mirrors banana-slides' `DETAIL_LEVEL_SPECS` modes in `scripts/generate_slide_spec.py`: `concise`, `default`, and `detailed`. The default is `default`: each slide should carry roughly 2-6 short phrases or sentences, each phrase around 15-20 characters, written for presentation delivery (not full paragraphs a speaker would just read aloud, and not 1-2 isolated 2-4 character keyword tags). Use `scripts/generate_slide_spec.py --detail-level default` to get this automatically; when hand-writing a spec, apply the density floor explicitly and expect `scripts/validate_content_density_and_roles.py` to fail if a default slide is reduced to bare keyword tags.

When source files are supplied to `generate_slide_spec.py --reference-file`, their contents are wrapped as banana-style `<uploaded_files>` XML for outline and per-page description prompts. `generation_meta.reference_files` records only filename and character count, not raw source text.

## Visual Style Contract

Build `visual_style_contract.json` with `scripts/build_style_contract.py` instead of hand-writing a short style label. A one-line style description ("clean, modern, blue and white") is not concrete enough to keep independent per-slide image-generation calls visually consistent; the same words get reinterpreted differently call to call. The preset library in `scripts/style_presets.py` carries exact hex colors, named fonts, and spacing/accent-usage percentages:

```powershell
python scripts/build_style_contract.py --list
python scripts/build_style_contract.py --preset scientific_tool_report --out visual_style_contract.json
```

Resulting shape (abbreviated; run the command above to see the full contract):

```json
{
  "style_source": "preset_library",
  "preset_key": "scientific_tool_report",
  "style_name": "科研工具产品汇报",
  "mood_and_lighting": "...",
  "palette": {
    "background": {"name": "暖白纸感背景", "hex": "#F7F4EC"},
    "primary": {"name": "深蓝", "hex": "#13315C"},
    "accent": {"name": "青绿色", "hex": "#2FA39A"},
    "neutral": {"name": "炭黑文字", "hex": "#1F2430"},
    "divider": {"name": "浅灰分割线", "hex": "#D9D4C7"}
  },
  "accent_usage_limit": "强调色使用面积不超过单页总面积的 10%，只用于图标高亮、连接线、关键数字和状态标记。",
  "typography": {"cjk_font": "思源黑体 / Noto Sans SC", "latin_font": "Inter", "title_weight": "...", "body_weight": "..."},
  "layout_density": "...",
  "grid_and_spacing": "...",
  "visual_language": "...",
  "icon_style": "...",
  "chart_rules": "...",
  "cover_treatment": "...",
  "forbidden_patterns": ["watermark", "logo", "fake English", "prompt labels", "markdown symbols"],
  "rendering_descriptor": "...",
  "resolution": "默认 1K；仅在用户明确要求高清重生成时使用 2K"
}
```

Pick the preset closest to the audience and topic (`scientific_tool_report` is the default for research-tool product decks). If none of the four presets fit, extend `scripts/style_presets.py` with a new preset at the same precision level rather than falling back to a short inline description. When a real template image or extracted style exists, keep `style_source` as `template_image` / `extracted_style` and still fill in explicit hex/font values so the contract stays concrete even if the reference image becomes unavailable later.

After slide 1 renders, add `reference_image` (absolute path to slide 1's rendered image) and a `reference_policy` string to the same contract file; see `references/aesthetic-generation.md` for the image-to-image style-anchor flow this enables.

## Page Description And Image Prompt

`page_description` is the design-facing handoff. It should read like a concise page brief, not a JSON dump:

- exact visible text that should appear on the slide
- layout plan and focal structure
- visual hierarchy and spacing expectations
- chart, table, icon, or mechanism-diagram intent
- material images or source figures if available
- special cover-page or closing-page treatment

`image_prompt` must be derived from `page_description` and the visual style contract. Do not expose internal field names as visible slide text.

`image_prompt` must describe the complete final slide image:

- full finished slide image with strong composition
- all intended page text, no placeholders
- enough visual richness to serve as the delivery image
- exact terms that must appear, and dense text that should be simplified rather than hallucinated
- sharp readable Chinese text, clear hierarchy, and a finished PPT-page aesthetic
- explicit forbidden text such as `Purpose`, `Visual brief`, `must_show`, `slide_spec`, and markdown markers

The final primary PPTX uses `rendered_image` as a full-slide picture.

## Optional Layer Prompts

For explicit editable reconstruction requests, `layer_prompts` may contain:

- `visual_target`: full finished slide image with strong composition, all intended page text, no placeholders, and enough visual richness to serve as the aesthetic benchmark.
- `background`: text-free, frame-free background image extracted from or regenerated against the visual target.
- `frame`: structural visuals only, including panels, cards, fills, separators, arrows, chart geometry, and non-text scaffolding extracted from the visual target.
- `icons`: icons, decorative marks, pictorial objects, and stylized text that should stay as images, extracted from the visual target.

Do not use `visual_target` as a final editable PPT layer. It is a design target for optional extraction and visual QA.

## Editable Text

Editable text is optional metadata for overlays or explicit reconstruction. It is ordinary PPT text, not OCR guessed after the fact.

`box` uses fractions of the slide canvas:

- `x`, `y`: top-left
- `w`, `h`: width and height
- all values in `[0, 1]`

When editable reconstruction is requested, use editable text boxes for:

- titles
- section labels
- body bullets
- key metrics
- citations
- short callouts

Use the icon/decoration layer for stylized text that ordinary PPT fonts cannot reproduce cleanly.

## Evidence

For source-grounded decks, evidence items should be compact:

```json
{
  "source_id": "paper-1",
  "locator": "section 3.2 or page 5",
  "claim": "Exact or paraphrased claim used on the slide",
  "confidence": "direct|inferred"
}
```

## Reference Guard

Use `reference_guard` whenever `deck.route` includes `style_reference` or when borrowing layout patterns from public examples.

`reference_guard` prevents semantic bleed from the reference image into the generated visual target:

- `style_reference_policy`: what may be copied from the reference, such as `layout_density_only`, `palette_only`, `preserve_structure`, or `component_style_only`.
- `allowed_text_source`: usually `slide_spec`; the visual target should use only text planned in the spec.
- `forbidden_terms`: reference brands, dates, metrics, labels, and topic-specific words that must not appear in planned content or the generated target.

If the generated slide visually contains forbidden reference terms, reject the image and regenerate before packaging.

## Existing Slide Edit

When `deck.route` includes `existing_slide_edit`, each slide must include `source_visual_target`, or the deck must include `source_visual_targets`.

Use this field to identify the exact source slide image or rendered page being packaged, redesigned, or optionally reconstructed. Missing source targets are a validation error because they make it impossible to audit which page was used.
