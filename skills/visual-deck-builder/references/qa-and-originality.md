# QA And Originality

## QA Gates

Before delivery, check:

- `slide_spec.json` exists and parses.
- `visual_style_contract.json` was built from `scripts/build_style_contract.py` (preset library) and passes `scripts/validate_style_contract_richness.py`: explicit hex colors for every palette role, named CJK/Latin fonts, and non-empty mood/accent-limit/grid/icon/chart/rendering fields. A bare color word or a one-line style label is a fail, not a pass-with-a-note.
- every primary slide has `slide_id`, `title`, `visual_brief`, `extra_fields` (or, for legacy decks, `page_description`), `image_prompt`, and `rendered_image`.
- `image_prompt` is derived from `extra_fields` (`visual_elements`/`visual_focus`/`layout_notes`, excluding `speaker_notes`) plus `visual_style_contract.json`, not from raw internal field labels.
- slide 1 has no reference image; slides 2..N were generated with slide 1's rendered image attached as a style-anchor reference (`reference_role: style_reference`), recorded in the run's technical evidence.
- an independent Stylist pass (`scripts/stylist_refine_slide.py`) ran once per slide before first generation, or its skip/blocker is recorded when `DASHSCOPE_API_KEY` was unavailable.
- every primary slide uses `delivery_mode: image_only` or clearly belongs to the image-only route.
- every slide has a generated or source-provided full-slide image recorded in `render_manifest.json`.
- `deck.json` covers all final slides.
- blocked images are clearly marked and not silently replaced by placeholders.
- output PPTX exists and has the expected number of slides.
- `scripts/audit_image_only_deck.py` passes: one full-slide picture per slide and no required editable text boxes.
- source slide images match the target PPT aspect ratio; do not accept a square source image stretched into a 16:9 slide.
- QA previews exist for every slide.
- visual comparison artifacts exist for each slide: source image, PPTX preview, and notes.
- `style_reference` routes include `reference_guard` and the target is visually checked for semantic bleed from the reference.
- source-grounded decks include evidence for specific claims.

## Visual QA

Generate previews or inspect the PPTX manually. Watch for:

- empty or placeholder pages
- weak full-slide image: sparse, flat, repetitive, or obviously less polished than a dense infographic benchmark
- missing design-facing extra_fields/page description or a prompt that reads like field concatenation
- missing or vague style contract, especially no explicit hex palette, named fonts, hierarchy, density, or visual-language guidance
- a page that visibly does not match its own style contract's colors/fonts/spacing (style_contract_compliance)
- a page that reads more like a standalone poster/illustration than a page from a formal presentation deck (formal_ppt_feel)
- every page using the same card-grid template regardless of content, with no compositional variation for flows vs. comparisons vs. conclusions (template_overfit)
- wrong aspect ratio
- stretched or squeezed page content caused by putting a non-16:9 source image into a 16:9 PPT page
- doubled text
- missing frame/card/chart geometry
- missing or duplicated icons
- unreadable text
- random English or fake labels
- reference semantic bleed: brand names, dates, metrics, topic labels, or example copy from a style reference that are not in `slide_spec.json`
- forbidden generic artifacts such as `LOGO`, page numbers, watermarks, markdown symbols, prompt labels, fake English, or internal field names unless the slide spec explicitly requires them
- visual style drift
- poor finished-slide aesthetics: no focal structure, weak hierarchy, cramped spacing, random decorations, or a generic template look
- repeated layouts when the deck should vary
- final PPTX preview substantially uglier or blurrier than the source slide image

Preview images are QA artifacts only. They must not be used as the source of final PPT pages.

## Image-Only QA

Use clear terms:

- `image-only deck`: PPTX made from one full-slide generated image per slide.
- `slide image`: the image-model output copied to `slides/`.
- `preview`: rendered QA image of the final PPTX, not the source image.

Run this for primary decks:

```powershell
python scripts/audit_image_only_deck.py out/deck-image.pptx --spec slide_spec.json --render-manifest render_manifest.json --deck-json deck.json --out qa/image-only-pptx.json
```

An image-only deck is expected to have zero editable text boxes. Do not run `audit_pptx_editability.py --fail-flattened` as the release gate for this route.

The full-slide image must already have the target page ratio. If the image backend returns `1024 x 1024` for a `16:9` deck, treat it as a failed visual target and regenerate or switch backend. Letterboxing, cropping, or stretching is not a valid release fix unless the user explicitly asks for that compromise.

## Editable QA

Use clear terms:

- `reconstructed editable deck`: layered PPTX made from background, frame, icon/decor, and real text layers.
- `layer asset`: an image-model-generated background, frame, or icon/decor image used by the editable deck.
- `preview`: rendered QA image of the editable PPTX; not the source of the editable deck.

Do not call a full-slide image plus text overlay "editable PPTX".

Run this only for explicit editable reconstruction decks:

```powershell
python scripts/audit_pptx_editability.py out/deck-reconstructed-editable.pptx --out qa/pptx-editability.json --fail-flattened
python scripts/audit_pptx_text_against_spec.py slide_spec.json out/deck-reconstructed-editable.pptx --out qa/pptx-text-vs-spec.json
python scripts/audit_layered_layout.py layered_deck.json --out qa/layered-layout.json --fail-on-warn
python scripts/audit_visual_quality.py layered_deck.json --out qa/visual-quality.json
python scripts/audit_visual_acceptance.py qa/visual-review.json --compare qa/visual/report.json --out qa/visual-acceptance.json --fail-on-warn
```

If layout QA warns or fails because of coordinate-level problems, repair and re-run the gate:

```powershell
python scripts/repair_layered_layout.py layered_deck.json --out layered_deck.repaired.json --report qa/layered-layout-repair.json
python scripts/audit_layered_layout.py layered_deck.repaired.json --out qa/layered-layout-repaired.json --fail-on-warn
```

Do not treat repair as a substitute for image-model extraction or visual inspection. It only moves existing editable-layer coordinates when the visual target and extracted assets are otherwise usable.

## Visual Acceptance Review

Write `qa/visual-review.json` after opening the slide image, final preview, side-by-side comparison, and PPTX structure audit. Populate it from direct human inspection or a multimodal model review. Required per-slide fields:

```json
{
  "slides": [
    {
      "slide_id": "01",
      "image_quality": "pass",
      "preview_readability": "pass",
      "text_rendering": "pass",
      "semantic_drift": "pass",
      "overall": "pass",
      "notes": ["Preview visually matches the source slide image and Chinese text is readable."]
    }
  ]
}
```

Use `warn` only for minor, documented deltas. Use `fail` when the slide has unreadable text, duplicated text, weak aesthetics, visible misalignment, missing expected icons/charts, chroma-key artifacts, or unsupported semantic drift. If the renderer cannot show Chinese text correctly, the visual acceptance review must say so and must not pass until another renderer or PowerPoint review confirms readability.

When building a VLM or visible-text review request, pass forbidden text from the style contract and reference guard into the review payload. If the reviewer observes a generic `LOGO`, watermark, page number, prompt label, markdown marker, or internal field name that was not in the allowed slide text, mark the slide `warn` or `fail` even if other quality axes pass.

## Originality Guard

Do not make the skill a clone of another public skill.

Reusable engineering ideas now intentionally embedded:

- spec-first planning
- full-slide image first, then image-only PPTX packaging
- manifest evidence
- isolated run directories
- image-model hard gate
- optional layered editable reconstruction
- QA scripts and visual previews

Avoid making the skill a plain wrapper or attribution-obscuring copy:

- another repo's skill names, stage names, directory structure, prompt text, marketing copy, or exact mandatory taxonomy
- reconstruction utilities as this skill's whole public identity
- copied upstream wording that makes the skill read as a renamed package
- removing attribution from vendored files or redistributed packages
- any claim that this skill is a wrapper around another repo

This skill's identity is **spec-first image-model PPT generation**:

- The plan exists before images.
- A full slide image is generated or selected to satisfy the plan.
- The final primary PPTX packages those page images with evidence and QA.
- Optional editable reconstruction is a separate implementation branch, not the default product.
