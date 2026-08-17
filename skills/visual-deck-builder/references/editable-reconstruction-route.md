# Editable Reconstruction Route

Use this route whenever a slide must be both visually strong and editable in PowerPoint.

## Design Intent

The visual target is the aesthetic contract. The editable deck is a reconstruction of that target using separately auditable layers:

1. `visual_target`: complete slide image used only as design reference and extraction target.
2. `background`: clean full-slide image with no ordinary text, panels, charts, or icons.
3. `frame`: full-slide transparent structure layer containing panels, fills, chart geometry, separators, arrows, ribbons, and non-text scaffolding.
4. `icons`: movable transparent image elements for icons, decorations, pictorial marks, and stylized text that should not become ordinary PPT text.
5. `texts`: real PowerPoint text boxes whose content comes from `slide_spec.json`.

The final PPTX must never be a full-slide screenshot with text placed over it.

## Reconstruction Loop

For each slide:

1. Generate or select the `visual_target`.
2. Inspect the target before extraction. Reject it if it is weak, contains forbidden reference terms, invents unsupported facts, or has unreadable text.
3. Generate the background, frame, and icon/decor assets from the target with an image editing backend.
4. Convert keyed frame and icon sheets into transparent PNG assets.
5. Build `layered_deck.json` from source-image pixel boxes or normalized fractions.
6. Compose the PPTX and QA previews.
7. Compare the reconstructed preview against the target.
8. Write `qa/visual-review.json` from human or multimodal inspection.
9. Run `audit_visual_acceptance.py`; iterate until the deck passes both structure QA and visual acceptance QA.

## Dense Page Rules

For complex business slides, dashboards, charts, tables, and process pages:

- Put chart/table geometry and panel fills in `frame`, not in PPT native shapes.
- Keep ordinary words out of `background` and `frame`.
- Generate more than one icon sheet when a single sheet cannot cover every source icon with enough padding.
- Review the icon contact sheet. Any missing, split, merged, or edge-clipped icon blocks delivery.
- For dense pages, write `qa/icon-coverage-expected.json` with source-image regions such as top nav, left KPI rail, central process icons, radar labels, bottom status strip, and micro-symbol bands. Run `scripts/gorden_image2pptx/icon_coverage_audit.py` before visual acceptance.
- Run placement QA before visual comparison; coordinate math can pass while the box still marks the wrong object.
- Keep text fitting enabled for dense editable text boxes by default. If fitting must be disabled, the visual review must explain why the text still fits in PowerPoint and in the QA preview.
- Treat the preview renderer as a diagnostic artifact. If the renderer cannot display CJK fonts correctly, inspect PPTX text structure and use PowerPoint or another renderer for final visual QA.
- On Windows, CJK preview QA must use an installed Chinese font path such as Microsoft YaHei, SimHei, SimSun, or KaiTi; black glyph boxes or overflow are release blockers, not renderer noise.
- Do not accept a dense page as release-ready when the only passing evidence is editability, text-vs-spec, or layout-coordinate QA.

## What This Skill Adds

This route is not just image-to-PPT conversion. This skill adds:

- input routing for topic-only, source-grounded, style-reference, existing-slide-edit, and mixed requests
- `slide_spec.json` as the text and evidence contract
- reference semantic bleed controls
- source-grounded claim discipline
- forbidden backend and layer provenance checks
- editable text vs spec audit
- dense layout, visual quality, and originality gates

## Implementation Notes

Some low-level utilities under `scripts/gorden_image2pptx/` are vendored implementation tools and keep their own notice. Do not expose those names as the user-facing workflow. Use the route names and artifact names in this file when explaining or packaging the skill.
