# Input Routing

Classify the user request into one or more routes before making slides.

## `topic_only`

Use when the user gives only a title, course topic, business theme, lesson goal, or broad prompt.

Behavior:

- Ask the model to create the narrative, sections, and examples.
- Avoid fabricated precise statistics, citations, customer names, or study results.
- Use assumption labels in `slide_spec.json` for invented framing.
- Prefer reusable structures: problem -> insight -> framework -> examples -> action.

## `source_grounded`

Use when the user provides materials such as PDFs, papers, reports, notes, datasets, screenshots, or pasted text.

Behavior:

- Extract the deck from the source.
- Preserve exact terms, names, and numbers.
- Add evidence fields to slide specs.
- If a claim cannot be found in source, either omit it or mark it as inferred.

## `style_reference`

Use when the user provides visual references, old PPTs, templates, brand assets, websites, or screenshots.

Reference roles:

- `style_only`: borrow color, density, typography mood, and composition energy.
- `preserve_structure`: keep the major layout arrangement.
- `use_elements`: reuse specific motifs or materials if allowed.
- `edit_target`: modify the provided image itself.

Never imply a local file path has been passed to an image model unless the runtime actually supports image input or the image has been shown/attached as the edit target.

## `existing_slide_edit`

Use when the input is already slide images, a PPT, or a PDF export of slides.

Behavior:

- Decide whether the goal is redesign, continuation, or editable reconstruction.
- For reconstruction, image inspection and layer extraction can be used.
- Keep reconstruction separate from spec-first generation: do not use reverse OCR as the default for decks you generated from specs.
- Record the actual source page path in `source_visual_target` for each slide, or `source_visual_targets` at deck level. Do not reconstruct from a remembered screenshot or old output directory.
- Run `scripts/preflight_extraction_backend.py` before extraction. If no true image-edit backend can receive the source page as an edit target, write a blocker instead of faking layers with local drawing code.

## Mixed Inputs

When multiple inputs exist, assign precedence:

1. User constraints and requested deliverable.
2. Source-grounded facts.
3. Style reference.
4. Topic-model expansion.

Write a short `input_summary.md` capturing this route decision.
