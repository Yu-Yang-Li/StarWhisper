# Aesthetic Generation Contract

Use this reference when creating or revising full-slide image prompts. Text-only generation is supported for slide 1 and for backends without image-to-image support, but the primary flow uses image-to-image once slide 1 exists: every later slide attaches slide 1's rendered image as a shared style-anchor reference, which measurably improves cross-slide visual consistency over text-only prompting alone (verified against the live Giiisp `/api/generate-async` endpoint).

## Required Flow

Generate in this order:

1. `slide_spec.json`: content truth, evidence, visible text, and slide purpose.
2. `visual_style_contract.json`: built from `scripts/build_style_contract.py` (preset library), carrying exact hex palette, named fonts, layout density, spacing/accent-usage percentages, icon/chart rules, rendering descriptor, and forbidden patterns.
3. `extra_fields` (`visual_elements`, `visual_focus`, `layout_notes`, `speaker_notes`): per-slide structured design brief, refined once by `scripts/stylist_refine_slide.py` before first generation.
4. `prompts/NN-slide.md`: provider prompt derived from `extra_fields` (image-safe subset only) plus the style contract serialized into an upstream-style `<page_style>` block.
5. full-slide image; from slide 2 onward, generated with slide 1's image attached as a style-anchor reference.
6. VLM review (`semantic_review_dashscope.py --style-contract ...`) covering content/text/aesthetic axes plus style-contract compliance, formal-PPT feel, and template-overfit; then image-only PPTX packaging.

Do not generate the provider prompt directly from raw spec fields or from a one-line style label. The structured `extra_fields` plus the precise style contract are the quality-control layer that prevents both mechanical field dumps and vague, inconsistently-interpreted style descriptions.

## Prompt Shape

Each prompt should contain these blocks:

- role: expert presentation/UI designer
- page description: exact slide text and design intent
- page_style: palette, typography, layout density, visual language, and cover handling, serialized from `visual_style_contract.json`
- design guidelines: 16:9, high-resolution, clear Chinese text, finished PPT page, no placeholders
- forbidden content: internal labels, markdown symbols, watermarks, logos, fake English, reference semantic bleed

Template images are useful but not required. If no template exists, the style contract must be concrete enough to replace it.

## No-Template Defaults

For topic-only or source-grounded decks without a style reference, build the contract from the preset library rather than writing a fresh description:

```powershell
python scripts/build_style_contract.py --preset scientific_tool_report --out visual_style_contract.json
```

Each preset in `scripts/style_presets.py` already specifies: mood/lighting, a five-role hex palette (background/primary/accent/neutral/divider), an accent-usage percentage limit, named CJK/Latin fonts, layout density, grid/spacing percentages, icon style, chart rules, cover treatment, and a rendering descriptor. Pick the preset closest to the audience/topic; extend the library with a new preset at the same precision instead of writing an ad hoc short label.

Never fall back to vague style words alone. "Modern, beautiful, high quality" is not enough, and neither is a bare color word like "深蓝" without a hex value — the same word gets reinterpreted differently by the image model on every independent generation call.

## Revision Behavior

When the user asks for a slide edit, reuse the parent page description and visual style contract. Change only the affected slide prompt unless the user requests a deck-wide style change. Keep one user-authorized repair pass for basic visual/text failures, then return the result for human judgment.
