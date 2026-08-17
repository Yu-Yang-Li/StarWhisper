# Reconstructed Editable PPTX

Use this path for this skill's final PPTX output.

## Position In This Skill

The overall workflow remains spec-first, but publish-grade pages now use a full visual target before extraction:

```
user input -> slide_spec.json -> visual_target image -> extracted layer assets -> editable PPTX -> preview QA
```

The deck is visual-target-first and layer-reconstructed. It must not use the full-slide visual target as a final background with duplicate text boxes on top.

## Layer Contract

Each slide in `layered_deck.json` uses this stack from bottom to top:

1. `background`: full-slide image with no ordinary text, no cards, no frame, no icons.
2. `frame`: full-slide transparent PNG containing structural visuals such as cards, panels, separators, arrows, chart geometry, fills, and decorative lines.
3. `icons`: positioned transparent PNG items for icons, decorations, pictorial objects, and stylized text that should remain visual.
4. `texts`: real PPT text boxes for ordinary editable text.

Minimal shape:

```json
{
  "slide_width_in": 13.333,
  "slide_height_in": 7.5,
  "units": "fraction",
  "assets_dir": ".",
  "slides": [
    {
      "slide_id": "01",
      "background": "layers/01/background.png",
      "frame": "layers/01/frame.png",
      "icons": [
        {"file": "layers/01/icons/chart.png", "x": 0.62, "y": 0.22, "w": 0.20, "h": 0.22}
      ],
      "texts": [
        {"text": "标题", "x": 0.06, "y": 0.08, "w": 0.55, "h": 0.10, "size": 30, "bold": true, "color": "111111"}
      ]
    }
  ]
}
```

## Generation Rules

For generated decks:

- Plan the slide structure in `slide_spec.json` first.
- Generate `visual_targets/NN.png` with the image model as the complete aesthetic benchmark for each slide.
- Generate or extract layer assets from the visual target with the image model, not with local drawing code.
- Keep ordinary editable text out of `background` and `frame`.
- Put chart geometry, cards, connectors, and visual scaffolding into `frame`.
- Put pictorial icons, decorative images, and stylized text into `icons`.
- Put ordinary wording into `texts`.

For existing slide images:

- Treat the source slide image as the `visual_target` and edit target for layer extraction.
- Produce a clean background, structural frame, icon/decor layer, and text layout.
- Record each image-model layer generation in `layer_manifest.json`.

## Provenance Manifest

Every layered editable slide should have entries for generated image layers:

```json
{
  "slides": [
    {
      "slide_id": "01",
      "layers": [
        {
          "layer": "visual_target",
          "backend": "codex-imagegen",
          "prompt_file": "prompts/01-visual-target.md",
          "generated_source": "provider output id or source path",
          "copied_to": "visual_targets/01.png",
          "status": "completed"
        },
        {
          "layer": "background",
          "backend": "codex-imagegen",
          "prompt_file": "prompts/01-background.md",
          "generated_source": "provider output id or source path",
          "copied_to": "layers/01/background.png",
          "status": "completed"
        }
      ]
    }
  ]
}
```

Invalid layer sources include `programmatic`, `PIL`, `SVG`, `HTML`, `Canvas`, `matplotlib`, screenshot renderers, or copied crops from the final full-slide image.

## QA Requirements

Before delivery:

- Run the layer validator or include the layered layout in `validate_visual_deck.py`.
- Compose a PPTX preview and inspect it visually.
- Compare the reconstructed preview against `visual_targets/NN.png`.
- Confirm there is no doubled text.
- Confirm background/frame/icon/text layers are present where the slide complexity requires them.
- Confirm text boxes are real PPT text and visually aligned.
- For source-grounded decks, evidence still lives in `slide_spec.json`; layer reconstruction does not remove source-grounding requirements.

Use clear output names:

- `deck-reconstructed-editable.pptx`: layer-built editable deck.
- `preview/slide-NN.png`: rendered QA preview only, not a source layer.
