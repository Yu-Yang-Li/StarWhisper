# Figures and cover art

The repository home uses one concept illustration. Paper figures, experiment
charts, and validation diagrams belong in the research notes, alongside their
source and scientific context.

Use repository-relative Markdown image links: `docs/assets/<file>` from a root
README, and `assets/<file>` from a document under `docs/`. This keeps images on
the same branch as the text and allows local preview. GitHub Pages also serves
these files at `https://yu-yang-li.github.io/StarWhisper/assets/<file>`.

## Current figures

| File | Used in | Source / purpose |
| --- | --- | --- |
| `starwhisper-cover.png` | Chinese and English repository home | AI-generated concept illustration; not a hardware photo or a scientific result |
| `starwhisper-lc.png` | Research notes | LC paper figure |
| `starwhisper-telescope.png` | Research notes | Telescope paper figure |
| `sitian.png` | Research notes | Sitian concept sketch |
| `goai-observe-loop-source.png` | Research notes | Original GOAI observing-loop diagram |
| `goai-metrics-source.png` | Research notes | Original GOAI policy trade-off figure |
| `goai-verification-source.png` | Research notes | Original GOAI validation-path diagram |

## Retained earlier assets

`starwhisper-hero.jpg`, `starwhisper-architecture.jpg`,
`starwhisper-observe-loop.jpg`, `starwhisper-skills-matrix.jpg`, and
`starwhisper-verification.jpg` are earlier generated illustrations. `demo-1.png`
and `demo-2.png` are earlier capability sketches. They remain available for old
links and provenance; the repository home no longer embeds them.

## Cover provenance

Created with the built-in `imagegen` tool, using `starwhisper-hero.jpg` as the edit
target. The illustration carries no observations, measurements, or interface
readouts.

<details>
<summary>Generation prompt</summary>

```text
Use case: illustration-story.
Asset type: wide GitHub README cover for the open-source astronomy research project StarWhisper.
Primary request: redesign the provided cover into a restrained, beautiful astronomy editorial illustration. Treat the input as the edit target: remove every overlay panel, every numeric readout, every schedule, every status card, the artificial connected-star network, every diagram, every title and subtitle. Keep the core idea of a telescope under the night sky, but simplify the composition dramatically.
Scene: a quiet mountain observatory at night beneath a delicate, scientifically plausible Milky Way. One small, physically coherent telescope silhouette in the lower right, a single low unobtrusive observatory dome along the horizon. The sky occupies most of the frame. A subtle warm light near the horizon, no laser beams.
Style: refined astronomy book-cover art, cinematic but calm, natural textures, no glossy sci-fi aesthetic. Midnight ink blue, restrained ivory starlight, a tiny warm amber accent. Rich dark tones without oversaturation.
Composition: panoramic 2.5:1 landscape banner, generous quiet space on the left and center, horizon in the lower fifth, telescope subordinate to sky. Clean edges, elegant visual balance, legible at a GitHub README width of 850 pixels.
Text: no text at all.
Constraints: this is conceptual cover art, not an actual hardware photo. No fabricated scientific results, no interface, no HUD, no cards, no badges, no charts, no axes, no code, no arrows, no pipelines, no labels, no network lattice, no watermark, no logos, no glowing AI brain. Do not merely blur the old panels: remove them completely and rebuild clean continuous sky.
```

</details>
