# Vendored Reconstruction Utilities

This file is retained only as an attribution and maintenance note.

The user-facing route for this skill is documented in [editable-reconstruction-route.md](editable-reconstruction-route.md). Use that file when explaining, packaging, or executing the workflow.

Low-level helper scripts in `scripts/gorden_image2pptx/` are vendored from GordenSun/GordenSuperPPTSkills and remain implementation utilities for:

- key-color probing
- chroma-key removal
- icon sheet slicing
- coordinate guard checks
- PPTX composition
- placement and visual comparison QA

Do not present those utilities as this skill's public identity. Do not remove their notice when redistributing the skill. Do not copy upstream wording into prompts, public descriptions, examples, or release notes.
