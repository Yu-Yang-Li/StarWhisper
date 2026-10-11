# PPT 运行记录与项目对照

[返回技能说明](../SKILL.md)

按本次任务阅读相关部分。本文件中的命令从技能目录执行，文件名与参数按实际输入替换。早期试运行与历史记录不表示外部服务当前可用。

## Boundary With Related PPT Skills

Treat this skill as the default entry for actual PPT/deck creation, redesign, packaging, or QA. Route away only when the user's request clearly matches a narrower asset:

The related skills below are optional and are not included in this package. Route to one only if it is installed and matches the task.

- Use `research-ppt-generation` when the task is specifically an academic or research deck and needs paper/PDF-to-narrative planning, research audience structure, figure grounding, or academic template logic.
- Use `ppt-image-generation` when the user is asking for slide-image prompting strategy, image-model page briefs, route selection, or diagnosis of weak PPT image prompts, not for an end-to-end deck package.
- Use `sci-employee-ppt-making` only for the sci-employee wrapper: research training course pages, PPT-to-video,讲稿/字幕/配音闭环, or persona-specific teaching delivery.
- Use `chronicle-ai-ppt-open-source-research-loop` only for the recurring AI PPT open-source ranking/gallery loop.
- Use `chronicle-weekly-ai-deck-script` only to turn an existing image-based AI frontier deck into a clean page-ordered speaker script.

Do not create another PPT skill for a new deck style until this routing fails for at least two real tasks.

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

## When To Read References

- Routing or mixed inputs: [references/input-routing.md](input-routing.md)
- JSON schema and editable layer contract: [references/slide-spec-contract.md](slide-spec-contract.md)
- Banana-style text-to-image aesthetic prompt contract: [references/aesthetic-generation.md](aesthetic-generation.md)
- Optional editable visual-target reconstruction route: [references/editable-reconstruction-route.md](editable-reconstruction-route.md)
- Optional reconstructed editable PPTX path: [references/reconstructed-editable.md](reconstructed-editable.md)
- Giiisp/SiTian, Codex imagegen, or provider token handling: [references/image-runtime.md](image-runtime.md)
- QA gates and avoiding "wrapper" positioning: [references/qa-and-originality.md](qa-and-originality.md)
