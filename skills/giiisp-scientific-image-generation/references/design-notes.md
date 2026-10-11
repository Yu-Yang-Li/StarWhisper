# 绘图设计来源与进度记录

[返回技能说明](../SKILL.md)

按本次任务阅读相关部分。本文件中的命令从技能目录执行，文件名与参数按实际输入替换。早期试运行与历史记录不表示外部服务当前可用。

## 用户可见进度输出

生成或改图时要主动输出有效状态，减少用户等待。进度绑定真实工作流节点，中文请求用中文；每条前缀当前本地时间或已耗时，但正文要保持科研图执行口径：有参数、有判断、有质量标准，不写成后台日志，也不写成陪聊安慰。只增加用户可见进度说明，不改变 Giiisp Imagine 接口、token 读取方式、请求体字段、轮询方式、run 目录规范或检查脚本。

进度文字要专业、短、具体。不要堆内部文件名，也不要把话说得太口语。优先输出用户真正关心的技术信息：图型、画幅、标签约束、参考图角色、生成状态、尺寸/格式、语义复查维度、是否可交付、下一轮修订策略。不必每轮全发，按真实发生的节点输出。

中文进度样式：

```text
[21:24 | 任务识别]
我先确认这张图的类型、画幅和必须保留的科学信息，避免生成时跑偏或自行扩写无关内容。

[21:25 | 作图简报]
图型、画幅、必须标签和禁止项已经定好；如果使用参考图，只参考指定部分，不改写原始科学含义。

[21:26 | 请求提交]
生成任务已经提交，正在等待图片返回；访问码只用于本次调用，不会写进记录里。

[21:31 | 图片生成]
图片已返回，尺寸和格式可读。接下来检查标签、伪英文、水印和版式层次。

[21:32 | 机器检查]
文件本身是有效图片，尺寸和格式没问题；但语义和标签还要复查，不能只看机器检查就交付。

[21:34 | 复查结论]
主体结构基本成立，关键标签也在；但仍发现一处文字或语义问题，这版适合做草稿，正式版建议再改一轮。

[21:35 | 交付索引]
本轮生成、检查和复查结论已经整理好；交付时会说明这张图能用到什么程度，以及下一轮该怎么改。
```

优先在这些节点输出：任务识别、作图约束确认、请求提交、轮询等待较久、图片下载、机器检查完成、复查开始、复查完成、manifest 更新、blocker、最终交付。每条包含一个真实中间信息即可，例如任务号、图片尺寸、必须标签数量、复查发现、是否建议二次修改。文件名可以出现，但不要每条都堆文件名；只在用户需要追溯或出现 blocker 时明确证据文件。遇到 blocker 时说明证据文件；如果接口返回错误 JSON，优先展示接口 `code` 和 `error`，例如 `INVALID_TOKEN / 访问码无效`，再说明衍生状态如缺 token、`ACCESS_TOKEN_REQUIRED`、无 `job_id`、轮询超时或无图片；明确不会伪造图片。

## 参考 PaperBanana 的地方

这个 skill 不照搬 PaperBanana 的模型调用栈，但照搬它的工作流结构：

| PaperBanana 做法 | 本 skill 对应实现 |
|---|---|
| `generate_diagram(source_context, caption, ...)` | `prompt` + 作图简报，生成单张科研图 |
| `continue_run(run_id, feedback, ...)` | `--run-kind edit`、`--source-run`、`--reference-image`，每次修改新建 run |
| `metadata.json` / `run_input.json` | 每轮生成或 blocker 都写 `run_input.json`、`metadata.json` 和 `figure_manifest.json` |
| `batch_manifest.yaml` | 多图交付用 `build_figure_package.py` 汇总每张图的 manifest，不用口头一次性塞多张图 |
| `batch_report.json` / checkpoint | 图包目录写 `package_plan.json`、`package_checkpoint.json`、`figure_package.json` |
| `evaluate_diagram` | 当前用 `check.json` + `semantic_review.json` + `manual_review.md`；有参考图时再补结构化对比 |
| 多候选择优 | 多个 run 完成后用 `select_figure_variant.py` 按完成状态、机器检查和人工质量轴选择候选 |

对齐源项目的关键链路是 Critic 看图后修改描述，再由 Visualizer 基于 revised description 重新文生图；本 skill 不默认接入 PaperBanana 的参考图库、Retriever、多候选并发筛选或三轮以上 Critic 循环。核心原则：每张图都必须有可追溯 run，不只保存最终图片。

Critic 输出字段要对齐 PaperBanana：

- `critic_suggestions`：具体审查意见；如果无需修改，写 `No changes needed.`。
- `revised_description`：合并修正后的完整详细描述；如果无需修改，写 `No changes needed.`。

Visualizer 重生 prompt 要以 revised description 为核心：

```text
Render an image based on the following detailed description: {revised_description}
Note that do not include figure titles in the image. Diagram:
```

本 skill 可以在这个 prompt 后追加中文标签、禁止项、Giiisp 输出要求和“一轮自动修订”边界，但不能跳过 `critic_suggestions` / `revised_description` 这组中间产物。

## Crafter 启发但不照搬的地方

Crafter 的有效启发是“科研图是结构化语义组件组合，不只是更长 prompt”。本 skill 只吸收轻量契约，不引入 OpenRouter、多 agent、SAM3 或 raster-to-SVG 依赖：

- 用 `figure_spec.json` 作为单张图的结构化事实源，避免多轮 prompt 追加后互相矛盾。
- 用 `reference_role` 明确参考图用途，避免“保留结构”“借用元素”“润色草图”“编辑当前图”混在一起。
- 用固定质量轴复核生成结果：`content_accuracy`、`layout_quality`、`text_readability`、`aesthetic_quality`、`artifact_severity`。
- 默认完整流程用 DashScope Qwen 做生成后语义复核，结果写 `semantic_review.json`，不替代本地机器检查和代理复查摘要。
- 继续坚持本 skill 的核心优势：Giiisp Imagine 专用、token 不落盘、每轮 run 可审计。
