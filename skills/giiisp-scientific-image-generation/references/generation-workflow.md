# 图像生成与检查

[返回技能说明](../SKILL.md)

按本次任务阅读相关部分。本文件中的命令从技能目录执行，文件名与参数按实际输入替换。早期试运行与历史记录不表示外部服务当前可用。

## 适用场景

用户要“画一张科研图”“生成论文配图”“把这段方法做成流程图”“按这张图继续改”时使用本 skill。

不要把它用于纯 SVG 手绘、PPT 排版或普通网页插图。这里的主线是图像生成模型。

## 默认交付深度

默认用户交付不是 smoke test。除非用户明确说“只测接口/只 smoke/只 dry-run”，真实生成后必须继续完成复查闭环：

1. 写 `figure_spec.json` 和脱敏 `request.json`。
2. 发起 `generate-async`，轮询 `generate-jobs/{job_id}`，下载图片。
3. 运行机器检查并写 `check.json`。
4. 默认运行 DashScope VLM 语义审查并写 `semantic_review.json`，检查语义符合度、必须标签、错字/伪英文/水印/广告感、布局层级和是否需要二次修改；如果 VLM 被 key、模型或接口阻断，保留 blocked 状态，不伪造审查结论。
5. 如果 VLM 明确建议 `edit` / `regenerate`，或发现必须标签缺失、禁止项出现、内容准确性/文字可读性/严重伪影失败，自动把 Critic 结论收敛成 PaperBanana 风格的 `critic_suggestions` 和 `revised_description`，再用 Visualizer prompt 重新文生图一次。最多自动修订一轮；第二版复查后停止，把后续选择权交给用户。
6. 写 `manual_review.md` 代理复查摘要：基于机器检查和 VLM 结果给出是否可交付、主要问题和下一轮修改 prompt；它不是最终人工签核。
7. 重新运行 `build_figure_manifest.py`，让 `figure_manifest.json` 纳入图片、机器检查、VLM/代理复查、自动修订状态和下一轮修改建议。
8. 最终回复必须说明：图片路径、检查结果、复查结论、是否达到交付标准、下一轮修改 prompt。

标准入口是 `run_scientific_image_workflow.py`，它串联真实生成、机器检查、VLM 语义审查、最多一轮自动修订、代理复查摘要和 manifest 重建。自动修订时会写 `auto_repair_prompt.json`，记录 `critic_suggestions`、`revised_description` 和最终发给生成器的 Visualizer prompt。`generate_scientific_image_smoke.py` 只是底层接口连通性和单次图片生成脚本，不代表默认完整交付已经结束。

## 工作流

1. 先写作图简报：图的用途、核心信息、必须出现的标签、画幅比例、风格。
2. 再整理生成提示词：中文标签优先保留中文，科研术语按用户材料原样保留。
3. 把作图简报落成 `figure_spec.json`：必须标签、布局、风格、参考图角色、允许/禁止修改项。
4. 构造 `generate-async` 请求。
5. 保存请求体、响应、图片、轮询记录和检查记录。
6. 生成图片后做机器检查和 VLM 语义审查；若有基础问题，自动输出 Critic 建议和 revised description，再用 revised description 重新文生图一次。
7. 对最终版写代理复查摘要，并给出是否需要用户继续判断或修改。
8. 根据用户反馈继续改图，不覆盖上一版；这属于人工接管后的下一轮，不进入无限自动迭代。
9. 生成或修改后重建 `figure_manifest.json`，把任务、图片、检查、复查、自动修订状态和人工判断串起来。

## Figure spec 契约

`figure_spec.json` 是每轮 run 的结构化作图简报，字段包括：

- `figure_kind`：`workflow`、`mechanism schematic`、`method diagram`、`comparison figure` 等。
- `caption` / `communicative_intent`：图题和要传达的科学信息。
- `required_labels` / `forbidden_labels`：必须出现和必须避免的标签。
- `layout_brief`：如“横向四步流程、每步一个卡片、箭头单向连接、留白充足”。
- `style_brief`：如“白底、蓝绿色学术配色、扁平风格、无广告感”。
- `reference_role`：空值或 `preserve_structure`、`use_elements`、`refine_sketch`、`edit_image`。
- `preserve_constraints`、`allowed_changes`、`disallowed_changes`：续改时的保留项、允许变化项和禁止变化项。

可从 JSON 直接构造 dry-run 或真实 run：

```powershell
python scripts/dry_run_scientific_image.py --input-json params.json
python scripts/generate_scientific_image_smoke.py --input-json params.json
```

`params.json` 可以直接放上述字段，也可以放在 `figure_spec` 子对象里。命令行参数优先级高于 JSON 默认值。

## 提示词质量改进建议

上一轮真实生成显示：接口能稳定返回 1024 x 1024 JPEG，主流程中文标签可读；但信息过密时容易出现小字拥挤、英文拼写错误和装饰元素偏多。第二轮收窄提示词后，结构更干净，但图内语义细节减少。因此提示词应优先控制信息层级：

- 主标签控制在 3-6 个短词或短句，逐字列出必须出现的标签。
- 不要要求模型生成密集说明文字；长解释放到图注、PPT 文本或后期排版层。
- 如果必须有英文术语，逐字给出并说明“不要改写、不要拼写变体”；不确定时优先中文。
- 明确版式约束，如“横向四步流程、每步一个卡片、箭头单向连接、留白充足”。
- 明确禁止项，如“小号正文、伪论文截图、随机英文、额外步骤、广告风格、水印”。
- 二次修改时只列允许变化的局部，避免重新生成整张图导致结构漂移。

检查报告必须写出“提示词质量改进建议”：指出本轮是标签问题、结构问题、语义缺失、风格偏差还是文字过密，并给出下一轮可直接复用的修改 prompt。

## 接口

| 项 | 内容 |
|---|---|
| 根页 | `http://images.sitianai.com/` |
| 生成 | `POST http://images.sitianai.com/api/generate-async` |
| 任务查询 | `/api/generate-jobs/{job_id}` |
| 鉴权 | 前端读取 `localStorage.giiisp_auth_token`，生成请求使用 `Authorization: Bearer <token>` |

请求体常用字段：

| 字段 | 说明 |
|---|---|
| `prompt` | 完整作图说明 |
| `negativePrompt` | 排除项，如水印、模糊文字、错乱标签 |
| `aspectRatio` | `1:1`、`4:3`、`16:9` 等 |
| `imageSize` | 前端当前使用如 `1K` |
| `numberOfImages` | 默认 1 |
| `responseModalities` | 默认 `["IMAGE","TEXT"]` |
| `outputMimeType` | 默认 `image/png` |
| `referenceImages` | 可选，参考图数组 |
| `imageBase64` / `imageMimeType` | 可选，用于图像编辑或参考图输入 |

## 访问码

真实生图必须测试结果。如果接口返回 `ACCESS_TOKEN_REQUIRED`，记录 blocker，不要伪造图片。

如果用户提供访问码或浏览器会话中已有 token：

- 可以使用 token 调 1 张低风险小样图。
- 不要把 token 写入文件、日志或最终回复。
- 只记录“已使用访问码 token”。
- 命令行测试只从环境变量 `GIIISP_AUTH_TOKEN` 读取 token。
- 没有 token 时提醒用户到 `https://giiisp.com/#/mcp/authenticate` 申请或刷新 Giiisp 认证，再设置 `GIIISP_AUTH_TOKEN`。

## 结果检查

生成后至少检查：

- 图片文件是否存在。
- 图片文件字节数是否大于 0。
- 图片类型是否为 PNG、JPEG 或 WebP。
- 图片尺寸是否可读取。
- 是否因为缺少 token 或接口返回 `ACCESS_TOKEN_REQUIRED` 被 blocker 阻断。
- 是否符合图题和提示词。
- 必须标签是否缺失或错乱。
- 是否有水印、广告感、模糊文字。

检查结果写入同一轮 run 目录。

机器检查字段至少包括：

| 字段 | 说明 |
|---|---|
| `image_exists` | 图片文件是否存在 |
| `file_size_bytes` | 图片字节数，无法读取时为 `null` |
| `image_type` | `png`、`jpeg`、`webp`、`unknown` 或 `missing` |
| `mime_type` | 推断出的 MIME 类型 |
| `width` / `height` | 可读取尺寸；不可读时为 `null` |
| `has_token_blocker` | 是否存在缺 token 或 `ACCESS_TOKEN_REQUIRED` blocker |
| `blocker_reason` | blocker 原因 |
| `manual_review_required` | 固定为 `true`，用于提醒仍需人工检查语义和标签 |

`check.json` 还会包含：

- `machine_check`：本地可判定的存在性、格式、尺寸、宽高比和像素问题。
- `quality_review_axes`：人工或 VLM 后续复核用的五个质量轴。分数默认为 `null`，不要伪造模型判断。
- `semantic_review`：使用 `--semantic-check` 时生成的待填写语义复核占位。

## DashScope 语义审查

生成图片并通过机器检查后，标准流程用 DashScope Qwen 做一次视觉语义审查：

```powershell
$env:DASHSCOPE_API_KEY = "<dashscope_api_key>"
python scripts/semantic_review_dashscope.py --run-dir "scientific_image_skill_runs/session_a/smoke_YYYYMMDD_HHMMSS" --model qwen3.7-plus
python scripts/build_figure_manifest.py --run-dir "scientific_image_skill_runs/session_a/smoke_YYYYMMDD_HHMMSS"
```

审查脚本使用 DashScope OpenAI-compatible `chat/completions` 接口，默认模型是 `qwen3.7-plus`，也可通过 `--model` 改成账号可用的 Qwen 视觉模型。访问码只从环境变量 `DASHSCOPE_API_KEY` 读取，不写入 `semantic_review.json`、manifest、package 或最终报告。没有 key 时提醒用户到 `https://help.aliyun.com/zh/model-studio/get-api-key` 申请 DashScope/百炼 API key。

审查整页 PPT 时，可加 `--style-contract visual_style_contract.json`。它在原有五项质量检查之外，增加样式符合度、正式汇报适用性和模板使用检查；八项均为 PASS 才可判为通过。缺项或 UNCERTAIN 需要人工复查，样式文件缺失或无法解析时写入 blocked 记录。普通科研图审查无需提供这个参数。

`semantic_review.json` 必须保留：

- `provider`、`endpoint`、`model`、`dashscope_status_code`。
- `quality_review_axes`：五个质量轴的 `PASS` / `FAIL` / `UNCERTAIN`、分数和理由。
- `observed_labels`、`missing_required_labels`、`forbidden_labels_seen`。
- `critic_suggestions`、`revised_description`：对齐 PaperBanana Critic 输出，用于一次自动修订。
- `overall_ready_to_ship`、`recommended_next_action`、`next_edit_prompt`。

如果缺少 `DASHSCOPE_API_KEY`、图片不存在、模型不可用或返回无法解析，脚本写 blocked 状态，不伪造语义判断。

人工检查报告建议使用 `templates/check_report_template.md`，字段至少包括：

- run 目录、源 run 目录、图片路径、请求摘要和生成时间。
- 机器检查摘要：存在性、字节数、类型、尺寸、blocker。
- 语义检查：是否符合图题、必须标签是否出现、是否有错字/伪英文/水印/广告感。
- 图像质量：布局、留白、层级、颜色、图标一致性、是否适合论文或汇报。
- 提示词质量改进建议：本轮问题归因、下一轮 prompt、negative prompt、是否需要二次修改参考图。

## Dry-run

无访问码时运行：

```powershell
python scripts/dry_run_scientific_image.py --prompt "画一个四步科研流程图：问题定义、数据整理、模型生成、结果检查"
```

dry-run 只构造请求体，不发起生成。

参考图 dry-run 默认不打印整张图片的 base64，只显示 `<redacted reference image base64>`，并在 `edit_metadata` 里记录图片 MIME、字节数和 SHA-256，避免把大段图片数据写进日志。

## 真实生成 smoke test

有访问码时运行：

```powershell
$env:GIIISP_AUTH_TOKEN = "<token>"
python scripts/generate_scientific_image_smoke.py --prompt "画一个四步科研流程图：问题定义、数据整理、模型生成、结果检查"
```

默认完整流程运行：

```powershell
$env:GIIISP_AUTH_TOKEN = "<token>"
$env:DASHSCOPE_API_KEY = "<dashscope_api_key>"
python scripts/run_scientific_image_workflow.py -- --prompt "画一个四步科研流程图：问题定义、数据整理、模型生成、结果检查" --required-labels 问题定义 数据整理 模型生成 结果检查
```

这个标准入口会依次调用 `generate_scientific_image_smoke.py`、`semantic_review_dashscope.py` 和 `build_figure_manifest.py`，必要时按 Critic 的 `revised_description` 自动重新文生图一次，并补写 `auto_repair_prompt.json`、`manual_review.md` 与 `workflow_status.json`。只有用户明确要求接口 smoke 时，才单独运行 `generate_scientific_image_smoke.py`；只有明确要求关闭自动修订时，才加 `--no-auto-repair`。

续改命令示例：

```powershell
python scripts/generate_scientific_image_smoke.py --run-kind edit --source-run "scientific_image_skill_runs/real_token_prompt_refine_20260602/smoke_20260602_162034" --reference-image "scientific_image_skill_runs/real_token_prompt_refine_20260602/smoke_20260602_162034/generated_image.jpg" --prompt "保留四步结构，只把第三步改为证据核验" --feedback "第三步语义不够准确，需要改成证据核验" --required-labels 提出问题 检索论文 证据核验 输出结论
```

脚本会发起 `POST /api/generate-async`，随后轮询 `GET /api/generate-jobs/{job_id}`，并把本轮证据保存到 run 目录：

- `request.json`：生成请求和脱敏后的请求头。
- `response.json`：`generate-async` 原始响应。
- `poll_history.json`：每次任务查询结果。
- `generated_image.*` 或 `image_url.txt`：下载成功的图片，或接口只返回远程地址时的地址记录。
- `check.json`：图片是否存在、字节数、PNG/JPEG/WebP 类型、尺寸、token/blocker 状态和失败原因。
- `semantic_review.json`：默认完整流程生成，DashScope Qwen 对生成图的语义审查；阻断时写 blocked 状态。
- `auto_repair_prompt.json`：如果触发一次自动修订，记录 Critic 建议、revised description 和实际 Visualizer prompt。
- `manual_review.md`：代理复查摘要，汇总机器检查、VLM 结论、交付判断和下一轮修改 prompt。
- `workflow_status.json`：标准入口的阶段状态，记录生成、VLM、一次自动修订、代理摘要和 manifest 重建是否完成。
- `run_input.json`：PaperBanana 式输入契约，记录 prompt、参考图哈希、source run、画幅、大小和 token 策略。
- `metadata.json`：本轮状态摘要，记录 job id、poll 次数、输出路径、机器检查和 blocker。
- `figure_manifest.json`：把请求、响应、图片、检查、人工复核和 source run 串成一份交付/续改索引。
- `blocker.json`：无 `GIIISP_AUTH_TOKEN`、接口返回 `ACCESS_TOKEN_REQUIRED`、无 `job_id`、轮询超时或没有图片时写入。

不要手写或伪造 `response.json`、`poll_history.json`、图片文件或 `check.json`。如果没有 token 或接口拒绝访问，保留 `blocker.json` 即可。
