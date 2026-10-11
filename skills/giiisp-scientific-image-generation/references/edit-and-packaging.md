# 改图、图包与候选选择

[返回技能说明](../SKILL.md)

按本次任务阅读相关部分。本文件中的命令从技能目录执行，文件名与参数按实际输入替换。早期试运行与历史记录不表示外部服务当前可用。

## 二次修改入口

用户说“按这张图继续改”“保留主体，只改标签/配色/布局/局部元素”时，走二次修改入口。默认先把用户反馈、上一轮 `semantic_review.json` 和原始 `figure_spec.json` 合成新版 prompt/spec，重新文生图；只有在用户明确要求按参考图结构修改，且接口实际可用时，才传 `--reference-image`。

1. 先确认上一轮图片路径或用户提供的参考图路径。
2. 先声明 `reference_role`：`preserve_structure` 保留结构，`use_elements` 借用元素，`refine_sketch` 把草图润色成成品，`edit_image` 编辑当前图；如果不走参考图输入，则记录为文本重生。
3. 把修改要求写进新的 `prompt`，明确“保留上一版的主体结构/构图/画幅”，再列出只允许变动的部分。
4. 若确认走参考图输入，再通过 `imageBase64` 和 `imageMimeType` 传入参考图；否则只传新版 prompt/spec 和 `source_run`。不要覆盖上一轮 run 目录。
5. 新建一轮 run 目录，保存新的 `figure_spec.json`、`request.json`、`response.json`、`poll_history.json`、图片和 `check.json`。
6. 如果没有 token、没有参考图或接口拒绝访问，只写 `blocker.json`，不要伪造图片或检查结果。

二次修改 run 目录规范：

- 所有运行目录放在 `scientific_image_skill_runs/<session_slug>/<run_slug>/`。
- 首次真实生成可用 `real_token_YYYYMMDD/smoke_YYYYMMDD_HHMMSS/`。
- 二次修改必须用新的 `edit_YYYYMMDD_HHMMSS/` 目录，放在当前 session slug 下，或放在 `edit_YYYYMMDD/` session 下。
- 二次修改目录必须写 `source_run.txt`，内容是上一轮 run 目录的绝对路径或相对稳定目录路径。
- 二次修改目录必须保留自己的 `request.json`、`response.json`、`poll_history.json`、`generated_image.*`、`check.json` 和人工检查报告，不复写上一轮文件。
- 如果只做 dry-run，保存或输出的请求仍要能看出 `reference_image_path`、`imageMimeType` 和 `source_run`，但不要写入 token。

二次修改必须保留这些 lineage 字段：

- 父 run：`source_run_id`、`source_run_dir`、`parent_run_input_path`、`parent_metadata_path`、`parent_manifest_path`。
- 父图：`source_image_path`、`source_image_sha256`。
- 用户反馈：`feedback`、`preserve_constraints`、`allowed_changes`、`disallowed_changes`、`new_prompt_delta`。
- 语义继承：`caption`、`communicative_intent`、`figure_kind`、`required_labels`。

dry-run 二次修改示例：

```powershell
python scripts/dry_run_scientific_image.py --prompt "保留上一版四步流程，只把第三步改成模型推断，并统一蓝绿色学术风格" --reference-image "path/to/generated_image.png" --reference-role preserve_structure --allowed-changes 替换第三步标签 统一配色 --disallowed-changes 新增步骤 改变画幅
```

## Figure manifest

生成或二次修改后运行：

```powershell
python scripts/build_figure_manifest.py --run-dir "scientific_image_skill_runs/real_token_prompt_refine_20260602/smoke_20260602_162034"
```

`figure_manifest.json` 至少记录：

- `run_id` 和 `run_dir`
- `figure_id`、`source_run`、`feedback`、`lineage`
- `caption`、`intent`、`figure_kind`、`required_labels`、`style_brief`
- `prompt`、`negativePrompt`、画幅、图片大小
- `request.json`、`response.json`、`poll_history.json`、图片、`check.json`、`manual_review.md`、`blocker.json`
- 输出摘要：图片路径、SHA-256、MIME、宽高
- 质量摘要：机器检查、人工检查摘要、是否需要重生成
- 机器检查摘要
- 人工检查摘要

它的作用类似 PaperBanana 的 run metadata/package report：让后续 continue、批量、复核和交付时知道这张图从哪里来、怎么生成、是否合格。

## Figure package

多张图或一个论文图包完成后，汇总所有 `figure_manifest.json`：

```powershell
python scripts/build_figure_package.py --input "scientific_image_skill_runs/real_token_prompt_refine_20260602" --package-dir "scientific_image_skill_runs/packages/paper_demo_20260602" --title "论文图像生成结果包"
```

图包目录会写：

- `package_plan.json`：锁定本次要交付的图、图题、类型、意图和 manifest 路径。
- `package_checkpoint.json`：记录每个 item 的状态、run id、manifest 和错误，便于后续恢复。
- `figure_package.json`：给用户查看的交付清单，统计完成/阻塞数量和每张图的图片路径、caption、错误。

这对应 PaperBanana 的 `orchestration_plan.json`、`orchestration_checkpoint.json` 和 `figure_package.json` 思路。

## Variant selection

如果同一张图生成了多个候选 run，先为每个 run 生成 `figure_manifest.json`，再运行：

```powershell
python scripts/select_figure_variant.py --input "scientific_image_skill_runs/session_a" --out "scientific_image_skill_runs/session_a/variant_selection.json"
```

选择逻辑只使用可审计字段：完成状态、是否有图片、`machine_check` 是否通过、DashScope `semantic_review_summary`、人工填写的 `quality_review_axes` 分数和 blocker。没有人工或模型语义分数时不会伪造语义判断。
