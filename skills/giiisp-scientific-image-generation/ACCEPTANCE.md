# 验收说明

## 当前定位

这个 skill 面向基于图像生成/编辑模型的科研配图流程。它参考 PaperBanana 的 run、continue、manifest、package 思路，但底层接口使用集思谱 Imagine。

它负责四件事：

- 把论文段落、图题或实验流程整理成作图简报和生成请求。
- 把作图简报保存成 `figure_spec.json`，并在 manifest/package 中暴露结构化字段。
- 真实生图时保存请求、响应、轮询、图片和检查记录。
- 按上一版继续改图时保留父 run、父图、反馈和修改约束。
- 把单图或多图汇总成可交付的 `figure_package.json`。

## 必须通过

```powershell
python scripts/dry_run_scientific_image.py --prompt "画一个四步科研流程图：问题定义、数据整理、模型生成、结果检查"
python scripts/dry_run_scientific_image.py --input-json path/to/params.json
python scripts/generate_scientific_image_smoke.py --prompt "画一个四步科研流程图：提出问题、检索论文、整理证据、输出结论"
python scripts/semantic_review_dashscope.py --run-dir path/to/completed_run --model qwen3.7-plus
python scripts/build_figure_package.py --input "../../scientific_image_skill_runs/real_token_prompt_refine_20260602" --package-dir "../../scientific_image_skill_runs/packages/acceptance_check" --title "科研图像生成结果包"
python scripts/select_figure_variant.py --input "../../scientific_image_skill_runs/real_token_prompt_refine_20260602" --out "../../scientific_image_skill_runs/real_token_prompt_refine_20260602/variant_selection.json"
```

预期：

- 无 token 时真实生成脚本只写 `blocker.json`，不会伪造图片。
- 每个 run 都保留 `figure_spec.json`、`run_input.json`、`metadata.json`、`figure_manifest.json`。
- `figure_manifest.json` 和 `figure_package.json` 暴露 `layout_brief`、`reference_role` 和 `figure_spec_path`。
- edit run 的 `continue_from` 记录父 run、父图 SHA-256、反馈、参考图角色、允许修改项和禁止修改项。
- 参考图 dry-run 不打印整张 base64。
- `check.json` 包含 `machine_check` 和 Crafter-style `quality_review_axes`，但语义分数默认为空，不能伪造。
- 有 `DASHSCOPE_API_KEY` 和图片时，`semantic_review_dashscope.py` 必须能写 `semantic_review.json`；无 key、缺图或模型不可用时必须写 blocked 状态，不能伪造审查结果。
- `figure_manifest.json` 和 `figure_package.json` 能追踪 `semantic_review.json`、审查模型和 `overall_ready_to_ship`。
- package 脚本生成 `package_plan.json`、`package_checkpoint.json`、`figure_package.json`。
- variant selector 只按 manifest 中的可审计字段选择候选，不从图片外观臆测语义质量。

## 可交付文件

- `SKILL.md`
- `agents/openai.yaml`
- `scripts/dry_run_scientific_image.py`
- `scripts/generate_scientific_image_smoke.py`
- `scripts/check_generated_image.py`
- `scripts/build_figure_manifest.py`
- `scripts/build_figure_package.py`
- `scripts/select_figure_variant.py`
- `scripts/semantic_review_dashscope.py`
- `templates/check_report_template.md`

## 真实生图边界

真实生图只从环境变量 `GIIISP_AUTH_TOKEN` 读取访问码。不要把 token 写进命令历史、请求文件、日志、进度记录或最终回复。若接口拒绝访问，保留 blocker 和 run contract 即可。

DashScope 审查只从环境变量 `DASHSCOPE_API_KEY` 读取访问码。不要把 key 写进命令历史、请求文件、日志、进度记录或最终回复。若模型或接口不可用，保留 blocked `semantic_review.json` 即可。
