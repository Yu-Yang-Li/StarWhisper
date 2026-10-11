---
name: photo-to-replan
description: 根据全天相机原始照片生成云区掩码和调整后的观测序列。用于 AllSky-Camera-XL 的 run_pipeline.py、标定参数选择和结果检查；生成文件，不执行望远镜观测。
---

# 全天相机观测序列重排

读取全天相机照片，生成掩码、调整后的观测计划和 NINA TargetSet。需要完整的 `AllSky-Camera-XL/` 项目、模型、依赖与标定参数；单独安装此说明文件不包含运行环境。

## Workflow

1. Locate the StarWhisper repository and its `AllSky-Camera-XL/` directory. Confirm the Python environment, model weights, and calibration group before running.

2. Run the single-entry pipeline script from that directory:

```bash
python run_pipeline.py --image /abs/path/to/image.jpg
```

Optional:

- Override parameter group with `--position-name position_X`
- Save segmentation overlay with `--save-overlay`

3. Read the generated report:

- `output/<image_stem>/pipeline_report.json`

4. From the report, inspect these outputs when needed:

- `scenario.json`
- `replanned_schedule.json`
- `deferred_targets.json`
- `replanned_sequence.ninaTargetSet`
- `sky_replan_plot.png`

## Input Requirements

- The input must be a raw all-sky camera photo.
- The filename should contain Beijing time, preferably in `YYYY_MM_DD_HH_MM_SS.jpg` form.

## Notes

- The mask stage uses the packaged `Pic2mask/full_image_infer/infer_full_image.py`.
- The replan stage uses the packaged `sequence_adjust_tool/scheme_horizon_to_image/replan.py`.
- The pipeline configuration is stored in `configs/default.json`.
- The packaged configuration uses a single conda environment for the whole pipeline.

## Validation

Check the process exit status and report, then confirm that these files exist and are readable:

- `pipeline_report.json`
- `scenario.json`
- `replanned_schedule.json`
- `replanned_sequence.ninaTargetSet`

Check `deferred_targets.json` and `replanned_schedule.json` to see which targets were scheduled or deferred. These files describe a proposed sequence; generating them does not mean the telescope loaded or executed it.
