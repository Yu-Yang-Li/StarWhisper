# 全天相机观测序列重排

根据原始全天相机照片生成云区掩码，再用掩码调整观测序列：

`原始全天相机照片 -> 整图掩码 -> 重排后的观测序列`

入口为 `run_pipeline.py`。输出包括调整后的计划和 NINA TargetSet 文件，实际执行仍需观测系统加载并确认。

[项目首页](../README.md) · [研究文档](../docs/research.md)

## 目录

```text
AllSky-Camera-XL/
  configs/
    default.json
  Pic2mask/
    deeplabv3_best_loss_3_16.pth
    full_image_infer/
      infer_full_image.py
  requirements.txt
  sequence_adjust_tool/
    parameters.json
    observe_config.json
    finals2000A.all
    de440s.bsp
    batch_0116_output/
    scheme_horizon_to_image/
      replan.py
      标注地平坐标系.py
      AltAzModeling.py
      all_sky_astro.py
  skill/
    photo-to-replan/
      SKILL.md
  output/
  run_pipeline.py
  README.md
```

## 默认配置

默认配置文件：

- [default.json](configs/default.json)

里面记录：

- 统一使用的 conda 环境名
- 默认 `position_name`

默认环境名为：

- `allsky-pipeline`

## 环境准备

推荐新建一个统一环境，例如：

```bash
conda create -n allsky-pipeline python=3.10
conda activate allsky-pipeline
```

先安装通用依赖：

```bash
pip install -r requirements.txt
```

再单独安装 `torch` 和 `torchvision`。

如果你使用 CUDA，请按本机 CUDA 版本改用对应安装命令。

## 运行

在这个目录内执行：

```bash
python run_pipeline.py --image /path/to/2025_03_02_20_33_49.jpg
```

如果要指定参数组：

```bash
python run_pipeline.py \
  --image /path/to/2025_03_02_20_33_49.jpg \
  --position-name position_2
```

如果要保存掩码叠加预览：

```bash
python run_pipeline.py \
  --image /path/to/2025_03_02_20_33_49.jpg \
  --position-name position_2 \
  --save-overlay
```

如果要使用新版 `AltAzModeling.py` 生成的标定结果，请先把
`x0y0_new.txt` 和 `parameters_new.txt` 写入
`sequence_adjust_tool/parameters.json` 的某个 `position` 条目，例如当前的
`position_5`，然后按参数组运行：

```bash
python run_pipeline.py \
  --image /path/to/2022_02_05_01_14_47_001991.jpg \
  --position-name position_5
```

其中：

- `x0y0_new.txt` 来自新版 `AltAzModeling.py`
- `parameters_new.txt` 来自新版 `AltAzModeling.py`
- `replan.py` 和 `标注地平坐标系.py` 都只从 `parameters.json` 读取标定参数
- 投影计算使用完整解析逆投影；`r = ze / k1` 近似不适用于此参数组

## 输出结构

每次运行会生成：

```text
AllSky-Camera-XL/output/
  <image_stem>/
    input/
      <original image>
    mask/
      <image_stem>.png
      <image_stem>_overlay.png   # 仅当 --save-overlay
    replan/
      <YYYYMMDD_HHMMSS_mask>/
        scenario.json
        replanned_schedule.json
        deferred_targets.json
        replanned_sequence.ninaTargetSet
        sky_replan_plot.png
    pipeline_report.json
```

其中：

- `mask/<image_stem>.png` 是整图掩码
- `replan/<...>/` 是重排输出目录
- `pipeline_report.json` 汇总了关键输出路径，便于后续 skill 直接读取

## 处理步骤

1. 输入原始全天相机照片
2. 用 `Pic2mask/full_image_infer/infer_full_image.py` 对整图推理，生成掩码
3. 用 `sequence_adjust_tool/scheme_horizon_to_image/replan.py` 调整序列：
   - `RA/Dec -> Az/Alt`
   - `Az/Alt -> 图像坐标`，使用 `E/a0/e/k1..k4` 完整解析逆投影
   - 在掩码上查像素
   - 输出调整后的观测计划和 NINA TargetSet

## 新版标定与地平标注

使用流程：

1. 准备 `targetlist.txt` 和 `x0y0.txt`
2. 在 `sequence_adjust_tool/scheme_horizon_to_image/` 下运行：

```bash
python AltAzModeling.py
```

3. 得到 `x0y0_new.txt` 和 `parameters_new.txt`
4. 将这两个 txt 的数值写入 `sequence_adjust_tool/parameters.json` 对应 position，例如 `position_5`
5. 运行 `run_pipeline.py` 时通过 `--position-name` 选择参数组，或单独运行地平标注：

```bash
python sequence_adjust_tool/scheme_horizon_to_image/标注地平坐标系.py \
  --image /path/to/image_or_mask.png \
  --parameters sequence_adjust_tool/parameters.json \
  --position-name position_5 \
  --output output/debug_horizon_overlay.png
```

## 通过技能使用

在本机准备好依赖和标定参数后，可使用 [photo-to-replan](skill/photo-to-replan/SKILL.md)。这是本项目附带的入口说明，未列入根目录 `skills/` 的 17 个安装技能。将其 `SKILL.md` 复制到：

```text
~/.codex/skills/photo-to-replan/SKILL.md
```

该技能仍需访问完整的 `AllSky-Camera-XL/` 项目目录；复制说明文件不会安装模型、依赖或标定数据。使用时应明确仓库位置，并从本目录运行入口脚本。
