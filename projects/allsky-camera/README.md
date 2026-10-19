# 全天相机观测序列重排

从全天相机照片识别云区，再据此调整观测序列。入口是 `run_pipeline.py`，输出包括云区掩码、调整后的计划和 NINA TargetSet 文件。

[项目目录](../README.md) · [研究文档](../../docs/research.md) · [技能用法](skill/photo-to-replan/SKILL.md)

仓库已附带分割模型、样例照片、样例观测计划和标定文件。安装 Conda 与推理依赖后，可以用这些材料试跑；换用自己的相机时，需要核对标定和观测计划。生成的 TargetSet 仍需由观测系统加载并确认，脚本本身不执行观测。

## 先用附带照片试跑

以下命令从本目录（`projects/allsky-camera/`）运行。先创建 Python 环境：

```bash
conda create -n cv python=3.10
conda activate cv
pip install -r requirements.txt
```

再按本机 CPU / CUDA 环境安装匹配的 `torch` 和 `torchvision`，安装方法见 [PyTorch](https://pytorch.org/get-started/locally/)。项目的 [default.json](configs/default.json) 当前指定环境 `cv`、标定组 `position_5`；入口会通过 `conda run -n cv` 调用后续程序。若使用其他环境名，需要同时修改配置中的 `conda_env`。

准备好后运行：

```bash
python run_pipeline.py --image test_seq_img/raw_img/2025_02_16_00_02_31.jpg --save-overlay
```

`--device cpu` 可指定 CPU，`--device cuda` 可指定 GPU，默认 `auto` 自动选择。虽然分割权重已在仓库中，当前 torchvision 模型构建仍可能在首次运行时下载骨干网络权重，需要网络或已有缓存。

## 已经提供的材料

| 材料 | 位置与用途 |
| --- | --- |
| 分割权重 | `Pic2mask/deeplabv3_best_loss_3_16.pth`，供整图云区推理使用 |
| 样例照片 | `test_seq_img/raw_img/`，可用于单张或批量处理 |
| 相机标定 | [parameters.json](sequence_adjust_tool/parameters.json)，按 `position` 分组 |
| 观测计划 | `sequence_adjust_tool/batch_0116_output/` 中的计划和 TargetSet 样例 |
| 观测设置与星历 | `sequence_adjust_tool/observe_config.json`、`de440s.bsp`、`finals2000A.all` |

这些材料属于已有相机和观测设置。用新照片生成计划前，应检查照片时间、台站、相机几何和计划设置是否相符。

## 换用自己的照片

输入应为原始全天相机照片，文件名包含北京时间，建议使用 `YYYY_MM_DD_HH_MM_SS.jpg`。将路径替换为真实照片，并选择与相机对应的标定组：

```bash
python run_pipeline.py --image /path/to/image.jpg --position-name position_5 --save-overlay
```

默认输出到 `output/<照片文件名>/`，可用 `--output-root /path/to/output` 指定其他位置。完成后先看 `pipeline_report.json`，其中列出了各项结果路径：

| 输出 | 内容 |
| --- | --- |
| `input/` | 输入照片的副本 |
| `mask/` | 云区掩码；使用 `--save-overlay` 时另有叠加预览 |
| `replan/<时间戳>_mask/` | 场景、重排计划、延后目标、`replanned_sequence.ninaTargetSet` 和天空分布图 |

批量处理附带照片可运行：

```bash
python run_pipeline.py --mode batch --position-name position_5 --count-mode model_count
```

批量结果写入 `output_batch/`。`model_count` 按模型掩码统计；默认的 `label_count` 还需要匹配的标签掩码，这些标签未随仓库提供。自己的序列可通过 `--raw-img-dir` 指定照片目录，使用标签统计时再通过 `--label-mask-dir` 指定标签目录。

## 相机标定

已有参数不一定适用于其他相机。重新标定时，需要自备星点表 `targetlist.txt` 和初始图像中心 `x0y0.txt`，放到 `sequence_adjust_tool/scheme_horizon_to_image/`，并核对 `AltAzModeling.py` 中的拍摄时间与台站设置，再从该目录运行：

```bash
python AltAzModeling.py
```

脚本生成 `x0y0_new.txt` 和 `parameters_new.txt`。将数值写入 `sequence_adjust_tool/parameters.json` 的相应 `position`，再用 `--position-name` 选择。重排与地平标注都从该 JSON 读参数，投影使用 `E/a0/e/k1..k4` 的完整解析逆投影；`r = ze / k1` 近似不适用于这组参数。

回到项目目录后，可单独检查地平标注：

```bash
python sequence_adjust_tool/scheme_horizon_to_image/标注地平坐标系.py --image /path/to/image_or_mask.png --parameters sequence_adjust_tool/parameters.json --position-name position_5 --output output/debug_horizon_overlay.png
```

## 代码与技能入口

`run_pipeline.py` 依次调用 `Pic2mask/full_image_infer/infer_full_image.py` 生成掩码，再由 `sequence_adjust_tool/scheme_horizon_to_image/replan.py` 将目标的 RA/Dec 转为 Az/Alt 和图像坐标，检查掩码后重排计划。

依赖和标定准备好后，也可按 [photo-to-replan](skill/photo-to-replan/SKILL.md) 交给 Codex 使用。这个项目自带的技能未列入根目录 `skills/` 的 17 个安装技能；可将其说明文件复制到 `~/.codex/skills/photo-to-replan/SKILL.md`。使用时仍须提供完整项目的位置，复制说明文件不会安装模型、依赖或标定材料。
