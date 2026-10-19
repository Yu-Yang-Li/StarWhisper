# 稀疏光变的早期分类

对稀疏 ZTF / ATLAS 光变片段进行分类，比较手工特征模型和直接读取时序的模型。主实验 `varlen` 使用每段 3–30 个观测点、7 个合并类别，以及 75 / 10 / 15 的训练、验证、测试划分（`random_state = 42`）。

[项目目录](../README.md) · [研究文档](../../docs/research.md) · [结果工具说明](../../skills/starwhisper-varlen/SKILL.md) · [模型权重](https://huggingface.co/castor0705/sparse-lc-early-classification)

仓库附有 11 个模型配置的结果、划分索引和图表，可以先离线比较指标。原始光变片段与训练使用的特征 CSV 未附带，重新训练需要另外准备；权重通过上面的 Hugging Face 链接获取。

## 先看已有结果

需要 Python 3.10 或更新版本，只依赖标准库。从本目录（`projects/sparse-light-curves/`）运行：

```bash
python ../../skills/starwhisper-varlen/scripts/eval_varlen.py contract
python ../../skills/starwhisper-varlen/scripts/eval_varlen.py table --pool varlen
python ../../skills/starwhisper-varlen/scripts/eval_varlen.py best --pool varlen
```

工具读取随技能附带的已发表指标，不需要安装训练依赖、下载模型或重建划分。`contract` 说明数据设定，`table` 和 `best` 在指定数据池内比较结果。`varlen`、`50obs` 与 `1121` 是不同的数据池；50 点预训练成绩不能作为 3–30 点早期分类的成绩。

模型比较包括：

| 模型组 | 配置 |
| --- | --- |
| 手工特征 + XGBoost | 完整 57 维特征；移除 Lomb–Scargle 后的 39 维特征，两者分别来自 `varlen` 和 `1121` 池 |
| 手工特征 + Transformer | 50 点预训练、变长微调，以及变长从头训练 |
| 端到端 Transformer | 轻量模型（约 11M）与同量级模型（约 227M）的预训练、微调配置，以及同量级从头训练 |
| 端到端 LSTM | 变长时序基线 |

每个实验的指标与输出位置见 [experiment_registry.py](train_models/experiment_registry.py)。对应目录保存 `test_metrics.json` 或 `metrics.txt`、配置和已有诊断图；汇总表、特征消融结果、推理耗时与图表在 [train_models/results/](train_models/results/)，论文图和绘图脚本在 [manuscript/figures/](manuscript/figures/)。

## 重新训练需要准备什么

训练使用 Python 3.10 或更新版本和本项目的依赖。从本目录安装：

```bash
pip install -r requirements.txt
```

GPU 训练还需按本机 CUDA 环境选择匹配的 [PyTorch](https://pytorch.org/get-started/locally/)。不同模型的训练入口见 [train_models/README.md](train_models/README.md)。

数据不能由权重文件代替，至少需要下面这些输入：

| 输入 | 放在哪里、怎样对应 |
| --- | --- |
| 光变片段 CSV | `train2/` 对应 3–30 点，`train4/` 对应固定 50 点；原始类别见 [category_config.py](data/category_config.py) |
| 匹配的特征 CSV | `features/`；文件名与数据池须同时对应 [prepare_data_split.py](data/prepare_data_split.py) 的 `POOLS` 和 [split_utils.py](train_models/split_utils.py) 的 `FEATURE_POOLS` |
| 划分与清单 | `data/split/{varlen,50obs,1121}/` 已附带索引；清单中的样本顺序和对应光变必须匹配 |
| 模型权重 | 重新推理或接续训练时，下载对应配置的权重，放到该实验脚本指定的位置；从头训练不需要已有权重 |

新实验的数据和主要输出可以放到单独目录。路径默认指向本项目；设置下面的变量可改用其他根目录，内部仍保留 `train2/`、`train4/`、`features/`、`data/split/` 和 `train_models/` 的相对布局：

```bash
export STARWHISPER_SPARSE_ROOT="/absolute/path/to/your/experiment"
```

PowerShell 对应 `$env:STARWHISPER_SPARSE_ROOT = 'D:/path/to/experiment'`。切片与划分脚本的部分日志、片段统计仍写到源码的 `data/` 下，运行前也要保留其中已有结果。

## 数据准备与训练顺序

下面以同量级端到端 Transformer 为例，命令都从本项目目录运行，并假定相应输入已经备齐。它们是不同步骤，切分原始光变并不会自动生成训练所需的全部特征 CSV。

如果从新的原始光变开始，可先生成变长片段：

```bash
python data/create_sparse_segments.py --source-dir /path/to/raw_curves --pool varlen
```

特征脚本在 `features/`。固定 50 点的完整特征提取与较早的变长完整特征提取使用不同输入池，选择前请看 [legacy/README.md](legacy/README.md)。新生成的 CSV 通常带时间戳，需核对内容、平衡方式与上述两处文件映射，不能只改文件名就视为原实验数据。

准备好匹配的特征 CSV 后，再建立划分与端到端数组：

```bash
python data/prepare_data_split.py
python data/prepare_e2e_data.py
python data/prepare_e2e_varlen.py
```

第一条会重新写入划分和清单。仅查看已发表结果时无需运行；复现原实验时，应保留对应的样本和划分。随后进行 50 点预训练与变长微调：

```bash
python train_models/train_e2e_transformer_50obs_matched_pretrain.py
python train_models/finetune_e2e_transformer_varlen_matched.py
```

微调需要上一步对应的预训练输出。其他模型按[训练脚本说明](train_models/README.md)选择入口；部分脚本执行即开始训练，不提供通用的 `--help` 模式。

在所选实验目录已有指标或模型后，可分别重新生成汇总和测量推理耗时：

```bash
python train_models/summarize_model_comparison.py
python train_models/run_benchmark_inference.py --device gpu --skip-legacy
```

汇总脚本会写表格与图，推理计时还需要模型和测试输入。只想看现有比较时，使用前面的离线读表命令即可。

## 其他脚本

`data/build_handcrafted_features.py` 根据当前清单额外构建 `data/features/` 缓存；上面的训练脚本读取的是 `FEATURE_POOLS` 指定的 CSV，两者不能直接混用。早期的原始目录划分、特定数据集特征与诊断脚本保留在 [legacy/](legacy/README.md)，对应各自的实验条件。

装好项目依赖后，可运行 `python -m pytest tests -q` 检查输入定位和预处理。这些检查使用临时文件，不训练模型。
