# 低信噪比恒星光谱分析

Low-SNR Stellar Spectra as Language 将恒星光谱表示为离散 token，并使用光谱扩散模型进行建模。预训练使用约 3030 像素的长序列；微调使用约 303 像素的短序列，按 SNR 25–30 → 9–11 → 7–9 等阶段接续训练。

[项目目录](../README.md) · [研究文档](../../docs/research.md) · [原始仓库](https://github.com/Jared-web03/Low-SNR-Stellar-Spectra-as-Language) · [模型权重](https://huggingface.co/Jaredxjc/Low-SNR-Stellar-Spectra-as-Language) · [MIT 许可](LICENSE)

这里提供模型、数据处理和训练代码，以及固定词表。可以直接阅读这些材料；完整的原始光谱和可用于训练的 CSV 未随仓库提供。预处理需要自备光谱，训练需要相应环境，微调还需要前一阶段的权重。本目录没有附带可直接运行的单条光谱预测示例。

## 先选入口

| 想做的事 | 从哪里开始 | 需要准备什么 |
| --- | --- | --- |
| 看模型和词表 | [src/spectral_lm/](src/spectral_lm/)、[vocab/](vocab/README.md) | 阅读不需要额外数据 |
| 将光谱转成训练 CSV | [scripts/preprocess_data.py](scripts/preprocess_data.py) | 符合输入格式的光谱 CSV，或按 SNR 分组的目录 |
| 预训练 | [examples/launch_pretrain.example.sh](examples/launch_pretrain.example.sh) → `scripts/pretrain.py` | 训练、验证 CSV，固定词表和训练环境 |
| 分阶段微调 | [examples/launch_finetune.example.sh](examples/launch_finetune.example.sh) → `scripts/finetune.py` | 本阶段数据、前一阶段权重，以及启用步级评估时的验证子集 |
| 从 PHOENIX / LAMOST 准备数据 | 下方“光谱从哪里来” | 原始模板或巡天数据、星表参数及相应访问权限 |

当前训练入口在 `scripts/`，数据生成工具在 `code/`。较早的预处理、预训练和台站作业脚本保留在 [legacy/](legacy/README.md)，不需要与当前入口一起运行。

## 环境

建议使用 Python 3.10 或更新版本。训练需要 NVIDIA GPU 与匹配的 CUDA / PyTorch；按 [PyTorch 安装说明](https://pytorch.org/get-started/locally/) 准备后，从本目录（`projects/stellar-spectra/`）安装依赖：

```bash
pip install -r requirements.txt
```

下面的数据处理与训练例子使用 Bash。除明确切换到输出目录的步骤外，都从本项目目录运行。原始数据处理还会按所选工具用到 `astropy`、`scipy` 等依赖，见对应脚本的导入项。

## 把光谱转成训练表

### 已有光谱 CSV

通用预处理器递归读取 `--data_dir` 下的 CSV。输入的文件命名需包含恒星参数，例如 `Z-1.0/10560+3.20-1.0.csv`；具体读取与参数提取规则见 `scripts/preprocess_data.py`。LAMOST 的原始 FITS 需先经过下方专用处理工具，不能直接作为这里的输入。

输出写到当前工作目录。下面先在项目目录记录代码位置，再切换到一个新的输出目录，避免覆盖已有数据：

```bash
export SPECTRA_PROJECT="$(pwd)"
export RAW_SPECTRA_DIR="/absolute/path/to/raw_spectrum_csvs"
export FIXED_VOCAB_PATH="${SPECTRA_PROJECT}/vocab/vocabulary.csv"
mkdir -p /absolute/path/to/tokenized
cd /absolute/path/to/tokenized
python "${SPECTRA_PROJECT}/scripts/preprocess_data.py" --data_dir "${RAW_SPECTRA_DIR}" --processes 8
```

生成的训练和验证表为 `spectrum_tokenized_train.csv`、`spectrum_tokenized_val.csv`。一条光谱占多行，每行对应一个像素；列顺序见脚本中的 `OUTPUT_COLUMNS`，包括恒星参数、四位 flux token 和 `<BOS>`、`<EOS>`、`<SEP>`。增广流程还会保留 `augmentation_id`。训练时使用仓库的固定 `vocab/vocabulary.csv`，不要直接用本批数据统计的 token 表替换其编号。

参数 token 分别用五位表示有效温度、三位加符号表示 log g、两位加符号表示 [Fe/H]。通用模式按 `spectrum_id` 做 9:1 的训练 / 验证划分，`random_state = 42`，同一条光谱的像素行保留在同一组。

### 按 SNR 分组的 CSV

若输入为 `FITS_ROOT/SNR10/.../*.csv` 这样的目录，可在新的输出目录中改用分组模式。这里的目录名沿用 `FITS_ROOT`，实际读取的仍是 CSV：

```bash
export SPECTRA_PROJECT="/absolute/path/to/StarWhisper/projects/stellar-spectra"
export FIXED_VOCAB_PATH="${SPECTRA_PROJECT}/vocab/vocabulary.csv"
mkdir -p /absolute/path/to/grouped_tokenized
cd /absolute/path/to/grouped_tokenized
python "${SPECTRA_PROJECT}/scripts/preprocess_data.py" --make_groups --fits_root /absolute/path/to/R1800FITS --pretrain_snrs SNR10,SNR20 --finetune_snrs SNR1 --processes 8 --batch_gb 2.0
```

预训练和微调表分别写入当前目录的 `pretrain_data/` 与 `finetune_data/`。SNR 目录名只是选择输入的示例，应按本次实验准备。`FIXED_VOCAB_PATH` 可显式指定固定词表；未指定时，程序依次查找当前目录的 `vocabulary.csv` 和项目自带词表。

分组模式按输入文件路径的哈希划分训练 / 验证，比例约为 9:1；改变输入路径会影响这一划分，复现时需保留相应清单。

## 预训练

返回项目目录，指定上一步实际生成的训练表所在目录和模型输出位置：

```bash
cd /absolute/path/to/StarWhisper/projects/stellar-spectra
export DATA_ROOT="/absolute/path/to/tokenized"
export OUTPUT_DIR="/absolute/path/to/output/pretrain"
NPROC_PER_NODE=1 bash examples/launch_pretrain.example.sh
```

`DATA_ROOT` 下应有 `spectrum_tokenized_train.csv` 和 `spectrum_tokenized_val.csv`。启动脚本默认使用项目词表，设置 `PYTHONPATH` 并调用 `torchrun`；`NPROC_PER_NODE=1` 表示单 GPU，脚本默认值为 4。需要其他位置时，可分别设置 `TRAIN_CSV`、`VAL_CSV`、`VOCAB_PATH`。可选的 `VAL_SUBSET_CSV` 用于步级验证，缺少时预训练程序回退到完整验证表。

## 分阶段微调

先复制对应的环境示例，在自己的副本中修改所有数据、权重、输出与日志路径，再通过 `FINETUNE_ENV_FILE` 加载。三个阶段分别使用各自的数据：

| 阶段 | 环境示例 | 起始权重 |
| --- | --- | --- |
| SNR 25–30 | [env_finetune_snr_25_30.example.sh](examples/env_finetune_snr_25_30.example.sh) | 预训练权重 |
| SNR 9–11 | [env_finetune_snr_9_11.example.sh](examples/env_finetune_snr_9_11.example.sh) | 前一阶段的微调权重 |
| SNR 7–9 | [env_finetune_snr_7_9.example.sh](examples/env_finetune_snr_7_9.example.sh) | 前一阶段的微调权重 |

启动前会检查 `TRAIN_CSV`、`VAL_CSV`、`VOCAB_PATH` 和 `PRETRAIN_CKPT_PATH`。默认每 500 步评估一次，还需明确指定本阶段的 `STEP_EVAL_CSV`；不会用其他数据组代替。若不做步级评估，显式设置 `STEP_EVAL_FREQ_STEPS=0`。

例如，可从本阶段验证表中选取前 1500 个 `spectrum_id`，保留各自完整的像素行，生成步级验证子集：

```bash
export VAL_CSV="/absolute/path/to/current_stage/spectrum_tokenized_val.csv"
export STEP_EVAL_CSV="/absolute/path/to/current_stage/spectrum_tokenized_val_first1500.csv"
python - <<'PY'
import os
import pandas as pd
frame = pd.read_csv(os.environ["VAL_CSV"])
ids = frame["spectrum_id"].drop_duplicates().iloc[:1500]
frame[frame["spectrum_id"].isin(ids)].to_csv(os.environ["STEP_EVAL_CSV"], index=False)
PY
```

环境示例还包含 `PARAM_STATS_JSON`。有对应统计文件时填写其路径；若希望从训练集估计参数统计量，移除该变量的设置，由 `PARAM_STATS_SAMPLES` 控制采样量。不同阶段不要沿用无关数据的统计文件。

输入准备好后，从项目目录启动：

```bash
export FINETUNE_ENV_FILE="/absolute/path/to/your_finetune_env.sh"
NPROC_PER_NODE=1 bash examples/launch_finetune.example.sh
```

已发布权重见 [Hugging Face 模型页](https://huggingface.co/Jaredxjc/Low-SNR-Stellar-Spectra-as-Language)。加载时使用 `src/spectral_lm/` 的对应结构与固定词表；下载权重不会同时获得原训练数据。

## 光谱从哪里来

| 数据来源或步骤 | 工具与输入 |
| --- | --- |
| PHOENIX 模板 | 从 [Göttingen PHOENIX grid](https://phoenix.astro.physik.uni-goettingen.de/?page_id=15) 获取原始高分辨率光谱；可用标准 FTP 客户端下载 |
| 模拟低分辨率光谱 | `code/interpolation&decrease_resolution/ShowOut/interpolation0415.py` 做波长重采样，`CaII.py` / `CaII_300p.py` 降分辨率、加入对应 SNR 的高斯噪声并提取 Ca II 区域；后者输出约 300 像素的短序列 |
| LAMOST DR12 | 在 [DR12 检索页](https://www.lamost.org/dr12/v1.1/search) 准备目标 CSV 和参数星表，在[用户页面](https://www.lamost.org/lmusers/user/)取得自己的访问 token；当前下载入口是 `code/pylamost-master/pylamost-master/lamost_dr12_pipeline.py`，与同目录的 `pylamost.py` 配合使用 |
| LAMOST 转训练表 | `code/lamost_sft_data/lamost_flux_preprocessing.py` 生成 flux token，`convert_lamost_to_pretrain_format.py` 按 `obsid` 合并含 `teff`、`logg`、`feh` 的星表；`mix_spectra.py` 用于混合表格，`run1207.sh` 是批量处理示例 |
| 数据增广 | `code/lamost_data_augmentation/` 的对应流程保留 `augmentation_id`，批量入口为 `run.sh` |

LAMOST 检索需按研究目标选择恒星类型、SNR / SNRZ 与金属丰度范围，并遵守数据访问规则。各处理脚本的路径设置如下：

| 流程 | 路径设置 |
| --- | --- |
| PHOENIX 处理 | `PHOENIX_INPUT_ROOT`、`PHOENIX_OUTPUT_ROOT`；默认在本项目 `data/` 下 |
| LAMOST 下载 | `LAMOST_TOKEN`、`LAMOST_DATA_ROOT`，可选 `LAMOST_WAVELENGTH_FILE` |
| LAMOST token 化 | `--data_dir`；批处理脚本使用 `LAMOST_DATA_ROOT` |
| 混合训练表 | 四个位置参数，或 `MIX_A_PATH`、`MIX_B1_PATH`、`MIX_B2_PATH`、`MIX_OUT_DIR` |

## 检查与引用

装好项目依赖后，可从本目录运行 `python -m pytest tests -q`，检查输入条件、词表定位与导入行为；这些检查不训练模型或下载光谱。

使用本工作时，请引用[原始仓库](https://github.com/Jared-web03/Low-SNR-Stellar-Spectra-as-Language)和[模型卡](https://huggingface.co/Jaredxjc/Low-SNR-Stellar-Spectra-as-Language)，并保留本目录的 MIT 许可。
