# Low-SNR Stellar Spectra as Language

Spectral diffusion and discrete-token modeling for stellar spectra at low signal-to-noise ratio. Pre-training uses long, high-SNR sequences (~3030 pixels); fine-tuning uses shorter sequences (~303 pixels) in SNR stages, such as 25–30 → 9–11 → 7–9.

[Model weights](https://huggingface.co/Jaredxjc/Low-SNR-Stellar-Spectra-as-Language) · [Original repository](https://github.com/Jared-web03/Low-SNR-Stellar-Spectra-as-Language) · [MIT license](LICENSE) · [StarWhisper](../README.md)

---

## Updates

- **Mar 31, 2026:** Public **training code**, **`code/` data pipeline**, and **English documentation**. **Model weights** are on [Hugging Face — Jaredxjc/Low-SNR-Stellar-Spectra-as-Language](https://huggingface.co/Jaredxjc/Low-SNR-Stellar-Spectra-as-Language) (MIT).
- **Mar 31, 2026:** **Full tokenized spectra datasets** for redistribution: **Coming soon** (will be linked from this README and the HF dataset card when released).

---

## Model weights

Pre-trained and staged fine-tuned checkpoints are hosted on Hugging Face:

[Jaredxjc/Low-SNR-Stellar-Spectra-as-Language on Hugging Face](https://huggingface.co/Jaredxjc/Low-SNR-Stellar-Spectra-as-Language)

Load with the same architecture as in `src/spectral_lm/` and training scripts under `scripts/` (set `VOCAB_PATH` to `vocab/vocabulary.csv`). The HF model card may be extended with exact file names and training details.

---

## Dataset

> **Data processing code — released.** The full pipeline for building tokenized training CSVs is in this repository: **`code/`** (LAMOST download, PHOENIX-based simulation, flux tokenization, catalog merge, augmentation) plus **`scripts/preprocess_data.py`** for generic CSV trees and **`vocab/vocabulary.csv`** for the fixed vocabulary.
>
> **Curated full dataset (ready-to-train CSVs) — coming soon.** We will publish a downloadable release (and/or a Hugging Face dataset) for users who do not wish to regenerate data from raw spectra. Until then, follow the **Data pipeline** section below and LAMOST data-access rules to produce files locally.

---

## Data pipeline

### Acquisition and processing workflow

1. **PHOENIX templates** — High-resolution synthetic spectra from the [PHOENIX grid (Göttingen)](https://phoenix.astro.physik.uni-goettingen.de/?page_id=15). Retrieve files over **FTP** with a standard client (e.g. FileZilla); a **full local mirror** is appropriate when generating large synthetic sets.
2. **Interpolation and low-resolution Ca II spectra** — Regrid PHOENIX onto the working wavelength scale in `code/interpolation&decrease_resolution/ShowOut/interpolation0415.py`. Then apply **`CaII.py`** / **`CaII_300p.py`**: reduce resolving power, inject **SNR-matched Gaussian noise**, and retain the **Ca II** region; **`CaII_300p.py`** yields **~300 pixels** per spectrum for short-sequence fine-tuning. Outputs use metallicity-style directories (e.g. `Z-1.0`–`Z-4.0`).
3. **LAMOST DR12** — Register at LAMOST; build a **CSV target list** via the [DR12 search portal](https://www.lamost.org/dr12/v1.1/search) (CSV output, SNR/SNRZ and [Fe/H] cuts, stellar type **Star**). Obtain a **PyLAMOST token** from the [user portal](https://www.lamost.org/lmusers/user/). Download and stage spectra with **`code/pylamost-master/pylamost.py`** and **`lamost_dr12_pipeline.py`**; stellar parameters come from the **survey catalog** bundled with the query.
4. **Augmentation and training CSVs** — Optional row-level augmentation in `code/lamost_data_augmentation/` (adds **`augmentation_id`**). Flux tokenization and catalog merge to the trainer’s column layout live in **`code/lamost_sft_data/`** (`lamost_flux_preprocessing.py`, `convert_lamost_to_pretrain_format.py`, `mix_spectra.py`).

### Pipeline summary (single reference)

| Topic | Summary |
|--------|---------|
| **Synthetic path** | PHOENIX → `interpolation0415.py` → `CaII.py` / `CaII_300p.py` (resolution + SNR + Ca II window; ~300 px variant for fine-tune). |
| **Observational path** | DR12 CSV → `pylamost` + `lamost_dr12_pipeline.py` → LAMOST spectra + catalog **Teff / log g / [Fe/H]**. |
| **Flux tokens** | Four digit columns per pixel (`flux_thu` … `flux_one`); sequence delimiters `<BOS>`, `<EOS>`, `<SEP>`. |
| **Label tokens** | \(T_\mathrm{eff}\) (five digits), log g (three digits + sign), [Fe/H] (two digits + sign); definition matches `convert_lamost_to_pretrain_format.py`. |
| **Training table** | One row per pixel; **20 columns** in the default layout (`spectrum_id`, `pixel_idx`, stellar and flux tokens, specials)—see `OUTPUT_COLUMNS` in `scripts/preprocess_data.py`. Augmented runs add **`augmentation_id`**. |
| **Splits** | Hash- or chunk-based **train/val** (e.g. ~9:1) in the streaming writers. |
| **Flux-only LAMOST** | `lamost_flux_preprocessing.py` writes flux tokens + `obsid` (+ `augmentation_id` if used); `convert_lamost_to_pretrain_format.py` **joins** a catalog with `obsid`, `teff`, `logg`, `feh`. |

### `code/` directory map

| Path | Purpose |
|------|---------|
| `code/pylamost-master/` | LAMOST API client, DR12 download pipeline (`lamost_dr12_pipeline.py`, `lamost_dr12_50.py`, samples). |
| `code/interpolation&decrease_resolution/` | Resolution reduction, SNR noise, Ca II extraction; `ShowOut/` holds helper notebooks and small parameter CSVs. |
| `code/lamost_sft_data/` | Flux tokenization, catalog merge to 20-column CSV, `mix_spectra.py`, `run1207.sh`. |
| `code/lamost_data_augmentation/` | As above plus **`augmentation_id`**; batch entry `run.sh`. |

*Dependencies (typical):* `pandas`, `numpy`, `tqdm`, `scikit-learn`; `astropy` / `scipy` where noted in scripts. Configure **absolute paths** inside those scripts for your environment.

---

## Training & reproduction

Run the commands below from `Low-SNR-Stellar-Spectra-as-Language/`. In the shell examples, `REPO_ROOT` means the absolute path to this directory, rather than the parent StarWhisper directory.

The canonical **model** and **train/finetune** entrypoints live outside `code/`:

| Path | Role |
|------|------|
| `src/spectral_lm/` | `SpectrumDiffusionModel` |
| `scripts/preprocess_data.py` | Tokenization & CSV generation for this repo’s trainer |
| `scripts/pretrain.py` | Pre-training |
| `scripts/finetune.py` | Fine-tuning |
| `vocab/vocabulary.csv` | Default vocabulary (`token_id`, `token`); see `vocab/README.md` |
| `legacy_pretrain/` | Optional legacy SLURM launcher (`launch_slurm_pretrain.sh`) |

### Prerequisites

- Python 3.10+ recommended, NVIDIA GPU(s), CUDA-compatible PyTorch.

```bash
pip install -r requirements.txt
```

Install `torch` for your CUDA version from [PyTorch](https://pytorch.org).

### Path placeholders (replace once, reuse everywhere)

| Placeholder | Meaning |
|-------------|---------|
| `REPO_ROOT` | Root of this repository (clone path). |
| `RAW_SPECTRA_DIR` | Directory tree of **input** spectrum CSV files (recursive scan). |
| `FITS_ROOT` | Parent folder that contains SNR subfolders (used with `--make_groups`). Example: `FITS_ROOT/SNR10/.../*.csv`. |
| `VOCAB_FILE` | Fixed vocabulary CSV. Default: `REPO_ROOT/vocab/vocabulary.csv`. |
| `PRETRAIN_DATA_DIR` | Folder with `spectrum_tokenized_train.csv`, `spectrum_tokenized_val.csv`, optional `spectrum_tokenized_val_subset.csv`. |
| `PRETRAIN_CKPT` | Checkpoint from pre-training (e.g. `checkpoint_step_9000.pth`). |
| `FT25_CKPT` / `FT911_CKPT` | Checkpoints after SNR stages. |
| `OUT_PRETRAIN`, `OUT_FT25`, … | Output directories for checkpoints and logs. |

**Linux / macOS**

```bash
export REPO_ROOT="/absolute/path/to/this/repo"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
```

**Windows (PowerShell)**

```powershell
$env:REPO_ROOT = "D:\absolute\path\to\this\repo"
$env:PYTHONPATH = "$env:REPO_ROOT\src"
```

---

### Stage 1 — Data preprocessing & tokenization (`scripts/preprocess_data.py`)

#### Mode A — Single root directory (CSVs in **current working directory**)

```bash
export REPO_ROOT="/absolute/path/to/this/repo"
export RAW_SPECTRA_DIR="/absolute/path/to/your/raw/csv_trees"

mkdir -p /absolute/path/to/output/pretrain_tokenized
cd /absolute/path/to/output/pretrain_tokenized

python "${REPO_ROOT}/scripts/preprocess_data.py" \
  --data_dir "${RAW_SPECTRA_DIR}" \
  --processes 8
```

#### Mode B — SNR folders under `FITS_ROOT` (`--make_groups`)

```bash
export REPO_ROOT="/absolute/path/to/this/repo"
export FITS_ROOT="/absolute/path/to/R1800FITS"

cd "${REPO_ROOT}"
cp -n vocab/vocabulary.csv ./vocabulary.csv   # optional OOV check in cwd

python scripts/preprocess_data.py \
  --make_groups \
  --fits_root "${FITS_ROOT}" \
  --pretrain_snrs "SNR10,SNR20" \
  --finetune_snrs "SNR1" \
  --processes 8 \
  --batch_gb 2.0
```

You can also set **`FIXED_VOCAB_PATH`** to an absolute path to `vocabulary.csv`.

#### Step-eval CSV for fine-tuning

```bash
export VAL_CSV_FULL="/absolute/path/to/spectrum_tokenized_val.csv"
export STEP_EVAL_CSV="/absolute/path/to/spectrum_tokenized_val_first1500.csv"

python - <<'PY'
import os
import pandas as pd
src = os.environ["VAL_CSV_FULL"]
dst = os.environ["STEP_EVAL_CSV"]
df = pd.read_csv(src)
ids = df["spectrum_id"].drop_duplicates().unique()[:1500]
df[df["spectrum_id"].isin(ids)].to_csv(dst, index=False)
print("wrote", dst, "spectra", len(ids))
PY
```

---

### Stage 2 — Pre-training

```bash
export REPO_ROOT="/absolute/path/to/this/repo"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
export PRETRAIN_DATA_DIR="/absolute/path/to/pretrain_tokenized"
export VOCAB_FILE="${REPO_ROOT}/vocab/vocabulary.csv"
export OUT_PRETRAIN="${REPO_ROOT}/output/pretrain"
export TRAIN_CSV="${PRETRAIN_DATA_DIR}/spectrum_tokenized_train.csv"
export VAL_CSV="${PRETRAIN_DATA_DIR}/spectrum_tokenized_val.csv"
export VAL_SUBSET_CSV="${PRETRAIN_DATA_DIR}/spectrum_tokenized_val_subset.csv"
export VOCAB_PATH="${VOCAB_FILE}"
export OUTPUT_DIR="${OUT_PRETRAIN}"
export MASK_TOKEN_ID="2"
mkdir -p "${OUTPUT_DIR}"
export NPROC_PER_NODE=4
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" \
  "${REPO_ROOT}/scripts/pretrain.py"
```

Single GPU: `NPROC_PER_NODE=1` and `torchrun --standalone --nproc_per_node=1 ...`.

---

### Stage 3 — Fine-tuning (staged SNR)

```bash
export REPO_ROOT="/absolute/path/to/this/repo"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
export VOCAB_FILE="${REPO_ROOT}/vocab/vocabulary.csv"
export BATCH_SIZE=64
export NPROC_PER_NODE=1
export PRELOAD_TO_MEMORY=0
export STREAMING_MODE=1
export PARAM_STATS_SAMPLES=256
```

**3a — SNR ~25–30** (from pretrain checkpoint)

```bash
export PRETRAIN_CKPT="/absolute/path/to/output/pretrain/checkpoint_step_9000.pth"
export FT_DATA="/absolute/path/to/finetune_snr25_30_tokenized"
export TRAIN_CSV="${FT_DATA}/spectrum_tokenized_train.csv"
export VAL_CSV="${FT_DATA}/spectrum_tokenized_val.csv"
export STEP_EVAL_CSV="${FT_DATA}/spectrum_tokenized_val_first1500.csv"
export VOCAB_PATH="${VOCAB_FILE}"
export PRETRAIN_CKPT_PATH="${PRETRAIN_CKPT}"
export RESUME_FROM="${PRETRAIN_CKPT}"
export CKPT_DIR="/absolute/path/to/output/finetune_snr25_30/ckpts"
export LOG_PATH="/absolute/path/to/output/finetune_snr25_30/logs/run.log"
mkdir -p "$(dirname "${LOG_PATH}")" "${CKPT_DIR}"
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  "${REPO_ROOT}/scripts/finetune.py"
```

**3b / 3c** — chain from `FT25_CKPT` and `FT911_CKPT` similarly (see previous README versions or `examples/env_finetune_snr_*.example.sh`).

**Shell helpers**

```bash
bash examples/launch_pretrain.example.sh
export FINETUNE_ENV_FILE="/absolute/path/to/your_env.sh"
bash examples/launch_finetune.example.sh
```

---

## Local configuration

Some scripts under `code/` retain paths from the original computing environment. Set the data and output paths for your own environment before running them. The complete tokenized dataset is still listed as coming soon; model weights are linked above.

## Citation

If you use this work, please cite the repository and the Hugging Face model card. A BibTeX entry will be added when the paper is public.
