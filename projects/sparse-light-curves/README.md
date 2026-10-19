# Early Classification from Sparse Light Curves

This directory contains scripts for segmenting sparse ZTF/ATLAS light curves, extracting handcrafted features or end-to-end sequences, training eleven model configurations, and aggregating test-set metrics. The main **varlen** benchmark uses **3–30 observations** per segment, **seven merged variability classes**, and a **75 / 10 / 15** train/validation/test split (`random_state = 42`).

[StarWhisper](../../README_EN.md) · [Research notes](../../docs/research-en.md) · [Read the published metrics](../../skills/starwhisper-varlen/SKILL.md)

**Models compared**

| Group | Models |
|-------|--------|
| Handcrafted + XGBoost | Full (57 features), Reduced (39 non–Lomb-Scargle features) |
| Handcrafted + Transformer | 50-obs pretrain → varlen finetune; varlen from scratch |
| End-to-end Transformer | Lightweight (~11M) and matched-size (~227M); 50-obs pretrain → varlen finetune; matched scratch |
| End-to-end LSTM | Varlen baseline |

Experiment IDs, output paths, and metric files are defined in `train_models/experiment_registry.py`.

## Repository contents

- `data/` — current preprocessing scripts, shared curve lookup/band encoding, and stratified split indices under `data/split/{varlen,50obs,1121}/`
- `features/` — current full/reduced feature extractors and CSV utilities
- `train_models/` — current training, fine-tuning, evaluation, and inference-benchmark scripts; see the [script map](train_models/README.md)
- `train_models/results/` — aggregated comparison tables, feature-ablation outputs, inference-timing reports, and summary plots
- `manuscript/figures/` — figures (`fig_*.pdf`) and regeneration scripts (`plot_*.py`)
- `legacy/` — earlier raw-folder splits, dataset-specific feature variants and diagnostic scripts; see the [original roles](legacy/README.md)

Per-run metrics and diagnostic plots (e.g. `test_metrics.json`, confusion matrices) are included under each model’s `results/` subdirectory where available.

## Pre-trained weights

The best model checkpoints for all configurations are available on Hugging Face Hub:
**[castor0705/sparse-lc-early-classification](https://huggingface.co/castor0705/sparse-lc-early-classification)**.
Use `hf download castor0705/sparse-lc-early-classification` to retrieve them locally.

## Environment

**Python ≥ 3.10** required.

```bash
pip install -r requirements.txt
# GPU: install the matching PyTorch build from https://pytorch.org/get-started/locally/
```

## Reproduction

Run from this directory (`projects/sparse-light-curves/`) after installing dependencies. Data and model paths default to this directory, independent of the shell's working directory. For a new run, use a separate data/output root:

```bash
export STARWHISPER_SPARSE_ROOT="/absolute/path/to/your/experiment"
```

On PowerShell, use `$env:STARWHISPER_SPARSE_ROOT = 'D:\path\to\experiment'`.
Keep raw curves under `train2/` or `train4/`, feature CSVs under `features/`,
splits under `data/split/`, and model outputs under `train_models/` in that root.

Raw curves and balanced feature CSVs are not bundled. The published runs use
the exact filenames listed in `data/prepare_data_split.py` (`POOLS`) and
`train_models/split_utils.py` (`FEATURE_POOLS`). Prepare those inputs or update
both mappings for a new run. The 50-point full-feature extractor and earlier
variable-length full-feature variant use different source pools; see
`legacy/README.md` before selecting a generator.

The commands below assume those inputs are ready. Reading the published
results does not require rebuilding split indices. New preprocessing and
training writes to the selected experiment root.

```bash
# Optional: segment a new raw light-curve collection
python data/create_sparse_segments.py --source-dir /path/to/raw_curves --pool varlen

# Build current splits from the prepared, matched feature CSVs
python data/prepare_data_split.py

# End-to-end models (example: matched-size Transformer)
python data/prepare_e2e_data.py
python data/prepare_e2e_varlen.py
python train_models/train_e2e_transformer_50obs_matched_pretrain.py
python train_models/finetune_e2e_transformer_varlen_matched.py

# Summarize & benchmark
python train_models/summarize_model_comparison.py
python train_models/run_benchmark_inference.py --device gpu --skip-legacy
```

Other training scripts follow the same naming pattern as their output directories (see `experiment_registry.py`).

`data/build_handcrafted_features.py` optionally builds an additional cache
under `data/features/` from the current manifest; the training scripts above
read the matched CSVs defined in `FEATURE_POOLS`. Earlier raw-folder splitting
scripts are kept under `legacy/data/` and are outside this workflow.

After installing the project dependencies, `python -m pytest tests -q` checks
input routing and preprocessing on temporary files without training a model.
