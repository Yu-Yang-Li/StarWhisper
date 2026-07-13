# Early Classification from Sparse Light Curves

This repository implements the full benchmark pipeline: segmenting sparse ZTF/ATLAS light curves, extracting handcrafted features or end-to-end sequences, training eleven model configurations, and aggregating test-set metrics. The main **varlen** benchmark uses **3–30 observations** per segment, **seven merged variability classes**, and a **75 / 10 / 15** train/validation/test split (`random_state = 42`).

**Models compared**

| Group | Models |
|-------|--------|
| Handcrafted + XGBoost | Full (57 features), Reduced (39 non–Lomb-Scargle features) |
| Handcrafted + Transformer | 50-obs pretrain → varlen finetune; varlen from scratch |
| End-to-end Transformer | Lightweight (~11M) and matched-size (~227M); 50-obs pretrain → varlen finetune; matched scratch |
| End-to-end LSTM | Varlen baseline |

Experiment IDs, output paths, and metric files are defined in `train_models/experiment_registry.py`.

## Repository contents

- `data/` — preprocessing scripts; stratified split indices under `data/split/{varlen,50obs,1121}/`
- `features/` — feature extraction scripts (`extract_features_full.py`, `extract_features_reduced.py`); `legacy/` — historical scripts
- `train_models/` — training, fine-tuning, evaluation, and inference-benchmark scripts
- `train_models/results/` — aggregated comparison tables, feature-ablation outputs, inference-timing reports, and summary plots
- `manuscript/figures/` — figures (`fig_*.pdf`) and regeneration scripts (`plot_*.py`)

Per-run metrics and diagnostic plots (e.g. `test_metrics.json`, confusion matrices) are included under each model’s `results/` subdirectory where available.

## Environment

**Python ≥ 3.10** required.

```bash
pip install -r requirements.txt
# GPU: install the matching PyTorch build from https://pytorch.org/get-started/locally/
```

## Reproduction

Run from the repository root after installing dependencies:

```bash
# Segments & splits
python data/create_sparse_segments.py --pool varlen
python data/build_split_manifests.py
python data/build_split_indices.py
python data/prepare_data_split.py

# Handcrafted features (57-feature set)
python features/extract_features_full.py
python data/build_handcrafted_features.py --set 1117

# End-to-end models (example: matched-size Transformer)
python data/prepare_e2e_varlen.py
python train_models/train_e2e_transformer_50obs_matched_pretrain.py
python train_models/finetune_e2e_transformer_varlen_matched.py

# Summarize & benchmark
python train_models/summarize_model_comparison.py
python train_models/run_benchmark_inference.py --device gpu --skip-legacy
```

Other training scripts follow the same naming pattern as their output directories (see `experiment_registry.py`).
