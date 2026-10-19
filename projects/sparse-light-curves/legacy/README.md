# Earlier experiment scripts

[Current project](../README.md)

These scripts retain the data conventions of earlier runs. Their sample pools
and outputs can differ from the current `varlen`, `50obs` and `1121` comparison.
Use them to reproduce the corresponding run; their results do not replace the
published split indices or metrics.

| Location | Original role |
| --- | --- |
| `data/build_split_manifests.py` | Build manifests directly from raw `train2` / `train4` folders |
| `data/build_split_indices.py` | Split those manifests into `data/split/train2` and `train4`; current trainers use `varlen`, `50obs` and `1121` instead |
| `features/extract_features_baseline.py` | Earlier fixed-length feature extraction |
| `features/extract_features_full_varlen.py` | Full-feature extraction for the earlier variable-length dataset |
| `features/extract_features_full_supernova.py` | Supernova feature extraction variant |
| `train_models/evaluate_xgboost_full.py` | Evaluate the earlier complete feature test CSV |
| `train_models/plot_test_distribution.py` | Plot the earlier feature CSV distributions |

Paths default to this project's root and accept `STARWHISPER_SPARSE_ROOT` for
external data and outputs. These scripts require the original input files.
The active entrypoints and shared benchmark helpers remain under `data/`,
`features/` and `train_models/`.
