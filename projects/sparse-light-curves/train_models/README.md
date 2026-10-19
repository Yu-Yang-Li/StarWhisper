# Training and evaluation scripts

[Project setup and data](../README.md)

Each training script has a corresponding experiment in `experiment_registry.py`.
That registry, `split_utils.py` and the entry scripts use the same project root
or `STARWHISPER_SPARSE_ROOT`. Model hyperparameters, class definitions, published
metrics and split files retain their original experimental settings.

| Task | Entry or shared code |
| --- | --- |
| XGBoost full / reduced features | `train_xgboost_full.py`, `train_xgboost_reduced.py` |
| Feature Transformer | `train_transformer_feat_50obs_pretrain.py`, `finetune_transformer_feat_varlen.py`, `train_transformer_feat_varlen_scratch.py` |
| End-to-end Transformer | `train_e2e_transformer_50obs_pretrain.py`, `finetune_e2e_transformer_varlen.py`, and the matching-size pretrain / finetune / scratch scripts |
| Variable-length LSTM | `train_lstm_varlen.py` |
| Shared arrays, splits and models | `split_utils.py`, `dl_common.py`, `e2e_transformer_matched_arch.py` |
| Current XGBoost diagnostics | `plot_xgboost_full_results.py`, `plot_xgboost_reduced_results.py` |
| Feature ablation | `run_feature_ablation.py`; its correlation diagnostic also needs the earlier `xgboost_optuna_1117` / `1121` feature-importance CSVs |
| Inference timing | `run_benchmark_inference.py` delegates execution to `benchmark_inference_core.py` and supplies shared feature-extraction / legacy-model helpers; both files are used |
| Aggregate published metrics | `summarize_model_comparison.py` |

Directories named after experiments contain their saved configurations,
results and diagnostic plots. They remain with the scripts they document.
The earlier complete-test-CSV evaluation and distribution plots are under
[`legacy/train_models`](../legacy/README.md), with their original cohort rules.
