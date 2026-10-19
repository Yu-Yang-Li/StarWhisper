# Earlier spectrum experiments

[Current project](../README.md)

Use `scripts/preprocess_data.py`, `scripts/pretrain.py` and
`scripts/finetune.py` for the documented workflow. This directory retains
earlier implementations and their own model source for reproducibility.

| Location | Contents |
| --- | --- |
| `preprocess_data.py` | Earlier tokenization implementation; requires explicit `--data_dir` or `--fits_root` |
| `pretrain/` | Earlier pretrainer, matching model source and site-specific SLURM launcher |
| `lamost/lamost_dr12_50.py` | Earlier spectrum-preview experiment; its historical sample limit is retained |

The SLURM launcher still contains cluster-specific module and resource settings;
review those before use. Its local data paths can be set through `DATA_ROOT`,
`TRAIN_CSV`, `VAL_CSV` and `VOCAB_PATH`. LAMOST experiments require your own
`LAMOST_TOKEN` and use `LAMOST_DATA_ROOT` for local files.

The incomplete `download_one.py` sketch was removed. Its source remains in
Git history; the current download pipeline is under `code/pylamost-master/`.
