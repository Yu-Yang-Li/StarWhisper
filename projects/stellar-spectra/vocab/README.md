# Vocabulary

This directory holds the **fixed token vocabulary** for training.

## File

| File | Description |
|------|-------------|
| `vocabulary.csv` | Two columns: `token_id` (int), `token` (string). The supplied vocabulary uses `<BOS>`, `<EOS>`, `<SEP>` at ids 0–2. |

Training scripts read the path from the environment variable **`VOCAB_PATH`**. If unset, `scripts/finetune.py` defaults to `vocab/vocabulary.csv` relative to the repository root.

## Preprocessing

For `scripts/preprocess_data.py --make_groups`, place a copy of this file in the working directory as `vocabulary.csv`, **or** set **`FIXED_VOCAB_PATH`** to the absolute path of `vocabulary.csv`. The script also checks `vocab/vocabulary.csv` when run from the repo root.

Keep generated tokenized datasets in a separate data directory. This directory holds the fixed vocabulary used by the training scripts.
