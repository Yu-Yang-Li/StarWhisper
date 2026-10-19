#!/usr/bin/env bash
set -euo pipefail
: "${LAMOST_DATA_ROOT:?Set LAMOST_DATA_ROOT to the directory containing your spectra and catalogue}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "${SCRIPT_DIR}/lamost_flux_preprocessing.py" \
  --data_dir "${LAMOST_DATA_ROOT}/LRS_SNR_3_11_303_augmented_spectra" \
  --output "${LAMOST_DATA_ROOT}/data_augment/lamost_flux_tokenized_full.csv" \
  --overwrite \
  --processes 24 --imap_chunksize 64 --batch_files 400 \
  --max_pixels 303 \
  --pixel_idx_start 4 --pixel_idx_step 10

python "${SCRIPT_DIR}/convert_lamost_to_pretrain_format.py" \
  --input "${LAMOST_DATA_ROOT}/data_augment/lamost_flux_tokenized_full.csv" \
  --catalog "${LAMOST_DATA_ROOT}/LRS_SNR_3_11_303_augmented_data.csv" \
  --output "${LAMOST_DATA_ROOT}/data_augment/lamost_flux_tokenized_full_temp.csv" \
  --output_dir "${LAMOST_DATA_ROOT}/data_augment" \
  --test_size 0.1 --random_seed 42 --chunksize 500000
