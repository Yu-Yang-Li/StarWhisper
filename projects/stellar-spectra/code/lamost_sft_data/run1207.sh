#!/usr/bin/env bash
set -euo pipefail
: "${LAMOST_DATA_ROOT:?Set LAMOST_DATA_ROOT to the directory containing your spectra}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "${SCRIPT_DIR}/lamost_flux_preprocessing.py" \
  --data_dir "${LAMOST_DATA_ROOT}/LRS_SNR_0_3_303" \
  --output "${LAMOST_DATA_ROOT}/lamost_flux_tokenized_full.csv" \
  --overwrite \
  --processes 24 --imap_chunksize 64 --batch_files 400 \
  --max_pixels 303 \
  --pixel_idx_start 4 --pixel_idx_step 10
