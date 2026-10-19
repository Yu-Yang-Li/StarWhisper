python lamost_flux_preprocessing.py \
  --data_dir "/data1/SpecTrain/lamost/LRS_SNR_0_3_303" \
  --output  "/data1/SpecTrain/lamost/lamost_flux_tokenized_full.csv" \
  --overwrite \
  --processes 24 --imap_chunksize 64 --batch_files 400 \
  --max_pixels 303 \
  --pixel_idx_start 4 --pixel_idx_step 10