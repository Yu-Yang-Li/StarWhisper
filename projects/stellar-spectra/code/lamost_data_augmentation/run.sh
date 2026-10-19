python /data1/SpecTrain/lamost/data_augment/lamost_flux_preprocessing.py \
  --data_dir "/data1/SpecTrain/lamost/LRS_SNR_3_11_303_augmented_spectra" \
  --output  "/data1/SpecTrain/lamost/data_augment/lamost_flux_tokenized_full.csv" \
  --overwrite \
  --processes 24 --imap_chunksize 64 --batch_files 400 \
  --max_pixels 303 \
  --pixel_idx_start 4 --pixel_idx_step 10 && \


python /data1/SpecTrain/lamost/data_augment/convert_lamost_to_pretrain_format.py \
  --input      "/data1/SpecTrain/lamost/data_augment/lamost_flux_tokenized_full.csv" \
  --catalog    "/data1/SpecTrain/lamost/LRS_SNR_3_11_303_augmented_data.csv" \
  --output     "/data1/SpecTrain/lamost/data_augment/lamost_flux_tokenized_full_temp.csv" \
  --output_dir "/data1/SpecTrain/lamost/data_augment" \
  --test_size 0.1 --random_seed 42 --chunksize 500000