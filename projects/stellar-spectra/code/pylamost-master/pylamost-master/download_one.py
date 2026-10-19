import gzip
import os
import sys


import numpy as np
import pandas as pd
from astropy.io import fits
from pylamost import lamost

obsid = 577702065

# =================== 初始化 LAMOST API ===================
lm = lamost(token="lLY1albuoI", dr_version="dr12", sub_version="v1.0", is_dev=False)

# =================== 处理单个 OBSID 的函数 ===================
def process_obsid(obsid):
        output_csv = os.path.join(f"/data1/SpecTrain/lamost/pylamost-master/pylamost-master/", f"{obsid}.csv")
        # --- 1. 下载 .fits.gz ---
        gz_path = lm.download_fits(obsid=obsid, savedir=f"/data1/SpecTrain/lamost/pylamost-master/pylamost-master", overwrite=False)
        with open(gz_path, 'rb') as f:
            gz_data = f.read()
        with gzip.GzipFile(fileobj=BytesIO(gz_data)) as gz_file:
            with fits.open(gz_file, memmap=False) as hdul:
                table = hdul[1].data
                flux = np.array(table['FLUX'][0])
                wavelength = np.array(table['WAVELENGTH'][0])

        sort_idx = np.argsort(wavelength)
        wavelength = wavelength[sort_idx]
        flux = flux[sort_idx]

        # --- 4. 归一化 ---
        f_min, f_max = flux.min(), flux.max()
        normalized_flux = (
            (flux - f_min) / (f_max - f_min)
            if f_max != f_min else np.zeros_like(interpolated_flux)
        )

        # --- 5. 保存 CSV ---
        pd.DataFrame({"flux": normalized_flux}).to_csv(output_csv, index=False)

        # --- 6. 清理 .gz 文件 ---
        if os.path.exists(gz_path):
            os.remove(gz_path)
            tqdm.write(f"[{obsid}] 🗑️ .fits.gz 已删除")

        return True


process_obsid(obsid)


