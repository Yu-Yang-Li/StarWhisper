"""
LAMOST DR12 整条光谱导出（不插值），仅处理前 50 条。
每条光谱画成一张图并保存为 PNG（波长 vs 归一化 flux），不保存 CSV。
"""
import gzip
import os
import sys
from datetime import datetime
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from pylamost import lamost
from tqdm import tqdm


# =================== 配置路径 ===================
CSV_FILE = r"/data1/SpecTrain/lamost/DR12LRS_SNR5_7_FEHleq0.csv"
TEMP_DIR = r"/data1/SpecTrain/lamost/temp_downloads"
# 整条光谱图 PNG 输出目录，仅 50 条
WORK_DIR = r"/data1/SpecTrain/lamost/LRS_full_spectrum_57_50"
LOG_DIR = r"/data1/SpecTrain/lamost/logs"

MAX_OBS = 500  # 只处理前 50 个

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(LOG_DIR, f"full_spectrum_50_{timestamp}.log")
sys.stdout = open(log_path, "w", encoding="utf-8")
sys.stderr = sys.stdout

print(f"LAMOST 整条光谱出图（50 条 PNG）| 时间: {timestamp}")
print(f"临时下载目录: {TEMP_DIR}")
print(f"输出目录: {WORK_DIR}")
print(f"日志: {log_path}")


# =================== 初始化 LAMOST API ===================
lm = lamost(
    token="lLY1albuoI", dr_version="dr12", sub_version="v1.0", is_dev=False
)

# =================== 读取 obsid 列表，只取前 50 个 ===================
try:
    df = pd.read_csv(CSV_FILE)
    if "obsid" not in df.columns:
        raise ValueError("CSV 中无 obsid 列")
    obsid_list = df["obsid"].dropna().astype(int).tolist()[:MAX_OBS]
    print(f"已读取 {len(obsid_list)} 个 OBSID（仅前 {MAX_OBS} 个）")
except Exception as e:
    print(f"读取 CSV 失败: {e}")
    raise


def process_obsid(obsid):
    output_png = os.path.join(WORK_DIR, f"{obsid}.png")
    if os.path.exists(output_png):
        tqdm.write(f"[{obsid}] 图已存在，跳过")
        return True

    gz_path = None
    try:
        gz_path = lm.download_fits(obsid=obsid, savedir=TEMP_DIR, overwrite=False)
        tqdm.write(f"[{obsid}] 下载完成 -> {os.path.basename(gz_path)}")

        with open(gz_path, "rb") as f:
            gz_data = f.read()
        with gzip.GzipFile(fileobj=BytesIO(gz_data)) as gz_file:
            with fits.open(gz_file, memmap=False) as hdul:
                table = hdul[1].data
                flux = np.array(table["FLUX"][0])
                wavelength = np.array(table["WAVELENGTH"][0])

        sort_idx = np.argsort(wavelength)
        wavelength = wavelength[sort_idx]
        flux = flux[sort_idx]

        f_min, f_max = flux.min(), flux.max()
        normalized_flux = (
            (flux - f_min) / (f_max - f_min)
            if f_max != f_min
            else np.zeros_like(flux)
        )

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(wavelength, normalized_flux, linewidth=0.5, color="C0")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Normalized flux")
        ax.set_title(f"OBSID {obsid}  ({len(flux)} points)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_png, dpi=150)
        plt.close(fig)
        tqdm.write(f"[{obsid}] 已保存图 -> {os.path.basename(output_png)}")

        if os.path.exists(gz_path):
            os.remove(gz_path)
        return True
    except Exception as e:
        tqdm.write(f"[{obsid}] 失败: {e}")
        if gz_path and os.path.exists(gz_path):
            try:
                os.remove(gz_path)
            except Exception:
                pass
        return False


# =================== 顺序处理 50 条 ===================
success = 0
for obsid in tqdm(obsid_list, desc="处理进度", unit="条"):
    if process_obsid(obsid):
        success += 1

print(f"\n完成: 成功 {success}/{len(obsid_list)}")
print(f"PNG 图目录: {WORK_DIR}")
print(f"日志: {log_path}")
sys.stdout.close()
