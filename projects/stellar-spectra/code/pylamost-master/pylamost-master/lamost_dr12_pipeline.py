import gzip
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from io import BytesIO
from threading import Lock

import numpy as np
import pandas as pd
from astropy.io import fits
from pylamost import lamost
from scipy.interpolate import interp1d
from tqdm import tqdm


# =================== 配置路径 ===================
CSV_FILE = r"/data1/SpecTrain/lamost/DR12LRS_SNRz0_3.csv"
TEMP_DIR = r"/data1/SpecTrain/lamost/temp_downloads"  # 存放 .fits.gz
WORK_DIR = r"/data1/SpecTrain/lamost/LRS_SNR_0_3_303"  # 存放 .csv
WAVELENGTH_DAT = r"/data1/R1800FITS_300/wavelength.dat"

# --- 自动创建目录 ---
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(WORK_DIR, exist_ok=True)
LOG_DIR = r"/data1/SpecTrain/lamost/logs"
os.makedirs(LOG_DIR, exist_ok=True)

# 日志文件名带时间戳
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(LOG_DIR, f"processing_{timestamp}.log")

# 重定向 stdout 和 stderr 到日志文件
sys.stdout = open(log_path, "w", encoding="utf-8")
sys.stderr = sys.stdout

print(f"🎯 LAMOST 光谱批处理开始 | 时间: {timestamp}")
print(f"📁 临时下载目录: {TEMP_DIR}")
print(f"📁 输出工作目录: {WORK_DIR}")
print(f"📊 目标波长文件: {WAVELENGTH_DAT}")
print(f"📝 日志文件: {log_path}")


# =================== 初始化 LAMOST API ===================
lm = lamost(token="lLY1albuoI", dr_version="dr12", sub_version="v1.0", is_dev=False)

# =================== 读取目标波长格点 ===================
try:
    target_wave = np.loadtxt(WAVELENGTH_DAT)
    print(f"✅ 成功加载 wavelength.dat，共 {len(target_wave)} 个波长点")
    print(f"   波长范围: [{target_wave.min():.1f}, {target_wave.max():.1f}] nm")
except Exception as e:
    print(f"❌ 无法读取 {WAVELENGTH_DAT}：{e}")
    raise

# =================== 读取 obsid 列表 ===================
try:
    df = pd.read_csv(CSV_FILE)
    if "obsid" not in df.columns:
        raise ValueError(f"CSV 文件中没有 'obsid' 列！列名为：{df.columns.tolist()}")
    obsid_list = df["obsid"].dropna().astype(int).tolist()
    print(f"✅ 成功读取 {len(obsid_list)} 个 OBSID")
except Exception as e:
    print(f"❌ 读取 CSV 文件失败：{e}")
    raise


# =================== 分批处理配置 ===================
BATCH_SIZE = 200
MAX_WORKERS = 10  # I/O 密集型任务，不宜过高
print(f"🔧 设置：每批 {BATCH_SIZE} 个，最大并发 {MAX_WORKERS} 线程")


# =================== 处理单个 OBSID 的函数 ===================
def process_obsid(obsid):
    output_csv = os.path.join(WORK_DIR, f"{obsid}.csv")
    if os.path.exists(output_csv):
        tqdm.write(f"[{obsid}] ✅ 已存在，跳过")
        return True

    gz_path = None
    try:
        # --- 1. 下载 .fits.gz ---
        gz_path = lm.download_fits(obsid=obsid, savedir=TEMP_DIR, overwrite=False)
        tqdm.write(f"[{obsid}] 📥 下载完成 → {os.path.basename(gz_path)}")

        # --- 2. 内存解压 + 读取 FITS ---
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
        tqdm.write(f"[{obsid}] 🔧 数据读取 → {len(flux)} 点 | λ范围: [{wavelength[0]:.1f}, {wavelength[-1]:.1f}] nm")

        # --- 3. 插值 ---
        valid_target = target_wave[
            (target_wave >= wavelength.min()) & (target_wave <= wavelength.max())
        ]
        if len(valid_target) == 0:
            tqdm.write(f"[{obsid}] ❌ 目标波长与光谱无交集")
            return False

        f_interp = interp1d(wavelength, flux, kind="linear", fill_value=np.nan, bounds_error=False)
        interpolated_flux = f_interp(valid_target)
        valid_mask = ~np.isnan(interpolated_flux)
        interpolated_flux = interpolated_flux[valid_mask]
        tqdm.write(f"[{obsid}] 📏 插值完成 → {len(interpolated_flux)} 个有效点")

        # --- 4. 归一化 ---
        f_min, f_max = interpolated_flux.min(), interpolated_flux.max()
        normalized_flux = (
            (interpolated_flux - f_min) / (f_max - f_min)
            if f_max != f_min else np.zeros_like(interpolated_flux)
        )
        tqdm.write(f"[{obsid}] 📊 归一化完成 → min={f_min:.2f}, max={f_max:.2f}")

        # --- 5. 保存 CSV ---
        pd.DataFrame({"flux": normalized_flux}).to_csv(output_csv, index=False)
        tqdm.write(f"[{obsid}] 💾 保存成功 → {output_csv}")

        # --- 6. 清理 .gz 文件 ---
        if os.path.exists(gz_path):
            os.remove(gz_path)
            tqdm.write(f"[{obsid}] 🗑️ .fits.gz 已删除")

        return True

    except Exception as e:
        tqdm.write(f"[{obsid}] ❌ 处理失败: {e}")
        if gz_path and os.path.exists(gz_path):
            try:
                os.remove(gz_path)
            except:
                pass
        return False


# =================== 分批并行处理 ===================
print(f"🚀 开始分批处理，每批 {BATCH_SIZE} 个...")

total_success = 0
total_processed = 0

for i in range(0, len(obsid_list), BATCH_SIZE):
    batch = obsid_list[i:i + BATCH_SIZE]
    print(f"\n🔄 处理第 {i//BATCH_SIZE + 1} 批: {len(batch)} 个文件 (OBSID {batch[0]} ~ {batch[-1]})")
    start_time = datetime.now()

    success_count = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_obsid, obsid) for obsid in batch]
        for future in tqdm(
            as_completed(futures),
            total=len(batch),
            desc="📦 本批进度",
            unit="file",
            ncols=100,
            colour="green"
        ):
            if future.result():
                success_count += 1

    batch_time = (datetime.now() - start_time).total_seconds()
    print(f"✅ 本批完成 | 成功: {success_count}/{len(batch)} | 耗时: {batch_time:.1f}s | 平均: {batch_time/len(batch):.2f}s/条")

    total_success += success_count
    total_processed += len(batch)

    # 可选：每批后暂停，缓解系统压力
    # time.sleep(2)

# =================== 清理临时目录（应已空）===================
if os.path.exists(TEMP_DIR):
    try:
        shutil.rmtree(TEMP_DIR)
        print(f"✅ 临时下载目录已清理: {TEMP_DIR}")
    except Exception as e:
        print(f"⚠️ 无法删除临时目录: {e}")

# =================== 最终总结 ===================
print(f"\n{'='*60}")
print("🎉 批处理全部完成！")
print(f"📊 总计处理: {total_processed} 个")
print(f"✅ 成功: {total_success} 个")
print(f"❌ 失败: {total_processed - total_success} 个")
print(f"📈 成功率: {total_success / total_processed * 100:.1f}%")
print(f"📁 结果保存在: {WORK_DIR}")
print(f"📄 日志文件: {log_path}")
print(f"🔚 处理结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 关闭日志
sys.stdout.close()