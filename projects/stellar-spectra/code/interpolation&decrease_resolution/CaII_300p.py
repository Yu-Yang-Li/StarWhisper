import os
import re
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d
import csv


# 参数设置
input_base = "/data1/R10000FITS"
output_base = "/data1/R1800FITS_300"

snr_list = [1, 2]
metallicities = ["Z-{:.1f}".format(i * 0.1) for i in range(10, 41)]  # Z-1.0 ~ Z-4.0

R_high = 10000
R_low = 1800
lambda_FWHM = 8580  # 参考波长（埃）

# 全局变量控制是否已经保存过 wavelength_range
WAVELENGTH_SAVED = False


# 创建输出目录
def create_output_dirs():
    for snr in snr_list:
        for z in metallicities:
            out_dir = os.path.join(output_base, f"SNR{snr}", z)
            os.makedirs(out_dir, exist_ok=True)


# 高斯卷积
def convolve_gaussian(flux, sigma):
    return gaussian_filter1d(flux, sigma=sigma)


# 计算噪声
def add_noise(signal, snr):
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / snr
    noise = np.random.randn(len(signal)) * np.sqrt(noise_power)
    return signal + noise


# 仅保存 noisy flux 到 CSV（不带 Index）
def save_to_csv(noisy_flux, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Noisy_Flux'])
        for flux in noisy_flux:
            writer.writerow([flux])


# 主函数
def process_spectrum(infile, outfile_prefix, snr):
    global WAVELENGTH_SAVED

    with fits.open(infile) as hdul:
        flux = hdul[0].data
        header = hdul[0].header

        # 获取参数
        teff = header.get('PHXTEFF', None)
        logg = header.get('PHXLOGG', None)
        m_h = header.get('PHXM_H', None)

        if None in [teff, logg, m_h]:
            print(f"跳过 {infile}: 缺少必要的头部信息")
            return

        # 波长计算
        crval1 = header.get('CRVAL1', 8.006368)
        cdelt1 = header.get('CDELT1', 1.0E-05)
        wavelength = np.exp(crval1 + cdelt1 * np.arange(len(flux)))

        # 卷积参数计算
        sigma_G = (lambda_FWHM / 2.355) * np.sqrt((1 / R_low ** 2) - (1 / R_high ** 2))

        # 对整个光谱进行高斯卷积（降分辨率）
        convolved_flux = convolve_gaussian(flux, sigma=sigma_G)

        # 截取 8450~8710 埃范围
        idx_range = (wavelength >= 8450) & (wavelength <= 8710)
        flux_range = convolved_flux[idx_range]
        wavelength_range = wavelength[idx_range]

        # ===========================================
        # 🔽 降采样：每10个点中取第5个点（索引为4, 14, 24, ...）
        # 即保留索引满足 idx % 10 == 4 的点
        # ===========================================
        sample_indices = np.arange(4, len(flux_range), 10)

        # 采样 flux 和 wavelength
        flux_sampled = flux_range[sample_indices]
        wavelength_sampled = wavelength_range[sample_indices]  # ← 这是我们要保存的波长

        # 在采样后的 flux 上添加噪声
        noisy_flux_sampled = add_noise(flux_sampled, snr)

        # ✅ 新增：仅在首次运行时保存【降采样后的】wavelength_sampled
        if not WAVELENGTH_SAVED:
            np.savetxt(os.path.join(output_base, "wavelength.dat"), wavelength_sampled)
            print(f"已保存降采样后的 wavelength.dat 到 {output_base}")
            WAVELENGTH_SAVED = True

        # 保存到 CSV（只保留 noisy flux）
        save_to_csv(noisy_flux_sampled, outfile_prefix + ".csv")

        print(f"已生成: {outfile_prefix}.csv")


# 主流程
def main():
    create_output_dirs()

    for metallicity in metallicities:
        input_dir = os.path.join(input_base, metallicity)
        if not os.path.exists(input_dir):
            continue

        for root, _, files in os.walk(input_dir):
            for file in files:
                if "PHOENIX" in file and "HiRes" in file and file.endswith(".fits"):
                    match = re.search(r'lte(\d+)-(\d+\.\d+)-(\d+\.\d+)', file)
                    if not match:
                        continue

                    # 提取 Teff, logg, [M/H] 并构建新文件名（去掉 lte）
                    teff, logg, mh = match.groups()
                    name_part = f"{teff}-{logg}-{mh}"

                    infile = os.path.join(root, file)

                    for snr in snr_list:
                        out_dir = os.path.join(output_base, f"SNR{snr}", metallicity)
                        outfile_prefix = os.path.join(out_dir, name_part)

                        process_spectrum(infile, outfile_prefix, snr)


if __name__ == "__main__":
    main()