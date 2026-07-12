#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZTF光变曲线特征提取脚本（最终完整版 - 推荐24特征）
专为 3~30 点稀疏观测设计，融合统计、时域、频域、物理与元数据特征

输出特征（共26列）：
基础统计 (9): num_points, mean_mag, std_mag, iqr_mag, amplitude_ratio,
              skewness, kurtosis, median_mag, min_mag, max_mag
时间结构 (5): zero_crossings, frac_rising, mag_change_rate, time_span, density
频域特征 (8): ls_power_to_median, ls_entropy, ls_harmonic_ratio,
              ls_significant_peaks, ls_peak_width, ls_max_power,
              ls_median_power, ls_power_variance, is_periodic
物理变异性 (2): excess_variance, stetson_j
元数据 (1): ndethist
元数据 (2): ndethist, band_code（g=1, r=2, z=3, 其他=4）
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from scipy import stats
from sklearn.model_selection import train_test_split

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("extract_features_final_complete.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# 配置参数
BASE_DIR = Path("/root/shared-nvme")
TRAIN_DIR = BASE_DIR / "train4"
OUTPUT_DIR = BASE_DIR

# 所有类别
CATEGORIES = [
    "BYDra",
    "CEP",
    "CEPII",
    "CV",
    "DSCT",
    "EA",
    "EW",
    "Mira",
    "RR",
    "RRc",
    "RSCVN",
    "SR",
]

TEST_SIZE = 0.1
RANDOM_STATE = 42


def safe_divide(a, b, default=0.0):
    """安全除法"""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.divide(a, b)
        if np.isnan(result) or np.isinf(result):
            return default
        return result


def calculate_iqr(mag):
    """四分位距"""
    if len(mag) < 4:
        return 0.0
    q75, q25 = np.percentile(mag, [75, 25])
    return q75 - q25


def calculate_skewness(mag):
    """偏度"""
    if len(mag) < 3:
        return 0.0
    return stats.skew(mag)


def calculate_kurtosis(mag):
    """峰度"""
    if len(mag) < 4:
        return 0.0
    return stats.kurtosis(mag)


def calculate_zero_crossings(time, mag):
    """跨越中位数的次数"""
    if len(mag) < 2:
        return 0
    mag_centered = mag - np.median(mag)
    signs = np.sign(mag_centered)
    signs[signs == 0] = 1  # 避免零值导致误判
    crossings = np.where(np.diff(signs))[0]
    return len(crossings)


def calculate_frac_rising(mag):
    """上升段占比：diff(mag) < 0 表示变亮"""
    if len(mag) < 2:
        return 0.0
    rising = np.sum(np.diff(mag) < 0)
    return rising / (len(mag) - 1)


def calculate_mag_change_rate(time, mag):
    """长期变化率：(last - first) / duration"""
    if len(time) < 2 or (max(time) - min(time)) == 0:
        return 0.0
    return (mag[-1] - mag[0]) / (time[-1] - time[0])


def calculate_density(time):
    """采样密度"""
    duration = np.max(time) - np.min(time)
    return safe_divide(len(time), duration) if duration > 0 else 0.0


def calculate_excess_variance(mag, magerr):
    """超出方差：判断是否为真实变源"""
    if len(mag) < 2 or np.any(magerr <= 0):
        return 0.0
    mean_mag = np.mean(mag)
    chi2 = np.sum(((mag - mean_mag) / magerr) ** 2)
    exc_var = (chi2 / len(mag)) - 1
    return max(exc_var, 0.0)


def stetson_j(mag, magerr):
    """Stetson J 指数：探测不规则变化"""
    if len(mag) < 2 or np.any(magerr <= 0):
        return 0.0
    delta = (mag - np.median(mag)) / magerr
    n_pairs = 0
    sum_product = 0.0
    for i in range(len(delta)):
        for j in range(i + 1, len(delta)):
            sum_product += np.sign(delta[i] * delta[j]) * np.sqrt(
                np.abs(delta[i] * delta[j])
            )
            n_pairs += 1
    return sum_product / n_pairs if n_pairs > 0 else 0.0


def analyze_frequency_domain(time, mag, magerr=None, min_period=0.1, max_period=100):
    """
    统一计算所有 Lomb-Scargle 衍生特征（7个）
    返回字典
    """
    try:
        # 过滤 NaN
        mask = ~np.isnan(mag)
        time = time[mask]
        mag = mag[mask]
        if magerr is not None:
            magerr = magerr[mask]
        else:
            magerr = np.ones_like(mag) * 0.05  # 默认误差

        if len(mag) < 5:
            return {
                f"ls_{k}": 0.0
                for k in [
                    "power_to_median",
                    "entropy",
                    "harmonic_ratio",
                    "significant_peaks",
                    "peak_width",
                    "max_power",
                    "median_power",
                    "power_variance",
                ]
            }

        duration = np.max(time) - np.min(time)
        max_p = min(max_period, duration * 2)
        min_freq = 1 / max_p
        max_freq = 1 / min_period

        ls = LombScargle(time, mag, dy=magerr)
        freq, power = ls.autopower(
            minimum_frequency=min_freq, maximum_frequency=max_freq
        )

        if len(power) == 0 or np.all(power <= 0):
            return dict.fromkeys(
                [
                    f"ls_{k}"
                    for k in [
                        "power_to_median",
                        "entropy",
                        "harmonic_ratio",
                        "significant_peaks",
                        "peak_width",
                        "max_power",
                        "median_power",
                        "power_variance",
                    ]
                ],
                0.0,
            )

        max_power = np.max(power)
        median_power = np.median(power)
        power_to_median = safe_divide(max_power, median_power)

        # 熵（分布集中度）
        prob = power / np.sum(power)
        ls_entropy = -np.sum(prob * np.log(prob + 1e-10))

        # 谐波比：次强峰 / 最强峰
        sorted_power = np.sort(power)[::-1]
        ls_harmonic_ratio = (
            safe_divide(sorted_power[1], sorted_power[0])
            if len(sorted_power) > 1
            else 0.0
        )

        # 显著峰数量（> median_power）
        ls_significant_peaks = np.sum(power > median_power)

        # 主峰宽度（半高全宽 FWHM）
        peak_idx = np.argmax(power)
        half_max = max_power / 2
        left = np.where(power[:peak_idx] < half_max)[0]
        right = np.where(power[peak_idx:] < half_max)[0]
        if len(left) > 0 and len(right) > 0:
            fwhm = freq[peak_idx + right[0]] - freq[peak_idx - left[-1]]
            ls_peak_width = 1 / fwhm if fwhm > 0 else 0.0
        else:
            ls_peak_width = 0.0

        # 其他统计
        ls_power_variance = np.var(power)

        return {
            "ls_power_to_median": power_to_median,
            "ls_entropy": ls_entropy,
            "ls_harmonic_ratio": ls_harmonic_ratio,
            "ls_significant_peaks": float(ls_significant_peaks),
            "ls_peak_width": ls_peak_width,
            "ls_max_power": max_power,
            "ls_median_power": median_power,
            "ls_power_variance": ls_power_variance,
        }

    except Exception as e:
        logger.debug(f"频域分析失败: {e}")
        return {
            f"ls_{k}": 0.0
            for k in [
                "power_to_median",
                "entropy",
                "harmonic_ratio",
                "significant_peaks",
                "peak_width",
                "max_power",
                "median_power",
                "power_variance",
            ]
        }


def is_periodic_suspected(ls_power_to_median, num_points):
    """经验判断周期性：max_power/median_power > log(N)/2"""
    if num_points < 5:
        return 0
    threshold = np.log(num_points) / 2.0
    return 1 if ls_power_to_median > threshold else 0


def infer_band_code(df: pd.DataFrame, csv_file: Path) -> int:
    """推断观测波段数值编码：g=1, r=2, z=3, 其他=4

    优先级：
    1) DataFrame 中的列：'filter' 或 'band'（字符串）
    2) DataFrame 中的列：'fid'（数值，1/2/3）
    3) 文件名包含: '_zg'/'zg' -> g, '_zr'/'zr' -> r, '_zz'/'zz' -> z
    4) 其他 -> 4
    """
    try:
        for col in ["filter", "band"]:
            if col in df.columns and df[col].notna().any():
                val = str(df[col].dropna().iloc[0]).lower()
                if "zg" in val or val == "g" or val.startswith("g"):
                    return 1
                if "zr" in val or val == "r" or val.startswith("r"):
                    return 2
                if "zz" in val or val == "z" or val.startswith("z"):
                    return 3
        if "fid" in df.columns and df["fid"].notna().any():
            fid = int(pd.to_numeric(df["fid"].dropna().iloc[0], errors="coerce"))
            if fid == 1:
                return 1
            if fid == 2:
                return 2
            if fid == 3:
                return 3
        name = str(csv_file.name).lower()
        if "_zg" in name or "zg" in name:
            return 1
        if "_zr" in name or "zr" in name:
            return 2
        if "_zz" in name or "zz" in name:
            return 3
    except Exception:
        pass
    return 4


def extract_features_from_file(csv_file, category):
    """从单个CSV文件提取特征"""
    try:
        df = pd.read_csv(csv_file)
        if "mjd" not in df.columns or "mag" not in df.columns:
            logger.warning(f"跳过 {csv_file}: 缺少 mjd 或 mag 列")
            return None

        time = df["mjd"].values
        mag = df["mag"].values

        # 检查是否有 magerr
        has_error = "magerr" in df.columns and df["magerr"].notna().any()
        magerr = df["magerr"].values if has_error else np.ones_like(mag) * 0.05

        if len(time) < 3 or len(mag) < 3:
            logger.warning(f"跳过 {csv_file}: 数据点太少 (<3)")
            return None

        features = {}

        # --- 基础统计 ---
        features["num_points"] = len(mag)
        features["mean_mag"] = np.mean(mag)
        features["std_mag"] = np.std(mag)
        features["iqr_mag"] = calculate_iqr(mag)
        features["amplitude_ratio"] = safe_divide(
            np.max(mag) - np.min(mag), np.mean(mag)
        )
        features["skewness"] = calculate_skewness(mag)
        features["kurtosis"] = calculate_kurtosis(mag)
        features["median_mag"] = np.median(mag)
        features["min_mag"] = np.min(mag)
        features["max_mag"] = np.max(mag)

        # --- 时间结构 ---
        features["zero_crossings"] = calculate_zero_crossings(time, mag)
        features["frac_rising"] = calculate_frac_rising(mag)
        features["mag_change_rate"] = calculate_mag_change_rate(time, mag)
        features["time_span"] = np.max(time) - np.min(time)
        features["density"] = calculate_density(time)

        # --- 频域特征 ---
        freq_features = analyze_frequency_domain(time, mag, magerr)
        features.update(freq_features)

        # --- 物理变异性 ---
        features["excess_variance"] = calculate_excess_variance(mag, magerr)
        features["stetson_j"] = stetson_j(mag, magerr)

        # --- 元数据 ---
        if "ndethist" in df.columns:
            features["ndethist"] = float(df["ndethist"].iloc[0])
        elif "Dethist" in df.columns:
            features["ndethist"] = float(df["Dethist"].iloc[0])
        else:
            features["ndethist"] = float(len(mag))  # 保守估计
        # 波段编码
        features["band_code"] = infer_band_code(df, csv_file)

        # --- 周期性判断 ---
        features["is_periodic"] = is_periodic_suspected(
            features["ls_power_to_median"], features["num_points"]
        )

        # --- 元信息 ---
        features["file_path"] = str(csv_file.relative_to(BASE_DIR))
        features["category"] = category

        return features

    except Exception as e:
        logger.error(f"处理文件 {csv_file} 时出错: {e}")
        return None


def process_category(category):
    """处理单个类别"""
    category_dir = TRAIN_DIR / category
    if not category_dir.exists():
        logger.warning(f"目录不存在: {category_dir}")
        return []

    csv_files = list(category_dir.glob("*.csv"))
    if not csv_files:
        logger.warning(f"无CSV文件: {category}")
        return []

    logger.info(f"处理类别 {category} ({len(csv_files)} 个文件)")

    features_list = []
    for file in csv_files:
        feat = extract_features_from_file(file, category)
        if feat:
            features_list.append(feat)

    logger.info(f"✅ {category}: 成功 {len(features_list)} / {len(csv_files)}")
    return features_list


def main():
    logger.info("=" * 80)
    logger.info("【ZTF 特征提取 - 最终完整版】")
    logger.info("目标：适用于 3~30 点稀疏数据，包含全部合理特征")
    logger.info(f"输入目录: {TRAIN_DIR}")
    logger.info(f"测试集比例: {TEST_SIZE}")
    logger.info("=" * 80)

    if not TRAIN_DIR.exists():
        logger.error(f"训练目录不存在: {TRAIN_DIR}")
        return

    all_features = []
    for cat in CATEGORIES:
        feats = process_category(cat)
        all_features.extend(feats)

    if not all_features:
        logger.error("未提取到任何有效特征，请检查数据路径和格式")
        return

    # 转DataFrame
    df = pd.DataFrame(all_features)

    # 固定列序（共26列）
    feature_order = [
        "file_path",
        "num_points",
        "mean_mag",
        "std_mag",
        "iqr_mag",
        "amplitude_ratio",
        "skewness",
        "kurtosis",
        "median_mag",
        "min_mag",
        "max_mag",
        "zero_crossings",
        "frac_rising",
        "mag_change_rate",
        "time_span",
        "density",
        "ls_power_to_median",
        "ls_entropy",
        "ls_harmonic_ratio",
        "ls_significant_peaks",
        "ls_peak_width",
        "ls_max_power",
        "ls_median_power",
        "ls_power_variance",
        "excess_variance",
        "stetson_j",
        "ndethist",
        "band_code",
        "is_periodic",
        "category",
    ]

    df = df[feature_order]

    # 创建输出目录
    output_features_dir = OUTPUT_DIR / "features"
    output_features_dir.mkdir(parents=True, exist_ok=True)

    # 划分数据集
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["category"],
    )

    # 保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_out = output_features_dir / f"train4_{timestamp}.csv"
    test_out = output_features_dir / f"test4_{timestamp}.csv"

    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)

    logger.info(f"✅ 训练集已保存: {train_out}")
    logger.info(f"✅ 测试集已保存: {test_out}")
    logger.info(f"📊 总样本: {len(df)}, 训练: {len(train_df)}, 测试: {len(test_df)}")

    # 类别分布
    logger.info("\n📈 各类别数量:")
    logger.info(df["category"].value_counts().sort_index().to_string())

    logger.info("\n✅ 特征提取完成！")


if __name__ == "__main__":
    main()
