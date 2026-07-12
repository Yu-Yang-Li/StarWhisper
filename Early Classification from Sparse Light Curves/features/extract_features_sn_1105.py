#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SN 数据特征提取脚本（基于 1105 版特征集）

针对 E:\\ZTF_variables\\SN 文件夹中的 ZTF 和 Atlas SN 采样数据，
使用与 extract_features_1105.py 相同的特征提取方法。

输出文件名：
- train2_SN_YYYYmmdd_HHMMSS.csv
- test2_SN_YYYYmmdd_HHMMSS.csv
- train2_SN_YYYYmmdd_HHMMSS_balanced.csv
- test2_SN_YYYYmmdd_HHMMSS_balanced.csv

说明：
- SN 文件夹中的文件无 category 子文件夹，需从文件名推断类型
- 文件名格式：ZTF_{name}_n{3-30}.csv 或 Atlas_{name}_n{3-30}.csv
- category 统一标记为 "SN"（或根据文件名前缀区分）
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from scipy import stats
from sklearn.model_selection import train_test_split

BASE_DIR = Path("/root/shared-nvme")
SN_DIR = BASE_DIR / "SN"
OUTPUT_DIR = BASE_DIR

TEST_SIZE = 0.1
RANDOM_STATE = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            BASE_DIR / "features/extract_features_sn_1105.log",
            encoding="utf-8",
        ),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def safe_divide(a, b, default=0.0):
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.divide(a, b)
        if np.isnan(result) or np.isinf(result):
            return default
        return result


def calculate_iqr(mag):
    if len(mag) < 4:
        return 0.0
    q75, q25 = np.percentile(mag, [75, 25])
    return q75 - q25


def calculate_mad(mag):
    """MAD (Median Absolute Deviation)"""
    if len(mag) < 2:
        return 0.0
    median = np.median(mag)
    return float(np.median(np.abs(mag - median)))


def calculate_beyond_std(mag, k=1):
    """超出 k 倍标准差的点占比"""
    if len(mag) < 2:
        return 0.0
    mean_mag = np.mean(mag)
    std_mag = np.std(mag)
    if std_mag == 0:
        return 0.0
    return float(np.mean(np.abs(mag - mean_mag) > k * std_mag))


def calculate_percent_amplitude(mag):
    """不对称振幅"""
    if len(mag) < 2:
        return 0.0
    median = np.median(mag)
    max_mag = np.max(mag)
    min_mag = np.min(mag)
    if median == 0:
        return 0.0
    upper = abs((max_mag - median) / median)
    lower = abs((median - min_mag) / median)
    return float(max(upper, lower))


def calculate_autocorr_lag1(mag):
    """Lag-1 自相关"""
    if len(mag) < 2:
        return 0.0
    try:
        corr_matrix = np.corrcoef(mag[:-1], mag[1:])
        return float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0
    except Exception:
        return 0.0


def calculate_longest_monotonic(mag):
    """最长连续上升/下降段长度（归一化）"""
    if len(mag) < 2:
        return 0.0, 0.0
    diffs = np.diff(mag)
    n = len(mag)

    up_max = 0
    down_max = 0
    curr_up = 0
    curr_down = 0

    for d in diffs:
        if d > 0:
            curr_up += 1
            curr_down = 0
        elif d < 0:
            curr_down += 1
            curr_up = 0
        else:
            curr_up = 0
            curr_down = 0
        up_max = max(up_max, curr_up)
        down_max = max(down_max, curr_down)

    return up_max / n, down_max / n


def calculate_quantile_features(mag):
    """分位数特征"""
    if len(mag) < 4:
        return 0.0, 0.0, 0.0, 0.0
    q10, q50, q90 = np.percentile(mag, [10, 50, 90])
    q90_minus_q10 = q90 - q10
    if q50 - q10 != 0:
        q_shape = (q90 - q50) / (q50 - q10)
    else:
        q_shape = 0.0
    return q10, q90, q90_minus_q10, q_shape


def calculate_gini(mag):
    """Gini 指数（经非负化处理）"""
    if len(mag) < 2:
        return 0.0
    # 非负化：shift to non-negative
    mag_shifted = mag - mag.min() + 1e-8
    sorted_mag = np.sort(mag_shifted)
    n = len(sorted_mag)
    index = np.arange(1, n + 1)
    return float(
        (2 * np.sum(index * sorted_mag)) / (n * np.sum(sorted_mag)) - (n + 1) / n
    )


def calculate_lomb_scargle_features(time, mag):
    """Lomb-Scargle 频域特征"""
    if len(time) < 5:
        return {
            "ls_power_to_median": 0.0,
            "ls_entropy": 0.0,
            "ls_harmonic_ratio": 0.0,
            "ls_significant_peaks": 0,
            "ls_peak_width": 0.0,
            "ls_max_power": 0.0,
            "ls_median_power": 0.0,
            "ls_power_variance": 0.0,
            "ls_dominant_period": 0.0,
            "ls_fap": 1.0,
        }

    try:
        ls = LombScargle(time, mag)
        freq, power = ls.autopower(minimum_frequency=0.01, maximum_frequency=10.0)

        if len(power) == 0:
            return {
                "ls_power_to_median": 0.0,
                "ls_entropy": 0.0,
                "ls_harmonic_ratio": 0.0,
                "ls_significant_peaks": 0,
                "ls_peak_width": 0.0,
                "ls_max_power": 0.0,
                "ls_median_power": 0.0,
                "ls_power_variance": 0.0,
                "ls_dominant_period": 0.0,
                "ls_fap": 1.0,
            }

        max_power = float(np.max(power))
        median_power = float(np.median(power))
        power_variance = float(np.var(power))
        power_to_median = safe_divide(max_power, median_power, 0.0)

        # Entropy
        power_norm = power / (power.sum() + 1e-10)
        entropy = -np.sum(power_norm * np.log(power_norm + 1e-10))

        # Harmonic ratio
        if len(power) > 2:
            sorted_power = np.sort(power)[::-1]
            harmonic_ratio = safe_divide(sorted_power[1], sorted_power[0], 0.0)
        else:
            harmonic_ratio = 0.0

        # Significant peaks
        threshold = median_power * 2
        sig_peaks = int(np.sum(power > threshold))

        # Peak width
        max_idx = np.argmax(power)
        half_max = max_power / 2
        above_half = power > half_max
        peak_width = float(np.sum(above_half)) / len(power) if len(power) > 0 else 0.0

        # Dominant period
        dominant_period = 1.0 / freq[max_idx] if freq[max_idx] > 0 else 0.0

        # False alarm probability
        try:
            fap = ls.false_alarm_probability(max_power)
            fap = float(fap) if not np.isnan(fap) else 1.0
        except Exception:
            fap = 1.0

        return {
            "ls_power_to_median": power_to_median,
            "ls_entropy": float(entropy),
            "ls_harmonic_ratio": harmonic_ratio,
            "ls_significant_peaks": sig_peaks,
            "ls_peak_width": peak_width,
            "ls_max_power": max_power,
            "ls_median_power": median_power,
            "ls_power_variance": power_variance,
            "ls_dominant_period": dominant_period,
            "ls_fap": fap,
        }
    except Exception as e:
        logger.debug(f"Lomb-Scargle 计算失败: {e}")
        return {
            "ls_power_to_median": 0.0,
            "ls_entropy": 0.0,
            "ls_harmonic_ratio": 0.0,
            "ls_significant_peaks": 0,
            "ls_peak_width": 0.0,
            "ls_max_power": 0.0,
            "ls_median_power": 0.0,
            "ls_power_variance": 0.0,
            "ls_dominant_period": 0.0,
            "ls_fap": 1.0,
        }


def calculate_weighted_stats(mag, magerr):
    """加权统计（使用误差作为权重）"""
    if len(mag) < 2:
        return 0.0, 0.0
    if magerr is None or len(magerr) == 0:
        return float(np.mean(mag)), float(np.std(mag))

    weights = 1.0 / (magerr**2 + 1e-8)
    w_mean = float(np.average(mag, weights=weights))
    w_var = float(np.average((mag - w_mean) ** 2, weights=weights))
    w_std = np.sqrt(w_var)
    return w_mean, w_std


def calculate_chi2_dof(mag, magerr):
    """Chi-square per degree of freedom"""
    if len(mag) < 2:
        return 0.0
    if magerr is None or len(magerr) == 0:
        return 0.0

    mean_mag = np.mean(mag)
    chi2 = np.sum(((mag - mean_mag) / (magerr + 1e-8)) ** 2)
    dof = len(mag) - 1
    return float(chi2 / dof) if dof > 0 else 0.0


def calculate_median_snr(mag, magerr):
    """中位数信噪比"""
    if len(mag) < 1:
        return 0.0
    if magerr is None or len(magerr) == 0:
        return 0.0
    median_mag = np.median(mag)
    snr = np.abs(mag - median_mag) / (magerr + 1e-8)
    return float(np.median(snr))


def calculate_time_gap_stats(time):
    """时间间隔统计"""
    if len(time) < 2:
        return 0.0, 0.0, 0.0
    sorted_time = np.sort(time)
    gaps = np.diff(sorted_time)
    return float(np.max(gaps)), float(np.std(gaps)), float(np.median(gaps))


def infer_category_from_filename(filename: str) -> str:
    """从文件名推断类别（统一为 SN）"""
    return "SN"


def extract_features_from_file(csv_file: Path) -> Optional[dict]:
    """从单个 CSV 提取特征"""
    try:
        df = pd.read_csv(csv_file)

        # 查找必需列（SN 数据只有 mjd 和 mag）
        mjd_col = None
        mag_col = None
        magerr_col = None

        for col in df.columns:
            col_lower = col.lower()
            if col_lower == "mjd":
                mjd_col = col
            elif col_lower in ["mag", "magnitude"]:
                mag_col = col
            elif col_lower in ["magerr", "mag_error", "e_mag"]:
                magerr_col = col

        if mjd_col is None or mag_col is None:
            logger.warning("缺少 mjd 或 mag 列: %s", csv_file)
            return None

        # 清洗数据
        df[mjd_col] = pd.to_numeric(df[mjd_col], errors="coerce")
        df[mag_col] = pd.to_numeric(df[mag_col], errors="coerce")
        df = df.dropna(subset=[mjd_col, mag_col])

        if len(df) < 3:
            return None

        mjd = df[mjd_col].values
        mag = df[mag_col].values

        # 误差列（可选）
        magerr = None
        if magerr_col is not None:
            df[magerr_col] = pd.to_numeric(df[magerr_col], errors="coerce")
            magerr = df[magerr_col].fillna(0).values

        # 初始化特征字典
        features = {}

        # 基础统计
        features["num_points"] = len(mag)
        features["mean_mag"] = float(np.mean(mag))
        features["std_mag"] = float(np.std(mag))
        features["iqr_mag"] = calculate_iqr(mag)
        features["amplitude_ratio"] = safe_divide(
            np.max(mag) - np.min(mag), np.mean(mag)
        )
        features["skewness"] = float(stats.skew(mag)) if len(mag) > 2 else 0.0
        features["kurtosis"] = float(stats.kurtosis(mag)) if len(mag) > 3 else 0.0
        features["median_mag"] = float(np.median(mag))
        features["min_mag"] = float(np.min(mag))
        features["max_mag"] = float(np.max(mag))

        # 1105 新增：MAD, Beyond, Percent Amplitude, Autocorr
        features["mad_mag"] = calculate_mad(mag)
        features["beyond1std"] = calculate_beyond_std(mag, k=1)
        features["beyond2std"] = calculate_beyond_std(mag, k=2)
        features["percent_amplitude"] = calculate_percent_amplitude(mag)
        features["autocorr_lag1"] = calculate_autocorr_lag1(mag)

        # 最长单调段
        con_up, con_down = calculate_longest_monotonic(mag)
        features["con_up_max_norm"] = con_up
        features["con_down_max_norm"] = con_down

        # 分位数
        q10, q90, q90_q10, q_shape = calculate_quantile_features(mag)
        features["q10"] = q10
        features["q90"] = q90
        features["q90_minus_q10"] = q90_q10
        features["q_shape_ratio"] = q_shape

        # Gini
        features["gini_mag"] = calculate_gini(mag)

        # 时序特征
        mag_diff = np.diff(mag)
        features["zero_crossings"] = int(np.sum(mag_diff[:-1] * mag_diff[1:] < 0))
        features["frac_rising"] = safe_divide(np.sum(mag_diff > 0), len(mag_diff))
        features["mag_change_rate"] = safe_divide(np.sum(np.abs(mag_diff)), len(mag))

        # 时间跨度与密度
        time_span = float(np.max(mjd) - np.min(mjd))
        features["time_span"] = time_span
        features["density"] = safe_divide(len(mag), time_span)

        # Lomb-Scargle
        ls_features = calculate_lomb_scargle_features(mjd, mag)
        features.update(ls_features)

        # 变异性指标
        if magerr is not None and len(magerr) > 0:
            mean_err_sq = np.mean(magerr**2)
            variance = np.var(mag)
            features["excess_variance"] = safe_divide(
                variance - mean_err_sq, np.mean(mag) ** 2
            )

            # Stetson J
            n = len(mag)
            delta = (mag - np.mean(mag)) / (magerr + 1e-8)
            if n > 1:
                sgn = np.sign(delta[:-1] * delta[1:])
                stetson_val = np.sum(sgn * np.sqrt(np.abs(delta[:-1] * delta[1:])))
                features["stetson_j"] = float(stetson_val / (n - 1))
            else:
                features["stetson_j"] = 0.0
        else:
            features["excess_variance"] = 0.0
            features["stetson_j"] = 0.0

        # 加权统计
        if magerr is not None and len(magerr) > 0:
            w_mean, w_std = calculate_weighted_stats(mag, magerr)
            features["w_mean_mag"] = w_mean
            features["w_std_mag"] = w_std
            features["chi2_dof"] = calculate_chi2_dof(mag, magerr)
            features["median_snr"] = calculate_median_snr(mag, magerr)
        else:
            features["w_mean_mag"] = features["mean_mag"]
            features["w_std_mag"] = features["std_mag"]
            features["chi2_dof"] = 0.0
            features["median_snr"] = 0.0

        # 时间间隔统计
        max_gap, gap_std, gap_median = calculate_time_gap_stats(mjd)
        features["max_gap"] = max_gap
        features["gap_std"] = gap_std
        features["gap_median"] = gap_median

        # 上限占比（SN 数据无此信息）
        features["fraction_upper_limits"] = 0.0

        # 元信息（SN 数据无 ndethist）
        features["ndethist"] = 0

        # band_code（SN 数据不区分 band，统一设为 0）
        features["band_code"] = 0

        # 周期性判断
        features["is_periodic"] = 1 if ls_features["ls_fap"] < 0.01 else 0

        # 元信息
        features["file_path"] = str(csv_file.relative_to(BASE_DIR))
        features["category"] = infer_category_from_filename(csv_file.name)

        return features

    except Exception as e:
        logger.error(f"处理文件 {csv_file} 时出错: {e}")
        return None


def process_sn_directory():
    """处理 SN 目录下的所有文件"""
    if not SN_DIR.exists():
        logger.warning(f"目录不存在: {SN_DIR}")
        return []

    csv_files = list(SN_DIR.glob("*.csv"))
    if not csv_files:
        logger.warning(f"无CSV文件: {SN_DIR}")
        return []

    logger.info(f"处理 SN 目录 ({len(csv_files)} 个文件)")
    features_list = []
    for i, file in enumerate(csv_files, 1):
        if i % 500 == 0:
            logger.info(f"进度: {i} / {len(csv_files)}")
        feat = extract_features_from_file(file)
        if feat:
            features_list.append(feat)

    logger.info(f"✅ SN: 成功 {len(features_list)} / {len(csv_files)}")
    return features_list


def main():
    logger.info("=" * 80)
    logger.info("【SN 特征提取 - 1105 扩展版】")
    logger.info("目标：适用于 SN 采样数据（3~30 点），使用 1105 特征集")
    logger.info(f"输入目录: {SN_DIR}")
    logger.info(f"测试集比例: {TEST_SIZE}")
    logger.info("=" * 80)

    if not SN_DIR.exists():
        logger.error(f"SN 目录不存在: {SN_DIR}")
        return

    all_features = process_sn_directory()

    if not all_features:
        logger.error("未提取到任何有效特征，请检查数据路径和格式")
        return

    df = pd.DataFrame(all_features)

    # 列顺序（与 1105 版保持一致）
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
        "mad_mag",
        "beyond1std",
        "beyond2std",
        "percent_amplitude",
        "autocorr_lag1",
        "con_up_max_norm",
        "con_down_max_norm",
        "q10",
        "q90",
        "q90_minus_q10",
        "q_shape_ratio",
        "gini_mag",
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
        "ls_dominant_period",
        "ls_fap",
        "excess_variance",
        "stetson_j",
        "w_mean_mag",
        "w_std_mag",
        "chi2_dof",
        "median_snr",
        "max_gap",
        "gap_std",
        "gap_median",
        "fraction_upper_limits",
        "ndethist",
        "band_code",
        "is_periodic",
        "category",
    ]

    df = df[[c for c in feature_order if c in df.columns]]

    output_features_dir = OUTPUT_DIR / "features"
    output_features_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["category"],
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_out = output_features_dir / f"train2_SN_{timestamp}.csv"
    test_out = output_features_dir / f"test2_SN_{timestamp}.csv"

    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)

    logger.info(f"✅ 训练集已保存: {train_out}")
    logger.info(f"✅ 测试集已保存: {test_out}")
    logger.info(f"📊 总样本: {len(df)}, 训练: {len(train_df)}, 测试: {len(test_df)}")
    logger.info("\n📈 类别数量:")
    logger.info(df["category"].value_counts().sort_index().to_string())
    logger.info("\n✅ 特征提取完成！")


if __name__ == "__main__":
    main()
