#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZTF光变曲线特征提取脚本（train4 版 · 完整 57 特征）

基于 legacy/extract_features_full_varlen.py，从 train4 目录提取特征。

新增特征：
- 各类统计量的缺失指示：skewness_is_missing、kurtosis_is_missing
- 频域特征缺失指示：ls_entropy_is_missing、ls_fap_is_missing、ls_max_power_is_missing、
  ls_median_power_is_missing、ls_power_variance_is_missing
- SN数据从 train4/SN 目录读取（不参与4倍平衡）

输出文件名包含 1117 标识：
train4_1117_YYYYmmdd_HHMMSS.csv / test4_1117_YYYYmmdd_HHMMSS.csv
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("extract_features_train4.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


BASE_DIR = Path("E:/ZTF_variables")
TRAIN_DIR = BASE_DIR / "train4"
SN_DIR = TRAIN_DIR / "SN"
OUTPUT_DIR = BASE_DIR

# 平衡数据集参数（从 balance_dataset.py 集成）
MERGE_MAPPING = {
    "RR": "RR",
    "RRc": "RR",
    "EA": "Eclipsing",
    "EW": "Eclipsing",
    "Mira": "LPV",
    "SR": "LPV",
    "BYDra": "Active",
    "RSCVN": "Active",
    "CV": "Cataclysmic",
    "CEP": "Pulsating",
    "CEPII": "Pulsating",
    "DSCT": "Pulsating",
    "SN": "SN",
}
MAX_RATIO = 4.0

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


def calculate_skewness(mag):
    if len(mag) < 3:
        return 0.0
    result = stats.skew(mag)
    return 0.0 if np.isnan(result) or np.isinf(result) else float(result)


def calculate_kurtosis(mag):
    if len(mag) < 4:
        return 0.0
    result = stats.kurtosis(mag)
    return 0.0 if np.isnan(result) or np.isinf(result) else float(result)


def calculate_zero_crossings(time, mag):
    if len(mag) < 2:
        return 0
    mag_centered = mag - np.median(mag)
    signs = np.sign(mag_centered)
    signs[signs == 0] = 1
    crossings = np.where(np.diff(signs))[0]
    return len(crossings)


def calculate_frac_rising(mag):
    if len(mag) < 2:
        return 0.0
    rising = np.sum(np.diff(mag) < 0)
    return rising / (len(mag) - 1)


def calculate_mag_change_rate(time, mag):
    if len(time) < 2 or (max(time) - min(time)) == 0:
        return 0.0
    return (mag[-1] - mag[0]) / (time[-1] - time[0])


def calculate_density(time):
    duration = np.max(time) - np.min(time)
    return safe_divide(len(time), duration) if duration > 0 else 0.0


def calculate_excess_variance(mag, magerr):
    if len(mag) < 2 or np.any(magerr <= 0):
        return 0.0
    mean_mag = np.mean(mag)
    chi2 = np.sum(((mag - mean_mag) / magerr) ** 2)
    exc_var = (chi2 / len(mag)) - 1
    return max(exc_var, 0.0)


def stetson_j(mag, magerr):
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


def analyze_frequency_domain(
    time,
    mag,
    magerr=None,
    min_period=0.1,
    max_period=100,
):
    base_keys = [
        "power_to_median",
        "entropy",
        "harmonic_ratio",
        "significant_peaks",
        "peak_width",
        "max_power",
        "median_power",
        "power_variance",
    ]

    def _default_result():
        return {f"ls_{k}": 0.0 for k in base_keys} | {
            "ls_dominant_period": 0.0,
            "ls_fap": 1.0,
        }

    def _missing_flags(value: int = 1):
        return {
            "ls_entropy_is_missing": value,
            "ls_fap_is_missing": value,
            "ls_max_power_is_missing": value,
            "ls_median_power_is_missing": value,
            "ls_power_variance_is_missing": value,
        }

    missing_flags = _missing_flags(0)

    try:
        mask = ~np.isnan(mag)
        time = time[mask]
        mag = mag[mask]
        if magerr is not None:
            magerr = magerr[mask]
        else:
            magerr = np.ones_like(mag) * 0.05

        if len(mag) < 5:
            return _default_result(), _missing_flags()

        duration = np.max(time) - np.min(time)
        max_p = min(max_period, max(duration * 2, 0.1))
        min_freq = 1 / max_p
        max_freq = 1 / min_period

        ls = LombScargle(time, mag, dy=magerr)
        freq, power = ls.autopower(
            minimum_frequency=min_freq, maximum_frequency=max_freq
        )

        if len(power) == 0 or np.all(power <= 0):
            return _default_result(), _missing_flags()

        max_power = float(np.max(power))
        median_power = float(np.median(power))
        power_to_median = safe_divide(max_power, median_power)

        prob = power / np.sum(power)
        ls_entropy = float(-np.sum(prob * np.log(prob + 1e-10)))

        sorted_power = np.sort(power)[::-1]
        ls_harmonic_ratio = (
            safe_divide(float(sorted_power[1]), float(sorted_power[0]))
            if len(sorted_power) > 1
            else 0.0
        )

        ls_significant_peaks = float(np.sum(power > median_power))

        peak_idx = int(np.argmax(power))
        half_max = max_power / 2
        left = np.where(power[:peak_idx] < half_max)[0]
        right = np.where(power[peak_idx:] < half_max)[0]
        if len(left) > 0 and len(right) > 0:
            fwhm = freq[peak_idx + right[0]] - freq[peak_idx - left[-1]]
            ls_peak_width = 1 / fwhm if fwhm > 0 else 0.0
        else:
            ls_peak_width = 0.0

        ls_power_variance = float(np.var(power))

        # 新增：主周期 + FAP
        dominant_freq = float(freq[peak_idx]) if len(freq) else 0.0
        ls_dominant_period = 1.0 / dominant_freq if dominant_freq > 0 else 0.0
        try:
            ls_fap = float(ls.false_alarm_probability(max_power))
        except Exception:
            ls_fap = 1.0
            missing_flags["ls_fap_is_missing"] = 1

        return (
            {
                "ls_power_to_median": float(power_to_median),
                "ls_entropy": float(ls_entropy),
                "ls_harmonic_ratio": float(ls_harmonic_ratio),
                "ls_significant_peaks": float(ls_significant_peaks),
                "ls_peak_width": float(ls_peak_width),
                "ls_max_power": float(max_power),
                "ls_median_power": float(median_power),
                "ls_power_variance": float(ls_power_variance),
                "ls_dominant_period": float(ls_dominant_period),
                "ls_fap": float(ls_fap),
            },
            missing_flags,
        )

    except Exception as e:
        logger.debug(f"频域分析失败: {e}")
        return _default_result(), _missing_flags()


def is_periodic_suspected(ls_power_to_median, num_points):
    if num_points < 5:
        return 0
    threshold = np.log(num_points) / 2.0
    return 1 if ls_power_to_median > threshold else 0


def infer_band_code(df: pd.DataFrame, csv_file: Path) -> int:
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


# ---------------- 新增特征函数 ---------------- #


def mad(mag: np.ndarray) -> float:
    if len(mag) == 0:
        return 0.0
    med = np.median(mag)
    return float(np.median(np.abs(mag - med)))


def beyond_k_std(mag: np.ndarray, mean_mag: float, std_mag: float, k: float) -> float:
    if len(mag) == 0 or std_mag <= 0:
        return 0.0
    return float(np.mean(np.abs(mag - mean_mag) > k * std_mag))


def percent_amplitude(min_mag: float, median_mag: float, max_mag: float) -> float:
    if median_mag == 0:
        return 0.0
    a = safe_divide(max_mag - median_mag, median_mag, 0.0)
    b = safe_divide(median_mag - min_mag, median_mag, 0.0)
    return float(max(a, b))


def autocorr_lag1(mag: np.ndarray) -> float:
    if len(mag) < 2:
        return 0.0
    with np.errstate(all="ignore"):
        c = np.corrcoef(mag[:-1], mag[1:])
    v = c[0, 1]
    if np.isnan(v) or np.isinf(v):
        return 0.0
    return float(v)


def longest_monotonic_runs(mag: np.ndarray) -> tuple[float, float]:
    if len(mag) < 2:
        return 0.0, 0.0
    diffs = np.diff(mag)
    up = (diffs < 0).astype(int)
    down = (diffs > 0).astype(int)

    def _max_run(arr):
        if len(arr) == 0:
            return 0
        max_run = cur = arr[0]
        for x in arr[1:]:
            if x == 1 and cur >= 1:
                cur += 1
            else:
                cur = x
            max_run = max(max_run, cur)
        return int(max_run)

    up_len = _max_run(up)
    down_len = _max_run(down)
    denom = max(len(mag) - 1, 1)
    return float(up_len / denom), float(down_len / denom)


def quantile_features(mag: np.ndarray) -> dict:
    if len(mag) == 0:
        return {
            "q10": 0.0,
            "q90": 0.0,
            "q90_minus_q10": 0.0,
            "q_shape_ratio": 0.0,
        }
    q10 = float(np.percentile(mag, 10))
    q50 = float(np.percentile(mag, 50))
    q90 = float(np.percentile(mag, 90))
    q90_minus_q10 = float(q90 - q10)
    denom = max(q50 - q10, 1e-8)
    q_shape_ratio = float((q90 - q50) / denom)
    return {
        "q10": q10,
        "q90": q90,
        "q90_minus_q10": q90_minus_q10,
        "q_shape_ratio": q_shape_ratio,
    }


def gini_mag(mag: np.ndarray) -> float:
    if len(mag) == 0:
        return 0.0
    x = np.sort(mag)
    # 非负化，避免分母~0
    x = x - x.min() + 1e-8
    n = len(x)
    s = np.sum(x)
    if s <= 0:
        return 0.0
    idx = np.arange(1, n + 1)
    g = (2.0 * np.sum(idx * x) / (n * s)) - (n + 1) / n
    return float(g)


def weighted_mean_std(mag: np.ndarray, magerr: np.ndarray) -> tuple[float, float]:
    if len(mag) == 0 or len(mag) != len(magerr):
        return 0.0, 0.0
    w = 1.0 / (magerr**2 + 1e-8)
    wmean = float(np.average(mag, weights=w))
    wvar = float(np.average((mag - wmean) ** 2, weights=w))
    return wmean, float(np.sqrt(max(wvar, 0.0)))


def chi2_dof(mag: np.ndarray, magerr: np.ndarray, ref: float | None = None) -> float:
    n = len(mag)
    if n < 2 or np.any(magerr <= 0):
        return 0.0
    ref_val = np.median(mag) if ref is None else ref
    chi2 = float(np.sum(((mag - ref_val) / magerr) ** 2))
    return float(chi2 / (n - 1))


def median_snr(mag: np.ndarray, magerr: np.ndarray) -> float:
    if len(mag) == 0 or np.any(magerr <= 0):
        return 0.0
    med = np.median(mag)
    snr = np.abs(mag - med) / magerr
    return float(np.median(snr))


def rise_decay_time_ratio(time: np.ndarray, mag: np.ndarray) -> float:
    if len(time) < 2 or len(mag) < 2:
        return 0.0
    idx_peak = int(np.argmin(mag))  # 亮度峰值（更亮对应更小的mag）
    t0, t1 = float(time[0]), float(time[-1])
    tp = float(time[idx_peak])
    # total 时间未直接使用，仅作为健壮性检查时可能参考
    rise = max(tp - t0, 0.0)
    decay = max(t1 - tp, 1e-8)
    return float(rise / decay)


def flux_asym_mean(time: np.ndarray, mag: np.ndarray) -> float:
    if len(time) == 0:
        return 0.0
    t_mid = 0.5 * (float(np.min(time)) + float(np.max(time)))
    first = mag[time <= t_mid]
    second = mag[time > t_mid]
    if len(first) == 0 or len(second) == 0:
        return 0.0
    return float(np.mean(first) - np.mean(second))


def time_gap_stats(time: np.ndarray) -> tuple[float, float, float]:
    if len(time) < 2:
        return 0.0, 0.0, 0.0
    t = np.sort(time)
    gaps = np.diff(t)
    if len(gaps) == 0:
        return 0.0, 0.0, 0.0
    return float(np.max(gaps)), float(np.std(gaps)), float(np.median(gaps))


def fraction_upper_limits(df: pd.DataFrame) -> float:
    try:
        if "isdiffpos" in df.columns:
            vals = pd.to_numeric(df["isdiffpos"], errors="coerce")
            return float(np.mean(vals.fillna(1) <= 0))
        if "catflags" in df.columns:
            vals = pd.to_numeric(df["catflags"], errors="coerce").fillna(0)
            return float(np.mean(vals != 0))
    except Exception:
        pass
    return 0.0


def extract_features_from_file(csv_file: Path, category: str):
    try:
        df = pd.read_csv(csv_file)
        if "mjd" not in df.columns or "mag" not in df.columns:
            logger.warning(f"跳过 {csv_file}: 缺少 mjd 或 mag 列")
            return None

        time = df["mjd"].values
        mag = df["mag"].values
        has_error = "magerr" in df.columns and df["magerr"].notna().any()
        magerr = df["magerr"].values if has_error else np.ones_like(mag) * 0.05

        if len(time) < 3 or len(mag) < 3:
            logger.warning(f"跳过 {csv_file}: 数据点太少 (<3)")
            return None

        features: dict[str, float | int | str] = {}

        # 基础统计
        features["num_points"] = int(len(mag))
        mean_mag = float(np.mean(mag))
        std_mag = float(np.std(mag))
        median_mag = float(np.median(mag))
        min_mag = float(np.min(mag))
        max_mag = float(np.max(mag))
        features["mean_mag"] = mean_mag
        features["std_mag"] = std_mag
        features["iqr_mag"] = float(calculate_iqr(mag))
        features["amplitude_ratio"] = safe_divide(max_mag - min_mag, mean_mag)
        features["skewness"] = float(calculate_skewness(mag))
        features["skewness_is_missing"] = int(len(mag) < 3)
        features["kurtosis"] = float(calculate_kurtosis(mag))
        features["kurtosis_is_missing"] = int(len(mag) < 4)
        features["median_mag"] = median_mag
        features["min_mag"] = min_mag
        features["max_mag"] = max_mag

        # 时间结构
        features["zero_crossings"] = int(calculate_zero_crossings(time, mag))
        features["frac_rising"] = float(calculate_frac_rising(mag))
        features["mag_change_rate"] = float(calculate_mag_change_rate(time, mag))
        features["time_span"] = float(np.max(time) - np.min(time))
        features["density"] = float(calculate_density(time))

        # 频域
        freq_features, freq_missing_flags = analyze_frequency_domain(time, mag, magerr)
        features.update(freq_features)
        features.update(freq_missing_flags)

        # 物理变异性
        features["excess_variance"] = float(calculate_excess_variance(mag, magerr))
        features["stetson_j"] = float(stetson_j(mag, magerr))

        # 元数据
        if "ndethist" in df.columns:
            features["ndethist"] = float(df["ndethist"].iloc[0])
        elif "Dethist" in df.columns:
            features["ndethist"] = float(df["Dethist"].iloc[0])
        else:
            features["ndethist"] = float(len(mag))
        features["band_code"] = int(infer_band_code(df, csv_file))

        # 周期性判断
        features["is_periodic"] = int(
            is_periodic_suspected(
                features["ls_power_to_median"], features["num_points"]
            )
        )

        # 1105 新增特征
        features["mad_mag"] = float(mad(mag))
        features["beyond1std"] = float(beyond_k_std(mag, mean_mag, std_mag, 1.0))
        features["beyond2std"] = float(beyond_k_std(mag, mean_mag, std_mag, 2.0))
        features["percent_amplitude"] = float(
            percent_amplitude(min_mag, median_mag, max_mag)
        )
        features["autocorr_lag1"] = float(autocorr_lag1(mag))
        up_run, down_run = longest_monotonic_runs(mag)
        features["con_up_max_norm"] = float(up_run)
        features["con_down_max_norm"] = float(down_run)
        features.update(quantile_features(mag))
        features["gini_mag"] = float(gini_mag(mag))
        wmean, wstd = weighted_mean_std(mag, magerr)
        features["w_mean_mag"] = float(wmean)
        features["w_std_mag"] = float(wstd)
        features["chi2_dof"] = float(chi2_dof(mag, magerr, ref=wmean))
        features["median_snr"] = float(median_snr(mag, magerr))
        features["rise_decay_time_ratio"] = float(rise_decay_time_ratio(time, mag))
        features["flux_asym_mean"] = float(flux_asym_mean(time, mag))
        max_gap, gap_std, gap_median = time_gap_stats(time)
        features["max_gap"] = float(max_gap)
        features["gap_std"] = float(gap_std)
        features["gap_median"] = float(gap_median)
        features["fraction_upper_limits"] = float(fraction_upper_limits(df))

        # 元信息
        features["file_path"] = str(csv_file.relative_to(BASE_DIR))
        features["category"] = category

        return features

    except Exception as e:
        logger.error(f"处理文件 {csv_file} 时出错: {e}")
        return None


def process_category(category: str):
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


def calculate_lomb_scargle_features_sn(time, mag):
    """SN版本的Lomb-Scargle特征计算（参考 legacy/extract_features_full_supernova.py）"""
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
        }, {
            "ls_entropy_is_missing": 1,
            "ls_fap_is_missing": 1,
            "ls_max_power_is_missing": 1,
            "ls_median_power_is_missing": 1,
            "ls_power_variance_is_missing": 1,
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
            }, {
                "ls_entropy_is_missing": 1,
                "ls_fap_is_missing": 1,
                "ls_max_power_is_missing": 1,
                "ls_median_power_is_missing": 1,
                "ls_power_variance_is_missing": 1,
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
            fap_missing = 0
        except Exception:
            fap = 1.0
            fap_missing = 1

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
        }, {
            "ls_entropy_is_missing": 0,
            "ls_fap_is_missing": fap_missing,
            "ls_max_power_is_missing": 0,
            "ls_median_power_is_missing": 0,
            "ls_power_variance_is_missing": 0,
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
        }, {
            "ls_entropy_is_missing": 1,
            "ls_fap_is_missing": 1,
            "ls_max_power_is_missing": 1,
            "ls_median_power_is_missing": 1,
            "ls_power_variance_is_missing": 1,
        }


def extract_sn_features_from_file(csv_file: Path) -> Optional[dict]:
    """从SN文件提取特征（与 legacy/extract_features_full_supernova.py 一致，并加上 is_missing 特征）"""
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
        features["skewness_is_missing"] = int(len(mag) < 3)
        features["kurtosis"] = float(stats.kurtosis(mag)) if len(mag) > 3 else 0.0
        features["kurtosis_is_missing"] = int(len(mag) < 4)
        features["median_mag"] = float(np.median(mag))
        features["min_mag"] = float(np.min(mag))
        features["max_mag"] = float(np.max(mag))

        # 1105 新增：MAD, Beyond, Percent Amplitude, Autocorr（使用SN版本的计算方式）
        features["mad_mag"] = (
            float(np.median(np.abs(mag - np.median(mag)))) if len(mag) >= 2 else 0.0
        )

        # Beyond std（SN版本）
        if len(mag) >= 2 and features["std_mag"] > 0:
            features["beyond1std"] = float(
                np.mean(np.abs(mag - features["mean_mag"]) > 1.0 * features["std_mag"])
            )
            features["beyond2std"] = float(
                np.mean(np.abs(mag - features["mean_mag"]) > 2.0 * features["std_mag"])
            )
        else:
            features["beyond1std"] = 0.0
            features["beyond2std"] = 0.0

        # Percent amplitude（SN版本）
        if features["median_mag"] != 0:
            upper = abs(
                (features["max_mag"] - features["median_mag"]) / features["median_mag"]
            )
            lower = abs(
                (features["median_mag"] - features["min_mag"]) / features["median_mag"]
            )
            features["percent_amplitude"] = float(max(upper, lower))
        else:
            features["percent_amplitude"] = 0.0

        features["autocorr_lag1"] = autocorr_lag1(mag)

        # 最长单调段（SN版本的计算方式）
        if len(mag) >= 2:
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
            features["con_up_max_norm"] = float(up_max / n)
            features["con_down_max_norm"] = float(down_max / n)
        else:
            features["con_up_max_norm"] = 0.0
            features["con_down_max_norm"] = 0.0

        # 分位数（SN版本，返回tuple）
        if len(mag) >= 4:
            q10, q50, q90 = np.percentile(mag, [10, 50, 90])
            q90_minus_q10 = q90 - q10
            if q50 - q10 != 0:
                q_shape = (q90 - q50) / (q50 - q10)
            else:
                q_shape = 0.0
            features["q10"] = float(q10)
            features["q90"] = float(q90)
            features["q90_minus_q10"] = float(q90_minus_q10)
            features["q_shape_ratio"] = float(q_shape)
        else:
            features["q10"] = 0.0
            features["q90"] = 0.0
            features["q90_minus_q10"] = 0.0
            features["q_shape_ratio"] = 0.0

        # Gini（SN版本）
        if len(mag) >= 2:
            mag_shifted = mag - mag.min() + 1e-8
            sorted_mag = np.sort(mag_shifted)
            n = len(sorted_mag)
            index = np.arange(1, n + 1)
            features["gini_mag"] = float(
                (2 * np.sum(index * sorted_mag)) / (n * np.sum(sorted_mag))
                - (n + 1) / n
            )
        else:
            features["gini_mag"] = 0.0

        # 时序特征（SN版本）
        mag_diff = np.diff(mag)
        features["zero_crossings"] = int(np.sum(mag_diff[:-1] * mag_diff[1:] < 0))
        features["frac_rising"] = safe_divide(np.sum(mag_diff > 0), len(mag_diff))
        features["mag_change_rate"] = safe_divide(np.sum(np.abs(mag_diff)), len(mag))

        # 时间跨度与密度
        time_span = float(np.max(mjd) - np.min(mjd))
        features["time_span"] = time_span
        features["density"] = safe_divide(len(mag), time_span)

        # Lomb-Scargle（使用SN版本）
        ls_features, ls_missing = calculate_lomb_scargle_features_sn(mjd, mag)
        features.update(ls_features)
        features.update(ls_missing)

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

        # 加权统计（SN版本）
        if magerr is not None and len(magerr) > 0:
            weights = 1.0 / (magerr**2 + 1e-8)
            w_mean = float(np.average(mag, weights=weights))
            w_var = float(np.average((mag - w_mean) ** 2, weights=weights))
            w_std = float(np.sqrt(max(w_var, 0.0)))
            features["w_mean_mag"] = w_mean
            features["w_std_mag"] = w_std

            # Chi2 DOF（SN版本）
            mean_mag = np.mean(mag)
            chi2 = np.sum(((mag - mean_mag) / (magerr + 1e-8)) ** 2)
            dof = len(mag) - 1
            features["chi2_dof"] = float(chi2 / dof) if dof > 0 else 0.0

            # Median SNR（SN版本）
            median_mag = np.median(mag)
            snr = np.abs(mag - median_mag) / (magerr + 1e-8)
            features["median_snr"] = float(np.median(snr))
        else:
            features["w_mean_mag"] = features["mean_mag"]
            features["w_std_mag"] = features["std_mag"]
            features["chi2_dof"] = 0.0
            features["median_snr"] = 0.0

        # 时间间隔统计（SN版本）
        if len(mjd) >= 2:
            sorted_time = np.sort(mjd)
            gaps = np.diff(sorted_time)
            features["max_gap"] = float(np.max(gaps))
            features["gap_std"] = float(np.std(gaps))
            features["gap_median"] = float(np.median(gaps))
        else:
            features["max_gap"] = 0.0
            features["gap_std"] = 0.0
            features["gap_median"] = 0.0

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
        features["category"] = "SN"

        return features

    except Exception as e:
        logger.error(f"处理文件 {csv_file} 时出错: {e}")
        return None


def process_sn_directory(sn_dir: Path) -> list[dict]:
    """处理SN目录，使用57个特征版本"""
    if not sn_dir.exists():
        logger.warning(f"SN目录不存在: {sn_dir}")
        return []
    csv_files = list(sn_dir.glob("*.csv"))  # 只处理直接子目录的CSV，不递归
    if not csv_files:
        logger.warning("SN目录下没有CSV文件")
        return []
    logger.info(f"处理 SN 数据 ({len(csv_files)} 个文件)")
    features_list = []
    for i, file in enumerate(csv_files, 1):
        if i % 500 == 0:
            logger.info(f"SN处理进度: {i} / {len(csv_files)}")
        feat = extract_sn_features_from_file(file)
        if feat:
            features_list.append(feat)
    logger.info(f"✅ SN: 成功 {len(features_list)} / {len(csv_files)}")
    return features_list


def merge_categories(df: pd.DataFrame) -> pd.DataFrame:
    """根据映射合并类别"""
    df_copy = df.copy()
    logger.info("\n合并前的类别分布:")
    original_counts = df_copy["category"].value_counts().sort_index()
    for cat, count in original_counts.items():
        logger.info(f"  {cat:12s}: {count:6d}")

    df_copy["category"] = df_copy["category"].map(MERGE_MAPPING)

    logger.info("\n合并后的类别分布:")
    merged_counts = df_copy["category"].value_counts().sort_index()
    for cat, count in merged_counts.items():
        logger.info(f"  {cat:12s}: {count:6d}")

    return df_copy


def balance_dataset(
    df: pd.DataFrame, max_ratio: float = 4.0, exclude_categories: list = None
) -> pd.DataFrame:
    """平衡数据集，使最大类别样本数不超过最小类别的指定倍数

    Args:
        df: 数据框
        max_ratio: 最大比例
        exclude_categories: 不参与平衡的类别列表（如SN）
    """
    if exclude_categories is None:
        exclude_categories = []

    # 分离需要平衡和不需要平衡的数据
    exclude_mask = df["category"].isin(exclude_categories)
    df_to_balance = df[~exclude_mask].copy()
    df_excluded = df[exclude_mask].copy()

    if len(df_to_balance) == 0:
        logger.info("没有需要平衡的数据")
        return df

    category_counts = df_to_balance["category"].value_counts().sort_values()

    logger.info("\n平衡前各类别样本数（排除SN等）:")
    for cat, count in category_counts.items():
        logger.info(f"  {cat:12s}: {count:6d}")

    min_count = category_counts.min()
    max_allowed = int(min_count * max_ratio)

    logger.info(f"\n最小类别样本数: {min_count}")
    logger.info(f"最大允许样本数: {max_allowed} ({max_ratio}倍)")

    balanced_dfs = []

    for category in category_counts.index:
        cat_df = df_to_balance[df_to_balance["category"] == category]
        current_count = len(cat_df)

        if current_count > max_allowed:
            cat_df_sampled = cat_df.sample(n=max_allowed, random_state=42)
            removed = current_count - max_allowed
            logger.info(
                f"  {category:12s}: {current_count:6d} -> "
                f"{max_allowed:6d} (删除 {removed})"
            )
            balanced_dfs.append(cat_df_sampled)
        else:
            logger.info(f"  {category:12s}: {current_count:6d} (保持不变)")
            balanced_dfs.append(cat_df)

    result_df = pd.concat(balanced_dfs, ignore_index=True)
    result_df = result_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 拼接回排除的类别
    if len(df_excluded) > 0:
        logger.info(f"\n保留未平衡的类别: {df_excluded['category'].unique()}")
        result_df = pd.concat([result_df, df_excluded], ignore_index=True)
        result_df = result_df.sample(frac=1, random_state=42).reset_index(drop=True)

    logger.info("\n平衡后各类别样本数:")
    final_counts = result_df["category"].value_counts().sort_values()
    for cat, count in final_counts.items():
        logger.info(f"  {cat:12s}: {count:6d}")

    return result_df


def balance_datasets(
    train_df: pd.DataFrame, test_df: pd.DataFrame, exclude_categories: list = None
):
    """平衡训练集和测试集

    Args:
        train_df: 训练集
        test_df: 测试集
        exclude_categories: 不参与平衡的类别列表（如SN）
    """
    if exclude_categories is None:
        exclude_categories = []

    logger.info("步骤1: 根据映射合并类别")
    train_df = merge_categories(train_df)
    test_df = merge_categories(test_df)

    logger.info(
        "\n步骤2: 平衡数据集（最大比例={}倍，排除{}）".format(
            MAX_RATIO, exclude_categories
        )
    )
    logger.info("\n--- 训练集 ---")
    train_df = balance_dataset(
        train_df, max_ratio=MAX_RATIO, exclude_categories=exclude_categories
    )
    logger.info("\n--- 测试集 ---")
    test_df = balance_dataset(
        test_df, max_ratio=MAX_RATIO, exclude_categories=exclude_categories
    )

    return train_df, test_df


def main():
    logger.info("=" * 80)
    logger.info("【ZTF 特征提取 - train4 版】")
    logger.info("目标：从train4目录提取特征，SN从train4/SN读取，不参与4倍平衡")
    logger.info(f"输入目录: {TRAIN_DIR}")
    logger.info(f"测试集比例: {TEST_SIZE}")
    logger.info("=" * 80)

    if not TRAIN_DIR.exists():
        logger.error(f"训练目录不存在: {TRAIN_DIR}")
        return

    # 步骤1: 处理train4数据（57个特征）
    logger.info("\n" + "=" * 80)
    logger.info("步骤1: 处理train4数据（提取57个特征）")
    logger.info("=" * 80)
    train4_features = []
    for cat in CATEGORIES:
        feats = process_category(cat)
        train4_features.extend(feats)

    if not train4_features:
        logger.error("train4数据未提取到任何有效特征")
        return

    train4_df = pd.DataFrame(train4_features)
    logger.info(f"train4数据提取完成: {len(train4_df)} 个样本")

    # 步骤2: 处理SN数据（57个特征，不参与平衡）
    logger.info("\n" + "=" * 80)
    logger.info("步骤2: 处理SN数据（提取57个特征，不参与平衡）")
    logger.info("=" * 80)
    sn_features = process_sn_directory(SN_DIR)

    sn_df = pd.DataFrame(sn_features) if sn_features else pd.DataFrame()
    if len(sn_df) > 0:
        logger.info(f"SN数据提取完成: {len(sn_df)} 个样本")
    else:
        logger.warning("SN数据未提取到任何有效特征")

    # 步骤3: 合并train4和SN数据
    logger.info("\n" + "=" * 80)
    logger.info("步骤3: 合并train4和SN数据")
    logger.info("=" * 80)
    if len(sn_df) > 0:
        df = pd.concat([train4_df, sn_df], ignore_index=True)
        logger.info(
            f"合并后总样本数: {len(df)} (train4: {len(train4_df)}, SN: {len(sn_df)})"
        )
    else:
        df = train4_df
        logger.info(f"总样本数: {len(df)} (仅train4)")

    # 列顺序（57个特征 + file_path + category）
    feature_order = [
        "file_path",
        "num_points",
        "mean_mag",
        "std_mag",
        "iqr_mag",
        "amplitude_ratio",
        "skewness",
        "skewness_is_missing",
        "kurtosis",
        "kurtosis_is_missing",
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
        "ls_entropy_is_missing",
        "ls_harmonic_ratio",
        "ls_significant_peaks",
        "ls_peak_width",
        "ls_max_power",
        "ls_max_power_is_missing",
        "ls_median_power",
        "ls_median_power_is_missing",
        "ls_power_variance",
        "ls_power_variance_is_missing",
        "ls_dominant_period",
        "ls_fap",
        "ls_fap_is_missing",
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

    # 缺列容错：仅保留并按顺序输出存在的列
    df = df[[c for c in feature_order if c in df.columns]]

    # 填充NaN值：对数值列填充0，保留字符串列（file_path, category）
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # 检查NaN数量
    nan_counts = df[numeric_cols].isna().sum()
    if nan_counts.sum() > 0:
        logger.warning("发现NaN值，将填充为0:")
        for col, count in nan_counts[nan_counts > 0].items():
            logger.warning("  {}: {} 个NaN".format(col, count))
        df[numeric_cols] = df[numeric_cols].fillna(0.0)
        logger.info("✅ 所有数值列的NaN已填充为0")
    else:
        logger.info("✅ 未发现NaN值")

    output_features_dir = OUTPUT_DIR / "features"
    output_features_dir.mkdir(parents=True, exist_ok=True)

    # 步骤4: 划分train/test
    logger.info("\n" + "=" * 80)
    logger.info("步骤4: 划分train/test数据集")
    logger.info("=" * 80)
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["category"],
    )

    logger.info(f"划分完成: 训练集 {len(train_df)}, 测试集 {len(test_df)}")
    logger.info("\n训练集各类别数量:")
    logger.info(train_df["category"].value_counts().sort_index().to_string())
    logger.info("\n测试集各类别数量:")
    logger.info(test_df["category"].value_counts().sort_index().to_string())

    # 步骤5: 对train4类别进行4倍平衡（SN不参与）
    logger.info("\n" + "=" * 80)
    logger.info("步骤5: 平衡数据集（train4类别4倍平衡，SN不参与）")
    logger.info("=" * 80)
    train_balanced, test_balanced = balance_datasets(
        train_df.copy(), test_df.copy(), exclude_categories=["SN"]
    )

    # 步骤6: 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存未平衡版本
    train_out = output_features_dir / f"train4_1117_{timestamp}.csv"
    test_out = output_features_dir / f"test4_1117_{timestamp}.csv"
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)
    logger.info(f"✅ 未平衡训练集已保存: {train_out}")
    logger.info(f"✅ 未平衡测试集已保存: {test_out}")

    # 保存平衡版本
    train_balanced_out = output_features_dir / f"train4_1117_{timestamp}_balanced.csv"
    test_balanced_out = output_features_dir / f"test4_1117_{timestamp}_balanced.csv"
    train_balanced.to_csv(train_balanced_out, index=False)
    test_balanced.to_csv(test_balanced_out, index=False)

    logger.info(f"✅ 平衡训练集已保存: {train_balanced_out}")
    logger.info(f"✅ 平衡测试集已保存: {test_balanced_out}")
    logger.info(f"📊 平衡后 - 训练: {len(train_balanced)}, 测试: {len(test_balanced)}")
    logger.info("\n平衡后训练集各类别数量:")
    logger.info(train_balanced["category"].value_counts().sort_index().to_string())
    logger.info("\n平衡后测试集各类别数量:")
    logger.info(test_balanced["category"].value_counts().sort_index().to_string())
    logger.info("\n✅ 特征提取与平衡完成！")


if __name__ == "__main__":
    main()
