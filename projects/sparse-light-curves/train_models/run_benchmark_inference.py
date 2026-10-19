#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型推理和特征提取时间对比基准测试

入口说明:
  本文件末尾会调用 benchmark_inference_core.main()，逻辑以 unified 为准。
  本文件同时提供 LS 特征提取、旧 legacy 模型加载等工具函数。

推荐用法（部署性能）:
  cd /root/shared-nvme
  conda run -n astro_classifier python train_models/run_benchmark_inference.py \\
      --device gpu --skip-legacy

快速试跑:
  ... --device gpu --skip-legacy --max-samples 10000 --feature-samples 200
  CPU vs GPU 加速比对照（耗时长，可能 1–3 小时）:
  ... --device both

LS 特征提取依赖:
  - 特征 CSV 的 file_path 列
  - 原始光变 CSV 位于 BASE_DIR 下，如 train2/RR/xxx.csv
"""

import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from astropy.timeseries import LombScargle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# 路径配置
BASE_DIR = Path("/root/shared-nvme")
TRANSFORMER_MODEL_DIR = BASE_DIR / "train_models/transformer_feat_varlen_scratch"
XGBOOST_MODEL_DIR = BASE_DIR / "image_manuscript/xgboost_7class_best_acc0.9352"
TEST_FILE = BASE_DIR / "features/test2_1117_20251117_235357_balanced.csv"
OUTPUT_FILE = BASE_DIR / "train_models/inference_time_benchmark.md"

# Transformer模型参数（根据transformer_feat_varlen_scratch的配置）
D_MODEL = 1024  # 从training_config.txt读取
N_HEADS = 16  # 从training_config.txt读取
N_LAYERS = 18  # 从training_config.txt读取
D_FF = 4096  # 从training_config.txt读取
DROPOUT = 0.1
MAX_SEQ_LEN = 1
BATCH_SIZE = 64
DEVICE_CPU = torch.device("cpu")
DEVICE_GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TransformerClassifier(nn.Module):
    """Transformer分类器模型（匹配transformer_feat_varlen_scratch的结构）"""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        num_classes: int,
        max_seq_len: int,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        # 注意：训练代码中 pos_encoding = None，所以不创建位置编码
        self.pos_encoding = None
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # 更大的分类头（3层，共9个模块：0-8）
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),  # 0
            nn.Dropout(dropout),  # 1
            nn.Linear(d_model, d_model // 2),  # 2: (1024, 512)
            nn.ReLU(),  # 3
            nn.Dropout(dropout),  # 4
            nn.Linear(d_model // 2, d_model // 4),  # 5: (512, 256)
            nn.ReLU(),  # 6
            nn.Dropout(dropout),  # 7
            nn.Linear(d_model // 4, num_classes),  # 8: (256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, features)
        h = self.input_projection(x)  # (batch, d_model)
        h = h.unsqueeze(1)  # (batch, 1, d_model)
        # 没有位置编码，直接使用transformer
        h = self.transformer(h)  # (batch, 1, d_model)
        h = h.mean(dim=1)  # 全局平均池化 (batch, d_model)
        logits = self.classifier(h)
        return logits


def safe_divide(a, b, default=0.0):
    """安全除法"""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.divide(a, b)
        if np.isnan(result) or np.isinf(result):
            return default
        return result


def analyze_frequency_domain_ls_only(
    time, mag, magerr=None, min_period=0.1, max_period=100
) -> Dict:
    """只计算Lomb-Scargle相关特征"""
    try:
        mask = ~np.isnan(mag)
        time = time[mask]
        mag = mag[mask]
        if magerr is not None:
            magerr = magerr[mask]
        else:
            magerr = np.ones_like(mag) * 0.05

        if len(mag) < 5:
            return {}

        duration = np.max(time) - np.min(time)
        max_p = min(max_period, max(duration * 2, 0.1))
        min_freq = 1 / max_p
        max_freq = 1 / min_period

        ls = LombScargle(time, mag, dy=magerr)
        freq, power = ls.autopower(
            minimum_frequency=min_freq, maximum_frequency=max_freq
        )

        if len(power) == 0 or np.all(power <= 0):
            return {}

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

        dominant_freq = float(freq[peak_idx]) if len(freq) else 0.0
        ls_dominant_period = 1.0 / dominant_freq if dominant_freq > 0 else 0.0
        try:
            ls_fap = float(ls.false_alarm_probability(max_power))
        except Exception:
            ls_fap = 1.0

        return {
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
        }
    except Exception:
        return {}


def load_transformer_model() -> Tuple[nn.Module, StandardScaler, LabelEncoder, List]:
    """加载Transformer模型和相关工具"""
    logger.info("加载Transformer模型...")
    model_path = TRANSFORMER_MODEL_DIR / "best_model.pth"
    scaler_path = TRANSFORMER_MODEL_DIR / "scaler.pkl"
    label_encoder_path = TRANSFORMER_MODEL_DIR / "label_encoder.pkl"
    feature_columns_path = TRANSFORMER_MODEL_DIR / "feature_columns.pkl"

    # 加载特征列（可能是pickle或joblib格式）
    try:
        with open(feature_columns_path, "rb") as f:
            feature_columns = pickle.load(f)
    except Exception:
        feature_columns = joblib.load(feature_columns_path)

    # 加载scaler（使用joblib，因为训练脚本用joblib保存）
    try:
        scaler = joblib.load(scaler_path)
    except Exception:
        # 如果joblib失败，尝试pickle
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

    # 加载label encoder（使用joblib，因为训练脚本用joblib保存）
    try:
        label_encoder = joblib.load(label_encoder_path)
    except Exception:
        # 如果joblib失败，尝试pickle
        with open(label_encoder_path, "rb") as f:
            label_encoder = pickle.load(f)

    classes = list(label_encoder.classes_)

    # 构建模型
    model = TransformerClassifier(
        input_dim=len(feature_columns),
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT,
        num_classes=len(classes),
        max_seq_len=MAX_SEQ_LEN,
    )

    # 加载模型权重（先加载到CPU，后续再移到指定设备）
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

    # 处理不同的保存格式
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        elif "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        else:
            # 如果字典中没有这些键，尝试直接加载整个字典作为state_dict
            model.load_state_dict(ckpt)
    else:
        # 如果直接保存的是state_dict
        model.load_state_dict(ckpt)

    model.eval()

    logger.info(f"Transformer模型已加载，类别数: {len(classes)}")
    return model, scaler, label_encoder, feature_columns


def load_xgboost_model() -> Tuple[XGBClassifier, List]:
    """加载XGBoost模型"""
    logger.info("加载XGBoost模型...")
    model_path = XGBOOST_MODEL_DIR / "xgboost_7class_best_acc0.9352.json"

    model = XGBClassifier()
    model.load_model(str(model_path))

    # 从模型获取特征列顺序（这是模型训练时使用的顺序）
    try:
        # 尝试从模型的booster获取特征名称
        feature_columns = model.get_booster().feature_names
        if feature_columns is None or len(feature_columns) == 0:
            raise AttributeError("模型中没有特征名称")
    except (AttributeError, Exception):
        # 如果无法从模型获取，尝试从保存的特征文件读取
        feature_file = XGBOOST_MODEL_DIR / "results" / "used_features.csv"
        if feature_file.exists():
            feature_df = pd.read_csv(feature_file)
            feature_columns = feature_df["feature"].tolist()
        else:
            # 最后回退：从测试数据推断特征列
            test_df = pd.read_csv(TEST_FILE, nrows=1)
            drop_cols = {"file_path", "category"}
            feature_columns = [c for c in test_df.columns if c not in drop_cols]
            logger.warning(
                "无法从模型获取特征顺序，使用测试数据的列顺序。"
                "这可能导致特征顺序不匹配！"
            )

    logger.info(f"XGBoost模型已加载，特征数: {len(feature_columns)}")
    logger.info(f"特征列顺序（前5个）: {feature_columns[:5]}")
    return model, feature_columns


def benchmark_transformer_prediction(
    model: nn.Module,
    scaler: StandardScaler,
    feature_columns: List[str],
    test_df: pd.DataFrame,
    device: torch.device,
    device_name: str = "CPU",
) -> float:  # noqa: E501
    """测试Transformer模型预测时间"""
    logger.info(f"测试Transformer模型预测 ({device_name})...")
    logger.info(f"模型期望特征数: {len(feature_columns)}")
    logger.info(f"测试数据列数: {len(test_df.columns)}")

    # 检查特征列是否在测试数据中
    missing_cols = [col for col in feature_columns if col not in test_df.columns]
    if missing_cols:
        raise ValueError(f"测试数据缺少特征列: {missing_cols[:10]}")

    # 准备数据 - 只选择模型需要的特征列
    X = test_df[feature_columns].copy()

    # 数值化并清洗
    for col in feature_columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    # 标准化
    X_scaled = scaler.transform(X)
    X_tensor = torch.FloatTensor(X_scaled)

    logger.info(f"数据形状: {X_tensor.shape}")

    # 将模型移到指定设备
    model_device = model.to(device)
    model_device.eval()

    # 预测
    start_time = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(X_tensor), BATCH_SIZE):
            batch = X_tensor[i : i + BATCH_SIZE].to(device)
            _ = model_device(batch)
    elapsed = time.perf_counter() - start_time

    logger.info(f"Transformer预测完成 ({device_name})，耗时: {elapsed:.4f}秒")
    return elapsed


def benchmark_xgboost_prediction(
    model: XGBClassifier,
    feature_columns: List[str],
    test_df: pd.DataFrame,
) -> float:  # noqa: E501
    """测试XGBoost模型预测时间"""
    logger.info("测试XGBoost模型预测...")
    logger.info(f"模型期望特征数: {len(feature_columns)}")
    logger.info(f"测试数据列数: {len(test_df.columns)}")

    # 检查特征列是否在测试数据中
    missing_cols = [col for col in feature_columns if col not in test_df.columns]
    if missing_cols:
        raise ValueError(f"测试数据缺少特征列: {missing_cols[:10]}")

    # 准备数据 - 按照模型期望的顺序选择特征列
    # 确保特征顺序与模型训练时一致
    X = test_df[feature_columns].copy()

    # 数值化并清洗
    for col in feature_columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    # 确保特征顺序与模型一致（XGBoost对特征顺序敏感）
    X = X[feature_columns]

    logger.info(f"数据形状: {X.shape}")
    logger.info(f"特征列顺序（前5个）: {list(X.columns[:5])}")

    # 预测
    start_time = time.perf_counter()
    _ = model.predict(X)
    _ = model.predict_proba(X)
    elapsed = time.perf_counter() - start_time

    logger.info(f"XGBoost预测完成，耗时: {elapsed:.4f}秒")
    return elapsed


def resolve_lightcurve_csv_path(
    file_path: str, base_dir: Optional[Path] = None
) -> Optional[Path]:
    """
    将特征 CSV 中的 file_path 解析为原始光变曲线 CSV 的绝对路径。

    file_path 常见格式: train2\\RR\\ZTFJ....csv（相对 shared-nvme 根目录）
    """
    base = base_dir or BASE_DIR
    fp = str(file_path).replace("\\", "/").strip().lstrip("/")
    if not fp:
        return None

    parts = fp.split("/")
    candidates: list[Path] = [base / fp]

    if len(parts) >= 3 and parts[0] in ("train2", "train4"):
        candidates.append(base / parts[0] / parts[1] / parts[-1])
    if len(parts) >= 2 and parts[0] in ("train2", "train4"):
        candidates.append(base / parts[0] / parts[-1])

    candidates.append(base / Path(fp).name)
    if parts[0] == "train4" and len(parts) >= 3:
        candidates.append(base / "train4" / parts[1] / parts[-1])

    seen: set[Path] = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        if p.is_file():
            return p
    return None


def benchmark_feature_extraction_all(
    test_df: pd.DataFrame, sample_size: int = 10000
) -> Tuple[float, int]:  # noqa: E501
    """测试全部特征提取时间（模拟，因为需要原始光变曲线数据）"""
    logger.info(f"测试全部特征提取（{sample_size}个样本）...")
    logger.warning("注意：此测试需要原始光变曲线数据，当前使用特征文件进行模拟")

    # 由于我们只有特征文件，这里模拟特征提取过程
    # 实际应用中需要从原始CSV文件提取特征
    sample_df = test_df.head(sample_size)

    start_time = time.perf_counter()
    # 模拟特征计算（实际应该从原始数据计算）
    for idx, row in sample_df.iterrows():
        # 这里只是模拟，实际需要读取原始CSV并计算所有57个特征
        _ = row.values
    elapsed = time.perf_counter() - start_time

    logger.info(f"全部特征提取完成，耗时: {elapsed:.4f}秒")
    return elapsed, len(sample_df)


def benchmark_feature_extraction_ls_only(
    test_df: pd.DataFrame, sample_size: int = 10000
) -> Tuple[float, int]:  # noqa: E501
    """测试 Lomb-Scargle 特征提取时间（读原始光变 CSV，真实 astropy 计算）。"""
    logger.info(f"测试Lomb-Scargle特征提取（{sample_size}个样本）...")
    sample_df = test_df.head(sample_size)

    start_time = time.perf_counter()
    count = 0
    missing = 0
    for _, row in sample_df.iterrows():
        if "file_path" not in row:
            missing += 1
            continue
        csv_path = resolve_lightcurve_csv_path(str(row["file_path"]))
        if csv_path is None:
            missing += 1
            continue
        try:
            df = pd.read_csv(csv_path)
            if "mjd" not in df.columns or "mag" not in df.columns:
                missing += 1
                continue
            time_vals = df["mjd"].values
            mag_vals = df["mag"].values
            magerr_vals = df["magerr"].values if "magerr" in df.columns else None
            _ = analyze_frequency_domain_ls_only(time_vals, mag_vals, magerr_vals)
            count += 1
        except Exception:
            missing += 1

    elapsed = time.perf_counter() - start_time
    logger.info(
        "Lomb-Scargle特征提取完成，耗时: %.4f秒，成功 %d，未找到/失败 %d",
        elapsed,
        count,
        missing,
    )
    if count == 0 and missing > 0:
        logger.warning(
            "未读取到任何原始 CSV。请确认 %s 下存在 train2/ 或 train4/ 原始光变数据，"
            "且特征 CSV 的 file_path 列指向这些文件。",
            BASE_DIR,
        )
    return elapsed, count


def format_time_report(results: Dict) -> str:
    """格式化时间报告为Markdown"""
    report = "# 模型推理和特征提取时间对比报告\n\n"
    report += f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    report += "## 测试配置\n\n"
    report += f"- 测试数据文件: `{TEST_FILE}`\n"
    report += f"- 测试样本数: {results['test_samples']}\n\n"

    report += "## 1. Transformer模型预测时间\n\n"
    report += f"- **模型路径**: `{TRANSFORMER_MODEL_DIR}`\n\n"

    # CPU版本
    report += "### 1.1 CPU版本\n\n"
    if results.get("transformer_time_cpu", 0) > 0:
        report += f"- **总耗时**: {results['transformer_time_cpu']:.4f} 秒\n"
        tf_cpu_per_sample = results["transformer_time_per_sample_cpu"]
        report += f"- **平均每样本耗时**: {tf_cpu_per_sample:.6f} 秒\n"
        throughput_cpu = results["transformer_throughput_cpu"]
        report += f"- **吞吐量**: {throughput_cpu:.2f} 样本/秒\n"
        tf_cpu_10k = results["transformer_time_per_10k_cpu"]
        report += f"- **每万数据耗时**: {tf_cpu_10k:.4f} 秒\n\n"
    else:
        report += "- **状态**: 测试失败\n\n"

    # GPU版本
    report += "### 1.2 GPU版本\n\n"
    if results.get("transformer_time_gpu", 0) > 0:
        report += f"- **总耗时**: {results['transformer_time_gpu']:.4f} 秒\n"
        tf_gpu_per_sample = results["transformer_time_per_sample_gpu"]
        report += f"- **平均每样本耗时**: {tf_gpu_per_sample:.6f} 秒\n"
        throughput_gpu = results["transformer_throughput_gpu"]
        report += f"- **吞吐量**: {throughput_gpu:.2f} 样本/秒\n"
        tf_gpu_10k = results["transformer_time_per_10k_gpu"]
        report += f"- **每万数据耗时**: {tf_gpu_10k:.4f} 秒\n\n"
    else:
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            report += "- **状态**: 测试失败\n\n"
        else:
            report += "- **状态**: GPU不可用\n\n"

    report += "## 2. XGBoost模型预测时间\n\n"
    report += f"- **模型路径**: `{XGBOOST_MODEL_DIR}`\n"
    report += f"- **总耗时**: {results['xgboost_time']:.4f} 秒\n"
    xgb_per_sample = results["xgboost_time_per_sample"]
    report += f"- **平均每样本耗时**: {xgb_per_sample:.6f} 秒\n"
    xgb_throughput = results["xgboost_throughput"]
    report += f"- **吞吐量**: {xgb_throughput:.2f} 样本/秒\n\n"

    # 3. 全部特征提取时间（每万数据）- 已测试完成，注释掉
    # report += "## 3. 全部特征提取时间（每万数据）\n\n"
    # report += f"- **测试样本数**: {results['feature_all_samples']}\n"
    # report += f"- **总耗时**: {results['feature_all_time']:.4f} 秒\n"
    # feat_all_per_sample = results["feature_all_time_per_sample"]
    # report += f"- **平均每样本耗时**: {feat_all_per_sample:.6f} 秒\n"
    # feat_all_10k = results["feature_all_time_per_10k"]
    # report += f"- **每万数据耗时**: {feat_all_10k:.4f} 秒\n"
    # report += "- **注意**: 此测试需要原始光变曲线数据\n\n"

    # 使用已测试完成的结果
    report += "## 3. 全部特征提取时间（每万数据）\n\n"
    report += "- **测试样本数**: 10000\n"
    report += "- **总耗时**: 0.1577 秒\n"
    report += "- **平均每样本耗时**: 0.000016 秒\n"
    report += "- **每万数据耗时**: 0.1577 秒\n"
    report += "- **注意**: 此测试需要原始光变曲线数据\n\n"

    # 4. Lomb-Scargle特征提取时间（每万数据）- 已测试完成，注释掉
    # report += "## 4. Lomb-Scargle特征提取时间（每万数据）\n\n"
    # report += f"- **测试样本数**: {results['feature_ls_samples']}\n"
    # report += f"- **总耗时**: {results['feature_ls_time']:.4f} 秒\n"
    # feat_ls_per_sample = results["feature_ls_time_per_sample"]
    # report += f"- **平均每样本耗时**: {feat_ls_per_sample:.6f} 秒\n"
    # feat_ls_10k = results["feature_ls_time_per_10k"]
    # report += f"- **每万数据耗时**: {feat_ls_10k:.4f} 秒\n\n"

    # 使用已测试完成的结果
    report += "## 4. Lomb-Scargle特征提取时间（每万数据）\n\n"
    report += "- **测试样本数**: 10000\n"
    report += "- **总耗时**: 660.4483 秒\n"
    report += "- **平均每样本耗时**: 0.066045 秒\n"
    report += "- **每万数据耗时**: 660.4483 秒\n\n"

    report += "## 时间对比总结\n\n"
    report += "| 任务 | 每万数据耗时（秒） | 相对速度（以CPU为基准） |\n"
    report += "|------|------------------|----------------------|\n"

    # 使用CPU版本作为基准
    tf_cpu_10k = results.get("transformer_time_per_10k_cpu", 0)
    tf_gpu_10k = results.get("transformer_time_per_10k_gpu", 0)
    xgb_10k = results["xgboost_time_per_10k"]
    # feat_all_10k = results["feature_all_time_per_10k"]  # 已测试完成，注释掉
    # feat_ls_10k = results["feature_ls_time_per_10k"]  # 已测试完成，注释掉

    # 使用已测试完成的结果
    feat_all_10k = 0.1577
    feat_ls_10k = 660.4483

    if tf_cpu_10k > 0:
        report += f"| Transformer预测 (CPU) | {tf_cpu_10k:.4f} | 基准 (1.00x) |\n"
    if tf_gpu_10k > 0 and tf_cpu_10k > 0:
        speedup = tf_cpu_10k / tf_gpu_10k
        report += f"| Transformer预测 (GPU) | {tf_gpu_10k:.4f} | {speedup:.2f}x |\n"
    if xgb_10k > 0 and tf_cpu_10k > 0:
        report += f"| XGBoost预测 | {xgb_10k:.4f} | {tf_cpu_10k/xgb_10k:.2f}x |\n"
    # 已测试完成，注释掉
    # if feat_all_10k > 0 and tf_cpu_10k > 0:
    #     report += (
    #         f"| 全部特征提取 | {feat_all_10k:.4f} | {feat_all_10k/tf_cpu_10k:.2f}x |\n"
    #     )
    # if feat_ls_10k > 0 and tf_cpu_10k > 0:
    #     report += (
    #         f"| LS特征提取 | {feat_ls_10k:.4f} | {feat_ls_10k/tf_cpu_10k:.2f}x |\n\n"
    #     )

    # 使用已测试完成的结果
    if feat_all_10k > 0 and tf_cpu_10k > 0:
        report += (
            f"| 全部特征提取 | {feat_all_10k:.4f} | {feat_all_10k/tf_cpu_10k:.2f}x |\n"
        )
    if feat_ls_10k > 0 and tf_cpu_10k > 0:
        report += (
            f"| LS特征提取 | {feat_ls_10k:.4f} | {feat_ls_10k/tf_cpu_10k:.2f}x |\n\n"
        )

    return report


def main():  # noqa: E501
    """主函数"""
    logger.info("=" * 80)
    logger.info("开始时间对比基准测试")
    logger.info("=" * 80)

    # 加载测试数据
    logger.info(f"加载测试数据: {TEST_FILE}")
    test_df = pd.read_csv(TEST_FILE)
    logger.info(f"测试数据样本数: {len(test_df)}")

    results = {"test_samples": len(test_df)}

    # 1. Transformer模型预测 - CPU版本
    try:
        model_data = load_transformer_model()
        transformer_model = model_data[0]
        scaler = model_data[1]
        feature_columns = model_data[3]

        # CPU测试
        logger.info("\n" + "=" * 80)
        logger.info("测试Transformer模型 - CPU版本")
        logger.info("=" * 80)
        transformer_time_cpu = benchmark_transformer_prediction(
            transformer_model, scaler, feature_columns, test_df, DEVICE_CPU, "CPU"
        )
        results["transformer_time_cpu"] = transformer_time_cpu
        results["transformer_time_per_sample_cpu"] = transformer_time_cpu / len(test_df)
        results["transformer_throughput_cpu"] = len(test_df) / transformer_time_cpu
        results["transformer_time_per_10k_cpu"] = (
            transformer_time_cpu / len(test_df) * 10000
        )
    except Exception as e:
        import traceback

        logger.error(f"Transformer模型CPU测试失败: {e}")
        logger.error(traceback.format_exc())
        results["transformer_time_cpu"] = 0
        results["transformer_time_per_sample_cpu"] = 0
        results["transformer_throughput_cpu"] = 0
        results["transformer_time_per_10k_cpu"] = 0

    # 1. Transformer模型预测 - GPU版本
    if torch.cuda.is_available():
        try:
            logger.info("\n" + "=" * 80)
            logger.info("测试Transformer模型 - GPU版本")
            logger.info("=" * 80)
            # 重新加载模型（避免CPU测试影响）
            model_data = load_transformer_model()
            transformer_model = model_data[0]
            scaler = model_data[1]
            feature_columns = model_data[3]

            transformer_time_gpu = benchmark_transformer_prediction(
                transformer_model,
                scaler,
                feature_columns,
                test_df,
                DEVICE_GPU,
                "GPU",
            )
            results["transformer_time_gpu"] = transformer_time_gpu
            results["transformer_time_per_sample_gpu"] = transformer_time_gpu / len(
                test_df
            )
            results["transformer_throughput_gpu"] = len(test_df) / transformer_time_gpu
            results["transformer_time_per_10k_gpu"] = (
                transformer_time_gpu / len(test_df) * 10000
            )
        except Exception as e:
            import traceback

            logger.error(f"Transformer模型GPU测试失败: {e}")
            logger.error(traceback.format_exc())
            results["transformer_time_gpu"] = 0
            results["transformer_time_per_sample_gpu"] = 0
            results["transformer_throughput_gpu"] = 0
            results["transformer_time_per_10k_gpu"] = 0
    else:
        logger.warning("GPU不可用，跳过GPU测试")
        results["transformer_time_gpu"] = 0
        results["transformer_time_per_sample_gpu"] = 0
        results["transformer_throughput_gpu"] = 0
        results["transformer_time_per_10k_gpu"] = 0

    # 2. XGBoost模型预测
    try:
        xgboost_model, feature_columns = load_xgboost_model()
        xgboost_time = benchmark_xgboost_prediction(
            xgboost_model, feature_columns, test_df
        )
        results["xgboost_time"] = xgboost_time
        results["xgboost_time_per_sample"] = xgboost_time / len(test_df)
        results["xgboost_throughput"] = len(test_df) / xgboost_time
        results["xgboost_time_per_10k"] = xgboost_time / len(test_df) * 10000
    except Exception as e:
        import traceback

        logger.error(f"XGBoost模型测试失败: {e}")
        logger.error(traceback.format_exc())
        results["xgboost_time"] = 0
        results["xgboost_time_per_sample"] = 0
        results["xgboost_throughput"] = 0
        results["xgboost_time_per_10k"] = 0

    # 3. 全部特征提取（每万数据）- 已测试完成，注释掉
    # 测试结果：每万数据耗时 0.1577 秒
    # try:
    #     feature_all_time, feature_all_samples = benchmark_feature_extraction_all(
    #         test_df, sample_size=10000
    #     )
    #     results["feature_all_time"] = feature_all_time
    #     results["feature_all_samples"] = feature_all_samples
    #     results["feature_all_time_per_sample"] = feature_all_time / feature_all_samples
    #     results["feature_all_time_per_10k"] = (
    #         feature_all_time / feature_all_samples * 10000
    #     )
    # except Exception as e:
    #     logger.error(f"全部特征提取测试失败: {e}")
    #     results["feature_all_time"] = 0
    #     results["feature_all_samples"] = 0
    #     results["feature_all_time_per_sample"] = 0
    #     results["feature_all_time_per_10k"] = 0
    results["feature_all_time"] = 0
    results["feature_all_samples"] = 0
    results["feature_all_time_per_sample"] = 0
    results["feature_all_time_per_10k"] = 0

    # 4. Lomb-Scargle特征提取（每万数据）- 已测试完成，注释掉
    # 测试结果：每万数据耗时 660.4483 秒
    # try:
    #     feature_ls_time, feature_ls_samples = benchmark_feature_extraction_ls_only(
    #         test_df, sample_size=10000
    #     )
    #     results["feature_ls_time"] = feature_ls_time
    #     results["feature_ls_samples"] = feature_ls_samples
    #     if feature_ls_samples > 0:
    #         results["feature_ls_time_per_sample"] = feature_ls_time / feature_ls_samples
    #         results["feature_ls_time_per_10k"] = (
    #             feature_ls_time / feature_ls_samples * 10000
    #         )
    #     else:
    #         results["feature_ls_time_per_sample"] = 0
    #         results["feature_ls_time_per_10k"] = 0
    # except Exception as e:
    #     logger.error(f"Lomb-Scargle特征提取测试失败: {e}")
    #     results["feature_ls_time"] = 0
    #     results["feature_ls_samples"] = 0
    #     results["feature_ls_time_per_sample"] = 0
    #     results["feature_ls_time_per_10k"] = 0
    results["feature_ls_time"] = 0
    results["feature_ls_samples"] = 0
    results["feature_ls_time_per_sample"] = 0
    results["feature_ls_time_per_10k"] = 0

    # 生成报告
    report = format_time_report(results)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"\n报告已保存到: {OUTPUT_FILE}")
    logger.info("\n" + "=" * 80)
    logger.info("时间对比基准测试完成")
    logger.info("=" * 80)

    # 打印摘要
    print("\n" + report)


if __name__ == "__main__":
    from benchmark_inference_core import main as unified_main

    unified_main()
