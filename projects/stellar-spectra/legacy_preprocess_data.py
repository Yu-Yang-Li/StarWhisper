   #!/usr/bin/env python3
"""
02_data_preprocessing.py - 光谱数据预处理器
- 支持并行处理所有CSV文件
- 位置编码与token化
- 生成训练/验证集和词汇表
"""

import argparse  # 命令行参数
import os  # 操作系统相关
import re  # 正则表达式
import warnings  # 忽略警告
from multiprocessing import Pool, cpu_count  # 并行处理
import sys

import numpy as np  # 数值计算
import pandas as pd  # 数据处理
from sklearn.model_selection import train_test_split  # 数据集划分
from tqdm import tqdm  # 进度条

warnings.filterwarnings("ignore")
import hashlib
from collections import Counter
import multiprocessing as mp

# ========= 固定词表支持 =========
def load_vocab_tokens(vocab_path: str):
    """
    从给定词表CSV读取 token 集合。
    - 支持包含列名 'token'（如 vocabulary.csv: token_id,token）
    - 若不存在或读取失败，返回 None
    """
    try:
        if not vocab_path or not os.path.exists(vocab_path):
            return None
        df = pd.read_csv(vocab_path)
        if 'token' in df.columns:
            return set(df['token'].astype(str).tolist())
        # 兜底：取最后一列为 token
        if len(df.columns) >= 1:
            last_col = df.columns[-1]
            return set(df[last_col].astype(str).tolist())
        return None
    except Exception:
        return None


def validate_df_tokens_against_vocab(df: pd.DataFrame, tokens_set: set):
    """对 df 的 token 列做一致性校验，若存在 OOV 则报错退出。"""
    token_cols = [
        c for c in df.columns
        if c.endswith('_token')
        or c.startswith('Teff_')
        or c.startswith('logg_')
        or c.startswith('FeH_')
        or c.startswith('flux_')
    ]
    missing = set()
    for c in token_cols:
        if c in df.columns:
            missing.update(set(pd.Series(df[c]).dropna().unique()) - tokens_set)
    if missing:
        print("❌ 发现不在固定词表中的 token，示例:", list(sorted(list(missing)))[:20])
        print("请先将 vocabulary.csv 更新为包含全部 token 的固定词表后重试。")
        sys.exit(1)

# 与 backup/stage1 完全一致的20列顺序
OUTPUT_COLUMNS = [
    'spectrum_id', 'pixel_idx',
    'Teff_tthou', 'Teff_thu', 'Teff_hun', 'Teff_ten', 'Teff_one',
    'logg_hun', 'logg_ten', 'logg_one', 'logg_sign',
    'FeH_ten', 'FeH_one', 'FeH_sign',
    'flux_thu', 'flux_hun', 'flux_ten', 'flux_one',
    'BOS_token', 'EOS_token', 'SEP_token'
]

# 新的参数提取，支持正负号和浮点数
# 文件名格式如：10560+3.20-1.0.csv 或 10560-3.20+1.0.csv
# 文件夹名如：/home/share/guofangkeda/wangcunshi/Spec/R1800FITS/SNR1/Z-1.0/10560+3.20-1.0.csv


def extract_parameters_from_filename(filename):
    """
    从文件名中提取恒星参数（Teff, logg, FeH）
    文件名格式如: 10560+3.20-1.0.csv 或 10560-3.20+1.0.csv
    返回: teff, logg, feh (float)
    """
    # 获取文件名（去除路径和扩展名）
    stem = os.path.splitext(os.path.basename(filename))[0]
    # 用正则表达式提取整数+符号浮点+符号浮点
    match = re.match(r"^(\d+)([+-]\d+\.\d+)([+-]\d+\.\d+)$", stem)
    if match:
        try:
            teff = float(match.group(1))  # 有效温度
            # 注意：第二段的 +/- 在现有文件命名中作为段分隔符，不表示 logg 的负号
            # 因此这里对 logg 取绝对值，确保按正值解析；FeH 仍保留真实符号
            logg = abs(float(match.group(2)))  # 表面重力（非负）
            feh = float(match.group(3))  # 金属丰度（保留符号）
            return teff, logg, feh
        except Exception:
            return None, None, None
    return None, None, None


# 递归遍历所有csv文件，返回绝对路径


def get_all_csv_files(root_dir):
    csv_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith(".csv"):
                csv_files.append(os.path.join(dirpath, f))
    return csv_files


# ============ 新增：分批工具 ============
def batch_files_by_size(file_paths, batch_size_gb: float = 2.0):
    """
    按累计文件大小分批返回文件列表（默认每批约2GB）。
    """
    batch = []
    batch_size_mb = 0.0
    target_mb = float(batch_size_gb) * 1024.0
    for fp in file_paths:
        try:
            size_mb = os.path.getsize(fp) / (1024.0 * 1024.0)
        except OSError:
            continue
        if batch and (batch_size_mb + size_mb) > target_mb:
            yield batch
            batch = []
            batch_size_mb = 0.0
        batch.append(fp)
        batch_size_mb += size_mb
    if batch:
        yield batch


def batch_files_by_count(file_paths, batch_count: int):
    """
    按固定文件数分批返回文件列表（batch_count<=0 则整体一批）。
    """
    if not batch_count or batch_count <= 0:
        yield file_paths
        return
    for i in range(0, len(file_paths), batch_count):
        yield file_paths[i:i + batch_count]


def get_next_spec_id_from_csv(csv_path: str) -> int:
    """
    流式扫描已存在的 CSV，返回应续写的下一个 spectrum_id（max+1）。
    若文件不存在或为空，返回 0。
    """
    try:
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
            return 0
        max_id = -1
        for chunk in pd.read_csv(csv_path, usecols=["spectrum_id"], chunksize=1_000_000):
            local_max = chunk["spectrum_id"].max()
            if pd.notna(local_max) and int(local_max) > max_id:
                max_id = int(local_max)
        return (max_id + 1) if max_id >= 0 else 0
    except Exception:
        return 0


# ============ 新增：按 SNR 列表在 R1800FITS 根目录收集文件 ============
def collect_csv_files_by_snrs(fits_root: str, snr_list):
    files = []
    for snr in snr_list:
        snr_dir = os.path.join(fits_root, snr)
        if not os.path.isdir(snr_dir):
            continue
        for dirpath, dirnames, filenames in os.walk(snr_dir):
            for f in filenames:
                if f.endswith('.csv'):
                    files.append(os.path.join(dirpath, f))
    return files


# SNR从根目录名提取


def extract_snr_from_path(path):
    # 假设SNR目录为.../SNR1/...
    parts = path.split(os.sep)
    for p in parts:
        if p.startswith("SNR"):
            try:
                return int(p.replace("SNR", ""))
            except Exception:
                return None
    return None


# 修改load_single_spectrum，传入绝对路径


def load_single_spectrum(args):
    """
    加载单个光谱文件，归一化并返回字典
    参数:
        args: (filepath)
    返回:
        dict: 包含teff, logg, feh, 归一化光谱, SNR, 路径
    """
    filepath = args
    teff, logg, feh = extract_parameters_from_filename(filepath)
    if teff is None:
        return None
    try:
        # 读取光谱数据
        spectrum = pd.read_csv(filepath, header=0).values.flatten()
        # 归一化到0-1范围
        spec_min = np.min(spectrum)
        spec_max = np.max(spectrum)
        if spec_max > spec_min:
            normalized_spectrum = (spectrum - spec_min) / (spec_max - spec_min)
        else:
            normalized_spectrum = spectrum * 0
        # 从路径中提取SNR
        snr = extract_snr_from_path(filepath)
        return {
            "filepath": filepath,
            "teff": teff,
            "logg": logg,
            "feh": feh,
            "spectrum": normalized_spectrum,
            "snr": snr,
        }
    except Exception as e:
        print(f"⚠️ Error loading {filepath}: {e}")
        return None


def tokenize_parameters(teff, logg, feh):
    """参数token化（英文缩写位置编码）"""
    tokens = {}

    # Teff: 5位数 万千百十个 -> tthou, thu, hun, ten, one
    teff_int = int(round(teff))
    teff_str = f"{teff_int:05d}"
    positions_teff = ["tthou", "thu", "hun", "ten", "one"]
    for i, digit_char in enumerate(teff_str):
        digit = int(digit_char)
        pos = positions_teff[i]
        tokens[f"Teff_{pos}"] = f"T{digit}_{pos}"

    # logg: 保留真实符号，数位拆分仍按绝对值；新增 logg_sign 表示正负
    logg_int = int(round(abs(logg) * 100))
    logg_str = f"{logg_int:03d}"
    positions_logg = ["hun", "ten", "one"]
    for i, digit_char in enumerate(logg_str):
        digit = int(digit_char)
        pos = positions_logg[i]
        tokens[f"logg_{pos}"] = f"L{digit}_{pos}"
    tokens["logg_sign"] = "L_pos" if logg >= 0 else "L_neg"
    

    # FeH: 2位数 + 符号 十个 -> ten, one
    feh_abs = abs(feh)
    feh_int = int(round(feh_abs * 10))
    feh_str = f"{feh_int:02d}"
    positions_feh = ["ten", "one"]
    for i, digit_char in enumerate(feh_str):
        digit = int(digit_char)
        pos = positions_feh[i]
        tokens[f"FeH_{pos}"] = f"F{digit}_{pos}"

    tokens["FeH_sign"] = "F_pos" if feh >= 0 else "F_neg"

    return tokens


def process_spectrum_chunk(args):
    """处理一个光谱的所有像素点"""
    spec_data, spec_idx, max_pixels = args

    param_tokens = tokenize_parameters(
        spec_data["teff"], spec_data["logg"], spec_data["feh"]
    )

    rows = []
    spectrum = spec_data["spectrum"]

    # 处理所有像素点（或限制数量用于测试）
    num_pixels = len(spectrum) if max_pixels is None else min(max_pixels, len(spectrum))

    for i in range(num_pixels):
        # 按需求将输出的 pixel_idx 从 4 开始并每点递增 10
        pixel_idx = 4 + i * 10
        flux_value = spectrum[i]

        row_data = {
            "spectrum_id": spec_idx,
            "pixel_idx": pixel_idx,
        }

        # 添加参数tokens
        row_data.update(param_tokens)

        # flux tokens 千百十个 -> thu, hun, ten, one
        scaled_value = int(flux_value * 9999)
        scaled_value = np.clip(scaled_value, 0, 9999)
        value_str = f"{scaled_value:04d}"

        positions = ["thu", "hun", "ten", "one"]
        for i, digit_char in enumerate(value_str):
            digit = int(digit_char)
            pos = positions[i]
            row_data[f"flux_{pos}"] = f"S{digit}_{pos}"

        # 添加特殊tokens（统一使用 <SEP> 作为分隔/填充）
        row_data["BOS_token"] = "<BOS>"
        row_data["EOS_token"] = "<EOS>"
        row_data["SEP_token"] = "<SEP>"

        rows.append(row_data)

    return rows


class SpectrumPreprocessor:
    """
    光谱数据预处理器主类
    用于批量加载、token化、分割、保存光谱数据
    """

    def __init__(
        self,
        data_dir="/home/share/guofangkeda/wangcunshi/Spec/R1800FITS/SNR1",
        n_processes=None,
        test_mode=False,
        max_files=None,
        max_pixels=None,
        vocab_path: str = "vocabulary.csv",
    ):
        """
        初始化预处理器
        参数:
            data_dir: 数据目录
            n_processes: 并行进程数
            test_mode: 是否测试模式
            max_files: 最大文件数
            max_pixels: 每条光谱最大像素数
        """
        self.data_dir = data_dir
        self.n_processes = n_processes or min(cpu_count(), 8)
        # 进程数：优先使用用户传入；未传则使用操作系统可见的 CPU 数
        self.n_processes = (
            n_processes if (n_processes is not None and n_processes > 0) else (os.cpu_count() or 1)
        )
        self.test_mode = test_mode
        self.max_files = max_files
        self.max_pixels = max_pixels
        self.tokenized_dataset = None
        self.full_df = None
        self.train_df = None
        self.val_df = None
        self.vocab_df = None
        # 固定词表（可选）
        self.vocab_path = vocab_path
        self.vocab_tokens_set = load_vocab_tokens(self.vocab_path)

    def run(self):
        """
        运行完整的预处理流程
        步骤：加载光谱 -> token化 -> 分割保存 -> 生成词表 -> 显示样本
        """
        print("🚀 光谱数据预处理器 v2.0")
        print("=" * 60)
        print(f"📂 数据目录: {self.data_dir}")
        print(f"⚡ 进程数: {self.n_processes}")
        print(f"🔤 位置编码: 万(tthou)千(thu)百(hun)十(ten)个(one)")
        if self.test_mode:
            print(
                f"🧪 测试模式: 最多 {self.max_files} 文件, {self.max_pixels} 像素/光谱"
            )
        if not os.path.exists(self.data_dir):
            print(f"❌ 数据目录不存在: {self.data_dir}")
            return
        # Step 1: 加载光谱数据
        spectra_data = self.load_spectra()
        if not spectra_data:
            print("❌ 没有成功加载任何光谱数据")
            return
        # Step 2: 创建token化数据集
        self.create_tokenized_dataset(spectra_data)
        # Step 3: 分割和保存数据
        self.split_and_save_data()
        # Step 4: 保存词汇表
        self.save_vocabulary()
        # Step 5: 显示样本
        self.show_samples()
        print(f"\n🎉 预处理完成！")
        print(f"   📊 完整数据: spectrum_tokenized_full.csv")
        print(f"   🚂 训练数据: spectrum_tokenized_train.csv")
        print(f"   📋 验证数据: spectrum_tokenized_val.csv")
        print(f"   📚 词汇表: token_vocabulary.csv")
        return self.train_df, self.val_df

    # ============ 新增：构建指定 SNR 集合的数据集（不改变原有 run 行为） ============
    def _build_group_dataset_df(self, fits_root: str, snr_list) -> pd.DataFrame:
        print(f"\n📂 组合构建数据集: SNR={snr_list}")
        csv_files = collect_csv_files_by_snrs(fits_root, snr_list)
        if self.max_files:
            csv_files = csv_files[: self.max_files]
        print(f"   发现 {len(csv_files)} 个CSV文件")
        # 并行加载（进度条）
        with Pool(self.n_processes) as pool:
            results = list(tqdm(
                pool.imap_unordered(load_single_spectrum, csv_files, chunksize=64),
                total=len(csv_files), desc="📥 加载光谱", smoothing=0.1
            ))
        spectra_data = [r for r in results if r is not None]
        print(f"✅ 成功加载 {len(spectra_data)} 个光谱")

        # 使用与原 create_tokenized_dataset 相同的流程构建逐像素行
        args_list = [
            (spec_data, idx, self.max_pixels)
            for idx, spec_data in enumerate(spectra_data)
        ]
        all_rows = []
        with Pool(self.n_processes) as pool:
            for chunk in tqdm(
                pool.imap_unordered(process_spectrum_chunk, args_list, chunksize=8),
                total=len(args_list), desc="🔣 逐像素token化", smoothing=0.1
            ):
                all_rows.extend(chunk)
        for i, row in enumerate(all_rows):
            spec_data = spectra_data[row["spectrum_id"]]
            row["teff"] = spec_data["teff"]
            row["logg"] = spec_data["logg"]
            row["feh"] = spec_data["feh"]
            row["SNR"] = spec_data.get("snr")
            row["path"] = spec_data.get("filepath")
            row["seq"] = i + 1
        df = pd.DataFrame(all_rows)
        if not df.empty:
            df = df[OUTPUT_COLUMNS]
        # 词表一致性校验（仅检查，不改词表）
        if self.vocab_tokens_set is not None and not df.empty:
            validate_df_tokens_against_vocab(df, self.vocab_tokens_set)
        else:
            print("⚠️ 未启用固定词表校验（未找到或未指定 vocabulary.csv）。")

        return df

    def build_group_dataset(self, fits_root: str, snr_list, output_path: str):
        df = self._build_group_dataset_df(fits_root, snr_list)
        # 确保输出目录存在并保存单文件
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"✅ 保存 {output_path} ({len(df):,} 行)")

    def load_spectra(self):
        """
        加载所有光谱文件，支持并行
        返回: spectra_data 列表，每个元素为单光谱字典
        """
        print(f"\n📂 加载光谱数据...")
        # 获取所有csv文件路径
        csv_files = get_all_csv_files(self.data_dir)
        if self.max_files:
            csv_files = csv_files[: self.max_files]
        print(f"   发现 {len(csv_files)} 个CSV文件")
        # 并行加载
        with Pool(self.n_processes) as pool:
            results = pool.map(load_single_spectrum, csv_files)
        # 过滤掉加载失败的
        spectra_data = [r for r in results if r is not None]
        print(f"✅ 成功加载 {len(spectra_data)} 个光谱")
        return spectra_data

    def create_tokenized_dataset(self, spectra_data):
        """
        创建token化数据集
        参数:
            spectra_data: 单光谱字典列表
        """
        print(f"\n🔄 创建token化数据集...")
        # 构造token化任务参数
        args_list = [
            (spec_data, idx, self.max_pixels)
            for idx, spec_data in enumerate(spectra_data)
        ]
        # 并行处理每个光谱
        with Pool(self.n_processes) as pool:
            chunk_results = pool.map(process_spectrum_chunk, args_list)
        all_rows = []
        for chunk in chunk_results:
            all_rows.extend(chunk)
        # 增加teff, logg, feh, SNR, 路径等信息
        for i, row in enumerate(all_rows):
            spec_data = spectra_data[row["spectrum_id"]]
            row["teff"] = spec_data["teff"]
            row["logg"] = spec_data["logg"]
            row["feh"] = spec_data["feh"]
            row["SNR"] = spec_data["snr"]
            row["path"] = spec_data["filepath"]
            row["seq"] = i + 1
        self.tokenized_dataset = pd.DataFrame(all_rows)
        print(f"✅ 创建完成: {self.tokenized_dataset.shape}")

    def split_and_save_data(self):
        """
        分割token化数据集为训练集和验证集，并保存为csv文件。
        训练集和验证集比例为9:1，同时保存人工可读的train_info.csv和val_info.csv。
        """
        print(f"\n🔀 分割数据集...")
        # 获取所有唯一的光谱ID
        unique_spectrum_ids = self.tokenized_dataset["spectrum_id"].unique()
        # 按9:1划分训练集和验证集
        train_spec_ids, val_spec_ids = train_test_split(
            unique_spectrum_ids, test_size=0.1, random_state=42  # 9:1
        )
        # 选出训练集和验证集的所有数据点
        full_base = self.tokenized_dataset[OUTPUT_COLUMNS]
        train_data = full_base[full_base["spectrum_id"].isin(train_spec_ids)]
        val_data = full_base[full_base["spectrum_id"].isin(val_spec_ids)]
        # 与 backup 完全一致的三份输出
        self.full_df = full_base
        self.train_df = train_data
        self.val_df = val_data
        # 保存token化csv
        self.full_df.to_csv("spectrum_tokenized_full.csv", index=False)
        self.train_df.to_csv("spectrum_tokenized_train.csv", index=False)
        self.val_df.to_csv("spectrum_tokenized_val.csv", index=False)
        # 保存人工可读的train_info.csv和val_info.csv
        # 只保留每个光谱一行（取每个spectrum_id的第一行即可）
        train_info = self.tokenized_dataset.groupby("spectrum_id").first().reset_index()
        val_info = self.tokenized_dataset.groupby("spectrum_id").first().reset_index()
        info_cols = [c for c in ["seq", "teff", "logg", "feh", "SNR", "path"] if c in self.tokenized_dataset.columns]
        train_info[info_cols].to_csv("train_info.csv", index=False)
        val_info[info_cols].to_csv("val_info.csv", index=False)
        print(f"   训练集: {len(train_spec_ids)} 光谱 ({len(self.train_df):,} 数据点)")
        print(f"   验证集: {len(val_spec_ids)} 光谱 ({len(self.val_df):,} 数据点)")

    def save_vocabulary(self):
        """
        统计所有token，生成词汇表并保存为csv文件。
        若已存在 token_vocabulary.csv，则不覆盖，避免与固定ID词表不一致。
        """
        # 若存在固定词表，则不再生成临时词表
        if os.path.exists("vocabulary.csv"):
            print(f"\n💾 检测到固定词表 vocabulary.csv，保持不变（不覆盖）。")
            return
        if os.path.exists("token_vocabulary.csv"):
            print(f"\n💾 检测到已有 token_vocabulary.csv，保持不变（不覆盖）。")
            return
        print(f"\n💾 生成并保存token词汇表...")
        # 统计所有token列
        token_cols = [
            col
            for col in self.tokenized_dataset.columns
            if col.endswith("_token")
            or col.startswith("Teff_")
            or col.startswith("logg_")
            or col.startswith("FeH_")
            or col.startswith("flux_")
        ]
        # 获取所有唯一token
        tokens = set()
        for col in token_cols:
            tokens.update(self.tokenized_dataset[col].unique())
        # 构建token到id的映射
        token_list = sorted(list(tokens))
        vocab_df = pd.DataFrame(
            {"token": token_list, "token_id": range(len(token_list))}
        )
        vocab_df.to_csv("token_vocabulary.csv", index=False)
        print(f"✅ 词汇表保存完成，共{len(token_list)}个token。")

    def show_samples(self):
        """
        打印部分样本，便于人工检查token化效果。
        """
        print(f"\n🔍 样本展示（前5条）：")
        print(self.tokenized_dataset.head(5))


def main():
    """
    数据预处理主入口函数。
    解析命令行参数，初始化预处理器并运行完整流程。
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="光谱数据预处理器")
    parser.add_argument(
        "--data_dir",
        default="/home/share/guofangkeda/wangcunshi/Spec/R1800FITS/SNR1",
        help="数据目录路径",
    )
    parser.add_argument(
        "--fits_root",
        default="/home/share/guofangkeda/wangcunshi/Spec/R1800FITS",
        help="R1800FITS 根目录（启用 --make_groups 时生效）",
    )
    parser.add_argument(
        "--pretrain_snrs",
        default="SNR10,SNR20",
        help="预训练 SNR 列表，逗号分隔（启用 --make_groups 时生效）",
    )
    parser.add_argument(
        "--finetune_snrs",
        default="SNR1",
        help="微调 SNR 列表，逗号分隔（启用 --make_groups 时生效）",
    )
    parser.add_argument(
        "--make_groups",
        action="store_true",
        help="按 SNR 分组生成 pretrain/finetune 两个CSV（保持原有逻辑不变）",
    )
    parser.add_argument("--processes", type=int, help="并行进程数")
    parser.add_argument("--test", action="store_true", help="测试模式")
    parser.add_argument("--max_files", type=int, help="最大文件数")
    parser.add_argument("--max_pixels", type=int, help="每条光谱最大像素数")
    parser.add_argument(
        "--emit_vocab",
        action="store_true",
        help="统计本次数据中出现过的token并输出为独立CSV（不覆盖token_vocabulary.csv）",
    )
    parser.add_argument(
        "--emit_vocab_counts",
        action="store_true",
        help="额外输出频次（默认只输出唯一token以提速）",
    )
    parser.add_argument(
        "--vocab_out",
        default=None,
        help="词表输出路径（可选）。若未指定，将按数据集自动命名到对应目录。",
    )
    parser.add_argument("--batch_gb", type=float, default=2.0, help="按GB分批（默认2GB）")
    parser.add_argument("--batch_files", type=int, default=0, help="按文件数分批（>0优先于按GB）")
    parser.add_argument("--resume", action="store_true", help="续写已有CSV并延续 spectrum_id")
    args = parser.parse_args()

    # 组装配置字典
    config = {
        "data_dir": args.data_dir,
        "n_processes": args.processes,
        "test_mode": args.test,
        "max_files": args.max_files,
        "max_pixels": args.max_pixels,
    }

    # 初始化预处理器
    preprocessor = SpectrumPreprocessor(
        data_dir=config["data_dir"],
        n_processes=config["n_processes"],
        test_mode=config["test_mode"],
        max_files=config["max_files"],
        max_pixels=config["max_pixels"],
    )
    
    # 工具函数：从DataFrame统计出现过的token，并输出到CSV
    def emit_vocab_from_df(df: pd.DataFrame, out_path: str, with_counts: bool = False):
        token_cols = [
            col
            for col in df.columns
            if col.endswith("_token")
            or col.startswith("Teff_")
            or col.startswith("logg_")
            or col.startswith("FeH_")
            or col.startswith("flux_")
        ]
        if not token_cols:
            print("⚠️ 未找到token列，跳过词表统计。")
            return
        # 默认仅输出唯一token；可选输出频次
        if with_counts:
            series_list = [df[col] for col in token_cols]
            all_tokens_series = pd.concat(series_list, ignore_index=True)
            counts = all_tokens_series.value_counts()
            vocab_df = counts.reset_index()
            vocab_df.columns = ["token", "count"]
        else:
            unique_tokens = set()
            for col in token_cols:
                unique_tokens.update(df[col].unique())
            vocab_df = pd.DataFrame({"token": sorted(unique_tokens)})
        # 确保输出目录存在
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        vocab_df.to_csv(out_path, index=False)
        if with_counts:
            print(f"📝 词表已输出: {out_path}（共 {len(vocab_df)} 个token，含频次）")
        else:
            print(f"📝 词表已输出: {out_path}（共 {len(vocab_df)} 个唯一token）")

    # 带进度条的CSV分块写出
    def write_csv_with_progress(df: pd.DataFrame, out_path: str, chunk_rows: int = 1000000):
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        total_rows = len(df)
        if total_rows == 0:
            pd.DataFrame(columns=df.columns).to_csv(out_path, index=False)
            print(f"✅ 保存 {out_path} (0 行)")
            return
        # 先写表头
        header_written = False
        with tqdm(total=total_rows, desc=f"💾 写出 {os.path.basename(out_path)}", unit="row", smoothing=0.1) as pbar:
            for start in range(0, total_rows, chunk_rows):
                end = min(start + chunk_rows, total_rows)
                df.iloc[start:end].to_csv(out_path, index=False, mode=('w' if not header_written else 'a'), header=not header_written)
                header_written = True
                pbar.update(end - start)
        print(f"✅ 保存 {out_path} ({total_rows:,} 行)")

    # 新模式：流式分批处理，避免内存爆炸
    if args.make_groups:
        # 尝试加载固定词表（用于校验）
        fixed_vocab_tokens = load_vocab_tokens("vocabulary.csv")
        pre_snrs = [s.strip() for s in args.pretrain_snrs.split(',') if s.strip()]
        ft_snrs = [s.strip() for s in args.finetune_snrs.split(',') if s.strip()]

        # --- 1) 预训练：流式写 full/train/val 三文件 ---
        if pre_snrs:
            all_pre = collect_csv_files_by_snrs(args.fits_root, pre_snrs)
            if args.max_files:
                all_pre = all_pre[:args.max_files]
            if not all_pre:
                print("❌ 未找到预训练文件，跳过。")
            else:
                os.makedirs('pretrain_data', exist_ok=True)
                full_path = 'pretrain_data/spectrum_tokenized_full.csv'
                train_path = 'pretrain_data/spectrum_tokenized_train.csv'
                val_path = 'pretrain_data/spectrum_tokenized_val.csv'

                # 续写/覆盖模式
                if args.resume:
                    full_header_written = os.path.exists(full_path) and os.path.getsize(full_path) > 0
                    train_header_written = os.path.exists(train_path) and os.path.getsize(train_path) > 0
                    val_header_written = os.path.exists(val_path) and os.path.getsize(val_path) > 0
                    next_spec_id = get_next_spec_id_from_csv(full_path)
                    print(f"⏯️ 续写模式：从 spectrum_id={next_spec_id} 继续")
                else:
                    for p in [full_path, train_path, val_path]:
                        try:
                            if os.path.exists(p):
                                os.remove(p)
                        except Exception:
                            pass
                    full_header_written = train_header_written = val_header_written = False
                    next_spec_id = 0

                # 可选词表统计（流式累计）
                seen_tokens = set() if args.emit_vocab and not args.emit_vocab_counts else None
                token_counter = Counter() if args.emit_vocab and args.emit_vocab_counts else None

                # 批次生成器
                batch_iter = (
                    batch_files_by_count(all_pre, args.batch_files)
                    if args.batch_files and args.batch_files > 0
                    else batch_files_by_size(all_pre, batch_size_gb=args.batch_gb)
                )

                for bi, file_batch in enumerate(batch_iter):
                    print(f"🔄 预训练 批次 {bi+1}: {len(file_batch)} 文件")
                    with mp.Pool(config['n_processes']) as pool:
                        spectra_batch = [r for r in pool.map(load_single_spectrum, file_batch) if r]

                    # 稳定划分 train/val（按路径哈希，9:1）
                    train_ids, val_ids = set(), set()
                    for local_idx, spec in enumerate(spectra_batch):
                        sid = next_spec_id + local_idx
                        path = spec.get('filepath', str(sid))
                        h = hashlib.md5(path.encode('utf-8')).hexdigest()
                        bucket = int(h[:8], 16) % 100
                        (val_ids if bucket < 10 else train_ids).add(sid)

                    # 逐像素token化（全局唯一ID）
                    args_list = [(spec, next_spec_id + idx, args.max_pixels) for idx, spec in enumerate(spectra_batch)]
                    all_rows = []
                    with mp.Pool(config['n_processes']) as pool:
                        for chunk in pool.imap_unordered(process_spectrum_chunk, args_list):
                            all_rows.extend(chunk)
                    df_batch = pd.DataFrame(all_rows, columns=OUTPUT_COLUMNS)
                    # 词表一致性校验（可选）
                    if fixed_vocab_tokens is not None and not df_batch.empty:
                        validate_df_tokens_against_vocab(df_batch, fixed_vocab_tokens)

                    # 1) 追加 full.csv
                    df_batch.to_csv(full_path, index=False, mode=('a' if full_header_written else 'w'), header=(not full_header_written))
                    full_header_written = True

                    # 2) 追加 train/val
                    if train_ids:
                        df_tr = df_batch[df_batch['spectrum_id'].isin(train_ids)]
                        if not df_tr.empty:
                            df_tr.to_csv(train_path, index=False, mode=('a' if train_header_written else 'w'), header=(not train_header_written))
                            train_header_written = True
                    if val_ids:
                        df_vl = df_batch[df_batch['spectrum_id'].isin(val_ids)]
                        if not df_vl.empty:
                            df_vl.to_csv(val_path, index=False, mode=('a' if val_header_written else 'w'), header=(not val_header_written))
                            val_header_written = True

                    # 3) 累计词表（可选）
                    if args.emit_vocab:
                        token_cols = [c for c in df_batch.columns if c.endswith('_token') or c.startswith('Teff_') or c.startswith('logg_') or c.startswith('FeH_') or c.startswith('flux_')]
                        if args.emit_vocab_counts:
                            for c in token_cols:
                                vc = df_batch[c].value_counts()
                                token_counter.update(vc.to_dict())
                        else:
                            for c in token_cols:
                                seen_tokens.update(df_batch[c].dropna().unique())

                    # 更新ID并释放内存
                    next_spec_id += len(spectra_batch)
                    del df_batch, all_rows, spectra_batch

                # 输出词表
                if args.emit_vocab:
                    out_path = args.vocab_out or 'pretrain_data/token_vocabulary_seen.csv'
                    os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
                    if args.emit_vocab_counts:
                        vocab_df = pd.DataFrame({'token': list(token_counter.keys()), 'count': list(token_counter.values())})
                        vocab_df = vocab_df.sort_values(by=['token']).reset_index(drop=True)
                        vocab_df.to_csv(out_path, index=False)
                        print(f"📝 词表已输出: {out_path}（共 {len(vocab_df)} 个token，含频次）")
                    else:
                        vocab_df = pd.DataFrame({'token': sorted(list(seen_tokens))})
                        vocab_df.to_csv(out_path, index=False)
                        print(f"📝 词表已输出: {out_path}（共 {len(vocab_df)} 个唯一token）")

        # --- 2) 微调：与预训练一致的流式三文件输出（full/train/val）+ 词表 ---
        if ft_snrs:
            all_ft = collect_csv_files_by_snrs(args.fits_root, ft_snrs)
            if args.max_files:
                all_ft = all_ft[:args.max_files]
            if not all_ft:
                print('ℹ️ 微调集合为空，跳过微调CSV输出。')
            else:
                # 输出目录与文件
                ft_dir = 'finetune_data'
                os.makedirs(ft_dir, exist_ok=True)
                ft_full_path = os.path.join(ft_dir, 'spectrum_tokenized_full.csv')
                ft_train_path = os.path.join(ft_dir, 'spectrum_tokenized_train.csv')
                ft_val_path = os.path.join(ft_dir, 'spectrum_tokenized_val.csv')

                # 续写/覆盖模式（与预训练一致）
                if args.resume:
                    ft_full_header_written = os.path.exists(ft_full_path) and os.path.getsize(ft_full_path) > 0
                    ft_train_header_written = os.path.exists(ft_train_path) and os.path.getsize(ft_train_path) > 0
                    ft_val_header_written = os.path.exists(ft_val_path) and os.path.getsize(ft_val_path) > 0
                    next_ft_id = get_next_spec_id_from_csv(ft_full_path)
                    print(f"⏯️ 微调续写：从 spectrum_id={next_ft_id} 继续")
                else:
                    for p in [ft_full_path, ft_train_path, ft_val_path]:
                        try:
                            if os.path.exists(p):
                                os.remove(p)
                        except Exception:
                            pass
                    ft_full_header_written = ft_train_header_written = ft_val_header_written = False
                    next_ft_id = 0

                # 词表累计（可选）
                seen_tokens_ft = set() if args.emit_vocab and not args.emit_vocab_counts else None
                token_counter_ft = Counter() if args.emit_vocab and args.emit_vocab_counts else None

                # 批次生成器
                batch_iter_ft = (
                    batch_files_by_count(all_ft, args.batch_files)
                    if args.batch_files and args.batch_files > 0
                    else batch_files_by_size(all_ft, batch_size_gb=args.batch_gb)
                )

                for bi, file_batch in enumerate(batch_iter_ft):
                    print(f"🔄 微调 批次 {bi+1}: {len(file_batch)} 文件")
                    with mp.Pool(config['n_processes']) as pool:
                        spectra_batch = [r for r in pool.map(load_single_spectrum, file_batch) if r]

                    # 与预训练一致的稳定 9:1 划分（按路径哈希）
                    train_ids_ft, val_ids_ft = set(), set()
                    for local_idx, spec in enumerate(spectra_batch):
                        sid = next_ft_id + local_idx
                        path = spec.get('filepath', str(sid))
                        h = hashlib.md5(path.encode('utf-8')).hexdigest()
                        bucket = int(h[:8], 16) % 100
                        (val_ids_ft if bucket < 10 else train_ids_ft).add(sid)

                    # 逐像素 token 化（全局唯一ID）
                    args_list = [(spec, next_ft_id + idx, args.max_pixels) for idx, spec in enumerate(spectra_batch)]
                    all_rows = []
                    with mp.Pool(config['n_processes']) as pool:
                        for chunk in pool.imap_unordered(process_spectrum_chunk, args_list):
                            all_rows.extend(chunk)
                    df_batch = pd.DataFrame(all_rows, columns=OUTPUT_COLUMNS)
                    # 词表一致性校验（可选）
                    if fixed_vocab_tokens is not None and not df_batch.empty:
                        validate_df_tokens_against_vocab(df_batch, fixed_vocab_tokens)

                    # 1) 追加 full
                    df_batch.to_csv(ft_full_path, index=False, mode=('a' if ft_full_header_written else 'w'), header=(not ft_full_header_written))
                    ft_full_header_written = True

                    # 2) 追加 train/val
                    if train_ids_ft:
                        df_tr = df_batch[df_batch['spectrum_id'].isin(train_ids_ft)]
                        if not df_tr.empty:
                            df_tr.to_csv(ft_train_path, index=False, mode=('a' if ft_train_header_written else 'w'), header=(not ft_train_header_written))
                            ft_train_header_written = True
                    if val_ids_ft:
                        df_vl = df_batch[df_batch['spectrum_id'].isin(val_ids_ft)]
                        if not df_vl.empty:
                            df_vl.to_csv(ft_val_path, index=False, mode=('a' if ft_val_header_written else 'w'), header=(not ft_val_header_written))
                            ft_val_header_written = True

                    # 3) 累计词表（可选）
                    if args.emit_vocab:
                        token_cols = [c for c in df_batch.columns if c.endswith('_token') or c.startswith('Teff_') or c.startswith('logg_') or c.startswith('FeH_') or c.startswith('flux_')]
                        if args.emit_vocab_counts:
                            for c in token_cols:
                                vc = df_batch[c].value_counts()
                                token_counter_ft.update(vc.to_dict())
                        else:
                            for c in token_cols:
                                seen_tokens_ft.update(df_batch[c].dropna().unique())

                    next_ft_id += len(spectra_batch)
                    del df_batch, all_rows, spectra_batch

                # 输出词表（默认到 finetune_data/）
                if args.emit_vocab:
                    out_path_ft = args.vocab_out or os.path.join(ft_dir, 'token_vocabulary_seen.csv')
                    os.makedirs(os.path.dirname(out_path_ft), exist_ok=True) if os.path.dirname(out_path_ft) else None
                    if args.emit_vocab_counts:
                        vocab_df = pd.DataFrame({'token': list(token_counter_ft.keys()), 'count': list(token_counter_ft.values())})
                        vocab_df = vocab_df.sort_values(by=['token']).reset_index(drop=True)
                        vocab_df.to_csv(out_path_ft, index=False)
                        print(f"📝 微调词表已输出: {out_path_ft}（共 {len(vocab_df)} 个token，含频次）")
                    else:
                        vocab_df = pd.DataFrame({'token': sorted(list(seen_tokens_ft))})
                        vocab_df.to_csv(out_path_ft, index=False)
                        print(f"📝 微调词表已输出: {out_path_ft}（共 {len(vocab_df)} 个唯一token）")
    else:
        # 旧模式：保持不变
        preprocessor.run()
        # 普通模式下的词表统计（基于 full.csv 的DataFrame），不覆盖固定词表
        if args.emit_vocab and preprocessor.full_df is not None:
            out_path = (
                args.vocab_out if args.vocab_out is not None else 'token_vocabulary_seen.csv'
            )
            emit_vocab_from_df(preprocessor.full_df, out_path, with_counts=args.emit_vocab_counts)


# 脚本入口
if __name__ == "__main__":
    main()
