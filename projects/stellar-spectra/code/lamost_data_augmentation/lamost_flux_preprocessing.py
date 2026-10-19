                                                                       #!/usr/bin/env python3
"""
lamost_flux_preprocessing.py

任务：
- 仅处理 flux，将每条光谱的每个像素转为一行并进行位置编码（Thu/Hun/Ten/One）。
- 不分 train/val，不输出任何恒星参数信息。
- 不从文件名读取参数；只记录来源（obsid 与原始文件路径）。
- 支持大规模目录的流式处理：按文件分批写入，避免一次性占用内存。

输出：
- 单一 CSV：默认 lamost_flux_tokenized_full.csv
  列：obsid, pixel_idx, flux_thu, flux_hun, flux_ten, flux_one, BOS_token, EOS_token, SEP_token, path

使用示例：
python lamost_flux_preprocessing.py \
  --data_dir /home/share/guofangkeda/wangcunshi/Spectrum/LAMOST_LRS/star_low_SN \
  --output lamost_flux_tokenized_full.csv \
  --processes 8 --batch_files 500 --max_files 3000 --max_pixels 1024
"""

import argparse
import os
import sys
from typing import Generator, Iterable, List, Optional, Set
import multiprocessing as mp
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm


# 固定输出列顺序（仅 flux + 源信息）
OUTPUT_COLUMNS = [
    "obsid", "augmentation_id", "pixel_idx",
    "flux_thu", "flux_hun", "flux_ten", "flux_one",
    "BOS_token", "EOS_token", "SEP_token",
]


def list_csv_files_stream(data_dir: str) -> Generator[str, None, None]:
    """惰性遍历目录下的一层 CSV 文件（不递归）。"""
    try:
        with os.scandir(data_dir) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith(".csv"):
                    yield entry.path
    except FileNotFoundError:
        return


def batch_iterable(items: Iterable[str], batch_count: int) -> Generator[List[str], None, None]:
    """按固定文件数进行分批。如果 batch_count <= 0，则整体一批。"""
    if not batch_count or batch_count <= 0:
        batch = list(items)
        if batch:
            yield batch
        return
    batch: List[str] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_count:
            yield batch
            batch = []
    if batch:
        yield batch


def tokenize_flux_array(
    flux: np.ndarray,
    max_pixels: Optional[int],
    pixel_idx_start: int = 0,
    pixel_idx_step: int = 1,
) -> pd.DataFrame:
    """
    将归一化到 [0,1] 的 flux 向量转为位置编码 token 的 DataFrame（仅像素级）。
    - 缩放到 0..9999，四位拆分为 thu/hun/ten/one
    - 添加 BOS/EOS/SEP 特殊 token
    """
    if max_pixels is not None:
        flux = flux[:max_pixels]

    # 安全裁剪
    flux = np.nan_to_num(flux, nan=0.0, posinf=1.0, neginf=0.0)
    flux = np.clip(flux, 0.0, 1.0)

    scaled = np.clip((flux * 9999.0).astype(int), 0, 9999)
    # 拆位：千/百/十/个
    thu = scaled // 1000
    hun = (scaled // 100) % 10
    ten = (scaled // 10) % 10
    one = scaled % 10

    df = pd.DataFrame({
        "pixel_idx": (pixel_idx_start + pixel_idx_step * np.arange(len(scaled), dtype=int)),
        "flux_thu": [f"S{d}_thu" for d in thu],
        "flux_hun": [f"S{d}_hun" for d in hun],
        "flux_ten": [f"S{d}_ten" for d in ten],
        "flux_one": [f"S{d}_one" for d in one],
        "BOS_token": "<BOS>",
        "EOS_token": "<EOS>",
        "SEP_token": "<SEP>",
    })
    return df


def process_single_file(
    file_path: str,
    max_pixels: Optional[int],
    pixel_idx_start: int,
    pixel_idx_step: int,
) -> Optional[pd.DataFrame]:
    """
    读取单个 CSV（包含一列 flux），生成像素级 token DataFrame，并附 obsid/path。
    返回 None 表示读取失败或数据为空。
    """
    try:
        df = pd.read_csv(file_path, usecols=["flux"])  # 期望列名为 flux
    except Exception:
        return None
    if df.empty or "flux" not in df.columns:
        return None

    flux = df["flux"].to_numpy(dtype=float, copy=False)
    token_df = tokenize_flux_array(
        flux,
        max_pixels=max_pixels,
        pixel_idx_start=pixel_idx_start,
        pixel_idx_step=pixel_idx_step,
    )

    # 解析文件名：支持 obsid_augmentationId.csv 或 obsid.csv
    stem = os.path.splitext(os.path.basename(file_path))[0]
    base_obsid = stem
    aug_id = ""
    if "_" in stem:
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            base_obsid, aug_id = parts[0], parts[1]
    token_df.insert(0, "obsid", str(base_obsid))
    token_df.insert(1, "augmentation_id", str(aug_id))
    return token_df[OUTPUT_COLUMNS]


def load_vocab_tokens(vocab_path: Optional[str]) -> Optional[Set[str]]:
    """
    从给定词表 CSV 读取 token 集合。
    - 优先列名 'token'；否则兜底使用最后一列作为 token 列。
    - 读取失败或未提供路径时返回 None。
    """
    try:
        if not vocab_path:
            return None
        if not os.path.exists(vocab_path):
            return None
        df = pd.read_csv(vocab_path)
        if 'token' in df.columns:
            series = df['token'].astype(str)
        else:
            last_col = df.columns[-1]
            series = df[last_col].astype(str)
        return set(series.tolist())
    except Exception:
        return None


def validate_df_tokens_against_vocab(
    df: pd.DataFrame,
    tokens_set: Set[str],
    warn_only: bool = False,
) -> None:
    """
    对批次 df 的 token 列做一致性校验。
    - 若存在 OOV：warn_only=False 时退出；为 True 时仅警告。
    """
    token_cols = [
        'flux_thu', 'flux_hun', 'flux_ten', 'flux_one',
        'BOS_token', 'EOS_token', 'SEP_token',
    ]
    missing: Set[str] = set()
    for c in token_cols:
        if c in df.columns:
            missing.update(set(pd.Series(df[c]).dropna().unique()) - tokens_set)
    if missing:
        sample = sorted(list(missing))[:20]
        msg = f"❌ 发现不在固定词表中的 token（示例前20个）: {sample}"
        if warn_only:
            print(msg)
        else:
            print(msg)
            print("请先更新/指定包含全部 token 的 vocabulary.csv 后重试。")
            sys.exit(1)


def write_batch(
    df_list: List[pd.DataFrame],
    out_path: str,
    header_written: bool,
    vocab_tokens: Optional[Set[str]] = None,
    warn_only: bool = False,
) -> bool:
    """将一批 DataFrame 追加写入到 CSV，返回是否已经写入表头。"""
    if not df_list:
        return header_written
    merged = pd.concat(df_list, ignore_index=True)
    if vocab_tokens is not None and not merged.empty:
        validate_df_tokens_against_vocab(merged, vocab_tokens, warn_only=warn_only)
    mode = "a" if header_written else "w"
    merged.to_csv(out_path, index=False, mode=mode, header=(not header_written))
    return True


def main():
    parser = argparse.ArgumentParser(description="LAMOST flux 预处理（仅flux，无参数/无切分）")
    parser.add_argument("--data_dir", default="/home/share/guofangkeda/wangcunshi/Spectrum/LAMOST_LRS/star_low_SN", help="包含 obsid.csv 的目录（不递归）")
    parser.add_argument("--output", default="lamost_flux_tokenized_full.csv", help="输出CSV路径")
    parser.add_argument("--processes", type=int, default=0, help="并行进程数（预留，当前顺序处理更省内存）")
    parser.add_argument("--batch_files", type=int, default=100, help="每批处理的文件数量（写一次磁盘）")
    parser.add_argument("--max_files", type=int, default=0, help="最多处理的文件数（0表示全部）")
    parser.add_argument("--max_pixels", type=int, default=0, help="每条光谱最多保留的像素数（0表示全部）")
    parser.add_argument("--overwrite", action="store_true", help="若目标文件存在则覆盖（默认追加写入）")
    parser.add_argument("--vocab_path", default=None, help="固定词表 CSV 路径（启用后将做OOV校验）")
    parser.add_argument("--oov_warn_only", action="store_true", help="OOV仅警告不退出")
    parser.add_argument("--imap_chunksize", type=int, default=64, help="并行imap_unordered的任务分片大小（并行时有效）")
    parser.add_argument("--pixel_idx_start", type=int, default=0, help="pixel_idx 起始值（默认0）")
    parser.add_argument("--pixel_idx_step", type=int, default=1, help="pixel_idx 步长（默认1）")
    args = parser.parse_args()

    data_dir = args.data_dir
    out_path = args.output
    max_pixels = args.max_pixels if args.max_pixels and args.max_pixels > 0 else None
    max_files = args.max_files if args.max_files and args.max_files > 0 else None

    if not os.path.isdir(data_dir):
        print(f"❌ 数据目录不存在或不可访问: {data_dir}")
        sys.exit(1)

    # 输出文件处理
    if os.path.exists(out_path):
        if args.overwrite:
            try:
                os.remove(out_path)
            except Exception:
                pass
            header_written = False
        else:
            header_written = os.path.getsize(out_path) > 0
            print(f"⏯️ 追加写入已存在文件: {out_path}")
    else:
        header_written = False

    # 词表（可选）
    vocab_tokens = load_vocab_tokens(args.vocab_path)
    if args.vocab_path and vocab_tokens is None:
        print(f"⚠️ 未能加载词表或词表不存在：{args.vocab_path}（将不进行校验）")

    # 文件流 + 分批写出
    file_stream = list_csv_files_stream(data_dir)
    if max_files is not None:
        # 惰性截断：将生成器裁剪到 max_files
        def limited_stream(stream: Generator[str, None, None], limit: int):
            count = 0
            for p in stream:
                yield p
                count += 1
                if count >= limit:
                    break
        file_stream = limited_stream(file_stream, max_files)

    total_processed = 0
    desc = "📥 读取并token化 (并行x%d)" % args.processes if (args.processes and args.processes > 0) else "📥 读取并token化"
    with tqdm(desc=desc, unit="file") as pbar:
        df_cache: List[pd.DataFrame] = []
        processed_in_batch = 0
        if args.processes and args.processes > 0:
            # 单个进程池贯穿全程，避免每批重建Pool的开销
            worker = partial(
                process_single_file,
                max_pixels=max_pixels,
                pixel_idx_start=args.pixel_idx_start,
                pixel_idx_step=args.pixel_idx_step,
            )
            with mp.Pool(processes=args.processes) as pool:
                for token_df in pool.imap_unordered(worker, file_stream, chunksize=max(1, args.imap_chunksize)):
                    if token_df is not None and not token_df.empty:
                        df_cache.append(token_df)
                        processed_in_batch += 1
                    pbar.update(1)
                    if processed_in_batch >= args.batch_files:
                        header_written = write_batch(
                            df_cache,
                            out_path,
                            header_written,
                            vocab_tokens=vocab_tokens,
                            warn_only=args.oov_warn_only,
                        )
                        total_processed += processed_in_batch
                        df_cache.clear()
                        processed_in_batch = 0
        else:
            # 顺序处理
            for file_path in file_stream:
                token_df = process_single_file(
                    file_path,
                    max_pixels=max_pixels,
                    pixel_idx_start=args.pixel_idx_start,
                    pixel_idx_step=args.pixel_idx_step,
                )
                if token_df is not None and not token_df.empty:
                    df_cache.append(token_df)
                    processed_in_batch += 1
                pbar.update(1)
                if processed_in_batch >= args.batch_files:
                    header_written = write_batch(
                        df_cache,
                        out_path,
                        header_written,
                        vocab_tokens=vocab_tokens,
                        warn_only=args.oov_warn_only,
                    )
                    total_processed += processed_in_batch
                    df_cache.clear()
                    processed_in_batch = 0

        # flush 余量
        if df_cache:
            header_written = write_batch(
                df_cache,
                out_path,
                header_written,
                vocab_tokens=vocab_tokens,
                warn_only=args.oov_warn_only,
            )
            total_processed += processed_in_batch

    print(f"\n✅ 完成：写出 {out_path}，源光谱数={total_processed}")


if __name__ == "__main__":
    main()


