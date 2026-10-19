#!/usr/bin/env python3
"""
convert_lamost_to_pretrain_format.py
将 LAMOST flux 数据转换为与预训练数据集相同的格式，用于微调

功能：
1. 读取 LAMOST flux tokenized CSV（只有 flux token）
2. 从目录文件获取恒星参数（Teff, logg, FeH）
3. 添加参数 token 化
4. 输出与预训练数据集相同的 20 列格式

使用：
python convert_lamost_to_pretrain_format.py \
  --input lamost_flux_tokenized_full.csv \
  --catalog /path/to/DR12LRS_SNRz0_3.csv \
  --output lamost_pretrain_format.csv \
  --max_rows 1000000
"""

import pandas as pd
import numpy as np
import argparse
import os
import hashlib
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# 与预训练数据集完全一致的20列顺序
OUTPUT_COLUMNS = [
    'spectrum_id', 'pixel_idx',
    'Teff_tthou', 'Teff_thu', 'Teff_hun', 'Teff_ten', 'Teff_one',
    'logg_hun', 'logg_ten', 'logg_one', 'logg_sign',
    'FeH_ten', 'FeH_one', 'FeH_sign',
    'flux_thu', 'flux_hun', 'flux_ten', 'flux_one',
    'BOS_token', 'EOS_token', 'SEP_token'
]

def tokenize_parameters(teff, logg, feh):
    """参数token化（与06_data_preprocessing.py完全一致）"""
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


def load_catalog_parameters(catalog_path):
    """加载参数文件，返回 obsid -> (teff, logg, feh) 的映射"""
    print(f"📂 加载参数文件: {catalog_path}")
    catalog = pd.read_csv(catalog_path, usecols=['obsid', 'teff', 'logg', 'feh'])
    catalog['obsid'] = catalog['obsid'].astype(str)
    
    # 统计有效参数的数量
    valid_params = catalog.dropna(subset=['teff', 'logg', 'feh'])
    print(f"   总记录: {len(catalog):,}")
    print(f"   有效参数: {len(valid_params):,} ({len(valid_params)/len(catalog)*100:.1f}%)")
    
    # 创建映射字典
    param_map = {}
    for _, row in valid_params.iterrows():
        obsid = str(row['obsid'])
        param_map[obsid] = (row['teff'], row['logg'], row['feh'])
    
    print(f"   参数映射创建完成: {len(param_map):,} 个有效映射")
    return param_map


def convert_lamost_to_pretrain_format(input_csv, catalog_path, output_csv, max_rows=None, chunksize=1_000_000):
    """
    将 LAMOST flux 数据转换为预训练格式
    """
    print(f"🔄 开始转换...")
    print(f"   输入: {input_csv}")
    print(f"   输出: {output_csv}")
    
    # 加载参数映射
    param_map = load_catalog_parameters(catalog_path)
    
    # 统计
    total_rows = 0
    converted_rows = 0
    missing_params = set()
    
    # 输出文件初始化
    header_written = False
    
    print(f"\n📥 开始逐块处理...")
    for chunk_idx, chunk in enumerate(pd.read_csv(input_csv, chunksize=chunksize)):
        if max_rows and total_rows >= max_rows:
            break
            
        print(f"   处理块 {chunk_idx + 1}: {len(chunk):,} 行")
        total_rows += len(chunk)
        
        # 转换当前块
        converted_chunk = []
        
        for _, row in chunk.iterrows():
            obsid = str(row['obsid'])
            
            # 检查是否有参数
            if obsid not in param_map:
                missing_params.add(obsid)
                continue
                
            teff, logg, feh = param_map[obsid]
            
            # 参数token化
            param_tokens = tokenize_parameters(teff, logg, feh)
            
            # 构建新行（与预训练格式一致）
            new_row = {
                'spectrum_id': obsid,  # 使用 obsid 作为 spectrum_id
                'pixel_idx': row['pixel_idx'],
                # 参数tokens
                **param_tokens,
                # flux tokens（直接复制）
                'flux_thu': row['flux_thu'],
                'flux_hun': row['flux_hun'],
                'flux_ten': row['flux_ten'],
                'flux_one': row['flux_one'],
                # 特殊tokens
                'BOS_token': row['BOS_token'],
                'EOS_token': row['EOS_token'],
                'SEP_token': row['SEP_token'],
            }
            
            converted_chunk.append(new_row)
            converted_rows += 1
            
            if max_rows and converted_rows >= max_rows:
                break
        
        # 写入当前块
        if converted_chunk:
            chunk_df = pd.DataFrame(converted_chunk)[OUTPUT_COLUMNS]
            mode = 'w' if not header_written else 'a'
            chunk_df.to_csv(output_csv, index=False, mode=mode, header=not header_written)
            header_written = True
        
        if max_rows and converted_rows >= max_rows:
            break
    
    print(f"\n✅ 转换完成!")
    print(f"   处理行数: {total_rows:,}")
    print(f"   转换行数: {converted_rows:,}")
    print(f"   缺失参数的 obsid: {len(missing_params):,}")
    print(f"   转换率: {converted_rows/total_rows*100:.1f}%")
    
    if missing_params and len(missing_params) <= 10:
        print(f"   缺失参数示例: {list(missing_params)[:10]}")
    
    return converted_rows


def _stable_shard_index(spectrum_id: str, num_shards: int, seed: int) -> int:
    # 稳定哈希（加入种子，避免文件名分布偏差），取 md5 前8位提升速度
    h = hashlib.md5((str(spectrum_id) + f"#{seed}").encode("utf-8")).hexdigest()
    return int(h[:8], 16) % max(1, num_shards)


def _append_file_to_file(src_path: str, dst_path: str, write_header: bool) -> bool:
    """将 src_path 追加到 dst_path。若 write_header=False，则跳过首行表头。返回是否已写过表头。"""
    if not os.path.exists(src_path) or os.path.getsize(src_path) == 0:
        return write_header
    with open(src_path, 'r', encoding='utf-8') as src, open(dst_path, 'a', encoding='utf-8') as dst:
        first = True
        for line in src:
            if not write_header and first:
                first = False
                continue
            dst.write(line)
    return True


def shuffle_and_split_data(
    input_csv,
    output_dir,
    test_size=0.1,
    random_state=42,
    num_shards: int = 64,
    expected_pixels: int = 303,
):
    """
    按光谱为单位划分 train/val，并进行光谱级乱序写出（保证单条光谱内部连续且 pixel_idx 顺序不变）。
    为提升效率，采用分片（shard）追加 + 最终按随机分片顺序合并的方式，避免全量加载到内存。
    """
    print(f"\n🔀 开始数据打乱和划分...")
    print(f"   输入文件: {input_csv}")
    print(f"   输出目录: {output_dir}")
    print(f"   验证集比例: {test_size}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 首先获取所有唯一的 spectrum_id
    print("   📊 统计光谱数量...")
    unique_spectrum_ids = set()
    for chunk in pd.read_csv(
        input_csv,
        usecols=['spectrum_id'],
        dtype={'spectrum_id': str},
        chunksize=1_000_000,
    ):
        unique_spectrum_ids.update(chunk['spectrum_id'].unique())
    
    spectrum_ids = list(unique_spectrum_ids)
    print(f"   总光谱数: {len(spectrum_ids):,}")
    
    # 按光谱ID进行9:1划分（确保同一光谱的所有像素都在同一集合中）
    train_ids, val_ids = train_test_split(
        spectrum_ids, 
        test_size=test_size, 
        random_state=random_state,
        shuffle=True
    )
    
    train_ids_set = set(train_ids)
    val_ids_set = set(val_ids)
    
    print(f"   训练集光谱数: {len(train_ids):,}")
    print(f"   验证集光谱数: {len(val_ids):,}")
    
    # 输出文件路径
    train_path = os.path.join(output_dir, 'spectrum_tokenized_train.csv')
    val_path = os.path.join(output_dir, 'spectrum_tokenized_val.csv')
    full_path = os.path.join(output_dir, 'spectrum_tokenized_full.csv')
    
    # 分片临时目录
    os.makedirs(output_dir, exist_ok=True)
    shards_dir = os.path.join(output_dir, "_shuffle_shards")
    os.makedirs(shards_dir, exist_ok=True)

    # 记录每个分片是否已写过表头
    train_shard_header = [False] * num_shards
    val_shard_header = [False] * num_shards

    train_rows = 0
    val_rows = 0
    total_rows = 0

    print("   📝 按光谱分片写入（流式，保证单光谱连续）...")
    for chunk in tqdm(
        pd.read_csv(
            input_csv,
            chunksize=500_000,
            dtype={'spectrum_id': str},
        ),
        desc="处理数据块",
    ):
        total_rows += len(chunk)

        # 仅保留在 train/val 中的行
        chunk = chunk[chunk['spectrum_id'].isin(train_ids_set | val_ids_set)]
        if chunk.empty:
            continue

        # 跨块连续性：对每个光谱累积到 expected_pixels 行再写出
        # 使用静态属性在多次调用中保存 carry（函数闭包外部定义一次）
        if not hasattr(shuffle_and_split_data, "_carry"):
            shuffle_and_split_data._carry = {}
        carry = shuffle_and_split_data._carry  # dict: sid -> DataFrame

        # 分片缓冲，减少磁盘写次数
        shard_buffers_train = {}
        shard_buffers_val = {}

        grouped = chunk.groupby('spectrum_id', sort=False)
        for sid, g in grouped:
            sid_str = str(sid)
            df = g
            if sid_str in carry:
                df = pd.concat([carry[sid_str], df], ignore_index=True)
                del carry[sid_str]

            # 由于输入来自第一步，理论上不会超过 expected_pixels；
            # 若 chunk 边界导致累计达到 expected_pixels，则写出；不足则缓存等待下一块。
            while len(df) >= expected_pixels:
                complete = df.iloc[:expected_pixels].copy()
                df = df.iloc[expected_pixels:]
                shard_idx = _stable_shard_index(sid_str, num_shards, random_state)
                if sid_str in train_ids_set:
                    shard_buffers_train.setdefault(shard_idx, []).append(complete)
                else:
                    shard_buffers_val.setdefault(shard_idx, []).append(complete)
                if df.empty:
                    break
            if not df.empty:
                carry[sid_str] = df

        # 将分片缓冲落盘
        def flush_buffers(buffers, prefix, header_flags):
            written = 0
            for shard_idx, dfs in buffers.items():
                shard_df = pd.concat(dfs, ignore_index=True)
                shard_path = os.path.join(shards_dir, f"{prefix}_shard_{shard_idx:03d}.csv")
                shard_df.to_csv(
                    shard_path,
                    index=False,
                    mode='a' if header_flags[shard_idx] else 'w',
                    header=not header_flags[shard_idx]
                )
                header_flags[shard_idx] = True
                written += len(shard_df)
            return written

        train_rows += flush_buffers(shard_buffers_train, "train", train_shard_header)
        val_rows += flush_buffers(shard_buffers_val, "val", val_shard_header)

    # 合并分片为最终文件，按随机分片顺序保证光谱级乱序
    rnd = random.Random(random_state)
    shard_order = list(range(num_shards))
    rnd.shuffle(shard_order)

    # 清理旧文件
    for p in [train_path, val_path]:
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

    train_header_written = False
    val_header_written = False
    for idx in shard_order:
        t_shard = os.path.join(shards_dir, f"train_shard_{idx:03d}.csv")
        v_shard = os.path.join(shards_dir, f"val_shard_{idx:03d}.csv")
        if os.path.exists(t_shard):
            train_header_written = _append_file_to_file(t_shard, train_path, write_header=train_header_written)
        if os.path.exists(v_shard):
            val_header_written = _append_file_to_file(v_shard, val_path, write_header=val_header_written)

    # 若存在未完结的光谱（异常），尝试写出并告警
    if hasattr(shuffle_and_split_data, "_carry") and shuffle_and_split_data._carry:
        orphan = list(shuffle_and_split_data._carry.keys())[:5]
        print(f"⚠️ 发现未完整的光谱（数量={len(shuffle_and_split_data._carry)}，示例={orphan}），已忽略其残余行。")
        shuffle_and_split_data._carry.clear()

    # 生成 full：简单合并 train 与 val（不强制全局乱序）
    try:
        if os.path.exists(full_path):
            os.remove(full_path)
    except Exception:
        pass
    # 先写 train，后写 val（不写第二个表头）
    _ = _append_file_to_file(train_path, full_path, write_header=False)
    _ = _append_file_to_file(val_path, full_path, write_header=True)

    # 删除分片文件
    for idx in range(num_shards):
        for prefix in ("train", "val"):
            fp = os.path.join(shards_dir, f"{prefix}_shard_{idx:03d}.csv")
            try:
                if os.path.exists(fp):
                    os.remove(fp)
            except Exception:
                pass
    try:
        os.rmdir(shards_dir)
    except Exception:
        pass
    
    print(f"\n✅ 数据划分完成!")
    print(f"   训练集: {train_path} ({train_rows:,} 行)")
    print(f"   验证集: {val_path} ({val_rows:,} 行)")
    print(f"   完整集: {full_path} ({total_rows:,} 行)")
    print(f"   训练集比例: {train_rows/total_rows*100:.1f}%")
    print(f"   验证集比例: {val_rows/total_rows*100:.1f}%")
    
    return train_rows, val_rows

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='将 LAMOST flux 数据转换为预训练格式')
    parser.add_argument('--input', required=True, help='输入的 LAMOST flux tokenized CSV')
    parser.add_argument('--catalog', required=True, help='参数 CSV 文件（含 obsid, teff, logg, feh）')
    parser.add_argument('--output', required=True, help='输出的预训练格式 CSV（临时文件）')
    parser.add_argument('--output_dir', default='finetune_data', help='最终输出目录（包含train/val/full）')
    parser.add_argument('--max_rows', type=int, default=None, help='最大处理行数（用于测试）')
    parser.add_argument('--chunksize', type=int, default=100_000, help='分块大小')
    parser.add_argument('--test_size', type=float, default=0.1, help='验证集比例（默认0.1即9:1划分）')
    parser.add_argument('--random_seed', type=int, default=42, help='随机种子')
    parser.add_argument('--skip_split', action='store_true', help='跳过数据划分，只进行格式转换')
    
    args = parser.parse_args()
    
    print("🚀 LAMOST 数据转换为预训练格式")
    print("=" * 60)
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"❌ 输入文件不存在: {args.input}")
        return
    
    if not os.path.exists(args.catalog):
        print(f"❌ 目录文件不存在: {args.catalog}")
        return
    
    # 执行转换
    converted_rows = convert_lamost_to_pretrain_format(
        args.input, 
        args.catalog, 
        args.output, 
        max_rows=args.max_rows,
        chunksize=args.chunksize
    )
    
    if converted_rows > 0:
        print(f"\n📊 格式转换完成:")
        print(f"   临时文件: {os.path.abspath(args.output)}")
        print(f"   转换行数: {converted_rows:,}")
        print(f"   列格式: {len(OUTPUT_COLUMNS)} 列（与预训练数据集一致）")
        
        # 显示前几行作为验证
        try:
            sample_df = pd.read_csv(args.output, nrows=3)
            print(f"\n📋 转换样本（前3行）:")
            print(sample_df.to_string(index=False))
        except Exception as e:
            print(f"⚠️ 无法读取输出文件样本: {e}")
        
        # 进行数据打乱和划分（除非跳过）
        if not args.skip_split:
            train_rows, val_rows = shuffle_and_split_data(
                args.output, 
                args.output_dir, 
                test_size=args.test_size,
                random_state=args.random_seed
            )
            
            # 删除临时文件
            try:
                os.remove(args.output)
                print(f"   🗑️ 已删除临时文件: {args.output}")
            except Exception as e:
                print(f"   ⚠️ 删除临时文件失败: {e}")
            
            print(f"\n🎉 全部完成!")
            print(f"   📁 输出目录: {os.path.abspath(args.output_dir)}")
            print(f"   📊 训练集: spectrum_tokenized_train.csv ({train_rows:,} 行)")
            print(f"   📊 验证集: spectrum_tokenized_val.csv ({val_rows:,} 行)")
            print(f"   📊 完整集: spectrum_tokenized_full.csv ({train_rows + val_rows:,} 行)")
        else:
            print(f"\n🎉 格式转换完成!")
            print(f"   📁 输出文件: {os.path.abspath(args.output)}")
    else:
        print(f"\n❌ 没有转换任何数据，请检查输入文件和参数文件的匹配情况")

if __name__ == "__main__":
    main()
