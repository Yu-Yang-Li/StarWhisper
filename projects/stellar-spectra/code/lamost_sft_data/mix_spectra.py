#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
按如下规则混合超大CSV光谱数据：
1) A源：每个完整光谱为连续3030行，且这3030行的`spectrumid`相同；
2) B源：由B1和B2两个CSV顺序组成（先读完B1再读B2）；
3) 输出顺序：每4个来自A的完整光谱后，插入1个来自B的完整光谱；
4) 为避免ID冲突，写出B来源光谱时，将`spectrumid`整体加10000000；
5) 当累计写出约22万条完整光谱后，自动切分输出一个新的CSV文件，并打印提示；
6) 直到A读完为止。

为保证内存占用稳定，采用流式逐行读取，按光谱（3030行）为单位写出。
"""

import csv
import os
import sys
import io
import threading
import queue
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor


DEFAULT_A_PATH = \
    "/home/share/guofangkeda/wangcunshi/Spectrum/Spectrum copy/finetune2/spectrum_tokenized_val_shuffled_global.csv"
DEFAULT_B1_PATH = \
    "/home/share/guofangkeda/wangcunshi/Spectrum/Spectrum copy/finetune_data/spectrum_tokenized_val_shuffled_global_partA.csv"
DEFAULT_B2_PATH = \
    "/home/share/guofangkeda/wangcunshi/Spectrum/Spectrum copy/finetune_data/spectrum_tokenized_val_shuffled_global_partB.csv"
DEFAULT_OUTPUT_DIR = \
    "/home/share/guofangkeda/wangcunshi/Spectrum/Spectrum copy/finetune_mix"


# 允许通过环境变量覆盖每个光谱的行数（默认3030）。
ROWS_PER_SPECTRUM = int(os.environ.get("MIX_ROWS_PER_SPECTRUM", "3030"))
B_ID_OFFSET = 10_000_000
PART_SPECTRA_THRESHOLD = 220_000  # 约22万条完整光谱/文件


class EOFErrorIncompleteSpectrum(RuntimeError):
    pass


def detect_spectrumid_index(header: List[str]) -> int:
    """返回`spectrumid`列索引；支持常见命名变体。"""
    candidates = ["spectrumid", "spectrum_id", "spectrumId", "id"]
    lower_header = [h.strip().lower() for h in header]
    for candidate in candidates:
        if candidate.lower() in lower_header:
            return lower_header.index(candidate.lower())
    raise ValueError(
        f"未在表头中找到spectrumid列，表头字段有：{header}"
    )

def detect_pixelidx_index(header: List[str]) -> Optional[int]:
    """返回像素索引列索引；若未找到则返回None。"""
    candidates = ["pixel_idx", "pixelindex", "pixel", "pixel_id", "pix", "pix_idx"]
    lower_header = [h.strip().lower() for h in header]
    for candidate in candidates:
        if candidate in lower_header:
            return lower_header.index(candidate)
    return None

def read_header(reader: csv.reader) -> List[str]:
    try:
        return next(reader)
    except StopIteration:
        raise ValueError("CSV为空，缺少表头")


def parse_header_from_line(header_line: str) -> List[str]:
    """从单行文本解析CSV表头。"""
    for row in csv.reader([header_line]):
        return row
    raise ValueError("无法解析表头行")


def read_next_spectrum(
    reader: csv.reader,
    spectrumid_index: int,
    source_name: str,
    pixelidx_index: Optional[int] = None,
    pixel_step: Optional[int] = None,
    pixel_start: Optional[int] = None,
) -> Optional[List[List[str]]]:
    """从reader中读取下一个完整光谱（3030行）。

    返回：
        - 3030行的二维数组（每行是list[str]），若到达文件末尾则返回None。
    约束：
        - 若在未满3030行就遇到EOF，将抛出EOFErrorIncompleteSpectrum，提示数据不完整。
        - 校验这3030行的spectrumid一致，否则抛出ValueError。
    """
    first_row: Optional[List[str]] = None
    try:
        first_row = next(reader)
    except StopIteration:
        return None  # 正常EOF

    spectrum_rows: List[List[str]] = [first_row]
    spectrum_id_value = first_row[spectrumid_index]

    # 继续读取剩余的ROWS_PER_SPECTRUM-1行
    for _ in range(ROWS_PER_SPECTRUM - 1):
        try:
            row = next(reader)
        except StopIteration:
            raise EOFErrorIncompleteSpectrum(
                f"{source_name} 在读取完整光谱时意外到达文件末尾（可能数据不完整）。"
            )
        spectrum_rows.append(row)

    # 校验ID一致性
    for row in spectrum_rows:
        if row[spectrumid_index] != spectrum_id_value:
            raise ValueError(
                f"{source_name} 检测到一个光谱的{ROWS_PER_SPECTRUM}行中spectrumid不一致："
                f"首行为{spectrum_id_value}，出现{row[spectrumid_index]}"
            )

    # 可选：校验像素索引步长与起始值
    if pixelidx_index is not None and (pixel_step is not None or pixel_start is not None):
        prev_pix: Optional[int] = None
        count = 0
        first_pix: Optional[int] = None
        for row in spectrum_rows:
            try:
                pv = int(str(row[pixelidx_index]).strip().strip('"').strip("'"))
            except Exception as exc:
                raise ValueError(f"{source_name} 像素索引无法解析为整数：{row[pixelidx_index]}") from exc
            if first_pix is None:
                first_pix = pv
            if prev_pix is not None:
                if pixel_step is not None and pv - prev_pix != pixel_step:
                    raise ValueError(
                        f"{source_name} 光谱像素步长异常：期望+{pixel_step}，实际 {prev_pix}->{pv}"
                    )
            prev_pix = pv
            count += 1
        if count != ROWS_PER_SPECTRUM:
            raise ValueError(f"{source_name} 光谱行数异常：{count}")
        if pixel_start is not None and first_pix is not None and first_pix != pixel_start:
            raise ValueError(
                f"{source_name} 光谱像素起始值异常：期望 {pixel_start}，实际 {first_pix}"
            )

    return spectrum_rows


def try_parse_int(value: str) -> int:
    value = value.strip().strip('"').strip("'")
    try:
        return int(value)
    except ValueError:
        try:
            return int(float(value))
        except Exception as exc:
            raise ValueError(f"无法将spectrumid值\"{value}\"解析为整数") from exc


def write_spectrum(
    writer: csv.writer,
    spectrum_rows: List[List[str]],
    spectrumid_index: int,
    id_offset: int = 0,
) -> None:
    if id_offset:
        # 在写出前，整体偏移ID
        # 拿第一行解析ID，避免重复解析3030次
        original_id = try_parse_int(spectrum_rows[0][spectrumid_index])
        new_id_str = str(original_id + id_offset)
        # 写出时逐行替换该列
        for row in spectrum_rows:
            row_out = list(row)
            row_out[spectrumid_index] = new_id_str
            writer.writerow(row_out)
    else:
        for row in spectrum_rows:
            writer.writerow(row)


def open_csv_reader(path: str) -> Tuple[csv.reader, 'io.TextIOWrapper']:
    # 提大缓冲区，提升吞吐
    f = open(path, mode="r", newline="", buffering=4 * 1024 * 1024)
    reader = csv.reader(f)
    return reader, f


def open_text_reader(path: str) -> io.TextIOWrapper:
    # 大缓冲文本读取
    return open(path, mode="r", newline="", buffering=8 * 1024 * 1024)


def ensure_same_header(base: List[str], other: List[str], name: str) -> None:
    if base != other:
        # 表头不完全一致时给出警告，但不强制中止（只要包含spectrumid列即可）
        print(
            f"[警告] {name} 的表头与主表头不一致。\n"
            f"主表头: {base}\n"
            f"{name}表头: {other}",
            file=sys.stderr,
            flush=True,
        )


def mix_stream(
    a_path: str,
    a2_path: Optional[str],
    b1_path: str,
    b2_path: Optional[str],
    output_dir: str,
    part_spectra_threshold: int = PART_SPECTRA_THRESHOLD,
    total_spectra_limit: Optional[int] = None,
    fast_a_raw: bool = False,
    a_prefetch_depth: int = 8,
    b_prefetch_depth: int = 4,
    out_buffer_bytes: int = 4 * 1024 * 1024,
    a_validate_fast: bool = False,
    a_group_n: int = 4,
    cycle_b: bool = False,
) -> None:
    # 目录/文件的创建逻辑下移到具体分支，避免把文件名当目录创建

    a_reader = None
    a_fp = None
    a_reader2 = None
    a_fp2 = None
    a_raw_fp = None
    a_raw_fp2 = None

    # 打开B源
    b1_reader, b1_fp = open_csv_reader(b1_path)
    b2_reader = None
    b2_fp = None
    b2_available = False
    if b2_path and b2_path.strip() and b2_path.strip() != "/dev/null":
        try:
            b2_reader, b2_fp = open_csv_reader(b2_path)
            # header 在下方与B1一并读取
            b2_available = True
        except Exception as _:
            b2_reader = None
            b2_fp = None
            b2_available = False

    try:
        if fast_a_raw:
            # A采用原始文本读取，提高吞吐；表头用首行文本解析
            a_raw_fp = open_text_reader(a_path)
            header_a_line = a_raw_fp.readline()
            if not header_a_line:
                raise ValueError("A源为空，缺少表头")
            header_a = parse_header_from_line(header_a_line)
            # 预读A2首行（可能是表头，也可能是数据）
            a2_first_line: Optional[str] = None
            a2_has_header = True
            if a2_path:
                a_raw_fp2 = open_text_reader(a2_path)
                header_a2_line = a_raw_fp2.readline()
                if not header_a2_line:
                    raise ValueError("A2源为空，缺少表头或数据")
                # 暂不立即校验，与A表头比对放到拿到spectrumid_index之后再做启发式判断
                a2_first_line = header_a2_line
        else:
            # 常规CSV读取
            a_reader, a_fp = open_csv_reader(a_path)
            header_a = read_header(a_reader)
            if a2_path:
                a_reader2, a_fp2 = open_csv_reader(a2_path)
                header_a2_candidate = read_header(a_reader2)
                # 暂存，待获得spectrumid_index后判断是否为无表头数据行
                header_a2 = header_a2_candidate
        # 读取B1表头（可容错：无表头或空文件）
        b1_headerless = False
        b1_size = None
        try:
            b1_size = os.path.getsize(b1_path)
        except Exception:
            b1_size = None
        try:
            header_b1 = read_header(b1_reader)
            # 若B1“表头”第一列为数字，视为无表头数据行 → 重开并从首行开始读
            is_numeric_first = False
            try:
                _ = int(str(header_b1[0]).strip().strip('"').strip("'"))
                is_numeric_first = True
            except Exception:
                is_numeric_first = False
            if is_numeric_first:
                print(
                    f"[警告] B1 首行看起来是数据而非表头，将按无表头处理并从首行读取：{b1_path}",
                    file=sys.stderr,
                    flush=True,
                )
                try:
                    b1_fp.close()
                except Exception:
                    pass
                b1_reader, b1_fp = open_csv_reader(b1_path)
                header_b1 = header_a
                b1_headerless = True
            else:
                ensure_same_header(header_a, header_b1, "B1")
        except ValueError as e:
            # /dev/null 等特殊文件：路径不是常规CSV，直接标记为空并报更清晰错误
            if b1_path.strip() == "/dev/null":
                raise ValueError("B1 指向 /dev/null，请提供真实的B CSV文件或将其设为空并开启CYCLE_B以仅循环B2（不推荐）。") from e
            if b1_size == 0:
                raise ValueError(f"B1文件为空：{b1_path}") from e
            # 认为B1无表头，使用A的表头；重开reader从首行开始读数据
            print(
                f"[警告] B1缺少表头或首行不可读，按无表头模式处理：{b1_path}",
                file=sys.stderr,
                flush=True,
            )
            try:
                b1_fp.close()
            except Exception:
                pass
            b1_reader, b1_fp = open_csv_reader(b1_path)
            header_b1 = header_a
            b1_headerless = True
        if b2_available and b2_reader is not None:
            try:
                header_b2 = read_header(b2_reader)
                ensure_same_header(header_a, header_b2, "B2")
            except Exception as e:
                print(f"[警告] B2不可用（跳过）：{e}", file=sys.stderr, flush=True)
                try:
                    b2_fp.close()  # type: ignore[union-attr]
                except Exception:
                    pass
                b2_reader = None
                b2_fp = None
                b2_available = False

        spectrumid_index = detect_spectrumid_index(header_a)
        pixelidx_index = detect_pixelidx_index(header_a)
        # 可选像素步长校验（如303点步长=10）
        try:
            pixel_step_env = os.environ.get("MIX_VALIDATE_PIXEL_STEP")
            pixel_step = int(pixel_step_env) if pixel_step_env else None
        except ValueError:
            pixel_step = None

        # 针对A2首行进行启发式判断：若首行的spectrumid列为数值，则认为A2无表头，需要将该行作为数据保留
        if fast_a_raw and a2_path and a_raw_fp2 is not None:
            if a2_first_line is not None:
                parsed = next(csv.reader([a2_first_line]))
                is_data = False
                try:
                    _ = int(str(parsed[spectrumid_index]).strip().strip('"').strip("'"))
                    is_data = True
                except Exception:
                    is_data = False
                if is_data:
                    # A2无表头：保留该首行数据，后续生产者先消费它
                    a2_has_header = False
                else:
                    # A2有表头：校验与A的一致性
                    ensure_same_header(header_a, parsed, "A2")
                    a2_first_line = None
                    a2_has_header = True
        elif (not fast_a_raw) and a2_path and a_reader2 is not None:
            # 普通CSV模式：若读到的“表头候选”是数据（spectrumid列可解析为数值），则重开A2并不跳过表头
            try:
                token = header_a2[spectrumid_index]  # type: ignore[name-defined]
                _ = int(str(token).strip().strip('"').strip("'"))
                # 识别为数据行：重开reader，从头开始不跳过
                try:
                    a_fp2.close()
                except Exception:
                    pass
                a_reader2, a_fp2 = open_csv_reader(a2_path)
            except Exception:
                # 识别为表头：做一致性提示
                ensure_same_header(header_a, header_a2, "A2")  # type: ignore[name-defined]

        part_index = 1
        spectra_in_current_part = 0
        total_spectra_written = 0

        # 打开第一个输出文件
        # 允许 output_dir 既可以是目录也可以是具体文件前缀；若是目录则按 partXXX 命名
        if os.path.isdir(output_dir) or output_dir.endswith("/"):
            os.makedirs(output_dir, exist_ok=True)
            current_out_path = os.path.join(
                output_dir, f"mix_A4_B1_offset1e7_part{part_index:03d}.csv"
            )
        else:
            # 传入的是一个文件路径或前缀；若无扩展名则自动补上part序号与.csv
            parent = os.path.dirname(output_dir) or "."
            os.makedirs(parent, exist_ok=True)
            base = os.path.basename(output_dir)
            if base.endswith(".csv"):
                # 直接使用给定文件名（忽略分片逻辑），仅输出一个文件
                current_out_path = output_dir
            else:
                current_out_path = os.path.join(parent, f"{base}_part{part_index:03d}.csv")
        out_fp = open(current_out_path, mode="w", newline="", buffering=out_buffer_bytes)
        out_writer = csv.writer(out_fp)
        # 统一使用A的表头写出
        out_writer.writerow(header_a)

        def rotate_output_file():
            nonlocal part_index, spectra_in_current_part, out_fp, out_writer, current_out_path
            out_fp.flush()
            out_fp.close()
            print(
                f"[完成] 生成CSV: {current_out_path} ，包含完整光谱 {spectra_in_current_part} 条。",
                flush=True,
            )
            part_index += 1
            spectra_in_current_part = 0
            if os.path.isdir(output_dir) or output_dir.endswith("/"):
                new_path = os.path.join(
                    output_dir, f"mix_A4_B1_offset1e7_part{part_index:03d}.csv"
                )
            else:
                parent = os.path.dirname(output_dir) or "."
                base = os.path.basename(output_dir)
                if base.endswith(".csv"):
                    # 若用户传了具体文件名，则不再轮转，继续覆盖（但我们避免覆盖，此处直接追加序号）
                    name, _ext = os.path.splitext(base)
                    new_path = os.path.join(parent, f"{name}_part{part_index:03d}.csv")
                else:
                    new_path = os.path.join(parent, f"{base}_part{part_index:03d}.csv")
            current_out_path = new_path
            new_fp = open(new_path, mode="w", newline="", buffering=1024 * 1024)
            new_writer = csv.writer(new_fp)
            new_writer.writerow(header_a)
            out_fp = new_fp
            out_writer = new_writer

        if fast_a_raw:
            # 使用生产者-消费者：A原始文本成块预取，B CSV解析预取
            a_queue: "queue.Queue[Optional[List[str]]]" = queue.Queue(maxsize=max(1, a_prefetch_depth))
            b_queue: "queue.Queue[Optional[List[List[str]]]]" = queue.Queue(maxsize=max(1, b_prefetch_depth))

            def a_producer():
                try:
                    using_a2 = False
                    current_fp = a_raw_fp
                    # 若A2无表头，需先消费首行
                    pending_a2_line: Optional[str] = a2_first_line if ('a2_first_line' in locals()) else None
                    while True:
                        block: List[str] = []
                        first_id: Optional[str] = None
                        for _ in range(ROWS_PER_SPECTRUM):
                            if using_a2 and pending_a2_line is not None:
                                line = pending_a2_line
                                pending_a2_line = None
                            else:
                                line = current_fp.readline()  # type: ignore[arg-type]
                            if not line:
                                # 切换到A2（若存在且尚未使用）
                                if not using_a2 and a_raw_fp2 is not None:
                                    using_a2 = True
                                    current_fp = a_raw_fp2
                                    # 切到A2后重新读这一行
                                    if pending_a2_line is not None:
                                        line = pending_a2_line
                                        pending_a2_line = None
                                    else:
                                        line = current_fp.readline()
                                    if not line:
                                        if len(block) == 0:
                                            a_queue.put(None)
                                            return
                                        raise EOFErrorIncompleteSpectrum("A2 空文件或缺少数据（fast）")
                                else:
                                    if len(block) == 0:
                                        a_queue.put(None)
                                        return
                                    # 中间EOF，数据不完整
                                    raise EOFErrorIncompleteSpectrum("A 在读取完整光谱时意外到达文件末尾（fast模式）。")
                            block.append(line)
                            if a_validate_fast:
                                # 仅解析spectrumid列进行一致性校验
                                for row in csv.reader([line]):
                                    sid = row[spectrumid_index]
                                    if first_id is None:
                                        first_id = sid
                                    elif sid != first_id:
                                        raise ValueError(
                                            f"A(fast) 检测到一个光谱的{ROWS_PER_SPECTRUM}行中spectrumid不一致：首为{first_id}，现为{sid}"
                                        )
                        a_queue.put(block)
                except Exception as e:
                    # 将异常通过特殊对象传递
                    a_queue.put(None)
                    print(f"[错误] A生产者异常: {e}", file=sys.stderr, flush=True)

            def b_producer():
                nonlocal b1_reader, b1_fp, b2_reader, b2_fp
                b1_exhausted_local = False
                b2_exhausted_local = not (b2_available and b2_reader is not None)
                try:
                    while True:
                        if not b1_exhausted_local:
                            spec = read_next_spectrum(
                                b1_reader, spectrumid_index, "B1", pixelidx_index, pixel_step,
                                pixel_start= int(os.environ.get("MIX_VALIDATE_PIXEL_START", "4")) if os.environ.get("MIX_VALIDATE_PIXEL_START") else None
                            )
                            if spec is not None:
                                b_queue.put(spec)
                                continue
                            b1_exhausted_local = True
                        if not b2_exhausted_local and b2_reader is not None:
                            spec = read_next_spectrum(
                                b2_reader, spectrumid_index, "B2", pixelidx_index, pixel_step,
                                pixel_start= int(os.environ.get("MIX_VALIDATE_PIXEL_START", "4")) if os.environ.get("MIX_VALIDATE_PIXEL_START") else None
                            )
                            if spec is not None:
                                b_queue.put(spec)
                                continue
                            b2_exhausted_local = True
                        # B完全耗尽
                        if cycle_b:
                            try:
                                b1_fp.close()
                            except Exception:
                                pass
                            try:
                                b2_fp.close()
                            except Exception:
                                pass
                            b1_reader, b1_fp = open_csv_reader(b1_path)
                            read_header(b1_reader)
                            if b2_path and b2_path.strip() and b2_path.strip() != "/dev/null":
                                try:
                                    b2_reader, b2_fp = open_csv_reader(b2_path)
                                    read_header(b2_reader)
                                    b2_exhausted_local = False
                                except Exception:
                                    b2_reader = None
                                    b2_fp = None
                                    b2_exhausted_local = True
                            b1_exhausted_local = False
                            continue
                        else:
                            b_queue.put(None)
                            return
                except Exception as e:
                    b_queue.put(None)
                    print(f"[错误] B生产者异常: {e}", file=sys.stderr, flush=True)

            ta = threading.Thread(target=a_producer, name="AProducer", daemon=True)
            tb = threading.Thread(target=b_producer, name="BProducer", daemon=True)
            ta.start(); tb.start()

            b_finished = False

            # 主循环：直到A读完或达到总光谱上限
            while True:
                wrote_any = False
                # 先写a_group_n个A
                for _ in range(max(1, a_group_n)):
                    block = a_queue.get()
                    if block is None:
                        # A耗尽，结束
                        out_fp.flush(); out_fp.close()
                        print(
                            f"[完成] 生成CSV: {current_out_path} ，包含完整光谱 {spectra_in_current_part} 条。",
                            flush=True,
                        )
                        print(
                            f"[结束] A源读取完毕。总计写出完整光谱 {total_spectra_written} 条。",
                            flush=True,
                        )
                        return
                    # 直接批量写出A的3030行
                    out_fp.writelines(block)
                    spectra_in_current_part += 1
                    total_spectra_written += 1
                    wrote_any = True
                    if total_spectra_limit is not None and total_spectra_written >= total_spectra_limit:
                        out_fp.flush(); out_fp.close()
                        print(
                            f"[完成] 生成CSV: {current_out_path} ，包含完整光谱 {spectra_in_current_part} 条。",
                            flush=True,
                        )
                        print(
                            f"[结束] 达到总光谱上限 {total_spectra_limit} 条，停止。总计写出完整光谱 {total_spectra_written} 条。",
                            flush=True,
                        )
                        return
                    if spectra_in_current_part >= part_spectra_threshold:
                        rotate_output_file()

                # 再写1个B（若还有）
                if not b_finished:
                    spec_b = b_queue.get()
                    if spec_b is not None:
                        write_spectrum(out_writer, spec_b, spectrumid_index, id_offset=B_ID_OFFSET)
                        spectra_in_current_part += 1
                        total_spectra_written += 1
                        wrote_any = True
                        if total_spectra_limit is not None and total_spectra_written >= total_spectra_limit:
                            out_fp.flush(); out_fp.close()
                            print(
                                f"[完成] 生成CSV: {current_out_path} ，包含完整光谱 {spectra_in_current_part} 条。",
                                flush=True,
                            )
                            print(
                                f"[结束] 达到总光谱上限 {total_spectra_limit} 条，停止。总计写出完整光谱 {total_spectra_written} 条。",
                                flush=True,
                            )
                            return
                        if spectra_in_current_part >= part_spectra_threshold:
                            rotate_output_file()
                    else:
                        b_finished = True

                if not wrote_any:
                    break
        else:
            # 原有并行预取（CSV解析路径）
            executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(max_workers=3)

            # 状态：B1先读，读完切到B2
            b1_exhausted = False
            b2_exhausted = False
            b_drained = False

            def read_next_b_task() -> Optional[List[List[str]]]:
                nonlocal b1_exhausted, b2_exhausted, b1_reader, b1_fp, b2_reader, b2_fp
                if not b1_exhausted:
                    spec = read_next_spectrum(
                        b1_reader, spectrumid_index, "B1", pixelidx_index, pixel_step,
                        pixel_start= int(os.environ.get("MIX_VALIDATE_PIXEL_START", "4")) if os.environ.get("MIX_VALIDATE_PIXEL_START") else None
                    )
                    if spec is not None:
                        return spec
                    b1_exhausted = True
                if not b2_exhausted and b2_reader is not None:
                    spec = read_next_spectrum(
                        b2_reader, spectrumid_index, "B2", pixelidx_index, pixel_step,
                        pixel_start= int(os.environ.get("MIX_VALIDATE_PIXEL_START", "4")) if os.environ.get("MIX_VALIDATE_PIXEL_START") else None
                    )
                    if spec is not None:
                        return spec
                    b2_exhausted = True
                if cycle_b:
                    # 重新打开B1/B2并从头开始
                    try:
                        b1_fp.close()
                    except Exception:
                        pass
                    try:
                        b2_fp.close()
                    except Exception:
                        pass
                    b1_reader, b1_fp = open_csv_reader(b1_path)
                    read_header(b1_reader)
                    if b2_path and b2_path.strip() and b2_path.strip() != "/dev/null":
                        try:
                            b2_reader, b2_fp = open_csv_reader(b2_path)
                            read_header(b2_reader)
                            b2_exhausted = False
                        except Exception:
                            b2_reader = None
                            b2_fp = None
                            b2_exhausted = True
                    b1_exhausted = False
                    # 再次尝试读取
                    return read_next_b_task()
                return None

            # 初始化预取任务
            # 注意：A、B分别预取一条；CSV解析路径下，reader不适合多线程并行读取同一文件
            def read_next_a_task() -> Optional[List[List[str]]]:
                nonlocal a_reader, a_reader2
                if a_reader is not None:
                    spec = read_next_spectrum(a_reader, spectrumid_index, "A1")
                    if spec is not None:
                        return spec
                    a_reader = None  # A1耗尽
                if a_reader2 is not None:
                    spec = read_next_spectrum(a_reader2, spectrumid_index, "A2")
                    if spec is not None:
                        return spec
                    a_reader2 = None
                return None

            future_a = executor.submit(read_next_a_task)
            future_b = executor.submit(read_next_b_task)

            # 主循环：直到A读完或达到总光谱上限
            while True:
                wrote_any = False
                # 先写a_group_n个A
                for _ in range(max(1, a_group_n)):
                    spec_a = future_a.result()
                    if spec_a is None:
                        # A读完，收尾并退出
                        out_fp.flush(); out_fp.close()
                        print(
                            f"[完成] 生成CSV: {current_out_path} ，包含完整光谱 {spectra_in_current_part} 条。",
                            flush=True,
                        )
                        print(
                            f"[结束] A源读取完毕。总计写出完整光谱 {total_spectra_written} 条。",
                            flush=True,
                        )
                        return
                    # 立刻预取下一条A
                    future_a = executor.submit(read_next_a_task)
                    # 写出当前A
                    write_spectrum(out_writer, spec_a, spectrumid_index, id_offset=0)
                    spectra_in_current_part += 1
                    total_spectra_written += 1
                    wrote_any = True
                    if total_spectra_limit is not None and total_spectra_written >= total_spectra_limit:
                        out_fp.flush(); out_fp.close()
                        print(
                            f"[完成] 生成CSV: {current_out_path} ，包含完整光谱 {spectra_in_current_part} 条。",
                            flush=True,
                        )
                        print(
                            f"[结束] 达到总光谱上限 {total_spectra_limit} 条，停止。总计写出完整光谱 {total_spectra_written} 条。",
                            flush=True,
                        )
                        return
                    if spectra_in_current_part >= part_spectra_threshold:
                        rotate_output_file()

                # 再写1个B（若还有）
                if not b_drained:
                    spec_b = future_b.result()
                    if spec_b is not None:
                        # 立刻预取下一条B
                        future_b = executor.submit(read_next_b_task)
                        # 写出B（加偏移）
                        write_spectrum(out_writer, spec_b, spectrumid_index, id_offset=B_ID_OFFSET)
                        spectra_in_current_part += 1
                        total_spectra_written += 1
                        wrote_any = True
                        if total_spectra_limit is not None and total_spectra_written >= total_spectra_limit:
                            out_fp.flush(); out_fp.close()
                            print(
                                f"[完成] 生成CSV: {current_out_path} ，包含完整光谱 {spectra_in_current_part} 条。",
                                flush=True,
                            )
                            print(
                                f"[结束] 达到总光谱上限 {total_spectra_limit} 条，停止。总计写出完整光谱 {total_spectra_written} 条。",
                                flush=True,
                            )
                            return
                        if spectra_in_current_part >= part_spectra_threshold:
                            rotate_output_file()
                    else:
                        b_drained = True

                if not wrote_any:
                    break

    except EOFErrorIncompleteSpectrum as err:
        print(f"[错误] {err}", file=sys.stderr, flush=True)
        raise
    finally:
        try:
            # 结束并行线程/生产者
            # executor 仅在CSV路径下存在
            executor.shutdown(wait=False, cancel_futures=True)  # type: ignore[name-defined]
        except Exception:
            pass
        try:
            a_fp.close()
        except Exception:
            pass
        try:
            if a_raw_fp is not None:
                a_raw_fp.close()
        except Exception:
            pass
        try:
            b1_fp.close()
        except Exception:
            pass
        try:
            b2_fp.close()
        except Exception:
            pass


def mix_insert_mode(
    base_path: str,
    insert_path: str,
    output_path: str,
    every_n: int = 4,
    insert_id_offset: int = 20_000_000,
    cycle_insert: bool = True,
    fast_base_raw: bool = False,
    validate_fast_base: bool = False,
    prefetch_base: int = 8,
    prefetch_insert: int = 4,
    out_buffer_bytes: int = 4 * 1024 * 1024,
) -> None:
    """以B为基准，每every_n条B插入1条A（A可循环），写出到一个新CSV。

    要求：
    - 每条光谱为3030行；
    - A插入时统一对spectrumid加 insert_id_offset；
    - B与A的表头需兼容（至少包含spectrumid列），以B的表头为输出表头；
    - 直到B读完为止。
    """
    # 准备B（base）reader
    if fast_base_raw:
        base_fp = open_text_reader(base_path)
        base_header_line = base_fp.readline()
        if not base_header_line:
            raise ValueError("基准文件为空，缺少表头")
        base_header = parse_header_from_line(base_header_line)
    else:
        base_reader, base_csv_fp = open_csv_reader(base_path)
        base_header = read_header(base_reader)

    # 准备A（insert）reader（CSV解析即可，便于ID偏移；可循环）
    insert_reader, insert_fp = open_csv_reader(insert_path)
    insert_header = read_header(insert_reader)
    ensure_same_header(base_header, insert_header, "INSERT(A)")
    spectrumid_index = detect_spectrumid_index(base_header)

    # 输出
    out_fp = open(output_path, mode="w", newline="", buffering=out_buffer_bytes)
    out_writer = csv.writer(out_fp)
    out_writer.writerow(base_header)

    # 生产者队列
    base_queue: "queue.Queue[Optional[List[str]]]" = queue.Queue(maxsize=max(1, prefetch_base))
    insert_queue: "queue.Queue[Optional[List[List[str]]]]" = queue.Queue(maxsize=max(1, prefetch_insert))

    def base_producer():
        try:
            if fast_base_raw:
                while True:
                    block: List[str] = []
                    first_id: Optional[str] = None
                    for _ in range(ROWS_PER_SPECTRUM):
                        line = base_fp.readline()  # type: ignore[arg-type]
                        if not line:
                            if len(block) == 0:
                                base_queue.put(None)
                                return
                            raise EOFErrorIncompleteSpectrum("B 在读取完整光谱时意外到达文件末尾（insert-fast）。")
                        block.append(line)
                        if validate_fast_base:
                            for row in csv.reader([line]):
                                sid = row[spectrumid_index]
                                if first_id is None:
                                    first_id = sid
                                elif sid != first_id:
                                    raise ValueError(
                                        f"B(insert-fast) 3030行spectrumid不一致：首为{first_id}，现为{sid}"
                                    )
                    base_queue.put(block)
            else:
                while True:
                    spec = read_next_spectrum(base_reader, spectrumid_index, "BASE(B)")
                    if spec is None:
                        base_queue.put(None)
                        return
                    base_queue.put([",".join(row) + "\n" for row in spec])
        except Exception as e:
            base_queue.put(None)
            print(f"[错误] BASE生产者异常: {e}", file=sys.stderr, flush=True)

    def insert_producer():
        nonlocal insert_reader, insert_fp
        try:
            while True:
                spec = read_next_spectrum(insert_reader, spectrumid_index, "INSERT(A)")
                if spec is None:
                    if not cycle_insert:
                        insert_queue.put(None)
                        return
                    # 重新打开insert源，从头循环（正确重绑定reader与文件句柄）
                    try:
                        insert_fp.close()
                    except Exception:
                        pass
                    insert_reader, insert_fp = open_csv_reader(insert_path)
                    # 跳过表头
                    read_header(insert_reader)
                    continue
                insert_queue.put(spec)
        except Exception as e:
            insert_queue.put(None)
            print(f"[错误] INSERT生产者异常: {e}", file=sys.stderr, flush=True)

    tb = threading.Thread(target=base_producer, name="BaseProducer", daemon=True)
    ta = threading.Thread(target=insert_producer, name="InsertProducer", daemon=True)
    tb.start(); ta.start()

    try:
        b_finished = False
        b_count = 0
        while True:
            # 写every_n条B
            for _ in range(every_n):
                block = base_queue.get()
                if block is None:
                    b_finished = True
                    break
                out_fp.writelines(block)
                b_count += 1
            if b_finished:
                break

            # 插入1条A（若还有）
            spec_a = insert_queue.get()
            if spec_a is not None:
                write_spectrum(out_writer, spec_a, spectrumid_index, id_offset=insert_id_offset)
            else:
                # A已完全耗尽且不循环，则跳过插入继续
                pass
        print(f"[完成] 插入模式输出: {output_path} ，B基准光谱数 {b_count} 条。", flush=True)
    finally:
        try:
            out_fp.flush(); out_fp.close()
        except Exception:
            pass
        try:
            if fast_base_raw:
                base_fp.close()
            else:
                base_csv_fp.close()
        except Exception:
            pass
        try:
            insert_fp.close()
        except Exception:
            pass


def main(argv: List[str]) -> int:
    # 允许通过命令行传参，也提供默认路径
    a_path = os.environ.get("MIX_A_PATH", (argv[1] if len(argv) > 1 else DEFAULT_A_PATH))
    a2_path = os.environ.get("MIX_A2_PATH")
    b1_path = os.environ.get("MIX_B1_PATH", (argv[2] if len(argv) > 2 else DEFAULT_B1_PATH))
    b2_path = os.environ.get("MIX_B2_PATH", (argv[3] if len(argv) > 3 else DEFAULT_B2_PATH))
    output_dir = os.environ.get("MIX_OUT_DIR", (argv[4] if len(argv) > 4 else DEFAULT_OUTPUT_DIR))
    try:
        threshold_env = os.environ.get("MIX_PART_SPECTRA_THRESHOLD")
        threshold = int(threshold_env) if threshold_env is not None else PART_SPECTRA_THRESHOLD
    except ValueError:
        threshold = PART_SPECTRA_THRESHOLD
    try:
        total_env = os.environ.get("MIX_TOTAL_SPECTRA_LIMIT")
        total_limit = int(total_env) if total_env is not None else None
    except ValueError:
        total_limit = None
    # 高级配置：快速模式与预取/缓冲
    fast_raw_env = os.environ.get("MIX_FAST_A_RAW", "0").strip().lower()
    fast_a_raw = fast_raw_env in ("1", "true", "yes", "on")
    try:
        a_prefetch = int(os.environ.get("MIX_PREFETCH_A", "8"))
    except ValueError:
        a_prefetch = 8
    try:
        b_prefetch = int(os.environ.get("MIX_PREFETCH_B", "4"))
    except ValueError:
        b_prefetch = 4
    try:
        out_buf_mb = int(os.environ.get("MIX_OUT_BUFFER_MB", "8"))
    except ValueError:
        out_buf_mb = 8
    out_buffer_bytes = max(1, out_buf_mb) * 1024 * 1024
    a_validate_fast = os.environ.get("MIX_A_VALIDATE_FAST", "0").strip().lower() in ("1", "true", "yes", "on")
    # 读取A:B比例与B循环
    try:
        a_group_n = int(os.environ.get("MIX_A_GROUP_N", "4"))
    except ValueError:
        a_group_n = 4
    cycle_b = os.environ.get("MIX_CYCLE_B", "0").strip().lower() in ("1", "true", "yes", "on")

    mode = os.environ.get("MIX_MODE", "default").strip().lower()
    if mode == "default":
        print(
            "配置:\n"
            f"  A: {a_path}\n"
            f"  A2: {a2_path}\n"
            f"  B1: {b1_path}\n"
            f"  B2: {b2_path}\n"
            f"  输出目录: {output_dir}\n"
                f"  每文件光谱数阈值: {threshold}\n"
                f"  总光谱上限: {total_limit}\n"
                f"  FAST_A_RAW: {fast_a_raw}\n"
                f"  PREFETCH_A: {a_prefetch}  PREFETCH_B: {b_prefetch}\n"
                f"  OUT_BUFFER_MB: {out_buf_mb}\n"
                f"  A_VALIDATE_FAST: {a_validate_fast}\n"
                f"  A_GROUP_N: {a_group_n}  CYCLE_B: {cycle_b}",
            flush=True,
        )

        mix_stream(
            a_path,
            a2_path,
            b1_path,
            b2_path,
            output_dir,
            threshold,
            total_limit,
            fast_a_raw,
            a_prefetch,
            b_prefetch,
            out_buffer_bytes,
            a_validate_fast,
            a_group_n,
            cycle_b,
        )
        return 0
    elif mode == "insert":
        base_path = os.environ.get("MIX_BASE_PATH")
        insert_path = os.environ.get("MIX_INSERT_PATH")
        output_file = os.environ.get("MIX_OUTPUT_FILE")
        if not base_path or not insert_path or not output_file:
            raise ValueError("insert模式需要设置 MIX_BASE_PATH、MIX_INSERT_PATH、MIX_OUTPUT_FILE 环境变量")
        try:
            insert_every_n = int(os.environ.get("MIX_INSERT_EVERY_N", "4"))
        except ValueError:
            insert_every_n = 4
        try:
            insert_id_offset = int(os.environ.get("MIX_INSERT_ID_OFFSET", "20000000"))
        except ValueError:
            insert_id_offset = 20_000_000
        cycle_insert = os.environ.get("MIX_CYCLE_INSERT", "1").strip().lower() in ("1", "true", "yes", "on")
        fast_base_raw = os.environ.get("MIX_INSERT_FAST_BASE_RAW", "0").strip().lower() in ("1", "true", "yes", "on")
        validate_fast_base = os.environ.get("MIX_INSERT_VALIDATE_FAST_BASE", "0").strip().lower() in ("1", "true", "yes", "on")
        try:
            prefetch_base = int(os.environ.get("MIX_INSERT_PREFETCH_BASE", "8"))
        except ValueError:
            prefetch_base = 8
        try:
            prefetch_insert = int(os.environ.get("MIX_INSERT_PREFETCH_INSERT", "4"))
        except ValueError:
            prefetch_insert = 4
        try:
            out_buf_mb_insert = int(os.environ.get("MIX_INSERT_OUT_BUFFER_MB", "8"))
        except ValueError:
            out_buf_mb_insert = 8
        out_buf_bytes_insert = max(1, out_buf_mb_insert) * 1024 * 1024

        print(
            "配置(插入模式):\n"
            f"  基准(B)路径: {base_path}\n"
            f"  插入(A)路径: {insert_path}\n"
            f"  输出文件: {output_file}\n"
            f"  规则: 每 {insert_every_n} 条B插入1条A\n"
            f"  插入A的ID偏移: +{insert_id_offset}\n"
            f"  A耗尽后循环: {cycle_insert}\n"
            f"  FAST_BASE_RAW: {fast_base_raw}  VALIDATE_FAST_BASE: {validate_fast_base}\n"
            f"  PREFETCH_BASE: {prefetch_base}  PREFETCH_INSERT: {prefetch_insert}\n"
            f"  OUT_BUFFER_MB: {out_buf_mb_insert}",
            flush=True,
        )

        mix_insert_mode(
            base_path=base_path,
            insert_path=insert_path,
            output_path=output_file,
            every_n=insert_every_n,
            insert_id_offset=insert_id_offset,
            cycle_insert=cycle_insert,
            fast_base_raw=fast_base_raw,
            validate_fast_base=validate_fast_base,
            prefetch_base=prefetch_base,
            prefetch_insert=prefetch_insert,
            out_buffer_bytes=out_buf_bytes_insert,
        )
        return 0
    else:
        raise ValueError(f"未知的 MIX_MODE: {mode}")


if __name__ == "__main__":
    sys.exit(main(sys.argv))


