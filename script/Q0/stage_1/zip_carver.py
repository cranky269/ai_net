# -*- coding: utf-8 -*-
"""分卷 ZIP 局部提取器（不依赖中央目录）。

背景
----
DroneRFb-DIR 在 ScienceDB 上以 ``.z01/.z02/.../*.zip`` 分卷形式发布，
正常解压必须等全部卷到齐（中央目录在最后一卷）。本模块改为顺序扫描
ZIP 的"本地文件头"（``PK\\x03\\x04``），直接从单个 ``.z01`` 卷中解出
其中**完整**的数据文件，用于"只下载 1 卷做数据探针"的场景。

能力边界
--------
1. 仅适用于从文件起始处开始的第一卷（``.z01``）；``.z02`` 及之后是续卷，
   不包含任何本地文件头，无法单独解析；
2. 跨卷截断的最后一个条目会被判定为不完整并丢弃；
3. 不支持加密条目（发现即跳过）；
4. 非目标扩展名的条目只做跳过，不做解压（长度可从头部得到时）。

对外接口
--------
- :func:`extract_from_volume`  主入口，从卷中提取若干文件
"""

import binascii
import logging
import os
import struct
import zlib

LOCAL_SIG = b"PK\x03\x04"    # 本地文件头
CENTRAL_SIG = b"PK\x01\x02"  # 中央目录项
END_SIG = b"PK\x05\x06"      # 中央目录结束
DESC_SIG = b"PK\x07\x08"     # 数据描述符

HEADER_SIZE = 30      # 本地文件头固定长度
CHUNK_SIZE = 1 << 20  # 流式解压块大小
SCAN_SIZE = 1 << 16   # 首次签名扫描范围


# --------------------------------------------------------------------------- #
# 基础解析
# --------------------------------------------------------------------------- #
def _read_exact(f, n):
    """精确读取 n 字节，不足返回 None。"""
    data = f.read(n)
    if len(data) != n:
        return None
    return data


def _decode_name(raw, flags):
    """按通用标志位解码条目名（UTF-8 标志位 / GBK / CP437 逐级回退）。"""
    if flags & 0x0800:
        return raw.decode("utf-8", errors="replace")
    for enc in ("utf-8", "gbk", "cp437"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def _parse_zip64_extra(extra, need_usize, need_csize):
    """从扩展字段里取出 zip64 的 64 位长度，返回 (usize, csize)。"""
    usize, csize, pos = None, None, 0
    while pos + 4 <= len(extra):
        tag, size = struct.unpack_from("<HH", extra, pos)
        body = extra[pos + 4: pos + 4 + size]
        if tag == 0x0001:
            off = 0
            if need_usize and off + 8 <= len(body):
                usize = struct.unpack_from("<Q", body, off)[0]
                off += 8
            if need_csize and off + 8 <= len(body):
                csize = struct.unpack_from("<Q", body, off)[0]
                off += 8
        pos += 4 + size
    return usize, csize


def read_local_header(f, pos):
    """解析 pos 处的本地文件头；不是本地头时返回 None。"""
    f.seek(pos)
    head = _read_exact(f, HEADER_SIZE)
    if head is None or head[:4] != LOCAL_SIG:
        return None
    (_, _ver, flags, method, _mt, _md, crc, csize, usize,
     name_len, extra_len) = struct.unpack("<4sHHHHHIIIHH", head)
    raw_name = _read_exact(f, name_len) or b""
    extra = _read_exact(f, extra_len) or b""
    zip64 = csize == 0xFFFFFFFF or usize == 0xFFFFFFFF
    if zip64:
        u64, c64 = _parse_zip64_extra(
            extra, usize == 0xFFFFFFFF, csize == 0xFFFFFFFF)
        usize = u64 if u64 is not None else usize
        csize = c64 if c64 is not None else csize
    return {
        "name": _decode_name(raw_name, flags),
        "flags": int(flags),
        "method": int(method),
        "crc": int(crc),
        "csize": int(csize),
        "usize": int(usize),
        "zip64": zip64,
        "data_offset": pos + HEADER_SIZE + name_len + extra_len,
    }


def _find_next_signature(f, start, total, window=CHUNK_SIZE):
    """从 start 起向后找下一个 PK\\x03\\x04，返回绝对偏移或 None。"""
    pos = start
    while pos < total:
        f.seek(pos)
        buf = f.read(min(window, total - pos))
        if not buf:
            return None
        idx = buf.find(LOCAL_SIG)
        if idx >= 0:
            return pos + idx
        pos += len(buf) - 3
    return None


# --------------------------------------------------------------------------- #
# 数据读取
# --------------------------------------------------------------------------- #
def _skip_data_descriptor(f, pos, zip64):
    """跳过条目末尾的数据描述符，返回下一条目的可能起始位置。"""
    f.seek(pos)
    head = f.read(4)
    if head == DESC_SIG:
        return pos + (24 if zip64 else 16)
    if head in (LOCAL_SIG, CENTRAL_SIG, END_SIG):
        return pos
    if len(head) < 4:
        return pos
    return pos + (20 if zip64 else 12)


def _inflate_from(f, start):
    """从 start 处流式 deflate 解压，返回 (明文, 消耗字节数)；截断时返回 (None, 0)。"""
    f.seek(start)
    dobj = zlib.decompressobj(-15)
    parts, consumed = [], 0
    while True:
        chunk = f.read(CHUNK_SIZE)
        if not chunk:
            return None, 0
        parts.append(dobj.decompress(chunk))
        consumed += len(chunk)
        if dobj.eof:
            consumed -= len(dobj.unused_data)
            return b"".join(parts), consumed


def _read_stored(f, start, csize):
    """读取未压缩（stored）条目。"""
    f.seek(start)
    data = f.read(csize)
    if len(data) != csize:
        return None, 0
    return data, csize


def _read_payload(f, header):
    """读取并还原条目内容，返回 (明文, 消耗字节数)。"""
    if header["flags"] & 0x0001:
        return None, 0                       # 加密条目，不支持
    if header["method"] == 0:
        if header["csize"]:
            return _read_stored(f, header["data_offset"], header["csize"])
        nxt = _find_next_signature(
            f, header["data_offset"], os.fstat(f.fileno()).st_size)
        if nxt is None:
            return None, 0
        f.seek(header["data_offset"])
        return f.read(nxt - header["data_offset"]), nxt - header["data_offset"]
    if header["method"] == 8:
        return _inflate_from(f, header["data_offset"])
    return None, 0


def _next_position(f, header):
    """仅凭头部推算下一条目位置；无法推算时返回 None。"""
    has_desc = bool(header["flags"] & 0x0008)
    if not has_desc and header["csize"] > 0:
        pos = header["data_offset"] + header["csize"]
        return _skip_data_descriptor(f, pos, header["zip64"])
    return None


# --------------------------------------------------------------------------- #
# 落盘
# --------------------------------------------------------------------------- #
def _match_ext(name, extensions):
    return os.path.splitext(name)[1].lower() in {e.lower() for e in extensions}


def _unique_path(out_dir, basename):
    target = os.path.join(out_dir, basename)
    if not os.path.exists(target):
        return target
    stem, ext = os.path.splitext(basename)
    idx = 1
    while os.path.exists(target):
        target = os.path.join(out_dir, "%s_%d%s" % (stem, idx, ext))
        idx += 1
    return target


def _save_entry(out_dir, name, data, header, verify_crc, log):
    """保存单个条目并校验，返回统计信息 dict。"""
    basename = os.path.basename(name.replace("\\", "/")) or "unnamed.bin"
    target = _unique_path(out_dir, basename)
    with open(target, "wb") as fo:
        fo.write(data)
    crc_ok = True
    if verify_crc and header["crc"]:
        crc_ok = (binascii.crc32(data) & 0xFFFFFFFF) == header["crc"]
        if not crc_ok:
            log.warning("CRC 校验失败：%s", basename)
    log.info("已提取 %-45s 体积=%8.2f MB  CRC=%s",
             basename, len(data) / 1048576.0, "OK" if crc_ok else "FAIL")
    return {
        "entry_name": name,
        "saved_path": target,
        "bytes": len(data),
        "crc_ok": crc_ok,
        "method": header["method"],
    }


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def extract_from_volume(volume_path, out_dir, max_files=8,
                        extensions=(".mat",), logger=None, verify_crc=True):
    """从单个分卷（.z01）中顺序提取前 max_files 个目标文件。

    参数
    ----
    volume_path : str   分卷路径（必须是第一巻）
    out_dir     : str   提取文件的保存目录
    max_files   : int   最多提取多少个目标文件
    extensions  : tuple 需要保存的扩展名
    logger      : logging.Logger
    verify_crc  : bool  是否校验 CRC32

    返回
    ----
    list[dict] 每个成功提取文件的信息
    """
    log = logger or logging.getLogger(__name__)
    os.makedirs(out_dir, exist_ok=True)
    results = []
    total = os.path.getsize(volume_path)
    log.info("开始解析分卷：%s（%.2f GB）", volume_path, total / 1073741824.0)

    with open(volume_path, "rb") as f:
        pos = _find_next_signature(f, 0, total, SCAN_SIZE)
        if pos is None:
            log.error("卷内未找到任何本地文件头，可能不是第一巻或已加密")
            return results

        while pos is not None and len(results) < max_files:
            header = read_local_header(f, pos)
            if header is None:
                log.info("在第 %d 字节处无法继续解析，结束", pos)
                break

            name = header["name"]
            nxt = _next_position(f, header)
            data = None
            if nxt is None:
                data, consumed = _read_payload(f, header)
                if data is None:
                    log.warning("条目「%s」跨卷截断或无法解析，停止", name)
                    break
                nxt = _skip_data_descriptor(
                    f, header["data_offset"] + consumed, header["zip64"])

            if not name.endswith("/") and _match_ext(name, extensions):
                if data is None:
                    data, _ = _read_payload(f, header)
                if data is None:
                    log.warning("条目「%s」内容读取失败，停止", name)
                    break
                results.append(
                    _save_entry(out_dir, name, data, header, verify_crc, log))

            pos = nxt

    log.info("共提取 %d 个文件到 %s", len(results), out_dir)
    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    import sys
    if len(sys.argv) < 3:
        print("用法: python zip_carver.py <volume.z01> <out_dir> [max_files]")
        sys.exit(1)
    _max = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    extract_from_volume(sys.argv[1], sys.argv[2], max_files=_max)
