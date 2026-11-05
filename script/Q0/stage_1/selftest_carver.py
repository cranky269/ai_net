# -*- coding: utf-8 -*-
"""zip_carver 的最小单元自测（不依赖真实数据集）。

构造一个"仿分卷 ZIP"：若干本地文件头 + 压缩数据 + 末尾一个跨卷截断的条目，
验证 :func:`zip_carver.extract_from_volume` 能正确取出完整条目、
跳过截断条目，并且取出的文件能被 :mod:`mat_probe` 正常读取。

用法
----
    python script/Q1/selftest_carver.py
"""

import logging
import os
import shutil
import struct
import sys
import zlib

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from zip_carver import extract_from_volume      # noqa: E402
from mat_probe import probe_file                # noqa: E402

WORK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "..", "output", "Q0", "_selftest")
WORK_DIR = os.path.abspath(WORK_DIR)

LOCAL_SIG = b"PK\x03\x04"
DESC_SIG = b"PK\x07\x08"


def make_fake_mat(path, n=8192, seed=0):
    """生成一个与数据集同结构的最小 .mat（HDF5，I/Q 双通道 float16）。"""
    rng = np.random.default_rng(seed)
    i_data = rng.normal(0, 1, n).astype(np.float16)
    q_data = rng.normal(0, 1, n).astype(np.float16)
    with h5py.File(path, "w") as f:
        f.create_dataset("I", data=i_data.reshape(1, -1))
        f.create_dataset("Q", data=q_data.reshape(1, -1))
    with open(path, "rb") as f:
        return f.read()


def build_entry(name, data, method=8, use_descriptor=False):
    """按 ZIP 本地文件头规范拼一个条目（不含中央目录）。"""
    raw_name = name.encode("utf-8")
    crc = zlib.crc32(data) & 0xFFFFFFFF
    if method == 8:
        payload = zlib.compressobj(9, zlib.DEFLATED, -15)
        body = payload.compress(data) + payload.flush()
    else:
        body = data
    if use_descriptor:
        flags, csize, usize = 0x0008, 0, 0
    else:
        flags, csize, usize = 0, len(body), len(data)
    head = struct.pack("<4sHHHHHIIIHH", LOCAL_SIG, 20, flags, method,
                       0, 0x21, crc, csize, usize, len(raw_name), 0)
    blob = head + raw_name + body
    if use_descriptor:
        blob += struct.pack("<4sIII", DESC_SIG, crc, len(body), len(data))
    return blob


def build_volume(files, out_path):
    """把若干 (文件名, 字节) 打包成仿分卷文件，末尾追加一个截断条目。"""
    blob = b""
    for idx, (name, data) in enumerate(files):
        method = 0 if idx == len(files) - 1 else 8
        blob += build_entry(name, data, method=method)
    rng = np.random.default_rng(99)
    noise = rng.integers(0, 256, 8192, dtype=np.uint8).tobytes()
    blob += build_entry("truncated_slice.mat", noise)[:200]  # 故意截断（随机数据不可压缩）
    with open(out_path, "wb") as f:
        f.write(blob)
    return len(blob)


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("selftest")

    if os.path.isdir(WORK_DIR):
        shutil.rmtree(WORK_DIR)
    os.makedirs(WORK_DIR, exist_ok=True)

    src_dir = os.path.join(WORK_DIR, "src")
    os.makedirs(src_dir, exist_ok=True)
    names = ["A1_IN_S0_slice_1.mat", "A1_IN_S0_slice_2.mat",
             "B_OUT_S1_slice_3.mat"]
    files = []
    for idx, name in enumerate(names):
        files.append((name, make_fake_mat(os.path.join(src_dir, name),
                                          n=8192, seed=idx)))

    volume = os.path.join(WORK_DIR, "fake.z01")
    size = build_volume(files, volume)
    log.info("构造仿分卷：%s（%.1f KB）", volume, size / 1024.0)

    out_dir = os.path.join(WORK_DIR, "extracted")
    results = extract_from_volume(volume, out_dir, max_files=10, logger=log)

    assert len(results) == 3, "应提取 3 个完整条目，实际 %d" % len(results)
    assert all(r["crc_ok"] for r in results), "CRC 校验应全部通过"
    assert not os.path.exists(os.path.join(out_dir, "truncated_slice.mat")), \
        "截断条目不应被写出"

    row = probe_file(results[0]["saved_path"], fs_hz=80_000_000)
    assert row["ok"], "探针失败：%s" % row["error"]
    assert row["itemsize"] == 2, "dtype 应为 2 字节，实际 %s" % row["itemsize"]
    assert row["n_samples"] == 8192, "采样点数应为 8192"

    log.info("PASS：提取=%d，dtype=%s，n_samples=%d，时长=%.3f ms",
             len(results), row["dtype"], row["n_samples"],
             row["duration_s"] * 1000.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
