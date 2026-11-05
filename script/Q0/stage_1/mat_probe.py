# -*- coding: utf-8 -*-
"""DroneRFb-DIR 的 .mat 探针：读取 IQ、统计基本参数、粗估带内信噪比。

用途
----
在只有少量真实文件的情况下，快速回答：
    1. 通道名是否为 I/Q，dtype 是否为 2 字节（float16）；
    2. 实际采样点数与片段时长（是否 ≥ 4M 点 / 50 ms）；
    3. 信号功率量级与粗估带内信噪比（区分 LOS/NLOS 是否有 20 dB 差异）。

文件格式说明
------------
数据集论文给出的读取方式为::

    data = h5py.File(filename)
    iq = data["I"][0] + 1j * data["Q"][0]

即 ``.mat`` 实为 HDF5，``I``/``Q`` 为独立数据集。
"""

import logging
import os

import numpy as np

try:
    import h5py
except ImportError:  # 便于在缺依赖时给出明确提示
    h5py = None


def load_iq(path):
    """读取 .mat 中的 I/Q 通道，返回 (iq 复数序列, 元信息 dict)。"""
    if h5py is None:
        raise ImportError("缺少 h5py，请先执行 pip install h5py")
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        if "I" not in keys or "Q" not in keys:
            raise ValueError("文件不含 I/Q 通道，实际键为: %s" % keys)
        i_data = np.asarray(f["I"][()]).reshape(-1)
        q_data = np.asarray(f["Q"][()]).reshape(-1)
    if i_data.size != q_data.size:
        raise ValueError("I/Q 长度不一致: %d vs %d" % (i_data.size, q_data.size))
    iq = i_data.astype(np.float32) + 1j * q_data.astype(np.float32)
    meta = {
        "keys": keys,
        "dtype": str(i_data.dtype),
        "itemsize": int(i_data.dtype.itemsize),
        "n_samples": int(i_data.size),
        "i_shape": tuple(np.asarray(i_data.shape)),
        "q_shape": tuple(np.asarray(q_data.shape)),
    }
    return iq, meta


def channel_stats(x, fs_hz):
    """统计单通道（实信号）的幅度与功率指标。"""
    xf = np.asarray(x, dtype=np.float64)
    n = int(xf.size)
    power = float(np.mean(xf ** 2)) if n else 0.0
    peak = float(np.max(np.abs(xf))) if n else 0.0
    return {
        "n": n,
        "duration_s": n / float(fs_hz) if fs_hz else float("nan"),
        "mean": float(np.mean(xf)) if n else 0.0,
        "std": float(np.std(xf)) if n else 0.0,
        "peak_abs": peak,
        "rms": float(np.sqrt(power)),
        "power_db": 10.0 * np.log10(power) if power > 0 else float("-inf"),
        "nan_count": int(np.count_nonzero(~np.isfinite(xf))),
        "zero_ratio": float(np.mean(xf == 0)) if n else 0.0,
        "crest_db": (20.0 * np.log10(peak / np.sqrt(power))
                     if power > 0 and peak > 0 else float("nan")),
    }


def estimate_snr_db(iq, segment_len=4096, max_segments=512,
                    signal_over_floor_db=10.0):
    """用平均周期图粗估带内信噪比（无外部真值，仅作量级参考）。

    思路：把片段切成若干等长小段，各段加 Hann 窗后做 FFT 并取功率平均，
    得到平均功率谱；以谱的中位数作为噪声底，高于底 ``signal_over_floor_db``
    的频点视为信号占用频点。SNR = 10log10(信号频点功率和 / 噪声频点功率和)。
    """
    x = np.asarray(iq, dtype=np.complex64)
    if x.size < segment_len * 2:
        return {"snr_db": float("nan"), "occupied_ratio": float("nan"),
                "noise_floor_db": float("nan")}
    hop = max(segment_len, x.size // max_segments)
    hop = hop - (hop % segment_len) if hop > segment_len else segment_len
    starts = range(0, x.size - segment_len + 1, hop)
    win = np.hanning(segment_len).astype(np.float32)
    acc = np.zeros(segment_len, dtype=np.float64)
    used = 0
    for s in starts:
        if used >= max_segments:
            break
        seg = x[s:s + segment_len] * win
        spec = np.fft.fftshift(np.fft.fft(seg))
        acc += np.abs(spec) ** 2
        used += 1
    if used == 0:
        return {"snr_db": float("nan"), "occupied_ratio": float("nan"),
                "noise_floor_db": float("nan")}
    psd = acc / used
    floor = float(np.median(psd))
    if floor <= 0:
        return {"snr_db": float("nan"), "occupied_ratio": float("nan"),
                "noise_floor_db": float("nan")}
    mask = psd > floor * (10.0 ** (signal_over_floor_db / 10.0))
    sig_power = float(np.sum(psd[mask]))
    noise_power = float(np.sum(psd[~mask]))
    ratio = (sig_power / noise_power) if noise_power > 0 else 0.0
    snr = 10.0 * np.log10(ratio) if ratio > 0 else float("nan")
    return {
        "snr_db": snr,
        "occupied_ratio": float(np.mean(mask)),
        "noise_floor_db": 10.0 * np.log10(floor),
        "segments_used": used,
    }


def probe_file(path, fs_hz, snr_cfg=None):
    """对单个文件做完整探针，返回一行汇总 dict；失败时返回错误行。"""
    snr_cfg = snr_cfg or {}
    row = {"file": os.path.basename(path), "path": path, "ok": False,
           "error": ""}
    try:
        iq, meta = load_iq(path)
        row.update(meta)
        row["duration_s"] = meta["n_samples"] / float(fs_hz)
        row["complex_power_db"] = (
            10.0 * np.log10(float(np.mean(np.abs(iq) ** 2)))
            if iq.size else float("-inf"))
        row["complex_peak"] = float(np.max(np.abs(iq))) if iq.size else 0.0
        snr = estimate_snr_db(
            iq,
            segment_len=int(snr_cfg.get("segment_len", 4096)),
            max_segments=int(snr_cfg.get("max_segments", 512)),
            signal_over_floor_db=float(snr_cfg.get("signal_over_floor_db", 10.0)),
        )
        row.update(snr)
        row["ok"] = True
    except Exception as exc:  # 单个文件失败不影响整体流程
        row["error"] = "%s: %s" % (type(exc).__name__, exc)
    return row


def probe_many(paths, fs_hz, snr_cfg=None, logger=None):
    """批量探针，逐个打印进度。"""
    log = logger or logging.getLogger(__name__)
    rows = []
    for idx, p in enumerate(paths, 1):
        row = probe_file(p, fs_hz, snr_cfg)
        rows.append(row)
        if row["ok"]:
            log.info("[%d/%d] %-42s dtype=%-8s n=%-8d 时长=%.1f ms  SNR≈%.1f dB",
                     idx, len(paths), row["file"], row["dtype"],
                     row["n_samples"], row["duration_s"] * 1000.0,
                     row.get("snr_db", float("nan")))
        else:
            log.warning("[%d/%d] %-42s 失败：%s",
                        idx, len(paths), row["file"], row["error"])
    return rows
