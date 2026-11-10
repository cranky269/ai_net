# -*- coding: utf-8 -*-
"""Q2.5 跨阶段通用层：路径常量、JSON/日志、STFT、复用 Q2 的加噪组件。

Q2.5 目录名含点（``Q2.5``），不能作为普通 Python 包导入；各阶段脚本先把
``python/Q2.5/common`` 加入 ``sys.path``，再 ``import q25_common``。
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np

# ==================== 路径常量 ====================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
Q25_DIR = Path(__file__).resolve().parents[1]

Q2_DIR = PROJECT_ROOT / "python" / "Q2"
Q2_STAGE1_CONFIG = Q2_DIR / "stage_1" / "config.json"
Q2_STAGE2_DIR = Q2_DIR / "stage_2"
Q2_STAGE2_CONFIG = Q2_STAGE2_DIR / "config.json"

PIC_CFG_PATH = PROJECT_ROOT / "config_pic.json"

IMAGE_DIR = PROJECT_ROOT / "image" / "Q2.5"
RESULT_DIR = PROJECT_ROOT / "result" / "Q2.5"
LOG_DIR = PROJECT_ROOT / "log" / "Q2.5"

# 随机流隔离标记：出图与探针共用同一条噪声，使“图”与“表”可互相对照；
# 与 Q2 的 PLOT_SALT、正式批量加噪的随机流三者互相独立。
SAMPLE_SALT = 777
# =================================================


def load_json(path):
    """读取 UTF-8 JSON 文件，返回 dict。"""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def setup_logging(log_path):
    """日志同时写文件（UTF-8）与控制台。"""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)],
    )


def import_q2_awgn():
    """把 Q2 stage_2 加入 sys.path，返回其 ``awgn`` 模块。

    复用其中的 ``load_iq`` / ``signal_power`` / ``sigma_for_snr`` /
    ``generate_noise`` / ``make_rng``，保证 Q2.5 与 Q2 的加噪语义完全一致。
    """
    path = str(Q2_STAGE2_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)
    import awgn  # noqa: E402  （需先插入 sys.path）

    return awgn


def apply_pic_style(pic):
    """把 config_pic.json 的字体、字号、边框样式应用到 matplotlib 全局。"""
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = pic["font_family"]
    plt.rcParams["font.size"] = pic["font_size"]
    plt.rcParams["axes.labelsize"] = pic["axes"]["labelsize"]
    plt.rcParams["axes.titlesize"] = pic["axes"]["titlesize"]
    plt.rcParams["axes.spines.top"] = pic["axes"]["spines_top"]
    plt.rcParams["axes.spines.right"] = pic["axes"]["spines_right"]
    plt.rcParams["figure.facecolor"] = pic["figure"]["facecolor"]
    plt.rcParams["axes.unicode_minus"] = False


def stft_mag(x, fs, frame_len, hop, n_fft, window="hann", chunk=8192):
    """复数 STFT，返回 (幅度矩阵[帧, 频点], 时间轴 ms, 频率轴 MHz)。

    与 ``python/Q2/stage_2/plot_noise_compare.py`` 的实现保持一致（分块 FFT）。
    """
    if x.size < frame_len:
        raise ValueError("信号仅 %d 点，短于帧长 %d" % (x.size, frame_len))
    n_frames = 1 + (x.size - frame_len) // hop
    win = (np.hanning(frame_len) if window == "hann"
           else np.ones(frame_len)).astype(np.float32)
    mag = np.empty((n_frames, n_fft), dtype=np.float32)
    for start in range(0, n_frames, chunk):
        stop = min(start + chunk, n_frames)
        idx = (np.arange(frame_len)[None, :]
               + hop * np.arange(start, stop)[:, None])
        spec = np.fft.fftshift(np.fft.fft(x[idx] * win, n=n_fft, axis=1), axes=1)
        mag[start:stop] = np.abs(spec).astype(np.float32)
    t_ms = (np.arange(n_frames) * hop + frame_len / 2.0) / fs * 1e3
    f_mhz = np.fft.fftshift(np.fft.fftfreq(n_fft, 1.0 / fs)) / 1e6
    return mag, t_ms, f_mhz


def pool_frames_max(arr, max_frames):
    """沿时间轴（第 0 轴）把帧数压到不超过 ``max_frames``，每段取最大值。

    出图时一帧远小于一个像素（6 万帧画在约 1600 px 宽的图上），逐帧保留只是
    白占内存（单格 float32 就有 61 MB）。取**最大值**而非均值，可保住突发与
    窄带峰。返回新数组，原数组不被修改。
    """
    n = int(arr.shape[0])
    m = min(int(max_frames), n)
    edges = np.linspace(0, n, m + 1).astype(np.int64)
    return np.maximum.reduceat(arr, edges[:-1], axis=0)


def mag_to_db(mag, ref, floor, max_frames=None):
    """幅度转 dB（相对统一参考 ref），并截断到色标下限。

    ``max_frames`` 非空时先沿时间轴做最大值池化再取 dB（见
    :func:`pool_frames_max`）。dB 对幅度单调、缩放也是线性，因此“先池化再取
    dB”与“先取 dB 再池化”结果相同，但峰值内存从整幅降到池化后的尺寸。
    """
    arr = np.asarray(mag, dtype=np.float32)
    if max_frames is not None and arr.shape[0] > int(max_frames):
        arr = pool_frames_max(arr, int(max_frames))
    db = np.multiply(arr, np.float32(1.0 / (float(ref) + 1e-12)))
    db += np.float32(1e-12)
    np.log10(db, out=db)
    db *= np.float32(20.0)
    np.maximum(db, np.float32(floor), out=db)
    return db


def resolve_show_len(cfg_q2, fs, override=None):
    """解析参与 STFT 的采样点数：override 优先，其次 Q2 的 show_ms。"""
    value = override if override is not None else cfg_q2.get("show_ms", 50.0)
    if isinstance(value, str) and value.strip().lower() == "full":
        return None
    return int(round(float(value) * 1e-3 * fs))


def stft_params(cfg_q2):
    """从 Q2 stage_1 config 提取 STFT 参数元组。

    返回 ``(fs, frame_len, hop, n_fft, window)``，可直接 ``*`` 展开传给
    :func:`stft_mag`。
    """
    fs = float(cfg_q2["fs_hz"])
    frame_len = int(cfg_q2["frame_len"])
    hop = max(1, int(round(frame_len * float(cfg_q2["hop_ratio"]))))
    n_fft = int(cfg_q2["n_fft"]) if cfg_q2.get("n_fft") else frame_len
    window = str(cfg_q2.get("window", "hann"))
    return fs, frame_len, hop, n_fft, window
