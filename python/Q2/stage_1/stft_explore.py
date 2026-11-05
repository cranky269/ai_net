# -*- coding: utf-8 -*-
"""Q2 stage_1：IQ 信号 STFT 时频图（帧长可调）。

从训练集每个型号各取 1 条 IQ 记录（统一 IN_S0 视距场景，background 单独 1 条），
做短时傅里叶变换并画成多子图，用来比较不同帧长下时间分辨率与频率分辨率的取舍。

用法::

    python python/Q2/stage_1/stft_explore.py                     # 用 config.json 的帧长
    python python/Q2/stage_1/stft_explore.py --frame-len 512     # 临时试别的帧长
    python python/Q2/stage_1/stft_explore.py --frame-len 1024 --freq-limit 10 --show

参数优先级：命令行 > ``config.json``。
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import h5py

# --show 时保留交互后端以便弹窗；否则用 Agg，服务器/无界面也能存图
if "--show" not in sys.argv:
    import matplotlib

    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==================== 可调参数（也可用命令行覆盖）====================
FRAME_LEN = None      # None → 用 config.json 里的 frame_len；这里填数字会覆盖它
HOP_RATIO = None      # None → 用 config.json 里的 hop_ratio
FREQ_LIMIT_MHZ = None # None → 用 config.json 里的 freq_limit_mhz，40 = 全带
# ===================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CFG_PATH = Path(__file__).with_name("config.json")
PIC_CFG_PATH = PROJECT_ROOT / "config_pic.json"
IMAGE_DIR = PROJECT_ROOT / "image" / "Q2"
LOG_PATH = PROJECT_ROOT / "log" / "Q2" / "stft_explore.log"

# 帧长候选：--sweep 会逐个出图，对照表也按这张表打印
FRAME_LEN_CANDIDATES = (256, 384, 512, 768, 1024, 2048,4096,8192,40000)


def setup_logging(log_path):
    """日志同时写文件（UTF-8）与控制台。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)],
    )


def load_json(path):
    """读取 UTF-8 JSON 文件，返回 dict。"""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def apply_pic_style(pic):
    """把 config_pic.json 的字体、字号、边框样式应用到 matplotlib 全局。"""
    plt.rcParams["font.family"] = pic["font_family"]
    plt.rcParams["font.size"] = pic["font_size"]
    plt.rcParams["axes.labelsize"] = pic["axes"]["labelsize"]
    plt.rcParams["axes.titlesize"] = pic["axes"]["titlesize"]
    plt.rcParams["axes.spines.top"] = pic["axes"]["spines_top"]
    plt.rcParams["axes.spines.right"] = pic["axes"]["spines_right"]
    plt.rcParams["figure.facecolor"] = pic["figure"]["facecolor"]
    plt.rcParams["axes.unicode_minus"] = False


def parse_args(cfg):
    """解析命令行参数；默认值取自 config.json。"""
    parser = argparse.ArgumentParser(description="IQ 信号 STFT 时频图（帧长可调）")
    parser.add_argument("--frame-len", type=int,
                        default=FRAME_LEN or int(cfg["frame_len"]),
                        help="帧长（采样点数），默认取 config.json")
    parser.add_argument("--hop-ratio", type=float,
                        default=HOP_RATIO or float(cfg["hop_ratio"]),
                        help="帧移 / 帧长，0.25 表示 75%% 重叠")
    parser.add_argument("--freq-limit", type=float,
                        default=FREQ_LIMIT_MHZ or float(cfg["freq_limit_mhz"]),
                        help="频率显示上限 MHz，40 = 全带")
    parser.add_argument("--sweep", action="store_true",
                        help="遍历 FRAME_LEN_CANDIDATES 全部帧长各出一张图，"
                             "并额外生成一张同信号的帧长对比图")
    parser.add_argument("--compare-signal",
                        default=str(cfg.get("sweep_compare_signal", "C")),
                        help="--sweep 生成对比图时用哪个型号（默认取 config.json）；"
                             "不存在则用第一个型号")
    parser.add_argument("--show", action="store_true", help="出图后弹窗显示")
    return parser.parse_args()


def parse_name(path, cfg):
    """从 `个体_场景[_S批次]_slice_N.mat` 解析型号、个体、场景；background 无场景。"""
    stem = path.stem
    if stem.startswith(cfg["background_prefix"]):
        return {"model": "BG", "unit": "background", "scene": "-"}
    head, _, _ = stem.rpartition(cfg["slice_sep"])
    parts = head.split(cfg["scene_sep"])
    unit = parts[0]
    scene = cfg["scene_sep"].join(parts[1:]) if len(parts) > 1 else "-"
    return {"model": unit[0], "unit": unit, "scene": scene}


def pick_one_per_model(paths, cfg):
    """每个型号保留 1 条：仅保留指定场景并按文件名排序取第一条；background 单列一组。"""
    buckets = {}
    for p in paths:
        info = parse_name(p, cfg)
        if info["model"] != "BG" and info["scene"] != cfg["scene"]:
            continue
        buckets.setdefault(info["model"], []).append(p)
    return {m: sorted(v)[0] for m, v in sorted(buckets.items())}


def load_iq(path):
    """读取 HDF5 格式 .mat 的 I/Q 通道，返回复基带信号（complex64）。"""
    with h5py.File(path, "r") as fh:
        keys = list(fh.keys())
        if not {"I", "Q"}.issubset(keys):
            raise ValueError("%s 缺少 I/Q 数据集，实际键: %s" % (path.name, keys))
        i_data = np.asarray(fh["I"][()], dtype=np.float32).reshape(-1)
        q_data = np.asarray(fh["Q"][()], dtype=np.float32).reshape(-1)
    if i_data.size != q_data.size:
        raise ValueError("%s 的 I/Q 长度不一致: %d vs %d"
                         % (path.name, i_data.size, q_data.size))
    return (i_data + 1j * q_data).astype(np.complex64)


def stft_db(x, fs, frame_len, hop, n_fft, db_floor, window="hann"):
    """复数 STFT，返回 (dB 矩阵[帧, 频点], 时间轴 ms, 频率轴 MHz)，dB 相对峰值。"""
    if x.size < frame_len:
        raise ValueError("信号仅 %d 点，短于帧长 %d" % (x.size, frame_len))
    n_frames = 1 + (x.size - frame_len) // hop
    idx = np.arange(frame_len)[None, :] + hop * np.arange(n_frames)[:, None]
    win = (np.hanning(frame_len) if window == "hann"
           else np.ones(frame_len)).astype(np.float32)
    spec = np.fft.fftshift(np.fft.fft(x[idx] * win, n=n_fft, axis=1), axes=1)
    mag = np.abs(spec)
    db = 20.0 * np.log10(mag / (mag.max() + 1e-12) + 1e-12)
    np.maximum(db, db_floor, out=db)
    t_ms = (np.arange(n_frames) * hop + frame_len / 2.0) / fs * 1e3
    f_mhz = np.fft.fftshift(np.fft.fftfreq(n_fft, 1.0 / fs)) / 1e6
    return db, t_ms, f_mhz


def log_frame_table(fs, hop_ratio, current, show_ms):
    """把候选帧长的时间/频率分辨率打到日志，帮助挑选帧长。"""
    logging.info("帧长对照（当前 %d 点）：", current)
    logging.info("%8s %12s %16s %12s %10s", "帧长", "帧时长(us)",
                 "频率分辨率(kHz)", "时间步长(us)", "帧数")
    for n in FRAME_LEN_CANDIDATES:
        hop = max(1, int(round(n * hop_ratio)))
        n_frames = int((int(show_ms * 1e-3 * fs) - n) // hop + 1)
        logging.info("%8s %12.2f %16.2f %12.2f %10d",
                     ("%d *" % n) if n == current else str(n),
                     n / fs * 1e6, fs / n / 1e3, hop / fs * 1e6, n_frames)


def plot_grid(samples, cfg, pic, frame_len, hop, n_fft, n_show,
              freq_limit_mhz, db_floor, out_png):
    """每型号 1 个子图的 STFT 时频图，共享坐标轴与色标；返回保存路径。"""
    fs = float(cfg["fs_hz"])
    fc_mhz = float(cfg["center_freq_hz"]) / 1e6
    n = len(samples)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=pic["stft"]["grid_figsize"],
                             sharex=True, sharey=True, squeeze=False)
    mesh = None
    for ax, (model, path) in zip(axes.ravel(), samples.items()):
        iq = load_iq(path)[:n_show]
        db, t_ms, f_mhz = stft_db(iq, fs, frame_len, hop, n_fft, db_floor,
                                  cfg["window"])
        band = np.abs(f_mhz) <= freq_limit_mhz
        mesh = ax.imshow(db[:, band].T, origin="lower", aspect="auto",
                         interpolation="nearest", vmin=db_floor, vmax=0.0,
                         cmap=pic["stft"]["colormap"],
                         extent=[t_ms[0], t_ms[-1],
                                 fc_mhz + f_mhz[band][0], fc_mhz + f_mhz[band][-1]])
        info = parse_name(path, cfg)
        ax.set_title("%s  (%s · %s)" % (model, info["unit"], info["scene"]))
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Frequency (MHz)")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("STFT of %.3g ms IQ | frame %d pts (%.2f us, %.1f kHz/bin) | hop %d pts "
                 "| %.0f-%.0f MHz"
                 % (n_show / fs * 1e3, frame_len, frame_len / fs * 1e6,
                    fs / n_fft / 1e3, hop,
                    fc_mhz - freq_limit_mhz, fc_mhz + freq_limit_mhz),
                 fontsize=pic["axes"]["titlesize"])
    cax = fig.add_axes([0.92, 0.12, 0.012, 0.76])
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label("Magnitude (dB re. peak)")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=pic["figure"]["dpi"],
                bbox_inches=pic["save"]["bbox_inches"],
                pad_inches=pic["save"]["pad_inches"])
    return out_png


def plot_sweep_compare(samples, cfg, pic, frame_lens, hop_ratio, n_show,
                       freq_limit_mhz, db_floor, out_png, compare_model=None):
    """同一个信号在多个帧长下的时频图并排，直观比较时间/频率分辨率的取舍。"""
    fs = float(cfg["fs_hz"])
    fc_mhz = float(cfg["center_freq_hz"]) / 1e6
    if compare_model and compare_model in samples:
        model, path = compare_model, samples[compare_model]
    else:
        model, path = next(iter(samples.items()))
    iq = load_iq(path)[:n_show]
    ncol = 5
    nrow = int(np.ceil(len(frame_lens) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(15.0, 3.9 * nrow),
                             sharex=True, sharey=True, squeeze=False)
    mesh = None
    for ax, n in zip(axes.ravel(), frame_lens):
        hop = max(1, int(round(n * hop_ratio)))
        db, t_ms, f_mhz = stft_db(iq, fs, n, hop, n, db_floor, cfg["window"])
        band = np.abs(f_mhz) <= freq_limit_mhz
        mesh = ax.imshow(db[:, band].T, origin="lower", aspect="auto",
                         interpolation="nearest", vmin=db_floor, vmax=0.0,
                         cmap=pic["stft"]["colormap"],
                         extent=[t_ms[0], t_ms[-1],
                                 fc_mhz + f_mhz[band][0], fc_mhz + f_mhz[band][-1]])
        ax.set_title("frame %d  (%.1f kHz/bin)" % (n, fs / n / 1e3))
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Frequency (MHz)")
    for ax in axes.ravel()[len(frame_lens):]:
        ax.axis("off")
    fig.suptitle("Same IQ (%s, %s) under %d frame lengths | %.3g ms window | %.0f-%.0f MHz"
                 % (model, path.name, len(frame_lens), n_show / fs * 1e3,
                    fc_mhz - freq_limit_mhz, fc_mhz + freq_limit_mhz),
                 fontsize=pic["axes"]["titlesize"])
    cax = fig.add_axes([0.92, 0.12, 0.012, 0.76])
    fig.colorbar(mesh, cax=cax).set_label("Magnitude (dB re. peak)")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=pic["figure"]["dpi"],
                bbox_inches=pic["save"]["bbox_inches"],
                pad_inches=pic["save"]["pad_inches"])
    plt.close(fig)
    return out_png


def main():
    """串起流程：读配置 → 选样 → 算 STFT → 存图。"""
    cfg = load_json(CFG_PATH)
    pic = load_json(PIC_CFG_PATH)
    setup_logging(LOG_PATH)
    apply_pic_style(pic)
    args = parse_args(cfg)

    fs = float(cfg["fs_hz"])
    fc_mhz = float(cfg["center_freq_hz"]) / 1e6
    frame_len = int(args.frame_len)
    hop = max(1, int(round(frame_len * float(args.hop_ratio))))
    n_fft = int(cfg["n_fft"]) if cfg.get("n_fft") else frame_len
    n_show = int(round(float(cfg["show_ms"]) * 1e-3 * fs))

    data_dir = Path(cfg["data_dir"])
    if not data_dir.is_dir():
        raise SystemExit("数据目录不存在: %s" % data_dir)
    all_mat = sorted(data_dir.glob(cfg["glob_pattern"]))
    samples = pick_one_per_model(all_mat, cfg)
    logging.info("匹配 .mat %d 个，入选型号 %d 个（场景 %s + background）",
                 len(all_mat), len(samples), cfg["scene"])
    for model, path in samples.items():
        logging.info("  %-2s %s", model, path.name)

    log_frame_table(fs, float(args.hop_ratio), frame_len, float(cfg["show_ms"]))
    db_floor = float(pic["stft"]["db_floor"])
    freq_limit = float(args.freq_limit)

    if args.sweep:
        for n in FRAME_LEN_CANDIDATES:
            out_png = IMAGE_DIR / ("stft_frame_%04d.png" % n)
            plot_grid(samples, cfg, pic, n,
                      max(1, int(round(n * float(args.hop_ratio)))),
                      n, n_show, freq_limit, db_floor, out_png)
            plt.close("all")
            logging.info("已保存: %s", out_png)
        cmp_png = plot_sweep_compare(samples, cfg, pic, FRAME_LEN_CANDIDATES,
                                     float(args.hop_ratio), n_show, freq_limit,
                                     db_floor, IMAGE_DIR / "stft_sweep_compare.png",
                                     compare_model=args.compare_signal)
        logging.info("帧长对比图: %s", cmp_png)
    else:
        out_png = IMAGE_DIR / ("stft_frame_%04d.png" % frame_len)
        plot_grid(samples, cfg, pic, frame_len, hop, n_fft, n_show,
                  freq_limit, db_floor, out_png)
        logging.info("时频图已保存: %s", out_png)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
