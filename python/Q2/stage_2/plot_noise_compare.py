# -*- coding: utf-8 -*-
"""Q2 stage_2：加噪前后 STFT 对比图（3 型号 × 无噪 + 各档 SNR）。

数据链路：读划分表选样 → 读原始 IQ → 原始 STFT（确定 dB 参考）
          → 按目标 SNR 加噪 → 用同一参考画加噪后 STFT。

所有 STFT 参数（采样率、中心频率、帧长、帧移、窗、FFT 点数、频率显示范围、
时长）均读 ``python/Q2/stage_1/config.json``；配色尺寸读项目根
``config_pic.json``；加噪参数读本阶段 ``config.json``。

用法::

    python python/Q2/stage_2/plot_noise_compare.py          # 存图
    python python/Q2/stage_2/plot_noise_compare.py --show   # 存图并弹窗
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

if "--show" not in sys.argv:
    import matplotlib

    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from awgn import (add_awgn, load_iq, load_json, load_split_table, make_rng,
                  split_to_id)

# ==================== 可调参数 ====================
# 对比图专用“盐”：与正式批量加噪的随机流隔离，调图不会污染正式数据
PLOT_SALT = 999
# STFT 分块帧数：控制 FFT 内存峰值（越大越快、越费内存）
STFT_CHUNK = 8192
# =================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STAGE2_DIR = Path(__file__).resolve().parent
CFG_S2_PATH = STAGE2_DIR / "config.json"
CFG_S1_PATH = PROJECT_ROOT / "python" / "Q2" / "stage_1" / "config.json"
PIC_CFG_PATH = PROJECT_ROOT / "config_pic.json"
LOG_PATH = PROJECT_ROOT / "log" / "Q2" / "stage2_awgn_compare.log"


def setup_logging(log_path):
    """日志同时写文件（UTF-8）与控制台。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)],
    )


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


def parse_args():
    """命令行参数：--show 控制是否弹窗。"""
    parser = argparse.ArgumentParser(description="加噪前后 STFT 对比图")
    parser.add_argument("--show", action="store_true", help="出图后弹窗显示")
    return parser.parse_args()


def pick_samples(table, cfg_s2):
    """每个型号按文件名排序取第一条，返回 [(model, file), ...]。"""
    models = [str(m) for m in cfg_s2["plot_models"]]
    split = str(cfg_s2["plot_split"])
    scene = cfg_s2.get("plot_scene")
    picked = []
    for model in models:
        cand = [r for r in table
                if str(r.get("split")) == split
                and str(r.get("model")) == model
                and (scene is None or str(r.get("scene")) == str(scene))]
        cand.sort(key=lambda r: str(r.get("file")))
        if not cand:
            raise SystemExit("型号 %s 在 %s/%s 下无样本" % (model, split, scene))
        picked.append((model, str(cand[0]["file"])))
    return picked


def resolve_show_len(cfg_s1, cfg_s2, fs):
    """解析参与 STFT 的采样点数：数字→毫秒换算；"full"/None→整段。"""
    value = cfg_s2.get("plot_show_ms")
    if value is None:
        value = cfg_s1.get("show_ms", 50.0)
    if isinstance(value, str) and value.strip().lower() == "full":
        return None
    return int(round(float(value) * 1e-3 * fs))


def stft_mag(x, fs, frame_len, hop, n_fft, window="hann", chunk=STFT_CHUNK):
    """复数 STFT，返回 (幅度矩阵[帧, 频点], 时间轴 ms, 频率轴 MHz)。

    分块计算，避免一次性构造超大的帧索引矩阵。
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


def mag_to_db(mag, ref, floor):
    """幅度转 dB（相对统一参考 ref），并截断到色标下限。"""
    db = 20.0 * np.log10(mag / (float(ref) + 1e-12) + 1e-12)
    return np.maximum(db, float(floor))


def build_row(model_index, model, file, cfg_s1, cfg_s2, snr_list, n_show, floor):
    """构造一行数据：无噪 STFT + 各档 SNR 加噪 STFT（共用同一 dB 参考）。"""
    fs = float(cfg_s1["fs_hz"])
    frame_len = int(cfg_s1["frame_len"])
    hop = max(1, int(round(frame_len * float(cfg_s1["hop_ratio"]))))
    n_fft = int(cfg_s1["n_fft"]) if cfg_s1.get("n_fft") else frame_len
    window = str(cfg_s1.get("window", "hann"))
    split = str(cfg_s2["plot_split"])
    split_id = split_to_id(split)

    data_path = PROJECT_ROOT / str(cfg_s2["data_dir"]) / file
    iq = load_iq(data_path)
    if n_show is not None:
        iq = iq[:n_show]

    mag0, t_ms, f_mhz = stft_mag(iq, fs, frame_len, hop, n_fft, window)
    ref = float(mag0.max())
    clean_db = mag_to_db(mag0, ref, floor)
    logging.info("  %-2s %-28s 无噪峰值参考 %.4g", model, file, ref)

    noisy_items = []
    for snr_idx, snr_db in enumerate(snr_list):
        rng = make_rng(cfg_s2["base_seed"], split_id, snr_idx,
                       extra=[PLOT_SALT, model_index])
        noisy, meta = add_awgn(iq, float(snr_db), rng)
        mag_n, _, _ = stft_mag(noisy, fs, frame_len, hop, n_fft, window)
        noisy_items.append((float(snr_db), mag_to_db(mag_n, ref, floor), meta))
        logging.info("      SNR %+6.1f dB -> 实测 %+6.2f dB (sigma=%.4g)",
                     float(snr_db), meta["snr_db_measured"], meta["sigma"])

    return {"model": model, "file": file, "clean_db": clean_db,
            "noisy": noisy_items, "t_ms": t_ms, "f_mhz": f_mhz}


def plot_compare(rows, cfg_s1, pic, snr_list, out_png):
    """画 型号×（无噪+各档 SNR）网格图，共享坐标轴与色标。"""
    fc_mhz = float(cfg_s1["center_freq_hz"]) / 1e6
    freq_limit = float(cfg_s1["freq_limit_mhz"])
    floor = float(pic["stft"]["db_floor"])
    ncol = 1 + len(snr_list)
    nrow = len(rows)

    fig, axes = plt.subplots(nrow, ncol,
                             figsize=pic["stage2"]["compare_figsize"],
                             sharex=True, sharey=True, squeeze=False)
    mesh = None
    for r, row in enumerate(rows):
        band = np.abs(row["f_mhz"]) <= freq_limit
        f_low = fc_mhz + float(row["f_mhz"][band][0])
        f_high = fc_mhz + float(row["f_mhz"][band][-1])
        extent = [float(row["t_ms"][0]), float(row["t_ms"][-1]), f_low, f_high]
        columns = [("clean", row["clean_db"], None)]
        columns += [(snr, db, meta) for snr, db, meta in row["noisy"]]
        for c, (label, db, meta) in enumerate(columns):
            ax = axes[r, c]
            mesh = ax.imshow(db[:, band].T, origin="lower", aspect="auto",
                             interpolation="nearest", vmin=floor, vmax=0.0,
                             cmap=pic["stft"]["colormap"], extent=extent)
            if c == 0:
                ax.set_title("%s · %s" % (row["model"], label))
                ax.set_ylabel("Frequency (MHz)")
            else:
                ax.set_title("SNR %+.0f dB (meas %+.1f)"
                             % (float(label), meta["snr_db_measured"]))
            if r == nrow - 1:
                ax.set_xlabel("Time (ms)")

    fig.suptitle("AWGN before/after STFT | %s | frame %d pts | shared dB ref (clean peak)"
                 % (str(cfg_s1.get("scene", "")), int(cfg_s1["frame_len"])),
                 fontsize=pic["axes"]["titlesize"])
    cax = fig.add_axes([0.92, 0.12, 0.012, 0.76])
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label("Magnitude (dB re. clean peak)")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=pic["figure"]["dpi"],
                bbox_inches=pic["save"]["bbox_inches"],
                pad_inches=pic["save"]["pad_inches"])
    plt.close(fig)
    return out_png


def main():
    """串起流程：读配置 → 选样 → 逐行加噪算 STFT → 出图。"""
    setup_logging(LOG_PATH)
    cfg_s2 = load_json(CFG_S2_PATH)
    cfg_s1 = load_json(CFG_S1_PATH)
    pic = load_json(PIC_CFG_PATH)
    apply_pic_style(pic)
    args = parse_args()

    fs = float(cfg_s1["fs_hz"])
    snr_list = [float(v) for v in cfg_s2["snr_db_list"]]
    floor = float(pic["stft"]["db_floor"])
    n_show = resolve_show_len(cfg_s1, cfg_s2, fs)

    table = load_split_table(str(PROJECT_ROOT / cfg_s2["split_table"]),
                             cfg_s2.get("split_sheet", "split"))
    samples = pick_samples(table, cfg_s2)
    logging.info("选样（%s / %s）：", cfg_s2["plot_split"], cfg_s2.get("plot_scene"))
    for model, file in samples:
        logging.info("  %-2s %s", model, file)
    logging.info("SNR 档位: %s dB | 参与 STFT 时长: %s",
                 snr_list, "整段" if n_show is None else "%.1f ms" % (n_show / fs * 1e3))

    rows = []
    for idx, (model, file) in enumerate(samples):
        rows.append(build_row(idx, model, file, cfg_s1, cfg_s2,
                              snr_list, n_show, floor))

    out_png = PROJECT_ROOT / str(cfg_s2["out_png"])
    plot_compare(rows, cfg_s1, pic, snr_list, out_png)
    logging.info("对比图已保存: %s", out_png)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
