# -*- coding: utf-8 -*-
"""Q2 stage_1：单个信号的 STFT 时频图，按「原版尺寸」保存。

数据区严格 1 帧 = 1 像素、1 个频点 = 1 像素（不做任何降采样），
用于查看不同时长/帧长下 PNG 的真实体积。坐标轴、色标照常保留，额外占像素。

用法::

    python python/Q2/stage_1/stft_native_size.py                      # 5 ms / 2048 点
    python python/Q2/stage_1/stft_native_size.py --show-ms 50         # 整段 50 ms
    python python/Q2/stage_1/stft_native_size.py --show-ms full --signal C
    python python/Q2/stage_1/stft_native_size.py --frame-len 512 --show-ms 10
"""

import argparse
import logging
import struct
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import stft_explore as se

IMAGE_DIR = se.PROJECT_ROOT / "image" / "Q2"
LOG_PATH = se.PROJECT_ROOT / "log" / "Q2" / "stft_native_size.log"

# 数据区之外的留白（像素）：左轴标签、右色标、下横轴、上留白
PAD_LEFT, PAD_RIGHT, PAD_BOTTOM, PAD_TOP = 110, 30, 70, 40
CBAR_GAP, CBAR_W = 24, 16


def parse_args(cfg, nat):
    """命令行参数；默认值全部取自 config.json 的 native_size 段。"""
    parser = argparse.ArgumentParser(description="单信号 STFT 原版尺寸出图")
    parser.add_argument("--signal", default=str(nat.get("signal", "A")),
                        help="型号（A/C/D/E/F/G/BG），不存在则用第一个")
    parser.add_argument("--frame-len", type=int,
                        default=int(nat.get("frame_len", 2048)), help="帧长（点数）")
    parser.add_argument("--hop-ratio", type=float,
                        default=float(nat.get("hop_ratio") or cfg["hop_ratio"]),
                        help="帧移 / 帧长")
    parser.add_argument("--show-ms", default=str(nat.get("show_ms", "config")),
                        help='"config" 用 config.json 的 show_ms；数字为毫秒；"full" 用整段')
    parser.add_argument("--dpi", type=int, default=int(nat.get("dpi", 100)),
                        help="必须与像素换算一致，一般不用改")
    return parser.parse_args()


def resolve_show_ms(text, cfg, fs, n_total):
    """把 --show-ms 的三种写法解析成毫秒数。"""
    key = str(text).strip().lower()
    if key == "full":
        return n_total / fs * 1e3
    if key == "config":
        return float(cfg["show_ms"])
    return float(text)


def png_pixel_size(path):
    """读 PNG 头拿像素尺寸（宽, 高）。"""
    with open(path, "rb") as fh:
        head = fh.read(24)
    return (struct.unpack(">I", head[16:20])[0],
            struct.unpack(">I", head[20:24])[0])


def render_native(db, t_ms, f_mhz, fc_mhz, pic, out_png, dpi):
    """数据区 1 帧 = 1 px、1 bin = 1 px 地渲染保存，返回数据区 (宽, 高)。"""
    # db 是 [帧, 频点]；转置后画出来是 行=频点、列=帧 → 宽 = 帧数，高 = 频点数
    w_px, h_px = db.shape[0], db.shape[1]
    fig_w = w_px + PAD_LEFT + PAD_RIGHT + CBAR_GAP + CBAR_W
    fig_h = h_px + PAD_BOTTOM + PAD_TOP
    fig = plt.figure(figsize=(fig_w / dpi, fig_h / dpi), dpi=dpi)
    ax = fig.add_axes([PAD_LEFT / fig_w, PAD_BOTTOM / fig_h,
                       w_px / fig_w, h_px / fig_h])
    mesh = ax.imshow(db.T, origin="lower", aspect="auto", interpolation="nearest",
                     vmin=float(pic["stft"]["db_floor"]), vmax=0.0,
                     cmap=pic["stft"]["colormap"],
                     extent=[t_ms[0], t_ms[-1],
                             fc_mhz + f_mhz[0], fc_mhz + f_mhz[-1]])
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (MHz)")
    cax = fig.add_axes([(PAD_LEFT + w_px + CBAR_GAP) / fig_w, PAD_BOTTOM / fig_h,
                        CBAR_W / fig_w, h_px / fig_h])
    fig.colorbar(mesh, cax=cax).set_label("Magnitude (dB re. peak)")
    fig.savefig(out_png, dpi=dpi)          # 不用 bbox_inches，否则像素数会变
    plt.close(fig)
    return w_px, h_px


def main():
    """串流程：选信号 → 截时长 → STFT → 1:1 出图 → 打印真实尺寸与体积。"""
    cfg = se.load_json(se.CFG_PATH)
    pic = se.load_json(se.PIC_CFG_PATH)
    se.setup_logging(LOG_PATH)
    se.apply_pic_style(pic)
    args = parse_args(cfg, cfg.get("native_size", {}))

    fs = float(cfg["fs_hz"])
    fc_mhz = float(cfg["center_freq_hz"]) / 1e6
    db_floor = float(pic["stft"]["db_floor"])

    data_dir = Path(cfg["data_dir"])
    samples = se.pick_one_per_model(sorted(data_dir.glob(cfg["glob_pattern"])), cfg)
    key = args.signal if args.signal in samples else next(iter(samples))
    path = samples[key]
    iq = se.load_iq(path)

    show_ms = resolve_show_ms(args.show_ms, cfg, fs, iq.size)
    n_show = min(iq.size, int(round(show_ms * 1e-3 * fs)))
    frame_len = int(args.frame_len)
    hop = max(1, int(round(frame_len * float(args.hop_ratio))))

    db, t_ms, f_mhz = se.stft_db(iq[:n_show], fs, frame_len, hop, frame_len,
                                 db_floor, cfg["window"])
    out_png = IMAGE_DIR / ("stft_native_f%d_%.3gms.png" % (frame_len, show_ms))
    w_px, h_px = render_native(db, t_ms, f_mhz, fc_mhz, pic, out_png, args.dpi)

    w_png, h_png = png_pixel_size(out_png)
    size_mb = out_png.stat().st_size / 1024 / 1024
    logging.info("信号 %s（%s）", key, path.name)
    logging.info("时长 %.4g ms | 帧长 %d | 帧移 %d | 帧数 %d | 频点 %d",
                 n_show / fs * 1e3, frame_len, hop, db.shape[0], db.shape[1])
    logging.info("数据区 %d x %d px（1 帧 = 1 px，1 bin = 1 px）", w_px, h_px)
    logging.info("整图 %d x %d px | 文件 %.2f MB", w_png, h_png, size_mb)
    logging.info("输出 %s", out_png)


if __name__ == "__main__":
    main()
