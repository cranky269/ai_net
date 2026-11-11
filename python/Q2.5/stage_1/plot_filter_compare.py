# -*- coding: utf-8 -*-
"""Q2.5 stage_1：预处理前后的 STFT 对比图（中值滤波 / 维纳滤波各一张）。

排版：行 = 无滤波 + 该方法各参数档，列 = 各档 SNR；
同一列（同一 SNR）所有行共用同一段噪声，因此可横向比较不同参数的抑噪效果。
所有子图共用“原始无噪信号的 STFT 峰值”作为 0 dB 参考。

分组由 ``config.json`` 的 ``plots`` 决定：每组一张图，只画该组注册名对应的方法。
两张图共用同一套噪声种子，故中值图与维纳图的同一列可直接对照。

STFT 参数取 ``python/Q2/stage_1/config.json``；
配色尺寸取项目根 ``config_pic.json``；方法与 SNR 取本阶段 ``config.json``。

用法::

    python python/Q2.5/stage_1/plot_filter_compare.py
    python python/Q2.5/stage_1/plot_filter_compare.py --show
    python python/Q2.5/stage_1/plot_filter_compare.py --only wiener
"""

import argparse
import gc
import logging
import sys
from pathlib import Path

import numpy as np

if "--show" not in sys.argv:
    import matplotlib

    matplotlib.use("Agg")
import matplotlib.pyplot as plt

COMMON_DIR = Path(__file__).resolve().parents[1] / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

import processors as q25_proc  # noqa: E402
import q25_common as q25  # noqa: E402

# ==================== 可调参数 ====================
CFG_PATH = Path(__file__).with_name("config.json")
LOG_PATH = q25.LOG_DIR / "stage1_filter_compare.log"
# =================================================


def parse_args():
    """命令行参数：--show 控制是否弹窗，--only 只画指定分组。"""
    parser = argparse.ArgumentParser(description="预处理方法 STFT 对比图")
    parser.add_argument("--show", action="store_true", help="出图后弹窗显示")
    parser.add_argument("--only", default=None,
                        help="只画某个分组，取 config.json 的 plots[].name")
    return parser.parse_args()


def build_noise_list(iq, snr_list, base_seed, split_id, salt, awgn, noise_cfg=None):
    """按 SNR 预生成噪声段；出图与探针共用同一套种子，保证图数一致。

    加噪方案由 ``noise_cfg``（Q2 stage_2 的 config）决定：``power`` 为原有
    按整段功率反推，``higuchi`` 为分形分段后补足差额。分段统计只算一次，
    各档 SNR 复用。
    """
    stats = awgn.fractal_stats(iq, noise_cfg)
    noise_list = []
    for snr_idx, snr_db in enumerate(snr_list):
        rng = awgn.make_rng(int(base_seed), int(split_id), snr_idx,
                            extra=[int(salt), 0])
        noise, _ = awgn.make_noise(iq, float(snr_db), rng, noise_cfg, stats=stats)
        noise_list.append(noise)
    return noise_list


def select_group(processors, names):
    """按注册名（median_freq / wiener）筛选方法，保持 make_processors 的顺序。"""
    wanted = {str(n) for n in names}
    return [p for p in processors if str(p["name"]) in wanted]


def build_rows(iq_clean, noise_list, processors, params, floor, max_frames=None,
               db_ref="panel", row_specs=None):
    """构造绘图数据：每行一个处理器或一条方法链，每列一档 SNR。

    流程：加噪 → STFT 得幅度谱 → 按链顺序做幅度域预处理 → dB。

    ``row_specs`` 给出时按它建行（每行 ``{"label": 显示名, "chain": [方法 label, ...]}``，
    链按顺序把 ``fn`` 套起来；``chain`` 为空即“无滤波”基线行）；
    为 None 时沿用旧行为：首行无滤波 + 每个处理器一行。

    ``db_ref`` 决定 0 dB 参考：

    - ``panel``（默认）：每格用自己的幅度峰值，等价于逐子图归一化，
      直接读出各方法把噪底压到相对峰值多少 dB；
    - ``column``：同一列（同档 SNR）所有行共用该列未滤波含噪谱的峰值，
      便于看出各参数相对“不处理”衰减了多少。

    ``max_frames`` 透传给 :func:`q25_common.mag_to_db`，沿时间轴做最大值池化：
    帧数远超图像像素，逐帧保留只会让单张图占到 GB 级内存。
    """
    fs, frame_len, hop, n_fft, window = params
    by_label = {str(p["label"]): p["fn"] for p in processors}

    if row_specs:
        rows = [{"label": str(spec["label"]),
                 "chain": [str(c) for c in spec.get("chain", [])],
                 "cells": []} for spec in row_specs]
        for row in rows:
            for label in row["chain"]:
                if label not in by_label:
                    raise SystemExit("rows 引用了未注册的方法 %r；可用: %s"
                                     % (label, sorted(by_label)))
    else:
        rows = [{"label": "no filter", "chain": [], "cells": []}]
        rows += [{"label": p["label"], "chain": [str(p["label"])], "cells": []}
                 for p in processors]
    per_panel = str(db_ref).lower() == "panel"

    refs, t_ms, f_mhz = [], None, None
    for noise in noise_list:
        noisy = (iq_clean + noise).astype(np.complex64)
        mag0, t_ms, f_mhz = q25.stft_mag(noisy, fs, frame_len, hop, n_fft, window)
        col_ref = float(mag0.max())           # 列参考：该档 SNR 未滤波谱的峰值
        refs.append(col_ref)
        for row in rows:
            mag_p = mag0
            for label in row["chain"]:        # 按链顺序串联
                mag_p = by_label[label](mag_p)
            ref_p = float(mag_p.max()) if per_panel else col_ref
            row["cells"].append(q25.mag_to_db(mag_p, ref_p, floor, max_frames))
            if mag_p is not mag0:
                del mag_p
        del mag0, noisy
    return rows, t_ms, f_mhz, refs


def plot_grid(rows, t_ms, f_mhz, cfg_q2, pic, snr_list, sample_name, title, out_png):
    """绘制 行=滤波方式、列=SNR 的网格图，共享坐标轴与色标。"""
    fc_mhz = float(cfg_q2["center_freq_hz"]) / 1e6
    freq_limit = float(cfg_q2["freq_limit_mhz"])
    floor = float(pic["stft"]["db_floor"])
    band = np.abs(f_mhz) <= freq_limit
    extent = [float(t_ms[0]), float(t_ms[-1]),
              fc_mhz + float(f_mhz[band][0]), fc_mhz + float(f_mhz[band][-1])]

    nrow, ncol = len(rows), len(snr_list)
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=pic["q25"]["filter_figsize"],
                             sharex=True, sharey=True, squeeze=False)
    mesh = None
    for r, row in enumerate(rows):
        for c, db in enumerate(row["cells"]):
            ax = axes[r, c]
            mesh = ax.imshow(db[:, band].T, origin="lower", aspect="auto",
                             interpolation="nearest", vmin=floor, vmax=0.0,
                             cmap=pic["stft"]["colormap"], extent=extent)
            if r == 0:
                ax.set_title("SNR %+.0f dB" % float(snr_list[c]))
            if c == 0:
                ax.set_ylabel("%s\nFrequency (MHz)" % row["label"])
            if r == nrow - 1:
                ax.set_xlabel("Time (ms)")

    fig.suptitle("%s | %s | frame %d pts | per-column dB ref (no-filter peak)"
                 % (title, sample_name, int(cfg_q2["frame_len"])),
                 fontsize=pic["axes"]["titlesize"])
    cax = fig.add_axes([0.92, 0.12, 0.012, 0.76])
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label("Magnitude (dB re. no-filter peak of same SNR)")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=pic["figure"]["dpi"],
                bbox_inches=pic["save"]["bbox_inches"],
                pad_inches=pic["save"]["pad_inches"])
    plt.close(fig)
    return out_png


def resolve_plots(cfg_s1):
    """取出 plots 分组配置，并校验必填字段（methods 与 rows 二选一）。"""
    plots = cfg_s1.get("plots")
    if not plots:
        raise SystemExit("config.json 缺少 plots 分组（每张对比图一组）")
    for item in plots:
        for key in ("name", "title", "out_png"):
            if key not in item:
                raise SystemExit("plots 分组缺少字段 %r：%s" % (key, item))
        if not item.get("methods") and not item.get("rows"):
            raise SystemExit("plots 分组需要 methods 或 rows 之一：%s" % item["name"])
    return plots


def main():
    """串起流程：读配置 → 读样本 → 生成噪声 → 逐组算 STFT/滤波并出图。"""
    q25.setup_logging(LOG_PATH)
    pic = q25.load_json(q25.PIC_CFG_PATH)
    q25.apply_pic_style(pic)
    cfg_s1 = q25.load_json(CFG_PATH)
    cfg_q2 = q25.load_json(q25.Q2_STAGE1_CONFIG)
    cfg_q2s2 = q25.load_json(q25.Q2_STAGE2_CONFIG)
    args = parse_args()

    awgn = q25.import_q2_awgn()
    params = q25.stft_params(cfg_q2)
    fs = params[0]
    n_show = q25.resolve_show_len(cfg_q2, fs)
    floor = float(pic["stft"]["db_floor"])
    snr_list = [float(v) for v in cfg_s1["snr_db_list"]]
    max_frames = cfg_s1.get("plot_max_frames", 4096)
    max_frames = None if max_frames in (None, 0) else int(max_frames)
    logging.info("时间轴池化上限: %s 帧（null/0 表示逐帧保留）", max_frames)

    db_ref = str(cfg_s1.get("db_ref", "panel")).lower()
    if db_ref not in ("panel", "column"):
        raise SystemExit("db_ref 只能是 panel / column，收到 %r" % db_ref)
    logging.info("dB 参考口径: %s", db_ref)

    plots = resolve_plots(cfg_s1)
    if args.only:
        plots = [p for p in plots if str(p["name"]) == str(args.only)]
        if not plots:
            raise SystemExit("--only %s 未匹配到任何 plots 分组" % args.only)

    all_procs = q25_proc.make_processors(cfg_s1["methods"])
    logging.info("已注册方法: %s", [p["label"] for p in all_procs])

    data_dir = q25.PROJECT_ROOT / str(cfg_q2s2["data_dir"])
    sample = str(cfg_s1["plot_sample"])
    split_id = awgn.split_to_id(str(cfg_s1["plot_split"]))
    iq = awgn.load_iq(data_dir / sample)
    if n_show is not None:
        iq = iq[:n_show]
    logging.info("样本 %s | %d 点 | 功率 %.4g", sample, iq.size,
                 awgn.signal_power(iq))
    logging.info("SNR: %s dB", snr_list)

    noise_list = build_noise_list(iq, snr_list, cfg_s1["base_seed"],
                                  split_id, q25.SAMPLE_SALT, awgn, cfg_q2s2)

    for plot in plots:
        row_specs = plot.get("rows")
        if row_specs:
            pool = all_procs
            logging.info("[%s] 链式行: %s", plot["name"],
                         [str(r.get("label")) for r in row_specs])
        else:
            group = select_group(all_procs, plot["methods"])
            if not group:
                logging.warning("分组 %s 没有匹配到已注册方法 %s，跳过",
                                plot["name"], plot["methods"])
                continue
            pool = group
            logging.info("[%s] 方法: %s", plot["name"], [p["label"] for p in group])
        rows, t_ms, f_mhz, refs = build_rows(iq, noise_list, pool, params, floor,
                                             max_frames, db_ref, row_specs)
        logging.info("[%s] 各列 dB 参考（该列无滤波峰值）: %s",
                     plot["name"], ["%.3g" % r for r in refs])
        out_png = q25.PROJECT_ROOT / str(plot["out_png"])
        plot_grid(rows, t_ms, f_mhz, cfg_q2, pic, snr_list, sample,
                  str(plot["title"]), out_png)
        logging.info("[%s] 对比图已保存: %s", plot["name"], out_png)
        # 单张图的行×列 dB 矩阵可达 GB 级，逐组释放，避免两张图叠加导致 OOM
        del rows
        gc.collect()

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
