# -*- coding: utf-8 -*-
"""Q2.5 stage_2：信号探针 / 噪声探针 / 交互效应探针（STFT 幅度域）。

设 ``x`` 为干净信号、``n`` 为与加噪同源的一段噪声、``M(.)`` 为预处理算子。
由于当前方法作用于 STFT 幅度谱，记::

    X_x  = |STFT(x)|      X_n = |STFT(n)|      X_xn = |STFT(x+n)|

四项指标（Frobenius 范数，除以 ``||X_x||_F^2``）::

    total          = ||M(X_xn) - X_x||_F^2 / ||X_x||_F^2
    signal_loss    = ||M(X_x)  - X_x||_F^2 / ||X_x||_F^2
    noise_residual = ||M(X_n)       ||_F^2 / ||X_x||_F^2
    interaction    = ||M(X_xn) - M(X_x) - M(X_n)||_F^2 / ||X_x||_F^2

恒等关系 ``M(X_xn) - X_x = (M(X_x)-X_x) + M(X_n) + I``，三项共同解释 total。

探针不绑定具体算法：方法全部来自 ``common/processors.py`` 的插件列表，
中值滤波只是当前注册的一种。

用法::

    python python/Q2.5/stage_2/probes.py
    python python/Q2.5/stage_2/probes.py --out-xlsx result/Q2.5/my_probe.xlsx
    python python/Q2.5/stage_2/probes.py --workers 8      # 全量时并行
"""

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

COMMON_DIR = Path(__file__).resolve().parents[1] / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

import processors as q25_proc  # noqa: E402
import q25_common as q25  # noqa: E402

# ==================== 可调参数 ====================
CFG_PATH = Path(__file__).with_name("config.json")
LOG_PATH = q25.LOG_DIR / "stage2_probes.log"
# =================================================


def parse_args():
    """命令行参数：输出路径与并行进程数可覆盖 config。"""
    parser = argparse.ArgumentParser(description="信号/噪声/交互效应探针")
    parser.add_argument("--out-xlsx", default=None, help="结果 xlsx 输出路径")
    parser.add_argument("--workers", type=int, default=None, help="并行进程数")
    return parser.parse_args()


def _identity(mag):
    """恒等算子：用于生成“无预处理”基线行（method=none）。"""
    return mag


def _sq(a, minus=(), chunk=4096):
    """``sum((a - minus 各项之和)^2)``，分块累加。

    整幅相减/平方各要一块与 ``|X|`` 同尺寸的临时数组（6.25 万帧 × 256 频点约
    61 MB），内存紧张的机器上会直接 MemoryError；分块后每块仅数 MB，且在
    float64 下累加，数值与原一次性求和无实质差别。
    """
    total = 0.0
    for start in range(0, a.shape[0], chunk):
        stop = min(start + chunk, a.shape[0])
        blk = np.asarray(a[start:stop], dtype=np.float64)
        if minus:
            blk = blk - np.asarray(minus[0][start:stop], dtype=np.float64)
            for other in minus[1:]:
                blk -= other[start:stop]
        total += float(np.einsum("ij,ij->", blk, blk))
    return total


def compute_metrics(spec_x, spec_n, spec_xn, fn, f_x=None):
    """在 STFT 幅度域计算四项指标（Frobenius 范数，除以 ||X_x||_F^2）。

    参数
    ----
    spec_x, spec_n, spec_xn : ndarray
        分别为干净信号、纯噪声、加噪信号的 STFT 幅度谱 ``|X|``。
    fn : callable
        作用在幅度谱上的预处理算子 ``fn(mag) -> mag``。
    f_x : ndarray, 可选
        ``fn(spec_x)`` 的缓存，每个方法只需算一次。
    """
    denom = _sq(spec_x)
    if denom <= 0:
        raise ValueError("干净信号 STFT 能量为 0，无法归一化")
    if f_x is None:
        f_x = fn(spec_x)
    f_n = fn(spec_n)
    f_xn = fn(spec_xn)

    return {
        "total": _sq(f_xn, (spec_x,)) / denom,
        "signal_loss": _sq(f_x, (spec_x,)) / denom,
        "noise_residual": _sq(f_n) / denom,
        "interaction": _sq(f_xn, (f_x, f_n)) / denom,
        "norm_x": denom,
    }


def probe_one_file(task):
    """处理单条样本的全部（方法 × SNR），返回明细行列表。

    流程：干净信号 STFT 幅度 →（每个 SNR）噪声/加噪信号的 STFT 幅度
    → 各方法在幅度谱上滤波 → 计算四项指标。

    作为模块级函数存在，便于 ``ProcessPoolExecutor`` 序列化调用。
    """
    if str(COMMON_DIR) not in sys.path:
        sys.path.insert(0, str(COMMON_DIR))
    import processors as proc_w
    import q25_common as q25w

    awgn = q25w.import_q2_awgn()
    path = Path(task["data_dir"]) / task["file"]
    x = awgn.load_iq(path)
    power = awgn.signal_power(x)
    params = tuple(task["stft_params"])

    spec_x = q25w.stft_mag(x, *params)[0]
    # 首行为恒等算子基线（无预处理），其余为真实预处理方法
    baseline = {"name": "none", "param": 0, "label": "no_filter",
                "fn": _identity, "cache_f_x": True}
    methods = [baseline] + proc_w.make_processors(task["methods_cfg"])
    # f_x 可跨 SNR 缓存以省时间，但整幅一份就 61 MB（6.25 万帧 × 256 频点），
    # 缓存整轮会让单进程峰值逼近 1 GB；内存紧张的机器把 config 的
    # ``cache_f_x`` 关掉（默认），代价只在中值滤波要按 SNR 重算。
    f_x_cache = {}
    if task.get("cache_f_x", False):
        f_x_cache = {m["label"]: m["fn"](spec_x)
                     for m in methods if m.get("cache_f_x")}

    rows = []
    for snr_idx, snr_db in enumerate(task["snr_list"]):
        rng = awgn.make_rng(int(task["base_seed"]), int(task["split_id"]),
                            snr_idx,
                            extra=[q25w.SAMPLE_SALT, int(task["sample_idx"])])
        sigma = awgn.sigma_for_snr(power, float(snr_db))
        noise = awgn.generate_noise(x.size, sigma, rng)
        spec_n = q25w.stft_mag(noise, *params)[0]
        spec_xn = q25w.stft_mag((x + noise).astype(np.complex64), *params)[0]
        del noise
        for method in methods:
            metrics = compute_metrics(spec_x, spec_n, spec_xn, method["fn"],
                                      f_x=f_x_cache.get(method["label"]))
            rows.append({
                "file": task["file"],
                "method": method["name"],
                "param": method["param"],
                "snr_db": float(snr_db),
                **metrics,
            })
        del spec_n, spec_xn
    return rows


def resolve_files(cfg_s2, cfg_q2, awgn):
    """确定探针样本列表：显式文件名列表，或 "all" 时按 split/scene 全量。"""
    spec = cfg_s2.get("probe_files", [])
    if isinstance(spec, str) and spec.strip().lower() == "all":
        table_path = q25.PROJECT_ROOT / cfg_q2["split_table"]
        table = awgn.load_split_table(str(table_path),
                                      cfg_q2.get("split_sheet", "split"))
        split = str(cfg_s2.get("probe_split", "train"))
        scene = cfg_s2.get("probe_scene")
        bg_model = str(cfg_q2.get("bg_model", "BG"))
        rows = [r for r in table
                if str(r.get("split")) == split
                and (scene is None or str(r.get("scene")) == str(scene))
                and str(r.get("model")) != bg_model]
        return sorted(str(r["file"]) for r in rows)
    return sorted(str(f) for f in spec)


def summarize(detail):
    """按 (method, param, snr_db) 聚合，输出各指标均值。

    ``param`` 原样保留（中值是整数窗口，维纳是浮点 alpha），不能取整，
    否则 ``alpha=1.0`` 与 ``1.5`` 会被合并成同一组。
    """
    groups = {}
    for row in detail:
        key = (str(row["method"]), row["param"], float(row["snr_db"]))
        groups.setdefault(key, []).append(row)
    summary = []
    for (method, param, snr_db), items in sorted(groups.items()):
        summary.append({
            "method": method,
            "param": param,
            "snr_db": snr_db,
            "n": len(items),
            "total_mean": float(np.mean([r["total"] for r in items])),
            "signal_loss_mean": float(np.mean([r["signal_loss"] for r in items])),
            "noise_residual_mean": float(np.mean([r["noise_residual"] for r in items])),
            "interaction_mean": float(np.mean([r["interaction"] for r in items])),
        })
    return summary


def _write_sheet(ws, header, rows):
    """向工作表写表头与数据行。"""
    ws.append(header)
    for row in rows:
        ws.append([row.get(h, "") for h in header])


def write_xlsx(detail, summary, out_path, meta_lines):
    """写出 xlsx：说明 / 明细 / 汇总 三个 sheet。"""
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "说明"
    ws.append(["项目", "内容"])
    for key, value in meta_lines:
        ws.append([key, value])

    detail_header = ["file", "method", "param", "snr_db", "total",
                     "signal_loss", "noise_residual", "interaction", "norm_x"]
    _write_sheet(wb.create_sheet("明细"), detail_header, detail)

    summary_header = ["method", "param", "snr_db", "n", "total_mean",
                      "signal_loss_mean", "noise_residual_mean", "interaction_mean"]
    _write_sheet(wb.create_sheet("汇总"), summary_header, summary)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        wb.save(str(out_path))
    except PermissionError as exc:
        raise SystemExit("无法写入 %s：文件可能正被 Excel 打开，请关闭后重试。\n%s"
                         % (out_path, exc))
    return out_path


def build_meta(cfg_s1, cfg_s2, files, workers):
    """整理写入“说明”sheet 的元信息。"""
    return [
        ("文件用途", "Q2.5 探针指标：信号损失 / 噪声残留 / 交互效应 / 总失真"),
        ("处理域", "STFT 幅度域：M 作用在 |X| 矩阵上（中值：沿频率方向取中值；维纳：沿时间估计噪声功率后逐点乘增益），相位不受影响"),
        ("参数含义", "method=median_freq 时 param 为中值窗口（频点数）；method=wiener 时 param 为过减系数 alpha（1.0 即标准维纳，Excel 可能显示为 1）"),
        ("基线行", "method=none、param=0 表示无预处理（恒等算子）；其 total 即加噪谱与干净谱的偏差，signal_loss 恒为 0"),
        ("记号", "X_x=|STFT(x)|，X_n=|STFT(n)|，X_xn=|STFT(x+n)|"),
        ("归一化", "全部指标除以 ||X_x||_F^2（干净信号 STFT 幅度能量），故三项可加"),
        ("total", "||M(X_xn)-X_x||_F^2 / ||X_x||_F^2（处理后与干净谱的总偏差）"),
        ("signal_loss", "||M(X_x)-X_x||_F^2 / ||X_x||_F^2（算子对干净谱的畸变）"),
        ("noise_residual", "||M(X_n)||_F^2 / ||X_x||_F^2（算子未抑制掉的噪声能量）"),
        ("interaction", "||M(X_xn)-M(X_x)-M(X_n)||_F^2 / ||X_x||_F^2（非线性交互效应）"),
        ("恒等关系", "M(X_xn)-X_x 由 (M(X_x)-X_x) + M(X_n) + I 共同解释"),
        ("方法配置", str(cfg_s1["methods"])),
        ("SNR 档位", str(cfg_s1["snr_db_list"])),
        ("随机种子根", str(cfg_s1["base_seed"])),
        ("种子派生", "SeedSequence([base_seed, split_id, snr_idx, SAMPLE_SALT, sample_idx])"),
        ("样本数", str(len(files))),
        ("样本列表", ", ".join(files[:20]) + (" ..." if len(files) > 20 else "")),
        ("并行进程数", str(workers)),
    ]


def main():
    """串起流程：读配置 → 生成任务 → 单/多进程计算 → 汇总并写 xlsx。"""
    q25.setup_logging(LOG_PATH)
    cfg_s1 = q25.load_json(q25.Q25_DIR / "stage_1" / "config.json")
    cfg_s2 = q25.load_json(CFG_PATH)
    cfg_q2s2 = q25.load_json(q25.Q2_STAGE2_CONFIG)
    cfg_q2 = q25.load_json(q25.Q2_STAGE1_CONFIG)
    args = parse_args()

    awgn = q25.import_q2_awgn()
    files = resolve_files(cfg_s2, cfg_q2s2, awgn)
    if not files:
        raise SystemExit("探针样本为空，请检查 probe_files / probe_split / probe_scene")

    workers = int(args.workers if args.workers is not None
                  else cfg_s2.get("workers", 1))
    out_xlsx = (Path(args.out_xlsx) if args.out_xlsx
                else q25.PROJECT_ROOT / str(cfg_s2["out_xlsx"]))
    data_dir = str(q25.PROJECT_ROOT / str(cfg_q2s2["data_dir"]))
    split_id = awgn.split_to_id(str(cfg_s2.get("probe_split", "train")))
    snr_list = [float(v) for v in cfg_s1["snr_db_list"]]

    logging.info("探针样本 %d 条 | SNR %s | 方法 %s | workers=%d",
                 len(files), snr_list,
                 [p["label"] for p in q25_proc.make_processors(cfg_s1["methods"])],
                 workers)

    params = q25.stft_params(cfg_q2)
    tasks = [{
        "file": name, "data_dir": data_dir, "sample_idx": idx,
        "split_id": split_id, "base_seed": int(cfg_s1["base_seed"]),
        "snr_list": snr_list, "methods_cfg": cfg_s1["methods"],
        "stft_params": list(params),
        "cache_f_x": bool(cfg_s2.get("cache_f_x", False)),
    } for idx, name in enumerate(files)]

    detail = []
    if workers > 1 and len(tasks) > 1:
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for rows in executor.map(probe_one_file, tasks):
                detail.extend(rows)
                done += 1
                logging.info("完成 %d/%d 条", done, len(files))
    else:
        for task in tasks:
            detail.extend(probe_one_file(task))
            logging.info("完成 %s", task["file"])

    summary = summarize(detail)
    meta = build_meta(cfg_s1, cfg_s2, files, workers)
    write_xlsx(detail, summary, out_xlsx, meta)
    logging.info("已写出: %s（明细 %d 行，汇总 %d 行）",
                 out_xlsx, len(detail), len(summary))


if __name__ == "__main__":
    main()
