# -*- coding: utf-8 -*-
"""预处理阶段 Q1 · 步骤 1：数据探针流水线。

流程
----
1. 读取 ``config.json``；
2. 在数据根目录定位第一巻分卷（``*.z01``）；
3. 用 :mod:`zip_carver` 从该卷中提取少量 ``.mat`` 文件；
4. 用 :mod:`mat_probe` 逐个探测 dtype / 采样点数 / 时长 / 粗估 SNR；
5. 结果写入 ``output/Q1/probe_summary.xlsx``，日志写入 ``log/Q1``。

用法
----
    python script/Q1/run_pipeline.py
    python script/Q1/run_pipeline.py --skip-extract   # 复用已提取文件
"""

import argparse
import glob
import json
import logging
import os
import sys

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from zip_carver import extract_from_volume          # noqa: E402
from mat_probe import probe_many                    # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

COLUMNS = [
    ("file", "文件名", 42),
    ("ok", "是否成功", 10),
    ("dtype", "I 通道 dtype", 14),
    ("itemsize", "单样本字节数", 14),
    ("n_samples", "采样点数", 12),
    ("duration_s", "片段时长(s)", 14),
    ("complex_power_db", "复信号功率(dB)", 16),
    ("complex_peak", "复信号峰值", 14),
    ("snr_db", "粗估带内 SNR(dB)", 18),
    ("occupied_ratio", "占用频点比例", 14),
    ("noise_floor_db", "噪声底(dB)", 14),
    ("path", "文件路径", 60),
    ("error", "错误信息", 40),
]


# --------------------------------------------------------------------------- #
# 基础设施
# --------------------------------------------------------------------------- #
def load_config(path=CONFIG_PATH):
    """读取并补全配置中的相对路径。"""
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    for key in ("output_xlsx", "log_file"):
        if key in cfg and not os.path.isabs(cfg[key]):
            cfg[key] = os.path.join(PROJECT_ROOT, cfg[key])
    return cfg


def setup_logging(log_file, level="INFO"):
    """同时向控制台与文件输出日志（控制台统一为 UTF-8，避免中文乱码）。"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for h in list(root.handlers):
        root.removeHandler(h)
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(logging.Formatter(fmt))
    file_h = logging.FileHandler(log_file, encoding="utf-8")
    file_h.setFormatter(logging.Formatter(fmt))
    root.addHandler(stream)
    root.addHandler(file_h)
    return logging.getLogger("Q1")


def find_volume(data_root, pattern):
    """在数据根目录（含子目录）中定位第一巻分卷，找不到时给出明确提示。"""
    matches = sorted(glob.glob(os.path.join(data_root, pattern), recursive=True))
    if not matches:
        return None, matches
    return matches[0], matches


# --------------------------------------------------------------------------- #
# 流水线步骤
# --------------------------------------------------------------------------- #
def step_extract(cfg, log, skip=False):
    """提取少量 .mat 文件，返回文件路径列表。"""
    extract_dir = cfg["extract_dir"]
    if skip:
        files = sorted(glob.glob(os.path.join(extract_dir, "*.mat")))
        log.info("跳过提取，复用已有 %d 个文件", len(files))
        return files
    volume, all_volumes = find_volume(cfg["data_root"], cfg["volume_pattern"])
    if volume is None:
        log.error("未在 %s 找到匹配 %s 的分卷，请先下载第一卷",
                  cfg["data_root"], cfg["volume_pattern"])
        return []
    log.info("共发现 %d 个分卷，本次仅解析第一卷", len(all_volumes))
    records = extract_from_volume(
        volume_path=volume,
        out_dir=extract_dir,
        max_files=int(cfg.get("max_extract_files", 6)),
        extensions=tuple(cfg.get("extensions", [".mat"])),
        logger=log,
        verify_crc=bool(cfg.get("verify_crc", True)),
    )
    return [r["saved_path"] for r in records if r.get("crc_ok", True)]


def write_xlsx(rows, path):
    """把探针结果写入 xlsx（含“说明”工作表）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    wb = Workbook()
    ws_desc = wb.active
    ws_desc.title = "说明"
    ws_desc.append(["字段", "含义"])
    for name, desc, _w in COLUMNS:
        ws_desc.append([name, desc])
    ws_desc.append(["来源", "DroneRFb-DIR 分卷局部提取文件，脚本 script/Q1/run_pipeline.py"])
    for cell in ws_desc["A1":"B1"][0]:
        cell.font = Font(bold=True)
    ws_desc.column_dimensions["A"].width = 22
    ws_desc.column_dimensions["B"].width = 60

    ws = wb.create_sheet("文件统计")
    header = [desc for _n, desc, _w in COLUMNS]
    ws.append(header)
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="4472C4")
        cell.alignment = Alignment(horizontal="center")
    for row in rows:
        ws.append([row.get(name, "") for name, _d, _w in COLUMNS])
    for idx, (_name, _desc, width) in enumerate(COLUMNS, 1):
        ws.column_dimensions[get_column_letter(idx)].width = width
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    wb.save(path)
    return path


def summarize(rows, log):
    """打印整体结论性摘要（不替代人工判断）。"""
    ok_rows = [r for r in rows if r.get("ok")]
    log.info("=" * 68)
    log.info("成功 %d / %d 个文件", len(ok_rows), len(rows))
    if not ok_rows:
        return
    dtypes = sorted({r["dtype"] for r in ok_rows})
    itemsizes = sorted({r["itemsize"] for r in ok_rows})
    ns = [r["n_samples"] for r in ok_rows]
    log.info("dtype 集合: %s（单样本 %s 字节）", dtypes, itemsizes)
    log.info("采样点数: min=%d, max=%d", min(ns), max(ns))
    snrs = [r["snr_db"] for r in ok_rows if r.get("snr_db") == r.get("snr_db")]
    if snrs:
        log.info("粗估 SNR(dB): min=%.1f, max=%.1f, mean=%.1f",
                 min(snrs), max(snrs), sum(snrs) / len(snrs))
    log.info("=" * 68)


def main():
    parser = argparse.ArgumentParser(description="Q1 数据探针流水线")
    parser.add_argument("--skip-extract", action="store_true",
                        help="跳过提取，直接复用 extract_dir 中已有 .mat")
    parser.add_argument("--config", default=CONFIG_PATH)
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = setup_logging(cfg["log_file"], cfg.get("log_level", "INFO"))
    log.info("配置：%s", cfg)

    files = step_extract(cfg, log, skip=args.skip_extract)
    if not files:
        log.error("没有可用文件，流程结束")
        return 1

    rows = probe_many(files, cfg["fs_hz"], cfg.get("snr_estimate", {}), log)
    out = write_xlsx(rows, cfg["output_xlsx"])
    log.info("结果已写入：%s", out)
    summarize(rows, log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
