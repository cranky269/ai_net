# -*- coding: utf-8 -*-
"""Q3 通用设施：路径、配置读写、日志、Q1 划分表读取、STFT（复用 Q2）。

本模块只放“多处要用”的东西，不含缓存构建 / 训练等业务逻辑。
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
Q2_STFT_DIR = PROJECT_ROOT / "python" / "Q2" / "stage_1"

# ---- 数据集不可调参数（按约定写死，不进 config）----
FRAME_LEN = 2048          # 帧长（采样点数）
CLIP_MS = 50.0            # 统一截取信号开头的时长（毫秒）

# ---- Q3 输入 / 输出路径 ----
DATA_ROOT = PROJECT_ROOT / "data"                    # 指向 D:/智能通信网络数据集 的 junction
CACHE_DIR = DATA_ROOT / "_q3_cache"                  # npy 缓存（放大盘）
SPLIT_XLSX = PROJECT_ROOT / "output" / "Q1" / "splits" / "dataset_split.xlsx"
RESULT_DIR = PROJECT_ROOT / "result" / "Q3"
IMAGE_DIR = PROJECT_ROOT / "image" / "Q3"
LOG_DIR = PROJECT_ROOT / "log" / "Q3"
PIC_CFG_PATH = PROJECT_ROOT / "config_pic.json"


def _load_q2():
    """把 Q2 stage_1 加入 sys.path 后导入，以复用其中的 load_iq / stft_db。"""
    if str(Q2_STFT_DIR) not in sys.path:
        sys.path.insert(0, str(Q2_STFT_DIR))
    import stft_explore

    return stft_explore


Q2 = _load_q2()


def load_iq(path):
    """读 .mat（HDF5）中的 I/Q，返回复基带信号（复用 Q2 实现）。"""
    return Q2.load_iq(Path(path))


def stft_db(x, fs, frame_len, hop, n_fft, db_floor, window="hann"):
    """复数 STFT → 相对峰值 dB 矩阵（复用 Q2 实现）。"""
    return Q2.stft_db(x, fs, frame_len, hop, n_fft, db_floor, window)


def load_json(path):
    """读取 UTF-8 JSON 文件。"""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def setup_logging(log_path, level=logging.INFO):
    """日志同时写文件（UTF-8）与控制台；重复调用不会叠加 handler。"""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)],
        force=True,
    )


def ensure_dirs(*paths):
    """批量创建目录（已存在则跳过）。"""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def read_split_rows(xlsx_path):
    """读 Q1 划分表，返回 dict 列表（file/individual/model/scene/label/split）。"""
    import openpyxl

    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb["split"]
    rows_iter = ws.iter_rows(values_only=True)
    header = [str(c).strip() for c in next(rows_iter)]
    rows = [dict(zip(header, r)) for r in rows_iter if r and r[0]]
    wb.close()
    return rows


def class_list(rows):
    """取划分表中出现过的型号名并排序，作为类别顺序（A、BG、C…G）。"""
    return sorted({str(r["model"]) for r in rows})


def db_to_dtype(db, db_floor, dtype):
    """按指定精度量化 dB 图：uint8 → 0~255，float16/float32 → 原始 dB 值。"""
    arr = np.asarray(db, dtype=np.float32)
    if dtype == "uint8":
        scaled = (arr - db_floor) / (-db_floor) * 255.0
        return np.clip(scaled, 0.0, 255.0).astype(np.uint8)
    if dtype in ("float16", "float32"):
        return arr.astype(dtype)
    raise ValueError("不支持的精度: %s（可用 uint8 / float16 / float32）" % dtype)
