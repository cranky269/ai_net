# -*- coding: utf-8 -*-
"""Q2 stage_2：IQ 信号加高斯白噪声（内存迭代，不落盘）。

目标信噪比反推噪声方差（复数复高斯噪声）::

    P_s   = mean(|x|^2)                      # 每条信号单独计算
    sigma = sqrt(P_s / 10^(SNR_dB/10))       # 噪声总功率 = sigma^2
    n     = sigma/sqrt(2) * (n_I + j*n_Q),  n_I, n_Q ~ N(0, 1)
    y     = x + n

可复现约定：
- 每个 ``(split, snr)`` 组合由 ``SeedSequence([base_seed, split_id, snr_idx])``
  派生一条独立随机流，任一组可脱离其他组单独复现；
- 每条信号按“映射表行序 + 文件名排序”的固定顺序依次消费该流；
- 每条信号按自身功率单独计算 sigma，保证都落在目标信噪比档位；
- 背景类（model == bg_model）默认跳过；``include_bg=True`` 时原样返回、不加噪。

用法::

    from awgn import load_json, load_split_table, iter_split

    cfg = load_json("config.json")
    table = load_split_table(cfg["split_table"], cfg["split_sheet"])
    for path, iq_noisy, meta in iter_split(table, cfg, "train", -5.0):
        ...  # 直接消费，不落盘
"""

import json
import logging
from pathlib import Path

import h5py
import numpy as np

# ==================== 可调参数 ====================
# 数据集名称 → 种子派生用的整数标签（只影响随机流，不参与计算）
SPLIT_ID = {"train": 0, "test": 1}
# =================================================

logger = logging.getLogger(__name__)


def load_json(path):
    """读取 UTF-8 JSON 文件，返回 dict。"""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_split_table(path, sheet="split"):
    """读取 Q1 数据划分表，返回按文件顺序排列的记录列表（每项为 dict）。"""
    import openpyxl  # 仅此处需要，避免 awgn 核心依赖表格库

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        rows = list(wb[sheet].values)
    finally:
        wb.close()
    if not rows:
        raise ValueError("划分表为空: %s" % path)
    header = [str(c).strip() if c is not None else "" for c in rows[0]]
    records = []
    for row in rows[1:]:
        if row is None or row[0] is None:
            continue
        records.append({header[i]: row[i] for i in range(len(header))})
    return records


def split_to_id(split):
    """把 split 名称转成整数标签（供 SeedSequence 使用）。"""
    if split not in SPLIT_ID:
        raise ValueError("未知 split: %r，可选 %s" % (split, list(SPLIT_ID)))
    return SPLIT_ID[split]


def snr_to_index(snr_db, snr_list):
    """把目标信噪比匹配到档位下标；不在档位表里则报错。"""
    values = [float(v) for v in snr_list]
    idx = int(np.argmin([abs(v - float(snr_db)) for v in values]))
    if abs(values[idx] - float(snr_db)) > 1e-6:
        raise ValueError("SNR %.3f dB 不在档位表 %s 中" % (float(snr_db), values))
    return idx


def make_rng(base_seed, split_id, snr_idx, extra=None):
    """派生独立随机流：SeedSequence([base_seed, split_id, snr_idx, *extra])。"""
    key = [int(base_seed), int(split_id), int(snr_idx)]
    if extra:
        key.extend(int(v) for v in extra)
    return np.random.default_rng(np.random.SeedSequence(key))


def signal_power(iq):
    """信号平均功率 P_s = mean(|x|^2)，以 float64 累加避免精度损失。"""
    return float(np.mean(np.abs(iq) ** 2, dtype=np.float64))


def sigma_for_snr(power, snr_db):
    """由信号功率与目标 SNR 反推复噪声总功率 sqrt，即 sigma。"""
    return float(np.sqrt(float(power) / (10.0 ** (float(snr_db) / 10.0))))


def generate_noise(n, sigma, rng):
    """生成复高斯白噪声：实虚部各 N(0, sigma^2/2)，总功率为 sigma^2。"""
    half = float(sigma) / np.sqrt(2.0)
    noise_i = rng.normal(0.0, half, n).astype(np.float32)
    noise_q = rng.normal(0.0, half, n).astype(np.float32)
    return (noise_i + 1j * noise_q).astype(np.complex64)


def make_noise(iq, snr_db, rng, cfg=None, stats=None):
    """统一加噪入口：返回 ``(noise, meta)``（只有原有 power 方案）。

    ``cfg`` / ``stats`` 仅为兼容 Q2.5 的调用签名保留，当前实现不用它们：
    早期试过的 Higuchi 分形分段方案对宽带调制数据无效，已整体移除。
    """
    power = signal_power(iq)
    sigma = sigma_for_snr(power, snr_db)
    noise = generate_noise(iq.size, sigma, rng)
    noise_power = float(np.mean(np.abs(noise) ** 2, dtype=np.float64))
    measured = 10.0 * np.log10(power / noise_power) if noise_power > 0 else float("inf")
    meta = {
        "snr_db_target": float(snr_db),
        "snr_db_measured": float(measured),
        "sigma": sigma,
        "signal_power": power,
        "noise_power": noise_power,
    }
    return noise, meta


def fractal_stats(iq, cfg=None):
    """兼容占位：Higuchi 分段方案已移除，恒返回 None（Q2.5 仍在调用）。"""
    return None


def add_awgn(iq, snr_db, rng, cfg=None):
    """给单条 IQ 加噪，返回 (加噪信号, meta)；meta 含实测 SNR 与 sigma。"""
    noise, meta = make_noise(iq, snr_db, rng)
    noisy = (iq + noise).astype(np.complex64)
    return noisy, meta


def load_iq(path):
    """读取 HDF5 格式 .mat 的 I/Q 通道，返回一维复数 IQ（complex64）。"""
    with h5py.File(path, "r") as fh:
        keys = list(fh.keys())
        if not {"I", "Q"}.issubset(keys):
            raise ValueError("%s 缺少 I/Q 数据集，实际键: %s" % (Path(path).name, keys))
        i_data = np.asarray(fh["I"][()], dtype=np.float32).reshape(-1)
        q_data = np.asarray(fh["Q"][()], dtype=np.float32).reshape(-1)
    if i_data.size != q_data.size:
        raise ValueError("%s 的 I/Q 长度不一致: %d vs %d"
                         % (Path(path).name, i_data.size, q_data.size))
    return (i_data + 1j * q_data).astype(np.complex64)


def select_rows(table, split, include_bg=False, bg_model="BG"):
    """筛出指定 split 的记录，按文件名排序保证处理顺序固定。"""
    rows = [r for r in table if str(r.get("split")) == str(split)]
    if not include_bg:
        rows = [r for r in rows if str(r.get("model")) != str(bg_model)]
    rows.sort(key=lambda r: str(r.get("file")))
    return rows


def iter_split(table, cfg, split, snr_db, data_dir=None, include_bg=None):
    """按固定顺序逐条产出 ``(path, iq_noisy, meta)``，不落盘。

    同一 ``(split, snr)`` 组合共用一条随机流，条目顺序固定，因而完全可复现。
    背景类在 ``include_bg=True`` 时原样返回，且不消耗随机数。
    """
    snr_idx = snr_to_index(snr_db, cfg["snr_db_list"])
    split_id = split_to_id(split)
    bg_model = str(cfg.get("bg_model", "BG"))
    if include_bg is None:
        include_bg = bool(cfg.get("include_bg", False))

    rows = select_rows(table, split, include_bg=include_bg, bg_model=bg_model)
    root = Path(data_dir) if data_dir is not None else Path(cfg["data_dir"])
    rng = make_rng(cfg["base_seed"], split_id, snr_idx)

    logger.info("iter_split | split=%s snr=%.1f dB | 共 %d 条 | 种子 [%d, %d, %d]",
                split, float(snr_db), len(rows),
                int(cfg["base_seed"]), split_id, snr_idx)

    for rec in rows:
        path = root / str(rec["file"])
        iq = load_iq(path)
        if str(rec.get("model")) == bg_model:
            yield path, iq, {"file": path.name, "model": bg_model,
                             "bg": True, "snr_db_target": None,
                             "snr_db_measured": None, "sigma": None}
            continue
        noisy, meta = add_awgn(iq, float(snr_db), rng)
        meta.update({"file": path.name, "model": rec.get("model"), "bg": False})
        yield path, noisy, meta
