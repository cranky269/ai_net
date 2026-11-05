# -*- coding: utf-8 -*-
"""Q3 stage_1：把 .mat 的 I/Q 转成 STFT 灰度图并缓存为 npy（memmap）。

流程：Q1 划分表 → 每类限量选片段 → 每片段算一次 STFT → 按 ``frames_per_image``
切图（余数丢弃）→ 按精度量化 → 写入 ``X_<split>.npy`` / ``y_<split>.npy``。
缓存索引用 openpyxl 写成 ``result/Q3/cache_index.xlsx``（含“说明”工作表）。

用法::

    python python/Q3/stage_1/build_cache.py
"""

import argparse
import logging
import math
import sys
from pathlib import Path

import numpy as np
import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import q3_common as q3  # noqa: E402

CFG_PATH = Path(__file__).with_name("config.json")


def compute_geometry(fs, hop_ratio, frames_per_image):
    """算 hop、50 ms 总帧数、每片段图数、单片段需要读多少采样点。"""
    hop = max(1, int(round(q3.FRAME_LEN * hop_ratio)))
    n_clip = int(round(q3.CLIP_MS * 1e-3 * fs))
    total_frames = 1 + (n_clip - q3.FRAME_LEN) // hop
    n_img = max(1, total_frames // frames_per_image)
    n_samples = (n_img * frames_per_image - 1) * hop + q3.FRAME_LEN
    return {"hop": hop, "total_frames": total_frames, "n_img": n_img,
            "n_samples": n_samples, "n_clip": n_clip,
            "dropped_frames": total_frames - n_img * frames_per_image}


def plan_split(rows, split, cfg, geom, class_index):
    """按类别限量选片段，展开成 [(file, label, image_idx), ...]。"""
    groups = {}
    for row in rows:
        if str(row["split"]) != split:
            continue
        groups.setdefault(str(row["model"]), []).append(str(row["file"]))

    max_cls = int(cfg["max_per_class"])
    need_files = None if max_cls <= 0 else int(math.ceil(max_cls / geom["n_img"]))
    plan = []
    for cls in sorted(groups):
        files = sorted(groups[cls])
        if need_files:
            files = files[:need_files]
        items = [(f, k) for f in files for k in range(geom["n_img"])]
        if cfg["bg_resample"] and max_cls > 0 and len(items) < max_cls:
            items = [items[k % len(items)] for k in range(max_cls)]
        plan.extend((f, class_index[cls], k) for f, k in items)
    return plan


def allocate(cache_dir, split, n_total, frames, bins, dtype):
    """预分配 X 的 memmap 与 y 数组，返回 (X, y, 路径)。"""
    cache_dir = Path(cache_dir)
    x_path = cache_dir / ("X_%s.npy" % split)
    y_path = cache_dir / ("y_%s.npy" % split)
    X = np.lib.format.open_memmap(str(x_path), mode="w+", dtype=np.dtype(dtype),
                                  shape=(n_total, frames, bins))
    y = np.zeros(n_total, dtype=np.int16)
    return X, y, x_path, y_path


def fill_split(X, y, plan, cfg, geom, mat_dir):
    """逐片段算 STFT 并写入缓存；同一片段连续出现时复用上一次的 STFT。"""
    fs = float(cfg["fs_hz"])
    db_floor = float(cfg["db_floor"])
    frames = int(cfg["frames_per_image"])
    dtype = str(cfg["dtype"])

    last_file, last_db = None, None
    for idx, (fname, label, img_idx) in enumerate(plan):
        if fname != last_file:
            iq = q3.load_iq(Path(mat_dir) / fname)[:geom["n_samples"]]
            last_db, _, _ = q3.stft_db(iq, fs, q3.FRAME_LEN, geom["hop"],
                                       q3.FRAME_LEN, db_floor, cfg["window"])
            last_file = fname
        start = img_idx * frames
        X[idx] = q3.db_to_dtype(last_db[start:start + frames, :], db_floor, dtype)
        y[idx] = label
        if (idx + 1) % 20 == 0 or idx + 1 == len(plan):
            logging.info("  [%d/%d] %s img%d", idx + 1, len(plan), fname, img_idx)
    return X, y


def write_index_xlsx(path, records, cfg, classes, geom, shapes):
    """写索引表：'说明' 表放字段与参数含义，'明细' 表每张图一行。"""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "说明"
    ws.append(["字段/参数", "含义"])
    for key, desc in [
        ("index", "该图在 X_<split>.npy 第一维的位置"),
        ("split", "train / test，来自 Q1 划分表"),
        ("file", "原始 .mat 文件名"),
        ("model", "型号（A/C/D/E/F/G/BG），即分类标签"),
        ("label_idx", "模型的类别下标，顺序见 class_order"),
        ("image_idx", "该片段切出的第几张图（从 0 开始）"),
        ("frames_per_image", cfg["frames_per_image"]),
        ("hop", geom["hop"]),
        ("frame_len", q3.FRAME_LEN),
        ("50ms总帧数", geom["total_frames"]),
        ("每片段图数", geom["n_img"]),
        ("丢弃余数帧", geom["dropped_frames"]),
        ("dtype", cfg["dtype"]),
        ("db_floor", cfg["db_floor"]),
        ("max_per_class", cfg["max_per_class"]),
        ("bg_resample", cfg["bg_resample"]),
        ("class_order", ", ".join(classes)),
        ("X形状", " / ".join("%s: %s" % (k, v) for k, v in shapes.items())),
    ]:
        ws.append([key, desc])

    wd = wb.create_sheet("明细")
    wd.append(["index", "split", "file", "model", "label_idx", "image_idx"])
    for rec in records:
        wd.append([rec["index"], rec["split"], rec["file"], rec["model"],
                   rec["label_idx"], rec["image_idx"]])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    wb.close()


def parse_args():
    """命令行只提供试跑用的类别数量覆盖，正式参数都在 config.json。"""
    parser = argparse.ArgumentParser(description="构建 Q3 STFT npy 缓存")
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="临时覆盖 config.json 的 max_per_class，便于小样本试跑")
    return parser.parse_args()


def main():
    """串流程：读配置与划分 → 逐子集构建缓存 → 写索引表。"""
    args = parse_args()
    cfg = q3.load_json(CFG_PATH)
    if args.max_per_class:
        cfg["max_per_class"] = int(args.max_per_class)
    q3.setup_logging(q3.PROJECT_ROOT / cfg["log_file"])

    geom = compute_geometry(float(cfg["fs_hz"]), float(cfg["hop_ratio"]),
                            int(cfg["frames_per_image"]))
    logging.info("帧长 %d | hop %d | 50 ms 共 %d 帧 | 每张 %d 帧 → 每片段 %d 张（丢 %d 帧）",
                 q3.FRAME_LEN, geom["hop"], geom["total_frames"],
                 cfg["frames_per_image"], geom["n_img"], geom["dropped_frames"])

    rows = q3.read_split_rows(q3.SPLIT_XLSX)
    classes = q3.class_list(rows)
    class_index = {c: i for i, c in enumerate(classes)}
    logging.info("类别顺序: %s", ", ".join(classes))

    cache_dir = q3.PROJECT_ROOT / cfg["cache_dir"]
    mat_dir = q3.PROJECT_ROOT / cfg["mat_dir"]
    q3.ensure_dirs(cache_dir, q3.RESULT_DIR)

    records, shapes = [], {}
    for split in cfg["splits"]:
        plan = plan_split(rows, split, cfg, geom, class_index)
        if not plan:
            logging.warning("%s 子集为空，跳过", split)
            continue
        X, y, x_path, y_path = allocate(cache_dir, split, len(plan),
                                        int(cfg["frames_per_image"]), q3.FRAME_LEN,
                                        cfg["dtype"])
        logging.info("%s: %d 张图 → %s", split, len(plan), x_path.name)
        X, y = fill_split(X, y, plan, cfg, geom, mat_dir)
        X.flush()
        del X
        np.save(y_path, y)
        shapes[split] = (len(plan), int(cfg["frames_per_image"]), q3.FRAME_LEN)
        for idx, (fname, label, img_idx) in enumerate(plan):
            records.append({"index": idx, "split": split, "file": fname,
                            "model": classes[label], "label_idx": int(label),
                            "image_idx": img_idx})

    write_index_xlsx(q3.PROJECT_ROOT / cfg["index_xlsx"], records, cfg, classes, geom, shapes)
    total_gib = sum(int(np.prod(s)) for s in shapes.values()) * np.dtype(cfg["dtype"]).itemsize / 1024 ** 3
    logging.info("完成：%d 张图，缓存约 %.2f GiB，索引 %s",
                 len(records), total_gib, cfg["index_xlsx"])


if __name__ == "__main__":
    main()
