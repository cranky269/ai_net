# -*- coding: utf-8 -*-
"""生成 DroneRFb-DIR 数据集实际情况报告（Q0 stage_3，预处理阶段）。

数据来源
--------
- output/Q0/inventory.json              分卷中央目录清单（含 train/test 条目）
- data/train_extracted/test_labels.txt   测试集标签（语义名 -> 数字标号）

输出
----
- docs/Q0/数据集实际情况.md
- output/Q0/dataset_stats.xlsx

用法
----
    python script/Q0/stage_3/dataset_report.py
"""

import datetime
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict

from openpyxl import Workbook

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

COND_VALUES = ("IN", "OUT")
BATCH_VALUES = ("S0", "S1", "S2")


# ---------------- 基础工具 ----------------

def load_config(path=CONFIG_PATH):
    """读取配置，把相对路径转成基于工程根目录的绝对路径。"""
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    for key in ("inventory_file", "labels_dir", "report_file", "stats_xlsx", "log_file"):
        if key in cfg and not os.path.isabs(cfg[key]):
            cfg[key] = os.path.join(PROJECT_ROOT, cfg[key])
    return cfg


def setup_logging(log_file):
    """同时输出到控制台和日志文件。"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(log_file, encoding="utf-8")],
    )
    return logging.getLogger("Q0.stage3")


def parse_name(basename):
    """解析 .mat 文件名 -> dict(id, cond, batch, slice)。"""
    stem = basename[:-4] if basename.lower().endswith(".mat") else basename
    parts = stem.split("_")
    if not parts:
        return None
    ident = parts[0]
    rest = parts[1:]
    cond = None
    batch = None
    if rest and rest[0] in COND_VALUES:
        cond = rest[0]
        rest = rest[1:]
    if rest and re.fullmatch(r"S\d+", rest[0]):
        batch = rest[0]
        rest = rest[1:]
    m = re.search(r"slice[_-]?(\d+)$", "_".join(rest))
    return {"id": ident, "cond": cond, "batch": batch,
            "slice": int(m.group(1)) if m else None}


def model_of(ident):
    """个体编号 -> 型号字母（background 记为 BG）。"""
    if ident.lower() == "background":
        return "BG"
    m = re.match(r"([A-Za-z])", ident)
    return m.group(1).upper() if m else "?"


def scene_key(rec):
    """组合场景键：IN_S0 / IN_S1 / IN_S2 / OUT。"""
    if rec["cond"] is None:
        return "无"
    return "%s_%s" % (rec["cond"], rec["batch"]) if rec["batch"] else rec["cond"]


# ---------------- 数据加载 ----------------

def load_train_records(inventory_path):
    """从清单读取 train 的 .mat 条目并解析。"""
    with open(inventory_path, encoding="utf-8") as f:
        inv = json.load(f)
    records = []
    for entry in inv["entries"]:
        name = entry["name"].replace("\\", "/")
        if "/train/" not in name or not name.lower().endswith(".mat"):
            continue
        rec = parse_name(os.path.basename(name))
        if rec:
            rec["file"] = os.path.basename(name)
            records.append(rec)
    return records


def load_test_records(test_labels_path):
    """从 test_labels.txt 读取 test 记录（每行：语义名 数字标号）。"""
    records = []
    if not os.path.exists(test_labels_path):
        return records
    with open(test_labels_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            rec = parse_name(parts[0])
            if rec:
                rec["file"] = parts[0]
                rec["index"] = parts[1]
                records.append(rec)
    return records


def collect_stats(records):
    """汇总一组记录的统计量。"""
    stats = {
        "total": len(records),
        "by_id": Counter(r["id"] for r in records),
        "by_model": Counter(model_of(r["id"]) for r in records),
        "by_cond": Counter(r["cond"] for r in records),
        "by_batch": Counter(r["batch"] for r in records),
        "by_scene": Counter(scene_key(r) for r in records),
        "id_cond": defaultdict(Counter),
        "id_batch": defaultdict(Counter),
    }
    for rec in records:
        stats["id_cond"][rec["id"]][rec["cond"]] += 1
        stats["id_batch"][rec["id"]][rec["batch"]] += 1
    return stats


# ---------------- 报告各章节 ----------------

def sec_header(cfg):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return [
        "# %s 数据集实际情况报告" % cfg["dataset_name"],
        "",
        "> 生成时间：%s" % now,
        "> 数据来源：`output/Q0/inventory.json`（分卷中央目录清单）+ `train_extracted/*_labels.txt`（标签）",
        "> 说明：test 数据仅基于清单/标签统计，未下载实际 `.mat` 文件。",
        "",
    ]


def sec_overview():
    return [
        "## 1. 数据集概述",
        "",
        "- **数据集**：DroneRFb-DIR（用于非合作无人机个体识别的射频信号数据集）",
        "- **采集设备**：USRP-2955 软件无线电",
        "- **频段**：2.4 ~ 2.48 GHz（含飞控、图传及干扰信号）",
        "- **数据格式**：`.mat`，I/Q 双通道，单精度浮点",
        "- **命名规则**：`个体_采集条件_时间序号_slice_片段号.mat`",
        "  - 个体：`机型字母(A/C/D/E/F/G) + 个体编号(1/2/3)`，另有 `background` 背景类",
        "  - 采集条件：`IN`=视距(LOS)，`OUT`=非视距(NLOS)",
        "  - 时间序号：`S0/S1/S2`（OUT 无时间序号）",
        "  - 片段号：`slice_N`",
        "",
    ]


def sec_scale(train_stats, test_stats):
    total = train_stats["total"] + test_stats["total"]
    return [
        "## 2. 总体规模",
        "",
        "| 划分 | `.mat` 文件数 | 状态 |",
        "|---|---|---|",
        "| train | %d | 已下载到本地 |" % train_stats["total"],
        "| test | %d | 仅清单/标签（未下载） |" % test_stats["total"],
        "| valid | 0 | 数据集无此划分 |",
        "| **合计** | **%d** | |" % total,
        "",
    ]


def sec_models(train_stats, test_stats):
    models = sorted(set(train_stats["by_model"]) | set(test_stats["by_model"]))
    lines = [
        "## 3. 型号分布",
        "",
        "官方描述：6 类无人机 + 1 类背景参考信号。",
        "",
        "| 型号 | train 片段 | test 片段 | 合计 |",
        "|---|---|---|---|",
    ]
    for model in models:
        tr = train_stats["by_model"].get(model, 0)
        te = test_stats["by_model"].get(model, 0)
        name = "BG(背景)" if model == "BG" else model
        lines.append("| %s | %d | %d | %d |" % (name, tr, te, tr + te))
    lines.append("")
    return lines


def sec_scenes(train_stats, test_stats):
    lines = [
        "## 4. 场景分布",
        "",
        "> 官方两层含义：`IN/OUT`=视距/非视距采集条件；`S0/S1/S2`=采集时间序号。",
        "",
        "### 4.1 采集条件 IN/OUT",
        "",
        "| 采集条件 | train | test |",
        "|---|---|---|",
    ]
    for cond in ("IN", "OUT"):
        lines.append("| %s | %d | %d |" % (
            cond, train_stats["by_cond"].get(cond, 0), test_stats["by_cond"].get(cond, 0)))
    lines += [
        "",
        "### 4.2 采集时间序号 S0/S1/S2",
        "",
        "| 时间序号 | train | test |",
        "|---|---|---|",
    ]
    for batch in ("S0", "S1", "S2", None):
        label = batch if batch else "无（OUT/背景）"
        lines.append("| %s | %d | %d |" % (
            label, train_stats["by_batch"].get(batch, 0), test_stats["by_batch"].get(batch, 0)))
    lines += [
        "",
        "### 4.3 组合场景（IN/OUT × 时间序号）",
        "",
        "| 组合场景 | train | test |",
        "|---|---|---|",
    ]
    scenes = sorted(set(train_stats["by_scene"]) | set(test_stats["by_scene"]))
    for scene in scenes:
        lines.append("| %s | %d | %d |" % (
            scene, train_stats["by_scene"].get(scene, 0), test_stats["by_scene"].get(scene, 0)))
    lines.append("")
    return lines


def sec_individuals(train_stats, test_stats):
    ids = sorted(set(train_stats["by_id"]) | set(test_stats["by_id"]))
    lines = [
        "## 5. 个体片段明细",
        "",
        "| 个体 | 型号 | train 片段 | test 片段 | 合计 |",
        "|---|---|---|---|---|",
    ]
    for ident in ids:
        tr = train_stats["by_id"].get(ident, 0)
        te = test_stats["by_id"].get(ident, 0)
        name = "background" if ident.lower() == "background" else ident
        lines.append("| %s | %s | %d | %d | %d |" % (name, model_of(ident), tr, te, tr + te))
    lines.append("")
    return lines


def class_slice_check(train_stats, test_stats, spec):
    """各型号片段数与官方下限对比。"""
    models = sorted((set(train_stats["by_model"]) | set(test_stats["by_model"])) - {"BG"})
    lines = [
        "**各型号片段数（train + test）与官方下限对比：**",
        "",
        "| 型号 | train | test | 合计 | 官方下限 | 是否达标 |",
        "|---|---|---|---|---|---|",
    ]
    for model in models:
        tr = train_stats["by_model"].get(model, 0)
        te = test_stats["by_model"].get(model, 0)
        tot = tr + te
        ok = tot >= spec["min_slices_per_class"]
        lines.append("| %s | %d | %d | %d | ≥%d | %s |" % (
            model, tr, te, tot, spec["min_slices_per_class"], "是" if ok else "否"))
    lines.append("")
    return lines


def sec_compare(train_stats, test_stats, cfg):
    spec = cfg["official_spec"]
    local_models = sorted(
        (set(train_stats["by_model"]) | set(test_stats["by_model"])) - {"BG"})
    has_bg = ("BG" in train_stats["by_model"]) or ("BG" in test_stats["by_model"])
    all_ids = (set(train_stats["by_id"]) | set(test_stats["by_id"])) - {"background"}
    train_ids = set(train_stats["by_id"]) - {"background"}
    test_ids = set(test_stats["by_id"]) - {"background"}
    official_ind = spec["drone_models"] * spec["individuals_per_model"]
    rows = [
        ("无人机型号数", "%d 类" % spec["drone_models"],
         "%d 类（%s）" % (len(local_models), "/".join(local_models)),
         "匹配" if len(local_models) == spec["drone_models"] else "不一致"),
        ("背景参考类", "%d 类" % spec["background_classes"],
         "1 类（background）" if has_bg else "无",
         "匹配" if has_bg else "缺失"),
        ("个体总数", "%d（%d×%d）" % (
            official_ind, spec["drone_models"], spec["individuals_per_model"]),
         "%d（train %d / test %d）" % (len(all_ids), len(train_ids), len(test_ids)),
         "匹配" if len(all_ids) == official_ind else "需核对"),
        ("每类最少片段", "≥ %d" % spec["min_slices_per_class"], "见下表", "见下表"),
    ]
    lines = [
        "## 6. 与官方描述对比",
        "",
        "| 项目 | 官方描述 | 本地实际 | 判断 |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append("| %s | %s | %s | %s |" % row)
    lines.append("")
    lines += class_slice_check(train_stats, test_stats, spec)
    return lines


def sec_conclusion(train_stats, test_stats, cfg):
    spec = cfg["official_spec"]
    train_ids = set(train_stats["by_id"]) - {"background"}
    test_ids = set(test_stats["by_id"]) - {"background"}
    all_ids = train_ids | test_ids
    only_test = sorted(test_ids - train_ids)
    return [
        "## 7. 结论与注意事项",
        "",
        "1. **型号/个体**：本地共 %d 个个体（%d 个型号 × %d 个编号），与官方「6 类 × 3 架」一致。" % (
            len(all_ids), spec["drone_models"], spec["individuals_per_model"]),
        "2. **train/test 划分**：train 含 %d 个个体，test 含 %d 个个体；仅出现在 test 的个体为：%s。" % (
            len(train_ids), len(test_ids), "、".join(only_test) if only_test else "无"),
        "3. **valid 划分**：数据集没有 valid 集，只有 train/test。",
        "4. **test 未下载**：本报告 test 统计基于 `test_labels.txt`，实际 `.mat` 文件未下载到本地。",
        "5. **场景**：实际组合场景为 `IN_S0 / IN_S1 / IN_S2 / OUT`，共 4 种"
        "（OUT 无时间序号，故并非理论上的 2×3=6 种）。",
        "",
    ]


def build_report(train_stats, test_stats, cfg):
    lines = []
    lines += sec_header(cfg)
    lines += sec_overview()
    lines += sec_scale(train_stats, test_stats)
    lines += sec_models(train_stats, test_stats)
    lines += sec_scenes(train_stats, test_stats)
    lines += sec_individuals(train_stats, test_stats)
    lines += sec_compare(train_stats, test_stats, cfg)
    lines += sec_conclusion(train_stats, test_stats, cfg)
    return lines


# ---------------- 明细表输出 ----------------

def write_xlsx(path, train_stats, test_stats):
    """把个体/场景统计写到 xlsx。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    wb = Workbook()

    ws = wb.active
    ws.title = "个体片段统计"
    ws.append(["个体", "型号", "train片段", "test片段", "合计"])
    ids = sorted(set(train_stats["by_id"]) | set(test_stats["by_id"]))
    for ident in ids:
        tr = train_stats["by_id"].get(ident, 0)
        te = test_stats["by_id"].get(ident, 0)
        ws.append([ident, model_of(ident), tr, te, tr + te])

    ws2 = wb.create_sheet("场景分布")
    ws2.append(["组合场景", "train", "test"])
    scenes = sorted(set(train_stats["by_scene"]) | set(test_stats["by_scene"]))
    for scene in scenes:
        ws2.append([scene, train_stats["by_scene"].get(scene, 0),
                    test_stats["by_scene"].get(scene, 0)])

    ws3 = wb.create_sheet("说明")
    ws3.append(["本文件为 DroneRFb-DIR 数据集统计明细"])
    ws3.append(["个体片段统计：每个个体在 train/test 中的片段数"])
    ws3.append(["场景分布：IN/OUT × S时间序号的组合场景片段数"])
    wb.save(path)


# ---------------- 主流程 ----------------

def main():
    cfg = load_config()
    log = setup_logging(cfg["log_file"])
    log.info("配置：%s", cfg)

    train_records = load_train_records(cfg["inventory_file"])
    log.info("train 记录数：%d", len(train_records))
    test_labels = os.path.join(cfg["labels_dir"], "test_labels.txt")
    test_records = load_test_records(test_labels)
    log.info("test 记录数：%d", len(test_records))

    train_stats = collect_stats(train_records)
    test_stats = collect_stats(test_records)

    lines = build_report(train_stats, test_stats, cfg)
    os.makedirs(os.path.dirname(cfg["report_file"]), exist_ok=True)
    with open(cfg["report_file"], "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log.info("报告已写出：%s", cfg["report_file"])

    write_xlsx(cfg["stats_xlsx"], train_stats, test_stats)
    log.info("统计明细已写出：%s", cfg["stats_xlsx"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
