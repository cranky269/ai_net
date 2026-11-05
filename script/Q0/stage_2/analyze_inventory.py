# -*- coding: utf-8 -*-
"""分析中央目录清单，统计每类/场景/采集批次的片段分布。

输入：list_inventory.py 输出的 inventory.json
输出：output/Q0/inventory_summary.xlsx + 控制台统计
"""

import json
import os
import re
import sys
from collections import Counter, defaultdict

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def parse_train_name(basename):
    """解析训练集文件名，返回 dict；解析失败返回 None。"""
    stem = basename[:-4] if basename.endswith(".mat") else basename
    parts = stem.split("_")
    if not parts:
        return None
    ident = parts[0]
    scene = None
    batch = None
    slice_no = None
    rest = parts[1:]
    if rest and rest[0] in ("IN", "OUT"):
        scene = rest[0]
        rest = rest[1:]
    if rest and re.fullmatch(r"S\d+", rest[0]):
        batch = rest[0]
        rest = rest[1:]
    m = re.search(r"slice[_-]?(\d+)$", "_".join(rest))
    if m:
        slice_no = int(m.group(1))
    return {"id": ident, "scene": scene, "batch": batch, "slice": slice_no}


def type_of(ident):
    """返回机型（A..G 或 Background）。"""
    m = re.match(r"([A-G])", ident)
    if m:
        return m.group(1)
    if ident.lower().startswith("b") or ident.lower() == "background":
        return "B(背景)"
    return "?"


def analyze(inventory_path, out_xlsx):
    with open(inventory_path, encoding="utf-8") as f:
        inv = json.load(f)
    entries = inv["entries"]

    train = [e for e in entries if "/train/" in e["name"].replace("\\", "/")]
    test = [e for e in entries if "/test/" in e["name"].replace("\\", "/")]
    print("train 文件数=%d, test 文件数=%d" % (len(train), len(test)))

    rows = []
    parsed = []
    for e in train:
        base = os.path.basename(e["name"].replace("\\", "/"))
        info = parse_train_name(base)
        if info is None:
            rows.append({"file": base, "id": "?", "type": "?",
                         "scene": None, "batch": None, "slice": None,
                         "csize": e["csize"], "usize": e["usize"]})
            continue
        info["file"] = base
        info["type"] = type_of(info["id"])
        info["csize"] = e["csize"]
        info["usize"] = e["usize"]
        parsed.append(info)

    # 1) 按个体（19 类）
    by_id = Counter(p["id"] for p in parsed)
    print("\n== 按个体(标签)片段数 ==")
    for k in sorted(by_id):
        print("  %-8s %d" % (k, by_id[k]))

    # 2) 按机型（型号级 7 类）
    by_type = Counter(p["type"] for p in parsed)
    print("\n== 按机型片段数 ==")
    for k in sorted(by_type):
        print("  %-10s %d" % (k, by_type[k]))

    # 3) IN/OUT 分布
    by_scene = Counter(p["scene"] for p in parsed)
    print("\n== 场景(IN/OUT)分布 ==", dict(by_scene))

    # 4) 每个(个体,场景)的 S 批次数量（判断能否按记录分组划分）
    batch_per_group = defaultdict(set)
    for p in parsed:
        key = (p["id"], p["scene"])
        if p["batch"]:
            batch_per_group[key].add(p["batch"])
    group_counts = {k: len(v) for k, v in batch_per_group.items()}
    print("\n== 每个(个体,场景)的 S 批次数量 ==")
    for k in sorted(group_counts):
        print("  %-16s %d 批次" % (str(k), group_counts[k]))
    print("  最少批次=%d, 最多批次=%d" % (
        min(group_counts.values()), max(group_counts.values())))

    # 5) 背景类命名样例
    bg = [p["file"] for p in parsed if p["type"] == "B(背景)"]
    print("\n== 背景类文件样例(前5) ==")
    for f in bg[:5]:
        print("  ", f)

    # 写出 xlsx
    os.makedirs(os.path.dirname(out_xlsx), exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.title = "train文件清单"
    header = ["文件名", "个体", "机型", "场景", "S批次", "slice序号",
              "压缩大小", "未压缩大小"]
    ws.append(header)
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="4472C4")
    for p in sorted(parsed, key=lambda x: x["file"]):
        ws.append([p["file"], p["id"], p["type"], p["scene"],
                   p["batch"], p["slice"], p["csize"], p["usize"]])
    wb.save(out_xlsx)
    print("\n已写出:", out_xlsx)


if __name__ == "__main__":
    inv = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        PROJECT_ROOT, "output", "Q0", "inventory.json")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        PROJECT_ROOT, "output", "Q0", "inventory_summary.xlsx")
    analyze(inv, out)
