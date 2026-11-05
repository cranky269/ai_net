# -*- coding: utf-8 -*-
"""按采集条件（IN/OUT/背景）统计数据集实际大小。

用途：评估删除 OUT 后可释放的空间，以及仅下载 IN 时 test 所需磁盘。
数据来源：output/Q0/inventory.json + data/train_extracted/test_labels.txt
"""

import json
import os
import sys
from collections import defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
GB = 1073741824


def cond_of(semantic):
    """从语义文件名判断采集条件：IN / OUT / BG / ?。"""
    stem = semantic[:-4] if semantic.lower().endswith(".mat") else semantic
    for part in stem.split("_")[1:]:
        if part in ("IN", "OUT"):
            return part
    return "BG" if "background" in semantic.lower() else "?"


def load_test_label_map(path):
    """test_labels.txt -> {标号: 语义名}。"""
    mapping = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                mapping[int(parts[1])] = parts[0]
    return mapping


def stat_train(inv):
    """统计 train 各条件的大小。"""
    size = defaultdict(int)
    cnt = defaultdict(int)
    for entry in inv["entries"]:
        name = entry["name"].replace("\\", "/")
        if "/train/" in name and name.lower().endswith(".mat"):
            cond = cond_of(os.path.basename(name))
            size[cond] += entry.get("usize", 0)
            cnt[cond] += 1
    return cnt, size


def stat_test(inv, label_map):
    """统计 test 各条件的大小（靠 label_map 把数字名映射回语义名）。"""
    size = defaultdict(int)
    cnt = defaultdict(int)
    for entry in inv["entries"]:
        name = entry["name"].replace("\\", "/")
        if "/test/" in name and name.lower().endswith(".mat"):
            idx = int(os.path.basename(name)[:-4])
            semantic = label_map.get(idx)
            if not semantic:
                continue
            cond = cond_of(semantic)
            size[cond] += entry.get("usize", 0)
            cnt[cond] += 1
    return cnt, size


def report(title, cnt, size):
    print("==", title, "==")
    for cond in sorted(size, key=str):
        print("  %-4s 文件 %5d  大小 %.2f GB" % (cond, cnt[cond], size[cond] / GB))
    print("  合计 文件 %5d  大小 %.2f GB" % (sum(cnt.values()), sum(size.values()) / GB))


def main():
    inv = json.load(open(os.path.join(PROJECT_ROOT, "output", "Q0", "inventory.json"),
                         encoding="utf-8"))
    label_map = load_test_label_map(
        os.path.join(PROJECT_ROOT, "data", "train_extracted", "test_labels.txt"))
    report("train", *stat_train(inv))
    report("test", *stat_test(inv, label_map))
    return 0


if __name__ == "__main__":
    sys.exit(main())
