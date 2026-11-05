# -*- coding: utf-8 -*-
"""下载 + 流式提取编排：只下载含 train 的卷，逐卷下载→提取→删除。

流程
----
1. 读 inventory.json（已由 list_inventory.py 从中央目录解析）；
2. 确定 train 文件所在的卷范围与起始偏移（跳过前面的 test 卷，省约 38GB 下载）；
3. 逐卷：多线程下载 → 流式提取 train/.mat 与标签 txt → 删除卷；
4. 断点续传：记录已完成卷序号，中断后重跑自动跳过。

用法
----
    python script/Q0/stage_2/run_download.py
"""

import argparse
import json
import logging
import os
import re
import sys
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from downloader import download_file, head_size   # noqa: E402
from stream_extract import StreamExtractor        # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
VOLUME_SIZE = 2147483648


def load_config(path=CONFIG_PATH):
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    for key in ("urls_file", "inventory_file", "log_file"):
        if key in cfg and not os.path.isabs(cfg[key]):
            cfg[key] = os.path.join(PROJECT_ROOT, cfg[key])
    return cfg


def setup_logging(log_file, level="INFO"):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", errors="replace")
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for h in list(root.handlers):
        root.removeHandler(h)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter(fmt))
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt))
    root.addHandler(sh)
    root.addHandler(fh)
    return logging.getLogger("Q0.stage2")


def parse_urls(path):
    """解析直链文件，返回按卷序排列的 (disk, name, url) 列表。"""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = parse_qs(urlparse(line).query)
            name = q.get("fileName", ["unknown"])[0]
            if name.endswith(".zip"):
                disk = 31
            else:
                m = re.search(r"z(\d+)", name)
                disk = int(m.group(1)) - 1 if m else -1
            rows.append((disk, name, line))
    return sorted(rows, key=lambda r: r[0])


def read_state(state_file):
    if os.path.exists(state_file):
        with open(state_file, encoding="utf-8") as f:
            return int(f.read().strip() or -1)
    return -1


def write_state(state_file, disk):
    os.makedirs(os.path.dirname(state_file), exist_ok=True)
    with open(state_file, "w", encoding="utf-8") as f:
        f.write(str(disk))


def load_test_name_map(path):
    """读取 test_labels.txt -> {数字名: 语义名}，用于 test 提取时重命名。"""
    name_map = {}
    if not path or not os.path.exists(path):
        return name_map
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                name_map["%s.mat" % parts[1]] = parts[0]
    return name_map


def resolve_target(split, cfg, entries, log):
    """根据划分确定下载目标：卷范围、起始偏移、目录、状态文件与过滤规则。"""
    if split == "test":
        test = [e for e in entries
                if "/test/" in e["name"].replace("\\", "/")
                and e["name"].lower().endswith(".mat")]
        if not test:
            raise SystemExit("清单中无 test 文件，请先运行 list_inventory.py")
        name_map = load_test_name_map(cfg.get("test_labels_file"))
        first_disk = min(e["disk"] for e in test)
        last_disk = max(e["disk"] for e in test)
        first_in_disk = [e for e in test if e["disk"] == first_disk]
        start_offset = min(e["offset"] for e in first_in_disk)
        log.info("test 文件 %d 个，标签映射 %d 条，卷 %d..%d",
                 len(test), len(name_map), first_disk, last_disk)
        return {
            "first_disk": first_disk,
            "last_disk": last_disk,
            "start_offset": start_offset,
            "extract_dir": cfg["test_extract_dir"],
            "state_file": cfg["test_state_file"],
            "name_map": name_map,
            "keep_conds": set(cfg.get("test_keep_conds", [])),
            "keep_test": True,
        }

    train = [e for e in entries if "/train/" in e["name"].replace("\\", "/")]
    txt = [e for e in entries if e["name"].lower().endswith(".txt")]
    if not train:
        raise SystemExit("清单中无 train 文件，请先运行 list_inventory.py")
    keep_entries = train + txt
    first_disk = min(e["disk"] for e in keep_entries)
    last_disk = max(e["disk"] for e in keep_entries)
    first_in_disk = [e for e in keep_entries if e["disk"] == first_disk]
    start_offset = min(e["offset"] for e in first_in_disk)
    total_bytes = sum(e["csize"] for e in train)
    log.info("train 文件 %d 个，压缩总量 %.2f GB，卷 %d..%d",
             len(train), total_bytes / 1073741824.0, first_disk, last_disk)
    return {
        "first_disk": first_disk,
        "last_disk": last_disk,
        "start_offset": start_offset,
        "extract_dir": cfg["extract_dir"],
        "state_file": cfg["state_file"],
        "name_map": {},
        "keep_conds": None,
        "keep_test": cfg.get("keep_test", False),
    }


def main():
    ap = argparse.ArgumentParser(description="下载并提取数据集（train/test）")
    ap.add_argument("--config", default=CONFIG_PATH)
    ap.add_argument("--split", choices=("train", "test"), default="train",
                    help="下载哪个划分（默认 train）")
    ap.add_argument("--limit", type=int, default=0,
                    help="最多处理几个卷（0=全部），用于测试")
    args = ap.parse_args()

    cfg = load_config(args.config)
    log = setup_logging(cfg["log_file"], cfg.get("log_level", "INFO"))
    log.info("配置：%s", cfg)

    with open(cfg["inventory_file"], encoding="utf-8") as f:
        inv = json.load(f)
    entries = inv["entries"]
    urls = parse_urls(cfg["urls_file"])
    disk_map = {d: (n, u) for d, n, u in urls}
    log.info("共 %d 个分卷" % len(urls))

    target = resolve_target(args.split, cfg, entries, log)

    orig_first_disk = target["first_disk"]
    first_disk = orig_first_disk
    state = read_state(target["state_file"])
    if state >= first_disk:
        log.info("从断点续传：已完成卷 %d，从卷 %d 开始", state, state + 1)
        first_disk = max(first_disk, state + 1)

    ext = StreamExtractor(target["extract_dir"], keep_test=target["keep_test"],
                          name_map=target["name_map"], keep_conds=target["keep_conds"],
                          logger=log)
    done = 0
    for disk in range(first_disk, target["last_disk"] + 1):
        name, url = disk_map[disk]
        dest = os.path.join(cfg["volume_dir"], name)
        size = head_size(url)
        log.info("---- 下载卷 %s（disk %d, %.2f GB）----", name, disk, size / 1073741824.0)
        download_file(url, dest, size=size, n_threads=cfg.get("download_threads", 16),
                      logger=log)
        off = target["start_offset"] if (disk == orig_first_disk and state < orig_first_disk) else 0
        ext.process_volume(dest, start_offset=off)
        if cfg.get("delete_volume_after_extract", True):
            os.remove(dest)
            log.info("已删除卷 %s", name)
        write_state(target["state_file"], disk)
        done += 1
        if args.limit and done >= args.limit:
            log.info("达到 --limit=%d，提前停止", args.limit)
            break

    ext.finish()
    log.info("全部完成：提取 %d 个文件到 %s", ext.count_kept, target["extract_dir"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
