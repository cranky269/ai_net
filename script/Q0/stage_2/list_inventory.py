# -*- coding: utf-8 -*-
"""抓取分卷归档的中央目录，生成完整文件清单（不下载数据本体）。

用法
----
    python script/Q0/stage_2/list_inventory.py <urls.txt> <out.json> [卷数=31]

输出
----
JSON：{zip_size, n_volumes, entries:[{name,crc,csize,usize,offset,method}]}
同时打印按目录/按类别的数量统计。
"""

import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zip_central import locate_central, parse_central_directory  # noqa: E402

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0"
VOLUME_SIZE = 2147483648  # 2 GiB


def http_get(url, start=None, end=None, timeout=120):
    headers = {"User-Agent": UA}
    if start is not None:
        rng = "bytes=%d-%s" % (start, "" if end is None else end)
        headers["Range"] = rng
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def head_size(url):
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return int(resp.headers["Content-Length"])


def parse_urls(path):
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(line).query)
            name = q.get("fileName", ["unknown"])[0]
            entries.append((name, line))
    def key(item):
        if item[0].endswith(".zip"):
            return (1, 0)
        m = re.search(r"z(\d+)", item[0])
        return (0, int(m.group(1)) if m else 0)
    return sorted(entries, key=key)


def summarize(entries):
    from collections import Counter
    c_dir = Counter()
    c_train = Counter()
    c_test = Counter()
    for e in entries:
        n = e["name"].replace("\\", "/")
        if n.endswith("/"):
            continue
        parts = n.split("/")
        top = parts[1] if len(parts) > 1 else "(root)"
        c_dir[top] += 1
        label = None
        m = re.match(r"([A-G])(\d)?_?(IN|OUT)?", parts[-1]) if len(parts) > 1 else None
        # 训练集文件名形如 A1_IN_S0_slice_2.mat，测试集为纯数字
        if "/train/" in n:
            c_train[n.split("_")[0]] += 1
        elif "/test/" in n:
            c_test["test"] += 1
    return c_dir, c_train, c_test


def main():
    if len(sys.argv) < 3:
        print("用法: python list_inventory.py <urls.txt> <out.json> [卷数=31]")
        return 1
    urls_file, out_json = sys.argv[1], sys.argv[2]
    n_vol = int(sys.argv[3]) if len(sys.argv) > 3 else 31

    urls = parse_urls(urls_file)
    if not urls:
        print("未解析到下载链接")
        return 1
    zip_name, zip_url = urls[-1]
    print("末卷:", zip_name)

    zip_size = head_size(zip_url)
    print("末卷大小: %.2f GB" % (zip_size / 1073741824.0))

    tail = http_get(zip_url, start=max(0, zip_size - 262144))
    info = locate_central(tail)
    if info is None:
        print("未找到中央目录（EOCD/Zip64 解析失败）")
        return 1
    coff, csize = info
    print("中央目录 offset=%d size=%d" % (coff, csize))

    within = coff  # coff 是相对末卷（.zip）起始的偏移
    data = http_get(zip_url, start=within, end=within + csize - 1)
    entries = parse_central_directory(data)
    print("解析条目数: %d" % len(entries))

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"zip_size": zip_size, "n_volumes": n_vol, "entries": entries},
                  f, ensure_ascii=False)

    c_dir, c_train, c_test = summarize(entries)
    print("\n== 顶层目录分布 ==")
    for k, v in c_dir.most_common():
        print("  %-20s %d" % (k, v))
    print("\n== train/ 各类(机型个体)数量 ==")
    for k in sorted(c_train):
        print("  %-10s %d" % (k, c_train[k]))
    print("\n== test/ 数量 ==", c_test.get("test", 0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
