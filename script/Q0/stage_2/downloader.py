# -*- coding: utf-8 -*-
"""纯 Python 多线程分块下载器（基于 HTTP Range，支持断点续传）。

不依赖 aria2/IDM。服务器需支持 Range（本数据集实测支持 206）。
每个文件切成 n_threads 块并发下载，每块写到 ``<dest>.partNNN``，
全部完成后合并并校验总大小；中断后重跑会自动跳过已完成的块。
"""

import logging
import os
import time
import urllib.request

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0"
DEFAULT_RETRIES = 5


def head_size(url, retries=DEFAULT_RETRIES):
    """返回远端文件大小。

    优先用 HEAD 请求；若服务器对 HEAD 响应慢/超时，则降级为
    GET + Range（只取第 1 字节），从 Content-Range 响应头解析总大小。
    全程带重试，避免偶发网络抖动导致整个程序崩溃。
    """
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return int(resp.headers["Content-Length"])
        except Exception as exc:  # 网络抖动重试
            last_err = exc
            time.sleep(min(2 ** attempt, 10))

    # HEAD 不可用，改用 GET + Range 探测（本数据集服务器实测支持 206）
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as resp:
                cr = resp.headers.get("Content-Range")
                if cr:
                    total = int(cr.rsplit("/", 1)[1])
                    if total > 0:
                        return total
                cl = resp.headers.get("Content-Length")
                if cl:
                    return int(cl)
        except Exception as exc:
            last_err = exc
            time.sleep(min(2 ** attempt, 10))
    raise RuntimeError("获取文件大小失败 %s: %s" % (url, last_err))


def fetch_range(url, start, end, retries=DEFAULT_RETRIES):
    """下载 [start, end] 闭区间字节，返回 bytes。"""
    headers = {"Range": "bytes=%d-%d" % (start, end), "User-Agent": UA}
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=180) as resp:
                return resp.read()
        except Exception as exc:  # 网络抖动重试
            last_err = exc
            time.sleep(min(2 ** attempt, 10))
    raise RuntimeError("下载 %s 区间 %d-%d 失败: %s" % (url, start, end, last_err))


def _download_part(url, start, end, part_path):
    """下载单个分块到 part 文件（已完整则跳过）。"""
    need = end - start + 1
    if os.path.exists(part_path) and os.path.getsize(part_path) == need:
        return part_path
    data = fetch_range(url, start, end)
    if len(data) != need:
        raise IOError("分块长度不符: 期望 %d 实际 %d" % (need, len(data)))
    with open(part_path, "wb") as f:
        f.write(data)
    return part_path


def download_file(url, dest, size=None, n_threads=16, logger=None):
    """多线程分块下载一个文件到 dest，返回 dest 路径。

    若 dest 已存在且大小正确则直接返回；否则按分块续传后合并。
    """
    log = logger or logging.getLogger(__name__)
    if size is None:
        size = head_size(url)
    if os.path.exists(dest) and os.path.getsize(dest) == size:
        log.info("已存在且完整，跳过：%s", os.path.basename(dest))
        return dest
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)

    chunk = (size + n_threads - 1) // n_threads
    ranges = [(i * chunk, min((i + 1) * chunk, size) - 1) for i in range(n_threads)]
    part_paths = ["%s.part%03d" % (dest, i) for i in range(n_threads)]

    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(max_workers=n_threads) as pool:
        futs = {pool.submit(_download_part, url, s, e, p): i
                for i, (s, e) in enumerate(ranges) for p in [part_paths[i]]}
        for fut in cf.as_completed(futs):
            fut.result()  # 抛出异常则中断

    with open(dest, "wb") as out:
        for p in part_paths:
            with open(p, "rb") as f:
                out.write(f.read())
    if os.path.getsize(dest) != size:
        raise IOError("合并后大小不符: 期望 %d 实际 %d" % (size, os.path.getsize(dest)))
    for p in part_paths:
        if os.path.exists(p):
            os.remove(p)
    log.info("下载完成：%s (%.2f GB)", os.path.basename(dest), size / 1073741824.0)
    return dest
