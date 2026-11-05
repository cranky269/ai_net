# -*- coding: utf-8 -*-
"""多卷流式 ZIP 提取器（处理跨卷文件，只提取指定目录）。

与普通解压不同：本类逐卷扫描"本地文件头"，不依赖中央目录，因此可以
"下载一卷 → 提取 → 删除一卷"。跨卷截断的文件会用 carry-over 缓冲，
在下一个卷开头续接完成。

用法
----
    ext = StreamExtractor(out_dir, keep_test=False, logger=log)
    ext.process_volume("vol.z18", start_offset=49193403)  # 首个卷可指定起始偏移
    ext.process_volume("vol.z19")
    ...
    ext.finish()
"""

import binascii
import logging
import os
import pickle
import sys
import zlib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage_1"))
from zip_carver import (  # noqa: E402
    read_local_header, _find_next_signature, _skip_data_descriptor)

LOCAL_SIG = b"PK\x03\x04"


def cond_of(basename):
    """从文件名判断采集条件：IN / OUT / BG / ?。"""
    stem = basename[:-4] if basename.lower().endswith(".mat") else basename
    for part in stem.split("_")[1:]:
        if part in ("IN", "OUT"):
            return part
    return "BG" if "background" in basename.lower() else "?"


class StreamExtractor:
    def __init__(self, out_dir, keep_test=False, verify_crc=True,
                 logger=None, extensions=(".mat", ".txt"),
                 name_map=None, keep_conds=None, pending_path=None):
        self.out_dir = out_dir
        self.keep_test = keep_test
        self.verify_crc = verify_crc
        self.log = logger or logging.getLogger(__name__)
        self.extensions = {e.lower() for e in extensions}
        self.name_map = name_map or {}            # 数字名 -> 语义名（test 专用重命名）
        self.keep_conds = set(keep_conds) if keep_conds else None  # 采集条件白名单
        self.pending_path = pending_path or os.path.join(out_dir, "_pending.pkl")
        self.pending = None          # 跨卷未完成的条目
        self.count_kept = 0
        self.count_skipped = 0
        os.makedirs(out_dir, exist_ok=True)
        self.pending = self._load_pending()

    # ---- 跨卷状态持久化 ----
    def _load_pending(self):
        """从磁盘恢复上次中断的跨卷条目，没有则返回 None。"""
        if not os.path.exists(self.pending_path):
            return None
        try:
            with open(self.pending_path, "rb") as f:
                pending = pickle.load(f)
            if pending:
                self.log.info("恢复跨卷文件 %s（已收 %d/%d 字节）",
                              pending["name"], len(pending["buf"]), pending["csize"])
            return pending
        except Exception as exc:
            self.log.warning("恢复跨卷状态失败，忽略：%s", exc)
            return None

    def _save_pending(self):
        """把未完成的跨卷条目写入磁盘；没有则删除旧文件。"""
        if self.pending is not None:
            with open(self.pending_path, "wb") as f:
                pickle.dump(self.pending, f)
        elif os.path.exists(self.pending_path):
            os.remove(self.pending_path)

    # ---- 判定 ----
    def _should_keep(self, name):
        if name.endswith("/"):
            return False
        if os.path.splitext(name)[1].lower() not in self.extensions:
            return False
        if not self.keep_test and "/test/" in name.replace("\\", "/"):
            return False
        return True

    # ---- 单条目还原与落盘 ----
    def _flush(self, header, buf):
        data = None
        if header["flags"] & 0x0001:
            self.log.warning("跳过加密条目 %s", header["name"])
            return
        if header["method"] == 0:
            data = buf
        elif header["method"] == 8:
            try:
                data = zlib.decompress(bytes(buf), -15)
            except zlib.error as exc:
                self.log.warning("解压失败 %s: %s", header["name"], exc)
                return
        else:
            self.log.warning("未知压缩方式 %d: %s", header["method"], header["name"])
            return
        if self.verify_crc and header["crc"]:
            if (binascii.crc32(data) & 0xFFFFFFFF) != header["crc"]:
                self.log.warning("CRC 失败 %s，丢弃", header["name"])
                return
        if len(data) != header["usize"]:
            self.log.warning("长度不符 %s: %d != %d", header["name"], len(data), header["usize"])
            return
        if not self._should_keep(header["name"]):
            self.count_skipped += 1
            return
        base = os.path.basename(header["name"].replace("\\", "/"))
        if self.name_map:
            if base not in self.name_map:
                self.count_skipped += 1
                return
            base = self.name_map[base]
        if self.keep_conds is not None and base.lower().endswith(".mat"):
            if cond_of(base) not in self.keep_conds:
                self.count_skipped += 1
                return
        target = self._unique(base)
        with open(target, "wb") as f:
            f.write(data)
        self.count_kept += 1
        self.log.info("提取 %s -> %s (%.2f MB)", header["name"], base, len(data) / 1048576.0)

    def _unique(self, base):
        target = os.path.join(self.out_dir, base)
        if not os.path.exists(target):
            return target
        stem, ext = os.path.splitext(base)
        idx = 1
        while os.path.exists(target):
            target = os.path.join(self.out_dir, "%s_%d%s" % (stem, idx, ext))
            idx += 1
        return target

    # ---- 逐卷处理 ----
    def process_volume(self, path, start_offset=0):
        total = os.path.getsize(path)
        kept_before = self.count_kept
        with open(path, "rb") as f:
            # 1) 先续接上一卷遗留的跨卷条目
            pos = start_offset
            if self.pending is not None:
                need = self.pending["csize"] - len(self.pending["buf"])
                chunk = f.read(need)
                self.pending["buf"] += chunk
                if len(self.pending["buf"]) < self.pending["csize"]:
                    self.log.info("文件 %s 仍跨卷，继续等待", self.pending["name"])
                    self._save_pending()
                    return 0
                self._flush(self.pending["header"], bytes(self.pending["buf"]))
                self.pending = None
                pos = len(chunk)
            # 2) 顺序扫描本地文件头
            while pos < total:
                f.seek(pos)
                if f.read(4) != LOCAL_SIG:
                    nxt = _find_next_signature(f, pos, total)
                    if nxt is None:
                        break
                    pos = nxt
                    f.seek(pos)
                header = read_local_header(f, pos)
                if header is None:
                    break
                name = header["name"]
                if name.endswith("/"):
                    pos = header["data_offset"]
                    continue
                doff = header["data_offset"]
                need = header["csize"]
                f.seek(doff)
                buf = f.read(need)
                if len(buf) < need:
                    self.pending = {"header": header, "buf": bytearray(buf),
                                    "name": name, "csize": need}
                    self.log.info("文件 %s 跨卷（%d/%d 字节），等待下一卷",
                                  name, len(buf), need)
                    break
                self._flush(header, buf)
                nxt = doff + need
                if header["flags"] & 0x0008:
                    nxt = _skip_data_descriptor(f, nxt, header["zip64"])
                pos = nxt
        self.log.info("卷 %s 完成：累计提取 %d、跳过 %d",
                      os.path.basename(path), self.count_kept, self.count_skipped)
        self._save_pending()
        return self.count_kept - kept_before

    def finish(self):
        if self.pending is not None:
            self.log.warning("存在未完成的跨卷文件 %s（缺少后续卷），已保存状态供下次续接",
                             self.pending["name"])
        self._save_pending()
