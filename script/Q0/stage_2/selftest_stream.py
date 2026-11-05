# -*- coding: utf-8 -*-
"""StreamExtractor 的最小单元自测（合成多卷 + 跨卷文件）。

构造一个逻辑流：目录条目 + test 文件(应跳过) + 跨卷 train 文件 + 普通 train 文件 + 标签 txt，
按字节切分成 3 个"卷"，验证流式提取能正确：
    1. 提取 train 文件与 txt；
    2. 跳过 test 文件；
    3. 正确续接跨卷文件；
    4. 校验 CRC 与内容。
"""

import logging
import os
import shutil
import struct
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stream_extract import StreamExtractor  # noqa: E402

WORK_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "output", "Q0", "_selftest_stream"))

LOCAL_SIG = b"PK\x03\x04"


def build_entry(name, data, method=8):
    raw = name.encode("utf-8")
    crc = zlib.crc32(data) & 0xFFFFFFFF
    if method == 8:
        body = zlib.compressobj(9, zlib.DEFLATED, -15)
        payload = body.compress(data) + body.flush()
    else:
        payload = data
    head = struct.pack("<4sHHHHHIIIHH", LOCAL_SIG, 20, 0, method, 0, 0x21,
                       crc, len(payload), len(data), len(raw), 0)
    return head + raw + payload


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("selftest_stream")

    if os.path.isdir(WORK_DIR):
        shutil.rmtree(WORK_DIR)
    os.makedirs(WORK_DIR)

    stream = b""
    stream += build_entry("root/", b"", method=0)                 # 目录条目
    stream += build_entry("root/test/0.mat", b"TEST" * 2000)      # 应跳过
    stream += build_entry("root/train/A.mat", b"A" * 5000)        # 跨卷文件
    stream += build_entry("root/train/B.mat", b"B" * 6000)        # 普通文件
    stream += build_entry("root/train_labels.txt", b"l1\nl2\n")   # 标签 txt

    # 切分点：落在 A.mat 的数据中间
    split1 = len(build_entry("root/", b"", method=0)) + len(build_entry("root/test/0.mat", b"TEST" * 2000)) + 500
    split2 = split1 + 3000
    vols = [stream[:split1], stream[split1:split2], stream[split2:]]
    paths = []
    for i, v in enumerate(vols):
        p = os.path.join(WORK_DIR, "vol%d.z" % (i + 1))
        with open(p, "wb") as f:
            f.write(v)
        paths.append(p)

    ext = StreamExtractor(os.path.join(WORK_DIR, "out"), keep_test=False, logger=log)
    for p in paths:
        ext.process_volume(p)
    ext.finish()

    out = os.path.join(WORK_DIR, "out")
    assert os.path.exists(os.path.join(out, "A.mat")), "A.mat 未提取"
    assert os.path.exists(os.path.join(out, "B.mat")), "B.mat 未提取"
    assert os.path.exists(os.path.join(out, "train_labels.txt")), "txt 未提取"
    assert not os.path.exists(os.path.join(out, "0.mat")), "test 文件不应被提取"

    with open(os.path.join(out, "A.mat"), "rb") as f:
        assert f.read() == b"A" * 5000, "A.mat 内容错误"
    with open(os.path.join(out, "B.mat"), "rb") as f:
        assert f.read() == b"B" * 6000, "B.mat 内容错误"
    assert ext.pending is None, "不应有未完成的跨卷文件"
    assert ext.count_skipped == 1, "应恰好跳过 1 个 test 文件"

    log.info("PASS：跨卷续接、train 提取、test 跳过、CRC 校验均正确")
    return 0


if __name__ == "__main__":
    sys.exit(main())
