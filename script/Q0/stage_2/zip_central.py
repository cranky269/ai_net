# -*- coding: utf-8 -*-
"""ZIP 中央目录解析（含 Zip64）。

用途
----
分卷 ZIP 的中央目录（central directory）位于最后一个卷的末尾。通过解析它，
可以在**不下载整份数据**的情况下，得到所有条目的：名称、压缩方式、CRC、
压缩/未压缩大小、以及在归档流中的绝对偏移。据此可对需要的文件做定点 Range 下载。

本模块只做"字节解析"，不做任何网络操作，便于单元测试。
"""

import struct

EOCD_SIG = b"PK\x05\x06"           # 中央目录结束标记
ZIP64_EOCD_SIG = b"PK\x06\x06"     # Zip64 中央目录结束记录
ZIP64_LOCATOR_SIG = b"PK\x06\x07"  # Zip64 结束定位器
CENTRAL_SIG = b"PK\x01\x02"        # 中央目录条目


def locate_central(tail):
    """从归档末尾字节中定位中央目录，返回 (offset, size) 或 None。

    tail 至少要包含：Zip64 EOCD 记录 + 定位器 + EOCD（含注释），一般取末尾 256KB 即可。
    """
    i = tail.rfind(EOCD_SIG)
    if i < 0:
        return None
    if i + 22 > len(tail):
        return None
    (_sig, _disk, _cdisk, _ndisk, ntot, csize, coff, clen) = struct.unpack_from(
        "<4sHHHHIIH", tail, i)
    if coff == 0xFFFFFFFF or csize == 0xFFFFFFFF:
        # 需要 Zip64 定位器
        j = tail.rfind(ZIP64_LOCATOR_SIG, 0, i)
        if j < 0 or j + 20 > len(tail):
            return None
        (_sig, _zdisk, zoff, _ztotal) = struct.unpack_from("<4sIQI", tail, j)
        del zoff
        k = tail.rfind(ZIP64_EOCD_SIG, 0, j)
        if k < 0 or k + 56 > len(tail):
            return None
        (_sig, _zsize, _vm, _vn, _d, _cd, _nd, _zntot,
         zcsize, zcoff) = struct.unpack_from("<4sQHHIIQQQQ", tail, k)
        csize, coff = zcsize, zcoff
    return int(coff), int(csize)


def parse_central_directory(data):
    """解析中央目录字节，返回条目列表（每个含 name/crc/csize/usize/offset/method）。"""
    entries = []
    pos = 0
    n = len(data)
    while pos + 46 <= n:
        if data[pos:pos + 4] != CENTRAL_SIG:
            break
        (_sig, _vmade, _vneed, flags, method, _mtime, _mdate, crc,
         csize, usize, nlen, elen, clen, diskstart, _iattr, _eattr,
         loff) = struct.unpack_from("<4sHHHHHHIIIHHHHHII", data, pos)
        name_start = pos + 46
        name = data[name_start:name_start + nlen].decode("utf-8", "replace")
        extra = data[name_start + nlen:name_start + nlen + elen]
        if csize == 0xFFFFFFFF or usize == 0xFFFFFFFF or loff == 0xFFFFFFFF:
            epos = 0
            while epos + 4 <= len(extra):
                tag, sz = struct.unpack_from("<HH", extra, epos)
                body = extra[epos + 4:epos + 4 + sz]
                if tag == 0x0001:
                    off = 0
                    if usize == 0xFFFFFFFF and off + 8 <= len(body):
                        usize = struct.unpack_from("<Q", body, off)[0]
                        off += 8
                    if csize == 0xFFFFFFFF and off + 8 <= len(body):
                        csize = struct.unpack_from("<Q", body, off)[0]
                        off += 8
                    if loff == 0xFFFFFFFF and off + 8 <= len(body):
                        loff = struct.unpack_from("<Q", body, off)[0]
                        off += 8
                epos += 4 + sz
        entries.append({
            "name": name,
            "flags": flags,
            "method": method,
            "crc": crc,
            "csize": int(csize),
            "usize": int(usize),
            "offset": int(loff),   # 相对本卷（disk）起始的偏移
            "disk": int(diskstart),  # 所在卷序号（0 = .z01，…，31 = .zip）
        })
        pos += 46 + nlen + elen + clen
    return entries
