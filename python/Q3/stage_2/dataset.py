# -*- coding: utf-8 -*-
"""Q3 数据集：用 memmap 从 npy 缓存按索引读单通道灰度图，不整份载入内存。"""

import numpy as np
import torch
from torch.utils.data import Dataset


def normalize(img, cache_dtype, db_floor):
    """把缓存值统一到 0~1：uint8 除 255；dB 值按 [db_floor, 0] 线性拉伸。"""
    arr = np.asarray(img, dtype=np.float32)
    if str(cache_dtype) == "uint8":
        return arr / 255.0
    return (arr - float(db_floor)) / (-float(db_floor))


class StftCacheDataset(Dataset):
    """只持有 memmap 句柄与索引列表，实际像素按需读取。"""

    def __init__(self, x_path, y_path, indices, cache_dtype, db_floor):
        self.X = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.cache_dtype = cache_dtype
        self.db_floor = float(db_floor)

    def __len__(self):
        return int(self.indices.size)

    def __getitem__(self, i):
        j = int(self.indices[i])
        img = normalize(self.X[j], self.cache_dtype, self.db_floor)
        return torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0), int(self.y[j])
