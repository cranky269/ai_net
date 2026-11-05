# -*- coding: utf-8 -*-
"""Q3 模型构建：把 YAML 架构描述翻译成 nn.Module。

两种写法：
  1) ``layers`` 逐层描述：conv / pool / adaptive_pool / flatten / dropout / fc
  2) 外部模型文件：``type: module`` + ``module: my_net.py`` + ``class: MyNet``
"""

import importlib.util
from pathlib import Path

import torch.nn as nn
import yaml

ACT_TABLE = {"relu": nn.ReLU, "relu6": nn.ReLU6, "gelu": nn.GELU,
             "leaky_relu": nn.LeakyReLU, "elu": nn.ELU, "silu": nn.SiLU}


def load_arch(path):
    """读 YAML 架构文件，并记录其所在目录（供外部模块相对路径解析）。"""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        arch = yaml.safe_load(fh)
    arch["_base_dir"] = str(path.parent)
    return arch


def _activation(name):
    """按名字取激活层。"""
    key = str(name or "relu").lower()
    if key not in ACT_TABLE:
        raise ValueError("未知激活函数: %s" % name)
    return ACT_TABLE[key](inplace=True)


def _norm_layer(name, channels):
    """按名字取归一化层，none 时用 Identity 占位。"""
    key = str(name or "none").lower()
    if key == "none":
        return nn.Identity()
    if key == "batch":
        return nn.BatchNorm2d(channels)
    if key == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    raise ValueError("未知归一化: %s" % name)


def _conv_block(spec, in_ch):
    """一个 conv(+norm+act+pool) 块，返回 (Sequential, 输出通道数)。"""
    out_ch = int(spec["out_channels"])
    conv = nn.Conv2d(in_ch, out_ch,
                     kernel_size=int(spec.get("kernel_size", 3)),
                     stride=int(spec.get("stride", 1)),
                     padding=int(spec.get("padding", 1)),
                     bias=str(spec.get("norm", "none")).lower() == "none")
    block = [conv, _norm_layer(spec.get("norm"), out_ch), _activation(spec.get("act"))]
    pool = int(spec.get("pool", 0) or 0)
    if pool > 1:
        block.append(nn.MaxPool2d(pool))
    return nn.Sequential(*block), out_ch


def _build_sequential(arch, num_classes):
    """按 layers 列表逐层搭建 nn.Sequential；最后一个 fc 的输出强制等于类别数。"""
    specs = arch["layers"]
    fc_positions = [i for i, s in enumerate(specs) if str(s.get("type")).lower() == "fc"]
    last_fc = fc_positions[-1] if fc_positions else -1
    ch = int(arch.get("input_channels", 1))
    flat_dim = None
    modules = []
    for i, spec in enumerate(specs):
        kind = str(spec.get("type", "")).lower()
        if kind == "conv":
            block, ch = _conv_block(spec, ch)
            modules.append(block)
        elif kind == "pool":
            modules.append(nn.MaxPool2d(int(spec.get("kernel_size", 2)),
                                        int(spec.get("stride", 2))))
        elif kind == "adaptive_pool":
            out = tuple(int(v) for v in spec["output"])
            modules.append(nn.AdaptiveAvgPool2d(out))
            flat_dim = ch * out[0] * out[1]
        elif kind == "flatten":
            modules.append(nn.Flatten())
        elif kind == "dropout":
            modules.append(nn.Dropout(float(spec.get("p", 0.5))))
        elif kind == "fc":
            out_f = num_classes if i == last_fc else int(spec["out_features"])
            in_f = int(spec.get("in_features") or flat_dim or 0)
            if in_f <= 0:
                raise ValueError("fc 之前需要有 adaptive_pool，或在 fc 里写明 in_features")
            modules.append(nn.Linear(in_f, out_f))
            flat_dim = out_f
        else:
            raise ValueError("未知层类型: %s" % kind)
    return nn.Sequential(*modules)


def _load_external(arch, num_classes):
    """从外部 .py 按类名加载模型（优先按 num_classes 构造，失败则无参构造）。"""
    path = Path(arch["module"])
    if not path.is_absolute():
        path = Path(arch["_base_dir"]) / path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = getattr(module, arch["class"])
    try:
        return cls(num_classes=num_classes)
    except TypeError:
        return cls()


def build_model(arch, num_classes):
    """按架构描述构建模型；type=module 时走外部模型文件。"""
    if str(arch.get("type", "")).lower() == "module":
        return _load_external(arch, num_classes)
    return _build_sequential(arch, num_classes)


def count_params(model):
    """可训练参数总量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
