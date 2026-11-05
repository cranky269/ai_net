# Q3 文件架构说明

本文件只记录“每个文件的作用与含义”，不记录任何运行得到的结论。

> Q3 为当前题目的第三阶段：STFT 灰度图 + CNN 型号识别。代码按阶段放 `python/Q3/stage_x`。

## python/Q3 —— 建模代码

### common —— 跨阶段通用

| 文件 | 作用 |
|---|---|
| `__init__.py` | 包标记 |
| `q3_common.py` | 路径常量（含 `data` junction、各产物目录）、JSON 读取、日志初始化、Q1 划分表读取、类别顺序、dB→指定精度量化；并以薄封装复用 Q2 的 `load_iq` / `stft_db` |

### stage_1 —— STFT 灰度图缓存构建

| 文件 | 作用 |
|---|---|
| `config.json` | 数据集可调参数：`hop_ratio`、`frames_per_image`、`dtype`、`db_floor`、`bg_resample`、`max_per_class`、`splits`；以及数据/缓存/索引/日志路径。帧长 2048、截取 50 ms 为不可调常量，写在 `common/q3_common.py` |
| `build_cache.py` | 读 Q1 划分 → 每类限量选片段 → 每片段算一次 STFT → 按帧数切图（余数丢弃）→ 量化写入 `X_<split>.npy` / `y_<split>.npy` → 写索引表 |
| `README.md` | 参数表、切片规则、体积参考、用法 |

### stage_2 —— CNN 训练与评估

| 文件 | 作用 |
|---|---|
| `config.json` | 训练参数：架构路径、预训练权重、索引表、批大小、轮数、学习率、两路独立随机种子（`seeds.init` / `seeds.loader`）、设备，以及权重/指标/图/日志输出路径 |
| `arch/cnn4.yaml` | 默认 4 层卷积架构；可改层参数，也可改成 `type: module` 指向自定义模型文件 |
| `model_zoo.py` | 把 YAML 翻译成 `nn.Module`：逐层构建（conv/pool/adaptive_pool/flatten/dropout/fc）或加载外部模型类；含参数量统计 |
| `dataset.py` | `StftCacheDataset`：memmap 按索引读图、按缓存精度归一化到 0~1 |
| `train.py` | 训练主程序：加载索引与架构 → 逐轮训练 → 在监控集合（优先 val，否则 test）上打分 → 保存 best/last 权重 → 用 best 在 test 上评估 → 写指标 xlsx 与曲线/混淆矩阵 |
| `README.md` | 用法、参数、架构 YAML 写法、切分与输出说明 |

### docs

| 文件 | 作用 |
|---|---|
| `file_tree.md` | 本文件：Q3 各目录与文件的含义说明 |

## 数据与产物

| 位置 | 作用 |
|---|---|
| `data/_q3_cache/X_train.npy`、`X_test.npy` | STFT 灰度图缓存，形状 `(N, frames_per_image, 2048)`；`data` 是指向 D 盘的 junction |
| `data/_q3_cache/y_train.npy`、`y_test.npy` | 对应的类别下标 |
| `result/Q3/cache_index.xlsx` | 缓存索引：“说明”表记录字段与参数含义，“明细”表每张图一行 |
| `result/Q3/train_metrics.xlsx` | 训练指标：“说明”“逐轮”“测试总览”“每类指标”四个工作表 |
| `result/Q3/weights/*.pt` | 权重（best/last），内含 `state_dict`、`arch`、`classes`、`epoch`、`eval_f1` |
| `image/Q3/loss_curve.png` | 训练 loss 与监控集合 acc / Macro-F1 曲线 |
| `image/Q3/confusion_matrix.png` | 测试集混淆矩阵 |
| `log/Q3/build_cache.log`、`train.log` | 缓存构建与训练日志 |
