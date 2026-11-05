# Q3 · stage_2：CNN 训练与评估

输入 stage_1 缓存的单通道灰度时频图，输出 7 类（A/C/D/E/F/G + BG）。

## 1. 运行

```bash
python python/Q3/stage_2/train.py                        # 按 config.json
python python/Q3/stage_2/train.py --epochs 2             # 小跑验证流程
python python/Q3/stage_2/train.py --weights result/Q3/weights/cnn4_best.pt
python python/Q3/stage_2/train.py --arch python/Q3/stage_2/arch/cnn4.yaml --tag exp1
```

| 命令行参数 | 含义 |
|---|---|
| `--arch` | 覆盖 config 的架构 YAML |
| `--weights` | 加载 `.pt` 权重继续训练 |
| `--epochs` / `--batch-size` / `--lr` | 覆盖训练超参 |
| `--limit-batches` | 调试：每个 epoch 只跑前 N 个 batch |
| `--tag` | 权重文件名后缀，便于区分实验 |

## 2. 训练参数（`config.json`）

| 参数 | 含义 | 默认 |
|---|---|---|
| `arch` | 架构 YAML 路径 | `arch/cnn4.yaml` |
| `weights` | 预训练权重，null = 从头训练 | null |
| `batch_size` | 批大小 | 8 |
| `epochs` | 训练轮数 | 10 |
| `lr` / `weight_decay` | Adam 学习率 / 权重衰减 | 1e-3 / 1e-4 |
| `val_ratio` | 从 train 按原始文件切出的验证集比例 | 0.1 |
| `dtype` / `db_floor` | 必须与 stage_1 一致，决定输入归一化 | uint8 / −80 |
| `device` | 计算设备 | cpu |
| `seeds.split` | 切分训练/验证集的随机种子 | 42 |
| `seeds.init` | 模型参数初始化的随机种子 | 2024 |
| `seeds.loader` | 训练集打乱（DataLoader）的随机种子 | 7 |

三个种子互相独立：只改其中一个，其余过程的结果不变，便于做单因素对比实验。

## 3. 架构文件写法（`arch/*.yaml`）

**写法一：逐层描述（默认）**

```yaml
name: cnn4
input_channels: 1
layers:
  - {type: conv, out_channels: 16, kernel_size: 5, stride: 2, padding: 2, norm: batch, act: relu, pool: 2}
  - {type: conv, out_channels: 32, kernel_size: 3, stride: 2, padding: 1, norm: batch, act: relu, pool: 2}
  - {type: conv, out_channels: 64, kernel_size: 3, stride: 2, padding: 1, norm: batch, act: relu, pool: 2}
  - {type: conv, out_channels: 64, kernel_size: 3, stride: 1, padding: 1, norm: batch, act: relu, pool: 2}
  - {type: adaptive_pool, output: [4, 8]}
  - {type: flatten}
  - {type: dropout, p: 0.3}
  - {type: fc, out_features: 7}
```

支持的层类型：`conv`（含 `norm`/`act`/`pool`）、`pool`、`adaptive_pool`、`flatten`、`dropout`、`fc`。
`act` 可选 `relu/relu6/gelu/leaky_relu/elu/silu`，`norm` 可选 `batch/instance/none`。
最后一个 `fc` 的输出维度会被自动改成实际类别数；输入尺寸由数据参数推导，不写死。

**写法二：外部模型文件**

```yaml
type: module
module: my_net.py     # 相对本 YAML 所在目录
class: MyNet
```

`MyNet` 优先按 `MyNet(num_classes=7)` 构造，不接受该参数时退回无参构造。

## 4. 数据切分

- train / test 直接取自 Q1 的 `dataset_split.xlsx`（按原始记录分组，不重叠）
- 验证集从 train 里按**原始文件**再切（默认 10%），保证同一片段切出的两张图落在同一侧
- 训练监控用验证集；`val_ratio=0` 时退化为用 test 监控（不推荐）
- test 只在最后用「验证集最优权重」评估一次

## 5. 输出

| 位置 | 内容 |
|---|---|
| `result/Q3/weights/<arch>_best.pt` / `_last.pt` | 权重 + `arch`/`classes`/`epoch`/`eval_f1` |
| `result/Q3/train_metrics.xlsx` | “说明”“逐轮”“测试总览”“每类指标”四个工作表 |
| `image/Q3/loss_curve.png` | 训练 loss 与监控集合 acc / Macro-F1 曲线 |
| `image/Q3/confusion_matrix.png` | 测试集混淆矩阵 |
| `log/Q3/train.log` | 训练日志 |
