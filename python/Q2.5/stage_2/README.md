# Q2.5 · stage_2：信号 / 噪声 / 交互效应探针

对预处理算子做“三探针”分解，量化它在低信噪比下的**特征损失**与**抑噪收益**。
探针**不绑定具体算法**：方法全部来自 `common/processors.py` 的插件列表。

## 1. 指标定义

预处理算子作用于 **STFT 幅度谱**，记：

```
X_x = |STFT(x)|      X_n = |STFT(n)|      X_xn = |STFT(x+n)|
```

四项指标（Frobenius 范数，除以 `||X_x||_F²`）：

```
total          = ||M(X_xn) - X_x||_F² / ||X_x||_F²
signal_loss    = ||M(X_x)  - X_x||_F² / ||X_x||_F²
noise_residual = ||M(X_n)       ||_F² / ||X_x||_F²
interaction    = ||M(X_xn) - M(X_x) - M(X_n)||_F² / ||X_x||_F²
```

恒等关系：

```
M(X_xn) - X_x = (M(X_x) - X_x) + M(X_n) + I
```

即“总失真”由**信号畸变 + 残留噪声 + 交互效应**共同构成（另含少量交叉项）。

- 全部以 `||X_x||_F²` 归一化，因此可加、可跨信号比较；
- 交互项反映算子的**非线性**：中值滤波是非线性的，`I ≠ 0` 正是其特性；
- 注意 `X_xn ≠ X_x + X_n`（取幅度这一步是非线性的），交互项里也包含这部分差异。

## 2. 运行

```bash
# 单条示例（config 里 probe_files 指定）
python python/Q2.5/stage_2/probes.py

# 指定输出文件名
python python/Q2.5/stage_2/probes.py --out-xlsx result/Q2.5/my_probe.xlsx

# 全量并行（先把 config 的 probe_files 改成 "all"）
python python/Q2.5/stage_2/probes.py --workers 8
```

## 3. 输出（xlsx，三个 sheet）

| sheet | 内容 |
|---|---|
| `说明` | 指标定义、公式、归一化方式、方法配置、种子派生规则、样本数 |
| `明细` | 每条样本 × 每个方法参数 × 每个 SNR 一行：`file, method, param, snr_db, total, signal_loss, noise_residual, interaction, norm_x` |
| `汇总` | 按 `(method, param, snr_db)` 聚合的均值：`n, total_mean, signal_loss_mean, noise_residual_mean, interaction_mean` |

表中 `method=none, param=0` 是**无预处理基线**（恒等算子）：其 `total` 即加噪谱与干净谱的偏差，
`signal_loss` 恒为 0，可作为其他方法的对照起点。

`method` / `param` 两列是通用字段，当前注册了两种方法：

| `method` | `param` 含义 | 取值 |
|---|---|---|
| `median_freq` | 频率方向中值窗口 | 3 / 5 / 9 / 17 |
| `wiener` | 维纳过减系数 `alpha` | 1.0 / 1.5 / 2.0 / 3.0（Excel 里 1.0 显示为 1） |

`param` 按原值写出、**不取整**，否则 `alpha=1.0` 与 `1.5` 会被合并成同一组。
新增算法只要在 `common/processors.py` 注册，**表结构不需要改**。

## 4. 参数（`config.json`）

| 字段 | 含义 | 默认 |
|---|---|---|
| `probe_files` | 样本列表；字符串 `"all"` 表示按 split/scene 全量 | `["A1_IN_S0_slice_1.mat"]` |
| `probe_split` | 全量时使用的 split，同时决定种子 | `train` |
| `probe_scene` | 全量时的场景筛选，`null` 不限 | `IN_S0` |
| `out_xlsx` | 结果路径 | `result/Q2.5/probe_metrics.xlsx` |
| `cache_f_x` | 是否跨 SNR 缓存方法在干净谱上的输出（快，但每个方法多占约 61 MB） | `false` |
| `workers` | 并行进程数 | `1` |

方法配置、SNR 档位、随机种子根从 `stage_1/config.json` 读取，保证探针与出图完全一致。

## 5. 可复现性

```
SeedSequence([base_seed, split_id, snr_idx, SAMPLE_SALT, sample_idx])
```

- `sample_idx` 是样本在**排序后列表中的下标**，因此顺序固定、可复现；
- 同一 `sample_idx × snr_idx` 下，不同方法共用**同一条噪声**，保证方法间对比公平；
- 该随机流与 Q2 批量加噪、Q2 对比图互相独立。

## 6. 扩展新方法

在 `common/processors.py` 的 `make_processors` 里追加一条：

```python
procs.append({
    "name": "median_freq_5",
    "domain": "stft",              # 作用在 STFT 幅度谱上
    "param": 5,
    "label": "median_freq_5",
    "cache_f_x": True,             # 重算贵的算子才标 True，见下文 cache_f_x
    "fn": lambda mag, _w=5: median_filter_freq(mag, _w),
})
```

若新方法作用在时域 IQ 上，设 `domain="time"`、`fn(iq) -> iq`，探针会改用
时域公式计算。本脚本无需改动，出图与探针都会自动包含新方法。

## 7. 全量耗时参考

单条样本是 400 万点的全长 STFT（约 6.25 万帧 × 256 频点），整幅幅度谱一份约 61 MB，
所以内存是主要约束：每个进程峰值约 0.5 GB，`cache_f_x=true` 时约 0.9 GB。

| 范围 | 耗时 | 说明 |
|---|---|---|
| 1 条示例（中值 4 档 + 维纳 4 档 × 5 SNR） | 约 150 秒 | `cache_f_x=false`（默认，省内存） |
| 同上 | 约 90 秒 | `cache_f_x=true`（内存宽裕时） |
| 全量 train（1056 条） | 约 40 小时 | 按 150 秒/条估算 |
| 全量 train（`--workers 8`） | 约 5 小时 | 需留出约 4 GB 可用内存 |
