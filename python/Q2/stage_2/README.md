# Q2 · stage_2：IQ 信号加高斯白噪声（内存迭代）+ 加噪前后 STFT 对比图

在划分好的数据集上按目标信噪比注入复高斯白噪声。**只在内存中加噪、不落盘**，
由后续流程（STFT / CNN）直接消费；可复现性由“固定种子 + 固定顺序 + 每条单独算 σ”保证。

## 1. 运行环境

与 stage_1 相同：conda `Lpytorch`（Python 3.8.20）

| 库 | 版本 |
|---|---|
| numpy | 1.24.4 |
| matplotlib | 3.7.2 |
| h5py | 3.11.0 |
| openpyxl | 读划分表用（Q1 环境已装） |

## 2. 运行

```bash
# 生成加噪前后 STFT 对比图（3 型号 × 无噪 + 5 档 SNR）
python python/Q2/stage_2/plot_noise_compare.py

# 出图后弹窗
python python/Q2/stage_2/plot_noise_compare.py --show
```


```
# 应用

from awgn import load_json, load_split_table, iter_split

cfg = load_json("python/Q2/stage_2/config.json")
table = load_split_table(cfg["split_table"], cfg["split_sheet"])

for path, iq_noisy, meta in iter_split(table, cfg, "train", -5.0):
    # iq_noisy: complex64；meta 含实测 SNR / sigma / 文件名 / 型号
    ...


```

## 3. 加噪公式

```
P_s   = mean(|x|^2)                       # 每条信号单独计算
sigma = sqrt(P_s / 10^(SNR_dB/10))        # 噪声总功率 = sigma^2
n     = sigma/sqrt(2) * (n_I + j*n_Q),   n_I, n_Q ~ N(0, 1)
y     = x + n
```

- 复数 IQ 加复高斯噪声：实部/虚部各承担一半功率，`E[|n|^2] = sigma^2`；
- 每条信号按自身功率反推 `sigma`，保证都落在目标 SNR 档位；
- 迭代器同时返回**实测 SNR**（`snr_db_measured`），可直接核对落档情况。

## 4. 可复现性约定

| 规则 | 实现 |
|---|---|
| 固定种子 | `SeedSequence([base_seed, split_id, snr_idx])`，每个 `(split, snr)` 一条独立流 |
| 固定顺序 | 按划分表行序 + 文件名排序遍历 |
| 逐条算方差 | 每条信号用自身 `P_s` 反推 `sigma` |
| 互不干扰 | train / test、各 SNR 档、对比图（额外 `PLOT_SALT`）使用彼此独立的随机流 |

> 对比图使用 `SeedSequence([base_seed, split_id, snr_idx, PLOT_SALT, model_idx])`，
> 与正式批量加噪的随机流隔离：反复调图不会改变正式实验数据。

## 5. 接口

```python
from awgn import load_json, load_split_table, iter_split

cfg = load_json("python/Q2/stage_2/config.json")
table = load_split_table(cfg["split_table"], cfg["split_sheet"])

for path, iq_noisy, meta in iter_split(table, cfg, "train", -5.0):
    # iq_noisy: complex64 加噪 IQ；meta: 文件名/型号/sigma/实测 SNR
    ...
```

| 函数 | 作用 |
|---|---|
| `load_split_table` | 读 Q1 划分表（xlsx） |
| `make_rng` | 由 `SeedSequence` 派生独立随机流 |
| `signal_power` / `sigma_for_snr` | 信号功率 / 由 SNR 反推 sigma |
| `add_awgn` | 单条加噪，返回 `(noisy, meta)` |
| `iter_split` | 按固定顺序批量迭代，逐条 yield |

> `iter_split` 是**流式迭代器**：请逐条消费（`for` 循环内处理完即丢弃），
> 不要 `list(iter_split(...))` 全量收集——单条 IQ 为 4M 点，上千条整体驻留会撑爆内存。

## 6. 配置

`config.json`（本阶段）：

| 字段 | 含义 | 默认 |
|---|---|---|
| `data_dir` | IQ 数据目录 | `data/train_extracted` |
| `split_table` / `split_sheet` | Q1 划分表与 sheet | `output/Q1/splits/dataset_split.xlsx` / `split` |
| `base_seed` | 随机种子根 | 2024 |
| `snr_db_list` | SNR 档位（≥4 档，可改任意数组） | `[-10, -5, 0, 5, 10]` |
| `bg_model` / `include_bg` | 背景类名 / 是否纳入迭代 | `BG` / `false` |
| `plot_models` | 对比图型号（每型号排序取第一条） | `["A","C","D"]` |
| `plot_split` / `plot_scene` | 对比图选样集合与场景 | `train` / `IN_S0` |
| `plot_show_ms` | 参与 STFT 的时长；`null` 沿用 stage_1 | `null` |
| `out_png` | 对比图输出路径 | `image/Q2/stage2/noise_compare_3x6.png` |

STFT 参数（采样率、中心频率、帧长 256、帧移 0.25、窗、FFT 点数、频率范围、时长）
全部读 `python/Q2/stage_1/config.json`；配色与尺寸读项目根 `config_pic.json` 的 `stft` / `stage2` 段。

## 7. 输出

| 位置 | 内容 |
|---|---|
| `image/Q2/stage2/noise_compare_3x6.png` | 3 行（A/C/D）× 6 列（无噪 + 5 档 SNR）的 STFT 对比图，全图共享同一 dB 参考（每行无噪峰值） |
| `log/Q2/stage2_awgn_compare.log` | 选样清单、每档实测 SNR 与 sigma、输出路径 |
