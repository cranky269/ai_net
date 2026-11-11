# Q2.5 · stage_1：预处理方法 + STFT 对比图

对加噪后的复 IQ 做预处理，并画出「无滤波 vs 各参数档」在「各档 SNR」下的时频图对比。
四种方法各出一张图，版式、配色、色标完全一致，同一列共用同一段噪声。

当前注册的方法：

| 方法 | `name` | `param` | 原理一句话 |
|---|---|---|---|
| 频率方向中值滤波 | `median_freq` | 窗口（频点数） | 固定时刻，沿频率取邻域中值 |
| 幅度域维纳滤波 | `wiener` | `alpha` | 谱减式增益，噪声由低分位自估计 |
| dB 域线性拉伸 | `db_lin` | 阈值（dB） | 阈值钳位后把 dB 区间线性铺满，等价 γ 律压缩 |
| 幅度域线性拉伸 | `amp_lin` | 阈值（dB） | 阈值钳位后把幅度区间仿射拉伸 |

前两种是**抑噪**算子，后两种是**动态范围整形**算子（不抑噪，见 1.3 / 1.4 的说明）。

## 1. 算法

四种方法都作用在 **STFT 幅度谱** `X[f, t] = |STFT(x)|` 上。

### 1.1 频率方向中值滤波

```
Y[f, t] = med{ X[f-k, t], ..., X[f+k, t] },   N = 2k+1
```

- **只在频率轴滤波，时间轴完全不动**；
- 为什么不在时域做：80 MHz 采样下相邻时域样本几乎去相关，时域中值会把窄带信号
  当成孤立尖峰抹掉；而 STFT 后信号能量在频域集中成窄带峰，沿频率取中值能保留峰、
  压掉散乱的噪底；
- 边界延拓默认 `reflect`（`config.json` 的 `median_mode`）；
- 实现用 `scipy.ndimage.median_filter`，`size=(1, N)` 只在最后一维（频率）滑动。

窗口以**频点数**计（当前 STFT 每频点 312.5 kHz）：

| 窗口 | 覆盖频宽 | 占 80 MHz |
|---|---|---|
| 3 | 0.94 MHz | 1.2% |
| 5 | 1.56 MHz | 2.0% |
| 9 | 2.81 MHz | 3.5% |
| 17 | 5.31 MHz | 6.6% |

### 1.2 幅度域维纳滤波

```
H[f, t] = max(beta, 1 - alpha * Snn[f, t] / Syy[f, t])
Y        = H .* X                # 相位不变
```

噪声功率 `Snn` 由含噪谱自身估计（三步）：

1. 沿时间一阶递归平滑 `Syy`（系数 `wiener_smooth`）；
2. 每 `step` 帧取长 `wiener_win` 的块，块内取 `wiener_q`（10%）低分位；
3. 低分位经 `wiener_bias = 9.49` 还原为噪声**平均**功率（指数分布 10% 分位 ≈ 均值 × 0.1054），
   再沿时间线性插值回逐帧。

- `alpha = 1` 即标准维纳；`alpha` 越大压制越强（`wiener_alphas`）；
- `beta` 是增益下限，避免过度衰减到 0；
- 低分位用“块采样 + 插值”而非逐帧滑窗：6 万帧上从约 80 s 降到约 1 s，
  中位相对误差 <1%（详见 `common/processors.py` 的 `estimate_noise_power`）。

### 1.3 dB 域线性拉伸（`db_lin`）

两阶段，全部在 dB 域：

```
M   = max(|X|)                                   # 该图自身峰值当作 0 dB
db  = 20·log10(|X|/M)
db1 = max(db, th)                                # 阶段一：低于阈值钳到阈值
db2 = db1 · floor/th                             # 阶段二：[th,0] 线性铺满 [floor,0]
|Y| = M · 10^(db2/20)
```

`floor` 由 `stretch_floor_db`（默认 -80，与 `config_pic.json` 的色标下限一致）给出。
由于两步都在 dB 域，整体等价于**幅度谱的 γ 律压缩**：

```
|Y| = M · (|X|/M)^(|floor/th|)
```

`th = -20` 时指数为 4，所以 -10 dB 的点会被映射到 **-40 dB**——所有层级等间距拉开。

### 1.4 幅度域线性拉伸（`amp_lin`）

阶段一相同（钳位），阶段二改到**线性幅度**上做仿射变换：

```
lo   = M · 10^(th/20)                            # 钳位后的最小幅度
Ymin = M · 10^(floor/20)                         # 拉伸目标下限（-80 dB → 1e-4·M）
|Y|  = Ymin + (max(|X|, lo) - lo) · (M - Ymin)/(M - lo)
```

与 1.3 的区别在第二步：这是仿射映射而不是幂律，因此**接近峰值的成分几乎不变、
越弱压得越狠**。同样 `th = -20` 时：

| 输入电平 | `db_lin` 输出 | `amp_lin` 输出 |
|---|---|---|
| 0 dB（峰值） | 0 dB | 0 dB |
| -10 dB | -40 dB | **-12.4 dB** |
| -20 dB | -80 dB | -80 dB |
| -30 dB | -80 dB | -80 dB |

四档阈值对应的幂次（仅 `db_lin` 有意义）：`-10 → γ=8`、`-20 → γ=4`、`-30 → γ=2.67`、`-40 → γ=2`。

> **注意**：1.3 / 1.4 不是抑噪算子。它们把低于阈值的成分压到 -80 dB 地板，
> 所以探针里的 `noise_residual` 会明显下降，但 `signal_loss` 同步上升
> （弱信号和噪声一起被压掉）。与中值/维纳的 `total` 相比时要注意这个差别。

## 2. 运行

```bash
python python/Q2.5/stage_1/plot_filter_compare.py             # 四张图都出
python python/Q2.5/stage_1/plot_filter_compare.py --only wiener
python python/Q2.5/stage_1/plot_filter_compare.py --only db_lin
python python/Q2.5/stage_1/plot_filter_compare.py --show      # 出图后弹窗
```

## 3. 输出

| 位置 | 内容 |
|---|---|
| `image/Q2.5/median_compare.png` | 行 = 无滤波 + 各中值窗口，列 = 各档 SNR |
| `image/Q2.5/wiener_compare.png` | 行 = 无滤波 + 各 `alpha`，列 = 各档 SNR |
| `image/Q2.5/db_lin_compare.png` | 行 = 无滤波 + 各阈值，列 = 各档 SNR |
| `image/Q2.5/amp_lin_compare.png` | 行 = 无滤波 + 各阈值，列 = 各档 SNR |
| `log/Q2.5/stage1_filter_compare.log` | 样本、功率、各列 dB 参考、各图输出路径 |

每张图都**以该列最上方（no filter）的 STFT 峰值为 0 dB 参考**，且同一列共用同一段噪声，
因此既能在列内看各参数衰减了多少，也能在不同方法的同一列之间直接对照。

## 4. 参数（`config.json`）

| 字段 | 含义 | 默认 |
|---|---|---|
| `methods.median_windows` | 中值窗口列表（奇数频点数） | `[3, 5, 9, 17]` |
| `methods.median_mode` | 中值边界延拓方式 | `reflect` |
| `methods.wiener_alphas` | 维纳过减系数档位 | `[1.0, 1.5, 2.0, 3.0]` |
| `methods.wiener_win` | 噪声分位估计窗口（帧） | `500` |
| `methods.wiener_q` | 噪声底分位数 | `0.1` |
| `methods.wiener_smooth` | 递归平滑系数 | `0.7` |
| `methods.wiener_bias` | 低分位 → 噪声均值的偏差补偿 | `9.49` |
| `methods.wiener_beta` | 增益下限 | `0.05` |
| `methods.wiener_step` | 分位块步长（帧）；`null` 表示按比例推导 | `null` |
| `methods.wiener_step_ratio` | `wiener_step` 为 null 时生效 | `0.125` |
| `methods.db_lin_thresholds` | dB 域线性拉伸的阈值档位（dB） | `[-10, -20, -30, -40]` |
| `methods.amp_lin_thresholds` | 幅度域线性拉伸的阈值档位（dB） | `[-10, -20, -30, -40]` |
| `methods.stretch_floor_db` | 两种拉伸的目标下限（dB） | `-80.0` |
| `plots[]` | 每张图一组：`name` / `methods`（注册名）/ `title` / `out_png` | 四组 |
| `plot_max_frames` | 出图前沿时间轴最大池化的帧数上限（`null`/`0` 为不池化） | `4096` |
| `plot_sample` | 对比图所用样本 | `A1_IN_S0_slice_1.mat` |
| `plot_split` | 样本所属集合（决定种子派生） | `train` |
| `snr_db_list` | SNR 档位 | `[-10, -5, 0, 5, 10]` |
| `base_seed` | 随机种子根 | `2024` |

`plots[].methods` 的取值是 `common/processors.py` 的注册名，新增算法只需在
`make_processors` 注册，再往某个 `plots[]` 分组里加上它的名字。

`plot_max_frames` 只影响**出图**（帧数远超图像像素，逐帧保留会白占 GB 级内存），
池化取每段最大值以保住突发与窄带峰，对探针指标无影响。

STFT 参数（采样率、中心频率、帧长、帧移、窗、FFT 点数、频率范围、时长）全部取
`python/Q2/stage_1/config.json`；配色尺寸取项目根 `config_pic.json` 的 `stft` / `q25` 段。

## 5. 与 stage_2 的一致性

- 噪声种子：`SeedSequence([base_seed, split_id, snr_idx, SAMPLE_SALT, sample_idx])`；
- stage_2 探针使用**同一套种子**，且读同一份 `methods`，因此图里看到的加噪信号，
  就是探针表格计算所依据的那一条；
- 该随机流与 Q2 批量加噪、Q2 对比图三者互相独立。
