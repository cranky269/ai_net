# Q2 · stage_1：IQ 信号 STFT 时频图

从训练集每个型号各取 1 条 IQ 记录（统一 `IN_S0` 视距场景，`background` 单独 1 条），
做短时傅里叶变换并画成一张多子图时频图，用来比较不同**帧长**下时间分辨率与频率分辨率的取舍。

频率轴为**绝对频率刻度**：基带 −40~+40 MHz 平移 `center_freq_hz`（2440 MHz）后显示为 2400~2480 MHz，
对应数据集的 2.4~2.48 GHz 采集范围。

## 1. 运行环境

实测环境：conda `Lpytorch`（**Python 3.8.20**）

| 库 | 版本 |
|---|---|
| numpy | 1.24.4 |
| scipy | 1.10.1 |
| matplotlib | 3.7.2 |
| h5py | 3.11.0 |

> 该环境的 pip 连不上 PyPI（SSL 报错），`h5py` 可用 cp38 wheel 离线安装：
> `pip install h5py-3.11.0-cp38-cp38-win_amd64.whl --no-index`

## 2. 运行

```bash
# 单帧长：用 config.json 里的值（默认 256）
python python/Q2/stage_1/stft_explore.py

# 单帧长：临时覆盖
python python/Q2/stage_1/stft_explore.py --frame-len 512

# 遍历 FRAME_LEN_CANDIDATES 里的全部帧长各出一张图 + 一张帧长对比图
python python/Q2/stage_1/stft_explore.py --sweep

# 指定对比图用哪个型号（默认 C）
python python/Q2/stage_1/stft_explore.py --sweep --compare-signal D
```

## 3. 参数

| 命令行参数 | 对应 `config.json` 字段 | 含义 | 默认 |
|---|---|---|---|
| `--frame-len` | `frame_len` | 帧长（采样点数） | 256 |
| `--hop-ratio` | `hop_ratio` | 帧移 = 帧长 × 该比例 | 0.25 |
| `--freq-limit` | `freq_limit_mhz` | 相对中心频率的显示范围 ±MHz，40 = 全带 | 40 |
| `--sweep` | — | 遍历 `FRAME_LEN_CANDIDATES` 全部帧长 | 关闭 |
| `--compare-signal` | — | 对比图用哪个型号 | C |
| `--show` | — | 出图后弹窗显示 | 关闭 |

`config.json` 另有：`data_dir`（数据目录）、`glob_pattern`、`fs_hz`（采样率）、
`center_freq_hz`（射频中心频率，2440 MHz）、`scene`（场景筛选，`IN_S0`）、`background_prefix`、
`show_ms`（参与 STFT 的时长）、`n_fft`（`null` 表示等于帧长）、`window`（窗函数）。

帧长遍历列表在脚本开头的 `FRAME_LEN_CANDIDATES` 里改。

绘图样式（colormap、色标下限、子图尺寸）在项目根 `config_pic.json` 的 `stft` 段。

## 4. 输出

| 位置 | 内容 |
|---|---|
| `image/Q2/stft_frame_XXXX.png` | 帧长为 XXXX 的时频图（7 类信号并排，共享色标，纵轴 2400~2480 MHz） |
| `image/Q2/stft_sweep_compare.png` | `--sweep` 时生成：同一信号在多个帧长下并排对比 |
| `log/Q2/stft_explore.log` | 运行日志：入选文件清单、帧长对照表、出图路径 |

## 5. 原版尺寸出图（`stft_native_size.py`）

只对**单个**信号出图，数据区严格 **1 帧 = 1 像素、1 个频点 = 1 像素**（不做任何降采样），
用来查看真实像素数与 PNG 体积。坐标轴、刻度、色标照常保留，额外占像素。

```bash
python python/Q2/stage_1/stft_native_size.py                  # 5 ms / 帧长 2048
python python/Q2/stage_1/stft_native_size.py --show-ms full   # 整段 50 ms
python python/Q2/stage_1/stft_native_size.py --show-ms 10 --frame-len 512 --signal C
```

| 参数 | 含义 | 默认 |
|---|---|---|
| `--signal` | 型号（A/C/D/E/F/G/BG） | A |
| `--frame-len` | 帧长（采样点数） | 2048 |
| `--hop-ratio` | 帧移 / 帧长 | 0.25 |
| `--show-ms` | 时长：`config` / 毫秒数 / `full`（整段） | config |
| `--dpi` | 像素换算基准，与 figsize 联动 | 100 |

输出：`image/Q2/stft_native_f<帧长>_<时长>ms.png`，同时打印数据区像素、整图像素、文件体积。
