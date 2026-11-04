# Q2 · stage_1：IQ 信号 STFT 时频图

从训练集每个型号各取 1 条 IQ 记录（统一 `IN_S0` 视距场景，`background` 单独 1 条），
做短时傅里叶变换并画成一张多子图时频图，用来比较不同**帧长**下时间分辨率与频率分辨率的取舍。

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
# 用 config.json 里的帧长（默认 256）
python python/Q2/stage_1/stft_explore.py

# 临时试别的帧长
python python/Q2/stage_1/stft_explore.py --frame-len 512

# 放大到 ±15 MHz 看占用带细节，并弹窗显示
python python/Q2/stage_1/stft_explore.py --frame-len 1024 --freq-limit 15 --show
```

## 3. 参数

| 命令行参数 | 对应 `config.json` 字段 | 含义 | 默认 |
|---|---|---|---|
| `--frame-len` | `frame_len` | 帧长（采样点数） | 256 |
| `--hop-ratio` | `hop_ratio` | 帧移 = 帧长 × 该比例 | 0.25 |
| `--freq-limit` | `freq_limit_mhz` | 频率显示上限（MHz），40 = 全带 | 40 |
| `--show` | — | 出图后弹窗显示 | 关闭 |

`config.json` 另有：`data_dir`（数据目录）、`glob_pattern`、`fs_hz`（采样率）、
`scene`（场景筛选，`IN_S0`）、`background_prefix`、`show_ms`（参与 STFT 的时长）、
`n_fft`（`null` 表示等于帧长）、`window`（窗函数）。

绘画样式（colormap、色标下限、子图尺寸）在项目根 `config_pic.json` 的 `stft` 段。

## 4. 输出

| 位置 | 内容 |
|---|---|
| `image/Q2/stft_frame_XXXX.png` | 帧长为 XXXX 的时频图（7 类信号并排，共享色标） |
| `log/Q2/stft_explore.log` | 运行日志：入选文件清单、帧长对照表、出图路径 |
