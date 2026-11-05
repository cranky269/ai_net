# Q2 文件架构说明

本文件只记录“每个文件的作用与含义”，不记录任何运行得到的结论。

> Q2 为当前题目的第二阶段：STFT 时频图生成（按《代码书写.md》第 10 条，代码按阶段放 `python/Q2/stage_x`）。

## python/Q2 —— 建模/信号处理代码

### stage_1 —— IQ 读取与 STFT 时频图

| 文件 | 作用 |
|---|---|
| `config.json` | 本阶段参数：数据目录与匹配规则、采样率、中心频率、场景筛选（`IN_S0`）、background 前缀、显示时长、帧长、帧移比例、FFT 点数、窗函数、频率显示上限、帧长对比图所用型号，以及 `native_size`（1:1 原版尺寸出图的信号/帧长/时长/dpi） |
| `stft_explore.py` | 主脚本：每型号取 1 条 IQ → 读取 I/Q 复信号 → 计算 STFT → 画多子图时频图并存图；帧长、帧移、频率范围均可用命令行覆盖；`--sweep` 遍历 `FRAME_LEN_CANDIDATES` 全部帧长并额外生成帧长对比图；频率轴按 `center_freq_hz`（2440 MHz）平移为 2400~2480 MHz 绝对刻度；同时打印各候选帧长的时间/频率分辨率对照表 |

| `stft_native_size.py` | 单信号 1:1 原版尺寸出图：数据区 1 帧 = 1 px、1 bin = 1 px，不做降采样；用于查看真实像素数与 PNG 体积，坐标轴与色标照常保留 |

### docs

| 文件 | 作用 |
|---|---|
| `file_tree.md` | 本文件：Q2 各目录与文件的含义说明 |

## image/Q2 —— 结果图

| 文件 | 作用 |
|---|---|
| `stft_frame_XXXX.png` | 帧长为 XXXX 点时，7 类信号（A/C/D/E/F/G 型号 + background）并排的 STFT 时频图，纵轴为 2400~2480 MHz 绝对频率；配色与尺寸取自项目根 `config_pic.json` |
| `stft_sweep_compare.png` | `--sweep` 生成：同一型号信号在 `FRAME_LEN_CANDIDATES` 各帧长下并排，用于选帧长 |
| `stft_native_f<帧长>_<时长>ms.png` | `stft_native_size.py` 输出：单信号 1:1 原版尺寸时频图（数据区 1 帧 = 1 px、1 bin = 1 px） |

## log/Q2 —— 运行日志

| 文件 | 作用 |
|---|---|
| `stft_explore.log` | `stft_explore.py` 的运行日志（UTF-8）：入选文件清单、帧长对照表、出图路径 |

## 项目根

| 文件 | 作用 |
|---|---|
| `config_pic.json` | 全项目统一绘图配置；本阶段新增 `stft` 段（时频图 colormap、色标下限、子图网格尺寸、色标宽度占比） |
