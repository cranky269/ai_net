# Q2.5 文件架构说明

本文件只记录“每个文件的作用与含义”，不记录任何运行得到的结论。

> Q2.5 为独立阶段：预处理算法（中值滤波）与探针诊断。代码按阶段放 `python/Q2.5/stage_x`，
> 跨阶段方法库放 `python/Q2.5/common`。

## python/Q2.5 —— 预处理算法与探针

### common —— 跨阶段通用

| 文件 | 作用 |
|---|---|
| `q25_common.py` | 路径常量（项目根、Q2.5 各产物目录、Q2 配置路径）、JSON 读取、日志初始化、`apply_pic_style`、`stft_mag`/`mag_to_db`、`resolve_show_len`，以及复用 Q2 加噪组件的薄封装 `import_q2_awgn`；并定义出图与探针共用的随机流标记 `SAMPLE_SALT` |
| `processors.py` | 预处理方法插件库：统一接口 `fn(mag) -> mag`（作用在 STFT 幅度谱）；实现 `median_filter_freq`（沿频率方向的一维中值）与 `make_processors`（按配置展开方法列表，含 `domain` 字段标明作用域），新增方法只需在此注册 |

### stage_1 —— 中值滤波与 STFT 对比图

| 文件 | 作用 |
|---|---|
| `config.json` | 方法参数（`median_windows`/`median_mode`）、对比图样本与所属 split、SNR 档位、随机种子根、输出图路径 |
| `plot_filter_compare.py` | 读样本 IQ → 按 SNR 预生成噪声 → STFT 得幅度谱 → 沿频率中值滤波 → 行=滤波方式、列=SNR 的网格图（每列以该列无滤波峰值为 dB 参考） |
| `README.md` | 算法、运行方式、参数表、与 stage_2 的种子一致性说明 |

### stage_2 —— 三探针

| 文件 | 作用 |
|---|---|
| `config.json` | 探针样本（列表或 `"all"`）、split/scene 筛选、结果 xlsx 路径、并行进程数 |
| `probes.py` | 计算 total / signal_loss / noise_residual / interaction 四项指标；支持单进程与 `--workers` 多进程；写出含“说明/明细/汇总”三个 sheet 的 xlsx |
| `README.md` | 指标公式、xlsx 结构、扩展新方法的方法、全量耗时参考 |

### docs

| 文件 | 作用 |
|---|---|
| `file_tree.md` | 本文件：Q2.5 各目录与文件的含义说明 |

## image/Q2.5 —— 结果图

| 文件 | 作用 |
|---|---|
| `median_compare.png` | 中值滤波对比图：行=无滤波+各窗口，列=各档 SNR，每列以该列无滤波峰值为 dB 参考 |

## result/Q2.5 —— 结果数据

| 文件 | 作用 |
|---|---|
| `probe_metrics.xlsx` | 探针结果：“说明”表记录指标定义与参数，“明细”表每条样本×方法参数×SNR 一行，“汇总”表按组合取均值 |

## log/Q2.5 —— 运行日志

| 文件 | 作用 |
|---|---|
| `stage1_median_compare.log` | `plot_filter_compare.py` 运行日志：样本、功率、无噪峰值参考、出图路径 |
| `stage2_probes.log` | `probes.py` 运行日志：样本数、SNR、方法列表、进度、输出路径 |

## 项目根

| 文件 | 作用 |
|---|---|
| `config_pic.json` | 全项目统一绘图配置；Q2.5 使用 `stft` 段（colormap/色标下限）与 `q25` 段（滤波对比图尺寸） |
