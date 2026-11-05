# Q0 文件架构说明

本文件只记录“每个文件的作用与含义”，不记录任何运行得到的结论。

> Q0 为当前题目（低信噪比无人机射频信号预处理与智能识别）。
> 按《代码书写.md》第 26~28 条（预处理阶段）：代码放 `script/Qx`，结果放 `output/Qx`；
> 按第 10 条，代码按阶段组织为 `Qx/stage_x`。

## script/Q0 —— 预处理阶段代码（分阶段）

### stage_1 —— 数据探针（下载最小样本，确认数据格式）

| 文件 | 作用 |
|---|---|
| `config.json` | 本阶段可调参数：数据根目录、分卷匹配规则、提取数量、采样率、SNR 估计参数、输出与日志路径 |
| `zip_carver.py` | 分卷 ZIP 局部提取器：顺序扫描本地文件头，从单个 `.z01` 卷中解出完整数据文件，不依赖中央目录 |
| `mat_probe.py` | `.mat` 探针：读取 I/Q 通道，统计 dtype/采样点数/时长/功率，用平均周期图粗估带内 SNR |
| `run_pipeline.py` | 流水线入口：定位分卷 → 提取样本 → 逐个探测 → 汇总写出 xlsx → 打印摘要 |
| `selftest_carver.py` | 最小单元自测：构造仿分卷（含跨卷截断条目），验证提取与探针逻辑，不依赖真实数据集 |
| `README.md` | 运行说明：环境准备、运行命令、参数含义、输出位置 |
| `.gitignore` | 忽略 `__pycache__/` 等缓存产物 |

### stage_2 —— 数据集下载与流式提取

| 文件 | 作用 |
|---|---|
| `config.json` | 本阶段可调参数：直链/清单/卷目录/提取目录（train/test 分开）、状态文件（train/test 分开）、test 标签文件与采集条件白名单、删除策略、下载线程数、日志路径 |
| `list_inventory.py` | 从各分卷中央目录解析出全量文件清单，写出 `output/Q0/inventory.json` |
| `analyze_inventory.py` | 分析清单，按个体/机型/场景/S批次统计 train 片段分布，写出 `output/Q0/inventory_summary.xlsx` |
| `downloader.py` | 纯 Python 多线程分块下载器（HTTP Range），支持 `.partNNN` 分块续传、完整性跳过、HEAD/Range 探尺寸 |
| `zip_central.py` | 中央目录解析工具：读取分卷末尾的中央目录，得到文件条目列表 |
| `stream_extract.py` | 多卷流式 ZIP 提取器：逐卷扫描本地文件头，处理跨卷文件，只提取指定扩展名；支持名称映射（数字名→语义名）与采集条件白名单过滤 |
| `run_download.py` | 下载+提取编排：逐卷下载→流式提取→删除卷，支持断点续传、`--limit` 分批、`--split train|test` 划分选择 |
| `selftest_stream.py` | 最小单元自测：构造仿多卷数据，验证流式提取的跨卷处理逻辑 |

### stage_3 —— 数据集实际情况统计与报告

| 文件 | 作用 |
|---|---|
| `config.json` | 本阶段可调参数：清单/标签路径、报告与统计输出路径、数据集官方描述元信息（型号数/个体数/片段下限等） |
| `dataset_report.py` | 读取清单与标签文件，统计型号/个体/场景/片段及 train/test/valid 分布，生成 `docs/Q0/数据集实际情况.md` 与 `output/Q0/dataset_stats.xlsx` |

### docs

| 文件 | 作用 |
|---|---|
| `file_tree.md` | 本文件：Q0 各目录与文件的含义说明 |

## docs/Q0 —— 人读报告（项目根目录下）

| 文件 | 作用 |
|---|---|
| `数据集实际情况.md` | stage_3 生成的数据集实际情况报告：规模、型号/个体/场景/片段分布、与官方描述对比 |
| `84cf9101e739402784b1396783881202.txt` | 数据集直链清单（分卷下载地址） |

## output/Q0 —— 预处理阶段结果

| 文件 | 作用 |
|---|---|
| `probe_summary.xlsx` | stage_1 探针汇总表，两个工作表：“说明”（字段含义）与“文件统计”（每个样本一行的探测结果） |
| `inventory.json` | stage_2 全量文件清单（各分卷中央目录解析结果，含 train/test 条目与大小信息） |
| `inventory_summary.xlsx` | stage_2 train 文件按个体/机型/场景/S批次的统计明细 |
| `dataset_stats.xlsx` | stage_3 数据集统计明细：个体片段统计、场景分布、字段说明 |
| `_probe_config.json` | stage_1 探针运行时的临时配置快照 |
| `_selftest/` | 自测临时目录，由 `selftest_carver.py` 自动创建，可随时删除 |

## python/Q0 —— 建模代码（后续阶段使用，本阶段暂无）

| 文件 | 作用 |
|---|---|
| （暂无） | 预留：CNN 训练、评估等建模代码按阶段放 `python/Q0/stage_x/`，通用函数放 `python/Q0/common/` |

## image/Q0 —— 图片（本阶段暂无）

| 文件 | 作用 |
|---|---|
| （暂无） | 预留：后续 STFT 时频图、处理前后对比图、性能曲线等放这里 |

## result/Q0 —— 重要数据（本阶段暂无）

| 文件 | 作用 |
|---|---|
| （暂无） | 预留：规范第 11 条“运行时产生的重要数据”放这里；本轮预处理结果按第 28 条放 `output/Q0` |

## log/Q0 —— 运行日志

| 文件 | 作用 |
|---|---|
| `probe_pipeline.log` | stage_1 流水线的完整运行日志（UTF-8），含每个文件的提取与探测过程 |
| `download_pipeline.log` | stage_2 下载与流式提取流水线日志（含断点续传、分批、错误记录） |
| `dataset_report.log` | stage_3 数据集统计报告的生成日志 |

## 数据目录（项目外，由用户手动下载）

| 路径 | 作用 |
|---|---|
| `D:/智能通信网络数据集/` | 原始分卷数据存放位置 |
| `D:/智能通信网络数据集/_volumes/` | 下载中的分卷临时目录（提取后按策略删除） |
| `D:/智能通信网络数据集/train_extracted/` | 流式提取出的 train 数据（`.mat`）与标签文件（`*_labels.txt`） |
| `C:/Users/lenovo/Desktop/ai_net_work/data` | 指向 `D:/智能通信网络数据集` 的 Junction 链接（工程内访问入口，不占 C 盘空间） |
