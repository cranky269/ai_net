# Q1 文件架构说明

本文件只记录“每个文件的作用与含义”，不记录任何运行得到的结论。

> Q1 为无人机个体识别任务；当前仍处于**预处理阶段**（数据准备与切分），尚未进入建模。

## script/Q1 —— 预处理阶段代码

### stage_1 —— 数据集切分

| 文件 | 作用 |
|---|---|
| `config.json` | 切分参数：数据目录、输出清单路径、日志路径；场景划分规则（`IN_S0/IN_S1 → train`、`IN_S2 → test`）；background 切分比例与固定随机种子 |
| `split_dataset.py` | 按场景切分数据集，生成划分清单 xlsx（只读取文件名，不移动/修改任何 `.mat` 文件） |

### docs

| 文件 | 作用 |
|---|---|
| `file_tree.md` | 本文件：Q1 各目录与文件的含义说明 |

## output/Q1 —— 预处理阶段结果

| 文件 | 作用 |
|---|---|
| `splits/dataset_split.xlsx` | 数据划分清单：每行一个文件，列为 `file / individual / model / scene / label / split`；训练/评估代码据此按 `split` 加载，文件本体保持原位 |

## log/Q1 —— 运行日志

| 文件 | 作用 |
|---|---|
| `split_dataset.log` | stage_1 切分脚本的运行日志（扫描数、background 分配、划分统计） |
