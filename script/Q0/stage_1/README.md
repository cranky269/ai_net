# Q0 预处理阶段（stage_1）· 数据探针运行说明

## 1. 环境准备

```bash
python -m pip install -r requirements.txt
```

已实测：Python 3.13.2 + numpy 2.5.1 + h5py 3.16.0 + openpyxl 3.1.5。

## 2. 数据准备（人工操作）

DroneRFb-DIR 以分卷压缩发布，**只需下载第一卷**（`.z01`，约 2 GB）即可做数据探针：

1. 打开 https://www.scidb.cn/detail?dataSetId=84cf9101e739402784b1396783881202
2. 下载列表中编号为 `.z01` 的那个文件（不要下 `.z02` 及之后，它们不含文件头）
3. 放到 `D:\智能通信网络数据集\` 下（子目录也可以，脚本会递归查找）

## 3. 运行

```bash
python script/Q0/stage_1/run_pipeline.py
```

复用已提取的文件（跳过解包）：

```bash
python script/Q0/stage_1/run_pipeline.py --skip-extract
```

自测（无需真实数据，验证代码本身）：

```bash
python script/Q0/stage_1/selftest_carver.py
```

## 4. 输出

| 位置 | 内容 |
|---|---|
| `output/Q0/probe_summary.xlsx` | 探针汇总表（“说明”+“文件统计”两个工作表） |
| `log/Q0/probe_pipeline.log` | 完整运行日志 |
| `D:\智能通信网络数据集\_probe_extracted\` | 从卷中提取出的 `.mat` 样本 |

## 5. 主要参数（`config.json`）

| 参数 | 含义 | 默认值 |
|---|---|---|
| `data_root` | 数据根目录 | `D:/智能通信网络数据集` |
| `volume_pattern` | 分卷匹配规则（支持 `**` 递归） | `**/*.z01` |
| `max_extract_files` | 单次最多提取多少文件 | `6` |
| `extensions` | 需要保存的扩展名 | `[".mat"]` |
| `fs_hz` | 采样率，用于换算片段时长 | `80000000` |
| `snr_estimate.segment_len` | 估算 SNR 时的分段长度 | `4096` |
| `snr_estimate.max_segments` | 最多平均多少段 | `512` |
| `snr_estimate.signal_over_floor_db` | 判为"信号频点"的门限 | `10.0` |

## 6. 原理与限制

脚本不依赖 ZIP 中央目录（它在最后一卷），而是顺序扫描每个条目的**本地文件头**
（`PK\x03\x04`），按头部记录的压缩长度就地解压，因此单卷即可取出一批完整文件。

限制：

- 只对第一卷（`.z01`）有效；
- 该卷末尾跨卷截断的文件会被丢弃，属正常现象；
- 若实际文件名形如 `xxx.7z.001`，请把 `volume_pattern` 改为 `**/*.001`（但 7z 分卷的头部格式不同，需要另行适配）。
