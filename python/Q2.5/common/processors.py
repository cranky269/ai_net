# -*- coding: utf-8 -*-
"""Q2.5 预处理方法插件库（均在 STFT 幅度谱上工作）。

当前已注册两类方法：

1. **频率方向中值滤波** —— 固定时刻，在每个频点取邻域中值；
2. **幅度域维纳滤波** —— ``|Y| = H·|X|``，其中
   ``H = max(beta, 1 - alpha·Snn/Syy)``，``Snn`` 由含噪谱自身估计。

为什么在频域做：80 MHz 采样下相邻时域样本几乎去相关，时域滤波会把窄带信号
当成孤立尖峰抹掉；而 STFT 后信号能量在频域集中成窄带峰，沿频率处理能保留峰、
压掉散乱的噪底。

统一接口：``fn(mag) -> mag``，``mag`` 为 ``|STFT|``（形状 ``[帧, 频点]``）。
新增方法只需在 :func:`make_processors` 注册，并用 ``domain`` 标明作用域。
"""

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import lfilter

# ==================== 中值滤波参数 ====================
DEFAULT_MEDIAN_WINDOWS = (3, 5, 9, 17)
DEFAULT_MEDIAN_MODE = "reflect"

# ==================== 均值滤波参数 ====================
DEFAULT_MEAN_WINDOWS = (3, 5, 9, 17)
DEFAULT_MEAN_MODE = "reflect"

# ==================== 维纳滤波参数 ====================
DEFAULT_WIENER_ALPHAS = (1.0, 1.5, 2.0, 3.0)
DEFAULT_WIENER_BETA = 0.05          # 谱下限，防止过度衰减
DEFAULT_WIENER_WIN = 500            # 噪声估计分位窗口（帧）
DEFAULT_WIENER_Q = 0.1              # 噪声底分位数
DEFAULT_WIENER_SMOOTH = 0.7         # 递归平滑系数
DEFAULT_WIENER_BIAS = 9.49          # 10% 分位 → 噪声均值 的偏差补偿（1/0.1054）
DEFAULT_WIENER_STEP_RATIO = 0.125   # 分位计算步长 = 窗口 × 该比例（越小越准、越慢）

# ==================== dB 阈值截断参数 ====================
DEFAULT_THRESH_DB = (-10.0, -20.0, -30.0, -40.0)   # 阈值档位（dB，相对该图峰值）
DEFAULT_STRETCH_FLOOR_DB = -80.0                   # 拉伸目标下限（dB，对应幅度 1e-4）
EPS = 1e-12
# =================================================


# ---------------------------------------------------------------- 中值滤波
def median_filter_freq(mag, window, mode=DEFAULT_MEDIAN_MODE):
    """沿频率轴（最后一维）做一维中值滤波，时间轴保持不变。"""
    window = int(window)
    if window < 3 or window % 2 == 0:
        raise ValueError("频率方向中值窗口必须为 >=3 的奇数，收到 %d" % window)
    out = median_filter(np.asarray(mag, dtype=np.float32),
                        size=(1, window), mode=mode)
    # median_filter 按输入 dtype 返回 float32，无需再整幅复制一份（6 万帧约 61 MB）
    return out if out.dtype == np.float32 else out.astype(np.float32)


def _median_freq_fn(window, mode):
    """生成单个频域中值滤波条目的调用函数。"""
    def _fn(mag, _w=window, _m=mode):
        return median_filter_freq(mag, _w, _m)

    return _fn


def mean_filter_freq(mag, window, mode=DEFAULT_MEAN_MODE):
    """沿频率轴（最后一维）做一维均值滤波，时间轴保持不变。

    与中值滤波的差别：均值会把邻域的强点摊到邻近频点，窄带谱线会被“抹宽”，
    但噪底的随机起伏压得更平；两者常作为链式后处理的可选项。
    """
    window = int(window)
    if window < 3 or window % 2 == 0:
        raise ValueError("频率方向均值窗口必须为 >=3 的奇数，收到 %d" % window)
    out = uniform_filter1d(np.asarray(mag, dtype=np.float32),
                           size=window, axis=-1, mode=mode)
    return out if out.dtype == np.float32 else out.astype(np.float32)


def _mean_freq_fn(window, mode):
    """生成单个频域均值滤波条目的调用函数。"""
    def _fn(mag, _w=window, _m=mode):
        return mean_filter_freq(mag, _w, _m)

    return _fn


# ---------------------------------------------------------------- 维纳滤波
def _smooth_power(syy, smooth):
    """沿时间轴做一阶递归平滑 ``s[t] = (1-a)·syy[t] + a·s[t-1]``。

    系数显式转成 float32，避免 ``lfilter`` 把结果升格为 float64（内存翻倍）。
    """
    b = np.array([1.0 - float(smooth)], dtype=np.float32)
    a = np.array([1.0, -float(smooth)], dtype=np.float32)
    return lfilter(b, a, syy, axis=0)


def _block_low_quantile(smoothed, win, q, step):
    """按步长取块、块内求分位，得到粗时间网格上的低分位曲线。

    不逐帧滑窗，而是每 ``step`` 帧取长度 ``win`` 的块求分位；
    复杂度从 O(帧数×窗长) 降到 O(帧数/步长×窗长)，结果再由调用方插值铺回。

    起始位置钳到最后能容纳整窗的位置，保证每块长度都是 ``win``，否则末尾残块
    会让 ``np.partition`` 的 ``kth`` 越界、且分位数失去可比性。
    """
    n_frames, n_freq = smoothed.shape
    rank = min(win, max(1, int(round(float(q) * win)))) - 1
    last = max(0, n_frames - win)
    starts = np.unique(np.minimum(np.arange(0, n_frames, step), last))
    low = np.empty((starts.size, n_freq), dtype=np.float32)
    for i, start in enumerate(starts):
        block = smoothed[start:start + win]
        low[i] = np.partition(block, rank, axis=0)[rank]
    return starts, low


def _interp_to_frames(starts, low, n_frames, win, bias):
    """把粗网格上的分位曲线沿时间线性插值回逐帧，并乘偏差补偿。

    插值权重钳在 ``[0, 1]``：两端不做外插（按 ``nearest`` 方式保持端块值），
    否则外插可能得到负的分位数，进而让增益 ``H > 1`` 反而放大信号。
    """
    if starts.size == 1:
        out = np.repeat(low, n_frames, axis=0)
        out *= np.float32(bias)
        return out

    centers = np.minimum(starts + win // 2, n_frames - 1).astype(np.float32)
    out = np.empty((n_frames, low.shape[1]), dtype=np.float32)
    chunk = 8192
    for start in range(0, n_frames, chunk):
        stop = min(start + chunk, n_frames)
        t = np.arange(start, stop, dtype=np.float32)
        pos = np.clip(np.searchsorted(centers, t, side="right") - 1,
                      0, centers.size - 2)
        weight = (t - centers[pos]) / np.maximum(centers[pos + 1] - centers[pos], 1.0)
        np.clip(weight, 0.0, 1.0, out=weight)
        lap = low[pos]                       # 分块，避免整幅临时数组
        lap = low[pos + 1] - lap
        lap *= weight[:, None]
        lap += low[pos]
        lap *= np.float32(bias)
        out[start:stop] = lap
    return out


def estimate_noise_power(syy, win=DEFAULT_WIENER_WIN, q=DEFAULT_WIENER_Q,
                         smooth=DEFAULT_WIENER_SMOOTH,
                         bias=DEFAULT_WIENER_BIAS, step=None):
    """沿时间轴估计噪声功率谱：递归平滑 → 块内低分位 → 时间插值 → 偏差补偿。

    指数分布的 10% 分位约为均值的 0.1054 倍，因此乘 ``bias≈9.49``
    才能还原为噪声平均功率，使 ``alpha=1`` 对应标准维纳。

    说明：低分位在“每 ``step`` 帧一个长度为 ``win`` 的块”上计算，再线性插值
    铺回逐帧（两端不外插）。与逐帧滑窗相比精度损失很小——6 万帧 × 257 频点上
    实测中位相对误差约 0.8%、p99 约 6.7%（端点首尾各数帧因旧实现按 ``nearest``
    补零边而偏差较大，属预期）；耗时约 1 s，原先逐帧 ``rank_filter`` 需约 80 s。
    """
    syy = np.asarray(syy, dtype=np.float32)
    n_frames = syy.shape[0]
    win = max(3, min(int(win), n_frames))
    if step is None:
        step = max(1, int(round(win * DEFAULT_WIENER_STEP_RATIO)))
    step = max(1, int(step))

    smoothed = _smooth_power(syy, smooth)
    starts, low = _block_low_quantile(smoothed, win, q, step)
    del smoothed
    return _interp_to_frames(starts, low, n_frames, win, bias)


def wiener_filter(mag, alpha, beta=DEFAULT_WIENER_BETA, noise=None,
                  win=DEFAULT_WIENER_WIN, q=DEFAULT_WIENER_Q,
                  smooth=DEFAULT_WIENER_SMOOTH, bias=DEFAULT_WIENER_BIAS,
                  step=None):
    """幅度域维纳滤波：``|Y| = H·|X|``，相位不变。

    增益 ``H = max(beta, 1 - alpha·Snn/Syy)``；全程在 float32 缓冲上原地运算，
    峰值内存约为 ``mag`` 的 3 倍（``syy`` 缓冲被复用为增益）。
    """
    mag = np.asarray(mag, dtype=np.float32)
    syy = np.square(mag)
    if noise is None:
        noise = estimate_noise_power(syy, win, q, smooth, bias, step)
    syy += EPS                                   # 防止除零
    gain = np.divide(noise, syy, out=syy)        # 复用 syy 缓冲
    gain *= -float(alpha)
    gain += 1.0
    np.maximum(gain, float(beta), out=gain)
    return np.multiply(gain, mag, out=gain)


class NoiseEstimator:
    """带缓存的噪声功率谱估计器，供多个 ``alpha`` 共享同一份 ``Snn``。

    以输入数组的 ``id`` 为键并保留数组引用，避免 ``id`` 复用导致的错配。
    """

    def __init__(self, win=DEFAULT_WIENER_WIN, q=DEFAULT_WIENER_Q,
                 smooth=DEFAULT_WIENER_SMOOTH, bias=DEFAULT_WIENER_BIAS,
                 step=None, max_items=3):
        self.win = int(win)
        self.q = float(q)
        self.smooth = float(smooth)
        self.bias = float(bias)
        self.step = None if step is None else int(step)
        self.max_items = int(max_items)
        self._cache = {}

    def get(self, mag):
        """返回该幅度谱对应的噪声功率估计（命中缓存则直接复用）。"""
        key = id(mag)
        hit = self._cache.get(key)
        if hit is not None and hit[0] is mag:
            return hit[1]
        syy = np.square(np.asarray(mag, dtype=np.float32))
        snn = estimate_noise_power(syy, self.win, self.q, self.smooth,
                                   self.bias, self.step)
        del syy
        if len(self._cache) >= self.max_items:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = (mag, snn)
        return snn


def _wiener_fn(alpha, beta, estimator):
    """生成单个维纳条目的调用函数（共享噪声估计器）。"""
    def _fn(mag, _a=alpha, _b=beta, _e=estimator):
        return wiener_filter(mag, _a, _b, noise=_e.get(mag))

    return _fn


# ------------------------------------------------------- dB 阈值截断
def _peak_of(mag):
    """幅度谱的峰值；空数组返回 0。"""
    return float(mag.max()) if mag.size else 0.0


def _check_threshold(th_db):
    """阈值必须是负的 dB 值（相对该图峰值），返回 float。"""
    th_db = float(th_db)
    if th_db >= 0.0:
        raise ValueError("阈值需为负的 dB 值（相对峰值），收到 %s" % th_db)
    return th_db


def _relative_db(mag, peak):
    """幅度 → 相对峰值的 dB（float32），0 dB 即峰值。"""
    db = np.multiply(np.asarray(mag, dtype=np.float32), np.float32(1.0 / peak))
    db += np.float32(EPS)                  # 防 log10(0) 得到 -inf
    np.log10(db, out=db)
    db *= np.float32(20.0)
    return db


def db_lin_compress(mag, th_db, floor_db=DEFAULT_STRETCH_FLOOR_DB):
    """dB 域线性：先按阈值钳位，再把 ``[th, 0]`` dB 线性铺满 ``[floor, 0]`` dB。

    等价于幅度谱的 γ 律压缩 ``|Y| = M·(|X|/M)^(|floor/th|)``：
    ``th=-20``、``floor=-80`` 时指数为 4，-10 dB 的点被映射到 -40 dB。
    """
    mag = np.asarray(mag, dtype=np.float32)
    peak = _peak_of(mag)
    if peak <= 0.0:
        return np.zeros_like(mag)
    th_db = _check_threshold(th_db)

    db = _relative_db(mag, peak)
    np.maximum(db, np.float32(th_db), out=db)       # 阶段一：钳到阈值
    db *= np.float32(float(floor_db) / th_db)       # 阶段二：[th, 0] → [floor, 0]
    db *= np.float32(1.0 / 20.0)
    np.power(np.float32(10.0), db, out=db)          # dB → 幅度
    db *= np.float32(peak)
    return db


def amp_lin_stretch(mag, th_db, floor_db=DEFAULT_STRETCH_FLOOR_DB):
    """幅度域线性：先按阈值钳位，再把幅度区间 ``[lo, M]`` 仿射拉伸到 ``[M·10^(floor/20), M]``。

    与 :func:`db_lin_compress` 的差别在第二步作用于**线性幅度**：接近峰值的成分
    几乎不变、越弱压得越狠，而不是把 dB 层级等间距拉开。
    """
    mag = np.asarray(mag, dtype=np.float32)
    peak = _peak_of(mag)
    if peak <= 0.0:
        return np.zeros_like(mag)
    th_db = _check_threshold(th_db)

    lo = peak * 10.0 ** (th_db / 20.0)              # 截断后的最小幅度
    ymin = peak * 10.0 ** (float(floor_db) / 20.0)  # 拉伸目标下限
    if peak - lo <= 0.0:                            # 阈值趋近 0 的退化情形
        return mag.copy()

    out = np.maximum(mag, np.float32(lo))           # 阶段一：幅度钳位
    out -= np.float32(lo)
    out *= np.float32((peak - ymin) / (peak - lo))
    out += np.float32(ymin)
    return out


def _db_lin_fn(th_db, floor_db):
    """生成单个 dB 域线性条目的调用函数。"""
    def _fn(mag, _t=th_db, _f=floor_db):
        return db_lin_compress(mag, _t, _f)

    return _fn


def _amp_lin_fn(th_db, floor_db):
    """生成单个幅度域线性条目的调用函数。"""
    def _fn(mag, _t=th_db, _f=floor_db):
        return amp_lin_stretch(mag, _t, _f)

    return _fn


def _threshold_proc(name, th_db, floor_db, factory):
    """构造一条阈值截断方法条目（``db_lin`` / ``amp_lin`` 共用结构）。"""
    th_db = float(th_db)
    return {
        "name": name,
        "domain": "stft",
        "param": th_db,
        "label": "%s_%g" % (name, th_db),
        "cache_f_x": False,          # 逐点映射，重算无代价
        "fn": factory(th_db, floor_db),
    }


# ---------------------------------------------------------------- 注册
def make_processors(cfg):
    """按配置生成方法列表。

    返回 ``[{"name", "domain", "param", "label", "fn"}, ...]``；
    当前所有方法 ``domain="stft"``，``fn`` 作用于 ``|STFT|`` 幅度矩阵。
    """
    procs = []

    mode = str(cfg.get("median_mode", DEFAULT_MEDIAN_MODE))
    for window in cfg.get("median_windows", list(DEFAULT_MEDIAN_WINDOWS)):
        window = int(window)
        procs.append({
            "name": "median_freq",
            "domain": "stft",
            "param": window,
            "label": "median_freq_%d" % window,
            # 沿频率滑窗的中值滤波单价高，探针可跨 SNR 缓存它在干净谱上的输出
            "cache_f_x": True,
            "fn": _median_freq_fn(window, mode),
        })

    mean_mode = str(cfg.get("mean_mode", DEFAULT_MEAN_MODE))
    for window in cfg.get("mean_windows", list(DEFAULT_MEAN_WINDOWS)):
        window = int(window)
        procs.append({
            "name": "mean_freq",
            "domain": "stft",
            "param": window,
            "label": "mean_freq_%d" % window,
            "cache_f_x": False,
            "fn": _mean_freq_fn(window, mean_mode),
        })

    alphas = cfg.get("wiener_alphas", list(DEFAULT_WIENER_ALPHAS))
    if alphas:
        win = int(cfg.get("wiener_win", DEFAULT_WIENER_WIN))
        step = cfg.get("wiener_step")
        if step is None:                     # null / 缺省 → 由窗口乘以比例推导
            ratio = float(cfg.get("wiener_step_ratio", DEFAULT_WIENER_STEP_RATIO))
            step = max(1, int(round(win * ratio)))
        estimator = NoiseEstimator(
            win=win,
            q=float(cfg.get("wiener_q", DEFAULT_WIENER_Q)),
            smooth=float(cfg.get("wiener_smooth", DEFAULT_WIENER_SMOOTH)),
            bias=float(cfg.get("wiener_bias", DEFAULT_WIENER_BIAS)),
            step=int(step),
            # 每份 Snn 约 61 MB；3 份刚好覆盖探针里 X_x / X_n / X_xn 三个输入的
            # 交替访问，再多没意义，少了会反复重估。
            max_items=int(cfg.get("wiener_cache_items", 3)),
        )
        beta = float(cfg.get("wiener_beta", DEFAULT_WIENER_BETA))
        for alpha in alphas:
            alpha = float(alpha)
            procs.append({
                "name": "wiener",
                "domain": "stft",
                "param": alpha,
                "label": "wiener_a%.1f" % alpha,
                # 增益运算便宜，且噪声估计已由 estimator 缓存，不必再存整幅 f_x
                "cache_f_x": False,
                "fn": _wiener_fn(alpha, beta, estimator),
            })

    # 两种 dB 阈值截断共用同一组阈值档位与拉伸下限
    floor_db = float(cfg.get("stretch_floor_db", DEFAULT_STRETCH_FLOOR_DB))
    for th_db in cfg.get("db_lin_thresholds", list(DEFAULT_THRESH_DB)):
        procs.append(_threshold_proc("db_lin", th_db, floor_db, _db_lin_fn))
    for th_db in cfg.get("amp_lin_thresholds", list(DEFAULT_THRESH_DB)):
        procs.append(_threshold_proc("amp_lin", th_db, floor_db, _amp_lin_fn))
    return procs
