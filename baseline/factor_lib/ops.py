# -*- coding: utf-8 -*-
"""因子库算子 / Factor-library operators.

所有算子输入输出统一为「日期 × 股票」的宽表 DataFrame，语义与 JoinQuant
Alpha191 / QLib Alpha158 / WorldQuant Alpha101 的常见实现保持一致：

- 横截面算子（rank / scale / row_max ...）沿 axis=1（股票维度）；
- 时序算子（ts_* / corr / cov / sma ...）沿时间轴滚动；
- 停牌（NaN）不填充，滚动窗口用 min_periods≈0.6n 容忍缺失；
- 所有除法经 safe_div，分母为 0 时返回 NaN。

All operators take and return date x symbol DataFrames. Cross-sectional
operators work along axis=1, time-series operators roll along the index.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

__all__ = [
    "delay", "delta", "rank", "ts_rank", "ts_sum", "ts_mean", "ts_std",
    "ts_max", "ts_min", "corr", "cov", "ema", "sma", "sign", "log", "absv",
    "emax", "emin", "scale", "decay_linear", "regbeta", "lowday", "highday",
    "count", "sumif", "product", "ts_argmax", "ts_argmin", "wma", "row_max",
    "row_min", "sequence", "returns", "adv", "safe_div", "clean",
]

EPS = 1e-12


# ===================== 基础工具 / Helpers =====================
def _minp(n: int) -> int:
    """滚动窗口最小有效样本数，约 60% 窗口且不超过窗口长度。
    Min valid samples in a window: about 60% of the window, capped at n.
    """
    return min(n, max(2, int(np.ceil(0.6 * n))))


def safe_div(num, den):
    """分母为 0 时返回 NaN 的除法。Division that maps zero denominators to NaN."""
    if np.isscalar(den):
        return num / den if den != 0 else num * np.nan
    return num / den.replace(0, np.nan)


def clean(x, limit: float = 1e6):
    """把 inf 置 NaN，并按上下限截断。Replace inf with NaN and clip the range."""
    return x.replace([np.inf, -np.inf], np.nan).clip(-limit, limit)


# ===================== 横截面算子 / Cross-sectional =====================
def rank(x: pd.DataFrame) -> pd.DataFrame:
    """横截面百分位排名（0~1）。Cross-sectional percentile rank."""
    return x.rank(axis=1, pct=True)


def scale(x: pd.DataFrame, a: float = 1.0) -> pd.DataFrame:
    """按截面绝对值和归一化，总和为 a。Scale so the cross-sectional abs sum is a."""
    denom = x.abs().sum(axis=1).replace(0, np.nan)
    return x.div(denom, axis=0) * a


def row_max(x: pd.DataFrame) -> pd.Series:
    """每行（每个交易日）跨股票取最大值。Row-wise max across symbols."""
    return x.max(axis=1)


def row_min(x: pd.DataFrame) -> pd.Series:
    """每行（每个交易日）跨股票取最小值。Row-wise min across symbols."""
    return x.min(axis=1)


# ===================== 时序算子 / Time series =====================
def delay(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.shift(n)


def delta(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x - x.shift(n)


def ts_rank(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """窗口内最后一个值的时序百分位。Time-series percentile rank of the latest value."""
    return x.rolling(n, min_periods=_minp(n)).rank(pct=True)


def ts_sum(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=_minp(n)).sum()


def ts_mean(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=_minp(n)).mean()


def ts_std(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=_minp(n)).std()


def ts_max(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=_minp(n)).max()


def ts_min(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=_minp(n)).min()


def corr(x: pd.DataFrame, y: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=_minp(n)).corr(y)


def cov(x: pd.DataFrame, y: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=_minp(n)).cov(y)


def adv(x: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """N 日均值（常用作平均成交额）。N-day average (used as ADV)."""
    return x.rolling(n, min_periods=_minp(n)).mean()


def ema(x: pd.DataFrame, n: int, m: int = 1) -> pd.DataFrame:
    """中国式 SMA：y_t = (m*x_t + (n-m)*y_{t-1}) / n。

    Chinese-style SMA, i.e. an EWMA with alpha = m / n. This is the Alpha191
    ``Sma(X, N, M)`` operator (not a simple moving average).
    """
    alpha = float(m) / float(n)
    return x.ewm(alpha=alpha, adjust=False, min_periods=1).mean()


#: Alpha191 里 ``Sma`` 指向中国式 EMA / Alpha191 ``Sma`` is the Chinese EMA.
sma = ema


def decay_linear(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """线性衰减加权平均，最新值权重最高；缺失值按有效权重归一。
    Linearly decaying weighted mean with the newest value weighted most.
    """
    num = None
    den = None
    for i in range(n):
        w = float(n - i)
        shifted = x.shift(i)
        term = shifted * w
        num = term if num is None else num.add(term, fill_value=0.0)
        mask = shifted.notna().astype(float) * w
        den = mask if den is None else den.add(mask, fill_value=0.0)
    return safe_div(num, den.where(den > 0))


def wma(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """加权移动平均，权重 1..n（最新值权重最大）。Weighted moving average."""
    return decay_linear(x, n)


def regbeta(y: pd.DataFrame, x) -> pd.DataFrame:
    """对固定自变量序列 x 的滚动回归斜率。

    Alpha191 的 ``Regbeta(y, Sequence(n))`` 自变量恒为 1..n，因此斜率是一个
    固定权重的线性滤波，可用 n 次 shift 向量化计算，无需逐窗回归。

    Rolling regression slope of y on a fixed regressor sequence x. Since
    Alpha191 only uses Sequence(n), the slope is a fixed-weight linear filter.
    """
    if isinstance(x, pd.Series):
        xv = x.to_numpy(dtype=float)
    else:
        xv = np.asarray(x, dtype=float).ravel()
    n = len(xv)
    xc = xv - xv.mean()
    denom = float((xc * xc).sum())
    if denom <= 0:
        return y * np.nan
    out = None
    for k in range(n):
        term = y.shift(n - 1 - k) * xc[k]
        out = term if out is None else out.add(term, fill_value=0.0)
    valid = y.notna().astype(float).rolling(n, min_periods=n).sum()
    out = out / denom
    return out.where(valid >= n)


def _rolling_arg(x: pd.DataFrame, n: int, mode: str = "max",
                 from_end: bool = False) -> pd.DataFrame:
    """滚动窗口内极值的位置（0=窗口最早，n-1=最新）。

    用 sliding_window_view 按列分块计算，避免 rolling.apply 的 Python 循环。

    Position of the rolling extreme. Computed block-wise with sliding windows.
    """
    arr = x.to_numpy(dtype=np.float64)
    t_len, s_len = arr.shape
    out = np.full((t_len, s_len), np.nan)
    if t_len < n:
        return pd.DataFrame(out, index=x.index, columns=x.columns)

    # 每块控制窗口张量在 2e8 字节以内 / keep the window tensor under 200 MB
    block = max(1, int(2e8 / max(1, (t_len - n + 1) * n * 8)))
    for start in range(0, s_len, block):
        stop = min(s_len, start + block)
        sub = arr[:, start:stop]
        windows = sliding_window_view(sub, n, axis=0)  # (T-n+1, cols, n)
        if mode == "max":
            pos = windows.argmax(axis=2)
        else:
            pos = windows.argmin(axis=2)
        bad = np.isnan(windows).any(axis=2)
        pos = pos.astype(float)
        pos[bad] = np.nan
        out[n - 1:, start:stop] = pos
        del windows, pos, bad
    if from_end:
        out = (n - 1) - out
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def ts_argmax(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """窗口内最大值的位置（0=最早）。Position of the rolling max."""
    return _rolling_arg(x, n, "max")


def ts_argmin(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """窗口内最小值的位置（0=最早）。Position of the rolling min."""
    return _rolling_arg(x, n, "min")


def lowday(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """距窗口最低值的天数（0=今天）。Days since the rolling minimum (0 = today)."""
    return _rolling_arg(x, n, "min", from_end=True)


def highday(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """距窗口最高值的天数（0=今天）。Days since the rolling maximum (0 = today)."""
    return _rolling_arg(x, n, "max", from_end=True)


def count(cond: pd.DataFrame, n: int) -> pd.DataFrame:
    """窗口内条件成立的天数。Number of true observations in the window."""
    return cond.astype(float).rolling(n, min_periods=_minp(n)).sum()


def sumif(part: pd.DataFrame, n: int, cond: pd.DataFrame) -> pd.DataFrame:
    """窗口内条件成立时对 part 求和。Rolling sum of part where cond is true."""
    return part.where(cond, 0.0).rolling(n, min_periods=_minp(n)).sum()


def product(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """窗口内连乘（用对数求和实现）。Rolling product via log-sum-exp."""
    if n <= 1:
        return x
    lx = np.log(x.where(x > 0))
    return np.exp(lx.rolling(n, min_periods=_minp(n)).sum())


# ===================== 逐元素算子 / Element-wise =====================
def sign(x: pd.DataFrame) -> pd.DataFrame:
    return np.sign(x)


def log(x: pd.DataFrame) -> pd.DataFrame:
    return np.log(x.where(x > 0))


def absv(x: pd.DataFrame) -> pd.DataFrame:
    return x.abs()


def emax(a, b) -> pd.DataFrame:
    """逐元素取大，b 可为标量或同形状表。Element-wise maximum."""
    out = np.maximum(a, b)
    if isinstance(out, pd.DataFrame):
        return out
    return pd.DataFrame(out, index=a.index, columns=a.columns)


def emin(a, b) -> pd.DataFrame:
    """逐元素取小，b 可为标量或同形状表。Element-wise minimum."""
    out = np.minimum(a, b)
    if isinstance(out, pd.DataFrame):
        return out
    return pd.DataFrame(out, index=a.index, columns=a.columns)


# ===================== 其它 / Misc =====================
def sequence(n: int) -> pd.Series:
    """1..n 的等差序列。Arithmetic sequence 1..n."""
    return pd.Series(np.arange(1, n + 1, dtype=float))


def returns(x: pd.DataFrame) -> pd.DataFrame:
    """简单收益率。Simple returns."""
    return x.pct_change(fill_method=None)
