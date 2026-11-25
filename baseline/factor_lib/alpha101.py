# -*- coding: utf-8 -*-
"""WorldQuant《101 Formulaic Alphas》向量化实现 / Vectorised Alpha101.

`compute_alpha101(matrices)` 是 generator，逐个 yield ``(因子名, 日频因子表)``，
因子名为 ``a101_001`` … ``a101_101``。全部公式在「日期 × 股票」宽表上整表
向量化计算，不逐股票循环。

口径说明 / Conventions
--------------------
- 算子全部来自 :mod:`baseline.factor_lib.ops`；横截面算子（rank / scale）
  沿 axis=1，时序算子沿时间轴滚动，缺失值由算子的 min_periods 容忍。
- ``adv{d}`` 取「d 日均成交额」= ``amount`` 的 d 日 rolling mean；论文中的
  ``adv20`` 即 20 日均成交额（``ADV_WINDOW``）。
- ``cap`` 取 ``total_mv``（缺失则跳过 Alpha#56）。
- ``IndNeutralize(x, IndClass.*)`` 用 ``utils.data_standardize._industry_map``
  的行业快照做横截面行业去均值（论文的 sector / industry / subindustry 三级
  在本地数据里只有一套行业字段，故统一用同一快照近似）。
- 公式中的小数窗口（如 3.92795）四舍五入到整数交易日。
- 除法统一走 ``safe_div``；分数次幂走 ``signedpower`` 并对指数截断到 ±5，
  防止溢出；输出统一经 ``clean`` 把 ±inf 置 NaN 并截断到 ±1e6。
- ``turnover_rate`` / ``is_st`` 在 Alpha101 原始公式中未被使用。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ops import (
    absv, clean, cov, decay_linear, delay, delta, emax, emin, log, product,
    rank, safe_div, scale, sign, ts_argmax, ts_argmin, ts_max, ts_mean, ts_min,
    ts_rank, ts_std, ts_sum,
)
from .ops import corr as _ops_corr

__all__ = ["compute_alpha101", "signedpower", "indneutralize", "ADV_WINDOW"]

#: 论文中 ``adv20`` 的窗口（交易日）/ window of the paper's ``adv20``.
ADV_WINDOW = 20


# ===================== 辅助函数 / Helpers =====================
def corr(x, y, n):
    """滚动相关；窗口内一方标准差为 0 时 pandas 会返回 ±inf，这里置 NaN。

    Rolling correlation. pandas returns ±inf when either side has zero variance
    in the window (possible with min_periods < n); map those to NaN.
    """
    return _ops_corr(x, y, n).replace([np.inf, -np.inf], np.nan)


def signedpower(x, a):
    """``sign(x) * |x|^a``；指数截断到 ±5，避免负数开方与幂溢出。

    Signed power as used by Alpha#1 / #84 etc. The exponent is clipped to
    [-5, 5] so that a large ``delta(close, n)`` exponent cannot overflow.
    """
    if isinstance(a, (pd.DataFrame, pd.Series)):
        a = a.clip(-5.0, 5.0)
    else:
        a = float(np.clip(a, -5.0, 5.0))
    return clean(sign(x) * absv(x) ** a)


def _load_industry(columns):
    """把行业快照对齐到股票列；不可得时返回 None。Align industry to columns."""
    try:
        from utils.data_standardize import _industry_map
        ind = _industry_map().reindex(columns)
    except Exception:  # 行业文件缺失 / scipy 不可用等，退化为全局去均值
        return None
    if ind.notna().sum() == 0:      # 代码口径对不上（如测试用的随机列名）
        return None
    return ind.fillna("未分类").to_numpy()


def indneutralize(x, ind=None):
    """横截面行业去均值；行业不可得时退化为全市场去均值。

    Cross-sectional industry demeaning via ``utils.data_standardize``'s
    industry snapshot. If the snapshot is unavailable or does not match the
    column codes, it falls back to cross-sectional (whole-market) demeaning.
    """
    if ind is None:
        ind = _load_industry(x.columns)
    if ind is None:
        return x.sub(x.mean(axis=1), axis=0)
    means = x.T.groupby(ind).transform("mean").T
    return x - means


def _num(cond):
    """布尔表转 0/1 浮点表（便于再 rank）。Cast a boolean frame to float."""
    return cond.astype(float)


def _frame_like(ref: pd.DataFrame, value: float) -> pd.DataFrame:
    """构造与 ref 同形状的常数表。Constant frame shaped like ``ref``."""
    return pd.DataFrame(value, index=ref.index, columns=ref.columns)


class _Ctx(object):
    """公式上下文：行情宽表 + 派生量 + 缓存。Formula context and caches."""

    def __init__(self, matrices: dict):
        self.close = matrices["close"]
        idx, cols = self.close.index, self.close.columns

        def get(name):
            frame = matrices.get(name)
            if frame is None:
                return None
            return frame.reindex(index=idx, columns=cols)

        # 缺失的价量字段用 close 兜底；成交额与 vwap 互相推导
        self.open = get("open")
        self.high = get("high")
        self.low = get("low")
        self.volume = get("volume")
        self.amount = get("amount")
        self.vwap = get("vwap")

        if self.open is None:
            self.open = self.close.copy()
        if self.high is None:
            self.high = self.close.copy()
        if self.low is None:
            self.low = self.close.copy()
        if self.volume is None and self.amount is not None:
            self.volume = safe_div(self.amount, self.close)
        if self.volume is None:
            raise KeyError("alpha101 需要 volume 或 amount 字段")
        if self.amount is None:
            self.amount = self.close * self.volume
        if self.vwap is None:
            self.vwap = safe_div(self.amount, self.volume)

        self.ret = get("returns")
        if self.ret is None:
            self.ret = self.close.pct_change(fill_method=None)
        self.cap = get("total_mv")

        self._adv = {}
        self._ind = _load_industry(cols)
        if self._ind is None:
            print("[alpha101] 行业快照不可用或与股票代码不匹配，"
                  "IndNeutralize 退化为全市场横截面去均值")

    def adv(self, n: int = ADV_WINDOW) -> pd.DataFrame:
        """n 日均成交额。n-day average traded amount."""
        if n not in self._adv:
            self._adv[n] = ts_mean(self.amount, n)
        return self._adv[n]

    def ind(self, x: pd.DataFrame) -> pd.DataFrame:
        """行业去均值（缓存的行业快照）。Industry-neutralise ``x``."""
        return indneutralize(x, self._ind)


def _as_frame(value, ref: pd.DataFrame) -> pd.DataFrame:
    """把标量 / Series / DataFrame 统一成与 ref 同形状的表。Coerce to frame."""
    if isinstance(value, pd.DataFrame):
        if value.shape == ref.shape:
            out = value.copy()
            out.index = ref.index
            out.columns = ref.columns
            return out
        return value.reindex(index=ref.index, columns=ref.columns)
    if isinstance(value, pd.Series):
        data = np.repeat(value.to_numpy(dtype=float)[:, None], ref.shape[1], axis=1)
        return pd.DataFrame(data, index=ref.index, columns=ref.columns)
    return _frame_like(ref, float(value))


# ===================== 公式 / Formulas =====================
# --- Alpha#1 ~ #10 ---
def _a001(m):
    inner = m.close.where(m.ret >= 0, ts_std(m.ret, 20))
    return rank(ts_argmax(signedpower(inner, 2.0), 5)) - 0.5


def _a002(m):
    return -1 * corr(rank(delta(log(m.volume), 2)),
                     rank(safe_div(m.close - m.open, m.open)), 6)


def _a003(m):
    return -1 * corr(rank(m.open), rank(m.volume), 10)


def _a004(m):
    return -1 * ts_rank(rank(m.low), 9)


def _a005(m):
    return rank(m.open - ts_mean(m.vwap, 10)) * (-1 * absv(rank(m.close - m.vwap)))


def _a006(m):
    return -1 * corr(m.open, m.volume, 10)


def _a007(m):
    adv20 = m.adv(20)
    inner = -1 * ts_rank(absv(delta(m.close, 7)), 60) * sign(delta(m.close, 7))
    return inner.where(adv20 < m.volume, -1.0)


def _a008(m):
    inner = ts_sum(m.open, 5) * ts_sum(m.ret, 5)
    return -1 * rank(inner - delay(inner, 10))


def _a009(m):
    d = delta(m.close, 1)
    cond = (ts_min(d, 5) > 0) | (ts_max(d, 5) < 0)
    return (-1 * d).where(~cond, d)


def _a010(m):
    d = delta(m.close, 1)
    cond = (ts_min(d, 4) > 0) | (ts_max(d, 4) < 0)
    return rank((-1 * d).where(~cond, d))


# --- Alpha#11 ~ #20 ---
def _a011(m):
    diff = m.vwap - m.close
    return (rank(ts_max(diff, 3)) + rank(ts_min(diff, 3))) * rank(delta(m.volume, 3))


def _a012(m):
    return sign(delta(m.volume, 1)) * (-1 * delta(m.close, 1))


def _a013(m):
    return -1 * rank(cov(rank(m.close), rank(m.volume), 5))


def _a014(m):
    return -1 * rank(delta(m.ret, 3)) * corr(m.open, m.volume, 10)


def _a015(m):
    return -1 * ts_sum(rank(corr(rank(m.high), rank(m.volume), 3)), 3)


def _a016(m):
    return -1 * rank(cov(rank(m.high), rank(m.volume), 5))


def _a017(m):
    adv20 = m.adv(20)
    return (-1 * rank(ts_rank(m.close, 10))
            * rank(delta(delta(m.close, 1), 1))
            * rank(ts_rank(safe_div(m.volume, adv20), 5)))


def _a018(m):
    return -1 * rank(ts_std(absv(m.close - m.open), 5)
                     + (m.close - m.open)
                     + corr(m.close, m.open, 10))


def _a019(m):
    return (-1 * sign((m.close - delay(m.close, 7)) + delta(m.close, 7))
            * (1 + rank(1 + ts_sum(m.ret, 250))))


def _a020(m):
    return (-1 * rank(m.open - delay(m.high, 1))
            * rank(m.open - delay(m.close, 1))
            * rank(m.open - delay(m.low, 1)))


# --- Alpha#21 ~ #30 ---
def _a021(m):
    c8, s8, c2 = ts_mean(m.close, 8), ts_std(m.close, 8), ts_mean(m.close, 2)
    cond1 = (c8 + s8) < c2
    cond2 = c2 < (c8 - s8)
    cond3 = safe_div(m.volume, m.adv(20)) >= 1
    out = _frame_like(m.close, -1.0)
    out = out.where(~cond3, 1.0)
    out = out.where(~cond2, 1.0)
    return out.where(~cond1, -1.0)


def _a022(m):
    return -1 * (delta(corr(m.high, m.volume, 5), 5) * rank(ts_std(m.close, 20)))


def _a023(m):
    return (-1 * delta(m.high, 2)).where(ts_mean(m.high, 20) < m.high, 0.0)


def _a024(m):
    ratio = safe_div(delta(ts_mean(m.close, 100), 100), delay(m.close, 100))
    return (-1 * (m.close - ts_min(m.close, 100))).where(
        ratio <= 0.05, -1 * delta(m.close, 3))


def _a025(m):
    return rank(((-1 * m.ret) * m.adv(20)) * m.vwap * (m.high - m.close))


def _a026(m):
    return -1 * ts_max(corr(ts_rank(m.volume, 5), ts_rank(m.high, 5), 5), 3)


def _a027(m):
    inner = rank(ts_mean(corr(rank(m.volume), rank(m.vwap), 6), 2))
    return _frame_like(m.close, 1.0).where(~(inner > 0.5), -1.0)


def _a028(m):
    return scale(corr(m.adv(20), m.low, 5) + (m.high + m.low) / 2 - m.close)


def _a029(m):
    inner = -1 * rank(delta(m.close - 1, 5))
    inner = rank(rank(inner))
    inner = ts_min(inner, 2)
    inner = ts_sum(inner, 1)
    inner = scale(log(inner))
    inner = rank(rank(inner))
    return (ts_min(rank(product(inner, 1)), 5)
            + ts_rank(delay(-1 * m.ret, 6), 5))


def _a030(m):
    s = (sign(delta(m.close, 1))
         + sign(delay(delta(m.close, 1), 1))
         + sign(delay(delta(m.close, 1), 2)))
    return safe_div((1.0 - rank(s)) * ts_sum(m.volume, 5), ts_sum(m.volume, 20))


# --- Alpha#31 ~ #40 ---
def _a031(m):
    return (rank(rank(rank(decay_linear(-1 * rank(rank(delta(m.close, 10))), 10))))
            + rank(-1 * delta(m.close, 3))
            + sign(scale(corr(m.adv(20), m.low, 12))))


def _a032(m):
    return (scale(ts_mean(m.close, 7) - m.close)
            + 20 * scale(corr(m.vwap, delay(m.close, 5), 230)))


def _a033(m):
    return rank(-1 * (1 - safe_div(m.open, m.close)))


def _a034(m):
    inner = safe_div(ts_std(m.ret, 2), ts_std(m.ret, 5))
    return rank((1 - rank(inner)) + (1 - rank(delta(m.close, 1))))


def _a035(m):
    return (ts_rank(m.volume, 32)
            * (1 - ts_rank(m.close + m.high - m.low, 16))
            * (1 - ts_rank(m.ret, 32)))


def _a036(m):
    return (2.21 * rank(corr(m.close - m.open, delay(m.volume, 1), 15))
            + 0.7 * rank(m.open - m.close)
            + 0.73 * rank(ts_rank(delay(-1 * m.ret, 6), 5))
            + rank(absv(corr(m.vwap, m.adv(20), 6)))
            + 0.6 * rank((ts_mean(m.close, 200) - m.open) * (m.close - m.open)))


def _a037(m):
    return (rank(corr(delay(m.open - m.close, 1), m.close, 200))
            + rank(m.open - m.close))


def _a038(m):
    return -1 * rank(ts_rank(m.close, 10)) * rank(safe_div(m.close, m.open))


def _a039(m):
    return (-1 * rank(delta(m.close, 7)
                      * (1 - rank(decay_linear(safe_div(m.volume, m.adv(20)), 9))))
            * (1 + rank(ts_sum(m.ret, 250))))


def _a040(m):
    return -1 * rank(ts_std(m.high, 10)) * corr(m.high, m.volume, 10)


# --- Alpha#41 ~ #50 ---
def _a041(m):
    return signedpower(m.high * m.low, 0.5) - m.vwap


def _a042(m):
    return safe_div(rank(m.vwap - m.close), rank(m.vwap + m.close))


def _a043(m):
    return (ts_rank(safe_div(m.volume, m.adv(20)), 20)
            * ts_rank(-1 * delta(m.close, 7), 8))


def _a044(m):
    return -1 * corr(m.high, rank(m.volume), 5)


def _a045(m):
    return -1 * (rank(ts_mean(delay(m.close, 5), 20))
                 * corr(m.close, m.volume, 2)
                 * rank(corr(ts_sum(m.close, 5), ts_sum(m.close, 20), 2)))


def _abc(m):
    """Alpha#46 / #49 / #51 共用的中期趋势项。Shared trend term."""
    return ((delay(m.close, 20) - delay(m.close, 10)) / 10.0
            - (delay(m.close, 10) - m.close) / 10.0)


def _a046(m):
    inner = _abc(m)
    out = (-1 * delta(m.close, 1)).where(~(inner < 0), 1.0)
    return out.where(~(inner > 0.25), -1.0)


def _a047(m):
    term = safe_div(rank(safe_div(1.0, m.close)) * m.volume, m.adv(20))
    return (term * safe_div(m.high * rank(m.high - m.close), ts_mean(m.high, 5))
            - rank(m.vwap - delay(m.vwap, 5)))


def _a048(m):
    d1 = delta(m.close, 1)
    num = m.ind(safe_div(corr(d1, delta(delay(m.close, 1), 1), 250) * d1, m.close))
    den = ts_sum(safe_div(d1, delay(m.close, 1)) ** 2, 250)
    return safe_div(num, den)


def _a049(m):
    inner = _abc(m)
    return (-1 * delta(m.close, 1)).where(~(inner < -0.1), 1.0)


def _a050(m):
    return -1 * ts_max(rank(corr(rank(m.volume), rank(m.vwap), 5)), 5)


# --- Alpha#51 ~ #60 ---
def _a051(m):
    inner = _abc(m)
    return (-1 * delta(m.close, 1)).where(~(inner < -0.05), 1.0)


def _a052(m):
    return ((-1 * ts_min(m.low, 5) + delay(ts_min(m.low, 5), 5))
            * rank((ts_sum(m.ret, 240) - ts_sum(m.ret, 20)) / 220.0)
            * ts_rank(m.volume, 5))


def _a053(m):
    inner = safe_div((m.close - m.low) - (m.high - m.close), m.close - m.low)
    return -1 * delta(inner, 9)


def _a054(m):
    return safe_div(-1 * (m.low - m.close) * m.open ** 5,
                    (m.low - m.high) * m.close ** 5)


def _a055(m):
    rng = ts_max(m.high, 12) - ts_min(m.low, 12)
    inner = safe_div(m.close - ts_min(m.low, 12), rng)
    return -1 * corr(rank(inner), rank(m.volume), 6)


def _a056(m):
    return -1 * (rank(safe_div(ts_sum(m.ret, 10), ts_sum(ts_sum(m.ret, 2), 3)))
                 * rank(m.ret * m.cap))


def _a057(m):
    return -1 * safe_div(m.close - m.vwap,
                         decay_linear(rank(ts_argmax(m.close, 30)), 2))


def _a058(m):
    return -1 * ts_rank(decay_linear(corr(m.ind(m.vwap), m.volume, 4), 8), 6)


def _a059(m):
    base = m.vwap * 0.728317 + m.vwap * (1 - 0.728317)
    return -1 * ts_rank(decay_linear(corr(m.ind(base), m.volume, 4), 16), 8)


def _a060(m):
    inner = safe_div((m.close - m.low) - (m.high - m.close), m.high - m.low) * m.volume
    return -1 * (2 * scale(rank(inner)) - scale(rank(ts_argmax(m.close, 10))))


# --- Alpha#61 ~ #70 ---
def _a061(m):
    return _num(rank(m.vwap - ts_min(m.vwap, 16))
                < rank(corr(m.vwap, m.adv(180), 18)))


def _a062(m):
    p1 = rank(corr(m.vwap, ts_sum(m.adv(20), 22), 10))
    p2 = rank(_num((rank(m.open) + rank(m.open))
                   < (rank((m.high + m.low) / 2) + rank(m.high))))
    return -1 * _num(p1 < p2)


def _a063(m):
    p1 = rank(decay_linear(delta(m.ind(m.close), 2), 8))
    p2 = rank(decay_linear(
        corr(m.vwap * 0.318108 + m.open * (1 - 0.318108),
             ts_sum(m.adv(180), 37), 14), 12))
    return -1 * (p1 - p2)


def _a064(m):
    p1 = rank(corr(ts_sum(m.open * 0.178404 + m.low * (1 - 0.178404), 13),
                   ts_sum(m.adv(120), 13), 17))
    p2 = rank(delta((m.high + m.low) / 2 * 0.178404
                    + m.vwap * (1 - 0.178404), 4))
    return -1 * _num(p1 < p2)


def _a065(m):
    p1 = rank(corr(m.open * 0.00817205 + m.vwap * (1 - 0.00817205),
                   ts_sum(m.adv(60), 9), 6))
    p2 = rank(m.open - ts_min(m.open, 14))
    return -1 * _num(p1 < p2)


def _a066(m):
    p1 = rank(decay_linear(delta(m.vwap, 4), 7))
    ratio = safe_div((m.low * 0.96633 + m.low * (1 - 0.96633)) - m.vwap,
                     m.open - (m.high + m.low) / 2)
    p2 = ts_rank(decay_linear(ratio, 11), 7)
    return -1 * (p1 + p2)


def _a067(m):
    return -1 * signedpower(
        rank(m.high - ts_min(m.high, 2)),
        rank(corr(m.ind(m.vwap), m.ind(m.adv(20)), 6)))


def _a068(m):
    p1 = ts_rank(corr(rank(m.high), rank(m.adv(15)), 9), 14)
    p2 = rank(delta(m.close * 0.518371 + m.low * (1 - 0.518371), 1))
    return -1 * _num(p1 < p2)


def _a069(m):
    return -1 * signedpower(
        rank(ts_max(delta(m.ind(m.vwap), 3), 5)),
        ts_rank(corr(m.close * 0.490655 + m.vwap * (1 - 0.490655),
                     m.adv(20), 5), 9))


def _a070(m):
    return -1 * signedpower(
        rank(delta(m.vwap, 1)),
        ts_rank(corr(m.ind(m.close), m.adv(50), 18), 18))


# --- Alpha#71 ~ #80 ---
def _a071(m):
    p1 = ts_rank(decay_linear(corr(ts_rank(m.close, 3),
                                   ts_rank(m.adv(180), 12), 18), 4), 16)
    p2 = ts_rank(decay_linear(signedpower(rank((m.low + m.open)
                                               - (m.vwap + m.vwap)), 2.0), 16), 4)
    return emax(p1, p2)


def _a072(m):
    p1 = rank(decay_linear(corr((m.high + m.low) / 2, m.adv(40), 9), 10))
    p2 = rank(decay_linear(corr(ts_rank(m.vwap, 4), ts_rank(m.volume, 19), 7), 3))
    return safe_div(p1, p2)


def _a073(m):
    p1 = rank(decay_linear(delta(m.vwap, 5), 3))
    ratio = safe_div(delta(m.open * 0.147155 + m.low * (1 - 0.147155), 2),
                     m.open * 0.147155 + m.low * (1 - 0.147155))
    p2 = ts_rank(decay_linear(-1 * ratio, 3), 17)
    return -1 * emax(p1, p2)


def _a074(m):
    p1 = rank(corr(m.close, ts_sum(m.adv(30), 37), 15))
    p2 = rank(corr(rank(m.high * 0.0261661 + m.vwap * (1 - 0.0261661)),
                   rank(m.volume), 11))
    return -1 * _num(p1 < p2)


def _a075(m):
    p1 = rank(corr(m.vwap, m.volume, 4))
    p2 = rank(corr(rank(m.low), rank(m.adv(50)), 12))
    return _num(p1 < p2)


def _a076(m):
    p1 = rank(decay_linear(delta(m.vwap, 1), 12))
    p2 = ts_rank(decay_linear(ts_rank(corr(m.ind(m.low), m.adv(81), 8), 20), 17), 19)
    return -1 * emax(p1, p2)


def _a077(m):
    p1 = rank(decay_linear(((m.high + m.low) / 2 + m.high)
                           - (m.vwap + m.high), 20))
    p2 = rank(decay_linear(corr((m.high + m.low) / 2, m.adv(40), 3), 6))
    return emin(p1, p2)


def _a078(m):
    p1 = rank(corr(ts_sum(m.low * 0.352233 + m.vwap * (1 - 0.352233), 20),
                   ts_sum(m.adv(40), 20), 7))
    p2 = rank(corr(rank(m.vwap), rank(m.volume), 6))
    return signedpower(p1, p2)


def _a079(m):
    p1 = rank(delta(m.ind(m.close * 0.60733 + m.open * (1 - 0.60733)), 1))
    p2 = rank(corr(ts_rank(m.vwap, 4), ts_rank(m.adv(150), 9), 15))
    return -1 * _num(p1 < p2)


def _a080(m):
    base = rank(sign(delta(m.ind(m.open * 0.868128 + m.high * (1 - 0.868128)), 4)))
    expo = ts_rank(corr(m.high, m.adv(10), 5), 6)
    return -1 * signedpower(base, expo)


# --- Alpha#81 ~ #90 ---
def _a081(m):
    inner = rank(signedpower(rank(corr(m.vwap, ts_sum(m.adv(10), 50), 8)), 4.0))
    p1 = rank(log(product(inner, 15)))
    p2 = rank(corr(rank(m.vwap), rank(m.volume), 5))
    return -1 * _num(p1 < p2)


def _a082(m):
    p1 = rank(decay_linear(delta(m.open, 1), 15))
    p2 = ts_rank(decay_linear(
        corr(m.ind(m.volume), m.open * 0.634196 + m.open * (1 - 0.634196), 17), 7), 13)
    return -1 * emin(p1, p2)


def _a083(m):
    hl = safe_div(m.high - m.low, ts_mean(m.close, 5))
    return safe_div(rank(delay(hl, 2)) * rank(rank(m.volume)),
                    safe_div(hl, m.vwap - m.close))


def _a084(m):
    return signedpower(ts_rank(m.vwap - ts_max(m.vwap, 15), 21),
                       delta(m.close, 5))


def _a085(m):
    p1 = rank(corr(m.high * 0.876703 + m.close * (1 - 0.876703), m.adv(30), 10))
    p2 = rank(corr(ts_rank((m.high + m.low) / 2, 4), ts_rank(m.volume, 10), 7))
    return signedpower(p1, p2)


def _a086(m):
    p1 = ts_rank(corr(m.close, ts_sum(m.adv(20), 15), 6), 20)
    p2 = rank((m.open + m.close) - (m.vwap + m.open))
    return -1 * _num(p1 < p2)


def _a087(m):
    p1 = rank(decay_linear(delta(m.close * 0.369701 + m.vwap * (1 - 0.369701), 2), 3))
    p2 = ts_rank(decay_linear(absv(corr(m.ind(m.adv(81)), m.close, 13)), 5), 14)
    return -1 * emax(p1, p2)


def _a088(m):
    p1 = rank(decay_linear((rank(m.open) + rank(m.low))
                           - (rank(m.high) + rank(m.close)), 8))
    p2 = ts_rank(decay_linear(corr(ts_rank(m.close, 8),
                                   ts_rank(m.adv(60), 21), 8), 7), 3)
    return emin(p1, p2)


def _a089(m):
    p1 = ts_rank(decay_linear(
        corr(m.low * 0.967285 + m.low * (1 - 0.967285), m.adv(10), 7), 6), 4)
    p2 = ts_rank(decay_linear(delta(m.ind(m.vwap), 3), 10), 15)
    return p1 - p2


def _a090(m):
    return -1 * signedpower(
        rank(m.close - ts_max(m.close, 5)),
        ts_rank(corr(m.ind(m.adv(40)), m.low, 5), 3))


# --- Alpha#91 ~ #101 ---
def _a091(m):
    p1 = ts_rank(decay_linear(decay_linear(corr(m.ind(m.close), m.volume, 10), 16), 4), 5)
    p2 = rank(decay_linear(corr(m.vwap, m.adv(30), 4), 3))
    return -1 * (p1 - p2)


def _a092(m):
    p1 = ts_rank(decay_linear(_num(((m.high + m.low) / 2 + m.close)
                                  < (m.low + m.open)), 15), 19)
    p2 = ts_rank(decay_linear(corr(rank(m.low), rank(m.adv(30)), 8), 7), 7)
    return emin(p1, p2)


def _a093(m):
    p1 = ts_rank(decay_linear(corr(m.ind(m.vwap), m.adv(81), 17), 20), 8)
    p2 = rank(decay_linear(delta(m.close * 0.524434 + m.vwap * (1 - 0.524434), 3), 16))
    return safe_div(p1, p2)


def _a094(m):
    return -1 * signedpower(
        rank(m.vwap - ts_min(m.vwap, 12)),
        ts_rank(corr(ts_rank(m.vwap, 20), ts_rank(m.adv(60), 4), 18), 3))


def _a095(m):
    inner = rank(corr(ts_sum((m.high + m.low) / 2, 19),
                      ts_sum(m.adv(40), 19), 13))
    return _num(rank(m.open - ts_min(m.open, 12))
                < ts_rank(signedpower(inner, 5.0), 12))


def _a096(m):
    p1 = ts_rank(decay_linear(corr(rank(m.vwap), rank(m.volume), 4), 4), 8)
    p2 = ts_rank(decay_linear(
        ts_argmax(corr(ts_rank(m.close, 7), ts_rank(m.adv(60), 4), 4), 13), 14), 13)
    return -1 * emax(p1, p2)


def _a097(m):
    base = m.low * 0.721001 + m.vwap * (1 - 0.721001)
    p1 = rank(decay_linear(delta(m.ind(base), 3), 20))
    p2 = ts_rank(decay_linear(ts_rank(corr(ts_rank(m.low, 8),
                                           ts_rank(m.adv(60), 17), 5), 19), 16), 7)
    return -1 * (p1 - p2)


def _a098(m):
    p1 = rank(decay_linear(corr(m.vwap, ts_sum(m.adv(5), 26), 5), 7))
    p2 = rank(decay_linear(ts_rank(ts_argmin(
        corr(rank(m.open), rank(m.adv(15)), 21), 9), 7), 8))
    return p1 - p2


def _a099(m):
    p1 = rank(corr(ts_sum((m.high + m.low) / 2, 20), ts_sum(m.adv(60), 20), 9))
    p2 = rank(corr(m.low, m.volume, 6))
    return -1 * _num(p1 < p2)


def _a100(m):
    base = safe_div((m.close - m.low) - (m.high - m.close), m.high - m.low) * m.volume
    p1 = 1.5 * scale(m.ind(m.ind(rank(base))))
    p2 = scale(m.ind(corr(m.close, rank(m.adv(20)), 5) - rank(ts_argmin(m.close, 30))))
    return -1 * (p1 - p2) * safe_div(m.volume, m.adv(20))


def _a101(m):
    return safe_div(m.close - m.open, (m.high - m.low) + 0.001)


_FACTORS = (
    ("a101_001", _a001), ("a101_002", _a002), ("a101_003", _a003),
    ("a101_004", _a004), ("a101_005", _a005), ("a101_006", _a006),
    ("a101_007", _a007), ("a101_008", _a008), ("a101_009", _a009),
    ("a101_010", _a010), ("a101_011", _a011), ("a101_012", _a012),
    ("a101_013", _a013), ("a101_014", _a014), ("a101_015", _a015),
    ("a101_016", _a016), ("a101_017", _a017), ("a101_018", _a018),
    ("a101_019", _a019), ("a101_020", _a020), ("a101_021", _a021),
    ("a101_022", _a022), ("a101_023", _a023), ("a101_024", _a024),
    ("a101_025", _a025), ("a101_026", _a026), ("a101_027", _a027),
    ("a101_028", _a028), ("a101_029", _a029), ("a101_030", _a030),
    ("a101_031", _a031), ("a101_032", _a032), ("a101_033", _a033),
    ("a101_034", _a034), ("a101_035", _a035), ("a101_036", _a036),
    ("a101_037", _a037), ("a101_038", _a038), ("a101_039", _a039),
    ("a101_040", _a040), ("a101_041", _a041), ("a101_042", _a042),
    ("a101_043", _a043), ("a101_044", _a044), ("a101_045", _a045),
    ("a101_046", _a046), ("a101_047", _a047), ("a101_048", _a048),
    ("a101_049", _a049), ("a101_050", _a050), ("a101_051", _a051),
    ("a101_052", _a052), ("a101_053", _a053), ("a101_054", _a054),
    ("a101_055", _a055), ("a101_056", _a056), ("a101_057", _a057),
    ("a101_058", _a058), ("a101_059", _a059), ("a101_060", _a060),
    ("a101_061", _a061), ("a101_062", _a062), ("a101_063", _a063),
    ("a101_064", _a064), ("a101_065", _a065), ("a101_066", _a066),
    ("a101_067", _a067), ("a101_068", _a068), ("a101_069", _a069),
    ("a101_070", _a070), ("a101_071", _a071), ("a101_072", _a072),
    ("a101_073", _a073), ("a101_074", _a074), ("a101_075", _a075),
    ("a101_076", _a076), ("a101_077", _a077), ("a101_078", _a078),
    ("a101_079", _a079), ("a101_080", _a080), ("a101_081", _a081),
    ("a101_082", _a082), ("a101_083", _a083), ("a101_084", _a084),
    ("a101_085", _a085), ("a101_086", _a086), ("a101_087", _a087),
    ("a101_088", _a088), ("a101_089", _a089), ("a101_090", _a090),
    ("a101_091", _a091), ("a101_092", _a092), ("a101_093", _a093),
    ("a101_094", _a094), ("a101_095", _a095), ("a101_096", _a096),
    ("a101_097", _a097), ("a101_098", _a098), ("a101_099", _a099),
    ("a101_100", _a100), ("a101_101", _a101),
)


def compute_alpha101(matrices: dict):
    """逐个产出 ``(因子名, 日频因子表)``。Yield ``(name, daily factor)``.

    Parameters
    ----------
    matrices : dict
        含 open/high/low/close/volume/amount/vwap/returns/turnover_rate/
        total_mv/is_st 的「日期 × 股票」宽表；缺失的价量字段用 close 推导，
        缺 ``total_mv`` 时跳过 Alpha#56（打印原因）。
    """
    m = _Ctx(matrices)
    for name, func in _FACTORS:
        if name == "a101_056" and m.cap is None:
            print("[alpha101] 跳过 a101_056：缺少 total_mv（cap），"
                  "无法计算 rank(returns * cap)")
            continue
        out = _as_frame(func(m), m.close)
        yield name, clean(out.astype(float))
