# -*- coding: utf-8 -*-
"""QLib Alpha158 量价因子（向量化实现）/ QLib Alpha158 price-volume factors.

158 个因子 = 9 个 K 线形态 + 4 个价格位置 + 29 个滚动算子 × 5 个窗口。
全部按 QLib ``Alpha158`` 的标准定义实现，只是把逐股票计算改成整表向量化。

158 factors = 9 K-bar shapes + 4 price positions + 29 rolling operators x
5 windows, following the QLib Alpha158 definitions, vectorised across symbols.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ops import _minp, safe_div, ts_argmax, ts_argmin, ts_rank

WINDOWS = (5, 10, 20, 30, 60)


def _roll(x: pd.DataFrame, n: int):
    return x.rolling(n, min_periods=_minp(n))


def compute_alpha158(matrices: dict):
    """逐个产出 (因子名, 日频因子表)。Yield (name, daily factor) one by one."""
    open_ = matrices["open"]
    high = matrices["high"]
    low = matrices["low"]
    close = matrices["close"]
    vwap = matrices["vwap"]
    vol = matrices["volume"]

    rng = (high - low).replace(0, np.nan)
    upper = np.maximum(open_, close)
    lower = np.minimum(open_, close)

    # --- 9 个 K 线形态 / 9 K-bar shapes ---
    yield "a158_kbar_kmid", (close - open_) / open_.replace(0, np.nan)
    yield "a158_kbar_klen", (high - low) / open_.replace(0, np.nan)
    yield "a158_kbar_kmid2", safe_div(close - open_, rng)
    yield "a158_kbar_kup", (high - upper) / open_.replace(0, np.nan)
    yield "a158_kbar_kup2", safe_div(high - upper, rng)
    yield "a158_kbar_klow", (lower - low) / open_.replace(0, np.nan)
    yield "a158_kbar_klow2", safe_div(lower - low, rng)
    yield "a158_kbar_ksft", (2 * close - high - low) / open_.replace(0, np.nan)
    yield "a158_kbar_ksft2", safe_div(2 * close - high - low, rng)

    # --- 4 个价格位置 / 4 price positions ---
    yield "a158_price_open", open_ / close.replace(0, np.nan)
    yield "a158_price_high", high / close.replace(0, np.nan)
    yield "a158_price_low", low / close.replace(0, np.nan)
    yield "a158_price_vwap", vwap / close.replace(0, np.nan)

    # --- 滚动基础量 / rolling building blocks ---
    ret1 = close.pct_change(fill_method=None)
    up = (close > close.shift(1)).astype(float)
    dn = (close < close.shift(1)).astype(float)
    diff = close.diff()
    abs_diff = diff.abs()
    pos_diff = diff.clip(lower=0.0)
    neg_diff = (-diff).clip(lower=0.0)

    dvol = vol.diff()
    abs_dvol = dvol.abs()
    pos_dvol = dvol.clip(lower=0.0)
    neg_dvol = (-dvol).clip(lower=0.0)
    abs_ret_vol = ret1.abs() * vol

    log_vol = np.log(vol.where(vol > 0))
    dlog_vol = log_vol.diff()

    t_seq = pd.Series(np.arange(len(close), dtype=float), index=close.index)

    for w in WINDOWS:
        max_c = _roll(close, w).max()
        min_c = _roll(close, w).min()
        mean_c = _roll(close, w).mean()
        std_c = _roll(close, w).std()

        yield f"a158_roc{w}", close / close.shift(w) - 1
        yield f"a158_ma{w}", mean_c / close.replace(0, np.nan)
        yield f"a158_std{w}", std_c / close.replace(0, np.nan)
        yield f"a158_max{w}", max_c / close.replace(0, np.nan)
        yield f"a158_min{w}", min_c / close.replace(0, np.nan)
        yield f"a158_qtlu{w}", _roll(close, w).quantile(0.8) / close.replace(0, np.nan)
        yield f"a158_qtld{w}", _roll(close, w).quantile(0.2) / close.replace(0, np.nan)
        yield f"a158_rank{w}", ts_rank(close, w)
        yield f"a158_rsv{w}", safe_div(close - min_c, max_c - min_c)

        # 滚动回归斜率 / R² / 残差 / rolling regression
        cov_t = _roll(close, w).cov(t_seq)
        var_t = (w * (w * w - 1) / 12.0) * w / (w - 1.0)
        beta = cov_t / var_t
        corr_t = _roll(close, w).corr(t_seq)
        resid = close - mean_c - beta * ((w - 1) / 2.0)
        yield f"a158_beta{w}", beta / close.replace(0, np.nan)
        yield f"a158_rsqr{w}", (corr_t ** 2) / close.replace(0, np.nan)
        yield f"a158_resi{w}", resid / close.replace(0, np.nan)

        imax = ts_argmax(high, w) / w
        imin = ts_argmin(low, w) / w
        yield f"a158_imax{w}", imax
        yield f"a158_imin{w}", imin
        yield f"a158_imxd{w}", imax - imin

        yield f"a158_corr{w}", _roll(close, w).corr(vol)
        yield f"a158_cord{w}", _roll(ret1, w).corr(dlog_vol)

        cntp = _roll(up, w).mean()
        cntn = _roll(dn, w).mean()
        yield f"a158_cntp{w}", cntp
        yield f"a158_cntn{w}", cntn
        yield f"a158_cntd{w}", cntp - cntn

        sum_abs = _roll(abs_diff, w).sum()
        sump = safe_div(_roll(pos_diff, w).sum(), sum_abs)
        sumn = safe_div(_roll(neg_diff, w).sum(), sum_abs)
        yield f"a158_sump{w}", sump
        yield f"a158_sumn{w}", sumn
        yield f"a158_sumd{w}", sump - sumn

        mean_vol = _roll(vol, w).mean()
        yield f"a158_vma{w}", safe_div(mean_vol, vol)
        yield f"a158_vstd{w}", safe_div(_roll(vol, w).std(), mean_vol)
        mean_arv = _roll(abs_ret_vol, w).mean()
        yield f"a158_wvma{w}", safe_div(_roll(abs_ret_vol, w).std(), mean_arv)

        sum_abs_dvol = _roll(abs_dvol, w).sum()
        vsump = safe_div(_roll(pos_dvol, w).sum(), sum_abs_dvol)
        vsumn = safe_div(_roll(neg_dvol, w).sum(), sum_abs_dvol)
        yield f"a158_vsump{w}", vsump
        yield f"a158_vsumn{w}", vsumn
        yield f"a158_vsumd{w}", vsump - vsumn


def count_factors() -> int:
    """Alpha158 因子个数。Number of Alpha158 factors."""
    return 13 + 29 * len(WINDOWS)
