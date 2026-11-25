# -*- coding: utf-8 -*-
"""清华因子白皮书（民生 56 因子）中的量价子集，共 22 个。

对应因子：size、beta、downside beta、idvol、vol、idskew、skew12、coskew12、
turn、std_turn、volumed、std_dvol、illq、LM、retnmax、sharechg、age、
mom12、mom6、momchg、imom、lagretn。

说明
----
1. 白皮书部分因子依赖 FF3 因子或分析师数据，这里只用日频行情与市值，
   个别口径做了近似（IDVOL 用单因子市场模型残差、IDSKEW 用残差偏度、
   LM 用换手与零成交天数构造），已在代码中标注。
2. 月度类因子直接在月末面板上计算；日频类因子先按日滚动，再取月末。

Price-volume subset of the Minsheng 56-factor white paper (22 factors). Some
definitions are approximated using only daily market data, as noted inline.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils.data_standardize import _industry_map

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
TRADING_DAYS = 252


def _ipo_series(columns) -> pd.Series:
    """上市日期，缺失记为很久以前。IPO dates aligned to stock columns."""
    basic = pd.read_csv(DATA_DIR / "sse_stock_basic.csv", dtype=str)
    ipo = pd.to_datetime(basic.set_index("code")["ipoDate"], errors="coerce")
    ipo.index = ipo.index.str[3:] + "." + ipo.index.str[:2].str.upper()
    ipo = ipo.reindex(columns).fillna(pd.Timestamp("2000-01-01"))
    return ipo


def _industry_align(columns) -> pd.Series:
    """行业快照，缺失归入“未分类”。Industry snapshot aligned to columns."""
    return _industry_map().reindex(columns).fillna("未分类")


def _roll(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=max(3, int(np.ceil(0.6 * n))))


def compute_tsinghua(matrices: dict, month_dates: pd.DatetimeIndex):
    """逐个产出 (因子名, 月末面板)。Yield (name, month-end panel)."""
    close = matrices["close"]
    ret_d = matrices["returns"]
    turn = matrices.get("turnover_rate")
    amount = matrices["amount"]
    vol = matrices["volume"]
    total_mv = matrices.get("total_mv")
    price = close.reindex(month_dates)
    ret_m = price.pct_change(fill_method=None)

    # 基准月收益 / benchmark monthly returns
    bench = pd.read_csv(DATA_DIR / "csi300_daily.csv", index_col=0, parse_dates=True)
    bm_close = bench["close"].astype(float).reindex(close.index)
    bm_m = bm_close.reindex(month_dates).pct_change(fill_method=None)

    def sample(daily: pd.DataFrame) -> pd.DataFrame:
        return daily.reindex(month_dates)

    # --- 日频类 / daily-based ---
    if turn is not None:
        yield "ts_turn", sample(_roll(turn, 21).mean())
        yield "ts_std_turn", sample(_roll(turn, 21).std())
        zero_turn = (turn.fillna(0) <= 0).astype(float)
        yield "ts_lm", sample(_roll(turn, 21).mean()) / (
            1.0 + zero_turn.rolling(TRADING_DAYS, min_periods=60).sum().reindex(month_dates))

    yield "ts_vol", sample(_roll(ret_d, 21).std())
    yield "ts_retnmax", sample(_roll(ret_d, 21).max())
    yield "ts_illq", sample(_roll(ret_d.abs() / amount.replace(0, np.nan), 21).mean())

    mean_v21 = _roll(vol, 21).mean()
    mean_v252 = _roll(vol, TRADING_DAYS).mean()
    yield "ts_volumed", sample(np.log(mean_v21.where(mean_v21 > 0)
                                      / mean_v252.where(mean_v252 > 0)))
    dvol = vol.pct_change(fill_method=None)
    yield "ts_std_dvol", sample(_roll(dvol, 21).std())

    # 流通股本变化：股本 = 成交量 / 换手率 / 流通股本变化
    if turn is not None:
        shares = vol / (turn.replace(0, np.nan) / 100.0)
        yield "ts_sharechg", sample(shares / shares.shift(TRADING_DAYS) - 1)

    # --- 月末类 / month-end based ---
    if total_mv is not None:
        mv = total_mv.reindex(month_dates)
        yield "ts_size", np.log(mv.where(mv > 0))

    # beta / downside beta / idvol / idskew / coskew12
    n = 12
    rm = bm_m
    mean_r = _roll(ret_m, n).mean()
    mean_m = rm.rolling(n, min_periods=8).mean()
    std_r = _roll(ret_m, n).std()
    std_m = rm.rolling(n, min_periods=8).std()
    cov_rm = _roll(ret_m, n).cov(rm)
    var_m = rm.rolling(n, min_periods=8).var()
    beta = cov_rm.div(var_m, axis=0)
    corr_rm = _roll(ret_m, n).corr(rm)

    yield "ts_beta", beta
    # 下行 beta：只用基准下跌的月份做回归 / downside beta on market-down months
    down = (rm < 0).astype(float)
    r_d = ret_m.mul(down, axis=0)
    m_d = rm * down
    cnt = down.rolling(n, min_periods=3).sum()
    sxy = _roll(ret_m.mul(rm, axis=0).mul(down, axis=0), n).sum()
    sx = m_d.rolling(n, min_periods=3).sum()
    sy = r_d.rolling(n, min_periods=3).sum()
    sxx = (rm ** 2 * down).rolling(n, min_periods=3).sum()
    num = sxy.mul(cnt, axis=0).sub(sy.mul(sx, axis=0), axis=0)
    den = sxx.mul(cnt, axis=0).sub(sx.pow(2), axis=0)
    yield "ts_downside_beta", num.div(den.replace(0, np.nan), axis=0)

    yield "ts_idvol", std_r.mul(np.sqrt((1 - corr_rm ** 2).clip(lower=0.0)))
    resid = ret_m.sub(beta.mul(rm, axis=0))
    yield "ts_idskew", _roll(resid, n).skew()
    yield "ts_skew12", _roll(ret_m, n).skew()

    # coskew12: E[(r-μr)(rm-μm)^2] / (σr σm^2)，用样本矩展开
    # coskew12 via sample-moment expansion
    e_rm2 = _roll(ret_m.mul(rm ** 2, axis=0), n).mean()
    e_rm = _roll(ret_m.mul(rm, axis=0), n).mean()
    e_m2 = (rm ** 2).rolling(n, min_periods=8).mean()
    num_cs = (e_rm2.sub(mean_r.mul(e_m2, axis=0))
              .sub(e_rm.mul(mean_m * 2.0, axis=0))
              .add(mean_r.mul(mean_m.pow(2) * 2.0, axis=0)))
    yield "ts_coskew12", num_cs.div(
        std_r.mul(std_m ** 2, axis=0).replace(0, np.nan))

    # 动量类 / momentum
    mom12 = price.shift(1) / price.shift(12) - 1
    mom6 = price.shift(1) / price.shift(6) - 1
    yield "ts_mom12", mom12
    yield "ts_mom6", mom6
    yield "ts_momchg", mom6 - mom12
    yield "ts_lagretn", price / price.shift(1) - 1

    industry = _industry_align(price.columns)
    ind_mean = mom12.T.groupby(industry.values).transform("mean").T
    yield "ts_imom", mom12 - ind_mean

    # 上市月龄 / months since IPO
    ipo = _ipo_series(price.columns)
    age = pd.DataFrame(
        [(d - ipo).dt.days.to_numpy() / 30.44 for d in month_dates],
        index=month_dates, columns=price.columns)
    yield "ts_age", age.clip(lower=0.0)


def count_factors() -> int:
    """因子个数。Number of factors."""
    return 22
