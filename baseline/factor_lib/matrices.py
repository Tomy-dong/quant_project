# -*- coding: utf-8 -*-
"""日频矩阵构建 / Daily matrix construction.

把 data/ 下按市场拆分的 CSV 拼成「日期 × 股票」的宽表，并统一复权口径：

- 价格类（open/high/low/close）乘以 adj_factor，得到后复权价格；
- 成交量调整为 vol / adj_factor，使 vwap = amount / volume 与后复权价格同尺度；
- amount、turnover_rate、total_mv、is_st 保持原始值；
- 不填充停牌 NaN，交由各算子的 min_periods 处理。

Prices are back-adjusted by adj_factor; volume is divided by adj_factor so that
vwap = amount / volume shares the same scale as prices. Suspended days are left
as NaN.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data_io import load_benchmark

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

PRICE_FIELDS = ("open", "high", "low", "close")


def _read_field(field: str) -> pd.DataFrame | None:
    """读取并合并沪深两市的同一字段。Read and concat one field across markets."""
    frames = []
    for market in ("sh", "sz"):
        path = DATA_DIR / f"{field}_{market}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col=0, parse_dates=True, dtype=np.float32)
        frames.append(df)
    if not frames:
        return None
    out = pd.concat(frames, axis=1)
    out = out.loc[~out.index.duplicated(keep="last")].sort_index()
    return out


def build_matrices(start: str = "2015-01-01",
                   end: str = "2025-12-31") -> tuple[dict, pd.DatetimeIndex]:
    """构建因子计算所需的日频矩阵字典与月末日期。

    Returns
    -------
    matrices : dict[str, pd.DataFrame]
        键为 close/open/high/low/volume/amount/vwap/returns/turnover/
        total_mv/is_st；行是交易日，列是股票代码。
    month_dates : pd.DatetimeIndex
        每月最后一个交易日。
    """
    close = _read_field("close")
    if close is None:
        raise FileNotFoundError(f"找不到 close 行情：{DATA_DIR}")

    close = close.loc[(close.index >= start) & (close.index <= end)]
    index = close.index
    columns = close.columns

    adj = _read_field("adj_factor")
    if adj is None:
        adj = pd.DataFrame(1.0, index=index, columns=columns)
    adj = adj.reindex(index=index, columns=columns).astype(np.float32)
    adj = adj.fillna(1.0).replace(0.0, 1.0).clip(lower=1e-6)

    adj_close = close * adj

    matrices: dict[str, pd.DataFrame] = {"close": adj_close}
    for field in PRICE_FIELDS:
        if field == "close":
            continue
        raw = _read_field(field)
        if raw is None:
            matrices[field] = adj_close.copy()
            continue
        raw = raw.reindex(index=index, columns=columns).astype(np.float32)
        matrices[field] = raw * adj

    vol = _read_field("vol")
    vol = vol.reindex(index=index, columns=columns).astype(np.float32)
    volume = vol / adj
    matrices["volume"] = volume.replace(0.0, np.nan)

    amount = _read_field("amount")
    amount = amount.reindex(index=index, columns=columns).astype(np.float32)
    matrices["amount"] = amount

    # vwap 与后复权价格同尺度 / vwap on the same scale as adjusted prices
    matrices["vwap"] = amount / matrices["volume"]

    matrices["returns"] = adj_close.pct_change(fill_method=None)

    for field in ("turnover_rate", "total_mv", "is_st"):
        df = _read_field(field)
        if df is not None:
            matrices[field] = df.reindex(index=index, columns=columns).astype(np.float32)

    month_dates = pd.DatetimeIndex(
        pd.Series(index).groupby(index.to_period("M")).max())
    return matrices, month_dates


def build_benchmark(start: str = "2015-01-01",
                    end: str = "2025-12-31",
                    index: str = "csi300") -> dict:
    """基准日线矩阵（供 Alpha191 的 075/181/182 与清华 Beta 类因子使用）。
    Benchmark daily matrices for benchmark-dependent factors.
    """
    bench = load_benchmark(start, end, index=index)
    out = {}
    for field in ("open", "high", "low", "close"):
        if field in bench.columns:
            out[field] = bench[[field]].rename(columns={field: "benchmark"})
    return out
