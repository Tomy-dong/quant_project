# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 16:16:57 2025

@author: lich5
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import boxcox

INDUSTRY_FILE = Path(__file__).resolve().parent.parent / "data" / "stk_company_info.csv"


def _my_boxcox(r_data, return_lambda=False):
    """对非缺失正值做 Box-Cox 变换；可选返回估计的 lambda。"""
    row_data = r_data.to_numpy(dtype=float, copy=True)
    nonempty = np.where(~np.isnan(row_data))[0]
    a, lam = boxcox(row_data[nonempty])
    row_data[nonempty] = a
    if return_lambda:
        return row_data, lam
    return row_data


@lru_cache(maxsize=1)
def _industry_map():
    """Baostock 行业快照：sh.600000 -> 600000.SH，取证监会行业名称。"""
    df = pd.read_csv(INDUSTRY_FILE, dtype=str).dropna(subset=["code", "industry"])
    code = df["code"].str[3:] + "." + df["code"].str[:2].str.upper()
    return pd.Series(df["industry"].values, index=code.values).groupby(level=0).last()


def industry_neutralize(data):
    """逐期截面按行业去均值；无行业归属的股票归入同一组。"""
    industry = _industry_map().reindex(data.columns).fillna("未分类")
    means = data.T.groupby(industry.values).transform("mean").T
    return data - means


def preprocess(data):
    """逐期截面 MAD 缩尾、行业去均值与 Z-score；输入为日期×股票宽表。"""
    median = data.median(axis=1)
    mad = data.sub(median, axis=0).abs().median(axis=1)
    lower = (median - 3 * 1.4826 * mad).where(mad > 0)
    upper = (median + 3 * 1.4826 * mad).where(mad > 0)
    data = data.clip(lower=lower, upper=upper, axis=0)
    data = industry_neutralize(data)
    return data.sub(data.mean(axis=1), axis=0).div(
        data.std(axis=1, ddof=1).replace(0, np.nan), axis=0)
