# -*- coding: utf-8 -*-
"""宽表转换公共函数（02a / 02b / 02 共用）。

把 data/raw/ 下的长表 CSV pivot 成宽表（index=date, columns=code），
输出到 data/，供因子计算与回测直接读取。

同时计算派生字段：
  adj_close  = close × backAdjustFactor          （后复权收盘价，动量/波动率因子用）
  total_mv   = 不复权 close × totalShare           （三因子 SMB 分组 + 容量筛选）
  circ_mv    = 不复权 close × liqaShare            （流通市值）
"""

import glob
from pathlib import Path
import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]  # scripts/ -> 仓库根目录
RAW_DIR = PACKAGE_ROOT / "data" / "raw"
OUT_DIR = PACKAGE_ROOT / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 长表字段 -> 宽表文件名前缀
FIELD_MAP = [
    ("open", "open"),
    ("high", "high"),
    ("low", "low"),
    ("close", "close"),
    ("volume", "vol"),
    ("amount", "amount"),
    ("turn", "turnover_rate"),
    ("pctChg", "pctChg"),
    ("isST", "is_st"),
    ("backAdjustFactor", "adj_factor"),
    ("peTTM", "pe_ttm"),
    ("pbMRQ", "pb"),
]


def norm_code(c):
    """Baostock 代码 -> 宽表列名：sh.600000 -> 600000.SH，sz.000001 -> 000001.SZ"""
    m, s = str(c).split(".")
    return f"{s}.{m.upper()}"


def load_raw():
    """读入 data/raw/ 下所有长表，拼成一个大 DataFrame。"""
    files = sorted(glob.glob(str(RAW_DIR / "*.csv")))
    print(f"读入 {len(files)} 只股票的长表...")
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, dtype={"code": str})
            if not df.empty:
                frames.append(df)
        except Exception as e:
            print(f"  跳过 {Path(f).name}: {e}")
    if not frames:
        raise RuntimeError("data/raw/ 下没有任何文件，请先跑 01a/01b/01")

    long_df = pd.concat(frames, ignore_index=True)
    long_df = long_df.dropna(subset=["code", "date"])
    long_df["date"] = pd.to_datetime(long_df["date"], format="%Y-%m-%d", errors="coerce")
    long_df = long_df.dropna(subset=["date"])
    long_df["code"] = long_df["code"].map(norm_code)
    return long_df


def pivot_market(long_df, market):
    """market = 'sh' 或 'sz'。把长表 pivot 成该市场的所有宽表。"""
    suffix = market.upper()
    sub = long_df[long_df["code"].str.endswith("." + suffix)].copy()
    print(f"\n=== {suffix} 市 {sub['code'].nunique()} 只，{len(sub)} 行 ===")
    if sub.empty:
        print(f"  警告：{suffix} 市无数据，跳过")
        return

    for field, prefix in FIELD_MAP:
        if field not in sub.columns:
            continue
        wide = sub.pivot(index="date", columns="code", values=field).sort_index()
        wide.index.name = "date"
        wide = wide.apply(pd.to_numeric, errors="coerce")
        out = OUT_DIR / f"{prefix}_{market}.csv"
        wide.to_csv(out, encoding="utf-8")
        print(f"  {out.name}: {wide.shape}")

    # 派生：后复权价 = close × backAdjustFactor
    if {"close", "backAdjustFactor"}.issubset(sub.columns):
        sub["close"] = pd.to_numeric(sub["close"], errors="coerce")
        sub["backAdjustFactor"] = pd.to_numeric(
            sub["backAdjustFactor"], errors="coerce"
        )
        sub["adj_close"] = sub["close"] * sub["backAdjustFactor"]
        wide = sub.pivot(index="date", columns="code", values="adj_close").sort_index()
        wide.index.name = "date"
        wide.to_csv(OUT_DIR / f"adj_close_{market}.csv", encoding="utf-8")
        print(f"  adj_close_{market}.csv: {wide.shape}")

    # 派生：总市值 = 不复权 close × totalShare
    if {"close", "totalShare"}.issubset(sub.columns):
        sub["totalShare"] = pd.to_numeric(sub["totalShare"], errors="coerce")
        sub["total_mv"] = sub["close"] * sub["totalShare"]
        wide = sub.pivot(index="date", columns="code", values="total_mv").sort_index()
        wide.index.name = "date"
        wide.to_csv(OUT_DIR / f"total_mv_{market}.csv", encoding="utf-8")
        print(f"  total_mv_{market}.csv: {wide.shape}")

    # 派生：流通市值 = 不复权 close × liqaShare
    if {"close", "liqaShare"}.issubset(sub.columns):
        sub["liqaShare"] = pd.to_numeric(sub["liqaShare"], errors="coerce")
        sub["circ_mv"] = sub["close"] * sub["liqaShare"]
        wide = sub.pivot(index="date", columns="code", values="circ_mv").sort_index()
        wide.index.name = "date"
        wide.to_csv(OUT_DIR / f"circ_mv_{market}.csv", encoding="utf-8")
        print(f"  circ_mv_{market}.csv: {wide.shape}")
