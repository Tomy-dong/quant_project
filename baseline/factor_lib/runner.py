# -*- coding: utf-8 -*-
"""因子库运行时：逐因子计算 → 月末采样 → 落盘。

四个家族：alpha191 / alpha158 / alpha101 / tsinghua。日频因子逐个计算后
立即取月末截面并以 float32 存成 npz（键为因子名），避免同时保留约 470 个
日频矩阵（约 27 GB）。月度类家族直接产出月末面板。

Each factor is computed on the daily matrices, immediately sampled at month
ends and stored as float32, so the ~470 daily matrices never coexist in memory.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .alpha158 import compute_alpha158
from .ops import clean
from .tsinghua import compute_tsinghua

POOL_DIR = Path(__file__).resolve().parents[2] / "results" / "factor_pool"
FAMILIES = ("alpha191", "alpha158", "alpha101", "tsinghua")


# ===================== 家族迭代器 / Family iterators =====================
def _iter_alpha191(matrices: dict, benchmark: dict | None):
    from .alpha191pkg.alpha191_formulas import Alpha191Engine
    fields = {k: matrices[k] for k in
              ("open", "high", "low", "close", "volume", "amount", "vwap")}
    engine = Alpha191Engine(fields, benchmark)
    for i in range(1, 192):
        name = f"alpha_{i:03d}"
        method = getattr(engine, f"alpha{i:03d}", None)
        if method is None:
            continue
        try:
            fac = method()
        except Exception as exc:  # noqa: BLE001
            print(f"  [alpha191] {name} 计算失败：{type(exc).__name__}: {exc}")
            continue
        if fac is None:
            print(f"  [alpha191] {name} 依赖外部数据，跳过")
            continue
        yield name, fac


def _iter_alpha101(matrices: dict):
    from .alpha101 import compute_alpha101
    yield from compute_alpha101(matrices)


def iter_family(family: str, matrices: dict, month_dates: pd.DatetimeIndex,
                benchmark: dict | None = None):
    """产出 (因子名, 日频或月频因子表)。Yield (name, factor table)."""
    if family == "alpha191":
        yield from _iter_alpha191(matrices, benchmark)
    elif family == "alpha158":
        yield from compute_alpha158(matrices)
    elif family == "alpha101":
        yield from _iter_alpha101(matrices)
    elif family == "tsinghua":
        yield from compute_tsinghua(matrices, month_dates)
    else:
        raise ValueError(f"未知因子家族：{family}")


# ===================== 采样与落盘 / Sampling & storage =====================
def sample_panel(fac, month_dates: pd.DatetimeIndex,
                 columns: pd.Index) -> pd.DataFrame | None:
    """把日频因子取月末截面，返回 float32 面板。Sample a daily factor at month ends."""
    if not isinstance(fac, pd.DataFrame):
        return None
    panel = fac.reindex(index=month_dates, columns=columns)
    panel = clean(panel).astype(np.float32)
    if panel.notna().sum().sum() == 0:
        return None
    return panel


def save_family(family: str, factors: dict, month_dates: pd.DatetimeIndex,
                columns: pd.Index) -> Path:
    """保存家族面板到 npz。Save a family's month-end panels to npz."""
    POOL_DIR.mkdir(parents=True, exist_ok=True)
    path = POOL_DIR / f"{family}.npz"
    arrays = {name: panel.to_numpy(dtype=np.float32)
              for name, panel in factors.items()}
    np.savez(path, **arrays)
    np.savez(POOL_DIR / "meta.npz",
             dates=month_dates.to_numpy(dtype="datetime64[ns]"),
             columns=np.asarray(columns, dtype=object))
    return path


def load_family(family: str) -> dict[str, pd.DataFrame]:
    """读取家族面板，返回 因子名 → DataFrame。Load a family's panels."""
    path = POOL_DIR / f"{family}.npz"
    if not path.exists():
        raise FileNotFoundError(f"缺少因子面板：{path}，请先运行 factor_screen.py")
    meta = np.load(POOL_DIR / "meta.npz", allow_pickle=True)
    dates = pd.DatetimeIndex(meta["dates"])
    columns = pd.Index(meta["columns"].astype(str))
    with np.load(path) as data:
        return {name: pd.DataFrame(data[name], index=dates, columns=columns)
                for name in data.files}


def load_all() -> dict[str, pd.DataFrame]:
    """合并四个家族的面板。Merge all families' panels."""
    out = {}
    for family in FAMILIES:
        path = POOL_DIR / f"{family}.npz"
        if path.exists():
            out.update(load_family(family))
    return out


# ===================== 全量计算 / Full computation =====================
def compute_family(family: str, matrices: dict, month_dates: pd.DatetimeIndex,
                   benchmark: dict | None = None, force: bool = False) -> dict:
    """计算并保存单个家族；已存在则直接读取。Compute/save one family."""
    path = POOL_DIR / f"{family}.npz"
    if path.exists() and not force:
        print(f"[{family}] 已存在，跳过计算：{path.name}")
        return load_family(family)

    columns = matrices["close"].columns
    factors: dict[str, pd.DataFrame] = {}
    t0 = time.time()
    for i, (name, fac) in enumerate(iter_family(family, matrices, month_dates, benchmark), 1):
        panel = sample_panel(fac, month_dates, columns)
        if panel is None:
            print(f"  [{family}] {name} 全为空，跳过")
            continue
        factors[name] = panel
        if i % 20 == 0:
            print(f"  [{family}] {i} 个因子，用时 {time.time() - t0:.0f}s")
    save_family(family, factors, month_dates, columns)
    print(f"[{family}] 完成：{len(factors)} 个因子，用时 {time.time() - t0:.0f}s")
    return factors
