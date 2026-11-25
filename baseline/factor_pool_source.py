# -*- coding: utf-8 -*-
"""因子池数据源：把初筛通过的因子接到优化器与滚动策略上。

供 ``optimizer1_heuristic.py`` / ``rolling_multifactor.py`` 等脚本使用：
当环境变量 ``FACTOR_SOURCE=factor_pool`` 时，用这里的月末面板替代
``price_volume_baseline.compute_price_volume_factors`` 的 38 个旧因子。

Data source that plugs the screened factor pool into the optimizers and the
rolling strategy when ``FACTOR_SOURCE=factor_pool``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import factor_screen as fs
import price_volume_baseline as pv
from factor_lib import matrices as flm
from factor_lib import runner

PASS_FILE = Path(__file__).resolve().parents[1] / "results" / "factor_screen" / "screen_pass.csv"


def pass_names() -> list[str]:
    """初筛通过的因子名单。Names of factors passing the screen."""
    if not PASS_FILE.exists():
        raise FileNotFoundError(f"缺少初筛结果：{PASS_FILE}，请先运行 factor_screen.py")
    return pd.read_csv(PASS_FILE)["因子"].tolist()


def load_pool(only_pass: bool = True, mask_valid: bool = True):
    """返回 (monthly, factors, dates)。

    monthly 与 ``price_volume_baseline.build_monthly`` 的字段兼容（含
    close/vol/turnover_rate/dates/returns/valid），factors 为月末因子面板。
    Masking by the tradable universe matches the legacy factor computation.
    """
    matrices, dates = flm.build_matrices(pv.START, pv.END)
    monthly = fs.build_monthly(matrices, dates)
    panels = runner.load_all()
    names = pass_names() if only_pass else list(panels)
    valid = monthly["valid"]
    factors = {}
    for name in names:
        if name not in panels:
            continue
        fac = panels[name].reindex(index=dates, columns=matrices["close"].columns)
        factors[name] = fac.where(valid) if mask_valid else fac
    return monthly, factors, dates
