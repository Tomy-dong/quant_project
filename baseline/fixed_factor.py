# -*- coding: utf-8 -*-
"""固定因子策略：2015-2019 选因子，2020-2025 因子与权重完全固定。

与滚动策略（rolling_multifactor.py）的唯一区别：滚动策略每年用过去 5 年
重选因子与权重，本脚本只在 2015-2019 上选一次，之后 2020-2025 不变，只按月
换股。其他口径（股票池、预处理、因子权重规则、费率、持仓数）完全一致。

用法
----
    python baseline/fixed_factor.py
    set FACTOR_SOURCE=factor_pool & set RM_TOP_N=30 & python baseline/fixed_factor.py
"""
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import price_volume_baseline as pv  # noqa: E402
import rolling_multifactor as rm  # noqa: E402
from utils.data_standardize import preprocess  # noqa: E402

OUTPUT_DIR = HERE.parent / "results" / os.environ.get("FIXED_OUTPUT", "fixed_pool")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SELECT_YEAR = rm.OOS_START_YEAR          # 用 2015-2019 选因子 / select on 2015-2019
OOS_START_YEAR = rm.OOS_START_YEAR
OOS_END_YEAR = rm.OOS_END_YEAR
TOP_N = int(os.environ.get("RM_TOP_N", "30"))
FACTOR_SOURCE = os.environ.get("FACTOR_SOURCE", "factor_pool").lower()


def main():
    print(f"固定因子策略：{SELECT_YEAR - rm.TRAIN_YEARS}-{SELECT_YEAR - 1} 选因子，"
          f"{OOS_START_YEAR}-{OOS_END_YEAR} 固定不变")
    print("=" * 70)

    print("[1/4] 载入因子与月末面板...")
    if FACTOR_SOURCE == "factor_pool":
        import factor_pool_source as fps
        monthly, factors, dates = fps.load_pool(mask_valid=True)
    else:
        daily = pv.load_market_data(pv.MARKET, pv.START, pv.END)
        monthly = pv.build_monthly(daily)
        dates = monthly["dates"]
        valid = pv.filter_pool(monthly)
        factors = pv.compute_price_volume_factors(monthly, valid)
    returns = monthly["returns"]
    print(f"  因子来源：{FACTOR_SOURCE}，{len(factors)} 个候选")

    print("[2/4] 截面预处理与 RankIC...")
    zfactors = {name: preprocess(fac) for name, fac in factors.items()}
    ic_series = rm.rank_ic_panel(zfactors, returns)

    print("[3/4] 用 2015-2019 选因子（此后固定）...")
    fit = rm.fit_fold(ic_series, SELECT_YEAR)
    print(f"  训练窗口 {fit['train']}，入选 {len(fit['weights'])} 个因子：")
    for name in fit["weights"].index:
        print(f"  {name:<20s} 权重 {fit['weights'][name]:.3f} "
              f"方向 {fit['signs'][name]:+d}")
    fit["picked"].to_csv(OUTPUT_DIR / "factor_selection.csv",
                         index=False, encoding="utf-8-sig")

    print("[4/4] 2020-2025 固定因子回测...")
    comps = [rm.build_composite(zfactors, fit, year)
             for year in range(OOS_START_YEAR, OOS_END_YEAR + 1)]
    composite = pd.concat(comps).sort_index()
    composite.to_csv(OUTPUT_DIR / "composite_factor.csv", encoding="utf-8-sig")

    oos_start = pd.Timestamp(f"{OOS_START_YEAR}-01-01")
    oos_end = pd.Timestamp(f"{OOS_END_YEAR}-12-31")
    comp_oos = composite.loc[(composite.index >= oos_start) & (composite.index <= oos_end)]
    ret_oos = returns.loc[(returns.index >= oos_start) & (returns.index <= oos_end)]

    outputs, perf = pv.backtest_portfolio(
        comp_oos, ret_oos, comp_oos.index, top_n=TOP_N, comsn=pv.COMSN,
        output_dir=OUTPUT_DIR,
        title=f"固定因子：{fit['train']} 选因子，因子固定、月频换股")

    pd.DataFrame({"样本外年份": range(OOS_START_YEAR, OOS_END_YEAR + 1),
                  "训练窗口": [fit["train"]] * (OOS_END_YEAR - OOS_START_YEAR + 1),
                  "K": [len(fit["weights"])] * (OOS_END_YEAR - OOS_START_YEAR + 1)}
                 ).to_csv(OUTPUT_DIR / "selection_summary.csv",
                          index=False, encoding="utf-8-sig")
    print(f"\n绩效已保存：{OUTPUT_DIR / 'performance.csv'}")


if __name__ == "__main__":
    main()
