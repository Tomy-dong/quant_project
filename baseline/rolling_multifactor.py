# -*- coding: utf-8 -*-
"""滚动多因子：年频换因子 + 月频换股。

与 price_volume_baseline.py 的区别只在因子层：

- 基线：用 2015–2025 全样本算 IC/ICIR，一次性选因子，属静态对照组，含测试期信息；
- 本脚本：每年首个调仓日用过去 5 年重训因子集合与权重，年内固定，只按月换股。

因子回看窗口不随年份改变（见 docs/strategy.md 3.1），每年变的只有
“用哪些因子”和“各给多少权重”。

用法
----
    python baseline/rolling_multifactor.py
"""

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

from price_volume_baseline import (MARKET, START, END, COMSN,
                                   backtest_portfolio, build_monthly,
                                   compute_price_volume_factors, filter_pool,
                                   load_market_data)
from utils.data_standardize import preprocess

OUTPUT_DIR = HERE.parent / "results" / "rolling"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===================== 配置 =====================
TRAIN_YEARS = 5           # 训练窗口：5 年
OOS_START_YEAR = 2020     # 样本外起始年份
OOS_END_YEAR = 2025       # 样本外结束年份
ICIR_THRESHOLD = 0.2      # 入选因子的 |ICIR| 下限
MAX_FACTORS = 20          # 入选因子数上限
MIN_FACTORS = 3           # 入选因子数下限


# ===================== 1. 训练窗口内的 RankIC =====================
def rank_ic_panel(factors, returns):
    """全样本 RankIC 序列：IC_t 用 t 期因子与 t+1 期收益，索引记在 t+1。

    只做截面排名相关，不涉及参数拟合，后续按训练窗口切片即可复用。
    """
    ret_rank = returns.rank(axis=1)
    rv = ret_rank.values
    out = {}
    for name, fac in factors.items():
        fv = fac.rank(axis=1).values
        ic = pd.Series(np.nan, index=fac.index, dtype=float)
        for i in range(len(fac.index) - 1):
            a, b = fv[i], rv[i + 1]
            mask = ~np.isnan(a) & ~np.isnan(b)
            if mask.sum() > 2:
                ic.iloc[i + 1] = np.corrcoef(a[mask], b[mask])[0, 1]
        out[name] = ic
    return out


def fit_fold(ic_series, year):
    """用 year 之前的 5 年估 ICIR，选因子并给出 |ICIR| 归一化权重与方向。"""
    train_start = pd.Timestamp(f"{year - TRAIN_YEARS}-01-01")
    train_end = pd.Timestamp(f"{year - 1}-12-31")

    rows = []
    for name, ic in ic_series.items():
        # IC 索引在 t+1，要求因子与收益两端都落在训练窗口内
        seg = ic.loc[(ic.index > train_start) & (ic.index <= train_end)].dropna()
        if len(seg) < 12:
            continue
        std = seg.std(ddof=1)
        rows.append({
            "因子": name,
            "平均IC": seg.mean(),
            "ICIR": seg.mean() / std if std > 0 else 0.0,
            "训练月数": len(seg),
        })

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise RuntimeError(f"{year} 年训练窗口内没有可用因子")

    summary["AbsICIR"] = summary["ICIR"].abs()
    picked = summary[summary["AbsICIR"] >= ICIR_THRESHOLD].sort_values(
        "AbsICIR", ascending=False)
    if len(picked) < MIN_FACTORS:
        picked = summary.sort_values("AbsICIR", ascending=False).head(MIN_FACTORS)
    picked = picked.head(MAX_FACTORS).copy()

    picked["权重"] = picked["AbsICIR"] / picked["AbsICIR"].sum()
    picked["方向"] = np.sign(picked["平均IC"]).replace(0, 1).astype(int)

    return {
        "year": year,
        "train": f"{year - TRAIN_YEARS}-{year - 1}",
        "picked": picked,
        "weights": picked.set_index("因子")["权重"],
        "signs": picked.set_index("因子")["方向"],
    }


# ===================== 2. 合成当期分数 =====================
def build_composite(zfactors, fit, year):
    """当年各月：sum_k w_k * sign_k * z_{i,k}，因子集与权重年内固定。"""
    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year}-12-31")

    comp = None
    for name in fit["weights"].index:
        z = zfactors[name]
        z = z.loc[(z.index >= start) & (z.index <= end)]
        term = (fit["weights"][name] * fit["signs"][name]) * z
        comp = term if comp is None else comp + term
    return comp


# ===================== 3. 滚动重训 =====================
def run_rolling(factors, returns, dates):
    print("预计算截面预处理（MAD + 行业中性化 + Z-score）...")
    zfactors = {name: preprocess(fac) for name, fac in factors.items()}

    print("预计算全样本 RankIC 序列...")
    ic_series = rank_ic_panel(zfactors, returns)

    comps, logs = [], []
    for year in range(OOS_START_YEAR, OOS_END_YEAR + 1):
        fit = fit_fold(ic_series, year)
        print(f"\n[{fit['train']}] 训练 -> 样本外 {year}："
              f"入选 {len(fit['weights'])} 个因子")
        for name in fit["weights"].index:
            print(f"  {name:<20s} 权重 {fit['weights'][name]:.3f} "
                  f"方向 {fit['signs'][name]:+d}")

        year_log = fit["picked"].copy()
        year_log.insert(0, "样本外年份", year)
        year_log.insert(1, "训练窗口", fit["train"])
        logs.append(year_log)

        comps.append(build_composite(zfactors, fit, year))

    composite = pd.concat(comps).sort_index()
    composite.to_csv(OUTPUT_DIR / "composite_factor.csv", encoding="utf-8-sig")

    log = pd.concat(logs, ignore_index=True)[
        ["样本外年份", "训练窗口", "因子", "平均IC", "ICIR", "权重", "方向", "训练月数"]]
    log.to_csv(OUTPUT_DIR / "factor_selection.csv", index=False, encoding="utf-8-sig")
    print("\n逐年因子切换记录：")
    print(log.round(4).to_string(index=False))

    oos_start = pd.Timestamp(f"{OOS_START_YEAR}-01-01")
    oos_end = pd.Timestamp(f"{OOS_END_YEAR}-12-31")
    comp_oos = composite.loc[(composite.index >= oos_start) &
                             (composite.index <= oos_end)]
    ret_oos = returns.loc[(returns.index >= oos_start) &
                          (returns.index <= oos_end)]

    print(f"\n样本外区间：{comp_oos.index[0].date()} ~ {comp_oos.index[-1].date()}"
          f"，共 {len(comp_oos)} 期")
    outputs, perf = backtest_portfolio(
        comp_oos, ret_oos, comp_oos.index, comsn=COMSN, output_dir=OUTPUT_DIR,
        title=f"滚动多因子：年频换因子 / 月频换股 ({MARKET.upper()})")
    return outputs, perf, log


# ===================== 主流程 =====================
def main():
    print(f"市场：{MARKET} | 训练窗口：{TRAIN_YEARS} 年 | "
          f"样本外：{OOS_START_YEAR}-{OOS_END_YEAR}")
    print("=" * 60)

    print("\n[1/4] 载入日频数据...")
    daily = load_market_data(MARKET, START, END)

    print("[2/4] 月末采样与股票池...")
    monthly = build_monthly(daily)
    dates = monthly["dates"]
    valid = filter_pool(monthly)
    print(f"  月末样本 {len(dates)} 期，{monthly['price'].shape[1]} 只股票")

    print("[3/4] 计算量价因子...")
    factors = compute_price_volume_factors(monthly, valid)
    print(f"  共 {len(factors)} 个因子")

    print("[4/4] 滚动重训与回测...")
    run_rolling(factors, monthly["returns"], dates)

    print(f"\n结果已保存到：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
