# -*- coding: utf-8 -*-
"""滚动多因子（S7 版）：年频换因子与权重，纯多头门控 + 多头 t 加权。

与 rolling_multifactor.py 的区别只在 fit_fold：
- 原版：|ICIR|>=0.2 选因子，|ICIR| 归一化权重；
- 本版：额外要求训练窗口内多头超额年化>0 且多头 t>=1，权重改为多头 t。

多头超额 = 方向校准后 top 组等权收益 - 当期全池等权收益（逐月，索引 t+1）。
方向由训练窗口 RankIC 符号决定：sign=+1 取原始 top 组，sign=-1 取原始
bottom 组（方向校准后 bottom 组即为多头组），全程只用训练窗口数据。

样本外 2020-2025，对照原版 rolling（ICIR 加权、top10=-6.40%）。

用法 / Usage
------------
    python baseline/rolling_top_weighted.py
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

import price_volume_baseline as pv  # noqa: E402
import rolling_multifactor as rm  # noqa: E402
from utils.data_standardize import preprocess  # noqa: E402

OUTPUT_DIR = HERE.parent / "results" / "rolling_top"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_T_GATE = 1.0       # 多头超额 t 门控 / min long-side t-stat
N_GROUP = 5
TOP_N_LIST = (10, 50, 100)


def excess_panels(zfactors, returns):
    """每因子原始方向的 top / bottom 组相对全池的逐月超额（索引 t+1）。
    Per-factor top/bottom group excess over the universe mean in the raw factor
    direction, indexed at t+1.
    """
    panels = {}
    for name, fac in zfactors.items():
        top = pd.Series(np.nan, index=fac.index, dtype=float)
        bot = pd.Series(np.nan, index=fac.index, dtype=float)
        for i in range(len(fac.index) - 1):
            df = pd.DataFrame({"s": fac.iloc[i], "r": returns.iloc[i + 1]}).dropna()
            if len(df) < N_GROUP * 20:
                continue
            try:
                g = pd.qcut(df["s"], N_GROUP, labels=False, duplicates="drop")
            except ValueError:
                continue
            means = df.groupby(g)["r"].mean()
            if len(means) < N_GROUP:
                continue
            top.iloc[i + 1] = means.iloc[-1] - df["r"].mean()
            bot.iloc[i + 1] = means.iloc[0] - df["r"].mean()
        panels[name] = (top, bot)
    return panels


def ann_t(series):
    """年化收益与 t 值。Annualized return and t-stat of a monthly series."""
    series = series.dropna()
    n = len(series)
    if n < 12 or series.std(ddof=1) == 0:
        return np.nan, np.nan
    nav = (1 + series).cumprod()
    ann = nav.iloc[-1] ** (12 / n) - 1
    t = series.mean() / series.std(ddof=1) * np.sqrt(n)
    return ann, t


def fit_fold_top(ic_series, panels, year):
    """训练窗口内：ICIR + 方向 + 多头门控 + 多头 t 权重。
    Within the training window: ICIR, direction, long-side gate, long-t weights.
    """
    train_start = pd.Timestamp(f"{year - rm.TRAIN_YEARS}-01-01")
    train_end = pd.Timestamp(f"{year - 1}-12-31")

    rows = []
    for name, ic in ic_series.items():
        seg = ic.loc[(ic.index > train_start) & (ic.index <= train_end)].dropna()
        if len(seg) < 12:
            continue
        std = seg.std(ddof=1)
        mean_ic = seg.mean()
        sign = int(np.sign(mean_ic)) or 1
        top, bot = panels[name]
        long_raw = top if sign > 0 else bot
        long_seg = long_raw.loc[(long_raw.index > train_start) &
                                (long_raw.index <= train_end)]
        long_ann, long_t = ann_t(long_seg)
        rows.append({"因子": name, "平均IC": mean_ic,
                     "ICIR": mean_ic / std if std > 0 else 0.0,
                     "方向": sign, "多头年化": long_ann, "多头t": long_t,
                     "训练月数": len(seg)})

    summary = pd.DataFrame(rows).dropna(subset=["多头t"])
    if summary.empty:
        raise RuntimeError(f"{year} 年训练窗口内没有可用因子")

    summary["AbsICIR"] = summary["ICIR"].abs()
    gate = ((summary["AbsICIR"] >= rm.ICIR_THRESHOLD) &
            (summary["多头年化"] > 0) & (summary["多头t"] >= TOP_T_GATE))
    picked = summary[gate].sort_values("多头t", ascending=False)
    if len(picked) < rm.MIN_FACTORS:
        picked = summary.sort_values("多头t", ascending=False).head(rm.MIN_FACTORS)
    picked = picked.head(rm.MAX_FACTORS).copy()

    picked["权重"] = picked["多头t"] / picked["多头t"].sum()
    return {"year": year, "train": f"{year - rm.TRAIN_YEARS}-{year - 1}",
            "picked": picked,
            "weights": picked.set_index("因子")["权重"],
            "signs": picked.set_index("因子")["方向"]}


def main():
    print(f"S7 滚动版：纯多头门控 + 多头 t 加权 | 样本外 "
          f"{rm.OOS_START_YEAR}-{rm.OOS_END_YEAR}")
    print("=" * 70)

    print("\n[1/5] 载入数据与月频面板...")
    daily = pv.load_market_data(pv.MARKET, pv.START, pv.END)
    monthly = pv.build_monthly(daily)
    dates = monthly["dates"]
    returns = monthly["returns"]
    valid = pv.filter_pool(monthly)
    print(f"  平均可交易股票数：{valid.sum(axis=1).mean():.0f}")

    print("[2/5] 计算量价因子与预处理...")
    factors = pv.compute_price_volume_factors(monthly, valid)
    zfactors = {name: preprocess(fac) for name, fac in factors.items()}

    print("[3/5] 预计算 RankIC 与多空分组超额面板...")
    ic_series = rm.rank_ic_panel(zfactors, returns)
    panels = excess_panels(zfactors, returns)

    print("[4/5] 逐年滚动重训...")
    comps, logs = [], []
    for year in range(rm.OOS_START_YEAR, rm.OOS_END_YEAR + 1):
        fit = fit_fold_top(ic_series, panels, year)
        print(f"\n[{fit['train']}] -> 样本外 {year}：入选 {len(fit['weights'])} 个因子")
        for name in fit["weights"].index:
            row = fit["picked"].set_index("因子").loc[name]
            print(f"  {name:<20s} 权重 {fit['weights'][name]:.3f} "
                  f"方向 {fit['signs'][name]:+d} 多头t {row['多头t']:.2f}")
        year_log = fit["picked"].copy()
        year_log.insert(0, "样本外年份", year)
        year_log.insert(1, "训练窗口", fit["train"])
        logs.append(year_log)
        comps.append(rm.build_composite(zfactors, fit, year))

    composite = pd.concat(comps).sort_index()
    composite.to_csv(OUTPUT_DIR / "composite_factor.csv", encoding="utf-8-sig")
    log = pd.concat(logs, ignore_index=True)[
        ["样本外年份", "训练窗口", "因子", "平均IC", "ICIR", "方向",
         "多头年化", "多头t", "权重", "训练月数"]]
    log.to_csv(OUTPUT_DIR / "factor_selection.csv", index=False, encoding="utf-8-sig")

    oos_start = pd.Timestamp(f"{rm.OOS_START_YEAR}-01-01")
    oos_end = pd.Timestamp(f"{rm.OOS_END_YEAR}-12-31")
    comp_oos = composite.loc[(composite.index >= oos_start) &
                             (composite.index <= oos_end)]
    ret_oos = returns.loc[(returns.index >= oos_start) & (returns.index <= oos_end)]
    print(f"\n样本外区间：{comp_oos.index[0].date()} ~ {comp_oos.index[-1].date()}"
          f"，共 {len(comp_oos)} 期")

    print("\n[5/5] 样本外回测（持仓数对照）...")
    rows = {}
    for tn in TOP_N_LIST:
        out_dir = OUTPUT_DIR / f"top{tn}"
        out_dir.mkdir(parents=True, exist_ok=True)
        _, perf = pv.backtest_portfolio(
            comp_oos, ret_oos, comp_oos.index, top_n=tn, comsn=pv.COMSN,
            output_dir=out_dir, title=f"S7 滚动 top{tn}")
        rows[f"top{tn}"] = perf
        print(f">>> S7 rolling top{tn}: 累计 {perf['累计收益']*100:.2f}% | "
              f"夏普 {perf['夏普比率']:.3f}")

    cmp = pd.DataFrame(rows).T
    cmp.index.name = "持仓数"
    cmp.to_csv(OUTPUT_DIR / "oos_performance.csv", encoding="utf-8-sig")
    show = cmp[["累计收益", "年化收益", "年化波动", "夏普比率",
                "最大回撤", "年化换手"]].apply(pd.to_numeric)
    for c in ["累计收益", "年化收益", "年化波动", "最大回撤"]:
        show[c] = (show[c] * 100).round(2)
    show[["夏普比率", "年化换手"]] = show[["夏普比率", "年化换手"]].round(3)
    print("\n样本外绩效（2020-2025）：")
    print(show.to_string())
    print(f"\n结果已保存到：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
