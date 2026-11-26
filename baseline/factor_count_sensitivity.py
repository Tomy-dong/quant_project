# -*- coding: utf-8 -*-
"""因子数量 K 敏感性测试。

只载入一次因子面板与统计量，循环 K = 3/4/5/6/8/10：
  - 每年用优化器一的门控+启发式搜索选 K 个因子（ICIR 下限 0.5，与
    optimizer1_pool 口径一致），合成样本外(2020-2025)分数；
  - Top30 等权回测（单边千二费率），算年化/夏普/回撤；
  - 相对沪深300/中证500 的超额年化与信息比率；
  - 三因子回归 alpha（复用 results/ff3/ff3_factors.csv）；
  - 统计 2020-2025 跨年份去重因子总数。

输出 results/factor_count_sensitivity/：
    summary.csv        各 K 的绩效、超额、FF3 alpha、去重因子数
    k_comparison.png   K vs 年化/夏普/回撤/alpha 对比图
    factor_sets.csv    各 K 每年入选因子明细
"""
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import price_volume_baseline as pv          # noqa: E402
import rolling_multifactor as rm            # noqa: E402
import rolling_top_weighted as rt           # noqa: E402
import optimizer1_heuristic as opt1         # noqa: E402
import factor_pool_source as fps            # noqa: E402
import ff3_regression as ff3                # noqa: E402
from data_io import load_benchmark          # noqa: E402
from utils.data_standardize import preprocess  # noqa: E402
from utils.stockselection import stock_selection  # noqa: E402
from utils.utils import calc_nav             # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

RESULT = HERE.parent / "results"
OUT = RESULT / "factor_count_sensitivity"
OUT.mkdir(parents=True, exist_ok=True)
FF3_PATH = RESULT / "ff3" / "ff3_factors.csv"

K_GRID = [3, 4, 5, 6, 8, 10]
TOP_N = 30
COMSN = pv.COMSN  # 0.002 单边费率


def metrics(pnl):
    s = pnl.dropna()
    if len(s) < 2:
        return {}
    nav = (1 + s).cumprod()
    total = float(nav.iloc[-1] - 1)
    years = len(s) / 12.0
    ann = float((1 + total) ** (1 / years) - 1)
    vol = float(s.std(ddof=1) * np.sqrt(12))
    sharpe = float(ann / vol) if vol > 0 else np.nan
    dd = float((nav / nav.cummax() - 1).min())
    return {"累计收益": total, "年化收益": ann, "年化波动": vol,
            "夏普比率": sharpe, "最大回撤": dd}


def backtest_top30(composite, returns):
    dates = composite.index
    holdings = np.full((len(dates), TOP_N), np.nan)
    for i in range(len(dates)):
        row = composite.iloc[i].dropna()
        if len(row) >= TOP_N:
            codes = row.nlargest(TOP_N).index
            holdings[i] = [composite.columns.get_loc(c) for c in codes]
    weights = stock_selection(returns.columns, holdings, dates)
    out = calc_nav(returns.fillna(0), weights, comsn=COMSN)
    return out["pnl"]


def bench_monthly(index, dates):
    close = load_benchmark(str(dates.min().date()),
                           str(dates.max().date()), index=index)["close"]
    return close.reindex(dates).pct_change(fill_method=None)


def main():
    print(f"因子数量 K 敏感性 | Top{TOP_N} 等权 | 费率 {COMSN:.4f} | K ∈ {K_GRID}")
    print("=" * 72)

    print("\n[1/4] 载入因子面板、预处理、预计算统计量（约 2-3 分钟）...")
    monthly, factors, dates = fps.load_pool(mask_valid=True)
    returns = monthly["returns"]
    zfactors = {name: preprocess(fac) for name, fac in factors.items()}
    names = list(zfactors)
    print(f"  因子池 {len(names)} 个（初筛通过），月末样本 {len(dates)} 期")

    # 与 optimizer1_pool 口径一致：ICIR 下限 0.5
    opt1.ICIR_MIN = 0.5
    print(f"  ICIR 下限 = {opt1.ICIR_MIN}")

    print("  计算 IC / 多空面板 / 规模暴露 / 相关矩阵...")
    ic_series = rm.rank_ic_panel(zfactors, returns)
    panels = rt.excess_panels(zfactors, returns)
    ls_panel = {n: panels[n][0] - panels[n][1] for n in names}
    size_beta, _ = opt1.compute_size_exposure(monthly, zfactors, dates)
    corr_mats, _ = opt1.compute_corr_mats(zfactors, dates)

    factors_df = pd.read_csv(FF3_PATH, index_col=0, parse_dates=True)
    factors_df.index.name = "日期"

    years = list(range(opt1.OOS_START_YEAR, opt1.OOS_END_YEAR + 1))
    oos_start = pd.Timestamp(f"{opt1.OOS_START_YEAR}-01-01")
    oos_end = pd.Timestamp(f"{opt1.OOS_END_YEAR}-12-31")

    rows, set_rows = [], []
    for K in K_GRID:
        print(f"\n[K={K}] 逐年选择 + 合成 + 回测...")
        opt1.K_TARGET = K
        opt1.K_MIN = K
        opt1.K_MAX = K
        fits = {}
        all_picked = set()
        for year in years:
            diag = opt1.build_diagnostics(ic_series, panels, ls_panel,
                                          size_beta, names, year)
            corr = opt1.average_corr(corr_mats, dates, year, names)
            problem = opt1.YearProblem(zfactors, returns, dates, diag,
                                       corr, names, year)
            fit = opt1.solve_year(year, diag, corr, names, problem)
            fits[year] = fit
            picked = fit["subset"]
            all_picked.update(picked)
            set_rows.append({"K": K, "年份": year, "入选因子": "; ".join(picked),
                             "K实际": len(picked),
                             "降级": fit["degraded"]})

        comps = [rm.build_composite(zfactors, fits[y], y) for y in years]
        composite = pd.concat(comps).sort_index()
        composite = composite.loc[(composite.index >= oos_start) &
                                  (composite.index <= oos_end)]
        ret_oos = returns.loc[composite.index]
        pnl = backtest_top30(composite, ret_oos)

        m = metrics(pnl)
        # 基准超额
        bench_ret = {b: bench_monthly(b, composite.index)
                     for b in ["csi300", "csi500"]}
        bench_ann = {b: (1 + bench_ret[b].dropna()).prod() **
                     (12 / len(bench_ret[b].dropna())) - 1
                     for b in bench_ret}
        ex, ir = {}, {}
        for b, br in bench_ret.items():
            diff = (pnl - br).dropna()
            ex[b] = m["年化收益"] - bench_ann[b]
            ir[b] = (diff.mean() / diff.std(ddof=1) * np.sqrt(12)
                     if diff.std(ddof=1) > 0 else np.nan)
        # FF3 alpha
        reg = ff3.regress_one(f"K{K}", pnl, factors_df)

        row = {"K": K, "去重因子数": len(all_picked), **m,
               "超额(沪深300)": ex["csi300"], "IR(沪深300)": ir["csi300"],
               "超额(中证500)": ex["csi500"], "IR(中证500)": ir["csi500"],
               "alpha年化": reg["alpha年化"], "alpha_t": reg["alpha_t"],
               "alpha_p": reg["alpha_p"], "R2": reg["R2"],
               "SMB": reg["SMB"]}
        rows.append(row)
        print(f"  K={K}: 去重因子 {len(all_picked)} | 年化 {m['年化收益']*100:.2f}% "
              f"夏普 {m['夏普比率']:.3f} 回撤 {m['最大回撤']*100:.2f}% | "
              f"超额CSI300 {ex['csi300']*100:+.2f}% 超额CSI500 {ex['csi500']*100:+.2f}% | "
              f"alpha {reg['alpha年化']*100:+.2f}%(p={reg['alpha_p']:.3f}) R2 {reg['R2']:.2f}")

    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(set_rows).to_csv(OUT / "factor_sets.csv", index=False,
                                  encoding="utf-8-sig")

    # 图：K vs 年化/夏普/回撤/alpha
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, col, title, pct in [
        (axes[0, 0], "年化收益", "年化收益", True),
        (axes[0, 1], "夏普比率", "夏普比率", False),
        (axes[1, 0], "最大回撤", "最大回撤", True),
        (axes[1, 1], "alpha年化", "FF3 alpha 年化", True),
    ]:
        v = summ[col] * (100 if pct else 1)
        ax.plot(summ["K"], v, "o-", color="#2f6f9f", linewidth=2)
        ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
        if col == "年化收益":
            ax.axhline(15, color="#c0392b", linewidth=0.8,
                       linestyle=":", label="目标 15%")
            ax.legend(fontsize=8)
        if col == "最大回撤":
            ax.axhline(-40, color="#c0392b", linewidth=0.8,
                       linestyle=":", label="目标 -40%")
            ax.legend(fontsize=8)
        ax.set_xlabel("每年因子数 K")
        ax.set_ylabel(("%" if pct else ""))
        ax.set_title(title)
        ax.grid(alpha=0.3)
        for x, y in zip(summ["K"], v):
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8)
    fig.suptitle("因子数量 K 敏感性（Top30 等权，2020-2025 样本外）", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "k_comparison.png", dpi=150)
    plt.close(fig)

    # 打印汇总
    print("\n" + "=" * 72)
    print("汇总（%）：")
    show = summ.copy()
    for c in ["累计收益", "年化收益", "年化波动", "最大回撤",
              "超额(沪深300)", "超额(中证500)", "alpha年化"]:
        show[c] = (show[c] * 100).round(2)
    for c in ["夏普比率", "IR(沪深300)", "IR(中证500)", "alpha_t"]:
        show[c] = show[c].round(3)
    show["alpha_p"] = show["alpha_p"].round(3)
    show["R2"] = show["R2"].round(3)
    show["SMB"] = show["SMB"].round(3)
    print(show.to_string(index=False))

    # 挑最优（去重因子<=10 前提下年化最高；若都<=10则直接年化最高）
    cand = summ[summ["去重因子数"] <= 10]
    if cand.empty:
        cand = summ
    best = cand.loc[cand["年化收益"].idxmax()]
    print(f"\n>>> 在去重因子≤10 中表现最好：K={int(best['K'])}，"
          f"年化 {best['年化收益']*100:.2f}%，夏普 {best['夏普比率']:.3f}，"
          f"回撤 {best['最大回撤']*100:.2f}%，去重因子 {int(best['去重因子数'])}")

    print(f"\n结果目录：{OUT}")


if __name__ == "__main__":
    main()
