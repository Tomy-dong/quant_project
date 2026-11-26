# -*- coding: utf-8 -*-
"""最终报告：候选策略 vs 沪深300/中证500，含 FF3 回归与全套图表。

候选策略（全部 Top30、单边千二、2020-2025 样本外）：
  - XGBoost 合成（读 results/xgb_strategy/composite_xgb.csv）
  - 线性固定因子 N=5 EW（compact 选出的 5 因子，因子等权）
  - 线性固定因子 N=10 EW（compact 选出的 10 因子，因子等权）
基准：沪深300、中证500

输出 results/final/：
    summary.csv          绩效 + 超额 + IR + FF3 alpha 汇总
    annual_returns.csv   分年度收益
    nav_compare.png      净值对比
    annual_returns.png   分年度收益柱状图
    drawdown.png         回撤对比
    ff3_alpha.png        FF3 alpha 年化 + 95% CI
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
import compact_strategy as cs               # noqa: E402
import optimizer2_weights as o2w            # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

RESULT = HERE.parent / "results"
OUT = RESULT / "final"
OUT.mkdir(parents=True, exist_ok=True)
FF3_PATH = RESULT / "ff3" / "ff3_factors.csv"
XGB_COMPOSITE = RESULT / "xgb_strategy" / "composite_xgb.csv"

TOP_N = 30
COMSN = pv.COMSN
SELECT_YEAR = rm.OOS_START_YEAR
RHO_MAX = 0.7

COLORS = {"XGBoost+IVar": "crimson", "XGBoost合成": "tomato",
          "线性N=5": "seagreen", "线性N=10": "darkviolet",
          "沪深300": "steelblue", "中证500": "darkorange"}


def metrics(pnl):
    s = pnl.dropna()
    nav = (1 + s).cumprod()
    total = float(nav.iloc[-1] - 1)
    years = len(s) / 12.0
    ann = float((1 + total) ** (1 / years) - 1)
    vol = float(s.std(ddof=1) * np.sqrt(12))
    return {"累计收益": total, "年化收益": ann, "年化波动": vol,
            "夏普比率": float(ann / vol) if vol > 0 else np.nan,
            "最大回撤": float((nav / nav.cummax() - 1).min())}


def backtest_top30(composite, returns):
    dates = composite.index
    codes = composite.columns
    weights = pd.DataFrame(0.0, index=dates, columns=codes)
    for i in range(len(dates)):
        row = composite.iloc[i].dropna()
        if len(row) >= TOP_N:
            sel = row.nlargest(TOP_N).index
            weights.iloc[i][sel] = 1.0 / TOP_N
    return calc_nav(returns.fillna(0), weights, comsn=COMSN)["pnl"]


def bench_monthly(index, dates):
    close = load_benchmark(str(dates.min().date()),
                           str(dates.max().date()), index=index)["close"]
    return close.reindex(dates).pct_change(fill_method=None)


def main():
    print("最终报告生成：载入因子面板（约2-3分钟）...")
    monthly, factors, dates = fps.load_pool(mask_valid=True)
    returns = monthly["returns"]
    zfactors = {name: preprocess(fac) for name, fac in factors.items()}
    names = list(zfactors)
    opt1.ICIR_MIN = 0.5
    ic_series = rm.rank_ic_panel(zfactors, returns)
    panels = rt.excess_panels(zfactors, returns)
    ls_panel = {n: panels[n][0] - panels[n][1] for n in names}
    size_beta, _ = opt1.compute_size_exposure(monthly, zfactors, dates)
    corr_mats, _ = opt1.compute_corr_mats(zfactors, dates)
    diag = opt1.build_diagnostics(ic_series, panels, ls_panel,
                                  size_beta, names, SELECT_YEAR)
    corr = opt1.average_corr(corr_mats, dates, SELECT_YEAR, names)

    years = list(range(rm.OOS_START_YEAR, rm.OOS_END_YEAR + 1))
    oos_start = pd.Timestamp(f"{rm.OOS_START_YEAR}-01-01")
    oos_end = pd.Timestamp(f"{rm.OOS_END_YEAR}-12-31")

    strategies = {}
    factor_sets = {}
    factor_counts = {}  # 实际输入因子数，与特征重要性展示数量分开记录

    # 线性固定因子 N=5 / N=10，因子等权
    for N in (5, 10):
        chosen = cs.pick_factors(diag, corr, names, N, rho_max=RHO_MAX)
        fit = cs.make_fit(diag, chosen, "EW")
        comps = [rm.build_composite(zfactors, fit, y) for y in years]
        composite = pd.concat(comps).sort_index()
        composite = composite.loc[(composite.index >= oos_start) &
                                  (composite.index <= oos_end)]
        pnl = backtest_top30(composite, returns.loc[composite.index])
        strategies[f"线性N={N}"] = pnl
        factor_sets[f"线性N={N}"] = chosen
        factor_counts[f"线性N={N}"] = len(chosen)

    # XGBoost 合成
    if XGB_COMPOSITE.exists():
        comp_xgb = pd.read_csv(XGB_COMPOSITE, index_col=0, parse_dates=True)
        comp_xgb = comp_xgb.loc[(comp_xgb.index >= oos_start) &
                                (comp_xgb.index <= oos_end)]
        pnl_xgb = backtest_top30(comp_xgb, returns.loc[comp_xgb.index])
        strategies["XGBoost合成"] = pnl_xgb
        imp = pd.read_csv(RESULT / "xgb_strategy" / "feature_importance.csv")
        factor_counts["XGBoost合成"] = len(imp)
        factor_sets["XGBoost合成"] = list(imp.iloc[:10, 0])  # 仅展示重要性 Top10
        # XGBoost 选股 + IVar 股票层风险权重（主方案）
        rets_xgb = returns.loc[comp_xgb.index]
        common = comp_xgb.columns.intersection(rets_xgb.columns)
        w_ivar = o2w.build_weight_mats(
            comp_xgb[common], rets_xgb[common], TOP_N, methods=["IVar"])["IVar"]
        out_ivar = calc_nav(rets_xgb[common].fillna(0), w_ivar, comsn=COMSN)
        strategies["XGBoost+IVar"] = out_ivar["pnl"]
        factor_sets["XGBoost+IVar"] = factor_sets["XGBoost合成"]
        factor_counts["XGBoost+IVar"] = factor_counts["XGBoost合成"]
    else:
        print("  警告：未找到 XGBoost composite，跳过")

    # 基准
    all_idx = list(strategies.values())[0].index
    b300 = bench_monthly("csi300", all_idx)
    b500 = bench_monthly("csi500", all_idx)
    strategies["沪深300"] = b300
    strategies["中证500"] = b500

    # 统一日期对齐
    common = all_idx
    for k in strategies:
        strategies[k] = strategies[k].reindex(common)

    factors_df = pd.read_csv(FF3_PATH, index_col=0, parse_dates=True)
    factors_df.index.name = "日期"

    rows, ann_rows = [], []
    for name, pnl in strategies.items():
        m = metrics(pnl)
        reg = ff3.regress_one(name, pnl, factors_df) if "XGB" in name or "线性" in name else {}
        ex300 = m["年化收益"] - metrics(b300)["年化收益"] if "线性" in name or "XGB" in name else np.nan
        ex500 = m["年化收益"] - metrics(b500)["年化收益"] if "线性" in name or "XGB" in name else np.nan
        diff300 = (pnl - b300).dropna()
        ir300 = (diff300.mean() / diff300.std(ddof=1) * np.sqrt(12)
                 if diff300.std(ddof=1) > 0 else np.nan) if "线性" in name or "XGB" in name else np.nan
        row = {"策略": name, "因子数": factor_counts.get(name, 0),
               **m, "超额沪深300": ex300, "IR沪深300": ir300,
               "超额中证500": ex500}
        if reg:
            row.update({"alpha年化": reg["alpha年化"], "alpha_t": reg["alpha_t"],
                        "alpha_p": reg["alpha_p"], "R2": reg["R2"],
                        "alpha_low": reg.get("alpha_95%下限", np.nan),
                        "alpha_high": reg.get("alpha_95%上限", np.nan)})
        rows.append(row)
        # 分年度
        yearly = pnl.groupby(pnl.index.year).apply(
            lambda x: (1 + x.fillna(0)).prod() - 1)
        for yr, v in yearly.items():
            ann_rows.append({"策略": name, "年份": int(yr), "收益": v})

    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(ann_rows).to_csv(OUT / "annual_returns.csv", index=False,
                                  encoding="utf-8-sig")
    pd.DataFrame([(k, "; ".join(v)) for k, v in factor_sets.items()],
                 columns=["策略", "因子"]).to_csv(OUT / "factor_sets.csv",
                                                 index=False, encoding="utf-8-sig")

    # 打印
    show = summ.copy()
    for c in ["累计收益", "年化收益", "年化波动", "最大回撤",
              "超额沪深300", "超额中证500", "alpha年化"]:
        if c in show:
            show[c] = (show[c] * 100).round(2)
    print(show.to_string(index=False))

    # 图1：净值对比
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name, pnl in strategies.items():
        nav = (1 + pnl.fillna(0)).cumprod()
        ax.plot(nav.index, nav.values, label=name, lw=2 if "XGB" in name else 1.4,
                color=COLORS.get(name))
    ax.set_title("候选策略 vs 基准 累计净值（Top30，2020-2025，单边千二）")
    ax.set_ylabel("累计净值"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "nav_compare.png", dpi=150); plt.close(fig)

    # 图2：分年度收益
    ann = pd.DataFrame(ann_rows).pivot(index="年份", columns="策略", values="收益")
    fig, ax = plt.subplots(figsize=(11, 5))
    ann.plot(kind="bar", ax=ax, color=[COLORS.get(c) for c in ann.columns])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("分年度收益对比"); ax.set_ylabel("年度收益")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "annual_returns.png", dpi=150); plt.close(fig)

    # 图3：回撤
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for name, pnl in strategies.items():
        nav = (1 + pnl.fillna(0)).cumprod()
        dd = nav / nav.cummax() - 1
        ax.plot(dd.index, dd.values, label=name, lw=1.6 if "XGB" in name else 1.1,
                color=COLORS.get(name))
    ax.set_title("回撤对比（Top30，2020-2025）"); ax.set_ylabel("回撤")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "drawdown.png", dpi=150); plt.close(fig)

    # 图4：FF3 alpha 对比（仅策略）
    strat = summ[summ["策略"].str.contains("线性|XGB")]
    if "alpha_low" in strat.columns:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = range(len(strat))
        ax.bar(x, strat["alpha年化"] * 100, yerr=[
            (strat["alpha年化"] - strat["alpha_low"]) * 100,
            (strat["alpha_high"] - strat["alpha年化"]) * 100],
            capsize=5, color=[COLORS.get(c, "gray") for c in strat["策略"]])
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(list(x)); ax.set_xticklabels(strat["策略"])
        ax.set_ylabel("FF3 alpha 年化（%）")
        ax.set_title("三因子模型 alpha 年化 + 95% 置信区间")
        for i, (_, r) in enumerate(strat.iterrows()):
            ax.text(i, r["alpha年化"] * 100, f"p={r['alpha_p']:.3f}",
                    ha="center", va="bottom", fontsize=8)
        fig.tight_layout(); fig.savefig(OUT / "ff3_alpha.png", dpi=150); plt.close(fig)

    print(f"\n结果目录：{OUT}")


if __name__ == "__main__":
    main()
