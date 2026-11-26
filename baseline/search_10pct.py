# -*- coding: utf-8 -*-
"""在 <=10 因子约束下搜索年化 >10% 的配置。

扫描：模式(固定/滚动) × K(4,5,10) × 因子权重(ICIR/多头t) × 股票权重(EW/IVar) × TopN(30,50,100)。
口径与 final_pipeline 一致：优化器一 ICIR>=0.5、共线<=0.7、波动分位 0.7，单边千二。
输出 results/search_10pct/summary.csv + annual_vs_dd.png
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
import optimizer2_weights as o2w            # noqa: E402
import factor_pool_source as fps            # noqa: E402
import ff3_regression as ff3                # noqa: E402
from data_io import load_benchmark          # noqa: E402
from utils.data_standardize import preprocess  # noqa: E402
from utils.utils import calc_nav             # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

RESULT = HERE.parent / "results"
OUT = RESULT / "search_10pct"
OUT.mkdir(parents=True, exist_ok=True)
FF3_PATH = RESULT / "ff3" / "ff3_factors.csv"
COMSN = pv.COMSN


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


def backtest_ew(comp, rets, top_n):
    dates = comp.index
    codes = comp.columns
    w = pd.DataFrame(0.0, index=dates, columns=codes)
    for i in range(len(dates)):
        row = comp.iloc[i].dropna()
        if len(row) >= top_n:
            w.iloc[i][row.nlargest(top_n).index] = 1.0 / top_n
    return calc_nav(rets.fillna(0), w, comsn=COMSN)["pnl"]


def main():
    print("<=10 因子下搜索年化 >10% 的配置")
    monthly, factors, dates = fps.load_pool(mask_valid=True)
    returns = monthly["returns"]
    zfactors = {n: preprocess(f) for n, f in factors.items()}
    names = list(zfactors)
    opt1.ICIR_MIN = 0.5
    ic_series = rm.rank_ic_panel(zfactors, returns)
    panels = rt.excess_panels(zfactors, returns)
    ls_panel = {n: panels[n][0] - panels[n][1] for n in names}
    size_beta, _ = opt1.compute_size_exposure(monthly, zfactors, dates)
    corr_mats, _ = opt1.compute_corr_mats(zfactors, dates)
    years = list(range(opt1.OOS_START_YEAR, opt1.OOS_END_YEAR + 1))
    oos_s = pd.Timestamp(f"{opt1.OOS_START_YEAR}-01-01")
    oos_e = pd.Timestamp(f"{opt1.OOS_END_YEAR}-12-31")

    diags, corrs = {}, {}
    for y in years:
        diags[y] = opt1.build_diagnostics(ic_series, panels, ls_panel,
                                          size_beta, names, y)
        corrs[y] = opt1.average_corr(corr_mats, dates, y, names)

    # 因子权重构造
    def fit_longt(diag, subset):
        d = diag.set_index("因子").loc[subset]
        t = np.clip(d["多头t"].to_numpy(float), 0, None)
        w = t / t.sum() if t.sum() > 0 else np.full(len(subset), 1 / len(subset))
        return {"weights": pd.Series(w, index=subset),
                "signs": d["方向"].astype(int)}

    factors_df = pd.read_csv(FF3_PATH, index_col=0, parse_dates=True)
    factors_df.index.name = "日期"

    rows = []
    for K in (4, 5, 10):
        opt1.K_TARGET = opt1.K_MIN = opt1.K_MAX = K
        fits = {}
        for y in years:
            problem = opt1.YearProblem(zfactors, returns, dates, diags[y],
                                       corrs[y], names, y)
            fits[y] = opt1.solve_year(y, diags[y], corrs[y], names, problem)
        for mode in ("固定", "滚动"):
            # 构造两种因子权重的 composite
            comps = {}
            if mode == "固定":
                f0 = fits[years[0]]
                comps["ICIR"] = pd.concat(
                    [rm.build_composite(zfactors, f0, y) for y in years]).sort_index()
                comps["多头t"] = pd.concat(
                    [rm.build_composite(zfactors, fit_longt(diags[years[0]], f0["subset"]), y)
                     for y in years]).sort_index()
            else:
                comps["ICIR"] = pd.concat(
                    [rm.build_composite(zfactors, fits[y], y) for y in years]).sort_index()
                comps["多头t"] = pd.concat(
                    [rm.build_composite(zfactors, fit_longt(diags[y], fits[y]["subset"]), y)
                     for y in years]).sort_index()
            for fw, comp in comps.items():
                comp = comp.loc[(comp.index >= oos_s) & (comp.index <= oos_e)]
                rets = returns.loc[comp.index]
                for sw in ("EW", "IVar"):
                    for tn in (30, 50, 100):
                        if sw == "EW":
                            pnl = backtest_ew(comp, rets, tn)
                        else:
                            common = comp.columns.intersection(rets.columns)
                            w = o2w.build_weight_mats(comp[common], rets[common], tn,
                                                      methods=["IVar"])["IVar"]
                            pnl = calc_nav(rets[common].fillna(0), w, comsn=COMSN)["pnl"]
                        m = metrics(pnl)
                        reg = ff3.regress_one("x", pnl, factors_df)
                        rows.append({"模式": mode, "K": K, "因子权重": fw,
                                     "股票权重": sw, "TopN": tn, **m,
                                     "alpha年化": reg["alpha年化"],
                                     "alpha_p": reg["alpha_p"]})
                        print(f"  {mode} K={K} {fw}/{sw} Top{tn}: "
                              f"年化 {m['年化收益']*100:6.2f}% 夏普 {m['夏普比率']:.3f} "
                              f"回撤 {m['最大回撤']*100:6.2f}% alpha {reg['alpha年化']*100:+.2f}%")

    summ = pd.DataFrame(rows).sort_values("年化收益", ascending=False)
    summ.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")

    # 图：年化 vs 回撤散点，标注 >10% 且回撤>-40% 的点
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ok = (summ["年化收益"] >= 0.10) & (summ["最大回撤"] > -0.40)
    ax.scatter(summ.loc[~ok, "最大回撤"] * 100, summ.loc[~ok, "年化收益"] * 100,
               c="lightgray", label="其他")
    ax.scatter(summ.loc[ok, "最大回撤"] * 100, summ.loc[ok, "年化收益"] * 100,
               c="crimson", label="年化≥10% 且回撤<40%")
    for _, r in summ[ok].iterrows():
        ax.annotate(f"{r['模式'][0]}K{r['K']}-{r['因子权重']}/{r['股票权重']}-T{r['TopN']}",
                    (r["最大回撤"] * 100, r["年化收益"] * 100),
                    textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7)
    ax.axhline(10, color="red", ls=":", lw=1)
    ax.axvline(-40, color="red", ls=":", lw=1)
    ax.set_xlabel("最大回撤（%）"); ax.set_ylabel("年化收益（%）")
    ax.set_title("≤10 因子搜索：年化 与 最大回撤")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "annual_vs_dd.png", dpi=150); plt.close(fig)

    print("\n达标（年化≥10% 且回撤>-40%）前 10：")
    print(summ[ok].head(10).to_string(index=False))
    print(f"\n结果：{OUT}")


if __name__ == "__main__":
    main()
