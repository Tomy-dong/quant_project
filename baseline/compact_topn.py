# -*- coding: utf-8 -*-
"""Top-N 持仓数敏感性：固定因子集(N=5/N=10) × 权重(EW/longt) × TopN。
摸清集中持仓能否提升年化到 12%+，以及回撤代价。
输出 results/compact_strategy/topn_sensitivity.csv 和 topn_sensitivity.png
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
OUT = RESULT / "compact_strategy"
FF3_PATH = RESULT / "ff3" / "ff3_factors.csv"

TOPN_GRID = [10, 15, 20, 30, 50]
N_FACTOR_GRID = [5, 10]
W_GRID = ["EW", "longt"]
SELECT_YEAR = rm.OOS_START_YEAR
RHO_MAX = 0.7


def metrics(pnl):
    s = pnl.dropna()
    if len(s) < 2:
        return {}
    nav = (1 + s).cumprod()
    total = float(nav.iloc[-1] - 1)
    years = len(s) / 12.0
    ann = float((1 + total) ** (1 / years) - 1)
    vol = float(s.std(ddof=1) * np.sqrt(12))
    return {"年化收益": ann, "年化波动": vol,
            "夏普比率": float(ann / vol) if vol > 0 else np.nan,
            "最大回撤": float((nav / nav.cummax() - 1).min())}


def backtest(composite, returns, top_n):
    dates = composite.index
    codes_all = composite.columns
    weights = pd.DataFrame(0.0, index=dates, columns=codes_all)
    for i in range(len(dates)):
        row = composite.iloc[i].dropna()
        if len(row) >= top_n:
            sel = row.nlargest(top_n).index
            weights.iloc[i][sel] = 1.0 / top_n
    out = calc_nav(returns.fillna(0), weights, comsn=pv.COMSN)
    return out["pnl"]


def pick_factors(diag, corr, names, n, rho_max=RHO_MAX):
    pos = {nm: i for i, nm in enumerate(names)}
    eligible = diag[diag["门控通过"]].sort_values("AbsICIR", ascending=False)
    chosen = []
    for name in eligible["因子"]:
        if len(chosen) >= n:
            break
        idx = pos[name]
        if all(abs(corr[idx, pos[c]]) <= rho_max for c in chosen):
            chosen.append(name)
    if len(chosen) < n:
        for name in diag.sort_values("AbsICIR", ascending=False)["因子"]:
            if len(chosen) >= n:
                break
            if name in chosen:
                continue
            idx = pos[name]
            if all(abs(corr[idx, pos[c]]) <= rho_max for c in chosen):
                chosen.append(name)
    return chosen[:n]


def make_fit(diag, chosen, mode):
    sub = diag.set_index("因子").loc[chosen]
    if mode == "longt":
        raw = np.clip(sub["多头t"].to_numpy(dtype=float), 0, None)
    else:
        raw = np.ones(len(chosen))
    w = raw / raw.sum() if raw.sum() > 0 else np.full(len(chosen), 1 / len(chosen))
    return {"weights": pd.Series(w, index=chosen),
            "signs": sub["方向"].astype(int)}


def main():
    print(f"Top-N 敏感性 | 因子N ∈ {N_FACTOR_GRID} | 权重 ∈ {W_GRID} | "
          f"TopN ∈ {TOPN_GRID}")
    print("载入因子面板与统计量（约 2-3 分钟）...")
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

    factors_df = pd.read_csv(FF3_PATH, index_col=0, parse_dates=True)
    factors_df.index.name = "日期"
    years = list(range(rm.OOS_START_YEAR, rm.OOS_END_YEAR + 1))

    rows = []
    for nf in N_FACTOR_GRID:
        chosen = pick_factors(diag, corr, names, nf)
        for mode in W_GRID:
            fit = make_fit(diag, chosen, mode)
            comps = [rm.build_composite(zfactors, fit, y) for y in years]
            composite = pd.concat(comps).sort_index()
            ret_oos = returns.loc[composite.index]
            for tn in TOPN_GRID:
                pnl = backtest(composite, ret_oos, tn)
                m = metrics(pnl)
                reg = ff3.regress_one(f"F{nf}_{mode}_T{tn}", pnl, factors_df)
                row = {"因子数": nf, "权重": mode, "TopN": tn, **m,
                       "alpha年化": reg["alpha年化"], "alpha_p": reg["alpha_p"]}
                rows.append(row)
                print(f"  F{nf} {mode:<6s} Top{tn:>2d}  年化 {m['年化收益']*100:6.2f}% "
                      f"夏普 {m['夏普比率']:.3f} 回撤 {m['最大回撤']*100:6.2f}% "
                      f"alpha {reg['alpha年化']*100:+.2f}%(p={reg['alpha_p']:.3f})")

    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "topn_sensitivity.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, col, title in [(axes[0], "年化收益", "年化收益（%）"),
                           (axes[1], "最大回撤", "最大回撤（%）")]:
        for nf in N_FACTOR_GRID:
            for mode in W_GRID:
                sub = summ[(summ["因子数"] == nf) & (summ["权重"] == mode)]
                ax.plot(sub["TopN"], sub[col] * 100, marker="o",
                        label=f"F{nf}-{mode}")
        ax.axhline(15 if col == "年化收益" else -40, color="red", ls="--",
                   lw=1, label="目标 15% / -40%")
        ax.set_xlabel("持仓数 TopN"); ax.set_ylabel(title); ax.set_title(title)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("Top-N 持仓数敏感性（固定因子，2020-2025）", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "topn_sensitivity.png", dpi=150)
    plt.close(fig)

    best = summ.loc[summ["年化收益"].idxmax()]
    print(f"\n>>> 年化最高：F{int(best['因子数'])} {best['权重']} "
          f"Top{int(best['TopN'])}，年化 {best['年化收益']*100:.2f}%，"
          f"回撤 {best['最大回撤']*100:.2f}%")
    print(f"结果：{OUT / 'topn_sensitivity.csv'}")


if __name__ == "__main__":
    main()
