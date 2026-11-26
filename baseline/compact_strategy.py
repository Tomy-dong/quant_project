# -*- coding: utf-8 -*-
"""精简策略实验：固定因子（2015-2019 选，带相关性剔除）× 因子数 × 因子权重。

目标：找到因子数 ≤10、Top30 下年化尽量高 / 回撤 <40% / 跑赢沪深300与中证500 /
FF3 alpha 尽量为正的配置。

实验矩阵：
  因子数 N ∈ {5, 6, 8, 10}（均 ≤10）
  因子权重 ∈ {ICIR（|ICIR|归一化）, longt（多头t）, EW（因子等权）}
  股票层：Top30 等权，单边千二费率
  选因子口径：2015-2019 训练窗口，门控通过（|ICIR|≥0.5 等），按 |ICIR| 降序
              贪心加入，两两 |corr| ≤ 0.7，取前 N 个。

输出 results/compact_strategy/：
    summary.csv         12 组配置的绩效 + 超额 + FF3 alpha
    factor_sets.csv     各 N 选中的因子明细
    n_vs_weight.png     因子数 × 权重 热力图（年化 / 回撤 / alpha）
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
OUT.mkdir(parents=True, exist_ok=True)
FF3_PATH = RESULT / "ff3" / "ff3_factors.csv"

N_GRID = [5, 6, 8, 10]
W_GRID = ["ICIR", "longt", "EW"]
TOP_N = 30
COMSN = pv.COMSN
SELECT_YEAR = rm.OOS_START_YEAR  # 2020 -> 训练窗口 2015-2019
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


def pick_factors(diag, corr, names, n, rho_max=RHO_MAX):
    """门控通过 -> |ICIR| 降序 -> 相关剔除 -> 取前 n 个。
    不足时放宽到 AbsICIR 前 n（仍做相关剔除）。"""
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
    """构造 fit dict 供 rm.build_composite 使用。"""
    sub = diag.set_index("因子").loc[chosen]
    if mode == "ICIR":
        raw = sub["AbsICIR"].to_numpy(dtype=float)
    elif mode == "longt":
        raw = np.clip(sub["多头t"].to_numpy(dtype=float), 0, None)
    else:  # EW
        raw = np.ones(len(chosen))
    w = raw / raw.sum() if raw.sum() > 0 else np.full(len(chosen), 1 / len(chosen))
    signs = sub["方向"].astype(int)
    picked = sub.reset_index().copy()
    picked["权重"] = w
    return {"weights": pd.Series(w, index=chosen),
            "signs": signs, "picked": picked}


def main():
    print(f"精简策略实验 | Top{TOP_N} | 费率 {COMSN:.4f} | "
          f"N ∈ {N_GRID} | 权重 ∈ {W_GRID}")
    print("=" * 72)

    print("\n[1/3] 载入因子面板、预计算统计量（约 2-3 分钟）...")
    monthly, factors, dates = fps.load_pool(mask_valid=True)
    returns = monthly["returns"]
    zfactors = {name: preprocess(fac) for name, fac in factors.items()}
    names = list(zfactors)
    opt1.ICIR_MIN = 0.5
    print(f"  因子池 {len(names)} 个，ICIR 下限 {opt1.ICIR_MIN}")

    ic_series = rm.rank_ic_panel(zfactors, returns)
    panels = rt.excess_panels(zfactors, returns)
    ls_panel = {n: panels[n][0] - panels[n][1] for n in names}
    size_beta, _ = opt1.compute_size_exposure(monthly, zfactors, dates)
    corr_mats, _ = opt1.compute_corr_mats(zfactors, dates)
    diag = opt1.build_diagnostics(ic_series, panels, ls_panel,
                                  size_beta, names, SELECT_YEAR)
    corr = opt1.average_corr(corr_mats, dates, SELECT_YEAR, names)
    print(f"  2015-2019 门控通过 {int(diag['门控通过'].sum())} / {len(diag)} 个因子")

    factors_df = pd.read_csv(FF3_PATH, index_col=0, parse_dates=True)
    factors_df.index.name = "日期"
    oos_start = pd.Timestamp(f"{rm.OOS_START_YEAR}-01-01")
    oos_end = pd.Timestamp(f"{rm.OOS_END_YEAR}-12-31")
    years = list(range(rm.OOS_START_YEAR, rm.OOS_END_YEAR + 1))

    bench_ret = {b: None for b in ["csi300", "csi500"]}

    rows, set_rows = [], []
    for N in N_GRID:
        chosen = pick_factors(diag, corr, names, N)
        set_rows.append({"N": N, "因子数": len(chosen),
                         "因子": "; ".join(chosen)})
        print(f"\n[N={N}] 选中 {len(chosen)} 个因子：{chosen}")
        for mode in W_GRID:
            fit = make_fit(diag, chosen, mode)
            comps = [rm.build_composite(zfactors, fit, y) for y in years]
            composite = pd.concat(comps).sort_index()
            composite = composite.loc[(composite.index >= oos_start) &
                                      (composite.index <= oos_end)]
            ret_oos = returns.loc[composite.index]
            pnl = backtest_top30(composite, ret_oos)

            if bench_ret["csi300"] is None:
                for b in bench_ret:
                    bench_ret[b] = bench_monthly(b, composite.index)
            m = metrics(pnl)
            bench_ann = {b: (1 + bench_ret[b].dropna()).prod() **
                         (12 / len(bench_ret[b].dropna())) - 1
                         for b in bench_ret}
            ex, ir = {}, {}
            for b, br in bench_ret.items():
                diff = (pnl - br).dropna()
                ex[b] = m["年化收益"] - bench_ann[b]
                ir[b] = (diff.mean() / diff.std(ddof=1) * np.sqrt(12)
                         if diff.std(ddof=1) > 0 else np.nan)
            reg = ff3.regress_one(f"N{N}_{mode}", pnl, factors_df)
            row = {"N": N, "权重": mode, **m,
                   "超额(沪深300)": ex["csi300"], "IR(沪深300)": ir["csi300"],
                   "超额(中证500)": ex["csi500"], "IR(中证500)": ir["csi500"],
                   "alpha年化": reg["alpha年化"], "alpha_t": reg["alpha_t"],
                   "alpha_p": reg["alpha_p"], "R2": reg["R2"],
                   "SMB": reg["SMB"]}
            rows.append(row)
            print(f"  {mode:<6s} 年化 {m['年化收益']*100:6.2f}% 夏普 {m['夏普比率']:.3f} "
                  f"回撤 {m['最大回撤']*100:6.2f}% | α300 {ex['csi300']*100:+5.2f}% "
                  f"α500 {ex['csi500']*100:+5.2f}% | alpha {reg['alpha年化']*100:+.2f}%"
                  f"(p={reg['alpha_p']:.3f})")

    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(set_rows).to_csv(OUT / "factor_sets.csv", index=False,
                                  encoding="utf-8-sig")

    # 热力图：N × 权重，指标 = 年化 / 回撤 / alpha
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col, title, fmt in [
        (axes[0], "年化收益", "年化收益（%）", "{:.1f}"),
        (axes[1], "最大回撤", "最大回撤（%）", "{:.1f}"),
        (axes[2], "alpha年化", "FF3 alpha 年化（%）", "{:.1f}"),
    ]:
        pivot = summ.pivot(index="N", columns="权重", values=col) * 100
        pivot = pivot[W_GRID]
        im = ax.imshow(pivot.values, cmap="RdYlGn" if col != "最大回撤" else "RdYlGn_r",
                       aspect="auto")
        ax.set_xticks(range(len(W_GRID))); ax.set_xticklabels(W_GRID)
        ax.set_yticks(range(len(N_GRID))); ax.set_yticklabels(N_GRID)
        ax.set_xlabel("因子权重"); ax.set_ylabel("因子数 N")
        ax.set_title(title)
        for i in range(len(N_GRID)):
            for j in range(len(W_GRID)):
                ax.text(j, i, fmt.format(pivot.values[i, j]),
                        ha="center", va="center", fontsize=9,
                        color="black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("精简策略：因子数 × 因子权重（Top30，2020-2025）", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "n_vs_weight.png", dpi=150)
    plt.close(fig)

    # 打印 + 挑最优
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
    print(show.to_string(index=False))

    # 达标筛选：年化≥15% 且 回撤>-40%
    ok = summ[(summ["年化收益"] >= 0.15) & (summ["最大回撤"] > -0.40)]
    if not ok.empty:
        best = ok.loc[ok["年化收益"].idxmax()]
        print(f"\n>>> 达标(年化≥15% & 回撤>-40%)最优：N={int(best['N'])} "
              f"权重={best['权重']}，年化 {best['年化收益']*100:.2f}%，"
              f"回撤 {best['最大回撤']*100:.2f}%")
    else:
        best = summ.loc[summ["年化收益"].idxmax()]
        print(f"\n>>> 无配置同时达标年化≥15% & 回撤>-40%。")
        print(f"    年化最高：N={int(best['N'])} 权重={best['权重']}，"
              f"年化 {best['年化收益']*100:.2f}%，回撤 {best['最大回撤']*100:.2f}%，"
              f"alpha {best['alpha年化']*100:+.2f}%(p={best['alpha_p']:.3f})")
        # 回撤最低
        lowdd = summ.loc[summ["最大回撤"].idxmax()]
        print(f"    回撤最低：N={int(lowdd['N'])} 权重={lowdd['权重']}，"
              f"回撤 {lowdd['最大回撤']*100:.2f}%，年化 {lowdd['年化收益']*100:.2f}%")

    print(f"\n结果目录：{OUT}")


if __name__ == "__main__":
    main()
