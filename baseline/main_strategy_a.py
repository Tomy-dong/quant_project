# -*- coding: utf-8 -*-
"""方案 A 主策略：固定因子 K=4 + ICIR 权重 + IVar 股票权重 + Top100。

同时回答“因子数超过 10 个是否更好”：固定因子下扫描 K ∈ {4,5,6,8,10,12,15,20}
（ICIR 权重 + IVar Top100），给出年化/夏普/回撤/alpha。

输出 results/main_strategy/：
    k_curve.csv / k_curve.png        因子数扩展曲线
    topn_curve.csv / topn_curve.png  持仓数曲线（K=4，IVar）
    final_compare.csv                固定因子/滚动因子/沪深300/中证500 + FF3 alpha
    monthly_pnl.csv                  主策略、滚动对照与双基准的月度收益
    nav_drawdown_main.png            主策略与双基准的独立净值—回撤双面板图
    nav_compare.png / drawdown.png / ff3_alpha.png
    holdings_main.xlsx               主策略逐月持仓
    factor_ic_grid.png / factor_ic_yearly_heatmap.png / ic_panels_top3.png
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
OUT = RESULT / "main_strategy"
OUT.mkdir(parents=True, exist_ok=True)
FF3_PATH = RESULT / "ff3" / "ff3_factors.csv"
COMSN = pv.COMSN
MAIN_K = 4
MAIN_TOPN = 100
K_GRID = [4, 5, 6, 8, 10, 12, 15, 20]
TOPN_GRID = [20, 30, 50, 100, 150, 200]
COLORS = {"固定因子(主策略)": "crimson", "滚动因子": "darkviolet",
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


def bt_ivar(comp, rets, top_n):
    common = comp.columns.intersection(rets.columns)
    w = o2w.build_weight_mats(comp[common], rets[common], top_n,
                              methods=["IVar"])["IVar"]
    return calc_nav(rets[common].fillna(0), w, comsn=COMSN)["pnl"], w


def bench_monthly(index, dates):
    close = load_benchmark(str(dates.min().date()),
                           str(dates.max().date()), index=index)["close"]
    return close.reindex(dates).pct_change(fill_method=None)


def main():
    print(f"方案 A：固定 K={MAIN_K} + ICIR + IVar Top{MAIN_TOPN}")
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

    def solve(y, K):
        opt1.K_TARGET = opt1.K_MIN = opt1.K_MAX = K
        problem = opt1.YearProblem(zfactors, returns, dates, diags[y],
                                   corrs[y], names, y)
        return opt1.solve_year(y, diags[y], corrs[y], names, problem)

    def comp_fixed(fit):
        c = pd.concat([rm.build_composite(zfactors, fit, y) for y in years]).sort_index()
        return c.loc[(c.index >= oos_s) & (c.index <= oos_e)]

    def comp_rolling(fits):
        c = pd.concat([rm.build_composite(zfactors, fits[y], y) for y in years]).sort_index()
        return c.loc[(c.index >= oos_s) & (c.index <= oos_e)]

    factors_df = pd.read_csv(FF3_PATH, index_col=0, parse_dates=True)
    factors_df.index.name = "日期"

    # ===== 1) 因子数扩展曲线（固定因子，IVar Top100）=====
    print("\n[1] 因子数扩展：K=4~20（固定因子，ICIR 权重，IVar Top100）")
    k_rows = []
    fitted = {}
    for K in K_GRID:
        fit = solve(years[0], K)
        fitted[K] = fit
        comp = comp_fixed(fit)
        pnl, _ = bt_ivar(comp, returns.loc[comp.index], MAIN_TOPN)
        m = metrics(pnl)
        reg = ff3.regress_one(f"K{K}", pnl, factors_df)
        k_rows.append({"K": K, "K实际": len(fit["subset"]), **m,
                       "alpha年化": reg["alpha年化"], "alpha_p": reg["alpha_p"]})
        print(f"  K={K:>2d}（实际 {len(fit['subset'])}）年化 {m['年化收益']*100:6.2f}% "
              f"夏普 {m['夏普比率']:.3f} 回撤 {m['最大回撤']*100:6.2f}% "
              f"alpha {reg['alpha年化']*100:+.2f}%(p={reg['alpha_p']:.3f})")
    kc = pd.DataFrame(k_rows)
    kc.to_csv(OUT / "k_curve.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, col, title, pct in [
        (axes[0], "年化收益", "年化收益（%）", True),
        (axes[1], "夏普比率", "夏普比率", False),
        (axes[2], "最大回撤", "最大回撤（%）", True),
    ]:
        v = kc[col] * (100 if pct else 1)
        ax.plot(kc["K"], v, "o-", color="#2f6f9f", lw=2)
        ax.axvline(MAIN_K, color="red", ls=":", lw=1, label=f"主策略 K={MAIN_K}")
        ax.axvline(10, color="gray", ls="--", lw=1, label="约束上限 K=10")
        ax.set_xlabel("因子数 K"); ax.set_ylabel(title); ax.set_title(title)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        for x, y in zip(kc["K"], v):
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8)
    fig.suptitle("因子数扩展（固定因子 + IVar Top100，2020-2025）", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "k_curve.png", dpi=150); plt.close(fig)

    # ===== 2) 持仓数曲线（K=4，IVar）=====
    print(f"\n[2] 持仓数曲线（K={MAIN_K}，IVar）")
    comp4 = comp_fixed(fitted[MAIN_K])
    t_rows = []
    for tn in TOPN_GRID:
        pnl, _ = bt_ivar(comp4, returns.loc[comp4.index], tn)
        m = metrics(pnl)
        t_rows.append({"TopN": tn, **m})
        print(f"  Top{tn:>3d} 年化 {m['年化收益']*100:6.2f}% 夏普 {m['夏普比率']:.3f} "
              f"回撤 {m['最大回撤']*100:6.2f}%")
    tc = pd.DataFrame(t_rows)
    tc.to_csv(OUT / "topn_curve.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for ax, col, title in [(axes[0], "年化收益", "年化收益（%）"),
                           (axes[1], "最大回撤", "最大回撤（%）")]:
        ax.plot(tc["TopN"], tc[col] * 100, "o-", color="#2f6f9f", lw=2)
        ax.axvline(MAIN_TOPN, color="red", ls=":", lw=1,
                   label=f"主策略 Top{MAIN_TOPN}")
        ax.axhline(10 if col == "年化收益" else -40, color="gray", ls="--",
                   lw=1, label="10% / -40%")
        ax.set_xlabel("持仓数 TopN"); ax.set_ylabel(title); ax.set_title(title)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle(f"持仓数曲线（固定 K={MAIN_K} + IVar，2020-2025）", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "topn_curve.png", dpi=150); plt.close(fig)

    # ===== 3) 最终对比：固定(主) / 滚动 / 双基准 =====
    print("\n[3] 最终对比")
    pnl_main, w_main = bt_ivar(comp4, returns.loc[comp4.index], MAIN_TOPN)
    fits_roll = {y: solve(y, MAIN_K) for y in years}
    comp_r = comp_rolling(fits_roll)
    pnl_roll, _ = bt_ivar(comp_r, returns.loc[comp_r.index], MAIN_TOPN)

    final = {"固定因子(主策略)": pnl_main, "滚动因子": pnl_roll}
    idx = pnl_main.index
    final["沪深300"] = bench_monthly("csi300", idx)
    final["中证500"] = bench_monthly("csi500", idx)

    ann300 = metrics(final["沪深300"])["年化收益"]
    ann500 = metrics(final["中证500"])["年化收益"]
    rows = []
    for name, pnl in final.items():
        m = metrics(pnl)
        row = {"策略": name, **m}
        if "因子" in name:
            reg = ff3.regress_one(name, pnl, factors_df)
            row.update({"超额沪深300": m["年化收益"] - ann300,
                        "超额中证500": m["年化收益"] - ann500,
                        "alpha年化": reg["alpha年化"], "alpha_t": reg["alpha_t"],
                        "alpha_p": reg["alpha_p"], "R2": reg["R2"],
                        "alpha_low": reg.get("alpha_95%下限"),
                        "alpha_high": reg.get("alpha_95%上限")})
        rows.append(row)
    fcmp = pd.DataFrame(rows)
    fcmp.to_csv(OUT / "final_compare.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(final).to_csv(OUT / "monthly_pnl.csv", encoding="utf-8-sig",
                               index_label="日期")
    print(fcmp.to_string(index=False))

    # 持仓导出
    hold = w_main.stack()
    hold = hold[hold > 1e-8].reset_index()
    hold.columns = ["日期", "股票代码", "权重"]
    hold.to_excel(OUT / "holdings_main.xlsx", index=False)
    print(f"  持仓已导出 holdings_main.xlsx（{len(hold)} 行）")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name in final:
        nav = (1 + final[name].fillna(0)).cumprod()
        ax.plot(nav.index, nav.values, label=name,
                lw=2.4 if "主策略" in name else 1.4, color=COLORS.get(name))
    ax.set_title(f"主策略：固定 K={MAIN_K} + ICIR + IVar Top{MAIN_TOPN}（2020-2025）")
    ax.set_ylabel("累计净值"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "nav_compare.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for name in final:
        nav = (1 + final[name].fillna(0)).cumprod()
        ax.plot(nav.index, (nav / nav.cummax() - 1).values, label=name,
                lw=1.9 if "主策略" in name else 1.1, color=COLORS.get(name))
    ax.set_title(f"回撤对比（Top{MAIN_TOPN}，2020-2025）"); ax.set_ylabel("回撤")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "drawdown.png", dpi=150); plt.close(fig)

    # 主结果表对应的独立图：仅保留主策略与双基准，不混入滚动方案或 Top30 附录。
    main_names = ["固定因子(主策略)", "沪深300", "中证500"]
    fig, (ax_nav, ax_dd) = plt.subplots(
        2, 1, figsize=(11, 8), sharex=True,
        gridspec_kw={"height_ratios": [1.55, 1.0], "hspace": 0.08})
    for name in main_names:
        nav = (1 + final[name].fillna(0)).cumprod()
        dd = nav / nav.cummax() - 1
        is_main = name == "固定因子(主策略)"
        label = f"主策略：固定 K={MAIN_K} + IVar Top{MAIN_TOPN}" if is_main else name
        ax_nav.plot(nav.index, nav.values, label=label,
                    lw=2.5 if is_main else 1.5, color=COLORS[name])
        ax_dd.plot(dd.index, dd.values, label=label,
                   lw=2.0 if is_main else 1.2, color=COLORS[name])
    ax_nav.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax_nav.set_title(f"主策略净值与回撤（2020-2025，单边千分之二，Top{MAIN_TOPN}）")
    ax_nav.set_ylabel("累计净值（初始=1）")
    ax_nav.legend(loc="upper left", ncol=3, fontsize=9)
    ax_nav.grid(alpha=0.3)
    ax_dd.axhline(-0.4, color="gray", lw=0.8, ls="--")
    ax_dd.set_ylabel("回撤")
    ax_dd.set_xlabel("日期")
    ax_dd.grid(alpha=0.3)
    fig.align_ylabels((ax_nav, ax_dd))
    fig.savefig(OUT / "nav_drawdown_main.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    sub = fcmp[fcmp["alpha年化"].notna()].copy()
    xs = np.arange(len(sub))
    ax.bar(xs, sub["alpha年化"] * 100,
           yerr=[(sub["alpha年化"] - sub["alpha_low"]) * 100,
                 (sub["alpha_high"] - sub["alpha年化"]) * 100],
           capsize=5, color=[COLORS.get(n, "gray") for n in sub["策略"]])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels(sub["策略"])
    ax.set_ylabel("FF3 alpha 年化（%）")
    ax.set_title("三因子回归：年化 alpha 与 95% 置信区间")
    for i, (_, r) in enumerate(sub.iterrows()):
        ax.text(i, r["alpha年化"] * 100,
                f"{r['alpha年化']*100:+.1f}%\np={r['alpha_p']:.3f}",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "ff3_alpha.png", dpi=150); plt.close(fig)

    # ===== 4) 4 个因子的 IC 图 =====
    feat = list(fitted[MAIN_K]["subset"])
    print(f"\n[4] 主策略因子（{len(feat)}）：{feat}")
    oic = {f: ic_series[f].loc[(ic_series[f].index >= oos_s) &
                               (ic_series[f].index <= oos_e)].dropna() for f in feat}
    fig, axes = plt.subplots(1, len(feat), figsize=(4.2 * len(feat), 3.4))
    for ax, f in zip(np.atleast_1d(axes), feat):
        s = oic[f]
        ax.bar(s.index, s.values, width=20, color="#4c78a8", alpha=0.85)
        ax.axhline(s.mean(), color="#d62728", lw=1.0, ls="--")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(f"{f}\n均值 {s.mean():.3f}  ICIR {s.mean()/s.std():.2f}", fontsize=9)
        ax.tick_params(labelsize=7)
    fig.suptitle("主策略入选因子：样本外（2020-2025）月度 RankIC", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "factor_ic_grid.png", dpi=140); plt.close(fig)

    yearly = pd.DataFrame({f: oic[f].groupby(oic[f].index.year).mean() for f in feat})
    fig, ax = plt.subplots(figsize=(7, 1.2 + 0.6 * len(feat)))
    vmax = np.nanmax(np.abs(yearly.values))
    im = ax.imshow(yearly.values, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(yearly.columns)))
    ax.set_xticklabels(yearly.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(yearly.index)))
    ax.set_yticklabels(yearly.index, fontsize=8)
    for i in range(len(yearly.index)):
        for j in range(len(yearly.columns)):
            ax.text(j, i, f"{yearly.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black")
    ax.set_title("入选因子 × 年份 平均 RankIC")
    fig.colorbar(im, ax=ax, fraction=0.05)
    fig.tight_layout(); fig.savefig(OUT / "factor_ic_yearly_heatmap.png", dpi=150)
    plt.close(fig)

    top3 = sorted(feat, key=lambda f: -abs(oic[f].mean()))[:min(3, len(feat))]
    fig, axes = plt.subplots(1, len(top3), figsize=(5 * len(top3), 3.8))
    for ax, f in zip(np.atleast_1d(axes), top3):
        s = oic[f]
        ax.bar(s.index, s.values, width=20, color="#4c78a8", alpha=0.8)
        ax2 = ax.twinx()
        ax2.plot(s.index, s.cumsum().values, color="#2ca02c", lw=1.6)
        ax.set_title(f"{f}（均值 {s.mean():.3f}）", fontsize=10)
        ax.tick_params(labelsize=7); ax2.tick_params(labelsize=7)
    fig.suptitle("前三因子：单期 IC 与累计 IC", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "ic_panels_top3.png", dpi=150); plt.close(fig)

    print(f"\n结果目录：{OUT}")


if __name__ == "__main__":
    main()
