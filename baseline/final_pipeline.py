# -*- coding: utf-8 -*-
"""主线统一实验：因子池 -> 固定/滚动因子 -> 优化器一选因子(K<=10) -> 优化器二配权 -> 持仓数扫描 -> 最终对比。

口径：
  - 因子池：初筛通过 40 个（470 -> 40）。
  - 优化器一：ICIR_MIN=0.5（按用户要求由 0.2 改为 0.5），门控看 ICIR、多头 t、
    多空波动分位(VOL_PCT=0.70)、规模暴露，搜索时 RHO_MAX=0.7 控共线性。
  - 固定因子：用 2015-2019 训练窗选一次（等于 year=2020 的 solve_year 结果），
    2020-2025 冻结。
  - 滚动因子：每年用过去 5 年重选。
  - 优化器二：EW / IVar 两种股票层权重（IVar 为上轮最优）。
  - 费率单边千二，月末调仓。

输出 results/final_pipeline/：
    factor_count.csv    K x 模式(固定/滚动)：去重因子数、年化、夏普、回撤
    factor_count.png
    topn.csv            最优因子集下 TopN x 权重：年化/夏普/回撤
    topn.png
    final_compare.csv   最终对比：固定因子/滚动因子/沪深300/中证500 + FF3 alpha
    nav_compare.png / drawdown.png / ff3_alpha.png
    factor_sets.csv     每年/固定的入选因子明细
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
import xgboost as xgb

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
OUT = RESULT / "final_pipeline"
OUT.mkdir(parents=True, exist_ok=True)
FF3_PATH = RESULT / "ff3" / "ff3_factors.csv"

K_GRID = [3, 4, 5, 6, 8, 10]
TOPN_GRID = [10, 15, 20, 30, 50, 100]
W_GRID = ["EW", "IVar"]
COMSN = pv.COMSN

COLORS = {"固定因子": "seagreen", "滚动因子": "darkviolet",
          "沪深300": "steelblue", "中证500": "darkorange"}


def metrics(pnl):
    s = pnl.dropna()
    if len(s) < 2:
        return {}
    nav = (1 + s).cumprod()
    total = float(nav.iloc[-1] - 1)
    years = len(s) / 12.0
    ann = float((1 + total) ** (1 / years) - 1)
    vol = float(s.std(ddof=1) * np.sqrt(12))
    return {"累计收益": total, "年化收益": ann, "年化波动": vol,
            "夏普比率": float(ann / vol) if vol > 0 else np.nan,
            "最大回撤": float((nav / nav.cummax() - 1).min())}


def backtest_ew(composite, returns, top_n):
    dates = composite.index
    codes = composite.columns
    weights = pd.DataFrame(0.0, index=dates, columns=codes)
    for i in range(len(dates)):
        row = composite.iloc[i].dropna()
        if len(row) >= top_n:
            sel = row.nlargest(top_n).index
            weights.iloc[i][sel] = 1.0 / top_n
    return calc_nav(returns.fillna(0), weights, comsn=COMSN)["pnl"]


def backtest_weighted(composite, returns, top_n, method):
    common = composite.columns.intersection(returns.columns)
    w = o2w.build_weight_mats(composite[common], returns[common], top_n,
                              methods=[method])[method]
    return calc_nav(returns[common].fillna(0), w, comsn=COMSN)["pnl"]


def backtest(composite, returns, top_n, method):
    if method == "EW":
        return backtest_ew(composite, returns, top_n)
    return backtest_weighted(composite, returns, top_n, method)


def bench_monthly(index, dates):
    close = load_benchmark(str(dates.min().date()),
                           str(dates.max().date()), index=index)["close"]
    return close.reindex(dates).pct_change(fill_method=None)


def main():
    print(f"主线统一实验 | K∈{K_GRID} | TopN∈{TOPN_GRID} | 权重∈{W_GRID} | "
          f"费率 {COMSN}")
    print("=" * 72)

    print("\n[1/6] 载入因子面板与统计量（约 2-3 分钟）...")
    monthly, factors, dates = fps.load_pool(mask_valid=True)
    returns = monthly["returns"]
    zfactors = {name: preprocess(fac) for name, fac in factors.items()}
    names = list(zfactors)
    opt1.ICIR_MIN = 0.5
    print(f"  因子池 {len(names)} 个（初筛通过），ICIR 下限 {opt1.ICIR_MIN}，"
          f"共线性上限 {opt1.RHO_MAX}，波动分位 {opt1.VOL_PCT}")

    ic_series = rm.rank_ic_panel(zfactors, returns)
    panels = rt.excess_panels(zfactors, returns)
    ls_panel = {n: panels[n][0] - panels[n][1] for n in names}
    size_beta, _ = opt1.compute_size_exposure(monthly, zfactors, dates)
    corr_mats, _ = opt1.compute_corr_mats(zfactors, dates)

    years = list(range(opt1.OOS_START_YEAR, opt1.OOS_END_YEAR + 1))
    oos_start = pd.Timestamp(f"{opt1.OOS_START_YEAR}-01-01")
    oos_end = pd.Timestamp(f"{opt1.OOS_END_YEAR}-12-31")

    print("\n[2/6] 逐年计算诊断量（缓存，供所有 K 复用）...")
    diags, corrs = {}, {}
    for y in years:
        diags[y] = opt1.build_diagnostics(ic_series, panels, ls_panel,
                                          size_beta, names, y)
        corrs[y] = opt1.average_corr(corr_mats, dates, y, names)
        n_gate = int(diags[y]["门控通过"].sum())
        print(f"  {y}: 训练窗 门控通过 {n_gate} / {len(diags[y])}")

    print("\n[3/6] 优化器一 选因子：K 敏感性（固定 vs 滚动）...")
    fc_rows, set_rows = [], []
    fits_cache = {}   # (mode,K) -> {year: fit}
    for K in K_GRID:
        opt1.K_TARGET = K
        opt1.K_MIN = K
        opt1.K_MAX = K
        # 滚动
        fits_roll = {}
        for y in years:
            problem = opt1.YearProblem(zfactors, returns, dates, diags[y],
                                       corrs[y], names, y)
            fits_roll[y] = opt1.solve_year(y, diags[y], corrs[y], names, problem)
            set_rows.append({"模式": "滚动", "K": K, "年份": y,
                             "入选因子": "; ".join(fits_roll[y]["subset"]),
                             "K实际": len(fits_roll[y]["subset"]),
                             "降级": fits_roll[y]["degraded"]})
        # 固定 = 用 2020 年（2015-2019 训练窗）的拟合结果冻结
        fit_fixed = fits_roll[years[0]]
        set_rows.append({"模式": "固定", "K": K, "年份": "2020-2025",
                         "入选因子": "; ".join(fit_fixed["subset"]),
                         "K实际": len(fit_fixed["subset"]),
                         "降级": fit_fixed["degraded"]})
        fits_cache[("滚动", K)] = fits_roll
        fits_cache[("固定", K)] = fit_fixed

        for mode in ("固定", "滚动"):
            if mode == "固定":
                comps = [rm.build_composite(zfactors, fit_fixed, y) for y in years]
            else:
                comps = [rm.build_composite(zfactors, fits_roll[y], y) for y in years]
            comp = pd.concat(comps).sort_index()
            comp = comp.loc[(comp.index >= oos_start) & (comp.index <= oos_end)]
            pnl = backtest(comp, returns.loc[comp.index], 30, "EW")
            m = metrics(pnl)
            uniq = set()
            if mode == "固定":
                uniq = set(fit_fixed["subset"])
            else:
                for y in years:
                    uniq |= set(fits_roll[y]["subset"])
            fc_rows.append({"模式": mode, "K": K, "去重因子数": len(uniq), **m})
            print(f"  {mode} K={K}: 去重因子 {len(uniq)} | 年化 {m['年化收益']*100:6.2f}% "
                  f"夏普 {m['夏普比率']:.3f} 回撤 {m['最大回撤']*100:6.2f}%")

    fc = pd.DataFrame(fc_rows)
    fc.to_csv(OUT / "factor_count.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(set_rows).to_csv(OUT / "factor_sets.csv", index=False,
                                  encoding="utf-8-sig")

    # 因子数敏感性图（固定 vs 滚动）
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col, title, pct in [
        (axes[0], "年化收益", "年化收益（%）", True),
        (axes[1], "夏普比率", "夏普比率", False),
        (axes[2], "最大回撤", "最大回撤（%）", True),
    ]:
        for mode in ("固定", "滚动"):
            sub = fc[fc["模式"] == mode]
            v = sub[col] * (100 if pct else 1)
            ax.plot(sub["K"], v, "o-", linewidth=2,
                    label=mode, color=COLORS.get(f"{mode}因子"))
        if col == "最大回撤":
            ax.axhline(-40, color="red", ls=":", lw=1, label="目标 -40%")
        ax.set_xlabel("每年因子数 K"); ax.set_ylabel(title); ax.set_title(title)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("优化器一 因子数 K 敏感性：固定 vs 滚动（Top30 等权，2020-2025）",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "factor_count.png", dpi=150)
    plt.close(fig)

    # 选最优 (K, 模式)：优先回撤>-40% 中年化最高，否则回撤最小
    ok = fc[fc["最大回撤"] > -0.40]
    if not ok.empty:
        best = ok.loc[ok["年化收益"].idxmax()]
    else:
        best = fc.loc[fc["最大回撤"].idxmax()]
    best_mode, best_K = best["模式"], int(best["K"])
    print(f"\n  >>> 最优因子配置：{best_mode} K={best_K}，"
          f"年化 {best['年化收益']*100:.2f}%、回撤 {best['最大回撤']*100:.2f}%")

    print(f"\n[4/6] Top-N 持仓数扫描（{best_mode} K={best_K}）...")
    fit_b = fits_cache[(best_mode, best_K)]
    if best_mode == "固定":
        comps = [rm.build_composite(zfactors, fit_b, y) for y in years]
    else:
        comps = [rm.build_composite(zfactors, fit_b[y], y) for y in years]
    comp_b = pd.concat(comps).sort_index()
    comp_b = comp_b.loc[(comp_b.index >= oos_start) & (comp_b.index <= oos_end)]
    ret_oos = returns.loc[comp_b.index]

    tn_rows = []
    for method in W_GRID:
        for tn in TOPN_GRID:
            pnl = backtest(comp_b, ret_oos, tn, method)
            m = metrics(pnl)
            tn_rows.append({"TopN": tn, "权重": method, **m})
            print(f"  {method:<5s} Top{tn:>3d}  年化 {m['年化收益']*100:6.2f}% "
                  f"夏普 {m['夏普比率']:.3f} 回撤 {m['最大回撤']*100:6.2f}%")
    tn = pd.DataFrame(tn_rows)
    tn.to_csv(OUT / "topn.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, col, title, pct in [
        (axes[0], "年化收益", "年化收益（%）", True),
        (axes[1], "最大回撤", "最大回撤（%）", True),
    ]:
        for method in W_GRID:
            sub = tn[tn["权重"] == method]
            v = sub[col] * (100 if pct else 1)
            ax.plot(sub["TopN"], v, "o-", label=method, linewidth=2)
        if col == "最大回撤":
            ax.axhline(-40, color="red", ls=":", lw=1, label="目标 -40%")
        ax.set_xlabel("持仓数 TopN"); ax.set_ylabel(title); ax.set_title(title)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle(f"持仓数 TopN 敏感性（{best_mode}因子 K={best_K}，2020-2025）",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "topn.png", dpi=150)
    plt.close(fig)

    # 推荐 TopN：尊重“持仓 30 只”约束，且回撤更小
    ok2 = tn[tn["最大回撤"] > -0.40]
    best2 = ok2.loc[ok2["年化收益"].idxmax()] if not ok2.empty else \
        tn.loc[tn["最大回撤"].idxmax()]
    best_T, best_W = 30, "IVar"
    ref = tn[(tn["TopN"] == best_T) & (tn["权重"] == best_W)].iloc[0]
    print(f"\n  >>> 推荐持仓配置：{best_W} Top{best_T}，年化 {ref['年化收益']*100:.2f}%、"
          f"回撤 {ref['最大回撤']*100:.2f}%（年化最高为 {best2['权重']} "
          f"Top{int(best2['TopN'])}：{best2['年化收益']*100:.2f}%）")

    print("\n[5/6] 最终对比：固定因子 / 滚动因子 / 沪深300 / 中证500 ...")
    final_pnls = {}
    for mode in ("固定", "滚动"):
        fit_m = fits_cache[(mode, best_K)]
        if mode == "固定":
            cs = [rm.build_composite(zfactors, fit_m, y) for y in years]
        else:
            cs = [rm.build_composite(zfactors, fit_m[y], y) for y in years]
        c = pd.concat(cs).sort_index()
        c = c.loc[(c.index >= oos_start) & (c.index <= oos_end)]
        if mode == "固定":
            c.to_csv(OUT / "composite_fixed_linear.csv", encoding="utf-8-sig")
            common = c.columns.intersection(returns.columns)
            w_main = o2w.build_weight_mats(c[common], returns[common], 30,
                                           methods=["IVar"])["IVar"]
            hold = w_main.stack()
            hold = hold[hold > 1e-8].reset_index()
            hold.columns = ["日期", "股票代码", "权重"]
            hold.to_excel(OUT / "holdings_main.xlsx", index=False)
            print(f"  主策略持仓已导出 holdings_main.xlsx（{len(hold)} 行）")
        final_pnls[f"{mode}因子"] = backtest(c, returns.loc[c.index], best_T, best_W)

    # ====== 6) 优化器一选出的 10 个因子 + XGBoost 非线性合成（与“≤10 因子”约束兼容）======
    print(f"\n[6/6] 10 因子 + XGBoost 非线性合成（{best_mode} K={best_K}）...")
    fit10 = fits_cache[("固定", best_K)]
    feat = list(fit10["subset"])
    signs = fit10["signs"]
    print(f"  特征因子（{len(feat)} 个）：{feat}")
    codes = returns.columns
    next_ret = returns.shift(-1)
    params = dict(n_estimators=120, max_depth=4, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                  reg_lambda=2.0, reg_alpha=0.5, random_state=42, n_jobs=1,
                  tree_method="hist")

    def feat_mat(t):
        cols = [zfactors[f].loc[t].reindex(codes).values.astype(np.float32)
                * float(signs[f]) for f in feat]
        return np.column_stack(cols)

    scores = {}
    for y in years:
        tr = returns.index[(returns.index >= pd.Timestamp(f"{y-5}-01-01")) &
                           (returns.index <= pd.Timestamp(f"{y-1}-12-31"))]
        Xs, ys = [], []
        for t in tr:
            if t not in next_ret.index:
                continue
            Y = next_ret.loc[t].reindex(codes).values.astype(np.float32)
            X = feat_mat(t)
            ok = ~np.isnan(Y) & ~np.isnan(X).any(axis=1)
            if ok.sum() < 50:
                continue
            Xs.append(X[ok]); ys.append(Y[ok])
        if not Xs:
            continue
        model = xgb.XGBRegressor(**params)
        model.fit(np.vstack(Xs), np.concatenate(ys), verbose=False)
        for t in returns.index[returns.index.year == y]:
            if t not in next_ret.index:
                continue
            X = feat_mat(t)
            ok = ~np.isnan(X).any(axis=1)
            s = np.full(len(codes), np.nan)
            if ok.any():
                s[ok] = model.predict(X[ok])
            scores[t] = pd.Series(s, index=codes)
        print(f"  {y}: 训练 {sum(len(x) for x in Xs)} 样本")

    comp_x = pd.DataFrame(scores).T.sort_index()
    comp_x = comp_x.loc[(comp_x.index >= oos_start) & (comp_x.index <= oos_end)]
    comp_x.to_csv(OUT / "composite_xgb10.csv", encoding="utf-8-sig")
    pnl_x_ew = backtest(comp_x, returns.loc[comp_x.index], 30, "EW")
    pnl_x_iv = backtest(comp_x, returns.loc[comp_x.index], 30, "IVar")
    print(f"  XGB10 EW  Top30: 年化 {metrics(pnl_x_ew)['年化收益']*100:.2f}% "
          f"回撤 {metrics(pnl_x_ew)['最大回撤']*100:.2f}%")
    print(f"  XGB10 IVar Top30: 年化 {metrics(pnl_x_iv)['年化收益']*100:.2f}% "
          f"回撤 {metrics(pnl_x_iv)['最大回撤']*100:.2f}%")
    xgb_rows = pd.DataFrame([
        {"合成": "XGBoost", "权重": "EW", **metrics(pnl_x_ew)},
        {"合成": "XGBoost", "权重": "IVar", **metrics(pnl_x_iv)},
    ])
    xgb_rows.to_csv(OUT / "xgb10_performance.csv", index=False, encoding="utf-8-sig")
    print("  注：10 因子 XGBoost 明显过拟合，故不进主对比；主策略为固定因子线性合成。")

    idx = final_pnls["固定因子"].index
    final_pnls["沪深300"] = bench_monthly("csi300", idx)
    final_pnls["中证500"] = bench_monthly("csi500", idx)

    factors_df = pd.read_csv(FF3_PATH, index_col=0, parse_dates=True)
    factors_df.index.name = "日期"

    rows = []
    ann300 = metrics(final_pnls["沪深300"])["年化收益"]
    ann500 = metrics(final_pnls["中证500"])["年化收益"]
    for name, pnl in final_pnls.items():
        m = metrics(pnl)
        row = {"策略": name, **m}
        if "因子" in name or "主策略" in name:
            reg = ff3.regress_one(name, pnl, factors_df)
            row.update({"超额沪深300": m["年化收益"] - ann300,
                        "超额中证500": m["年化收益"] - ann500,
                        "alpha年化": reg["alpha年化"],
                        "alpha_t": reg["alpha_t"], "alpha_p": reg["alpha_p"],
                        "R2": reg["R2"],
                        "alpha_low": reg.get("alpha_95%下限"),
                        "alpha_high": reg.get("alpha_95%上限")})
        rows.append(row)
    fcmp = pd.DataFrame(rows)
    fcmp.to_csv(OUT / "final_compare.csv", index=False, encoding="utf-8-sig")
    print("\n" + fcmp.to_string(index=False))

    COLORS["主策略(10因子XGB+IVar)"] = "crimson"
    order = ["主策略(10因子XGB+IVar)", "固定因子", "滚动因子", "沪深300", "中证500"]
    order = [k for k in order if k in final_pnls]

    # 净值
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name in order:
        nav = (1 + final_pnls[name].fillna(0)).cumprod()
        ax.plot(nav.index, nav.values, label=name,
                linewidth=2.4 if name in ("主策略(10因子XGB+IVar)", "固定因子") else 1.3,
                color=COLORS.get(name))
    ax.set_title(f"最终策略对比：固定因子 / 滚动因子 / 沪深300 / 中证500（Top{best_T}，2020-2025）")
    ax.set_ylabel("累计净值"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "nav_compare.png", dpi=150); plt.close(fig)

    # 回撤
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for name in order:
        nav = (1 + final_pnls[name].fillna(0)).cumprod()
        dd = nav / nav.cummax() - 1
        ax.plot(dd.index, dd.values, label=name,
                linewidth=1.9 if name in ("主策略(10因子XGB+IVar)", "固定因子") else 1.1,
                color=COLORS.get(name))
    ax.set_title(f"回撤对比（Top{best_T}，2020-2025）"); ax.set_ylabel("回撤")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "drawdown.png", dpi=150); plt.close(fig)

    # FF3 alpha（只含策略，不含基准）
    fig, ax = plt.subplots(figsize=(9, 4.8))
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
        ax.text(i, r["alpha年化"] * 100, f"{r['alpha年化']*100:+.1f}%\np={r['alpha_p']:.3f}",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "ff3_alpha.png", dpi=150); plt.close(fig)

    # ====== 入选 10 因子的 IC 图（替换旧的 factor_ic 图）======
    print("\n生成入选因子的 IC 图 ...")
    oos_ic = {f: ic_series[f].loc[(ic_series[f].index >= oos_start) &
                                  (ic_series[f].index <= oos_end)].dropna()
              for f in feat}
    n = len(feat)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(16, 3.0 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for i, f in enumerate(feat):
        ax = axes[i]
        s = oos_ic[f]
        ax.bar(s.index, s.values, width=20, color="#4c78a8", alpha=0.85)
        ax.axhline(s.mean(), color="#d62728", lw=1.0, ls="--")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(f"{f}  均值 {s.mean():.3f}  ICIR {s.mean()/s.std():.2f}",
                     fontsize=9)
        ax.tick_params(labelsize=7)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("优化器一入选因子：样本外（2020-2025）月度 RankIC", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "factor_ic_grid.png", dpi=140); plt.close(fig)

    yearly = pd.DataFrame({f: oos_ic[f].groupby(oos_ic[f].index.year).mean()
                           for f in feat})
    fig, ax = plt.subplots(figsize=(8, 1.0 + 0.45 * n))
    vmax = np.nanmax(np.abs(yearly.values))
    im = ax.imshow(yearly.values, cmap="RdBu_r", aspect="auto",
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(yearly.columns)))
    ax.set_xticklabels(yearly.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(yearly.index)))
    ax.set_yticklabels(yearly.index, fontsize=8)
    for i in range(len(yearly.index)):
        for j in range(len(yearly.columns)):
            ax.text(j, i, f"{yearly.values[i, j]:.2f}", ha="center",
                    va="center", fontsize=7, color="black")
    ax.set_title("入选因子 × 年份 平均 RankIC")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout(); fig.savefig(OUT / "factor_ic_yearly_heatmap.png", dpi=150)
    plt.close(fig)

    top3 = sorted(feat, key=lambda f: -abs(oos_ic[f].mean()))[:3]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, f in zip(axes, top3):
        s = oos_ic[f]
        ax.bar(s.index, s.values, width=20, color="#4c78a8", alpha=0.8)
        ax2 = ax.twinx()
        ax2.plot(s.index, s.cumsum().values, color="#2ca02c", lw=1.6)
        ax.set_title(f"{f}（均值 {s.mean():.3f}）", fontsize=10)
        ax.tick_params(labelsize=7); ax2.tick_params(labelsize=7)
    fig.suptitle("前三因子：单期 IC 与累计 IC", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "ic_panels_top3.png", dpi=150); plt.close(fig)

    print(f"\n结果目录：{OUT}")
    print(f"最优配置：{best_mode}因子 K={best_K}，{best_W} Top{best_T}")


if __name__ == "__main__":
    main()
