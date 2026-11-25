# -*- coding: utf-8 -*-
"""优化器二：八种资金权重配置 / Optimizer-2: eight capital-weighting schemes

定位
----
优化器一只决定"买哪几只"（合成分数排序取 Top N）与"等权"；优化器二在
**已固定的名单内**回答"每只配多少资金"。名单来自
results/optimizer1/composite_factor.csv，与优化器一回测完全一致。

八种方法
--------
EW    等权             w_i = 1/N
IV    逆波动率         w_i ∝ 1/σ_i
IVar  逆方差           w_i ∝ 1/σ_i²
GMVP  全局最小方差     min w'Σw
ERC   等风险贡献       min ½x'Σx - Σ b_i log x，b_i = 1/N
RB    风险预算         b_i ∝ 1/σ_i（逆波动率预算）
MDP   最大分散化       max (w'σ)/√(w'Σw)
MDCP  最大去相关       min w'Cw，C 为相关矩阵

统一约定
--------
- 满仓 Σw=1、只做多 w≥0；单票上限 = 2/N（Top10 即 20%）。
- 协方差矩阵 Σ 用调仓日前 COV_WINDOW=36 个月的月收益估计，与所有方法共用，
  保证可比；向对角矩阵收缩 SHRINK=0.2 以避免 N>T 时矩阵奇异、解爆炸。
- 权重在调仓日确定，收益从下一期开始计入（calc_nav 内部 w[i-1]·r[i]）。

输出
----
results/optimizer2/
    monthly_returns.csv.gz       月频收益缓存（供协方差估计与复现）
    comparison.csv               8 方法 × Top10/50/100 绩效
    summary_bars.png             累计收益对比柱状图
    window_sensitivity.csv       GMVP Top10 协方差窗口敏感性（24/36/60）
    window_annual.csv            三个窗口的分年度收益
    window_compare.png           三个窗口的净值对比
    window{24,36,60}_top10/      窗口敏感性明细
    top{N}/{方法}/               绩效、全部结果图、holdings.csv/xlsx
    top{N}/nav_compare.png       N 只名单内 8 种权重净值对比

用法
----
    python baseline/optimizer2_weights.py
"""
import os
import sys
import time
import warnings
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Values in x were outside bounds")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import price_volume_baseline as pv  # noqa: E402
import rolling_multifactor as rm  # noqa: E402
import optimizer1_heuristic as o1  # noqa: E402
from utils import report_plots as rp  # noqa: E402
from utils.utils import calc_nav  # noqa: E402

OUTPUT_DIR = HERE.parent / "results" / os.environ.get("OPT2_OUTPUT", "optimizer2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
COMPOSITE_FILE = HERE.parent / "results" / os.environ.get(
    "OPT2_COMPOSITE", "optimizer1/composite_factor.csv")

TOP_N_LIST = tuple(int(x) for x in os.environ.get(
    "OPT2_TOP_N", "10,50,100").split(","))
METHODS = ("EW", "IV", "IVar", "GMVP", "ERC", "RB", "MDP", "MDCP")
METHOD_CN = {"EW": "等权", "IV": "逆波动率", "IVar": "逆方差",
             "GMVP": "最小方差", "ERC": "等风险贡献", "RB": "风险预算",
             "MDP": "最大分散化", "MDCP": "最大去相关"}

COV_WINDOW = 36      # 协方差估计窗口（月）/ covariance window in months
COV_WINDOWS = (24, 36, 60)   # 窗口敏感性 / window sensitivity grid
MIN_OBS = 12         # 最少有效观测 / minimum observations
SHRINK = 0.2         # 向对角收缩强度 / shrinkage toward diagonal


# ===================== 数据 / Data =====================
def load_monthly_returns(start, end):
    """只读后复权价格并取月末，构造月频收益面板。
    Build the monthly return panel from back-adjusted close only.
    """
    price_parts = []
    for mkt in ("sh", "sz"):
        close = pd.read_csv(pv.DATA_DIR / f"close_{mkt}.csv",
                            index_col=0, parse_dates=True)
        adj = pd.read_csv(pv.DATA_DIR / f"adj_factor_{mkt}.csv",
                          index_col=0, parse_dates=True)
        close, adj = close.align(adj, join="inner")
        price_parts.append(close * adj)
    price = pd.concat(price_parts, axis=1)
    price = price.loc[~price.index.duplicated(keep="last")]
    price = price.loc[:, ~price.columns.duplicated(keep="first")]
    price = price.loc[(price.index >= start) & (price.index <= end)]

    dates = pd.DatetimeIndex(
        pd.Series(price.index).groupby(price.index.to_period("M")).max())
    return price.reindex(dates).pct_change(fill_method=None)


def trailing_cov(rets, dt, codes, window=COV_WINDOW, shrink=SHRINK):
    """调仓日前 window 个月的协方差，缺失收益按 0 处理并做对角收缩。
    Trailing covariance estimated up to dt, shrunk toward its diagonal.
    """
    hist = rets.loc[:dt, codes].tail(window)
    if hist.notna().sum().sum() < MIN_OBS * 2:
        return np.eye(len(codes))
    hist = hist.fillna(0.0)
    cov = hist.cov(ddof=1).to_numpy(dtype=float)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    cov = 0.5 * (cov + cov.T)
    diag = np.maximum(np.diag(cov), 1e-10)
    cov = (1 - shrink) * cov + shrink * np.diag(diag)
    return cov + np.eye(len(cov)) * 1e-10


# ===================== 权重求解 / Weight solvers =====================
def _capped(w, cap):
    """把权重压到单票上限并重新归一。Clip to the single-name cap, renormalise."""
    w = np.clip(np.asarray(w, dtype=float), 0, None)
    total = w.sum()
    if total <= 0 or not np.isfinite(total):
        return np.full(len(w), 1.0 / len(w))
    w = w / total
    for _ in range(100):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        free = ~over
        free_sum = w[free].sum()
        if free_sum <= 0:
            w[free] = excess / free.sum()
        else:
            w[free] += excess * w[free] / free_sum
    return w / w.sum()


def _solve(cost, n, cap):
    """带满仓与上下限约束的 SLSQP 求解，失败时退回等权。
    SLSQP with full-investment and box constraints; fall back to EW on failure.
    """
    x0 = np.full(n, 1.0 / n)
    res = minimize(cost, x0, method="SLSQP", bounds=[(0.0, cap)] * n,
                   constraints=({"type": "eq", "fun": lambda w: w.sum() - 1.0},),
                   options={"maxiter": 500, "ftol": 1e-12})
    w = np.asarray(res.x, dtype=float)
    if not np.isfinite(w).all() or w.sum() <= 0:
        return x0
    return _capped(w, cap)


def _risk_budget(cov, b, cap):
    """风险预算凸问题 min ½x'Σx - Σb_i log x，归一后压上限。
    Risk-budgeting convex program; normalised then capped.
    """
    n = len(b)
    x0 = np.full(n, 1.0 / n)
    cost = lambda x: 0.5 * float(x @ cov @ x) - float(b @ np.log(x))  # noqa: E731
    res = minimize(cost, x0, method="SLSQP", bounds=[(1e-8, None)] * n,
                   options={"maxiter": 500, "ftol": 1e-12})
    x = np.asarray(res.x, dtype=float)
    if not np.isfinite(x).all() or (x <= 0).any():
        x = x0
    return _capped(x / x.sum(), cap)


def method_weights(method, cov, cap):
    """按方法名生成一组权重。Weights for one method on a fixed name list."""
    n = cov.shape[0]
    diag = np.clip(np.diag(cov), 1e-12, None)
    sig = np.sqrt(diag)

    if method == "EW":
        return np.full(n, 1.0 / n)
    if method == "IV":
        w = 1.0 / sig
    elif method == "IVar":
        w = 1.0 / diag
    elif method == "GMVP":
        return _solve(lambda w: float(w @ cov @ w), n, cap)
    elif method == "MDCP":
        corr = cov / np.outer(sig, sig)
        np.fill_diagonal(corr, 1.0)
        return _solve(lambda w: float(w @ corr @ w), n, cap)
    elif method == "MDP":
        def cost(w):
            var = max(float(w @ cov @ w), 1e-16)
            return -float(w @ sig) / np.sqrt(var)
        return _solve(cost, n, cap)
    elif method in ("ERC", "RB"):
        if method == "ERC":
            b = np.full(n, 1.0 / n)
        else:
            b = (1.0 / sig) / (1.0 / sig).sum()
        return _risk_budget(cov, b, cap)

    return _capped(w / w.sum(), cap)


def build_weight_mats(composite, rets, top_n, methods=METHODS,
                      window=COV_WINDOW, shrink=SHRINK):
    """对每个调仓日生成指定方法的权重矩阵（日期 × 股票）。
    Build the weight matrices for the requested methods and holding count.
    """
    dates = composite.index
    codes_all = composite.columns
    col_idx = {c: i for i, c in enumerate(codes_all)}
    cap = 2.0 / top_n
    mats = {m: np.zeros((len(dates), len(codes_all))) for m in methods}

    for i, dt in enumerate(dates):
        row = composite.loc[dt].dropna()
        if len(row) < top_n:
            continue
        codes = row.nlargest(top_n).index
        cov = trailing_cov(rets, dt, codes, window=window, shrink=shrink)
        idx = [col_idx[c] for c in codes]
        for m in methods:
            mats[m][i, idx] = method_weights(m, cov, cap)

    return {m: pd.DataFrame(v, index=dates, columns=codes_all)
            for m, v in mats.items()}


# ===================== 回测与输出 / Backtest & outputs =====================
def performance(outputs):
    """月度净值与 pnl 的标准绩效。Standard performance from nav/pnl."""
    nav, pnl = outputs["nav"], outputs["pnl"]
    years = (nav.index[-1] - nav.index[0]).days / 365.24
    ann_ret = nav.iloc[-1] ** (1 / years) - 1
    ann_vol_ = pnl.std() * np.sqrt(12)
    return pd.Series({
        "累计收益": nav.iloc[-1] - 1,
        "年化收益": ann_ret,
        "年化波动": ann_vol_,
        "夏普比率": ann_ret / ann_vol_ if ann_vol_ > 0 else 0,
        "最大回撤": (nav / nav.cummax() - 1).min(),
        "年化换手": outputs["turnover"].mean() * 12,
    })


def run_one(weights, rets, out_dir, title, ic=None, grp=None):
    """给定权重矩阵回测，输出绩效、持仓表与全部结果图。
    Backtest a given weight matrix and export everything.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = calc_nav(rets.fillna(0), weights, comsn=pv.COMSN)
    perf = performance(outputs)
    perf.to_csv(out_dir / "performance.csv", encoding="utf-8-sig")
    o1.export_holdings(weights, out_dir)
    rp.plot_all(outputs, o1.make_bench_nav(weights.index), out_dir, title,
                ic=ic, grp=grp)
    return outputs, perf


def plot_nav_compare(navs, path, title):
    """同一名单下 8 种权重的净值曲线。NAV curves of the eight schemes."""
    fig, ax = plt.subplots(figsize=(11, 5.8))
    for name, nav in navs.items():
        label = f"{name} {METHOD_CN[name]}" if name in METHOD_CN else str(name)
        ax.plot(nav.index, nav.values, lw=1.5, label=label)
    ax.axhline(1.0, color="black", lw=0.8, ls="--", alpha=0.6)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("净值")
    ax.grid(alpha=0.3, ls=":")
    ax.legend(fontsize=8, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_summary_bars(cmp, path):
    """8 方法 × 3 种持仓数的累计收益对比。Cumulative return comparison."""
    methods = list(METHODS)
    x = np.arange(len(methods))
    width = 0.26
    fig, ax = plt.subplots(figsize=(12, 5.6))
    for k, tn in enumerate(TOP_N_LIST):
        vals = [cmp.loc[(f"top{tn}", m), "累计收益"] * 100 for m in methods]
        ax.bar(x + (k - 1) * width, vals, width, label=f"Top{tn}")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n{METHOD_CN[m]}" for m in methods], fontsize=9)
    ax.set_ylabel("累计收益（%）")
    ax.set_title("优化器二：八种资金权重配置样本外累计收益（2020-2025）",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, ls=":")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ===================== 主流程 / Main =====================
def main():
    t0 = time.time()
    print("优化器二：八种资金权重配置 | 名单来自优化器一 Top N")
    print("=" * 72)

    print("[1/3] 读取合成分数与月频收益...")
    composite = pd.read_csv(COMPOSITE_FILE, index_col=0, parse_dates=True)
    rets_full = load_monthly_returns(pv.START, pv.END)
    rets_full.to_csv(OUTPUT_DIR / "monthly_returns.csv.gz",
                     compression="gzip", encoding="utf-8-sig")
    rets = rets_full.reindex(index=composite.index, columns=composite.columns)
    print(f"  调仓 {len(composite)} 期，候选 {composite.shape[1]} 只，"
          f"协方差窗口 {COV_WINDOW} 个月")

    ic = rm.rank_ic_panel({"合成因子": composite}, rets)["合成因子"]
    grp = rp.group_returns(composite, rets, rp.N_GROUP)
    print(f"  合成因子样本外 IC 均值 {ic.mean():.3f}")

    print("[2/3] 逐月求解 8 种权重并回测...")
    rows = []
    for tn in TOP_N_LIST:
        print(f"\n  --- Top{tn}（单票上限 {2.0 / tn:.1%}）---")
        mats = build_weight_mats(composite, rets_full, tn)
        navs = {}
        for m in METHODS:
            out_dir = OUTPUT_DIR / f"top{tn}" / m
            outputs, perf = run_one(
                mats[m], rets, out_dir,
                f"优化器二 {METHOD_CN[m]}({m}) Top{tn}",
                ic=ic if m == "EW" else None,
                grp=grp if m == "EW" else None)
            navs[m] = outputs["nav"]
            rows.append({"持仓数": f"top{tn}", "方法": m,
                         "方法中文": METHOD_CN[m], **perf.to_dict()})
            print(f"    {m:<5s} 累计 {perf['累计收益'] * 100:7.2f}% | "
                  f"年化 {perf['年化收益'] * 100:6.2f}% | "
                  f"夏普 {perf['夏普比率']:6.3f} | "
                  f"回撤 {perf['最大回撤'] * 100:7.2f}% | "
                  f"换手 {perf['年化换手']:5.2f}")
        plot_nav_compare(navs, OUTPUT_DIR / f"top{tn}" / "nav_compare.png",
                         f"优化器二 Top{tn}：八种资金权重净值对比")

    cmp = pd.DataFrame(rows).set_index(["持仓数", "方法"])
    cmp.to_csv(OUTPUT_DIR / "comparison.csv", encoding="utf-8-sig")
    plot_summary_bars(cmp, OUTPUT_DIR / "summary_bars.png")

    sens_top = TOP_N_LIST[-1]
    print(f"\n补充：协方差窗口敏感性（GMVP Top{sens_top}，24/36/60 个月）...")
    sens_rows, sens_navs, sens_pnl = [], {}, {}
    for w in COV_WINDOWS:
        mats = build_weight_mats(composite, rets_full, sens_top,
                                 methods=("GMVP",), window=w)
        outputs, perf = run_one(
            mats["GMVP"], rets, OUTPUT_DIR / f"window{w}_top{sens_top}" / "GMVP",
            f"优化器二 GMVP Top{sens_top}（Σ 窗口 {w} 个月）")
        label = f"窗口{w}个月"
        sens_navs[label] = outputs["nav"]
        sens_pnl[label] = outputs["pnl"]
        sens_rows.append({"协方差窗口": w, **perf.to_dict()})
        print(f"    窗口 {w:>2d} 个月：累计 {perf['累计收益'] * 100:7.2f}% | "
              f"夏普 {perf['夏普比率']:6.3f} | "
              f"回撤 {perf['最大回撤'] * 100:7.2f}%")
    sens = pd.DataFrame(sens_rows).set_index("协方差窗口")
    sens.to_csv(OUTPUT_DIR / "window_sensitivity.csv", encoding="utf-8-sig")
    plot_nav_compare(sens_navs, OUTPUT_DIR / "window_compare.png",
                     f"优化器二 GMVP Top{sens_top}：协方差窗口敏感性")

    ann = pd.DataFrame({k: rp.annual_returns(v) for k, v in sens_pnl.items()})
    ann.index.name = "年份"
    ann.to_csv(OUTPUT_DIR / "window_annual.csv", encoding="utf-8-sig")
    print("    分年度收益：\n" + (ann * 100).round(2).to_string())

    print("\n[3/3] 汇总（按累计收益排序）")
    show = cmp[["累计收益", "年化收益", "年化波动", "夏普比率",
                "最大回撤", "年化换手"]].apply(pd.to_numeric)
    for c in ["累计收益", "年化收益", "年化波动", "最大回撤"]:
        show[c] = (show[c] * 100).round(2)
    show[["夏普比率", "年化换手"]] = show[["夏普比率", "年化换手"]].round(3)
    print(show.sort_values("累计收益", ascending=False).to_string())

    print("\n相对等权基线的增量（累计收益）：")
    for tn in TOP_N_LIST:
        base = cmp.loc[(f"top{tn}", "EW"), "累计收益"]
        delta = (cmp.loc[f"top{tn}", "累计收益"] - base) * 100
        print(f"  Top{tn}：" + " | ".join(
            f"{m} {delta[m]:+.2f}pp" for m in METHODS if m != "EW"))

    print(f"\n完成，用时 {time.time() - t0:.0f}s；结果在 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
