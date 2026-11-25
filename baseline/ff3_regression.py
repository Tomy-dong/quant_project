# -*- coding: utf-8 -*-
"""Fama-French 三因子构造与回归对比。

目的
----
在同一套股票池、同一区间、同一费率下，把主策略与对照策略的月度超额收益
对 MKT、SMB、HML 做时序回归，回答两个问题：

1. 策略收益有多少能被市场、规模、价值三种风格解释（R²）；
2. 扣除三因子暴露后，是否还剩下显著的正 alpha（截距）。

三因子构造
----------
每个月末 t，在可交易股票池内做 2×3 独立双重排序：

- 规模：总市值高于/低于中位数，分 Small / Big；
- 价值：B/M = 1/PB，按 30%/40%/30% 分 Low / Neutral / High；
- 六个组合按 t 月末总市值加权，持有 t+1 月收益。

    SMB = mean(SL, SN, SH) - mean(BL, BN, BH)
    HML = mean(SH, BH) - mean(SL, BL)
    MKT = 全池市值加权收益 - Rf

无风险利率沿用课程实验假设，年化 2%，折算为月度复利。

对比对象
--------
- 优化器二 GMVP Top10（最终主策略）
- 优化器一 Top100（因子筛选后的等权对照）
- S7 Top100（多头门控基准）
- 沪深300、中证500（指数基准）

输出
----
results/ff3/
    ff3_factors.csv              月度三因子与无风险利率
    ff3_factor_stats.csv         三因子描述统计
    strategy_monthly_returns.csv 各策略/基准月度收益
    ff3_regression.csv           回归结果（alpha、beta、t 值、R²）
    ff3_factor_cum.png           三因子累计净值
    ff3_alpha.png                各策略年化 alpha 与 95% 置信区间
    ff3_loadings.png             各策略三因子暴露
    ff3_main_fit.png             主策略实际与拟合月度收益散点

用法
----
    python baseline/ff3_regression.py

Fama-French three-factor construction and regression comparison.

Build MKT/SMB/HML on the tradable universe with 2x3 size and book-to-market
sorts, then regress each strategy's monthly excess return on the three factors.
The intercept (alpha) tells whether the strategy adds value beyond style
exposure. Risk-free rate follows the course assumption: 2% per year.
"""
import sys
import warnings
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import price_volume_baseline as pv  # noqa: E402
import rolling_multifactor as rm  # noqa: E402
from data_io import load_benchmark  # noqa: E402
from utils.utils import calc_nav  # noqa: E402
from utils.stockselection import stock_selection  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = HERE.parent / "results" / "ff3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===================== 配置 / Config =====================
RF_ANNUAL = 0.02          # 年化无风险利率，课程实验假设 / course assumption
MIN_STOCKS = 50           # 单月构造三因子的最少股票数 / min stocks per month
MIN_GROUP = 10            # 单个组合最少股票数 / min stocks in a 2x3 cell
HAC_LAGS = 3              # Newey-West 滞后阶数 / HAC lag length
TOP_N_S7 = 100            # S7 对照的持仓数 / holdings of the S7 control

GMVP_HOLD = (HERE.parent / "results" / "optimizer2" / "top10" / "GMVP"
             / "holdings.csv")
OPT1_HOLD = (HERE.parent / "results" / "optimizer1" / "top100"
             / "holdings.csv")
S7_COMPOSITE = HERE.parent / "results" / "rolling_top" / "composite_factor.csv"


# ===================== 数据 / Data =====================
def load_wide(field):
    """读取沪市 + 深市宽表并按列合并。Load and merge the sh/sz wide tables."""
    parts = []
    for mkt in ("sh", "sz"):
        path = pv.DATA_DIR / f"{field}_{mkt}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"缺少 {path.name}，请先补齐 {field} 数据")
        parts.append(pd.read_csv(path, index_col=0, parse_dates=True))
    df = pd.concat(parts, axis=1)
    df = df.loc[~df.index.duplicated(keep="last")]
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    return df.sort_index()


# ===================== 三因子构造 / FF3 construction =====================
def build_ff3(returns, valid, pb, mv):
    """逐月做 2x3 双重排序，构造 MKT / SMB / HML。

    Build MKT/SMB/HML with monthly 2x3 size and B/M sorts. Portfolio returns
    are value-weighted by the market cap at the formation date and realised in
    the next month.
    """
    rf_m = (1 + RF_ANNUAL) ** (1 / 12) - 1
    rows = []
    skipped = []

    for i in range(len(returns.index) - 1):
        t, t_next = returns.index[i], returns.index[i + 1]
        pool = valid.loc[t].fillna(False).astype(bool)

        df = pd.DataFrame({
            "mv": mv.loc[t] if t in mv.index else np.nan,
            "pb": pb.loc[t] if t in pb.index else np.nan,
            "ret": returns.loc[t_next],
        })
        df = df.reindex(pool.index)
        df = df[pool.reindex(df.index).fillna(False)]
        df = df[(df["mv"] > 0) & (df["pb"] > 0)].dropna()
        if len(df) < MIN_STOCKS:
            skipped.append(t_next)
            continue

        # 规模分组：中位数 / size split at the median
        df["sz"] = np.where(df["mv"] >= df["mv"].median(), "B", "S")
        # 价值分组：按 B/M 的 30/40/30 / B/M terciles
        df["bm"] = 1.0 / df["pb"]
        rank = df["bm"].rank(method="first")
        df["bm_g"] = pd.qcut(rank, [0, 0.3, 0.7, 1.0], labels=["L", "N", "H"])

        def vw(sub):
            return float(np.average(sub["ret"], weights=sub["mv"]))

        grp = df.groupby(["sz", "bm_g"], observed=True)
        counts = grp.size()
        if counts.min() < MIN_GROUP or len(counts) < 6:
            skipped.append(t_next)
            continue
        gr = grp.apply(vw)

        smb = gr[["S"]].mean() - gr[["B"]].mean()
        hml = gr.xs("H", level="bm_g").mean() - gr.xs("L", level="bm_g").mean()
        mkt = vw(df)

        rows.append({"日期": t_next, "MKT": mkt - rf_m, "SMB": float(smb),
                     "HML": float(hml), "RF": rf_m, "股票数": len(df)})

    factors = pd.DataFrame(rows).set_index("日期").sort_index()
    print(f"  三因子构造完成：{len(factors)} 个月，跳过 {len(skipped)} 个月")
    return factors


def factor_stats(factors):
    """三因子描述统计。Descriptive statistics of the three factors."""
    rf = factors["RF"].iloc[0] if len(factors) else 0.0
    rows = []
    for name in ("MKT", "SMB", "HML"):
        s = factors[name].dropna()
        mean = s.mean()
        std = s.std(ddof=1)
        rows.append({
            "因子": name,
            "月均": mean,
            "年化均值": (1 + mean) ** 12 - 1,
            "年化波动": std * np.sqrt(12),
            "t值": mean / std * np.sqrt(len(s)) if std > 0 else np.nan,
            "累计收益": (1 + s).prod() - 1,
            "为正月份占比": (s > 0).mean(),
            "样本数": len(s),
        })
    return pd.DataFrame(rows)


# ===================== 策略收益 / Strategy returns =====================
def returns_from_holdings(path, returns):
    """按持仓权重与单边千二费率重建月度净收益。

    Rebuild monthly net returns from saved holdings and the one-way cost.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"未找到 {path}，请先运行对应优化器脚本")
    h = pd.read_csv(path, dtype={"股票代码": str})
    w = h.pivot_table(index="调仓日", columns="股票代码", values="权重",
                      aggfunc="sum")
    w.index = pd.to_datetime(w.index)
    w = w.reindex(columns=returns.columns).fillna(0.0)
    rets = returns.reindex(index=w.index, columns=w.columns).fillna(0.0)
    out = calc_nav(rets, w, comsn=pv.COMSN)
    return pd.Series(out["pnl"].values, index=w.index)


def returns_equal_weight_topn(composite, returns, top_n):
    """按合成分数每月等权持有前 top_n 只，含单边千二费率。

    Equal-weight the top_n names each month, with the one-way cost.
    """
    dates = composite.index
    holdings = np.full((len(dates), top_n), np.nan)
    for i in range(len(dates)):
        row = composite.iloc[i].dropna()
        if len(row) >= top_n:
            codes = row.nlargest(top_n).index
            holdings[i] = [composite.columns.get_loc(c) for c in codes]

    weights = stock_selection(returns.columns, holdings, dates)
    rets = returns.reindex(index=dates, columns=returns.columns).fillna(0.0)
    out = calc_nav(rets, weights, comsn=pv.COMSN)
    return pd.Series(out["pnl"].values, index=dates)


def benchmark_monthly(index_name, dates):
    """指数月度收益。Monthly index returns on the given month-end dates."""
    close = load_benchmark(pv.START, pv.END, index=index_name)["close"]
    close = close.reindex(dates)
    return close.pct_change(fill_method=None)


# ===================== 回归 / Regression =====================
def regress_one(name, raw_ret, factors):
    """单条收益序列对三因子回归。Regress one return series on MKT/SMB/HML."""
    df = pd.concat([raw_ret.rename("ret"), factors], axis=1).dropna()
    y = df["ret"] - df["RF"]
    x = sm.add_constant(df[["MKT", "SMB", "HML"]])
    model = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})

    alpha_m = model.params["const"]
    alpha_ann = (1 + alpha_m) ** 12 - 1
    se_ann = model.bse["const"] * 12
    return {
        "策略": name,
        "alpha月": alpha_m,
        "alpha年化": alpha_ann,
        "alpha_t": model.tvalues["const"],
        "alpha_p": model.pvalues["const"],
        "alpha_95%下限": alpha_ann - 1.96 * se_ann,
        "alpha_95%上限": alpha_ann + 1.96 * se_ann,
        "MKT": model.params["MKT"],
        "MKT_t": model.tvalues["MKT"],
        "SMB": model.params["SMB"],
        "SMB_t": model.tvalues["SMB"],
        "HML": model.params["HML"],
        "HML_t": model.tvalues["HML"],
        "R2": model.rsquared,
        "调整R2": model.rsquared_adj,
        "年化波动": y.std(ddof=1) * np.sqrt(12),
        "样本数": int(model.nobs),
    }


# ===================== 绘图 / Plots =====================
def plot_factor_cum(factors):
    """三因子累计净值。Cumulative value of the three factors."""
    cum = (1 + factors[["MKT", "SMB", "HML"]]).cumprod()
    fig, ax = plt.subplots(figsize=(10, 5))
    cum.plot(ax=ax, linewidth=1.6)
    ax.axhline(1.0, color="black", linewidth=0.7, linestyle="--")
    ax.set(xlabel="日期", ylabel="累计净值（初始=1）",
           title="Fama-French 三因子累计表现（2020-2025）")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "ff3_factor_cum.png", dpi=150)
    plt.close(fig)


def plot_alpha(summary):
    """各策略年化 alpha 与 95% 置信区间。Annualized alpha with 95% CI."""
    df = summary.sort_values("alpha年化")
    y = np.arange(len(df))
    lo = (df["alpha_95%下限"] * 100).to_numpy()
    hi = (df["alpha_95%上限"] * 100).to_numpy()
    val = (df["alpha年化"] * 100).to_numpy()
    colors = ["#c0392b" if v < 0 else "#2f6f9f" for v in val]

    fig, ax = plt.subplots(figsize=(9, max(3.8, 0.55 * len(df) + 1.5)))
    ax.barh(y, val, color=colors, alpha=0.85)
    ax.errorbar(val, y, xerr=[val - lo, hi - val], fmt="none",
                ecolor="black", capsize=3, linewidth=1)
    for i, v in enumerate(val):
        # 标签放在置信区间外侧，避免负号被柱体遮住
        # Put labels outside the confidence interval so minus signs stay visible.
        xpos = hi[i] + 0.4 if v >= 0 else lo[i] - 0.4
        ax.text(xpos, i, f"{v:+.1f}%", va="center",
                ha="left" if v >= 0 else "right", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(df["策略"])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(xlabel="年化 alpha（%）", title="三因子回归：年化 alpha 与 95% 置信区间")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "ff3_alpha.png", dpi=150)
    plt.close(fig)


def plot_loadings(summary):
    """各策略三因子暴露。Factor loadings of each strategy."""
    cols = ["MKT", "SMB", "HML"]
    x = np.arange(len(summary))
    width = 0.26
    fig, ax = plt.subplots(figsize=(10, 5))
    for k, c in enumerate(cols):
        ax.bar(x + (k - 1) * width, summary[c], width, label=c)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["策略"], rotation=15, ha="right", fontsize=9)
    ax.set(ylabel="回归系数", title="三因子回归：策略因子暴露")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "ff3_loadings.png", dpi=150)
    plt.close(fig)


def plot_main_fit(raw_ret, factors, name):
    """主策略实际与拟合月度收益散点。Actual vs fitted monthly returns."""
    df = pd.concat([raw_ret.rename("ret"), factors], axis=1).dropna()
    y = df["ret"] - df["RF"]
    x = sm.add_constant(df[["MKT", "SMB", "HML"]])
    model = sm.OLS(y, x).fit()
    fitted = model.fittedvalues

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.scatter(fitted * 100, y * 100, s=22, alpha=0.7, color="#2f6f9f")
    lim = [min(fitted.min(), y.min()) * 100 - 1,
           max(fitted.max(), y.max()) * 100 + 1]
    ax.plot(lim, lim, color="black", linewidth=0.9, linestyle="--")
    ax.set(xlabel="三因子拟合月收益（%）", ylabel="实际超额月收益（%）",
           title=f"{name}：实际 vs 拟合（R²={model.rsquared:.2f}）")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "ff3_main_fit.png", dpi=150)
    plt.close(fig)


# ===================== 主流程 / Main =====================
def main():
    print("Fama-French 三因子回归对比 | 样本外 "
          f"{rm.OOS_START_YEAR}-{rm.OOS_END_YEAR}")
    print("=" * 72)

    print("\n[1/4] 载入行情、PB 与总市值数据...")
    daily = pv.load_market_data(pv.MARKET, pv.START, pv.END)
    monthly = pv.build_monthly(daily)
    returns = monthly["returns"]
    valid = pv.filter_pool(monthly)
    pb = load_wide("pb").reindex(index=returns.index, columns=returns.columns)
    mv = load_wide("total_mv").reindex(index=returns.index,
                                       columns=returns.columns)
    print(f"  PB 覆盖 {pb.notna().sum().sum():,} 个股票-月，"
          f"总市值覆盖 {mv.notna().sum().sum():,} 个股票-月")

    print("[2/4] 构造 MKT / SMB / HML...")
    factors = build_ff3(returns, valid, pb, mv)
    oos = (factors.index >= pd.Timestamp(f"{rm.OOS_START_YEAR}-01-01")) & (
        factors.index <= pd.Timestamp(f"{rm.OOS_END_YEAR}-12-31"))
    factors = factors.loc[oos]
    factors.index.name = "日期"
    factors.to_csv(OUTPUT_DIR / "ff3_factors.csv", encoding="utf-8-sig")

    stats = factor_stats(factors)
    stats.to_csv(OUTPUT_DIR / "ff3_factor_stats.csv", index=False,
                 encoding="utf-8-sig")
    print("\n三因子描述统计（2020-2025）：")
    show = stats.copy()
    for c in ["月均", "年化均值", "年化波动", "累计收益", "为正月份占比"]:
        show[c] = (show[c] * 100).round(2)
    show["t值"] = show["t值"].round(2)
    print(show.to_string(index=False))

    print("\n[3/4] 重建各策略月度收益...")
    series = {
        "优化器二 GMVP Top10": returns_from_holdings(GMVP_HOLD, returns),
        "优化器一 Top100": returns_from_holdings(OPT1_HOLD, returns),
    }
    composite = pd.read_csv(S7_COMPOSITE, index_col=0, parse_dates=True)
    series["S7 Top100"] = returns_equal_weight_topn(
        composite, returns, TOP_N_S7)

    dates = pd.DatetimeIndex(factors.index)
    series["沪深300"] = benchmark_monthly("csi300", dates)
    series["中证500"] = benchmark_monthly("csi500", dates)

    ret_df = pd.DataFrame(series).sort_index()
    ret_df.index.name = "日期"
    ret_df.to_csv(OUTPUT_DIR / "strategy_monthly_returns.csv",
                  encoding="utf-8-sig")
    for name, s in series.items():
        print(f"  {name:<20s} {s.dropna().index.min().date()} ~ "
              f"{s.dropna().index.max().date()}，{s.dropna().shape[0]} 个月")

    print("[4/4] 三因子时序回归...")
    summary = pd.DataFrame([regress_one(name, s, factors)
                            for name, s in series.items()])
    summary.to_csv(OUTPUT_DIR / "ff3_regression.csv", index=False,
                   encoding="utf-8-sig")

    show = summary.copy()
    for c in ["alpha月", "alpha年化", "alpha_95%下限", "alpha_95%上限",
              "年化波动"]:
        show[c] = (show[c] * 100).round(2)
    for c in ["alpha_t", "MKT", "MKT_t", "SMB", "SMB_t", "HML", "HML_t",
              "R2", "调整R2"]:
        show[c] = show[c].round(3)
    print("\n回归结果（%）：")
    print(show.to_string(index=False))

    plot_factor_cum(factors)
    plot_alpha(summary)
    plot_loadings(summary)
    plot_main_fit(series["优化器二 GMVP Top10"], factors, "优化器二 GMVP Top10")

    print(f"\n结果已保存到：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
