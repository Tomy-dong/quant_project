# -*- coding: utf-8 -*-
"""回撤审计：指标扩展 + 多次随机回测 + 压力情景。

对应课程第九章：
- 绩效指标表：Sharpe、年化收益、年化波动、Sortino、最大回撤、CVaR95；
- 四种回测方法中的“多次随机回测”：随机抽资产子集与连续时间窗，重复 k 次，
  汇总最大回撤等指标的分布，而不是只看一条历史路径；
- 合成数据/压力测试：单独检查 2020Q1、2022 全年、2024Q1 微盘危机三个情景。

主策略：results/optimizer2_wq03_top30/top30/IVar/holdings.csv（WQ 4–5 因子、Top30、IVar）。

输出（results/audit/drawdown/）：
    metrics_main.csv          主策略月频净收益指标 + 日频回撤
    random_groups.csv         40 组随机回测 × 3 种权重的逐组指标
    random_summary.csv        各方法指标的中位数 / 均值 / 5% 分位
    paired_comparison.csv     IVar、GMVP 相对 EW 的最大回撤配对比较与胜率
    stress_windows.csv        三个压力情景的收益与回撤
    dd_boxplot.png            最大回撤箱线图
    paired_diff.png           配对回撤差值分布
    main_drawdown_daily.png   主策略日频回撤曲线（标注压力区间）
    drawdown_summary.md       结论汇总

用法
----
    python baseline/drawdown_audit.py
"""
import os
import sys
import warnings
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Values in x were outside bounds")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import price_volume_baseline as pv  # noqa: E402
import optimizer2_weights as o2  # noqa: E402
from utils.utils import calc_nav  # noqa: E402

OUT = HERE.parent / "results" / "audit" / "drawdown"
OUT.mkdir(parents=True, exist_ok=True)

COMPOSITE = HERE.parent / "results" / "optimizer1_wq03" / "composite_factor.csv"
HOLDINGS = (HERE.parent / "results" / "optimizer2_wq03_top30"
            / "top30" / "IVar" / "holdings.csv")

GROUPS = int(os.environ.get("DDA_GROUPS", 40))
WINDOW = int(os.environ.get("DDA_WINDOW", 36))
ASSETS = int(os.environ.get("DDA_ASSETS", 300))
TOPN = int(os.environ.get("DDA_TOP_N", 30))
COV_WINDOW = 36
SEED = 20261125
METHODS = ("EW", "IVar", "GMVP")
METHOD_CN = {"EW": "等权", "IVar": "逆方差", "GMVP": "最小方差"}

STRESS = {
    "2020Q1 疫情冲击": ("2020-01-23", "2020-03-31"),
    "2022 全年": ("2021-12-31", "2022-12-30"),
    "2024Q1 微盘危机": ("2024-01-02", "2024-02-29"),
}

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ===================== 数据 / Data =====================
def load_daily_prices(start, end):
    """后复权日收盘价。Back-adjusted daily close."""
    parts = []
    for mkt in ("sh", "sz"):
        close = pd.read_csv(pv.DATA_DIR / f"close_{mkt}.csv",
                            index_col=0, parse_dates=True)
        adj = pd.read_csv(pv.DATA_DIR / f"adj_factor_{mkt}.csv",
                          index_col=0, parse_dates=True)
        close, adj = close.align(adj, join="inner")
        parts.append(close * adj)
    price = pd.concat(parts, axis=1)
    price = price.loc[~price.index.duplicated(keep="last")]
    price = price.loc[:, ~price.columns.duplicated(keep="first")]
    return price.loc[(price.index >= start) & (price.index <= end)]


def load_holdings():
    """调仓日 × 股票的权重矩阵。Rebalance-date weight matrix."""
    h = pd.read_csv(HOLDINGS, parse_dates=["调仓日"])
    w = h.pivot_table(index="调仓日", columns="股票代码", values="权重",
                      aggfunc="sum").fillna(0.0)
    return w


# ===================== 指标 / Metrics =====================
def dd_curve(nav):
    """回撤序列。Drawdown series."""
    return nav / nav.cummax() - 1


def dd_duration(nav):
    """最长水下期（期数）与当前是否在水下。Longest underwater run."""
    dd = dd_curve(nav)
    under = dd < -1e-12
    longest = cur = 0
    for flag in under:
        cur = cur + 1 if flag else 0
        longest = max(longest, cur)
    return int(longest), bool(under.iloc[-1])


def metrics(pnl):
    """月频收益序列的标准指标。Standard metrics from a monthly PnL series."""
    pnl = pnl.iloc[1:] if len(pnl) > 12 else pnl
    pnl = pnl.dropna()
    n = len(pnl)
    nav = (1 + pnl).cumprod()
    years = n / 12
    ann = nav.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol = pnl.std(ddof=1) * np.sqrt(12)
    downside = np.sqrt(np.mean(np.minimum(pnl, 0.0) ** 2)) * np.sqrt(12)
    q05 = pnl.quantile(0.05)
    longest, underwater = dd_duration(nav)
    return pd.Series({
        "累计收益": nav.iloc[-1] - 1,
        "年化收益": ann,
        "年化波动": vol,
        "夏普": ann / vol if vol > 0 else np.nan,
        "Sortino": ann / downside if downside > 0 else np.nan,
        "最大回撤": dd_curve(nav).min(),
        "CVaR95": pnl[pnl <= q05].mean(),
        "最长水下月数": longest,
        "期末在水下": underwater,
        "月数": n,
    })


def daily_max_dd(nav):
    """日频最大回撤与最长水下天数。Daily max drawdown and longest underwater run."""
    dd = dd_curve(nav)
    return float(dd.min()), int(dd_duration(nav)[0])


# ===================== 1. 主策略指标 / Main strategy =====================
def main_metrics(rets, w, daily_ret):
    """月频净收益指标 + 日频回撤。Monthly net metrics plus daily drawdown."""
    out = calc_nav(rets.loc[w.index, w.columns].fillna(0), w, comsn=pv.COMSN)
    m = metrics(out["pnl"])
    m["年化换手"] = out["turnover"].mean() * 12

    # 日频重建：月内权重固定，忽略成本，仅用于看月内回撤深度
    idx = daily_ret.index
    wd = w.reindex(idx, method="ffill").shift(1).fillna(0.0)
    cols = [c for c in w.columns if c in daily_ret.columns]
    d_pnl = (wd[cols] * daily_ret[cols].reindex(idx).fillna(0.0)).sum(axis=1)
    d_nav = (1 + d_pnl).cumprod()
    m["日频最大回撤"], m["日频最长水下天数"] = daily_max_dd(d_nav)
    m.to_csv(OUT / "metrics_main.csv", encoding="utf-8-sig")
    return m, out, d_nav


# ===================== 2. 随机回测 / Random backtests =====================
def one_group(comp, rets, dates, rng, gid):
    """一组随机回测：随机资产子集 + 连续时间窗。One random group."""
    start = int(rng.integers(0, len(dates) - WINDOW))
    win = dates[start:start + WINDOW]
    pool = comp.loc[win].dropna(axis=1, how="all").columns
    size = min(ASSETS, len(pool))
    codes = rng.choice(pool.to_numpy(), size=size, replace=False)

    mats = {m: pd.DataFrame(0.0, index=win, columns=comp.columns)
            for m in METHODS}
    for dt in win:
        row = comp.loc[dt, codes].dropna()
        if len(row) < TOPN:
            continue
        pick = row.nlargest(TOPN).index
        cov = o2.trailing_cov(rets, dt, pick, window=COV_WINDOW)
        cap = 2.0 / TOPN
        for m in METHODS:
            wt = o2.method_weights(m, cov, cap)
            mats[m].loc[dt, pick] = wt

    rows = []
    for m in METHODS:
        out = calc_nav(
            rets.loc[win].reindex(columns=mats[m].columns).fillna(0),
            mats[m], comsn=pv.COMSN)
        met = metrics(out["pnl"])
        rows.append({"组": gid, "窗口起点": win[0], "窗口终点": win[-1],
                     "方法": m, **met.to_dict()})
    return rows


def random_backtests(comp, rets):
    """40 组随机回测。Forty random asset/time resamples."""
    dates = comp.index
    rng = np.random.default_rng(SEED)
    rows = []
    for gid in range(GROUPS):
        try:
            rows.extend(one_group(comp, rets, dates, rng, gid))
        except Exception as exc:  # 单组失败也记录，不静默丢弃
            rows.append({"组": gid, "方法": "FAILED", "错误": str(exc)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "random_groups.csv", index=False, encoding="utf-8-sig")

    ok = df[df["方法"].isin(METHODS)]
    summary = ok.groupby("方法").agg(
        有效组数=("组", "nunique"),
        最大回撤中位数=("最大回撤", "median"),
        最大回撤均值=("最大回撤", "mean"),
        最大回撤最差5pct=("最大回撤", lambda s: s.quantile(0.05)),
        年化收益中位数=("年化收益", "median"),
        夏普中位数=("夏普", "median"),
        Sortino中位数=("Sortino", "median"),
        CVaR95中位数=("CVaR95", "median"),
    ).reindex(list(METHODS))
    summary.to_csv(OUT / "random_summary.csv", encoding="utf-8-sig")
    return ok, summary


def paired(ok):
    """IVar、GMVP 相对 EW 的配对差与胜率。Paired differences vs EW."""
    wide = ok.pivot(index="组", columns="方法", values="最大回撤")
    shp = ok.pivot(index="组", columns="方法", values="夏普")
    rows = []
    for m in ("IVar", "GMVP"):
        diff = (wide[m] - wide["EW"]).dropna()
        ds = (shp[m] - shp["EW"]).dropna()
        rows.append({
            "对比": f"{m} - EW",
            "有效组数": int(len(diff)),
            "回撤差值中位数": float(diff.median()),
            "回撤改善组数": int((diff > 0).sum()),
            "回撤回撤胜率": float((diff > 0).mean()) if len(diff) else np.nan,
            "夏普差值中位数": float(ds.median()) if len(ds) else np.nan,
            "夏普胜率": float((ds > 0).mean()) if len(ds) else np.nan,
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "paired_comparison.csv", index=False, encoding="utf-8-sig")
    return df, wide


# ===================== 3. 压力情景 / Stress windows =====================
def stress(daily_ret, d_nav):
    """三个压力情景的收益与回撤。Return and drawdown in three stress windows."""
    # 全A等权日收益作为参照
    mkt = daily_ret.mean(axis=1, skipna=True)
    mkt_nav = (1 + mkt.fillna(0)).cumprod()
    rows = []
    for name, (a, b) in STRESS.items():
        seg = d_nav.loc[a:b]
        mk = mkt_nav.loc[a:b]
        me = seg.groupby(seg.index.to_period("M")).last()
        me = pd.concat([pd.Series([seg.iloc[0]]), me.reset_index(drop=True)],
                       ignore_index=True)
        if len(seg) < 2:
            continue
        dd_s = dd_curve(seg)
        dd_m = dd_curve(me)
        rows.append({
            "情景": name,
            "区间": f"{a} ~ {b}",
            "策略收益": seg.iloc[-1] / seg.iloc[0] - 1,
            "策略区间最大回撤": float(dd_s.min()),
            "策略最差单日": float((seg / seg.shift(1) - 1).min()),
            "月频区间最大回撤": float(dd_m.min()),
            "全A等权收益": mk.iloc[-1] / mk.iloc[0] - 1,
            "全A等权区间最大回撤": float(dd_curve(mk).min()),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "stress_windows.csv", index=False, encoding="utf-8-sig")
    return df


# ===================== 4. 图 / Plots =====================
def plots(ok, wide, d_nav, summary):
    # 箱线图：最大回撤
    data = [ok.loc[ok["方法"] == m, "最大回撤"].dropna() * 100 for m in METHODS]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bp = ax.boxplot(data, labels=[f"{m} {METHOD_CN[m]}" for m in METHODS],
                    patch_artist=True, medianprops={"color": "black"})
    for patch, color in zip(bp["boxes"], ("#9ecae1", "#fdae6b", "#a1d99b")):
        patch.set_facecolor(color)
    ax.axhline(summary.loc["IVar", "最大回撤中位数"] * 100, color="C1",
               ls="--", lw=1, alpha=0.7)
    ax.set_ylabel("最大回撤（%）")
    ax.set_title(f"{GROUPS} 组随机回测的最大回撤分布")
    ax.grid(axis="y", alpha=0.3, ls=":")
    fig.tight_layout()
    fig.savefig(OUT / "dd_boxplot.png", dpi=150)
    plt.close(fig)

    # 配对差值分布
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for m, color in (("IVar", "#fdae6b"), ("GMVP", "#a1d99b")):
        diff = (wide[m] - wide["EW"]).dropna() * 100
        ax.hist(diff, bins=12, alpha=0.55, label=f"{m} - EW", color=color)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("最大回撤差值（百分点，>0 表示回撤更浅）")
    ax.set_ylabel("组数")
    ax.set_title("相对等权的配对回撤改善")
    ax.legend()
    ax.grid(alpha=0.3, ls=":")
    fig.tight_layout()
    fig.savefig(OUT / "paired_diff.png", dpi=150)
    plt.close(fig)

    # 主策略日频回撤
    dd = dd_curve(d_nav) * 100
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.fill_between(dd.index, dd.values, 0, color="#d62728", alpha=0.35)
    ax.plot(dd.index, dd.values, color="#d62728", lw=1)
    for name, (a, b) in STRESS.items():
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="grey", alpha=0.18)
    ax.set_ylabel("回撤（%）")
    ax.set_title("主策略日频回撤（灰底为压力区间，未计交易成本）")
    ax.grid(alpha=0.3, ls=":")
    fig.tight_layout()
    fig.savefig(OUT / "main_drawdown_daily.png", dpi=150)
    plt.close(fig)


# ===================== 5. 汇总 / Summary =====================
def pct(x):
    return f"{x * 100:.2f}%"


def write_summary(main, summary, pair, st):
    lines = [
        "# 回撤审计（第九章方法）",
        "",
        f"样本：2020-01 至 2025-12，主策略 Top30 IVar，单边费率 {pv.COMSN:.1%}。",
        "",
        "## 1. 主策略指标",
        "",
        "| 指标 | 月频（净收益） |",
        "| --- | ---: |",
        f"| 累计收益 | {pct(main['累计收益'])} |",
        f"| 年化收益 | {pct(main['年化收益'])} |",
        f"| 年化波动 | {pct(main['年化波动'])} |",
        f"| 夏普 | {main['夏普']:.3f} |",
        f"| Sortino | {main['Sortino']:.3f} |",
        f"| 最大回撤 | {pct(main['最大回撤'])} |",
        f"| CVaR95（月度） | {pct(main['CVaR95'])} |",
        f"| 最长水下月数 | {int(main['最长水下月数'])} |",
        f"| 日频最大回撤（未计成本） | {pct(main['日频最大回撤'])} |",
        "",
        "## 2. 多次随机回测",
        "",
        f"每组随机抽 {ASSETS} 只股票、连续 {WINDOW} 个月，重复 {GROUPS} 组；"
        "组内每月按合成分数取 Top30，分别用 EW / IVar / GMVP 配权。",
        "",
        "| 方法 | 有效组数 | 最大回撤中位数 | 最差5%分位 | 年化收益中位数 | 夏普中位数 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for m in METHODS:
        r = summary.loc[m]
        lines.append(
            f"| {m} {METHOD_CN[m]} | {int(r['有效组数'])} | "
            f"{pct(r['最大回撤中位数'])} | {pct(r['最大回撤最差5pct'])} | "
            f"{pct(r['年化收益中位数'])} | {r['夏普中位数']:.3f} |")
    lines += [
        "",
        "## 3. 配对比较",
        "",
        "| 对比 | 有效组数 | 回撤差值中位数 | 回撤改善组数 | 胜率 | 夏普差值中位数 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, r in pair.iterrows():
        lines.append(
            f"| {r['对比']} | {int(r['有效组数'])} | "
            f"{r['回撤差值中位数'] * 100:+.2f}pp | {int(r['回撤改善组数'])} | "
            f"{r['回撤回撤胜率']:.0%} | {r['夏普差值中位数']:+.3f} |")
    lines += [
        "",
        "## 4. 压力情景",
        "",
        "| 情景 | 区间 | 策略收益 | 策略区间最大回撤 | 全A等权收益 | 全A等权回撤 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for _, r in st.iterrows():
        lines.append(
            f"| {r['情景']} | {r['区间']} | {pct(r['策略收益'])} | "
            f"{pct(r['策略区间最大回撤'])} | {pct(r['全A等权收益'])} | "
            f"{pct(r['全A等权区间最大回撤'])} |")
    lines += [
        "",
        "## 5. 结论",
        "",
        "- 月频最大回撤与日频最大回撤分开报告：月频口径会低估月内回撤，"
        "对外介绍应引用日频数字。",
        "- 随机回测给出回撤分布，而不是单一历史路径；若中位数与最差 5% 分位差距大，"
        "说明回撤对样本选择敏感。",
        f"- IVar 在 {int(pair.loc[0, '有效组数'])} 组配对中回撤优于等权的比例为 "
        f"{pair.loc[0, '回撤回撤胜率']:.0%}，差值中位数 "
        f"{pair.loc[0, '回撤差值中位数'] * 100:+.2f}pp；"
        "这是“优化器二是否值得”的直接证据，而不是只看全样本一条路径。",
        f"- 随机回测的 IVar 年化收益中位数为 {pct(summary.loc['IVar', '年化收益中位数'])}，"
        f"低于全样本 {pct(main['年化收益'])}：每组只从约 {ASSETS} 只股票中选 Top30，"
        "候选池被压缩，因此随机回测用于比较方法优劣，不代表主策略真实收益。",
        "- 压力情景只覆盖三段历史，不等于完整压力测试；极端情景下的损失仍可能超出样本。",
    ]
    (OUT / "drawdown_summary.md").write_text("\n".join(lines), encoding="utf-8")


# ===================== main =====================
def main():
    print("[1/5] 读取数据 ...")
    comp = pd.read_csv(COMPOSITE, index_col=0, parse_dates=True)
    rets = o2.load_monthly_returns(pv.START, pv.END)
    rets = rets.loc[rets.index >= comp.index[0]]
    w = load_holdings()
    daily = load_daily_prices("2020-01-01", "2025-12-31")
    daily_ret = daily.pct_change(fill_method=None)

    print("[2/5] 主策略指标 ...")
    main, _, d_nav = main_metrics(rets, w, daily_ret)
    print(main.drop(labels=["期末在水下"]).astype(float).round(4).to_string())

    print("[3/5] 多次随机回测 ...")
    ok, summary = random_backtests(comp, rets)
    print(summary.round(4).to_string())

    print("[4/5] 配对比较与压力情景 ...")
    pair, wide = paired(ok)
    st = stress(daily_ret, d_nav)
    print(pair.round(4).to_string())
    print(st.round(4).to_string())

    print("[5/5] 绘图与汇总 ...")
    plots(ok, wide, d_nav, summary)
    write_summary(main, summary, pair, st)
    print(f"完成，结果见 {OUT}")


if __name__ == "__main__":
    main()
