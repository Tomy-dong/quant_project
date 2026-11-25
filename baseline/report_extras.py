# -*- coding: utf-8 -*-
"""组合过程图：净值与回撤区间、每月调仓只数、权重视图、绩效柱状图、有效前沿。

对应课件“净值与持仓调整 / 有效前沿 / 净值与回撤”三类图。全部基于
results/optimizer2 下已缓存的月频收益与持仓表计算，不重跑因子，几秒完成。

The process figures (NAV with drawdown span, monthly buy/sell counts, weights,
performance bars, efficient frontier). Everything is built from the cached
monthly returns and holdings under results/optimizer2, so no factor
recomputation is needed.

输出 results/optimizer2/extras/
    nav_drawdown.png      策略/基准净值 + 最大回撤区间高亮 + 回撤曲线
    rebalance_counts.png  每月买入/卖出股票只数
    weight_heatmap.png    日期 × 股票 权重热力图
    weights_latest.png    最新一期 GMVP 与等权权重对比
    weights_rolling.gif   每月权重滚动条形图
    perf_bars.png         年化收益 / 年化波动 / 夏普 三面板（8 种方法）
    efficient_frontier.png 末期名单随机组合散点 + GMVP / 等权位置

用法
----
    python baseline/report_extras.py
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
from matplotlib.animation import FuncAnimation, PillowWriter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import price_volume_baseline as pv  # noqa: E402
from data_io import load_benchmark  # noqa: E402
from utils.utils import calc_nav  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = HERE.parent / "results" / "optimizer2"
EXTRA = OUT / "extras"
EXTRA.mkdir(parents=True, exist_ok=True)
COMSN = 0.002
TOP_N = 10


def load_weights(path, columns):
    """长表持仓转权重矩阵并对齐收益列。Long holdings to a weight matrix."""
    df = pd.read_csv(path, dtype={"股票代码": str}, parse_dates=["调仓日"])
    w = df.pivot_table(index="调仓日", columns="股票代码", values="权重",
                       aggfunc="sum").fillna(0.0)
    return w.reindex(columns=columns, fill_value=0.0)


def plot_nav_drawdown(nav, benches, path):
    """净值 + 双基准 + 最大回撤区间阴影 + 回撤曲线。
    NAV with two benchmarks, the maximum-drawdown span and the drawdown curve.
    """
    dd = nav / nav.cummax() - 1
    trough = dd.idxmin()
    peak = nav.loc[:trough].idxmax()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1.4]})
    ax1.plot(nav.index, nav.values, color="#1f4e79", lw=1.6, label="策略净值")
    for label, series in benches.items():
        ax1.plot(series.index, series.values, lw=1.3, label=label)
    ax1.axvspan(peak, trough, color="#c00000", alpha=0.12)
    ax1.annotate(f"最大回撤 {dd.min() * 100:.2f}%", xy=(trough, nav.loc[trough]),
                 xytext=(0.02, 0.08), textcoords="axes fraction",
                 color="#c00000", fontsize=10,
                 arrowprops=dict(arrowstyle="->", color="#c00000", lw=1))
    ax1.axhline(1.0, color="black", lw=0.7, ls="--", alpha=0.6)
    ax1.set_ylabel("净值（初始=1）")
    ax1.set_title("优化器二 GMVP Top10：净值与最大回撤区间",
                  fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3, ls=":")

    ax2.fill_between(dd.index, dd.values * 100, 0, color="#c00000", alpha=0.35)
    ax2.plot(dd.index, dd.values * 100, color="#c00000", lw=1)
    ax2.set_ylabel("回撤（%）")
    ax2.grid(alpha=0.3, ls=":")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def benchmark_table(nav, benches):
    """策略与双基准的全期指标和分年度收益。Full-sample stats and annual returns."""
    series = {"策略（GMVP Top10）": nav, **benches}
    rows = []
    for label, s in series.items():
        r = s.pct_change().fillna(s.iloc[0] - 1)
        total = s.iloc[-1] - 1
        ann = (1 + total) ** (12 / len(s)) - 1
        vol = r.std(ddof=1) * np.sqrt(12)
        rows.append({
            "组合": label,
            "累计收益": total,
            "年化收益": ann,
            "年化波动": vol,
            "夏普": ann / vol,
            "最大回撤": (s / s.cummax() - 1).min(),
        })
    table = pd.DataFrame(rows).set_index("组合")
    annual = pd.DataFrame({
        label: (1 + s.pct_change().fillna(s.iloc[0] - 1)).groupby(
            s.index.year).prod() - 1
        for label, s in series.items()})
    return table, annual


def plot_rebalance_counts(long_csv, path):
    """每月新进 / 退出股票只数。Monthly buys and sells."""
    df = pd.read_csv(long_csv, dtype={"股票代码": str}, parse_dates=["调仓日"])
    sets = {dt: set(sub["股票代码"]) for dt, sub in df.groupby("调仓日")}
    dates = sorted(sets)
    buys, sells = [], []
    for prev, cur in zip(dates[:-1], dates[1:]):
        buys.append(len(sets[cur] - sets[prev]))
        sells.append(len(sets[prev] - sets[cur]))

    fig, ax = plt.subplots(figsize=(11, 3.4))
    x = np.arange(len(buys))
    ax.bar(x - 0.2, buys, 0.4, color="#1f4e79", label="买入数量")
    ax.bar(x + 0.2, sells, 0.4, color="#c00000", label="卖出数量")
    ax.set_xticks(x[::6])
    ax.set_xticklabels([dates[i + 1].strftime("%Y-%m") for i in range(0, len(buys), 6)],
                       rotation=0, fontsize=8)
    ax.set_ylabel("股票数量")
    ax.set_title("优化器二 GMVP Top10：每月买入 / 卖出只数",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, ls=":", axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_weight_heatmap(w, path, max_cols=40):
    """日期 × 股票 权重热力图。Date-by-stock weight heatmap."""
    cols = w.columns[w.sum() > 0]
    cols = cols[np.argsort(-w[cols].mean().to_numpy())][:max_cols]
    mat = w[cols].T.to_numpy()

    fig, ax = plt.subplots(figsize=(12, 0.28 * len(cols) + 2))
    im = ax.imshow(mat, aspect="auto", cmap="YlGnBu", vmin=0, vmax=0.25)
    ax.set_xticks(range(0, len(w.index), 6))
    ax.set_xticklabels([d.strftime("%y-%m") for d in w.index[::6]],
                       rotation=0, fontsize=8)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(list(cols), fontsize=7)
    ax.set_title("优化器二 GMVP Top10：持仓权重随时间变化",
                 fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.7, label="权重")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_weights_latest(w_gmvp, w_ew, path):
    """最新一期 GMVP 与等权权重对比（图二下半）。"""
    dt = w_gmvp.index[-1]
    a = w_gmvp.loc[dt]
    b = w_ew.reindex(columns=w_gmvp.columns, fill_value=0.0).loc[dt]
    codes = list(a[a > 0].index) + [c for c in b[b > 0].index if a.get(c, 0) == 0]
    codes = codes[::-1]
    y = np.arange(len(codes))

    fig, ax = plt.subplots(figsize=(8.4, 0.42 * len(codes) + 1.6))
    ax.barh(y + 0.18, [a.get(c, 0) * 100 for c in codes], 0.36,
            color="#1f4e79", label="全局最小方差（GMVP）")
    ax.barh(y - 0.18, [b.get(c, 0) * 100 for c in codes], 0.36,
            color="#c00000", label="等权（EW）")
    ax.set_yticks(y)
    ax.set_yticklabels(codes, fontsize=8)
    ax.set_xlabel("组合权重（%）")
    ax.set_title(f"最新一期权重对比（{dt:%Y-%m}）",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, ls=":", axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_perf_bars(cmp, path):
    """年化收益 / 波动 / 夏普 三面板（图六）。"""
    sub = cmp.loc[f"top{TOP_N}"]
    methods = list(sub.index)
    panels = [("年化收益", "年化收益", "#1f4e79"),
              ("年化波动", "年化波动", "#8c8c8c"),
              ("夏普比率", "夏普比率", "#2e7d32")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    for ax, (col, title, color) in zip(axes, panels):
        vals = sub[col].to_numpy(dtype=float)
        ax.bar(range(len(methods)), vals, 0.62, color=color)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=0, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3, ls=":", axis="y")
    fig.suptitle(f"优化器二 Top{TOP_N}：8 种权重方法绩效对比",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_frontier(rets, w_gmvp, path, n_sim=4000, seed=42):
    """末期名单的随机组合散点 + GMVP / 等权位置（图二上半）。"""
    dt = w_gmvp.index[-1]
    codes = list(w_gmvp.loc[dt][w_gmvp.loc[dt] > 0].index)
    hist = rets.loc[:dt, codes].tail(36).fillna(0.0)
    mu = hist.mean().to_numpy() * 12
    cov = hist.cov().to_numpy() * 12
    cap = 2.0 / len(codes)

    rng = np.random.default_rng(seed)
    raw = rng.random((n_sim, len(codes)))
    raw /= raw.sum(axis=1, keepdims=True)
    raw = np.minimum(raw, cap)
    raw /= raw.sum(axis=1, keepdims=True)
    vols = np.sqrt(np.einsum("ij,jk,ik->i", raw, cov, raw))
    rets_ = raw @ mu

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.scatter(vols * 100, rets_ * 100, s=5, color="#8c8c8c", alpha=0.25,
               label="随机权重组合")
    for label, weights, color in [
        ("GMVP（本月）", w_gmvp.loc[dt, codes].to_numpy(), "#1f4e79"),
        ("等权（本月）", np.full(len(codes), 1 / len(codes)), "#c00000"),
    ]:
        v = float(np.sqrt(weights @ cov @ weights))
        r_ = float(weights @ mu)
        ax.scatter([v * 100], [r_ * 100], s=90, color=color, label=label,
                   edgecolor="white", zorder=3)
    ax.set_xlabel("年化波动（%）")
    ax.set_ylabel("年化收益（%）")
    ax.set_title(f"末期名单的有效前沿（{dt:%Y-%m}，36 个月估计）",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, ls=":")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_weights_gif(w, path, n_show=12):
    """每月权重条形图滚动动画。Rolling weight bar animation."""
    cols = w.columns[w.sum() > 0]
    cols = list(cols[np.argsort(-w[cols].mean().to_numpy())][:n_show])
    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    def draw(i):
        ax.clear()
        vals = w.iloc[i][cols] * 100
        ax.barh(list(cols)[::-1], vals.to_numpy()[::-1], color="#1f4e79")
        ax.set_xlim(0, 25)
        ax.set_xlabel("权重（%）")
        ax.set_title(f"优化器二 GMVP Top10 权重（{w.index[i]:%Y-%m}）",
                     fontsize=11)
        ax.grid(alpha=0.3, ls=":", axis="x")

    anim = FuncAnimation(fig, draw, frames=len(w), interval=250)
    anim.save(path, writer=PillowWriter(fps=4))
    plt.close(fig)


def main():
    print("载入缓存数据...")
    rets = pd.read_csv(OUT / "monthly_returns.csv.gz", index_col=0,
                       parse_dates=True)
    gmvp_csv = OUT / f"top{TOP_N}" / "GMVP" / "holdings.csv"
    ew_csv = OUT / f"top{TOP_N}" / "EW" / "holdings.csv"
    w_gmvp = load_weights(gmvp_csv, rets.columns)
    w_ew = load_weights(ew_csv, rets.columns)

    r = rets.loc[w_gmvp.index].fillna(0.0)
    out = calc_nav(r, w_gmvp, comsn=COMSN)
    nav = out["nav"]

    csi300_close = load_benchmark(pv.START, pv.END)["close"]
    csi300 = csi300_close.reindex(nav.index)
    csi300 = (1 + csi300.pct_change(fill_method=None).fillna(0)).cumprod()
    csi500_close = load_benchmark(pv.START, pv.END, index="csi500")["close"]
    csi500 = csi500_close.reindex(nav.index)
    csi500 = (1 + csi500.pct_change(fill_method=None).fillna(0)).cumprod()
    benches = {"沪深300": csi300, "中证500": csi500}

    print("[1/7] 净值与回撤区间（双基准）")
    plot_nav_drawdown(nav, benches, EXTRA / "nav_drawdown.png")
    table, annual = benchmark_table(nav, benches)
    table.to_csv(EXTRA / "benchmark_compare.csv", encoding="utf-8-sig")
    annual.to_csv(EXTRA / "benchmark_annual.csv", encoding="utf-8-sig")
    print((table * 100).round(2).to_string())
    print((annual * 100).round(2).to_string())
    print(f"  净值终值 {nav.iloc[-1]:.3f}，最大回撤 "
          f"{(nav / nav.cummax() - 1).min() * 100:.2f}%")

    print("[2/7] 每月买卖只数")
    plot_rebalance_counts(gmvp_csv, EXTRA / "rebalance_counts.png")

    print("[3/7] 权重热力图")
    plot_weight_heatmap(w_gmvp, EXTRA / "weight_heatmap.png")

    print("[4/7] 最新一期权重对比")
    plot_weights_latest(w_gmvp, w_ew, EXTRA / "weights_latest.png")

    print("[5/7] 绩效三面板")
    cmp = pd.read_csv(OUT / "comparison.csv", index_col=[0, 1])
    plot_perf_bars(cmp, EXTRA / "perf_bars.png")

    print("[6/7] 有效前沿")
    plot_frontier(rets, w_gmvp, EXTRA / "efficient_frontier.png")

    print("[7/7] 权重滚动 GIF")
    plot_weights_gif(w_gmvp, EXTRA / "weights_rolling.gif")

    print(f"完成：{EXTRA}")


if __name__ == "__main__":
    main()
