# -*- coding: utf-8 -*-
"""回测结果统一出图 / Standard backtest report plots.

课程要求的结果图集中在这里，避免每个脚本各画一套：

1. 净值 + 基准
2. 策略回撤
3. 月度收益热力图（年 × 月）
4. 年度收益柱状图
5. 月度换手率
6. 合成因子 IC 时序（月度 IC + 12 月滚动均值 + 累计 IC）
7. 五分组累计收益

每个函数只负责画图，指标计算放在调用方，便于复用同一套图对比不同策略。

Usage
-----
    from utils.report_plots import plot_all
    plot_all(outputs, bench_nav, out_dir, title, ic_series=ic, group_ret=grp)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

N_GROUP = 5
GROUP_COLORS = ["#8c8c8c", "#5b9bd5", "#70ad47", "#ed7d31", "#c00000"]


# ===================== 基础指标 / Basic metrics =====================
def annual_returns(pnl):
    """把月度收益按年复利成年度收益。Compound monthly returns into yearly returns."""
    grouped = (1 + pnl).groupby(pnl.index.year).prod() - 1
    grouped.index.name = "年份"
    return grouped


def cumulative(series):
    """累计净值或累计收益。Cumulative wealth index."""
    return (1 + series.fillna(0)).cumprod()


def drawdown(nav):
    """相对历史高点的回撤（负值）。Drawdown from the running peak."""
    return nav / nav.cummax() - 1


def group_returns(score, returns, n_group=N_GROUP):
    """按合成分数逐月五分组，返回各组等权月收益（索引 t+1）。
    Monthly equal-weighted returns of n_group score-sorted portfolios, indexed t+1.
    """
    out = []
    idx = []
    cols = [f"Q{i + 1}" for i in range(n_group)]
    for i in range(len(score.index) - 1):
        df = pd.DataFrame({"s": score.iloc[i], "r": returns.iloc[i + 1]}).dropna()
        if len(df) < n_group * 20:
            continue
        try:
            g = pd.qcut(df["s"], n_group, labels=False, duplicates="drop")
        except ValueError:
            continue
        means = df.groupby(g)["r"].mean()
        if len(means) < n_group:
            continue
        out.append(means.to_numpy())
        idx.append(score.index[i + 1])
    return pd.DataFrame(out, index=idx, columns=cols)


# ===================== 单张图 / Individual figures =====================
def plot_nav(nav, bench_nav, path, title):
    """策略净值与基准净值。Strategy NAV versus benchmark NAV."""
    fig, ax = plt.subplots(figsize=(10, 5))
    nav.plot(ax=ax, linewidth=1.6, color="#1f4e79", label="策略净值")
    if bench_nav is not None:
        bench_nav.plot(ax=ax, linewidth=1.4, color="#8c8c8c", label="沪深300")
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set(xlabel="日期", ylabel="净值（初始=1）", title=title)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_drawdown(nav, path, title):
    """策略回撤曲线。Strategy drawdown."""
    dd = drawdown(nav) * 100
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.fill_between(dd.index, dd.to_numpy(), 0, color="#c0392b", alpha=0.35)
    dd.plot(ax=ax, color="#c0392b", linewidth=1.2)
    ax.set(xlabel="日期", ylabel="回撤（%）", title=title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_monthly_heatmap(pnl, path, title):
    """月度收益热力图，行=年份，列=月份。Monthly return heatmap (year x month)."""
    df = pnl.to_frame("r")
    df["year"] = df.index.year
    df["month"] = df.index.month
    piv = df.pivot_table(index="year", columns="month", values="r") * 100
    values = piv.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1.0

    fig, ax = plt.subplots(figsize=(10, 0.55 * len(piv) + 1.8))
    im = ax.imshow(values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{m}月" for m in piv.columns])
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                        color="black" if abs(v) < 0.6 * vmax else "white")
    ax.set(title=title, xlabel="月份", ylabel="年份")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="月收益（%）")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_annual_bars(pnl, path, title):
    """年度收益柱状图。Annual returns bar chart."""
    ann = annual_returns(pnl) * 100
    colors = ["#2e7d32" if v >= 0 else "#c0392b" for v in ann]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(ann.index.astype(str), ann.to_numpy(), color=colors, width=0.6)
    for x, v in zip(range(len(ann)), ann.to_numpy()):
        ax.text(x, v, f"{v:.1f}%", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(xlabel="年份", ylabel="收益（%）", title=title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_turnover(turnover, path, title, roll=12):
    """月度单边换手率与滚动均值。Monthly one-way turnover and its rolling mean."""
    to = turnover * 100
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(to.index, to.to_numpy(), width=20, color="#5b9bd5", alpha=0.55,
           label="月度换手")
    to.rolling(roll, min_periods=1).mean().plot(
        ax=ax, color="#c0392b", linewidth=1.6, label=f"{roll}月滚动均值")
    ax.set(xlabel="日期", ylabel="单边换手率（%）", title=title)
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ic_series(ic, path, title, roll=12):
    """合成因子 IC 时序：月度 IC 柱 + 滚动均值 + 累计 IC。Composite IC over time."""
    ic = ic.dropna()
    if ic.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1.2]})
    ax = axes[0]
    colors = ["#2e7d32" if v >= 0 else "#c0392b" for v in ic]
    ax.bar(ic.index, ic.to_numpy(), width=20, color=colors, alpha=0.6,
           label="月度 RankIC")
    ic.rolling(roll, min_periods=1).mean().plot(
        ax=ax, color="#1f4e79", linewidth=1.6, label=f"{roll}月滚动均值")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(ic.mean(), color="#ed7d31", linewidth=1.0, linestyle="--",
               label=f"均值 {ic.mean():.3f}")
    ax.set(ylabel="RankIC", title=title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ic.cumsum().plot(ax=ax2, color="#1f4e79", linewidth=1.6)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set(xlabel="日期", ylabel="累计 IC")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_group_returns(grp, path, title):
    """五分组累计净值，Q1 最低分组、Q5 最高分组。Cumulative NAV of 5 score groups."""
    if grp is None or grp.empty:
        return
    nav = (1 + grp.fillna(0)).cumprod()
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, col in enumerate(grp.columns):
        color = GROUP_COLORS[i % len(GROUP_COLORS)]
        nav[col].plot(ax=ax, linewidth=1.5, color=color, label=col)
        ann = nav[col].iloc[-1] ** (12 / len(nav)) - 1
        ax.annotate(f"{col} 年化 {ann * 100:.1f}%",
                    xy=(nav.index[-1], nav[col].iloc[-1]),
                    xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color=color, va="center")
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set(xlabel="日期", ylabel="累计净值", title=title)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ===================== 一次出全套 / Full report =====================
def plot_all(outputs, bench_nav, out_dir, title, ic=None, grp=None):
    """按课程要求一次性输出全部结果图。Write the full figure set.

    outputs : calc_nav 返回的 nav / pnl / turnover
    bench_nav : 基准净值（与 nav 同索引），可为 None
    ic : 合成因子月度 RankIC，可为 None
    grp : group_returns 输出的五分组月收益，可为 None
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_nav(outputs["nav"], bench_nav, out_dir / "nav.png", title)
    plot_drawdown(outputs["nav"], out_dir / "drawdown.png", title)
    plot_monthly_heatmap(outputs["pnl"], out_dir / "monthly_heatmap.png", title)
    plot_annual_bars(outputs["pnl"], out_dir / "annual_returns.png", title)
    plot_turnover(outputs["turnover"], out_dir / "turnover.png", title)
    if ic is not None:
        plot_ic_series(ic, out_dir / "ic_series.png", title)
    if grp is not None:
        plot_group_returns(grp, out_dir / "group_returns.png", title)
    return sorted(p.name for p in out_dir.glob("*.png"))
