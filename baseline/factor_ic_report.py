# -*- coding: utf-8 -*-
"""候选因子 IC 对比图（对应课件“因子处理与检验”一节）。

因子本身是数据字段，每年都在算，不存在“只在某一年有效”的因子；动态变化的
是优化器一每年挑中哪些因子、给多少权重。因此这里对每个因子在样本外
2020-2025 全区间计算 IC，并用年份底纹 + 标题标注它在哪些年份被选中，
既保留课件那种“单期 IC 柱 + 平均 IC 线 + 累计 IC 曲线”的画法，又能看出
因子集随年份变化。

The factors are data fields computed every year; nothing is “a factor of one
year”. What changes is which factors Optimizer 1 picks each year. This script
therefore computes IC over the whole 2020-2025 out-of-sample window and marks
the years in which each factor was selected.

输出 results/factor_ic/
    factor_ic_summary.csv        每个因子的平均 IC / ICIR / 胜率 / 累计 IC / 入选年份
    factor_ic_grid.png           入选因子的单期 IC + 平均 IC + 累计 IC 小图矩阵
    factor_ic_yearly_heatmap.png 因子 × 年份 平均 IC 热力图（体现动态变化）
    ic_panels_top3.png           三个代表因子的 Rank IC / Pearson IC 双联图

用法
----
    python baseline/factor_ic_report.py
"""
import sys
import warnings
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import price_volume_baseline as pv  # noqa: E402
import rolling_multifactor as rm  # noqa: E402
from utils.data_standardize import preprocess  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = HERE.parent / "results" / "factor_ic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SELECTION_FILE = HERE.parent / "results" / "optimizer1" / "factor_selection.csv"

Y0, Y1 = rm.OOS_START_YEAR, rm.OOS_END_YEAR
BAR_COLOR = "#4c78a8"
MEAN_COLOR = "#d62728"
CUM_COLOR = "#2ca02c"
PICK_COLOR = "#ffd966"


def load_selection():
    """读取逐年入选因子，汇总每个因子的入选年份与方向。
    Aggregate the yearly selections into years and directions per factor.
    """
    df = pd.read_csv(SELECTION_FILE)
    agg = df.groupby("因子").agg(
        入选年份=("样本外年份", lambda s: sorted(int(v) for v in s)),
        方向=("方向", lambda s: "/".join(sorted({str(int(v)) for v in s}))),
    )
    return agg


def ic_stats(ic):
    """IC 描述统计。/ IC descriptive statistics."""
    mean, std = ic.mean(), ic.std()
    return pd.DataFrame({
        "平均IC": mean,
        "ICIR": mean / std,
        "IC>0月度占比": (ic > 0).mean(),
        "累计IC": ic.sum(),
    })


def draw_ic_panel(ax, series, title):
    """单期 IC 柱 + 平均 IC 线 + 累计 IC（右轴）。 """
    s = series.dropna()
    ax.bar(s.index, s.values, width=20, color=BAR_COLOR, alpha=0.85,
           label="单期 IC")
    ax.axhline(s.mean(), color=MEAN_COLOR, lw=1.2, ls="--", label="平均 IC")
    ax.axhline(0, color="black", lw=0.6, alpha=0.6)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=8)

    ax2 = ax.twinx()
    ax2.plot(s.index, s.cumsum().values, color=CUM_COLOR, lw=1.5,
             label="累计 IC")
    ax2.tick_params(labelsize=8)


def plot_grid(ic, years_map, path):
    """入选因子的 IC 小图矩阵，选中年份加底纹。Small-multiple IC grid."""
    cols = 4
    rows = int(np.ceil(len(ic.columns) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 2.9 * rows),
                             squeeze=False)
    for ax, name in zip(axes.flat, ic.columns):
        for y in years_map.get(name, []):
            ax.axvspan(pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31"),
                       color=PICK_COLOR, alpha=0.35, lw=0)
        years = "/".join(str(y) for y in years_map.get(name, []))
        draw_ic_panel(ax, ic[name], f"{name}（入选 {years}）")
    for ax in list(axes.flat)[len(ic.columns):]:
        ax.axis("off")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=BAR_COLOR, alpha=0.85),
        plt.Line2D([], [], color=MEAN_COLOR, ls="--", lw=1.2),
        plt.Line2D([], [], color=CUM_COLOR, lw=1.5),
        plt.Rectangle((0, 0), 1, 1, color=PICK_COLOR, alpha=0.35),
    ]
    labels = ["单期 IC（左轴）", "平均 IC（左轴）",
              "累计 IC（右轴）", "入选年份"]
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=10,
               frameon=False)
    fig.suptitle(f"入选因子逐月 Rank IC（样本外 {Y0}-{Y1}）",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_yearly_heatmap(ic, path):
    """因子 × 年份 平均 IC 热力图，直接展示动态变化。
    Factor-by-year mean IC heatmap.
    """
    yearly = ic.groupby(ic.index.year).mean().T
    yearly = yearly.loc[yearly.mean(axis=1).sort_values(ascending=False).index]
    vmax = float(np.nanmax(np.abs(yearly.to_numpy())))

    fig, ax = plt.subplots(figsize=(8.4, 0.42 * len(yearly) + 2.2))
    im = ax.imshow(yearly.to_numpy(), cmap="RdYlGn", vmin=-vmax, vmax=vmax,
                   aspect="auto")
    ax.set_xticks(range(len(yearly.columns)))
    ax.set_xticklabels([str(c) for c in yearly.columns], fontsize=9)
    ax.set_yticks(range(len(yearly.index)))
    ax.set_yticklabels(list(yearly.index), fontsize=8)
    for i in range(len(yearly.index)):
        for j in range(len(yearly.columns)):
            v = yearly.iloc[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7.5)
    ax.set_title(f"入选因子分年度平均 Rank IC（样本外 {Y0}-{Y1}）",
                 fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.75, label="平均 IC")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pearson_ic_series(z, returns):
    """逐月 Pearson IC（因子 t 与收益 t+1）。Monthly Pearson IC."""
    fwd = returns.shift(-1)
    out = {}
    for t in z.index:
        if t not in fwd.index:
            continue
        pair = pd.concat([z.loc[t], fwd.loc[t]], axis=1).dropna()
        if len(pair) > 20:
            out[t] = pair.iloc[:, 0].corr(pair.iloc[:, 1])
    return pd.Series(out).sort_index()


def plot_top3(rank_ic, zfactors, returns, names, path):
    """代表因子的 Rank / Pearson 双联图，复刻课件排版。
    Dual-panel Rank/Pearson IC for representative factors.
    """
    fig, axes = plt.subplots(len(names), 2, figsize=(13, 3.1 * len(names)),
                             squeeze=False)
    for i, name in enumerate(names):
        z = zfactors[name].loc[f"{Y0}-01-01":f"{Y1}-12-31"]
        draw_ic_panel(axes[i, 0], rank_ic[name], f"{name} · Rank IC")
        draw_ic_panel(axes[i, 1], pearson_ic_series(z, returns),
                      f"{name} · Pearson IC")
    fig.suptitle(f"代表因子逐月 IC 与累计 IC（样本外 {Y0}-{Y1}）",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    print(f"因子 IC 对比：样本外 {Y0}-{Y1}，输出到 {OUTPUT_DIR}")
    print("[1/4] 载入数据并计算因子...")
    daily = pv.load_market_data(pv.MARKET, pv.START, pv.END)
    monthly = pv.build_monthly(daily)
    returns = monthly["returns"]
    valid = pv.filter_pool(monthly)
    factors = pv.compute_price_volume_factors(monthly, valid)
    zfactors = {name: preprocess(fac) for name, fac in factors.items()}
    print(f"  {len(zfactors)} 个候选因子")

    print("[2/4] 计算逐月 Rank IC...")
    ic_all = pd.DataFrame(rm.rank_ic_panel(zfactors, returns))
    ic_all = ic_all.loc[f"{Y0}-01-01":f"{Y1}-12-31"]

    sel = load_selection()
    names = [n for n in sel.index if n in ic_all.columns]
    ic = ic_all[names]
    years_map = {n: sel.loc[n, "入选年份"] for n in names}
    print(f"  入选因子（并集）{len(names)} 个，样本外 {len(ic)} 期")

    print("[3/4] 汇总与热力图...")
    summary = ic_stats(ic).join(sel[["入选年份", "方向"]])
    summary["入选年份"] = summary["入选年份"].map(
        lambda ys: "/".join(str(y) for y in ys))
    summary = summary.sort_values("ICIR", key=lambda s: s.abs(),
                                  ascending=False)
    summary.index.name = "因子"
    summary.to_csv(OUTPUT_DIR / "factor_ic_summary.csv",
                   encoding="utf-8-sig")
    plot_grid(ic, years_map, OUTPUT_DIR / "factor_ic_grid.png")
    plot_yearly_heatmap(ic, OUTPUT_DIR / "factor_ic_yearly_heatmap.png")

    print("[4/4] 代表因子双联图...")
    top3 = list(summary.index[:3])
    plot_top3(ic, zfactors, returns, top3,
              OUTPUT_DIR / "ic_panels_top3.png")

    show = summary[["平均IC", "ICIR", "IC>0月度占比", "累计IC", "入选年份"]]
    print((show.drop(columns="入选年份") * 100).round(2)
          .assign(入选年份=show["入选年份"]).to_string())
    print(f"\n完成：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
