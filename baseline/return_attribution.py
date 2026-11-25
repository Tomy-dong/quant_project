# -*- coding: utf-8 -*-
"""收益归因：把最终组合（优化器二 GMVP Top10）的月度收益拆成截距、因子贡献与残差。

设计见 docs/strategy.md 第 7 节。每个月末用当期标准化因子暴露对下一月
个股收益做横截面回归：

    r_{i,t+1} = alpha_t + sum_k beta_{k,t} f_{i,k,t} + eps_{i,t}

持仓权重加权后，组合月收益被精确拆成三块：

    r^p_{t+1} = alpha_t
              + sum_k (sum_i w_{i,t} f_{i,k,t}) beta_{k,t}
              + sum_i w_{i,t} eps_{i,t}

因子集合与方向读取优化器一（optimizer1_heuristic.py）的
results/optimizer1/factor_selection.csv；持仓与**实际个股权重**读取优化器二
results/optimizer2/top10/GMVP/holdings.csv，不再假设等权，因此归因结果与
results/optimizer2/top10/GMVP/ 的回测一一对应。

用法
----
    python baseline/optimizer1_heuristic.py    # 生成因子集与合成分数
    python baseline/optimizer2_weights.py      # 生成 GMVP Top10 持仓与权重
    python baseline/return_attribution.py

Return attribution: decompose the monthly returns of the final portfolio
(Optimizer 2, GMVP Top10) into an intercept, factor contributions and a
residual. The factor set and directions come from
results/optimizer1/factor_selection.csv; holdings and actual weights come from
results/optimizer2/top10/GMVP/holdings.csv, so no equal-weight assumption is
made and the decomposition matches that backtest exactly.
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

OUTPUT_DIR = HERE.parent / "results" / "attribution"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SELECTION_FILE = HERE.parent / "results" / "optimizer1" / "factor_selection.csv"
HOLDINGS_FILE = (HERE.parent / "results" / "optimizer2" / "top10" / "GMVP"
                 / "holdings.csv")

CONFIG_LABEL = "优化器二 GMVP Top10"
MIN_STOCKS = 50           # 回归所需最少股票数 / minimum cross-section size
MIN_COVERAGE = 0.80       # 持仓权重覆盖率下限 / minimum matched weight mass


def load_fits(path=SELECTION_FILE):
    """读取优化器一的逐年因子集合、方向与最终权重。
    Load the optimizer-1 per-year factor set, directions and final weights.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"未找到 {path}，请先运行 baseline/optimizer1_heuristic.py")
    df = pd.read_csv(path)
    fits = {}
    for year, sub in df.groupby("样本外年份"):
        sub = sub.sort_values("最终权重", ascending=False)
        fits[int(year)] = {
            "year": int(year),
            "picked": sub,
            "weights": sub.set_index("因子")["最终权重"],
            "signs": sub.set_index("因子")["方向"].astype(int),
        }
    return fits


def load_holdings(path=HOLDINGS_FILE):
    """读取优化器二导出的逐月持仓与权重。Load monthly holdings and weights."""
    if not path.exists():
        raise FileNotFoundError(
            f"未找到 {path}，请先运行 baseline/optimizer2_weights.py")
    df = pd.read_csv(path, dtype={"股票代码": str})
    holdings = {}
    for dt, sub in df.groupby("调仓日"):
        series = sub.set_index("股票代码")["权重"].astype(float)
        holdings[pd.Timestamp(dt)] = series[series > 0]
    return holdings


def attribute_month(zfactors, names, t, t_next, returns, weights):
    """单月截面回归并分解组合收益。One-month regression and decomposition.

    t       调仓日（因子暴露取这一天）/ rebalance date, exposures are taken here
    t_next  收益月（下一月末）/ return month
    weights 当期持仓权重 / holdings weights of this period
    """
    expo = pd.concat({n: zfactors[n].loc[t] for n in names}, axis=1)
    y = returns.loc[t_next].rename("__ret__")
    df = expo.join(y, how="inner").dropna()
    if len(df) < max(MIN_STOCKS, 10 * len(names)):
        return None

    x = df[names].to_numpy(dtype=float)
    yv = df["__ret__"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(design, yv, rcond=None)
    alpha, betas = coef[0], coef[1:]

    # 持仓权重按回归样本对齐；缺失因子或收益的股票剔除后重新归一化
    # Align weights to the regression sample, then renormalise dropped names.
    w_all = weights.reindex(df.index).fillna(0.0)
    coverage = float(w_all.sum())
    if coverage < MIN_COVERAGE:
        return None
    w = w_all / coverage

    exposure_p = w.to_numpy(dtype=float) @ x
    contrib = exposure_p * betas
    port_ret = float(w.to_numpy(dtype=float) @ yv)
    resid = port_ret - alpha - float(contrib.sum())

    row = {
        "收益月": t_next,
        "调仓日": t,
        "组合收益": port_ret,
        "截距": alpha,
        "残差": resid,
        "回归股票数": len(df),
        "权重覆盖率": coverage,
    }
    for name, c in zip(names, contrib):
        row[f"因子:{name}"] = c
    for name, e in zip(names, exposure_p):
        row[f"暴露:{name}"] = e
    return row


def summarize(monthly, factor_cols):
    """汇总各部分的月均、年化、累计贡献与 t 值。Summarize each component."""
    n = len(monthly)
    series = {"截距": monthly["截距"], "残差": monthly["残差"]}
    for c in factor_cols:
        series[c.replace("因子:", "")] = monthly[c]
    total_cum = monthly["组合收益"].sum()

    rows = []
    for name, s in series.items():
        mean = s.mean()
        std = s.std(ddof=1)
        rows.append({
            "项目": name,
            "月均贡献": mean,
            "年化贡献": mean * 12,
            "累计贡献": s.sum(),
            "累计占比": s.sum() / total_cum if total_cum != 0 else np.nan,
            "t值": mean / std * np.sqrt(n) if std > 0 else np.nan,
        })
    rows.append({
        "项目": "组合收益（合计）",
        "月均贡献": monthly["组合收益"].mean(),
        "年化贡献": monthly["组合收益"].mean() * 12,
        "累计贡献": total_cum,
        "累计占比": 1.0,
        "t值": (monthly["组合收益"].mean() /
                monthly["组合收益"].std(ddof=1) * np.sqrt(n)
                if monthly["组合收益"].std(ddof=1) > 0 else np.nan),
    })
    return pd.DataFrame(rows)


def plot_contribution_bar(summary):
    """各因子累计贡献（含截距与残差）。Cumulative contribution bar chart."""
    only = summary[summary["项目"] != "组合收益（合计）"].copy()
    only = only.sort_values("累计贡献")
    colors = ["#8a8a8a" if k in ("截距", "残差") else "#2f6f9f"
              for k in only["项目"]]
    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.28 * len(only) + 1.6)))
    ax.barh(only["项目"], only["累计贡献"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(xlabel="累计贡献（收益口径）", ylabel="",
           title=f"收益归因：各因子累计贡献（{CONFIG_LABEL}，样本外）")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "attribution_bar.png", dpi=150)
    plt.close(fig)


def plot_cumulative(monthly, factor_cols):
    """累计拆解曲线：组合、截距、因子合计、残差。Cumulative decomposition."""
    factor_total = monthly[factor_cols].sum(axis=1)
    # 先取 numpy 再建表，避免按新的日期索引对齐导致整列为 NaN
    # Convert to numpy first, otherwise the dict of Series is realigned to the
    # new date index and every column becomes NaN.
    cum = pd.DataFrame({
        "组合收益": monthly["组合收益"].cumsum().to_numpy(),
        "截距": monthly["截距"].cumsum().to_numpy(),
        "因子合计": factor_total.cumsum().to_numpy(),
        "残差": monthly["残差"].cumsum().to_numpy(),
    }, index=pd.to_datetime(monthly["收益月"]).to_numpy())

    fig, ax = plt.subplots(figsize=(10, 5))
    cum["组合收益"].plot(ax=ax, linewidth=2, color="black", label="组合收益")
    cum["因子合计"].plot(ax=ax, linewidth=1.5, color="#2f6f9f", label="因子合计")
    cum["截距"].plot(ax=ax, linewidth=1.2, color="#c0392b", label="截距")
    cum["残差"].plot(ax=ax, linewidth=1.2, color="#7f8c8d", label="残差")
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax.set(xlabel="日期", ylabel="累计贡献（收益口径）",
           title=f"收益归因：累计拆解（{CONFIG_LABEL}，样本外）")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "attribution_cum.png", dpi=150)
    plt.close(fig)


def plot_exposure(exposure):
    """组合因子暴露热力图（因子 × 年份）。Portfolio factor exposure heatmap.

    暴露是持仓权重加权后的截面标准化因子值，正数表示超配高因子值股票。
    某年未入选的因子留空（不是 0），避免把“未使用”误读成“零暴露”。
    """
    yearly = exposure.groupby(exposure.index.year).mean().T
    order = yearly.abs().mean(axis=1).sort_values(ascending=False).index
    yearly = yearly.loc[order]
    vmax = float(np.nanmax(np.abs(yearly.to_numpy())))
    masked = np.ma.masked_invalid(yearly.to_numpy())

    fig, ax = plt.subplots(figsize=(8.0, 0.36 * len(yearly) + 2.0))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#e8e8e8")
    im = ax.imshow(masked, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(yearly.columns)))
    ax.set_xticklabels([str(c) for c in yearly.columns], fontsize=9)
    ax.set_yticks(range(len(yearly.index)))
    ax.set_yticklabels(list(yearly.index), fontsize=8)
    for i in range(len(yearly.index)):
        for j in range(len(yearly.columns)):
            v = yearly.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5)
    ax.set_title(f"组合因子暴露（{CONFIG_LABEL}，标准化单位，灰=当年未入选）",
                 fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.75, label="暴露")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "exposure_heatmap.png", dpi=150)
    plt.close(fig)


def main():
    print(f"收益归因：{CONFIG_LABEL} | 样本外 "
          f"{rm.OOS_START_YEAR}-{rm.OOS_END_YEAR}")
    print("=" * 70)

    print("\n[1/4] 载入数据、计算因子与预处理（约数分钟）...")
    daily = pv.load_market_data(pv.MARKET, pv.START, pv.END)
    monthly = pv.build_monthly(daily)
    returns = monthly["returns"]
    valid = pv.filter_pool(monthly)
    factors = pv.compute_price_volume_factors(monthly, valid)
    zfactors = {name: preprocess(fac) for name, fac in factors.items()}
    print(f"  可交易股票池均值：{valid.sum(axis=1).mean():.0f} 只")

    print("[2/4] 读取优化器一因子集与优化器二 GMVP Top10 持仓...")
    fits = load_fits()
    holdings = load_holdings()
    print(f"  持仓 {len(holdings)} 期，平均持股 "
          f"{np.mean([len(w) for w in holdings.values()]):.1f} 只")

    print("[3/4] 逐月横截面回归归因...")
    rows, skipped = [], 0
    for t, wts in holdings.items():
        if t not in returns.index:
            skipped += 1
            continue
        loc = returns.index.get_loc(t)
        if loc + 1 >= len(returns.index):
            skipped += 1
            continue
        t_next = returns.index[loc + 1]
        if t.year not in fits:
            skipped += 1
            continue
        names = list(fits[t.year]["weights"].index)
        row = attribute_month(zfactors, names, t, t_next, returns, wts)
        if row is None:
            skipped += 1
            continue
        rows.append(row)

    monthly_attr = pd.DataFrame(rows)
    if monthly_attr.empty:
        raise RuntimeError("归因样本为空，请检查数据区间与股票池")
    factor_cols = [c for c in monthly_attr.columns if c.startswith("因子:")]

    # 校验：截距 + 因子贡献 + 残差 应与组合收益完全相等
    check = (monthly_attr["截距"] + monthly_attr[factor_cols].sum(axis=1) +
             monthly_attr["残差"] - monthly_attr["组合收益"]).abs().max()
    print(f"  归因月份 {len(monthly_attr)} 期，跳过 {skipped} 期，"
          f"拆分误差上限 {check:.2e}")

    monthly_attr.to_csv(OUTPUT_DIR / "monthly_attribution.csv",
                        index=False, encoding="utf-8-sig")

    expo_cols = [c for c in monthly_attr.columns if c.startswith("暴露:")]
    exposure = monthly_attr[["调仓日"] + expo_cols].copy()
    exposure.columns = ["调仓日"] + [c.replace("暴露:", "") for c in expo_cols]
    exposure["调仓日"] = pd.to_datetime(exposure["调仓日"])
    exposure = exposure.set_index("调仓日")
    exposure.to_csv(OUTPUT_DIR / "exposure.csv", encoding="utf-8-sig")

    print("[4/4] 汇总与绘图...")
    summary = summarize(monthly_attr, factor_cols)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")

    show = summary.copy()
    for c in ["月均贡献", "年化贡献", "累计贡献", "累计占比"]:
        show[c] = (show[c] * 100).round(2)
    show["t值"] = show["t值"].round(2)
    print("\n归因汇总（%，按累计贡献排序）：")
    order = (summary["项目"] != "组合收益（合计）")
    print(show[order].reindex(
        summary[order].sort_values("累计贡献", ascending=False).index
    ).to_string(index=False))
    print(f"\n组合月均收益 {monthly_attr['组合收益'].mean() * 100:.2f}%，"
          f"权重覆盖率均值 {monthly_attr['权重覆盖率'].mean() * 100:.1f}%")

    plot_contribution_bar(summary)
    plot_cumulative(monthly_attr, factor_cols)
    plot_exposure(exposure)
    print(f"\n结果已保存到：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
