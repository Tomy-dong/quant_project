# -*- coding: utf-8 -*-
"""因子池重叠度审计：WorldQuant Alpha101 与华泰量价因子是否重复。

目的
----
老师建议先用 WorldQuant 量价因子、暂时不用华泰因子。本脚本回答两个问题：

1. 两套因子之间是否存在公式级重复；
2. 是否存在高度相关的“近似重复”因子，加入华泰后会不会只是重复计数。

方法
----
- 用与主策略完全相同的月末面板、可交易池和预处理口径；
- 每月计算因子两两之间的 Spearman 截面相关，再对月份取平均；
- 以 |平均相关| >= 0.7 视为近似重复，列出配对与聚类；
- 同时给出全样本与 2015-2019 训练窗口的 RankIC / ICIR，便于下一步筛选。

输出
----
results/rescreen/
    factor_ic.csv               全样本与训练窗口 IC / ICIR
    high_corr_pairs.csv         |平均相关| >= 0.7 的因子配对
    factor_corr_heatmap.png     因子平均截面相关热力图

用法
----
    python baseline/factor_corr_audit.py

Factor-pool overlap audit: are WorldQuant Alpha101 and Huatai price-volume
factors duplicates? Monthly cross-sectional Spearman correlations are averaged
over time; pairs with |corr| >= 0.7 are treated as near-duplicates.
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
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import price_volume_baseline as pv  # noqa: E402
from utils.data_standardize import preprocess  # noqa: E402

OUT = HERE.parent / "results" / "rescreen"
OUT.mkdir(parents=True, exist_ok=True)

WQ = ["Alpha4", "Alpha6", "Alpha9", "Alpha12", "Alpha13", "Alpha14", "Alpha15",
      "Alpha16", "Alpha18", "Alpha20", "Alpha23", "Alpha26", "Alpha30",
      "Alpha33", "Alpha34", "Alpha35", "Alpha38", "Alpha40", "Alpha43",
      "Alpha53", "Alpha101"]
HUATAI = ["return_1m", "return_3m", "return_6m", "return_12m",
          "wgt_return_1m", "wgt_return_3m", "wgt_return_6m", "wgt_return_12m",
          "exp_wgt_return_3m", "exp_wgt_return_6m", "exp_wgt_return_12m",
          "HAlpha"]
EXT = ["volatility_3m", "volatility_6m", "turnover_mean_3m",
       "turnover_std_3m", "amihud_3m"]


def pool_of(name):
    if name in WQ:
        return "WorldQuant"
    if name in HUATAI:
        return "Huatai"
    return "Extension"


def avg_corr(zfactors, dates):
    """月度截面 Spearman 相关的平均值。Average monthly cross-sectional corr."""
    names = list(zfactors)
    total = np.zeros((len(names), len(names)))
    count = 0
    for dt in dates:
        df = pd.DataFrame({n: zfactors[n].loc[dt] for n in names})
        if df.notna().sum().min() < 100:
            continue
        c = df.corr(method="spearman", min_periods=100).to_numpy()
        c = np.nan_to_num(c, nan=0.0)
        total += c
        count += 1
    return pd.DataFrame(total / max(count, 1), index=names, columns=names)


def ic_stats(zfactors, returns, dates):
    """全样本与训练窗口 RankIC / ICIR。Full-sample and training-window IC/ICIR."""
    ret_rank = returns.rank(axis=1)
    train_start = pd.Timestamp("2015-01-01")
    train_end = pd.Timestamp("2019-12-31")
    rows = []
    for name, fac in zfactors.items():
        fv = fac.rank(axis=1).values
        rv = ret_rank.values
        ic = pd.Series(np.nan, index=fac.index, dtype=float)
        for i in range(len(fac.index) - 1):
            a, b = fv[i], rv[i + 1]
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() > 2:
                ic.iloc[i + 1] = np.corrcoef(a[m], b[m])[0, 1]
        ic = ic.dropna()
        tr = ic.loc[(ic.index > train_start) & (ic.index <= train_end)]
        std = ic.std(ddof=1)
        tr_std = tr.std(ddof=1)
        rows.append({
            "因子": name,
            "因子池": pool_of(name),
            "全样本平均IC": ic.mean(),
            "全样本ICIR": ic.mean() / std if std > 0 else np.nan,
            "训练期平均IC": tr.mean(),
            "训练期ICIR": tr.mean() / tr_std if tr_std > 0 else np.nan,
            "样本月数": len(ic),
        })
    return pd.DataFrame(rows)


def high_corr_pairs(corr, threshold=0.7):
    """提取高度相关的因子配对。Extract highly correlated factor pairs."""
    rows = []
    names = list(corr.index)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = corr.iloc[i, j]
            if abs(r) >= threshold:
                rows.append({"因子A": names[i], "因子B": names[j],
                             "平均相关": r, "池A": pool_of(names[i]),
                             "池B": pool_of(names[j])})
    return pd.DataFrame(rows).sort_values("平均相关", key=abs, ascending=False)


def corr_clusters(corr, threshold=0.7):
    """按 |corr| 层次聚类，返回每个因子的簇编号。Cluster factors by |corr|."""
    dist = 1 - corr.abs().to_numpy()
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2
    labels = fcluster(linkage(squareform(dist, checks=False), method="average"),
                      1 - threshold, criterion="distance")
    return pd.Series(labels, index=corr.index, name="簇")


def plot_heatmap(corr):
    """相关热力图。Correlation heatmap."""
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.index, fontsize=7)
    ax.set_title("因子平均截面 Spearman 相关（2015-2025）")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT / "factor_corr_heatmap.png", dpi=150)
    plt.close(fig)


def main():
    print("因子池重叠度审计 | WorldQuant vs 华泰")
    print("=" * 72)
    print("[1/3] 载入数据并计算 38 个量价因子（约数分钟）...")
    daily = pv.load_market_data(pv.MARKET, pv.START, pv.END)
    monthly = pv.build_monthly(daily)
    dates = monthly["dates"]
    returns = monthly["returns"]
    valid = pv.filter_pool(monthly)
    factors = pv.compute_price_volume_factors(monthly, valid)
    zfactors = {n: preprocess(f) for n, f in factors.items()}

    print("[2/3] 计算平均截面相关与 IC...")
    corr = avg_corr(zfactors, dates)
    corr.to_csv(OUT / "factor_corr.csv", encoding="utf-8-sig")
    pairs = high_corr_pairs(corr, 0.7)
    pairs.to_csv(OUT / "high_corr_pairs.csv", index=False, encoding="utf-8-sig")
    ic = ic_stats(zfactors, returns, dates)
    ic.to_csv(OUT / "factor_ic.csv", index=False, encoding="utf-8-sig")

    print("[3/3] 输出热力图与聚类...")
    plot_heatmap(corr)
    clusters = corr_clusters(corr, 0.7)

    print("\n高度相关配对（|平均相关| >= 0.7，共 "
          f"{len(pairs)} 对）：")
    show = pairs.head(40).copy()
    show["平均相关"] = show["平均相关"].round(3)
    print(show.to_string(index=False))

    print("\n跨因子池的高相关配对（WorldQuant vs 华泰）：")
    cross = pairs[(pairs["池A"] != pairs["池B"])]
    if cross.empty:
        print("  无：两套因子没有 |相关| >= 0.7 的跨池配对")
    else:
        c = cross.copy()
        c["平均相关"] = c["平均相关"].round(3)
        print(c.to_string(index=False))

    print("\n相关簇（|corr| >= 0.7 视为同一簇，仅列多成员簇）：")
    members = clusters.groupby(clusters).groups
    for k, idx in members.items():
        if len(idx) > 1:
            print(f"  簇{k}: {', '.join(idx)}")

    print("\n近似重复（每个因子的最高相关对象）：")
    dup = []
    for name in corr.index:
        row = corr.loc[name].drop(name)
        other = row.abs().idxmax()
        dup.append({"因子": name, "最相关因子": other,
                    "相关": corr.loc[name, other]})
    dup = pd.DataFrame(dup).sort_values("相关", key=abs, ascending=False)
    print(dup.round(3).to_string(index=False))

    print(f"\n结果已保存到：{OUT}")


if __name__ == "__main__":
    main()
