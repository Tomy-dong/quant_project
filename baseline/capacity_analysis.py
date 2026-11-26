# -*- coding: utf-8 -*-
"""主策略资金容量（AUM）测算：持仓 × 60 日 ADV → 参与率与冲击成本。

口径
----
- 持仓：results/main_strategy/holdings_main.xlsx（日期 / 股票代码 / 权重），
  与 12.1 的主策略完全一致（固定 K=4 + ICIR + IVar，Top100，月频）。
- 流动性：个股 60 日平均成交额 ADV（data/amount_{sh,sz}.csv）。
- 换手：先按 6.2 节的漂移口径把上期权重顺延到本期（w×(1+r) 后归一化），
  再与本期目标权重取绝对差，首次建仓按整仓计。这样与主回测的换手口径一致。
- 冲击成本：平方根模型，单边成本 = k × sqrt(参与率%)，k 为 1% 参与率对应的
  bp 数，基准取 k=10，另做 k=5 / k=20 敏感性。参与率 = 交易金额 / ADV。
- 11.24% 已是扣除单边千二后的年化，因此净年化 = 11.24% − 年化冲击成本。

输出 results/capacity/
    capacity.csv    AUM × k → 平均/95 分位参与率、超限比例、年化冲击成本、净年化
    capacity.png    AUM ↔ 净年化（三档 k）+ 95 分位参与率

用法
----
    python baseline/capacity_analysis.py
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

from factor_lib.matrices import _read_field  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

RESULT = HERE.parent / "results"
HOLD_PATH = RESULT / "main_strategy" / "holdings_main.xlsx"
OUT = RESULT / "capacity"
OUT.mkdir(parents=True, exist_ok=True)

BASE_ANN = 0.1124          # 12.1 主策略年化（已扣单边千二）
COMSN = 0.002              # 单边费率，仅用于对照换手成本
AUM_GRID = [1e8, 3e8, 5e8, 1e9, 2e9, 3e9, 5e9, 1e10]
K_GRID = [5, 10, 20]       # 1% 参与率对应的单边冲击（bp）
P_CAP = 0.10               # 单票参与率上限 10%
ADV_WINDOW = 60


def load_weights():
    df = pd.read_excel(HOLD_PATH)
    df.columns = [str(c).strip() for c in df.columns]
    df["日期"] = pd.to_datetime(df["日期"])
    w = df.pivot_table(index="日期", columns="股票代码", values="权重",
                       aggfunc="sum").fillna(0.0)
    return w.sort_index()


def load_adv(dates, codes):
    amount = _read_field("amount")
    adv = amount.rolling(ADV_WINDOW, min_periods=20).mean()
    adv = adv.reindex(index=dates, columns=codes)
    adv = adv.ffill()
    # 缺失时用当日横截面中位数兜底，避免把缺失当成零流动性
    adv = adv.apply(lambda row: row.fillna(row.median()), axis=1)
    return adv


def monthly_returns(dates, codes):
    """月末后复权收益（用于权重漂移）。Month-end adjusted returns."""
    close = _read_field("close")
    adj = _read_field("adj_factor")
    adj = adj.reindex(index=close.index, columns=close.columns)
    adj = adj.fillna(1.0).replace(0.0, 1.0)
    px = (close * adj).reindex(index=dates, columns=codes)
    ret = px.pct_change(fill_method=None)
    return ret.fillna(0.0)


def drifted_weights(w, ret):
    """上期目标权重按本期实际收益漂移后归一化。Drift then normalise."""
    drift = w.shift(1).fillna(0.0) * (1.0 + ret)
    tot = drift.sum(axis=1)
    return drift.div(tot.replace(0.0, np.nan), axis=0).fillna(0.0)


def main():
    w = load_weights()
    dates = w.index
    codes = w.columns
    print(f"持仓：{len(dates)} 期 × {len(codes)} 只（{dates.min().date()} ~ "
          f"{dates.max().date()}）")

    dw = (w - drifted_weights(w, monthly_returns(dates, codes))).abs()
    turn_m = dw.sum(axis=1)
    turn_ann = float(turn_m.mean() * 12)
    fee_ann = float((dw.mul(COMSN).sum(axis=1)).mean() * 12)
    print(f"年化单边换手 {turn_ann:.2f} 倍；按千二计的费率拖累 {fee_ann*100:.2f}%/年")

    adv = load_adv(dates, codes)
    print(f"ADV 覆盖：{adv.notna().mean().mean()*100:.1f}%")

    rows = []
    for aum in AUM_GRID:
        trade = dw * aum
        part = (trade / adv).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        for k in K_GRID:
            impact_bps = k * np.sqrt(part * 100.0)
            cost_rate = (trade * impact_bps / 1e4).sum(axis=1) / aum
            ann_cost = float(cost_rate.mean() * 12)
            flat = part.values.ravel()
            flat = flat[flat > 0]
            rows.append({
                "AUM(亿)": aum / 1e8,
                "k(bp@1%)": k,
                "平均参与率(%)": float(flat.mean() * 100),
                "95分位参与率(%)": float(np.percentile(flat, 95) * 100),
                "参与率>5%笔数占比(%)": float((flat > 0.05).mean() * 100),
                "参与率>10%笔数占比(%)": float((flat > 0.10).mean() * 100),
                "年化冲击成本(%)": ann_cost * 100,
                "净年化(%)": (BASE_ANN - ann_cost) * 100,
            })

    cap = pd.DataFrame(rows)
    cap.to_csv(OUT / "capacity.csv", index=False, encoding="utf-8-sig")

    print("\nAUM 容量测算（基准 k=10bp@1%）")
    print(cap[cap["k(bp@1%)"] == 10].to_string(index=False,
                                               float_format=lambda x: f"{x:.2f}"))

    # 容量阈值：净年化仍高于沪深300（2.49%）的最大 AUM
    base = cap[cap["k(bp@1%)"] == 10]
    ok = base[base["净年化(%)"] > 2.49]
    print(f"\n净年化仍高于沪深300（2.49%）的最大 AUM："
          f"{(ok['AUM(亿)'].max() if not ok.empty else 0):.0f} 亿")
    ok2 = base[base["95分位参与率(%)"] <= P_CAP * 100]
    print(f"95 分位参与率 ≤ 10% 的最大 AUM："
          f"{(ok2['AUM(亿)'].max() if not ok2.empty else 0):.0f} 亿")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for k in K_GRID:
        sub = cap[cap["k(bp@1%)"] == k]
        axes[0].plot(sub["AUM(亿)"], sub["净年化(%)"], "o-", label=f"k={k}bp")
    axes[0].axhline(2.49, color="steelblue", ls=":", lw=1, label="沪深300 2.49%")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("AUM（亿元，对数轴）")
    axes[0].set_ylabel("净年化收益（%）")
    axes[0].set_title("资金规模 vs 净年化（已扣千二）")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(base["AUM(亿)"], base["95分位参与率(%)"], "o-",
                 color="darkorange", label="95 分位参与率")
    axes[1].axhline(P_CAP * 100, color="red", ls=":", lw=1, label="10% 上限")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("AUM（亿元，对数轴）")
    axes[1].set_ylabel("单票参与率（%）")
    axes[1].set_title("资金规模 vs 单票参与率（k=10bp）")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.suptitle("主策略资金容量测算（2020–2025，Top100，月频）", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "capacity.png", dpi=150)
    plt.close(fig)
    print(f"\n已输出 {OUT/'capacity.csv'} 与 {OUT/'capacity.png'}")


if __name__ == "__main__":
    main()
