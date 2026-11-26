# -*- coding: utf-8 -*-
"""XGBoost v3 选股 + 股票层风险权重，寻找回撤<40%且保住 alpha 的配置。

读 results/xgb_strategy/composite_xgb.csv（v3 预测分数），对 Top30 持仓分别用
EW / IVar / GMVP / MDCP 四种资金权重回测（复用 optimizer2_weights 的协方差估计
与权重求解），比较年化/夏普/回撤/FF3 alpha。
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

import price_volume_baseline as pv          # noqa: E402
import ff3_regression as ff3                # noqa: E402
from data_io import load_benchmark          # noqa: E402
from utils.utils import calc_nav             # noqa: E402
import optimizer2_weights as o2w            # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

RESULT = HERE.parent / "results"
OUT = RESULT / "xgb_riskweight"
OUT.mkdir(parents=True, exist_ok=True)
FF3_PATH = RESULT / "ff3" / "ff3_factors.csv"
XGB_COMPOSITE = RESULT / "xgb_strategy" / "composite_xgb.csv"

TOP_N = 30
COMSN = pv.COMSN
METHODS = ["EW", "IVar", "GMVP", "MDCP"]


def metrics(pnl):
    s = pnl.dropna()
    nav = (1 + s).cumprod()
    total = float(nav.iloc[-1] - 1)
    years = len(s) / 12.0
    ann = float((1 + total) ** (1 / years) - 1)
    vol = float(s.std(ddof=1) * np.sqrt(12))
    return {"年化收益": ann, "年化波动": vol,
            "夏普比率": float(ann / vol) if vol > 0 else np.nan,
            "最大回撤": float((nav / nav.cummax() - 1).min())}


def main():
    print(f"XGBoost v3 + 风险权重 | Top{TOP_N} | 方法 {METHODS}")
    composite = pd.read_csv(XGB_COMPOSITE, index_col=0, parse_dates=True)
    print(f"  载入 composite: {composite.shape}")

    # 月频收益面板（与 optimizer2 同口径）
    rets = o2w.load_monthly_returns(pv.START, pv.END)
    # 对齐到 composite 的日期与股票
    common_codes = composite.columns.intersection(rets.columns)
    composite = composite[common_codes]
    rets = rets[common_codes].reindex(composite.index)

    factors_df = pd.read_csv(FF3_PATH, index_col=0, parse_dates=True)
    factors_df.index.name = "日期"

    rows = {}
    for method in METHODS:
        weights = o2w.build_weight_mats(composite, rets, TOP_N,
                                         methods=[method])[method]
        out = calc_nav(rets.fillna(0), weights, comsn=COMSN)
        pnl = out["pnl"]
        mtr = metrics(pnl)
        reg = ff3.regress_one(method, pnl, factors_df)
        rows[method] = {"pnl": pnl, "metrics": mtr, "reg": reg,
                        "weights": weights}
        print(f"  {method:<5s} 年化 {mtr['年化收益']*100:6.2f}% 夏普 {mtr['夏普比率']:.3f} "
              f"回撤 {mtr['最大回撤']*100:6.2f}% | alpha {reg['alpha年化']*100:+.2f}%"
              f"(p={reg['alpha_p']:.3f}) R2={reg['R2']:.3f}")

    # 汇总
    summ = pd.DataFrame([{
        "方法": m, **d["metrics"],
        "alpha年化": d["reg"]["alpha年化"], "alpha_p": d["reg"]["alpha_p"],
        "R2": d["reg"]["R2"],
    } for m, d in rows.items()])
    summ.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")

    # 导出最优（回撤<40% 中年化最高，否则全部）的持仓到 xlsx
    ok = summ[summ["最大回撤"] > -0.40]
    best_method = (ok.loc[ok["年化收益"].idxmax(), "方法"]
                   if not ok.empty else summ.loc[summ["年化收益"].idxmax(), "方法"])
    w_best = rows[best_method]["weights"]
    hold = w_best.stack()
    hold = hold[hold > 1e-8].reset_index()
    hold.columns = ["日期", "股票代码", "权重"]
    hold.to_csv(OUT / f"holdings_{best_method}.csv", index=False,
                encoding="utf-8-sig")
    hold.to_excel(OUT / f"holdings_{best_method}.xlsx", index=False)
    # 宽表（每月一行，值为权重）
    wide = hold.pivot(index="日期", columns="股票代码", values="权重").fillna(0)
    wide.to_excel(OUT / f"holdings_{best_method}_wide.xlsx")
    print(f"  最优方法 {best_method} 持仓已导出 holdings_{best_method}.xlsx")

    # 净值对比图
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for method, d in rows.items():
        nav = (1 + d["pnl"].fillna(0)).cumprod()
        axes[0].plot(nav.index, nav.values, label=method, lw=1.8)
        dd = nav / nav.cummax() - 1
        axes[1].plot(dd.index, dd.values, label=method, lw=1.5)
    axes[0].axhline(1, color="black", lw=0.7); axes[0].set_title("累计净值")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].axhline(-0.4, color="red", ls="--", lw=1, label="-40%")
    axes[1].set_title("回撤"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.suptitle("XGBoost v3 Top30 × 股票层风险权重（2020-2025）", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "riskweight_compare.png", dpi=150)
    plt.close(fig)

    # 挑达标
    ok = summ[summ["最大回撤"] > -0.40]
    if not ok.empty:
        best = ok.loc[ok["年化收益"].idxmax()]
        print(f"\n>>> 回撤<40% 且年化最高：{best['方法']}，年化 "
              f"{best['年化收益']*100:.2f}%，回撤 {best['最大回撤']*100:.2f}%，"
              f"alpha {best['alpha年化']*100:+.2f}%(p={best['alpha_p']:.3f})")
    else:
        lowdd = summ.loc[summ["最大回撤"].idxmax()]
        print(f"\n>>> 无配置回撤<40%。回撤最低：{lowdd['方法']}，"
              f"回撤 {lowdd['最大回撤']*100:.2f}%")
    print(f"结果：{OUT}")


if __name__ == "__main__":
    main()
