# -*- coding: utf-8 -*-
"""主策略（固定 4 因子）因子收益归因与时变暴露。

口径
----
- 因子：方案 A 固定入选的 4 个因子（a101_040 / a101_088 / alpha_097 /
  a158_cord20），经 MAD 去极值 + 行业中性 + Zscore。
- 持仓：results/main_strategy/holdings_main.xlsx（与 12.1 同源）。
- 因子收益：每月对可投池做截面回归 r_{t} = β0 + Σ β_k z_{k,t-1} + ε，
  β_k 即该月因子收益（Fama-MacBeth 式），β0 为截距。
- 因子贡献：contribution_k,t = 暴露_k,t-1 × β_k,t，
  暴露_k,t-1 = Σ_i w_{i,t-1} z_{i,k,t-1}（持仓加权标准化暴露）。
- 组合收益 r_p,t = Σ_i w_{i,t-1} r_{i,t}，与 calc_nav 的 pnl 口径一致。
- 残差 = r_p,t − Σ_k contribution_k,t − β0,t。

输出 results/attribution_main/
    attribution_monthly.csv   逐月：组合收益 / 各因子贡献 / 截距 / 残差
    attribution_summary.csv   各因子累计贡献、年平均贡献、平均暴露
    factor_contrib_bar.png    各因子累计贡献（含截距与残差）
    exposure_heatmap.png      年份 × 因子的平均标准化暴露

用法
----
    python baseline/attribution_main.py
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

import factor_pool_source as fps          # noqa: E402
from utils.data_standardize import preprocess  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

RESULT = HERE.parent / "results"
OUT = RESULT / "attribution_main"
OUT.mkdir(parents=True, exist_ok=True)
HOLD_PATH = RESULT / "main_strategy" / "holdings_main.xlsx"

NAMES = ["a101_040", "a101_088", "alpha_097", "a158_cord20"]
LABEL = {
    "a101_040": "a101_040",
    "a101_088": "a101_088",
    "alpha_097": "alpha_097",
    "a158_cord20": "a158_cord20",
}


def load_weights(dates, codes):
    df = pd.read_excel(HOLD_PATH)
    df.columns = [str(c).strip() for c in df.columns]
    df["日期"] = pd.to_datetime(df["日期"])
    w = df.pivot_table(index="日期", columns="股票代码", values="权重",
                       aggfunc="sum")
    w = w.reindex(index=dates, columns=codes).fillna(0.0)
    return w


def main():
    print("[1/3] 载入因子面板与持仓 ...")
    monthly, factors, dates = fps.load_pool(mask_valid=True)
    returns = monthly["returns"].reindex(index=dates)
    valid = monthly["valid"].reindex(index=dates)
    z = {n: preprocess(factors[n]).where(valid) for n in NAMES}
    w = load_weights(dates, returns.columns)

    rows = []
    hold_dates = w.index[w.sum(axis=1) > 0]
    start = hold_dates.min()
    idx = dates[(dates >= start) & (dates <= hold_dates.max())]
    print(f"  区间 {idx.min().date()} ~ {idx.max().date()}，共 {len(idx)} 期")

    for i in range(1, len(idx)):
        t_prev, t = idx[i - 1], idx[i]
        r = returns.loc[t]
        # 截面回归：r_t = β0 + Σ β_k z_{k,t-1} + ε
        X = pd.DataFrame({n: z[n].loc[t_prev] for n in NAMES})
        data = X.join(r.rename("r"), how="inner").dropna()
        if len(data) < 50:
            continue
        A = np.column_stack([np.ones(len(data)),
                             data[NAMES].to_numpy(dtype=float)])
        y = data["r"].to_numpy(dtype=float)
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        b0, bk = float(beta[0]), beta[1:]

        w_all = w.loc[t_prev].reindex(data.index).fillna(0.0)
        coverage = float(w_all.sum())
        if coverage < 0.80:
            continue
        w_prev = w_all / coverage
        x = data[NAMES].to_numpy(dtype=float)
        expo = w_prev.to_numpy(dtype=float) @ x
        contrib = expo * bk
        r_p = float(w_prev.to_numpy(dtype=float) @ y)
        resid = r_p - b0 - float(contrib.sum())

        row = {"日期": t, "组合收益": r_p, "截距": b0, "残差": resid,
               "合计贡献": float(contrib.sum())}
        for n, c, e in zip(NAMES, contrib, expo):
            row[f"{n}_贡献"] = float(c)
            row[f"{n}_暴露"] = float(e)
        rows.append(row)

    att = pd.DataFrame(rows).set_index("日期")
    att.to_csv(OUT / "attribution_monthly.csv", encoding="utf-8-sig")

    n_months = len(att)
    years = n_months / 12.0
    summ = []
    for n in NAMES:
        c = att[f"{n}_贡献"]
        summ.append({"因子": n, "累计贡献": c.sum(),
                     "年化贡献": c.sum() / years,
                     "平均暴露": att[f"{n}_暴露"].mean(),
                     "t值": c.mean() / c.std(ddof=1) * np.sqrt(n_months)})
    summ = pd.DataFrame(summ)
    extra = pd.DataFrame([
        {"因子": "截距", "累计贡献": att["截距"].sum(),
         "年化贡献": att["截距"].sum() / years, "平均暴露": np.nan,
         "t值": att["截距"].mean() / att["截距"].std(ddof=1) * np.sqrt(n_months)},
        {"因子": "残差", "累计贡献": att["残差"].sum(),
         "年化贡献": att["残差"].sum() / years, "平均暴露": np.nan,
         "t值": att["残差"].mean() / att["残差"].std(ddof=1) * np.sqrt(n_months)},
    ])
    allsum = pd.concat([summ, extra], ignore_index=True)
    allsum.to_csv(OUT / "attribution_summary.csv", index=False,
                  encoding="utf-8-sig")
    print(allsum.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---- 图 1：各因子累计贡献 ----
    order = NAMES + ["截距", "残差"]
    vals = [att[f"{n}_贡献"].sum() for n in NAMES] + \
           [att["截距"].sum(), att["残差"].sum()]
    colors = ["#4c78a8"] * len(NAMES) + ["#999999", "#999999"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ypos = np.arange(len(order))[::-1]
    ax.barh(ypos, np.array(vals) * 100, color=colors, height=0.62)
    ax.set_yticks(ypos)
    ax.set_yticklabels(order)
    for yy, v in zip(ypos, vals):
        ax.text(v * 100 + (1.2 if v >= 0 else -1.2), yy, f"{v*100:+.2f}%",
                va="center", ha="left" if v >= 0 else "right", fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    lo, hi = min(vals) * 100, max(vals) * 100
    ax.set_xlim(lo - abs(lo) * 0.30 - 4, hi + 14)
    ax.set_xlabel("2020–2025 累计贡献（%）")
    ax.set_title("收益归因：各因子累计贡献（主策略 4 因子 + IVar Top100，样本外）")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "factor_contrib_bar.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    en_order = NAMES + ["Intercept", "Residual"]
    ax.barh(ypos, np.array(vals) * 100, color=colors, height=0.62)
    ax.set_yticks(ypos); ax.set_yticklabels(en_order)
    for yy, v in zip(ypos, vals):
        ax.text(v * 100 + (1.2 if v >= 0 else -1.2), yy, f"{v*100:+.2f}%",
                va="center", ha="left" if v >= 0 else "right", fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlim(lo - abs(lo) * 0.30 - 4, hi + 14)
    ax.set_xlabel("Cumulative Contribution, 2020–2025 (%)")
    ax.set_title("Return Attribution: Main Strategy (4 Factors + IVar Top 100, Out of Sample)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "factor_contrib_bar_en.png", dpi=150)
    plt.close(fig)

    # ---- 图 2：年份 × 因子 平均暴露 ----
    yr = att.index.year
    expo_tab = pd.DataFrame({n: att[f"{n}_暴露"] for n in NAMES})
    expo_tab["年"] = yr
    heat = expo_tab.groupby("年")[NAMES].mean().T
    fig, ax = plt.subplots(figsize=(9, 3.6))
    vmax = float(np.nanmax(np.abs(heat.values))) or 1.0
    im = ax.imshow(heat.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   aspect="auto")
    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels([f"{y}" for y in heat.columns])
    ax.set_yticks(range(len(NAMES)))
    ax.set_yticklabels(NAMES)
    for i in range(len(NAMES)):
        for j in range(len(heat.columns)):
            v = heat.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=9,
                    color="white" if abs(v) > vmax * 0.55 else "black")
    ax.set_title("组合因子暴露（主策略 4 因子，标准化单位，年均值）")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="暴露")
    fig.tight_layout()
    fig.savefig(OUT / "exposure_heatmap.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.6))
    im = ax.imshow(heat.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels([f"{y}" for y in heat.columns])
    ax.set_yticks(range(len(NAMES))); ax.set_yticklabels(NAMES)
    ax.set_xlabel("Year"); ax.set_ylabel("Factor")
    for i in range(len(NAMES)):
        for j in range(len(heat.columns)):
            v = heat.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(v) > vmax * 0.55 else "black")
    ax.set_title("Portfolio Factor Exposures: Annual Means (Main Strategy, Standardized Units)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Exposure")
    fig.tight_layout()
    fig.savefig(OUT / "exposure_heatmap_en.png", dpi=150)
    plt.close(fig)

    print(f"\n已输出 {OUT}")


if __name__ == "__main__":
    main()
