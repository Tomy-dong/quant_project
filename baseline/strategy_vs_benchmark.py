# -*- coding: utf-8 -*-
"""新因子池策略 vs 沪深300 / 中证500 基准对比 + 三因子回归。

回答两个问题：
1. 扩容后的策略（固定因子、优化器二 Top50）在 2020-2025 样本外有没有跑赢
   沪深300 / 中证500？（同区间、同口径，含单边千二费率）
2. 扣除 Fama-French 三因子（MKT / SMB / HML）暴露后，策略是否还剩下
   显著的正 alpha？

三因子直接复用 ``results/ff3/ff3_factors.csv``（已由 ff3_regression.py 构造好，
2x3 规模/价值双重排序，2020-2025），不重新构造以节省内存。

输出 ``results/benchmark_compare/``：
    summary.csv            策略与基准的绩效 + 超额 + 信息比率
    nav_compare.png        净值曲线（策略 + 沪深300 + 中证500）
    annual_returns.csv     分年度收益
    annual_returns.png     分年度收益柱状图
    excess_vs_csi300.png   相对沪深300的累计超额净值
    ff3_regression.csv     各策略三因子回归（alpha / 暴露 / t / R²）
    ff3_alpha.png          年化 alpha 与 95% 置信区间
"""
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import price_volume_baseline as pv          # noqa: E402
import ff3_regression as ff3                # noqa: E402
from data_io import load_benchmark          # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

RESULT = HERE.parent / "results"
OUT = RESULT / "benchmark_compare"
OUT.mkdir(parents=True, exist_ok=True)
FF3_DIR = RESULT / "ff3"

RF_ANNUAL = 0.02  # 与 ff3_regression 一致 / consistent with FF3

# 要对比的策略：(展示名, 类型, 路径)
# 类型 nav    -> 从 nav.csv 读月度净值再算收益
# 类型 hold   -> 从 holdings.csv 用 ff3.returns_from_holdings 重建
STRATEGIES = [
    ("固定因子 Top30", "nav",  RESULT / "fixed_pool" / "nav.csv"),
    ("滚动因子 Top30", "nav",  RESULT / "rolling_pool" / "nav.csv"),
    ("优化器二 Top50 IVar", "hold",
     RESULT / "optimizer2_pool" / "top50" / "IVar" / "holdings.csv"),
    ("优化器二 Top50 EW", "hold",
     RESULT / "optimizer2_pool" / "top50" / "EW" / "holdings.csv"),
]
BENCHMARKS = [("沪深300", "csi300"), ("中证500", "csi500")]


# ---------- 工具 / helpers ----------
def perf_metrics(monthly_ret: pd.Series) -> dict:
    """由月度净收益算累计/年化/波动/夏普/回撤。Metrics from monthly net returns."""
    s = monthly_ret.dropna()
    if len(s) < 2:
        return {}
    nav = (1 + s).cumprod()
    total = float(nav.iloc[-1] - 1)
    years = len(s) / 12.0
    ann = float((1 + total) ** (1 / years) - 1)
    vol = float(s.std(ddof=1) * np.sqrt(12))
    # 与项目既有 performance.csv 口径一致：夏普 = 年化/波动，不扣无风险利率
    sharpe = float(ann / vol) if vol > 0 else np.nan
    dd = float((nav / nav.cummax() - 1).min())
    return {"累计收益": total, "年化收益": ann, "年化波动": vol,
            "夏普比率": sharpe, "最大回撤": dd}


def annual_returns(monthly_ret: pd.Series) -> pd.Series:
    nav = (1 + monthly_ret.fillna(0)).cumprod()
    return nav.groupby(nav.index.year).apply(lambda s: s.iloc[-1] / s.iloc[0] - 1)


def plot_alpha(summary, path):
    """年化 alpha 与 95% 置信区间图（本地保存，不覆盖旧 ff3 图）。"""
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
        xpos = hi[i] + 0.4 if v >= 0 else lo[i] - 0.4
        ax.text(xpos, i, f"{v:+.1f}%", va="center",
                ha="left" if v >= 0 else "right", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(df["策略"])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(xlabel="年化 alpha（%）",
           title="三因子回归：年化 alpha 与 95% 置信区间")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------- 主流程 ----------
def main():
    print("策略 vs 基准 + 三因子回归 | 样本外 2020-2025")
    print("=" * 72)

    # 1. 月度收益矩阵（仅用于从 holdings 重建优化器二收益）
    print("\n[1/5] 载入市场数据构建月度收益矩阵...")
    daily = pv.load_market_data(pv.MARKET, pv.START, pv.END)
    monthly = pv.build_monthly(daily)
    returns = monthly["returns"]
    del daily, monthly
    print(f"  月度收益矩阵：{returns.shape[0]} 月 × {returns.shape[1]} 股票")

    # 2. 三因子（复用已构造结果）
    print("[2/5] 载入已构造的三因子...")
    factors = pd.read_csv(FF3_DIR / "ff3_factors.csv",
                          index_col=0, parse_dates=True)
    factors.index.name = "日期"
    print(f"  三因子区间：{factors.index.min().date()} ~ "
          f"{factors.index.max().date()}，共 {len(factors)} 个月")

    # 3. 各策略月度收益
    print("[3/5] 重建各策略月度收益...")
    series = {}
    for label, kind, path in STRATEGIES:
        if not Path(path).exists():
            print(f"  [跳过] {label}: 缺少 {path}")
            continue
        if kind == "nav":
            nav = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
            nav.index = pd.to_datetime(nav.index)
            s = nav.pct_change(fill_method=None)  # 首点 NaN / first point NaN
        else:
            s = ff3.returns_from_holdings(Path(path), returns)
            # 首点为建仓月(0)，置 NaN 与 nav 口径对齐
            s = s.copy()
            if len(s):
                s.iloc[0] = np.nan
        s.index.name = "日期"
        series[label] = s
        print(f"  {label:<18s} {s.dropna().index.min().date()} ~ "
              f"{s.dropna().index.max().date()}，{s.dropna().shape[0]} 个月")

    dates = pd.DatetimeIndex(factors.index)
    for label, idx in BENCHMARKS:
        close = load_benchmark(str(dates.min().date()),
                               str(dates.max().date()), index=idx)["close"]
        close = close.reindex(dates)
        series[label] = close.pct_change(fill_method=None)
        print(f"  {label:<18s} {series[label].dropna().index.min().date()} ~ "
              f"{series[label].dropna().index.max().date()}，"
              f"{series[label].dropna().shape[0]} 个月")

    ret_df = pd.DataFrame(series).sort_index()

    # 4. 基准对比
    print("\n[4/5] 基准对比...")
    rows = []
    for label in ret_df.columns:
        m = perf_metrics(ret_df[label])
        m["方案"] = label
        rows.append(m)
    summary = pd.DataFrame(rows).set_index("方案")

    bench_ann = {b: summary.loc[b, "年化收益"] for b, _ in BENCHMARKS}
    for b_label, _ in BENCHMARKS:
        col_ex = f"超额年化({b_label})"
        col_ir = f"信息比率({b_label})"
        for label in ret_df.columns:
            if label == b_label:
                summary.loc[label, col_ex] = 0.0
                summary.loc[label, col_ir] = np.nan
                continue
            diff = (ret_df[label] - ret_df[b_label]).dropna()
            summary.loc[label, col_ex] = summary.loc[label, "年化收益"] - bench_ann[b_label]
            summary.loc[label, col_ir] = (
                diff.mean() / diff.std(ddof=1) * np.sqrt(12)
                if diff.std(ddof=1) > 0 else np.nan)

    summary.to_csv(OUT / "summary.csv", encoding="utf-8-sig")
    show = summary.copy()
    for c in ["累计收益", "年化收益", "年化波动", "最大回撤",
              "超额年化(沪深300)", "信息比率(沪深300)",
              "超额年化(中证500)", "信息比率(中证500)"]:
        if c in show:
            show[c] = (show[c] * 100).round(2)
    for c in ["夏普比率"]:
        show[c] = show[c].round(3)
    print("\n绩效与基准对比（%）：")
    print(show.to_string())

    # 净值曲线
    navs = (1 + ret_df.fillna(0)).cumprod()
    navs = navs / navs.iloc[0]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = {"沪深300": "#888888", "中证500": "#bbbbbb"}
    for label in ret_df.columns:
        lw = 1.3 if label in dict(BENCHMARKS) else 1.8
        ax.plot(navs.index, navs[label], label=label, linewidth=lw,
                color=colors.get(label))
    ax.set(xlabel="日期", ylabel="净值（初始=1）",
           title="新因子池策略 vs 沪深300 / 中证500（样本外 2020-2025，含单边千二费率）")
    ax.legend(ncol=3, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "nav_compare.png", dpi=150)
    plt.close(fig)

    # 分年度收益
    ann = pd.DataFrame({lab: annual_returns(ret_df[lab]) for lab in ret_df.columns})
    ann.index.name = "年份"
    ann.to_csv(OUT / "annual_returns.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    (ann * 100).plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(xlabel="年份", ylabel="年度收益（%）", title="分年度收益对比")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(axis="y", alpha=0.3, ls=":")
    fig.tight_layout()
    fig.savefig(OUT / "annual_returns.png", dpi=150)
    plt.close(fig)

    # 相对沪深300累计超额
    fig, ax = plt.subplots(figsize=(11, 5))
    base = ret_df["沪深300"]
    for label in ret_df.columns:
        if label == "沪深300":
            continue
        excess = (1 + (ret_df[label] - base).fillna(0)).cumprod()
        ax.plot(excess.index, excess, label=label, linewidth=1.6)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set(xlabel="日期", ylabel="累计超额净值（初始=1）",
           title="相对沪深300的累计超额收益")
    ax.legend(ncol=3, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "excess_vs_csi300.png", dpi=150)
    plt.close(fig)

    # 5. 三因子回归
    print("\n[5/5] 三因子时序回归（复用已构造 MKT/SMB/HML）...")
    reg_rows = [ff3.regress_one(lab, ret_df[lab], factors)
                for lab in ret_df.columns]
    reg = pd.DataFrame(reg_rows)
    reg.to_csv(OUT / "ff3_regression.csv", index=False, encoding="utf-8-sig")

    show = reg.copy()
    for c in ["alpha月", "alpha年化", "alpha_95%下限", "alpha_95%上限", "年化波动"]:
        show[c] = (show[c] * 100).round(2)
    for c in ["alpha_t", "alpha_p", "MKT", "MKT_t", "SMB", "SMB_t",
              "HML", "HML_t", "R2", "调整R2"]:
        show[c] = show[c].round(3)
    print("\n三因子回归结果（%）：")
    print(show.to_string(index=False))

    # alpha 图（本地绘制，不覆盖 results/ff3/ 下的旧图）
    plot_alpha(reg, OUT / "ff3_alpha.png")

    # ---------- 结论 ----------
    print("\n" + "=" * 72)
    print("结论摘要：")
    for b_label, _ in BENCHMARKS:
        print(f"\n  相对 {b_label}（年化 {bench_ann[b_label]*100:.2f}%）：")
        for label in ret_df.columns:
            if label == b_label:
                continue
            ex = summary.loc[label, f"超额年化({b_label})"]
            ir = summary.loc[label, f"信息比率({b_label})"]
            beat = "跑赢" if ex > 0 else "跑输"
            print(f"    {label:<18s} {beat}  超额 {ex*100:+.2f}%  IR={ir:.2f}")
    print("\n  三因子 alpha（年化）：")
    for _, r in reg.iterrows():
        sig = "显著" if r["alpha_p"] < 0.1 else "不显著"
        print(f"    {r['策略']:<18s} alpha={r['alpha年化']*100:+.2f}%  "
              f"t={r['alpha_t']:.2f}  p={r['alpha_p']:.3f}  "
              f"R²={r['R2']:.2f}  ({sig})")

    print(f"\n结果目录：{OUT}")


if __name__ == "__main__":
    main()
