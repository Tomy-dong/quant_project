"""市场中性分支：多头组合 + 股指期货空头。

Market-neutral branch: the long-only GMVP Top10 portfolio plus an index-futures
short leg that neutralises the market beta.

由于 Baostock 不提供期货行情，空头腿用指数收益近似（不考虑基差之外的
结算细节），基差以年化成本的形式单独做敏感性分析。β 用 24 个月滚动
回归估计，且只使用 t-1 期及之前的信息，避免前视偏差。

输出：
    results/market_neutral/neutral_performance.csv
    results/market_neutral/neutral_annual.csv
    results/market_neutral/basis_sensitivity.csv
    results/market_neutral/monthly_neutral.csv
    results/market_neutral/nav_neutral.png
    results/market_neutral/rolling_beta.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "baseline")]

from data_io import load_benchmark
from utils.utils import calc_nav

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = ROOT / "results" / "market_neutral"
HOLDINGS = ROOT / "results" / "optimizer2" / "top10" / "GMVP" / "holdings.csv"
RET_PATH = ROOT / "results" / "optimizer2" / "monthly_returns.csv.gz"

COMSN = 0.002          # 单边费率，与优化器二保持一致
WINDOW = 24            # β 估计窗口（月）
MIN_PERIODS = 12       # 最少观测数
BASIS_MAIN = 0.04      # 主表假设的基差成本（年化）
BASIS_GRID = (0.0, 0.02, 0.04, 0.06, 0.08)


def load_nav():
    """读取 GMVP Top10 持仓并复算月度净值。Rebuild the long-only NAV."""
    rets = pd.read_csv(RET_PATH, index_col=0, parse_dates=True)
    hold = pd.read_csv(HOLDINGS, dtype={"股票代码": str}, parse_dates=["调仓日"])
    w = (hold.pivot_table(index="调仓日", columns="股票代码", values="权重",
                          aggfunc="sum")
             .fillna(0.0)
             .reindex(columns=rets.columns, fill_value=0.0))
    r = rets.loc[w.index].fillna(0.0)
    nav = calc_nav(r, w, comsn=COMSN)["nav"]
    return nav, w


def monthly_returns(nav, start, end):
    """组合与两条基准的月度收益。Monthly returns of the portfolio and benchmarks."""
    rp = nav.pct_change()
    rp.iloc[0] = nav.iloc[0] - 1
    rb = {}
    for label, idx in (("沪深300", "csi300"), ("中证500", "csi500")):
        close = load_benchmark(start, end, index=idx)["close"]
        close = close.reindex(nav.index)
        rb[label] = close.pct_change().fillna(0.0)
    return rp, rb


def rolling_beta(rp, rb):
    """24 个月滚动 β，仅用 t-1 期信息做下一期对冲。Trailing rolling beta."""
    cov = rp.rolling(WINDOW, min_periods=MIN_PERIODS).cov(rb)
    var = rb.rolling(WINDOW, min_periods=MIN_PERIODS).var()
    return (cov / var).shift(1)


def metrics(r):
    """由月度收益序列计算累计、年化、波动、夏普、回撤。Performance stats."""
    nav = (1 + r.fillna(0)).cumprod()
    total = nav.iloc[-1] - 1
    ann = (1 + total) ** (12 / len(r)) - 1
    vol = r.std(ddof=1) * np.sqrt(12)
    return {
        "累计收益": total,
        "年化收益": ann,
        "年化波动": vol,
        "夏普": ann / vol if vol > 0 else np.nan,
        "最大回撤": (nav / nav.cummax() - 1).min(),
    }


def plot_nav(series, path):
    """中性组合净值对比。NAV comparison across hedge choices."""
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for label, nav in series.items():
        ax.plot(nav.index, nav.values, lw=1.5, label=f"{label}（{nav.iloc[-1] - 1:+.1%}）")
    ax.axhline(1.0, color="grey", lw=0.8, ls="--")
    ax.set_title("市场中性分支：多头 vs 期货对冲后的净值")
    ax.set_ylabel("净值")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_beta(betas, path):
    """滚动 β。Rolling beta to both benchmarks."""
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for label, b in betas.items():
        ax.plot(b.index, b.values, lw=1.5, label=label)
    ax.axhline(1.0, color="grey", lw=0.8, ls="--")
    ax.axhline(0.0, color="#c00000", lw=0.8, ls=":")
    ax.set_title(f"多头组合对基准的 {WINDOW} 个月滚动 β")
    ax.set_ylabel("β")
    ax.set_ylim(-0.5, 2.0)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nav, _ = load_nav()
    rp, rb = monthly_returns(nav, str(nav.index[0].date()), str(nav.index[-1].date()))

    betas = {label: rolling_beta(rp, series) for label, series in rb.items()}
    valid = betas["沪深300"].notna() & betas["中证500"].notna()
    rp_v = rp[valid]

    legs = {"多头（不对冲）": rp_v}
    for label in rb:
        for basis, tag in ((0.0, "0基差"), (BASIS_MAIN, f"{BASIS_MAIN:.0%}基差")):
            hedge = betas[label][valid] * rb[label][valid]
            legs[f"{label}对冲·{tag}"] = rp_v - hedge - basis / 12

    table = pd.DataFrame({k: metrics(v) for k, v in legs.items()}).T
    table.index.name = "组合"
    table.to_csv(OUT_DIR / "neutral_performance.csv", encoding="utf-8-sig")

    annual = pd.DataFrame({
        k: (1 + v).groupby(v.index.year).prod() - 1 for k, v in legs.items()})
    annual.to_csv(OUT_DIR / "neutral_annual.csv", encoding="utf-8-sig")

    basis_rows = []
    for label in rb:
        for basis in BASIS_GRID:
            r = rp_v - betas[label][valid] * rb[label][valid] - basis / 12
            m = metrics(r)
            m.update({"对冲标的": label, "年化基差": basis})
            basis_rows.append(m)
    basis_tbl = pd.DataFrame(basis_rows).set_index(["对冲标的", "年化基差"])
    basis_tbl.to_csv(OUT_DIR / "basis_sensitivity.csv", encoding="utf-8-sig")

    monthly = pd.DataFrame({
        "组合收益": rp_v,
        "沪深300": rb["沪深300"][valid],
        "中证500": rb["中证500"][valid],
        "beta_300": betas["沪深300"][valid],
        "beta_500": betas["中证500"][valid],
        "净收益_IF_4%": legs[f"沪深300对冲·{BASIS_MAIN:.0%}基差"],
        "净收益_IC_4%": legs[f"中证500对冲·{BASIS_MAIN:.0%}基差"],
    })
    monthly.to_csv(OUT_DIR / "monthly_neutral.csv", encoding="utf-8-sig")

    navs = {k: (1 + v).cumprod() for k, v in legs.items()}
    plot_nav(navs, OUT_DIR / "nav_neutral.png")
    plot_beta(betas, OUT_DIR / "rolling_beta.png")

    print(f"中性区间：{rp_v.index[0].date()} ~ {rp_v.index[-1].date()}（{len(rp_v)} 个月）")
    print((table * 100).round(2).to_string())
    print("\n分年度收益（%）：")
    print((annual * 100).round(2).to_string())
    print(f"\n基差敏感性（夏普）：")
    print(basis_tbl["夏普"].unstack().round(3).to_string())
    print(f"\n结果已保存到：{OUT_DIR}")


if __name__ == "__main__":
    main()
