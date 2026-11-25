# -*- coding: utf-8 -*-
"""滚动 vs 固定因子策略对比。

读取两套策略的绩效与净值，输出对比表、净值曲线与分年度收益。

用法
----
    python baseline/compare_rolling_fixed.py
    # 可通过环境变量指定目录
    set ROLLING_DIR=rolling_pool & set FIXED_DIR=fixed_pool & python baseline/compare_rolling_fixed.py
"""
import os
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

from data_io import load_benchmark  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

RESULT_DIR = HERE.parent / "results"
OUT_DIR = RESULT_DIR / "rolling_vs_fixed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ROLLING_DIR = RESULT_DIR / os.environ.get("ROLLING_DIR", "rolling_pool")
FIXED_DIR = RESULT_DIR / os.environ.get("FIXED_DIR", "fixed_pool")


def annual_returns(nav: pd.Series) -> pd.Series:
    """由净值序列计算分年度收益。Annual returns from a NAV series."""
    return nav.groupby(nav.index.year).apply(lambda s: s.iloc[-1] / s.iloc[0] - 1)


def main():
    items = {"滚动因子": ROLLING_DIR, "固定因子": FIXED_DIR}
    rows, navs = [], {}
    for label, path in items.items():
        perf_file = path / "performance.csv"
        nav_file = path / "nav.csv"
        if not perf_file.exists():
            raise FileNotFoundError(f"缺少 {label} 绩效：{perf_file}")
        perf = pd.read_csv(perf_file, index_col=0).iloc[:, 0]
        rows.append({"方案": label, **perf.to_dict()})
        if nav_file.exists():
            navs[label] = pd.read_csv(nav_file, index_col=0, parse_dates=True).iloc[:, 0]

    cmp = pd.DataFrame(rows).set_index("方案")
    cmp.to_csv(OUT_DIR / "comparison.csv", encoding="utf-8-sig")

    if navs:
        ann = pd.DataFrame({k: annual_returns(v) for k, v in navs.items()})
        ann.index.name = "年份"
        ann.to_csv(OUT_DIR / "annual_returns.csv", encoding="utf-8-sig")

        fig, ax = plt.subplots(figsize=(10, 5))
        first = next(iter(navs.values()))
        bench = load_benchmark(str(first.index[0].date()),
                               str(first.index[-1].date()))["close"]
        bench = bench.reindex(first.index)
        bench_nav = (1 + bench.pct_change(fill_method=None).fillna(0)).cumprod()
        bench_nav = bench_nav / bench_nav.iloc[0]
        for label, nav in navs.items():
            nav = nav / nav.iloc[0]
            nav.plot(ax=ax, label=label, linewidth=1.6)
        bench_nav.plot(ax=ax, label="沪深300", color="gray", linewidth=1.2)
        ax.set(xlabel="日期", ylabel="净值（初始=1）",
               title="滚动因子 vs 固定因子（样本外 2020-2025）")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "nav_compare.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        (ann * 100).plot(kind="bar", ax=ax)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set(xlabel="年份", ylabel="年度收益（%）", title="分年度收益对比")
        ax.grid(axis="y", alpha=0.3, ls=":")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "annual_returns.png", dpi=150)
        plt.close(fig)

    show = cmp.copy()
    for col in ("累计收益", "年化收益", "年化波动", "最大回撤"):
        if col in show:
            show[col] = (show[col] * 100).round(2)
    print("滚动 vs 固定（样本外）：")
    print(show.round(3).to_string())
    print(f"\n结果目录：{OUT_DIR}")


if __name__ == "__main__":
    main()
