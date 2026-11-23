# -*- coding: utf-8 -*-
"""38 个量价因子的多空收益诊断。

对每个因子按当期预处理后的分数分 5 组，计算下一期 Top 组减 Bottom 组的
等权收益，再按 IC 方向校准（正数表示因子在其 IC 方向上有效）。

输出 results/baseline/longshort.csv，并与 factor_summary.csv 的 IC/ICIR 合并展示。

用法
----
    python baseline/factor_longshort.py
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from price_volume_baseline import (MARKET, START, END, OUTPUT_DIR,
                                   build_monthly, compute_price_volume_factors,
                                   filter_pool, load_market_data)
from utils.data_standardize import preprocess

N_GROUP = 5


def long_short_series(score, returns):
    """逐月分 N_GROUP 组，返回 Top - Bottom 的等权收益序列。"""
    out = []
    idx = []
    for i in range(len(score.index) - 1):
        df = pd.DataFrame({"s": score.iloc[i], "r": returns.iloc[i + 1]}).dropna()
        if len(df) < N_GROUP * 20:
            continue
        try:
            groups = pd.qcut(df["s"], N_GROUP, labels=False, duplicates="drop")
        except ValueError:
            continue
        means = df.groupby(groups)["r"].mean()
        if len(means) < N_GROUP:
            continue
        out.append(means.iloc[-1] - means.iloc[0])
        idx.append(score.index[i + 1])
    return pd.Series(out, index=idx, dtype=float)


def stats(ls):
    """年化收益、夏普、t 值、月胜率、最大回撤。"""
    ls = ls.dropna()
    n = len(ls)
    nav = (1 + ls).cumprod()
    ann = nav.iloc[-1] ** (12 / n) - 1
    vol = ls.std(ddof=1) * np.sqrt(12)
    return {
        "多空年化": ann,
        "多空夏普": ls.mean() / ls.std(ddof=1) * np.sqrt(12),
        "t值": ls.mean() / ls.std(ddof=1) * np.sqrt(n),
        "月胜率": (ls > 0).mean(),
        "最大回撤": (nav / nav.cummax() - 1).min(),
        "样本月数": n,
    }


def main():
    print(f"市场：{MARKET} | 多空诊断：{N_GROUP} 分组")
    print("=" * 60)

    print("\n[1/3] 载入数据与月频面板...")
    monthly = build_monthly(load_market_data(MARKET, START, END))
    valid = filter_pool(monthly)

    print("[2/3] 计算量价因子与截面预处理...")
    factors = compute_price_volume_factors(monthly, valid)
    scores = {name: preprocess(fac) for name, fac in factors.items()}

    print("[3/3] 逐因子多空回测...")
    ic = pd.read_csv(OUTPUT_DIR / "factor_summary.csv")
    ic = ic[ic["指标"] == "RankIC"].set_index("因子")

    rows = []
    for name, score in scores.items():
        ls = long_short_series(score, monthly["returns"])
        if len(ls) < 24:
            continue
        sign = np.sign(ic.loc[name, "平均IC"]) or 1.0
        row = {"因子": name, "平均IC": ic.loc[name, "平均IC"],
               "ICIR": ic.loc[name, "ICIR"], "方向": int(sign)}
        row.update(stats(sign * ls))
        rows.append(row)

    table = pd.DataFrame(rows).sort_values("ICIR", key=abs, ascending=False)
    table.to_csv(OUTPUT_DIR / "longshort.csv", index=False, encoding="utf-8-sig")

    print("\n多空诊断（按 |ICIR| 排序，收益已按 IC 方向校准）：")
    show = table.copy()
    for col in ["多空年化", "多空夏普", "月胜率", "最大回撤"]:
        show[col] = (show[col] * 100).round(1).astype(str) + "%"
    show["t值"] = show["t值"].round(2)
    print(show[["因子", "平均IC", "ICIR", "多空年化", "多空夏普", "t值", "月胜率", "最大回撤"]]
          .to_string(index=False))

    good = table[(table["多空年化"] > 0) & (table["t值"] > 2)]
    print(f"\n合计 {len(table)} 个因子；多空年化为正且 t>2 的有 {len(good)} 个。")
    print(f"结果已保存到：{OUTPUT_DIR / 'longshort.csv'}")


if __name__ == "__main__":
    main()
