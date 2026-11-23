"""优化器一：可交易池内的因子筛选与加权实验 / Optimizer 1: factor selection.

固定前提：容量过滤开 + 口径统一（可交易池）。在此之上比较不同筛选规则：
    S0 当前规则：|ICIR|>=0.2，按 |ICIR| 取前 20，等权（对照组，应复现 -28.62%）
    S1 S0 + 多空门控（多空年化>0 且 t>=2）
    S2 S1 + 多头门控（多头超额年化>0 且 t>0）
    S3 S2 的因子集合，按 |ICIR| 加权
    S4 纯多头导向：多头超额 t>=1 且 |ICIR|>=0.2（不要求多空显著）
    S5 S2 + |ICIR|>=0.3
    S6 S2 + |ICIR|>=0.4
最后对夏普最高的配置扫描持仓数 TOP_N ∈ {10, 20, 50}。

多头超额 = 方向校准后 top 组等权收益 - 当期全池等权收益（逐月）。
IC / 多空诊断直接复用 results/baseline/ 下当前配置的 factor_summary.csv 与
longshort.csv，避免重复运行 factor_test。

注意：本实验为全样本内筛选，存在过拟合风险，结论需经 rolling OOS 验证。

Usage:
    python baseline/optimizer_factor_selection.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import price_volume_baseline as pv  # noqa: E402
from utils.data_standardize import preprocess  # noqa: E402

OUT = HERE.parent / "results" / "optimizer"
OUT.mkdir(parents=True, exist_ok=True)
BASE = HERE.parent / "results" / "baseline"

N_GROUP = 5


def top_excess_series(score, returns, sign):
    """方向校准后 top 组等权收益 - 全池等权收益（逐月）。
    Direction-calibrated top-group equal-weight return minus the universe mean.
    """
    s = sign * score
    out, idx = [], []
    for i in range(len(s.index) - 1):
        df = pd.DataFrame({"s": s.iloc[i], "r": returns.iloc[i + 1]}).dropna()
        if len(df) < N_GROUP * 20:
            continue
        try:
            g = pd.qcut(df["s"], N_GROUP, labels=False, duplicates="drop")
        except ValueError:
            continue
        means = df.groupby(g)["r"].mean()
        if len(means) < N_GROUP:
            continue
        out.append(means.iloc[-1] - df["r"].mean())
        idx.append(s.index[i + 1])
    return pd.Series(out, index=idx, dtype=float)


def ann_stats(r):
    """年化收益与 t 值。Annualized return and t-stat of a monthly series."""
    r = r.dropna()
    n = len(r)
    if n < 24 or r.std(ddof=1) == 0:
        return np.nan, np.nan
    nav = (1 + r).cumprod()
    ann = nav.iloc[-1] ** (12 / n) - 1
    t = r.mean() / r.std(ddof=1) * np.sqrt(n)
    return ann, t


def build_diagnostics(scores, returns):
    """合并 IC / 多空 / 多头超额三张诊断。Merge IC, long-short and top-excess."""
    ic = pd.read_csv(BASE / "factor_summary.csv")
    ic = ic[ic["指标"] == "RankIC"].set_index("因子")[["平均IC", "ICIR"]]
    ls = pd.read_csv(BASE / "longshort.csv").set_index("因子")[["方向", "多空年化", "t值"]]
    diag = ic.join(ls, how="inner")
    diag = diag.rename(columns={"t值": "多空t"})

    top_ann, top_t = {}, {}
    for name, score in scores.items():
        if name not in diag.index:
            continue
        sign = int(diag.loc[name, "方向"]) or 1
        ser = top_excess_series(score, returns, sign)
        a, t = ann_stats(ser)
        top_ann[name], top_t[name] = a, t
    diag["多头年化"] = pd.Series(top_ann)
    diag["多头t"] = pd.Series(top_t)
    return diag.dropna(subset=["ICIR", "多空t"])


def pick(diag, icir_th=0.2, ls_gate=False, top_gate=False, top_t=0.0, max_n=20):
    """按门控筛选因子，按 |ICIR| 降序截断。Gate factors then truncate by |ICIR|."""
    d = diag[diag["ICIR"].abs() >= icir_th]
    if ls_gate:
        d = d[(d["多空年化"] > 0) & (d["多空t"] >= 2)]
    if top_gate:
        d = d[(d["多头年化"] > 0) & (d["多头t"] > top_t)]
    d = d.sort_values("ICIR", key=abs, ascending=False).head(max_n)
    return d


def composite(scores, factors_df, weight_col=None):
    """方向校准后等权或按 weight_col 列的绝对值加权合成。
    Blend signed scores equally or weighted by |weight_col| (e.g. ICIR, 多头t).
    """
    total = None
    wsum = 0.0
    for name, row in factors_df.iterrows():
        sign = np.sign(row["平均IC"]) or 1.0
        w = sign * (abs(row[weight_col]) if weight_col else 1.0)
        term = w * scores[name]
        total = term if total is None else total + term
        wsum += abs(w)
    return total / wsum if wsum > 0 else total


def evaluate(tag, factors_df, scores, returns, dates, weight_col=None, top_n=pv.TOP_N):
    """合成并回测一个配置。Blend and backtest one configuration."""
    comp = composite(scores, factors_df, weight_col)
    out_dir = OUT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    _, perf = pv.backtest_portfolio(comp, returns, dates, top_n=top_n,
                                    output_dir=out_dir, title=tag)
    perf["入选数"] = len(factors_df)
    perf["入选因子"] = "、".join(factors_df.index)
    print(f">>> {tag}: 入选 {len(factors_df)} 个 | 累计 {perf['累计收益']*100:.2f}% | "
          f"夏普 {perf['夏普比率']:.3f}")
    return perf


def main():
    print(f"优化器一：可交易池因子筛选 | {pv.START} ~ {pv.END}")
    print("=" * 70)

    print("\n[1/4] 载入数据、计算因子与预处理（约数分钟）...")
    daily = pv.load_market_data(pv.MARKET, pv.START, pv.END)
    monthly = pv.build_monthly(daily)
    dates = monthly["dates"]
    returns = monthly["returns"]
    valid = pv.filter_pool(monthly)
    print(f"  平均可交易股票数：{valid.sum(axis=1).mean():.0f}")
    factors = pv.compute_price_volume_factors(monthly, valid)
    scores = {name: preprocess(fac) for name, fac in factors.items()}

    print("\n[2/4] 构建因子诊断（IC + 多空 + 多头超额）...")
    diag = build_diagnostics(scores, returns)
    diag.to_csv(OUT / "factor_diagnostics.csv", encoding="utf-8-sig")
    show = diag.copy()
    for c in ["平均IC", "ICIR", "多空年化", "多空t", "多头年化", "多头t"]:
        show[c] = show[c].round(3)
    print(show.sort_values("ICIR", key=abs, ascending=False).to_string())

    print("\n[3/4] 运行筛选配置对照...")
    rows = {}
    rows["S0_当前"] = evaluate("S0_current", pick(diag), scores, returns, dates)
    rows["S1_多空门控"] = evaluate("S1_ls", pick(diag, ls_gate=True), scores, returns, dates)
    rows["S2_多空+多头"] = evaluate("S2_ls_top", pick(diag, ls_gate=True, top_gate=True),
                                    scores, returns, dates)
    rows["S3_S2加权"] = evaluate("S3_ls_top_w", pick(diag, ls_gate=True, top_gate=True),
                                 scores, returns, dates, weight_col="ICIR")
    rows["S4_纯多头"] = evaluate("S4_top_only", pick(diag, top_gate=True, top_t=1.0),
                                 scores, returns, dates)
    rows["S5_S2_icir03"] = evaluate("S5_icir03", pick(diag, icir_th=0.3, ls_gate=True,
                                                      top_gate=True), scores, returns, dates)
    rows["S6_S2_icir04"] = evaluate("S6_icir04", pick(diag, icir_th=0.4, ls_gate=True,
                                                      top_gate=True), scores, returns, dates)

    cmp = pd.DataFrame(rows).T
    cmp.index.name = "配置"
    cmp.to_csv(OUT / "selection_compare.csv", encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print("筛选配置对照（全样本内，可交易池）")
    print("=" * 70)
    tbl = cmp[["累计收益", "年化收益", "年化波动", "夏普比率", "最大回撤",
               "年化换手", "入选数"]].apply(pd.to_numeric)
    for c in ["累计收益", "年化收益", "年化波动", "最大回撤"]:
        tbl[c] = (tbl[c] * 100).round(2)
    tbl[["夏普比率", "年化换手"]] = tbl[["夏普比率", "年化换手"]].round(3)
    print(tbl.to_string())

    print("\n[4/4] 配置 × 持仓数对照（区分筛选贡献与分散化贡献）...")
    scan_configs = {
        "S0等权": (pick(diag), None),
        "S3_ICIR加权": (pick(diag, ls_gate=True, top_gate=True), "ICIR"),
        "S4纯多头等权": (pick(diag, top_gate=True, top_t=1.0), None),
        "S7纯多头_t加权": (pick(diag, top_gate=True, top_t=1.0), "多头t"),
    }
    scan = {}
    for label, (fdf, wcol) in scan_configs.items():
        for tn in (10, 20, 50, 100):
            tag = f"{label}_top{tn}"
            scan[tag] = evaluate(tag, fdf, scores, returns, dates,
                                 weight_col=wcol, top_n=tn)
    scan_df = pd.DataFrame(scan).T
    scan_df.index.name = "配置"
    scan_df.to_csv(OUT / "topn_scan.csv", encoding="utf-8-sig")
    st = scan_df[["累计收益", "年化收益", "夏普比率", "最大回撤"]].apply(pd.to_numeric)
    st["累计收益"] = (st["累计收益"] * 100).round(2)
    st["年化收益"] = (st["年化收益"] * 100).round(2)
    st["最大回撤"] = (st["最大回撤"] * 100).round(2)
    st["夏普比率"] = st["夏普比率"].round(3)
    print(st.to_string())
    print(f"\n结果已保存到：{OUT}")


if __name__ == "__main__":
    main()
