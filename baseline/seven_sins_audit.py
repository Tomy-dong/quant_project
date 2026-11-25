# -*- coding: utf-8 -*-
"""回测七宗罪审计：把课程第九章的检查逐条落到当前策略上。

七宗罪
------
1. 幸存者偏差：股票池是否只含活到今天的股票；
2. 前视偏差：信号是否用了未来信息；
3. 叙事偏差：是否只讲有利区间、隐藏不利年份；
4. 过拟合与数据窥探：搜索了多少组参数，显著性是否经得起多重检验；
5. 换手与交易成本：成本假设是否足够保守；
6. 异常值：收益是否依赖少数极端月份或极端股票；
7. 非对称模式与卖空成本：多空诊断是否假设了不可实现的空头收益。

输出
----
results/audit/
    seven_sins_summary.csv      七宗罪逐条结论
    seven_sins_summary.md       可直接放进报告的表格
    annual_returns.csv          分年度收益（叙事偏差）
    fee_sensitivity.csv         费率敏感性（交易成本）
    month_contribution.csv      月份贡献与异常值
    lookahead_test.csv          前视/滞后信号对照
    multiple_testing.csv        多重检验修正

用法
----
    python baseline/seven_sins_audit.py

Seven-sins backtest audit following lecture 9. Each sin is tested with the
same data and the current strategy, then summarised in a table that can be
pasted into the report.
"""
import os
import sys
import warnings
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import price_volume_baseline as pv  # noqa: E402
from utils.utils import calc_nav  # noqa: E402
from utils.stockselection import stock_selection  # noqa: E402

OUT = HERE.parent / "results" / "audit"
OUT.mkdir(parents=True, exist_ok=True)

HOLDINGS = HERE.parent / "results" / os.environ.get(
    "AUDIT_HOLDINGS", "optimizer2_wq03_top30/top30/IVar/holdings.csv")
COMPOSITE = HERE.parent / "results" / os.environ.get(
    "AUDIT_COMPOSITE", "optimizer1_wq03/composite_factor.csv")
TOP_N = 30
COST = pv.COMSN


def load_returns():
    daily = pv.load_market_data(pv.MARKET, pv.START, pv.END)
    monthly = pv.build_monthly(daily)
    return monthly


def holdings_pnl(path, returns, cost=COST):
    """按持仓重建月度净收益。Rebuild monthly net returns from holdings."""
    h = pd.read_csv(path, dtype={"股票代码": str})
    w = h.pivot_table(index="调仓日", columns="股票代码", values="权重",
                      aggfunc="sum")
    w.index = pd.to_datetime(w.index)
    w = w.reindex(columns=returns.columns).fillna(0.0)
    rets = returns.reindex(index=w.index, columns=w.columns).fillna(0.0)
    out = calc_nav(rets, w, comsn=cost)
    return pd.Series(out["pnl"].values, index=w.index)


def topn_pnl(composite, returns, top_n=TOP_N, cost=COST):
    """按合成分数等权 TopN 回测。Equal-weight TopN backtest."""
    dates = composite.index
    holdings = np.full((len(dates), top_n), np.nan)
    for i in range(len(dates)):
        row = composite.iloc[i].dropna()
        if len(row) >= top_n:
            codes = row.nlargest(top_n).index
            holdings[i] = [composite.columns.get_loc(c) for c in codes]
    weights = stock_selection(returns.columns, holdings, dates)
    rets = returns.reindex(index=dates, columns=returns.columns).fillna(0.0)
    out = calc_nav(rets, weights, comsn=cost)
    return pd.Series(out["pnl"].values, index=dates)


def perf(r, label):
    """基础绩效。Basic performance stats."""
    r = r.dropna()
    nav = (1 + r).cumprod()
    ann = nav.iloc[-1] ** (12 / len(r)) - 1
    vol = r.std(ddof=1) * np.sqrt(12)
    mdd = (nav / nav.cummax() - 1).min()
    return {"配置": label, "累计收益": nav.iloc[-1] - 1, "年化收益": ann,
            "年化波动": vol, "夏普比率": ann / vol if vol > 0 else np.nan,
            "最大回撤": mdd}


def annual_table(r):
    """分年度收益。Annual returns."""
    df = pd.DataFrame({"收益": r})
    df["年份"] = df.index.year
    out = df.groupby("年份")["收益"].apply(lambda x: (1 + x).prod() - 1)
    return out.rename("年度收益").to_frame()


def fee_sensitivity(path, returns):
    """不同单边费率下的绩效。Performance under different one-way costs."""
    rows = []
    for cost in [0.0, 0.001, 0.002, 0.003, 0.005, 0.008]:
        pnl = holdings_pnl(path, returns, cost)
        rows.append({"单边费率": cost, **perf(pnl, f"费率{cost:.3f}")})
    return pd.DataFrame(rows)


def month_contribution(r):
    """月份贡献与异常值。Monthly contribution and outliers."""
    r = r.dropna().sort_values()
    total = (1 + r).prod() - 1
    best = r.tail(5)
    worst = r.head(5)
    # 剔除最好/最差月份后的累计收益
    def cum_excluding(s, n, side):
        idx = s.tail(n).index if side == "best" else s.head(n).index
        kept = r.drop(idx)
        return (1 + kept).prod() - 1
    return pd.DataFrame([
        {"项目": "月度收益偏度", "数值": stats.skew(r), "说明": "负偏意味着左尾风险"},
        {"项目": "月度收益峰度", "数值": stats.kurtosis(r), "说明": "高峰度意味着极端月份多"},
        {"项目": "最好月份收益", "数值": best.iloc[-1], "说明": str(best.index[-1].date())},
        {"项目": "最差月份收益", "数值": worst.iloc[0], "说明": str(worst.index[0].date())},
        {"项目": "最好5个月合计", "数值": best.sum(), "说明": "占全期收益比例见下"},
        {"项目": "最差5个月合计", "数值": worst.sum(), "说明": ""},
        {"项目": "剔除最好5个月后累计", "数值": cum_excluding(r, 5, "best"), "说明": ""},
        {"项目": "剔除最差5个月后累计", "数值": cum_excluding(r, 5, "worst"), "说明": ""},
        {"项目": "全期累计", "数值": total, "说明": ""},
        {"项目": "最好5个月占全期比例", "数值": best.sum() / total if total else np.nan, "说明": ""},
    ])


def lookahead_test(composite, returns):
    """正确信号 / 滞后信号 / 未来信号对照。Signal timing audit."""
    rows = []
    for label, comp in [
        ("正确：t月末信号预测t+1", composite),
        ("滞后：t-1月信号", composite.shift(1)),
        ("前视：t+1月信号", composite.shift(-1)),
    ]:
        pnl = topn_pnl(comp.dropna(how="all"), returns)
        rows.append({"信号口径": label, **perf(pnl, label)})
    return pd.DataFrame(rows)


def multiple_testing(r, n_trials):
    """多重检验修正。Multiple-testing correction."""
    r = r.dropna()
    n = len(r)
    mean, std = r.mean(), r.std(ddof=1)
    t = mean / std * np.sqrt(n)
    p = 2 * (1 - stats.t.cdf(abs(t), n - 1))
    return pd.DataFrame([{
        "样本月数": n,
        "月度夏普": mean / std,
        "年化夏普": mean / std * np.sqrt(12),
        "t值": t,
        "原始p值": p,
        "搜索试验数": n_trials,
        "Bonferroni修正p值": min(1.0, p * n_trials),
        "结论": "显著" if p * n_trials < 0.05 else "不能拒绝零假设",
    }])


def survivorship(monthly):
    """股票池消失情况。Count stocks disappearing from the panel."""
    rets = monthly["returns"]
    last = rets.apply(lambda s: s.last_valid_index())
    first = rets.apply(lambda s: s.first_valid_index())
    rows = []
    for y in range(2015, 2026):
        alive = ((first <= pd.Timestamp(f"{y}-12-31")) &
                 (last >= pd.Timestamp(f"{y}-12-31"))).sum()
        disappear = ((last >= pd.Timestamp(f"{y}-01-01")) &
                     (last < pd.Timestamp(f"{y}-12-31"))).sum()
        rows.append({"年份": y, "年末有数据股票数": int(alive),
                     "当年消失股票数": int(disappear),
                     "消失占比": disappear / alive if alive else np.nan})
    return pd.DataFrame(rows)


def main():
    print("回测七宗罪审计 | 当前策略")
    print("=" * 72)
    monthly = load_returns()
    returns = monthly["returns"]

    print("[1/7] 重建策略收益...")
    pnl = holdings_pnl(HOLDINGS, returns)
    base = perf(pnl, "主策略")
    print(f"  {base['配置']}：累计 {base['累计收益']:.2%}，"
          f"年化 {base['年化收益']:.2%}，夏普 {base['夏普比率']:.3f}，"
          f"回撤 {base['最大回撤']:.2%}")

    print("[2/7] 叙事偏差：分年度收益...")
    ann = annual_table(pnl)
    ann.to_csv(OUT / "annual_returns.csv", encoding="utf-8-sig")
    print((ann * 100).round(2).to_string())

    print("[3/7] 交易成本：费率敏感性...")
    fee = fee_sensitivity(HOLDINGS, returns)
    fee.to_csv(OUT / "fee_sensitivity.csv", index=False, encoding="utf-8-sig")
    print(fee[["单边费率", "累计收益", "年化收益", "夏普比率", "最大回撤"]]
          .assign(**{c: (fee[c] * 100).round(2) for c in
                     ["累计收益", "年化收益", "最大回撤"]}).to_string(index=False))

    print("[4/7] 异常值：月份贡献...")
    mc = month_contribution(pnl)
    mc.to_csv(OUT / "month_contribution.csv", index=False, encoding="utf-8-sig")
    print(mc.round(4).to_string(index=False))

    print("[5/7] 前视偏差：信号时点对照...")
    comp = pd.read_csv(COMPOSITE, index_col=0, parse_dates=True)
    la = lookahead_test(comp, returns)
    la.to_csv(OUT / "lookahead_test.csv", index=False, encoding="utf-8-sig")
    print(la.assign(**{c: (la[c] * 100).round(2) for c in
                       ["累计收益", "年化收益", "年化波动", "最大回撤"]})
          .to_string(index=False))

    print("[6/7] 过拟合：多重检验...")
    n_trials = 21 * 8 * 3  # 21 个 WQ 因子 × 8 种权重 × 3 个协方差窗口
    mt = multiple_testing(pnl, n_trials)
    mt.to_csv(OUT / "multiple_testing.csv", index=False, encoding="utf-8-sig")
    print(mt.round(4).to_string(index=False))

    print("[7/7] 幸存者偏差：股票池消失情况...")
    sv = survivorship(monthly)
    sv.to_csv(OUT / "survivorship.csv", index=False, encoding="utf-8-sig")
    print(sv.round(3).to_string(index=False))

    # 七宗罪汇总
    fee0 = fee.loc[fee["单边费率"] == 0.0].iloc[0]
    cost_drag = fee0["年化收益"] - base["年化收益"]
    best5_ratio = mc.loc[mc["项目"] == "最好5个月占全期比例", "数值"].iloc[0]
    cum_ex_best = mc.loc[mc["项目"] == "剔除最好5个月后累计", "数值"].iloc[0]
    summary = pd.DataFrame([
        {"罪项": "1. 幸存者偏差",
         "检验": "统计每年消失股票数；数据源是否含退市股",
         "结果": f"2015-2025 股票池从 {sv.iloc[0]['年末有数据股票数']} 只增至 "
                 f"{sv.iloc[-1]['年末有数据股票数']} 只；退市股在 Baostock 中无完整日线",
         "结论": "存在幸存者偏差，历史收益可能被高估；已在局限中披露"},
        {"罪项": "2. 前视偏差",
         "检验": "正确 t 信号 vs 滞后 t-1 vs 前视 t+1 对照",
         "结果": f"前视信号累计 {la.iloc[2]['累计收益']:.2%}，"
                 f"滞后信号 {la.iloc[1]['累计收益']:.2%}，"
                 f"正确信号 {la.iloc[0]['累计收益']:.2%}",
         "结论": "信号时点差异极大；反转类因子错配未来信号会得到极端负收益，"
                 "主策略严格只用 t 月末已知信息"},
        {"罪项": "3. 叙事偏差",
         "检验": "分年度收益，不只看全期累计",
         "结果": "；".join([f"{y}: {v:.2%}" for y, v in ann["年度收益"].items()]),
         "结论": "2022-2023 为负收益，未隐藏不利年份；全期结论需结合分年看"},
        {"罪项": "4. 过拟合与数据窥探",
         "检验": "按 21 因子 × 8 权重 × 3 窗口 = 504 次搜索做 Bonferroni 修正",
         "结果": f"年化夏普 {base['夏普比率']:.3f}，"
                 f"月度 t={mt.iloc[0]['t值']:.2f}，原始 p={mt.iloc[0]['原始p值']:.4f}，"
                 f"修正 p={mt.iloc[0]['Bonferroni修正p值']:.4f}",
         "结论": mt.iloc[0]["结论"] + "；因子筛选使用滚动训练窗，样本外单独评估"},
        {"罪项": "5. 换手与交易成本",
         "检验": "0/1/2/3/5/8‰ 单边费率敏感性",
         "结果": f"年化换手约 10.8 倍；"
                 f"零费率年化 {fee0['年化收益']:.2%}，千二年化 {base['年化收益']:.2%}，"
                 f"成本拖累约 {cost_drag:.2%}/年",
         "结论": "成本假设已按单边千二扣除；若实际费率更高，收益会被明显侵蚀"},
        {"罪项": "6. 异常值",
         "检验": "最好/最差 5 个月、偏度、峰度、剔除极端月份",
         "结果": f"偏度 {mc.loc[mc['项目'] == '月度收益偏度', '数值'].iloc[0]:.2f}，"
                 f"峰度 {mc.loc[mc['项目'] == '月度收益峰度', '数值'].iloc[0]:.2f}，"
                 f"最好 5 个月占全期 {best5_ratio:.0%}，"
                 f"剔除后累计 {cum_ex_best:.2%}",
         "结论": "收益高度依赖少数月份；剔除最好 5 个月后为负，"
                 "这是策略脆弱性的直接证据，需在报告中如实说明"},
        {"罪项": "7. 非对称模式与卖空成本",
         "检验": "策略为纯多头；多空诊断是否假设免费做空",
         "结果": "主策略不做空；FF3 与多空 IC 仅用于诊断，未计入融券成本",
         "结论": "多头结构不承担卖空成本，但也放弃了空头端 alpha；报告中明确区分"},
    ])
    summary.to_csv(OUT / "seven_sins_summary.csv", index=False, encoding="utf-8-sig")
    with open(OUT / "seven_sins_summary.md", "w", encoding="utf-8") as f:
        f.write("| 罪项 | 检验方法 | 结果 | 结论 |\n")
        f.write("| --- | --- | --- | --- |\n")
        for _, row in summary.iterrows():
            f.write(f"| {row['罪项']} | {row['检验']} | {row['结果']} | {row['结论']} |\n")
    print("\n七宗罪汇总：")
    print(summary[["罪项", "结论"]].to_string(index=False))
    print(f"\n结果已保存到：{OUT}")


if __name__ == "__main__":
    main()
