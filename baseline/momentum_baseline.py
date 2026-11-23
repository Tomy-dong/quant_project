# -*- coding: utf-8 -*-
"""单因子基线：动量 / 反转类因子在 A 股的多头选股实验。

用途
----
作为多因子策略的对照组，回答一个最基本的问题：只用动量类因子做多头
选股，在 A 股能不能赚钱。结论是否定的，具体数字见 results/baseline/。

流程
----
数据载入 -> 月末采样 -> 股票池过滤 -> 因子计算 -> 单因子检验(IC/分组/多空)
-> 按 ICIR 选因子并等权合成 -> Top N 等权持有、月末调仓 -> 绩效输出

注意
----
1. 因子只用后复权 OHLCV 与换手率，不引入财务数据。
2. wgt_return / exp_wgt_return 用月频换手率均值近似日频换手率加权，
   与研报原始定义存在口径差异，仅用于基线对照。
3. 回测为等权 Top N，不含资金配置优化，费率单边 0.002。

用法
----
    python baseline/momentum_baseline.py
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

from data_io import load_benchmark
from utils.FactorTest import factor_test
from utils.data_standardize import preprocess
from utils.utils import calc_nav
from utils.stockselection import stock_selection

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ===================== 配置 =====================
MARKET = "all"            # sh 沪市 / sz 深市 / all 全A
START = "2015-01-01"
END = "2025-12-31"
TOP_N = 10                # 每期持仓只数
COMSN = 0.002             # 单边交易成本
NUM_GROUP = 5             # 因子分组数
ICIR_THRESHOLD = 0.2      # 入选因子的 |ICIR| 下限

DATA_DIR = HERE.parent / "data"
OUTPUT_DIR = HERE.parent / "results" / "baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ===================== 1. 数据载入 =====================
def load_market_data(market, start, end):
    """载入指定市场的日频宽表。market: 'sh' / 'sz' / 'all'。"""
    fields = ["close", "adj_factor", "open", "high", "low",
              "vol", "amount", "turnover_rate", "is_st"]
    daily = {}
    markets = ["sh", "sz"] if market == "all" else [market]

    for fld in fields:
        parts = []
        for mkt in markets:
            path = DATA_DIR / f"{fld}_{mkt}.csv"
            if not path.exists():
                print(f"  警告：{path.name} 不存在，跳过 {mkt} 市场的 {fld}")
                continue
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df = df.loc[(df.index >= start) & (df.index <= end)]
            parts.append(df)
        if parts:
            daily[fld] = pd.concat(parts, axis=1)
            daily[fld] = daily[fld].loc[
                ~daily[fld].index.duplicated(keep="last")]
        else:
            daily[fld] = pd.DataFrame()
    return daily


# ===================== 2. 月末采样与股票池过滤 =====================
def build_monthly(daily):
    """取每月最后一个交易日，构建月频面板。"""
    close = daily["close"]
    adj = daily["adj_factor"]
    price = close * adj                      # 后复权收盘价

    dates = pd.DatetimeIndex(
        pd.Series(price.index).groupby(price.index.to_period("M")).max())
    dates = dates[(dates >= START) & (dates <= END)]

    monthly = {}
    for fld, df in daily.items():
        if df.empty:
            monthly[fld] = pd.DataFrame()
        else:
            monthly[fld] = df.reindex(dates)

    monthly["price"] = price.reindex(dates)
    monthly["returns"] = monthly["price"].pct_change(fill_method=None)
    monthly["dates"] = dates
    return monthly


def filter_pool(monthly, min_amount=10_000):
    """可交易股票池：非 ST、有成交、成交额达标。"""
    is_st = monthly["is_st"].fillna(1)
    amount = monthly["amount"].fillna(0)
    vol = monthly["vol"].fillna(0)

    return (is_st == 0) & (vol > 0) & (amount >= min_amount)


# ===================== 3. 因子计算 =====================
def compute_momentum_factors(monthly, valid):
    """计算 19 个动量 / 反转类因子。

    华泰金工系列（13 个）：return_Nm、wgt_return_Nm、exp_wgt_return_Nm、HAlpha
    Alpha101 动量反转（6 个）：Alpha4、Alpha9、Alpha20、Alpha30、Alpha35、Alpha101
    """
    price = monthly["price"]
    turn = monthly["turnover_rate"]
    open_ = monthly["open"] * monthly["adj_factor"]
    high = monthly["high"] * monthly["adj_factor"]
    low = monthly["low"] * monthly["adj_factor"]
    close = monthly["close"] * monthly["adj_factor"]
    vol = monthly["vol"]

    factors = {}
    months = [1, 3, 6, 12]

    # --- 简单收益动量 return_Nm ---
    for N in months:
        fac = price / price.shift(N) - 1
        factors[f"return_{N}m"] = fac.where(valid).replace([np.inf, -np.inf], np.nan)

    # --- 换手率加权收益 wgt_return_Nm ---
    # 研报按日频换手率加权；这里用近 N 个月平均换手率乘区间收益近似。
    for N in months:
        ret_N = price / price.shift(N) - 1
        turn_N = turn.rolling(N, min_periods=1).mean()
        factors[f"wgt_return_{N}m"] = (turn_N * ret_N).where(valid).replace(
            [np.inf, -np.inf], np.nan)

    # --- 指数衰减换手率加权 exp_wgt_return_Nm ---
    for N in months:
        ret_N = price / price.shift(N) - 1
        decay = np.exp(-np.arange(N)[::-1] / (N / 4.0))
        decay = decay / decay.sum()
        turn_N = turn.rolling(N, min_periods=1).apply(
            lambda x: np.dot(x[-len(decay):], decay[-len(x):]), raw=True)
        factors[f"exp_wgt_return_{N}m"] = (turn_N * ret_N).where(valid).replace(
            [np.inf, -np.inf], np.nan)

    # --- HAlpha：近 12 个月收益对基准回归的截距 ---
    bench = load_benchmark(START, END)["close"].reindex(monthly["dates"])
    bench_ret = bench.pct_change(fill_method=None)
    halpha = pd.DataFrame(np.nan, index=price.index, columns=price.columns)
    for col in price.columns:
        stock_ret = price[col].pct_change()
        pair = pd.concat([stock_ret, bench_ret], axis=1).dropna()
        pair.columns = ["stock", "bench"]
        if len(pair) < 12:
            continue
        for i in range(11, len(pair)):
            window = pair.iloc[i - 11:i + 1]
            x = window["bench"].values
            y = window["stock"].values
            if np.std(x) > 1e-8:
                beta = np.cov(x, y)[0, 1] / np.var(x)
                halpha.loc[pair.index[i], col] = np.mean(y) - beta * np.mean(x)
    factors["HAlpha"] = halpha.where(valid)

    # --- Alpha4: -1 * ts_rank(rank(low), 9) ---
    rank_low = low.rank(axis=1, pct=True)
    factors["Alpha4"] = (-1 * rank_low.rolling(9, min_periods=1).rank(pct=True)
                         ).where(valid)

    # --- Alpha9: 条件反转 ---
    delta_close = close.diff(1)
    ts_min = delta_close.rolling(5, min_periods=1).min()
    ts_max = delta_close.rolling(5, min_periods=1).max()
    ts_rank_dc = delta_close.rolling(5, min_periods=1).rank(pct=True)
    alpha9 = ts_rank_dc.copy()
    alpha9[(ts_min > 0) | (ts_max < 0)] = ts_rank_dc[(ts_min > 0) | (ts_max < 0)]
    alpha9[(ts_min <= 0) & (ts_max >= 0)] = -delta_close[(ts_min <= 0) & (ts_max >= 0)]
    factors["Alpha9"] = alpha9.where(valid)

    # --- Alpha20: 跳空回补 ---
    gap_h = open_ - high.shift(1)
    gap_c = open_ - close.shift(1)
    gap_l = open_ - low.shift(1)
    factors["Alpha20"] = (-1 * gap_h.rank(axis=1, pct=True)
                          * gap_c.rank(axis=1, pct=True)
                          * gap_l.rank(axis=1, pct=True)).where(valid)

    # --- Alpha30: 方向符号 × 量能 ---
    sign1 = np.sign(close.diff(1))
    sign2 = np.sign(close.diff(1).shift(1))
    sign3 = np.sign(close.diff(1).shift(2))
    dir_sig = sign1 + sign2 + sign3
    vol5 = vol.rolling(5, min_periods=1).sum()
    vol20 = vol.rolling(20, min_periods=1).sum()
    factors["Alpha30"] = ((1 - (dir_sig * vol5).rank(axis=1, pct=True))
                          / vol20.replace(0, np.nan)).where(valid)

    # --- Alpha35: 三周期共振 ---
    ts_rank_vol = vol.rolling(32, min_periods=1).rank(pct=True)
    bull = close + high - low
    ts_rank_bull = bull.rolling(16, min_periods=1).rank(pct=True)
    rets = close.pct_change()
    ts_rank_ret = rets.rolling(32, min_periods=1).rank(pct=True)
    factors["Alpha35"] = (ts_rank_vol * (1 - ts_rank_bull)
                          * (1 - ts_rank_ret)).where(valid)

    # --- Alpha101: (close-open)/(high-low) ---
    factors["Alpha101"] = ((close - open_) / (high - low + 0.001)).where(valid)

    return factors


# ===================== 4. 因子检验 =====================
def run_factor_test(factors, returns, dates):
    """对每个因子做 IC、分组与多空检验，汇总到 factor_summary.csv。"""
    results = {}
    summary_rows = []
    for name, fac in factors.items():
        print(f"\n=== 因子检验：{name} ===")
        res = factor_test(fac, returns, NumOfGroup=NUM_GROUP,
                          DispFlag=True, PlotFlag=False, comsn=0, name=name)
        results[name] = res

        for metric, key in [("RankIC", "Rank_IC"), ("NormalIC", "Normal_IC")]:
            ic = pd.Series(res[key][1:], index=dates[1:]).dropna()
            summary_rows.append({
                "因子": name, "指标": metric,
                "平均IC": ic.mean(), "IC标准差": ic.std(ddof=1),
                "ICIR": ic.mean() / ic.std(ddof=1) if ic.std(ddof=1) > 0 else 0,
                "p+": (ic > 0).mean(),
            })
        grp = pd.DataFrame(res["GroupRtn"][1:], index=dates[1:])
        ls = grp.iloc[:, -1] - grp.iloc[:, 0]
        summary_rows.append({
            "因子": name, "指标": "多空月均收益",
            "平均IC": ls.mean(), "IC标准差": ls.std(ddof=1),
            "ICIR": ls.mean() / ls.std(ddof=1) if ls.std(ddof=1) > 0 else 0,
            "p+": (ls > 0).mean(),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "factor_summary.csv", index=False, encoding="utf-8-sig")
    print("\n因子检验汇总：")
    print(summary.round(4).to_string(index=False))
    return results, summary


# ===================== 5. 因子合成与回测 =====================
def select_factors(summary, icir_threshold=ICIR_THRESHOLD):
    """按 |ICIR| 阈值选因子，至少保留 3 个。"""
    pivot = summary.pivot_table(index="因子", columns="指标", values="ICIR")
    if "RankIC" not in pivot.columns:
        return pivot.index.tolist()

    selected = pivot[pivot["RankIC"].abs() >= icir_threshold].index.tolist()
    if len(selected) < 3:
        selected = pivot["RankIC"].abs().nlargest(3).index.tolist()
    print(f"\n入选因子（|ICIR|>={icir_threshold}）：{selected}")
    return selected


def composite_factor(factors, selected, summary):
    """等权合成入选因子，IC 为负的因子取反。"""
    pivot = summary.pivot_table(index="因子", columns="指标", values="平均IC")
    composite = pd.DataFrame(0.0, index=factors[selected[0]].index,
                             columns=factors[selected[0]].columns)
    for name in selected:
        sign = 1.0
        if "RankIC" in pivot.columns:
            sign = np.sign(pivot.loc[name, "RankIC"]) or 1.0
        composite = composite + sign * preprocess(factors[name])
    return composite / len(selected)


def backtest_portfolio(composite, returns, dates, top_n=TOP_N, comsn=COMSN):
    """复合因子取每期前 top_n 只，等权持有，月末调仓。"""
    holdings = np.full((len(dates), top_n), np.nan)
    for i in range(len(dates)):
        row = composite.iloc[i].dropna()
        if len(row) >= top_n:
            codes = row.nlargest(top_n).index
            holdings[i] = [composite.columns.get_loc(c) for c in codes]

    weights = stock_selection(returns.columns, holdings, dates)
    outputs = calc_nav(returns.fillna(0), weights, comsn=comsn)

    nav = outputs["nav"]
    pnl = outputs["pnl"]
    years = (nav.index[-1] - nav.index[0]).days / 365.24
    ann_ret = nav.iloc[-1] ** (1 / years) - 1
    ann_vol = pnl.std() * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    max_dd = (nav / nav.cummax() - 1).min()

    perf = pd.Series({
        "累计收益": nav.iloc[-1] - 1,
        "年化收益": ann_ret,
        "年化波动": ann_vol,
        "夏普比率": sharpe,
        "最大回撤": max_dd,
        "年化换手": outputs["turnover"].mean() * 12,
    })
    perf.to_csv(OUTPUT_DIR / "performance.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(10, 5))
    nav.plot(ax=ax, label="策略净值", linewidth=1.5)
    bench = load_benchmark(START, END)["close"].reindex(dates)
    bench_nav = (1 + bench.pct_change(fill_method=None).fillna(0)).cumprod()
    bench_nav.plot(ax=ax, label="沪深300", linewidth=1.5, color="gray")
    ax.set(xlabel="日期", ylabel="净值（初始=1）", title=f"动量基线 ({MARKET.upper()})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "nav.png", dpi=150)
    plt.close(fig)

    print("\n策略绩效：")
    print(perf.round(4).to_string())
    return outputs, perf


# ===================== 主流程 =====================
def main():
    print(f"市场：{MARKET} | 区间：{START} ~ {END} | 持仓数：{TOP_N}")
    print("=" * 60)

    print("\n[1/5] 载入日频数据...")
    daily = load_market_data(MARKET, START, END)
    print(f"  close: {daily['close'].shape}")

    print("\n[2/5] 月末采样...")
    monthly = build_monthly(daily)
    dates = monthly["dates"]
    print(f"  月末样本：{len(dates)} 期，{monthly['price'].shape[1]} 只股票")

    print("\n[3/5] 构建可交易股票池...")
    valid = filter_pool(monthly)
    print(f"  平均每期可交易股票数：{valid.sum(axis=1).mean():.0f}")

    print("\n[4/5] 计算动量因子...")
    factors = compute_momentum_factors(monthly, valid)
    print(f"  共 {len(factors)} 个因子")

    print("\n[5/5] 因子检验与回测...")
    returns = monthly["returns"]
    _, summary = run_factor_test(factors, returns, dates)

    selected = select_factors(summary)
    composite = composite_factor(factors, selected, summary)
    composite.to_csv(OUTPUT_DIR / "composite_factor.csv", encoding="utf-8-sig")
    backtest_portfolio(composite, returns, dates)

    print(f"\n结果已保存到：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
