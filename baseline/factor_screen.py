# -*- coding: utf-8 -*-
"""大因子库单因子初筛：|IC| >= 0.05 且 |ICIR| >= 0.5。

流程
----
1. 用 factor_lib 计算四个家族的月末因子面板（已存在则直接读取）；
2. 复用基线口径构造股票池与月末收益（非 ST、上市满 120 日、ADV 前 80%）；
3. 每个因子做截面预处理（MAD 缩尾 + 行业去均值 + Zscore），计算月频 RankIC；
4. 全样本 2015-2025 与子样本 2015-2019 分别统计 IC / ICIR / t 值；
5. 按全样本阈值筛出因子池，输出汇总表，并可选做 Top30 单因子基线回测。

用法
----
    python baseline/factor_screen.py                # 计算并筛选
    python baseline/factor_screen.py --force        # 强制重算因子面板
    python baseline/factor_screen.py --backtest     # 同时做单因子回测
    python baseline/factor_screen.py --backtest-only
"""
import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import price_volume_baseline as pv  # noqa: E402
from factor_lib import matrices as flm  # noqa: E402
from factor_lib import runner  # noqa: E402
from utils.data_standardize import preprocess  # noqa: E402

OUTPUT_DIR = HERE.parent / "results" / "factor_screen"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SINGLE_DIR = HERE.parent / "results" / "single_factor"
SINGLE_DIR.mkdir(parents=True, exist_ok=True)

# ===================== 配置 / Config =====================
IC_MIN = 0.05            # |平均IC| 下限 / min |mean IC|
ICIR_MIN = 0.5           # |ICIR| 下限 / min |ICIR|
MIN_MONTHS = 24          # 有效 IC 月数下限 / min valid months
TOP_N = 30               # 单因子回测持仓数 / holdings per period
TRAIN_END = "2019-12-31"  # 固定因子策略的训练期末 / training cut for the fixed strategy

FAMILY_LABELS = {
    "alpha": "Alpha191", "a158": "Alpha158", "a101": "Alpha101",
    "ts": "清华22",
}


# ===================== 股票池与收益 / Universe and returns =====================
def build_monthly(matrices: dict, month_dates: pd.DatetimeIndex):
    """按基线口径构造月末收益、可交易池与 60 日 ADV。

    与 price_volume_baseline.build_monthly / filter_pool 等价，只是直接复用
    已载入的矩阵，避免重复读取 CSV。

    Month-end returns, tradable pool and 60-day ADV, equivalent to the
    baseline pipeline but reusing the already loaded matrices.
    """
    close = matrices["close"]
    amount = matrices["amount"]
    volume = matrices["volume"]
    is_st = matrices.get("is_st")

    price = close.reindex(month_dates)
    returns = price.pct_change(fill_method=None)
    adv = amount.rolling(pv.ADV_WINDOW, min_periods=20).mean().reindex(month_dates)

    basic = pd.read_csv(flm.DATA_DIR / "sse_stock_basic.csv", dtype=str)
    ipo = pd.to_datetime(basic.set_index("code")["ipoDate"], errors="coerce")
    ipo.index = ipo.index.str[3:] + "." + ipo.index.str[:2].str.upper()
    ipo = ipo.reindex(close.columns).fillna(pd.Timestamp("2100-01-01"))
    end_pos = close.index.searchsorted(month_dates, side="right") - 1
    ipo_pos = close.index.searchsorted(ipo.to_numpy())
    listed = end_pos[:, None] - ipo_pos[None, :] + 1
    listed = np.where(ipo.to_numpy() < close.index[0], 10_000, listed)
    listed_days = pd.DataFrame(listed, index=month_dates, columns=close.columns)

    traded = (volume.reindex(month_dates).fillna(0) > 0) & (
        amount.reindex(month_dates).fillna(0) > 0)
    st_flag = (is_st.reindex(month_dates).fillna(1) == 0) if is_st is not None else True
    pool = traded & (listed_days >= pv.MIN_LISTED_DAYS) & st_flag
    adv_m = adv.where(pool)
    threshold = adv_m.quantile(pv.ADV_PCT, axis=1)
    valid = pool & adv_m.gt(threshold, axis=0)
    turn = matrices.get("turnover_rate")
    return {
        "price": price, "returns": returns, "valid": valid, "adv": adv,
        "dates": month_dates,
        # 供优化器一的规模暴露回归使用（复权价×复权量 与原始市值一致）
        # Used by optimizer-1's size exposure regression.
        "close": close.reindex(month_dates),
        "vol": volume.reindex(month_dates),
        "turnover_rate": (turn.reindex(month_dates) if turn is not None
                          else pd.DataFrame(index=month_dates, columns=close.columns)),
        "is_st": (is_st.reindex(month_dates) if is_st is not None else None),
        "amount": amount.reindex(month_dates),
    }


# ===================== RankIC =====================
def rank_ic(zfac: pd.DataFrame, fwd_rank: pd.DataFrame) -> pd.Series:
    """逐月 RankIC（因子 t 对收益 t+1）。Monthly rank IC."""
    return zfac.rank(axis=1).corrwith(fwd_rank, axis=1)


def ic_stats(ic: pd.Series, start=None, end=None) -> dict:
    """IC 描述统计。Descriptive statistics of an IC series."""
    seg = ic.dropna()
    if start is not None:
        seg = seg[seg.index >= start]
    if end is not None:
        seg = seg[seg.index <= end]
    n = len(seg)
    if n < 3:
        return {"months": n, "ic": np.nan, "ic_std": np.nan,
                "icir": np.nan, "t": np.nan, "pos": np.nan}
    std = seg.std(ddof=1)
    mean = seg.mean()
    return {
        "months": n,
        "ic": mean,
        "ic_std": std,
        "icir": mean / std if std > 0 else np.nan,
        "t": mean / std * np.sqrt(n) if std > 0 else np.nan,
        "pos": float((seg > 0).mean()),
    }


# ===================== 主流程 / Main =====================
def ensure_panels(families, force=False):
    """确保四个家族的面板已计算并落盘。Make sure family panels exist."""
    matrices, month_dates = flm.build_matrices(pv.START, pv.END)
    benchmark = flm.build_benchmark(pv.START, pv.END)
    for family in families:
        runner.compute_family(family, matrices, month_dates, benchmark, force=force)
    return matrices, month_dates


def screen(families, force=False):
    """计算全部因子并输出初筛结果。Compute and screen all factors."""
    matrices, month_dates = ensure_panels(families, force=force)
    columns = matrices["close"].columns
    monthly = build_monthly(matrices, month_dates)
    valid = monthly["valid"]
    returns = monthly["returns"]
    fwd_rank = returns.shift(-1).rank(axis=1)

    rows = []
    ic_series = {}
    panels = runner.load_all()
    print(f"\n共加载 {len(panels)} 个因子，开始计算 RankIC ...")

    for i, (name, panel) in enumerate(panels.items(), 1):
        panel = panel.reindex(index=month_dates, columns=columns)
        z = preprocess(panel).where(valid)
        ic = rank_ic(z, fwd_rank)
        ic_series[name] = ic

        full = ic_stats(ic)
        sub = ic_stats(ic, end=TRAIN_END)
        coverage = float(z.notna().sum(axis=1).div(valid.sum(axis=1)).mean())
        family = FAMILY_LABELS.get(name.split("_")[0], name.split("_")[0])
        passed = (full["months"] >= MIN_MONTHS
                  and abs(np.nan_to_num(full["ic"])) >= IC_MIN
                  and abs(np.nan_to_num(full["icir"])) >= ICIR_MIN)
        rows.append({
            "因子": name, "家族": family, "窗口": "2015-2025",
            "平均IC": full["ic"], "IC标准差": full["ic_std"], "ICIR": full["icir"],
            "t值": full["t"], "IC>0占比": full["pos"], "月数": full["months"],
            "覆盖度": coverage, "通过": passed,
            "训练期IC": sub["ic"], "训练期ICIR": sub["icir"],
        })
        if i % 50 == 0:
            print(f"  已处理 {i}/{len(panels)}")

    summary = pd.DataFrame(rows).sort_values("ICIR", key=np.abs, ascending=False)
    summary.to_csv(OUTPUT_DIR / "factor_summary.csv", index=False, encoding="utf-8-sig")

    passed = summary[summary["通过"]].copy()
    passed["方向"] = np.sign(passed["平均IC"]).replace(0, 1).astype(int)
    passed.to_csv(OUTPUT_DIR / "screen_pass.csv", index=False, encoding="utf-8-sig")

    digest = summary.groupby("家族").agg(
        候选=("因子", "size"), 通过=("通过", "sum"),
        最大绝对ICIR=("ICIR", lambda s: s.abs().max())).reset_index()
    digest.to_csv(OUTPUT_DIR / "family_digest.csv", index=False, encoding="utf-8-sig")

    ic_panel = pd.DataFrame(ic_series)
    ic_panel.to_csv(OUTPUT_DIR / "ic_panel.csv", encoding="utf-8-sig")

    if len(passed) >= 2:
        z_panels = {}
        for name in passed["因子"]:
            z_panels[name] = preprocess(
                panels[name].reindex(index=month_dates, columns=columns)).where(valid)
        corr = pd.DataFrame(
            {a: [z_panels[a].corrwith(z_panels[b], axis=1).mean() for b in z_panels]
             for a in z_panels}, index=list(z_panels))
        corr.to_csv(OUTPUT_DIR / "ic_corr_matrix.csv", encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print(f"家族汇总（阈值 |IC|>={IC_MIN}, |ICIR|>={ICIR_MIN}）：")
    print(digest.to_string(index=False))
    print(f"\n通过因子 {len(passed)} 个：")
    if len(passed):
        print(passed[["因子", "家族", "平均IC", "ICIR", "t值", "覆盖度",
                      "训练期IC", "训练期ICIR"]].round(4).to_string(index=False))
    print(f"\n结果目录：{OUTPUT_DIR}")
    return summary, passed, monthly


def single_factor_backtests(passed: pd.DataFrame, monthly: dict,
                            month_dates: pd.DatetimeIndex):
    """对通过初筛的因子逐个做 Top30 等权回测。Top30 equal-weight backtests."""
    panels = runner.load_all()
    valid = monthly["valid"]
    returns = monthly["returns"]
    columns = returns.columns

    rows = []
    for name in passed["因子"]:
        panel = panels[name].reindex(index=month_dates, columns=columns)
        z = preprocess(panel).where(valid)
        sign = float(np.sign(passed.loc[passed["因子"] == name, "平均IC"].iloc[0]) or 1.0)
        score = z * sign
        out_dir = SINGLE_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        _, perf = pv.backtest_portfolio(score, returns, month_dates, top_n=TOP_N,
                                        output_dir=out_dir, title=f"单因子 {name}")
        row = perf.to_dict()
        row["因子"] = name
        rows.append(row)
        print(f"  {name}: 年化 {perf['年化收益']:.2%}，夏普 {perf['夏普比率']:.2f}，"
              f"最大回撤 {perf['最大回撤']:.2%}")

    summary = pd.DataFrame(rows)
    cols = ["因子"] + [c for c in summary.columns if c != "因子"]
    summary[cols].sort_values("夏普比率", ascending=False).to_csv(
        SINGLE_DIR / "summary.csv", index=False, encoding="utf-8-sig")
    print(f"\n单因子回测汇总：{SINGLE_DIR / 'summary.csv'}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", default=",".join(runner.FAMILIES))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--backtest-only", action="store_true")
    args = parser.parse_args()
    families = [x.strip() for x in args.families.split(",") if x.strip()]

    pass_path = OUTPUT_DIR / "screen_pass.csv"
    if args.backtest_only:
        passed = pd.read_csv(pass_path)
        matrices, month_dates = flm.build_matrices(pv.START, pv.END)
        monthly = build_monthly(matrices, month_dates)
        single_factor_backtests(passed, monthly, month_dates)
        return

    _, passed, monthly = screen(families, force=args.force)
    if args.backtest and len(passed):
        matrices, month_dates = flm.build_matrices(pv.START, pv.END)
        monthly = build_monthly(matrices, month_dates)
        single_factor_backtests(passed, monthly, month_dates)


if __name__ == "__main__":
    main()
