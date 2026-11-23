"""归因对照实验 / Attribution ablation.

把“容量过滤”与“因子口径统一”拆成 2x2，在同一份数据上各跑一遍基线，
确认收益转负的主因。配置：
    A: 容量开 + 口径统一   (当前)
    B: 容量关 + 口径统一
    C: 容量开 + 口径分裂
    D: 容量关 + 口径分裂   (修复前)
容量关 = 仅保留 非ST、有成交、成交额>0；口径分裂 = 用原始因子算 IC 选因子，
用预处理后因子合成（与修复前一致）。

Usage:
    python baseline/ablation_attribution.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "baseline")
sys.path.insert(0, ".")

import price_volume_baseline as pv  # noqa: E402
from utils.data_standardize import preprocess  # noqa: E402

ABLATION_DIR = Path("results/ablation")
ABLATION_DIR.mkdir(parents=True, exist_ok=True)
pv.OUTPUT_DIR = ABLATION_DIR  # 让 factor_test / backtest 写到这里，不覆盖正式结果


def make_valid(monthly, capacity_on):
    """构造可交易股票池。Build the tradable-stock mask.

    capacity_on=True  走正式的 ADV+次新+ST 过滤；
    capacity_on=False 只保留 非ST、有成交、成交额>0（修复前口径）。
    """
    if capacity_on:
        return pv.filter_pool(monthly)
    is_st = monthly["is_st"].fillna(1)
    amount = monthly["amount"].fillna(0)
    vol = monthly["vol"].fillna(0)
    return (is_st == 0) & (vol > 0) & (amount > 0)


def run_once(capacity_on, unified, monthly, returns, dates):
    """跑单个配置。Run a single 2x2 configuration."""
    valid = make_valid(monthly, capacity_on)
    avg_pool = valid.sum(axis=1).mean()

    factors = pv.compute_price_volume_factors(monthly, valid)
    scores = {name: preprocess(fac) for name, fac in factors.items()}

    # IC 来源：统一用 scores，分裂用 raw factors
    # IC source: unified uses scores, split uses raw factors
    ic_input = scores if unified else factors
    tag = f"cap{'on' if capacity_on else 'off'}_{'uni' if unified else 'split'}"
    print(f"\n{'#'*60}\n# 配置 {tag.upper()}  容量={'开' if capacity_on else '关'}  "
          f"口径={'统一' if unified else '分裂'}\n{'#'*60}")
    print(f"  平均可交易股票数：{avg_pool:.0f}")

    _, summary = pv.run_factor_test(ic_input, returns, dates)
    selected = pv.select_factors(summary)
    composite = pv.composite_factor(scores, selected, summary)

    out_dir = ABLATION_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    _, perf = pv.backtest_portfolio(composite, returns, dates,
                                    output_dir=out_dir, title=f"Ablation {tag}")
    perf["入选因子数"] = len(selected)
    perf["入选因子"] = "、".join(selected)
    return perf


def main():
    print(f"市场：{pv.MARKET} | 区间：{pv.START} ~ {pv.END}")
    print("[1/3] 载入日频数据并构建月频面板...")
    daily = pv.load_market_data(pv.MARKET, pv.START, pv.END)
    monthly = pv.build_monthly(daily)
    dates = monthly["dates"]
    returns = monthly["returns"]
    print(f"  月末样本：{len(dates)} 期，{monthly['price'].shape[1]} 只股票")

    print("\n[2/3] 运行 2x2 归因配置...")
    rows = {}
    for cap in (True, False):
        for uni in (True, False):
            label = f"cap{'on' if cap else 'off'}_{'uni' if uni else 'split'}"
            rows[label] = run_once(cap, uni, monthly, returns, dates)

    print("\n[3/3] 汇总对比...")
    cmp = pd.DataFrame(rows).T
    cmp.index.name = "配置"
    show = cmp[["累计收益", "年化收益", "年化波动", "夏普比率",
                "最大回撤", "年化换手", "入选因子数"]].copy()
    # 转置后列可能是 object dtype，先强转数值再取整
    # After .T columns may be object dtype; coerce to numeric before rounding
    for col in ["累计收益", "年化收益", "年化波动", "最大回撤"]:
        show[col] = (pd.to_numeric(show[col]) * 100).round(2)
    show["夏普比率"] = pd.to_numeric(show["夏普比率"]).round(3)
    show["年化换手"] = pd.to_numeric(show["年化换手"]).round(3)
    print("\n" + "=" * 80)
    print("归因对照表（容量过滤 × 口径统一，2015-2025 全区间）")
    print("=" * 80)
    print(show.to_string())
    print("=" * 80)
    print("\n入选因子详情：")
    for label, perf in rows.items():
        print(f"  {label}: {perf['入选因子']}")

    cmp.to_csv(ABLATION_DIR / "comparison.csv", encoding="utf-8-sig")
    print(f"\n结果已保存到：{ABLATION_DIR / 'comparison.csv'}")


if __name__ == "__main__":
    main()
