# -*- coding: utf-8 -*-
"""辅助数据下载：沪深300 行情 + 交易日历 + 行业分类 + 基本资料。

用途：
  - csi300_daily.csv：主动管理 & 市场中性统一基准（沪深300价格指数，Baostock sh.000300）
  - trade_calendar.csv：月末调仓、容量筛选日期对齐
  - stk_company_info.csv：行业分类，因子中性化 + 行业配置对照
  - sse_stock_basic.csv：上市日期 + 退市日期，剔除次新股（ipoDate 距今 < 60 日）
"""

import time
from pathlib import Path
import baostock as bs
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]  # scripts/ -> 仓库根目录
DATA_DIR = PACKAGE_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

START = "2015-01-01"
END = "2025-12-31"


def to_frame(result):
    if result.error_code != "0":
        raise RuntimeError(f"{result.error_code}: {result.error_msg}")
    rows = []
    while result.next():
        rows.append(result.get_row_data())
    return pd.DataFrame(rows, columns=result.fields)


if __name__ == "__main__":
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"BaoStock login failed: {login.error_msg}")

    try:
        t0 = time.time()

        # 1) 沪深300 价格指数（主动管理 & 市场中性统一基准）
        print("\n[1/4] 沪深300 指数行情...")
        idx = to_frame(bs.query_history_k_data_plus(
            "sh.000300",
            "date,code,open,high,low,close,preclose,volume,amount",
            start_date=START, end_date=END, frequency="d",
        ))
        idx["date"] = pd.to_datetime(idx["date"])
        idx.to_csv(DATA_DIR / "csi300_daily.csv", index=False, encoding="utf-8-sig")
        print(f"  csi300_daily.csv: {idx.shape}")

        # 2) 交易日历
        print("[2/4] 交易日历...")
        cal = to_frame(bs.query_trade_dates(start_date=START, end_date=END))
        cal["calendar_date"] = pd.to_datetime(cal["calendar_date"])
        cal.to_csv(DATA_DIR / "trade_calendar.csv", index=False, encoding="utf-8-sig")
        print(f"  trade_calendar.csv: {cal.shape}")

        # 3) 行业分类（证监会，当期快照）
        print("[3/4] 行业分类...")
        ind = to_frame(bs.query_stock_industry())
        ind.to_csv(DATA_DIR / "stk_company_info.csv", index=False, encoding="utf-8-sig")
        print(f"  stk_company_info.csv: {ind.shape}")

        # 4) 股票基本资料（上市 / 退市日期）
        print("[4/4] 股票基本资料...")
        basic = to_frame(bs.query_stock_basic())
        # 同 01 脚本，只保留沪深 A 股
        mask = (basic["type"] == "1") & (
            basic["code"].str.startswith("sh.6") | basic["code"].str.startswith("sz.")
        )
        basic = basic.loc[mask].reset_index(drop=True)
        basic.to_csv(DATA_DIR / "sse_stock_basic.csv", index=False, encoding="utf-8-sig")
        print(f"  sse_stock_basic.csv: {basic.shape}")

        print(f"\n全部完成，总用时 {time.time()-t0:.1f}s，目录 {DATA_DIR}")
    finally:
        bs.logout()
