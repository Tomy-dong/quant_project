# -*- coding: utf-8 -*-
"""Baostock 数据下载公共函数（01a / 01b / 01 共用）。

为保证「沪市 / 深市」两个下载脚本可以独立复现，核心逻辑集中在这里，
01a 和 01b 只负责过滤不同市场的股票代码。
"""

from pathlib import Path
import atexit
import multiprocessing as mp
import time
import baostock as bs
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]  # scripts/ -> 仓库根目录
RAW_DIR = PACKAGE_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Baostock 偶尔返回「网络接收错误 10002007」，全市场下载时必须自动重试
MAX_RETRY = 5
RETRY_SLEEP = 3  # 秒

# 黑名单错误码：请求频率过高会被 Baostock 临时封禁，此时继续请求只会延长封禁
BLACKLIST_CODES = ("10001011",)

# 并行下载进程数。实测 4 进程会触发服务端封禁，默认降到 2；
# 如需临时提高，命令行传参即可（但风险自负）。
DEFAULT_WORKERS = 2

# 股本快照取全部四个季度（Q1/Q2/Q3/Q4），保证送转股等股本变动最及时
SHARE_QUARTERS = (1, 2, 3, 4)


class BlacklistedError(RuntimeError):
    """Baostock 判定请求频率过高、临时封禁（错误码 10001011）。

    遇到此错误必须立即停止整个下载任务，继续请求只会延长封禁时间。
    """


def _is_blacklist(error_code="", error_msg=""):
    msg = str(error_msg or "")
    return (str(error_code) in BLACKLIST_CODES) or ("黑名单" in msg)


def _call(retry_fn, *args, **kwargs):
    """带重试的 Baostock 调用，遇到网络错误自动重试。"""
    last_err = None
    for attempt in range(1, MAX_RETRY + 1):
        try:
            return retry_fn(*args, **kwargs)
        except BlacklistedError:
            raise  # 封禁不重试，直接向上抛，由下载主循环熔断
        except Exception as e:
            last_err = e
            msg = str(e)
            if attempt < MAX_RETRY and (
                "网络" in msg or "10002007" in msg or "WinError" in msg
                or "接收数据异常" in msg or "连接" in msg
            ):
                print(f"    网络异常，第 {attempt} 次重试 ({RETRY_SLEEP}s)...")
                time.sleep(RETRY_SLEEP)
                continue
            raise
    raise last_err

START_DATE = "2015-01-01"
END_DATE = "2025-12-31"
FACTOR_START_DATE = "1990-01-01"  # 复权因子从尽可能早的日期开始

# 日线字段（不复权价格；复权因子走独立接口 query_adjust_factor 拼进来）
K_FIELDS = (
    "date,code,open,high,low,close,preclose,volume,amount,"
    "turn,tradestatus,pctChg,isST,peTTM,pbMRQ"
)

# 补齐模式专用：本地已有 OHLCV 时，日线接口只查估值两字段，省去完整日线下载
VALUATION_FIELDS = "date,peTTM,pbMRQ"

# 外部 OHLCV 长表目录（可选，已有基础行情时可跳过完整日线下载）
EXTERNAL_OHLCV_DIR = PACKAGE_ROOT / "data" / "external_ohlcv"


def to_frame(result):
    """把 BaoStock ResultData 转成 DataFrame。"""
    if result.error_code != "0":
        if _is_blacklist(result.error_code, result.error_msg):
            raise BlacklistedError(f"{result.error_code}: {result.error_msg}")
        raise RuntimeError(f"{result.error_code}: {result.error_msg}")
    rows = []
    while result.next():
        rows.append(result.get_row_data())
    return pd.DataFrame(rows, columns=result.fields)


def get_stock_list(market="all"):
    """取 A 股列表。market = 'all' / 'sh' / 'sz'，排除北交所。"""
    cache_file = PACKAGE_ROOT / "data" / "sse_stock_basic.csv"
    if cache_file.exists():
        basic = pd.read_csv(cache_file, dtype=str).fillna("")
        print(f"使用本地股票清单：{cache_file}", flush=True)
    else:
        print("本地股票清单不存在，正在从 Baostock 获取……", flush=True)
        basic = to_frame(bs.query_stock_basic())
    mask = (basic["type"] == "1") & (
        basic["code"].str.startswith("sh.6") | basic["code"].str.startswith("sz.")
    )
    stocks = basic.loc[mask].copy()
    if market == "sh":
        stocks = stocks[stocks["code"].str.startswith("sh.6")]
    elif market == "sz":
        stocks = stocks[stocks["code"].str.startswith("sz.")]
    stocks = stocks.sort_values("code").reset_index(drop=True)
    n_sh = int(stocks["code"].str.startswith("sh.6").sum())
    n_sz = int(stocks["code"].str.startswith("sz.").sum())
    print(f"待下载 {len(stocks)} 只（沪市 {n_sh}、深市 {n_sz}）")
    return stocks


def download_one_stock(code, overwrite=False, ipo_date="", out_date="", base_ohlcv=None):
    """下载单只股票：不复权日线 + 后复权因子 + 季度股本快照。

    ipo_date / out_date 用于裁剪股本查询年份，避免为上市前的年份白跑接口。
    base_ohlcv: 可选，本地 OHLCV 长表路径。提供时跳过完整日线下载，
                仅以它为行情基础、另查估值两字段补齐，
                可把日线接口耗时从约 15s 降到约 3s。
    输出到 data/raw/{code}.csv，字段：
      date, code, open, high, low, close, preclose, volume, amount,
      turn, tradestatus, pctChg, isST, peTTM, pbMRQ,
      backAdjustFactor, totalShare, liqaShare
    """
    out_file = RAW_DIR / f"{code}.csv"

    if out_file.exists() and not overwrite:
        existing_cols = set(pd.read_csv(out_file, nrows=0).columns)
        if {"backAdjustFactor", "totalShare"}.issubset(existing_cols):
            return "skipped"

    # 1) 行情数据
    if base_ohlcv is not None and Path(base_ohlcv).exists():
        # 补齐模式：读本地 OHLCV，裁剪到回测区间，再补估值两字段
        price = pd.read_csv(base_ohlcv)
        price["date"] = pd.to_datetime(price["date"], format="%Y-%m-%d")
        price = price[(price["date"] >= START_DATE) & (price["date"] <= END_DATE)]
        if price.empty:
            return "empty (skipped, maybe delisted)"
        price = price.sort_values("date").reset_index(drop=True)
        val = _call(lambda: to_frame(bs.query_history_k_data_plus(
            code, VALUATION_FIELDS, start_date=START_DATE, end_date=END_DATE,
            frequency="d", adjustflag="3",
        )))
        if not val.empty:
            val["date"] = pd.to_datetime(val["date"], format="%Y-%m-%d")
            price = price.merge(val, on="date", how="left")
        else:
            price["peTTM"] = pd.NA
            price["pbMRQ"] = pd.NA
    else:
        # 完整下载模式：不复权日线（adjustflag=3 表示不复权）
        price = _call(lambda: to_frame(bs.query_history_k_data_plus(
            code, K_FIELDS, start_date=START_DATE, end_date=END_DATE,
            frequency="d", adjustflag="3",
        )))
        if price.empty:
            return "empty (skipped, maybe delisted)"
        price["date"] = pd.to_datetime(price["date"], format="%Y-%m-%d")

    # 2) 后复权因子（独立接口，merge_asof 按日期前向填充）
    factor = _call(lambda: to_frame(bs.query_adjust_factor(
        code=code, start_date=FACTOR_START_DATE, end_date=END_DATE,
    )))
    if factor.empty:
        price["backAdjustFactor"] = 1.0
    else:
        factor["dividOperateDate"] = pd.to_datetime(
            factor["dividOperateDate"], errors="coerce"
        )
        factor["backAdjustFactor"] = pd.to_numeric(
            factor["backAdjustFactor"], errors="coerce"
        )
        factor = (factor[["dividOperateDate", "backAdjustFactor"]]
                  .dropna().sort_values("dividOperateDate")
                  .drop_duplicates("dividOperateDate", keep="last"))
        price = pd.merge_asof(
            price.sort_values("date"), factor,
            left_on="date", right_on="dividOperateDate", direction="backward",
        ).drop(columns="dividOperateDate")
        price["backAdjustFactor"] = price["backAdjustFactor"].fillna(1.0)

    # 3) 股本（季度更新，用来算市值 = 不复权 close × 股本）
    #    只查该股票上市期间（2015-2025 ∩ [ipo, out]）的四个季度，减少接口调用
    start_year = max(int(START_DATE[:4]), int(str(ipo_date)[:4]) if ipo_date else 0)
    end_year = int(END_DATE[:4])
    if out_date:
        end_year = min(end_year, int(str(out_date)[:4]))
    share_rows = []
    for year in range(start_year, end_year + 1):
        for quarter in SHARE_QUARTERS:
            try:
                df = _call(lambda y=year, q=quarter: to_frame(
                    bs.query_profit_data(code=code, year=y, quarter=q)
                ))
                if not df.empty:
                    share_rows.extend(
                        df[["code", "statDate", "totalShare", "liqaShare"]].to_dict("records")
                    )
            except BlacklistedError:
                raise  # 封禁必须冒泡到主循环熔断，不能被这里吞掉
            except Exception:
                pass
    if share_rows:
        shares = pd.DataFrame(share_rows)
        shares["statDate"] = pd.to_datetime(
            shares["statDate"], format="%Y-%m-%d", errors="coerce"
        )
        # code 列与 price 重复，丢弃避免 merge 后产生 code_x/code_y
        shares = shares.drop(columns=["code"], errors="ignore")
        shares = shares.dropna().sort_values("statDate").drop_duplicates(
            "statDate", keep="last"
        )
        price = pd.merge_asof(
            price.sort_values("date"), shares,
            left_on="date", right_on="statDate", direction="backward",
        ).drop(columns=["statDate"], errors="ignore")
        price["totalShare"] = pd.to_numeric(
            price["totalShare"], errors="coerce"
        ).ffill()
        price["liqaShare"] = pd.to_numeric(
            price["liqaShare"], errors="coerce"
        ).ffill()
    else:
        price["totalShare"] = pd.NA
        price["liqaShare"] = pd.NA

    price["date"] = price["date"].dt.strftime("%Y-%m-%d")
    price = price.sort_values("date").reset_index(drop=True)
    price.to_csv(out_file, index=False, encoding="utf-8-sig")
    return f"{len(price):,} rows"


# ---------------------------------------------------------------- 并行下载
# Baostock 客户端不是线程安全的（单 socket 会话），但多进程各登录一次是安全的。
# 实测单只约 30s，4 进程并发可到 5-7s/只；超过 6 进程服务端会限流。

def _worker_init():
    """子进程初始化：各自登录一次，退出时登出。"""
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"BaoStock login failed: {login.error_msg}")
    atexit.register(bs.logout)


def _worker_task(args):
    code, ipo_date, out_date, overwrite, base_ohlcv = args
    try:
        return code, download_one_stock(
            code, overwrite=overwrite, ipo_date=ipo_date, out_date=out_date,
            base_ohlcv=base_ohlcv,
        ), None, False
    except BlacklistedError as e:
        return code, None, str(e), True
    except Exception as e:
        return code, None, str(e), False


def download_many(stocks, market_label="ALL", workers=DEFAULT_WORKERS,
                  overwrite=False, pause=0.0, base_dir=None):
    """并行下载股票列表，打印进度与预计剩余时间。

    stocks:   get_stock_list() 返回的 DataFrame（含 code / ipoDate / outDate）
    pause:    每只股票下载前额外等待秒数，用于降低请求频率（默认 0）
    base_dir: 可选，本地 OHLCV 长表目录。
              提供后，凡目录里已有 {code}.csv 的股票走「补齐模式」，
              只补估值/复权/股本字段，不重下完整日线。
    熔断:     一旦返回黑名单错误，立即终止本次任务，不再空转剩余股票。
    """
    tasks = []
    for r in stocks.itertuples(index=False):
        base_ohlcv = None
        if base_dir is not None:
            candidate = Path(base_dir) / f"{r.code}.csv"
            if candidate.exists():
                base_ohlcv = candidate
        tasks.append((r.code, r.ipoDate, r.outDate, overwrite, base_ohlcv))
    total = len(tasks)
    t0 = time.time()
    done = 0
    failed = []
    blacklisted = False
    pool = mp.Pool(workers, initializer=_worker_init)
    try:
        for code, status, err, is_black in pool.imap_unordered(_worker_task, tasks):
            if is_black:
                blacklisted = True
                print(f"\n{'='*66}\n[熔断] {code} 返回黑名单错误：{err}")
                print("已立即停止下载，避免延长封禁时间。请等待封禁解除后再重跑本脚本，")
                print("已下载的文件不会重复下载（断点续传）。\n" + "="*66)
                break

            done += 1
            elapsed = time.time() - t0
            eta = elapsed / done * (total - done)
            if err:
                failed.append((code, err))
                print(f"[{market_label} {done:>5}/{total}] {code}: FAILED - {err}")
            else:
                print(f"[{market_label} {done:>5}/{total}] {code}: {status}"
                      f"  ({elapsed/done:.1f}s/只, 剩余约 {eta/3600:.1f}h)")
            if pause:
                time.sleep(pause)
    finally:
        pool.terminate()
        pool.join()

    if blacklisted:
        return failed

    if failed:
        print(f"\n失败 {len(failed)} 只（重跑同一命令会自动补下）：")
        for c, m in failed[:50]:
            print(f"  {c}: {m}")
    print(f"\n{market_label} 下载完成：{total} 只，用时 {time.time()-t0:.0f}s，输出 {RAW_DIR}")
    return failed
