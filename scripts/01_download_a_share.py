# -*- coding: utf-8 -*-
"""数据下载：A 股全市场（沪深两市）一次性下载。

等价于依次跑 01a_download_sh.py + 01b_download_sz.py。
沪市、深市可单独重跑，见 01a / 01b。

并发说明：Baostock 对请求频率敏感，4 进程会触发「黑名单」封禁（错误码 10001011），
故默认 **2 进程**。一旦出现黑名单错误，脚本立即熔断退出，不会空转剩余股票，
把封禁时间拖长。已下载文件自动跳过，封禁解除后重跑即可续传。

用法：
    python 01_download_a_share.py          # 默认 2 进程
    python 01_download_a_share.py 2 0.5    # 2 进程 + 每只额外停 0.5 秒（更稳）

输出：data/raw/{code}.csv，每只股票一个长表
字段：date, code, open, high, low, close, preclose, volume, amount,
      turn, tradestatus, pctChg, isST, peTTM, pbMRQ,
      backAdjustFactor, totalShare, liqaShare
"""

import sys
import baostock as bs
from _baostock_common import get_stock_list, download_many, DEFAULT_WORKERS

OVERWRITE = False  # True 强制重下；False 跳过已有完整文件


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_WORKERS
    pause = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    if workers > 2:
        print(f"警告：{workers} 进程有再次被 Baostock 封禁的风险（4 进程已实测被封）。")

    login = bs.login()
    if login.error_code != "0":
        if "黑名单" in (login.error_msg or ""):
            print("登录失败：当前 IP/账号仍处于黑名单，请稍后再试。")
            return
        raise RuntimeError(f"BaoStock login failed: {login.error_msg}")
    stocks = get_stock_list(market="all")
    bs.logout()  # 父进程只负责取列表，子进程各自登录

    print(f"并行进程数：{workers}，每只间隔：{pause}s")
    download_many(stocks, market_label="ALL", workers=workers,
                  overwrite=OVERWRITE, pause=pause)


if __name__ == "__main__":
    main()
