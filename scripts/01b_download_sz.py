# -*- coding: utf-8 -*-
"""数据下载：只下载深市 A 股。

代码与 01a_download_sh.py 对称，两个脚本可独立运行、各自复现。
核心下载逻辑在 _baostock_common.py（沪市、深市、全市场共用，避免重复 100+ 行）。

并发说明：Baostock 对请求频率敏感，4 进程会触发「黑名单」封禁（错误码 10001011），
故默认 **2 进程**。一旦出现黑名单错误，脚本立即熔断退出。已下载文件自动跳过。

用法：
    python 01b_download_sz.py          # 默认 2 进程
    python 01b_download_sz.py 2 0.5    # 2 进程 + 每只额外停 0.5 秒

输出：data/raw/sz.000001.csv ...
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
    stocks = get_stock_list(market="sz")
    bs.logout()  # 父进程只负责取列表，子进程各自登录

    print(f"并行进程数：{workers}，每只间隔：{pause}s")
    download_many(stocks, market_label="SZ", workers=workers,
                  overwrite=OVERWRITE, pause=pause)


if __name__ == "__main__":
    main()
