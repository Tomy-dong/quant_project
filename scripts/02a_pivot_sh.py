# -*- coding: utf-8 -*-
"""宽表转换：只转换沪市。

读 data/raw/ 下的 sh.*.csv 长表，输出沪市宽表到 data/。
与 02b_pivot_sz.py 对称，可独立运行。

全市场一起转直接跑 02_pivot_to_wide.py。
"""
from _pivot_common import load_raw, pivot_market

if __name__ == "__main__":
    long_df = load_raw()
    pivot_market(long_df, "sh")
    print("\n沪市宽表转换完成。")
