# -*- coding: utf-8 -*-
"""宽表转换：只转换深市。

读 data/raw/ 下的 sz.*.csv 长表，输出深市宽表到 data/。
与 02a_pivot_sh.py 对称，可独立运行。

全市场一起转直接跑 02_pivot_to_wide.py。
"""
from _pivot_common import load_raw, pivot_market

if __name__ == "__main__":
    long_df = load_raw()
    pivot_market(long_df, "sz")
    print("\n深市宽表转换完成。")
