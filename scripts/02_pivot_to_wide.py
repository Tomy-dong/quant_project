# -*- coding: utf-8 -*-
"""宽表转换：全市场（沪市 + 深市）一次性转换。

等价于依次跑 02a_pivot_sh.py + 02b_pivot_sz.py。
读 data/raw/ 长表，输出宽表到 data/，供因子计算与回测读取。

派生字段：
  adj_close  = close × backAdjustFactor   （后复权收盘价）
  total_mv   = 不复权 close × totalShare   （总市值，三因子 + 容量用）
  circ_mv    = 不复权 close × liqaShare    （流通市值）
"""

from _pivot_common import load_raw, pivot_market

if __name__ == "__main__":
    long_df = load_raw()
    pivot_market(long_df, "sh")
    pivot_market(long_df, "sz")
    print("\n全市场宽表转换完成。")
