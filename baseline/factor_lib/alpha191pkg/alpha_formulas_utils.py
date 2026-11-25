# -*- coding: utf-8 -*-
"""Alpha191 运行时依赖：算子模块。

原实现 ``alpha191_formulas.py`` 通过 ``from .alpha_formulas_utils import ...``
引用算子；这里统一转发到 :mod:`factor_lib.ops`，保持公式零改动。

Operator module required by the upstream Alpha191 implementation. Everything
is re-exported from :mod:`factor_lib.ops`.
"""
from ..ops import (  # noqa: F401
    absv, adv, corr, count, cov, decay_linear, delay, delta, ema, emax, emin,
    highday, log, lowday, product, rank, regbeta, returns, row_max, row_min,
    safe_div, scale, sequence, sign, sumif, ts_argmax, ts_argmin, ts_max,
    ts_mean, ts_min, ts_rank, ts_std, ts_sum, wma,
)

# Alpha191 里的 Sma 是中国式 EMA（非简单移动平均）
# Alpha191's Sma is the Chinese EMA, not a simple moving average.
sma = ema
