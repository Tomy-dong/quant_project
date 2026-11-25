# -*- coding: utf-8 -*-
"""Alpha191 运行时依赖：并行执行器（本实现退化为串行）。

原 skill 用线程池并行计算，这里默认串行以控制内存占用；因子计算本身是
pandas 向量化的，串行已足够。

Upstream helper module, reduced to a serial executor to keep memory bounded.
"""
from __future__ import annotations

import traceback


def default_alpha_n_jobs() -> int:
    """默认并行度：串行。Default parallelism: serial."""
    return 1


def compute_engine_alpha_methods(engine, names, n_jobs=None, show_progress=False,
                                 label="Alpha191"):
    """按名称调用 engine 的 alphaNNN 方法，异常或 None 记为 None。

    Call ``engine.alphaNNN`` for each name; failures and ``None`` results are
    recorded as ``None`` so one bad formula does not stop the batch.
    """
    results = {}
    for name in names:
        method_name = str(name).replace("alpha_", "alpha")
        method = getattr(engine, method_name, None)
        if method is None:
            results[name] = None
            continue
        try:
            results[name] = method()
        except Exception:  # noqa: BLE001
            traceback.print_exc(limit=1)
            results[name] = None
    return results
