# -*- coding: utf-8 -*-
"""XGBoost 非线性多因子合成（滚动训练，月频）。

特征：初筛 40 因子（screen_pass），按 screen_pass 的方向对齐为正向预测收益，
      截面已 zscore（zfactors）。
标签：t 月末因子 -> t+1 月收益（与 calc_nav 的 w[t-1]·r[t] 对齐）。
训练：滚动 5 年窗口，每年 OOS 前重训；XGBRegressor + 最后 1 年验证早停。
回测：预测分数排序 Top30 等权，单边千二费率。

输出 results/xgb_strategy/：
    performance.csv   主结果 + 基准超额 + FF3 alpha
    feature_importance.csv  各年特征重要性平均 Top20
    nav_compare.png   策略 vs 沪深300/中证500 净值
"""
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import price_volume_baseline as pv          # noqa: E402
import factor_pool_source as fps            # noqa: E402
import ff3_regression as ff3                # noqa: E402
from data_io import load_benchmark          # noqa: E402
from utils.data_standardize import preprocess  # noqa: E402
from utils.stockselection import stock_selection  # noqa: E402
from utils.utils import calc_nav             # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

RESULT = HERE.parent / "results"
OUT = RESULT / "xgb_strategy"
OUT.mkdir(parents=True, exist_ok=True)
FF3_PATH = RESULT / "ff3" / "ff3_factors.csv"
SCREEN_PATH = RESULT / "factor_screen" / "screen_pass.csv"

TOP_N = 30
COMSN = pv.COMSN
TRAIN_YEARS = 5
OOS_START, OOS_END = 2020, 2025
LABEL_MODE = "raw"    # "rank" 截面排名标签 / "raw" 原始收益（raw 保留幅度信息，实测更优）
N_ESTIMATORS = 120    # 折中：避免早停过激到1轮，又不过拟合到300轮


def metrics(pnl):
    s = pnl.dropna()
    nav = (1 + s).cumprod()
    total = float(nav.iloc[-1] - 1)
    years = len(s) / 12.0
    ann = float((1 + total) ** (1 / years) - 1)
    vol = float(s.std(ddof=1) * np.sqrt(12))
    return {"累计收益": total, "年化收益": ann, "年化波动": vol,
            "夏普比率": float(ann / vol) if vol > 0 else np.nan,
            "最大回撤": float((nav / nav.cummax() - 1).min())}


def backtest(composite, returns):
    dates = composite.index
    codes = composite.columns
    holdings = np.full((len(dates), TOP_N), np.nan)
    for i in range(len(dates)):
        row = composite.iloc[i].dropna()
        if len(row) >= TOP_N:
            sel = row.nlargest(TOP_N).index
            holdings[i] = [codes.get_loc(c) for c in sel]
    weights = stock_selection(codes, holdings, dates)
    return calc_nav(returns.fillna(0), weights, comsn=COMSN)["pnl"]


def bench_monthly(index, dates):
    close = load_benchmark(str(dates.min().date()),
                           str(dates.max().date()), index=index)["close"]
    return close.reindex(dates).pct_change(fill_method=None)


def feat_mat(t, feat, direction, zfactors, codes):
    """t 时点截面特征矩阵 (n_codes, K)，已乘方向。"""
    cols = []
    for f in feat:
        v = zfactors[f].reindex(columns=codes).loc[t].values.astype(np.float32)
        cols.append(v * direction[f])
    return np.column_stack(cols)


def build_train_set(train_dates, next_ret, feat, direction, zfactors, codes):
    """拼接训练窗口内所有截面：X (n, K), y (n,)。"""
    Xs, ys = [], []
    for t in train_dates:
        if t not in next_ret.index:
            continue
        y = next_ret.loc[t].reindex(codes).values.astype(np.float32)
        X = feat_mat(t, feat, direction, zfactors, codes)
        valid = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
        if valid.sum() < 50:
            continue
        yv = y[valid]
        if LABEL_MODE == "rank":
            # 截面排名归一化到 [-0.5, 0.5]，模型学相对排序，抗极值
            yv = (pd.Series(yv).rank(pct=True).values.astype(np.float32) - 0.5)
        Xs.append(X[valid])
        ys.append(yv)
    if not Xs:
        return None, None
    return np.vstack(Xs), np.concatenate(ys)


def main():
    print(f"XGBoost 合成 | Top{TOP_N} | 费率 {COMSN} | 滚动 {TRAIN_YEARS} 年 | "
          f"OOS {OOS_START}-{OOS_END}")
    print("=" * 72)

    # 初筛因子 + 方向
    sp = pd.read_csv(SCREEN_PATH, encoding="utf-8-sig")
    col_fac = "因子" if "因子" in sp.columns else sp.columns[0]
    col_dir = "方向" if "方向" in sp.columns else None
    feat = list(sp[col_fac])
    direction = {r[col_fac]: int(r[col_dir]) if col_dir else 1
                 for _, r in sp.iterrows()}
    print(f"  初筛因子 {len(feat)} 个（方向对齐后全部正向）")

    print("\n载入因子面板（约 2-3 分钟）...")
    monthly, factors, dates = fps.load_pool(mask_valid=True)
    returns = monthly["returns"]
    codes = returns.columns
    zfactors = {name: preprocess(fac) for name, fac in factors.items()}
    # 只用初筛因子，且必须存在于面板
    feat = [f for f in feat if f in zfactors]
    print(f"  面板可用因子 {len(feat)} 个")

    # t 时点的标签 = t+1 期收益（next_ret.loc[t] = returns.loc[t+1]）
    next_ret = returns.shift(-1)

    all_dates = returns.index
    oos_dates = all_dates[(all_dates.year >= OOS_START) & (all_dates.year <= OOS_END)]
    years = sorted(set(oos_dates.year))

    xgb_params = dict(
        n_estimators=N_ESTIMATORS, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_lambda=2.0, reg_alpha=0.5, random_state=42, n_jobs=1,
        tree_method="hist",
    )

    scores = {}  # t -> pd.Series(预测分数, index=codes)
    imp_rows = []
    for y in years:
        tr_end = pd.Timestamp(f"{y-1}-12-31")
        tr_start = pd.Timestamp(f"{y-TRAIN_YEARS}-01-01")
        train_dates = all_dates[(all_dates > tr_start) & (all_dates <= tr_end)]
        # 全训练窗口训练（不早停：相邻年份风格切换会让验证集早停过激）
        X_fit, y_fit = build_train_set(train_dates, next_ret, feat, direction,
                                       zfactors, codes)
        if X_fit is None:
            print(f"  {y}: 训练集为空，跳过"); continue

        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_fit, y_fit, verbose=False)
        imp = dict(zip(feat, model.feature_importances_))
        imp_rows.append({"year": y, **imp})
        print(f"  {y}: 训练 {len(X_fit)} 样本（{LABEL_MODE}标签，{N_ESTIMATORS}轮），"
              f"Top3 因子 {sorted(imp, key=imp.get, reverse=True)[:3]}")

        # 预测 y 年各月
        for t in oos_dates[oos_dates.year == y]:
            if t not in next_ret.index:
                continue
            X = feat_mat(t, feat, direction, zfactors, codes)
            valid = ~np.isnan(X).any(axis=1)
            s = np.full(len(codes), np.nan)
            if valid.any():
                s[valid] = model.predict(X[valid])
            scores[t] = pd.Series(s, index=codes)

    composite = pd.DataFrame(scores).T.sort_index()
    composite.index.name = "日期"
    ret_oos = returns.loc[composite.index]

    pnl = backtest(composite, ret_oos)
    m = metrics(pnl)
    factors_df = pd.read_csv(FF3_PATH, index_col=0, parse_dates=True)
    factors_df.index.name = "日期"
    reg = ff3.regress_one("xgb", pnl, factors_df)

    b300 = bench_monthly("csi300", composite.index)
    b500 = bench_monthly("csi500", composite.index)
    b_ann = {b: (1 + br.dropna()).prod() ** (12 / len(br.dropna())) - 1
             for b, br in [("csi300", b300), ("csi500", b500)]}
    ex300 = m["年化收益"] - b_ann["csi300"]
    ex500 = m["年化收益"] - b_ann["csi500"]

    print("\n" + "=" * 72)
    print(f"XGBoost Top{TOP_N}: 年化 {m['年化收益']*100:.2f}% 夏普 {m['夏普比率']:.3f} "
          f"回撤 {m['最大回撤']*100:.2f}%")
    print(f"  超额沪深300 {ex300*100:+.2f}% | 超额中证500 {ex500*100:+.2f}% | "
          f"FF3 alpha {reg['alpha年化']*100:+.2f}% (t={reg['alpha_t']:.2f}, "
          f"p={reg['alpha_p']:.3f}) R2={reg['R2']:.3f}")

    perf = pd.DataFrame([{
        "策略": "XGBoost合成", "TopN": TOP_N, **m,
        "沪深300年化": b_ann["csi300"], "中证500年化": b_ann["csi500"],
        "超额沪深300": ex300, "超额中证500": ex500,
        "alpha年化": reg["alpha年化"], "alpha_t": reg["alpha_t"],
        "alpha_p": reg["alpha_p"], "R2": reg["R2"],
    }])
    perf.to_csv(OUT / "performance.csv", index=False, encoding="utf-8-sig")

    # 特征重要性汇总
    imp_df = pd.DataFrame(imp_rows).set_index("year")
    avg_imp = imp_df.mean().sort_values(ascending=False)
    avg_imp.to_csv(OUT / "feature_importance.csv", encoding="utf-8-sig",
                   header=["importance"])
    print(f"\n平均特征重要性 Top10（合计占比 {avg_imp.head(10).sum():.1%}）：")
    for f, v in avg_imp.head(10).items():
        print(f"  {f:<18s} {v:.4f}")

    # 净值对比图
    nav_strat = (1 + pnl.fillna(0)).cumprod()
    nav300 = (1 + b300.fillna(0)).cumprod()
    nav500 = (1 + b500.fillna(0)).cumprod()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(nav_strat.index, nav_strat.values, label="XGBoost策略", lw=2, color="crimson")
    ax.plot(nav300.index, nav300.values, label="沪深300", lw=1.3, color="steelblue")
    ax.plot(nav500.index, nav500.values, label="中证500", lw=1.3, color="darkorange")
    ax.set_title(f"XGBoost多因子合成 vs 基准（Top{TOP_N}，{OOS_START}-{OOS_END}）")
    ax.set_ylabel("累计净值"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "nav_compare.png", dpi=150)
    plt.close(fig)

    composite.to_csv(OUT / "composite_xgb.csv", encoding="utf-8-sig")
    print(f"\n结果目录：{OUT}")


if __name__ == "__main__":
    main()
