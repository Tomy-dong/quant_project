# -*- coding: utf-8 -*-
"""优化器一（完整版）：带约束的启发式因子筛选 + λ 权重收缩。

与 S7（rolling_top_weighted.py）的区别
--------------------------------------
S7 是固定规则：|ICIR|>=0.2 + 多头门控 + 多头 t 加权。
本脚本把 docs/strategy.md 4.3–4.7 的设计目标全部落地：

1. 候选过滤（4.4）
   - |ICIR| >= 0.2；
   - 多头超额年化 > 0 且多头 t >= 1；
   - 单因子多空年化波动 <= 候选池 70% 分位；
   - 规模暴露 |β| <= 0.30（逐月把因子对 log 流通市值做横截面回归，
     两端都先标准化，斜率即相关系数；在约 3000 只股票的截面上 t 值几乎
     必然显著，因此以经济幅度 β 为准，t 值仅作诊断输出）。
2. 损失函数（4.3）
   L = -a*z(IC_comp) - b*z(ICIR_comp) + g*max|rho|
       + d*|K - K_target| + e*N_flip
   其中 IC_comp / ICIR_comp 用候选池单因子 IC / ICIR 的均值与标准差做
   z-score，避免量纲不一致；max|rho| 为入选因子平均截面相关的最大绝对值。
3. 约束（4.4）
   - 3 <= K <= 20，K_target = 10；
   - 两两相关 <= 0.7（平均截面相关矩阵）；
   - 规模暴露 |β| <= 0.30（候选阶段已过滤）。
4. 启发式搜索（4.5）
   以 |ICIR| 前 10 且通过共线检查的因子为初值，每轮随机执行
   add / remove / swap，只有损失严格下降才接受；连续 40 轮无改进或
   达到 200 轮即停。
5. 降级（4.5）
   合格候选不足 3 个时，退化为 ICIR 过滤 + 共线剔除，并尽量保留 3 个。
6. 权重收缩（4.7）
   omega = (1 - lambda) * omega_ICIR + lambda / K，lambda = 0.3；
   另做 0.2 / 0.5 敏感性，并对同一子集用多头 t 权重做对照。

输出
----
results/optimizer1/
    factor_diagnostics.csv     逐年全部因子的门控与诊断
    factor_selection.csv       逐年入选因子、权重与损失
    selection_summary.csv      逐年 K、损失、最大相关、是否降级
    composite_factor.csv       样本外合成分数
    oos_performance.csv        Top10/50/100 绩效
    sensitivity.csv            λ / 多头 t 权重敏感性
    top10|top50|top100/        绩效、全部结果图、holdings.csv/xlsx
    lambda0.2_top100|lambda0.5_top100|longt_top100/

用法
----
    python baseline/optimizer1_heuristic.py
"""
import os
import sys
import warnings
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib
matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import price_volume_baseline as pv  # noqa: E402
import rolling_multifactor as rm  # noqa: E402
import rolling_top_weighted as rt  # noqa: E402
from data_io import load_benchmark  # noqa: E402
from utils import report_plots as rp  # noqa: E402
from utils.data_standardize import preprocess  # noqa: E402
from utils.stockselection import stock_selection  # noqa: E402
from utils.utils import calc_nav  # noqa: E402

OUTPUT_DIR = HERE.parent / "results" / os.environ.get("OPT1_OUTPUT", "optimizer1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===================== 配置 / Config =====================
TRAIN_YEARS = rm.TRAIN_YEARS            # 5 年训练窗 / 5-year training window
OOS_START_YEAR = rm.OOS_START_YEAR
OOS_END_YEAR = rm.OOS_END_YEAR

# 可通过环境变量覆盖，默认值与旧版保持一致
# Overridable via environment variables; defaults keep the legacy behaviour.
ICIR_MIN = float(os.environ.get("OPT1_ICIR_MIN", 0.2))
LONG_T_MIN = 1.0                        # 多头 t 下限 / min long-side t
VOL_PCT = 0.70                          # 多空波动分位上限 / vol percentile cap
RHO_MAX = 0.7                           # 两两相关上限 / max pairwise correlation
SIZE_EXPO_MAX = 0.30                    # 标准化规模暴露上限 / max |standardised size exposure|

K_MIN = int(os.environ.get("OPT1_K_MIN", 3))
K_MAX = int(os.environ.get("OPT1_K_MAX", 20))
K_TARGET = int(os.environ.get("OPT1_K_TARGET", 10))

ALPHA = 1.0
BETA = 1.0
GAMMA = 0.1
DELTA = 0.1
ETA = 0.05

LAMBDA_MAIN = 0.3
LAMBDA_GRID = (0.2, 0.3, 0.5)
MAX_ITER = 200
PATIENCE = 40
SEED = 42

TOP_N_LIST = tuple(int(x) for x in os.environ.get(
    "OPT1_TOP_N", "10,50,100").split(","))
SENS_TOP_N = int(os.environ.get("OPT1_SENS_TOP_N", TOP_N_LIST[-1]))
N_GROUP = 5

# WorldQuant Alpha101 子集（项目已实现的 21 个）
# WorldQuant Alpha101 subset already implemented in this project.
WQ_FACTORS = {
    "Alpha4", "Alpha6", "Alpha9", "Alpha12", "Alpha13", "Alpha14", "Alpha15",
    "Alpha16", "Alpha18", "Alpha20", "Alpha23", "Alpha26", "Alpha30",
    "Alpha33", "Alpha34", "Alpha35", "Alpha38", "Alpha40", "Alpha43",
    "Alpha53", "Alpha101",
}
FACTOR_POOL = os.environ.get("OPT1_POOL", "all").lower()


def select_pool(names):
    """按配置截取因子池。Slice the factor pool by configuration.

    all      : 全部 38 个因子（旧版行为 / legacy behaviour）
    wq       : 仅 WorldQuant Alpha101
    wq_huatai: WorldQuant + 华泰（用于对比重叠度）
    """
    if FACTOR_POOL == "wq":
        return [n for n in names if n in WQ_FACTORS]
    if FACTOR_POOL == "wq_huatai":
        return [n for n in names if n in WQ_FACTORS or n in {
            "return_1m", "return_3m", "return_6m", "return_12m",
            "wgt_return_1m", "wgt_return_3m", "wgt_return_6m",
            "wgt_return_12m", "exp_wgt_return_3m", "exp_wgt_return_6m",
            "exp_wgt_return_12m", "HAlpha"}]
    return list(names)


# ===================== 基础统计 / Basic statistics =====================
def ann_t(series):
    """年化收益与 t 值。Annualized return and t-stat of a monthly series."""
    s = series.dropna()
    n = len(s)
    if n < 12 or s.std(ddof=1) == 0:
        return np.nan, np.nan
    nav = (1 + s).cumprod()
    ann = nav.iloc[-1] ** (12 / n) - 1
    t = s.mean() / s.std(ddof=1) * np.sqrt(n)
    return ann, t


def ann_vol(series):
    """年化波动率。Annualized volatility of a monthly series."""
    s = series.dropna()
    if len(s) < 12:
        return np.nan
    return s.std(ddof=1) * np.sqrt(12)


def _rank_ic(a, b):
    """两列之间的 Spearman 相关（快速实现）。
    Spearman correlation between two vectors.
    """
    ra = rankdata(a)
    rb = rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else np.nan


# ===================== 规模暴露 / Size exposure =====================
def compute_size_exposure(monthly, zfactors, dates):
    """逐月把因子对 log 流通市值做横截面回归，返回月度标准化暴露与 t 值。

    流通股本 = vol / (turnover_rate / 100)，流通市值 = 不复权 close × 流通股本。
    每月先把因子与 log 市值都做截面标准化，斜率即为相关系数 β；规模暴露取
    月度 β 的均值，t 值仅作诊断（每期约 3000 只股票，t 值几乎必然显著）。

    Monthly cross-sectional regression of each factor on log float market cap.
    Both sides are standardised cross-sectionally each month, so the slope is a
    correlation; the size exposure is the average monthly beta. The t-stat is
    reported for diagnostics only, since it is almost always significant with
    ~3000 stocks per cross-section.
    """
    close = monthly["close"]
    vol = monthly["vol"]
    turn = monthly["turnover_rate"]
    shares = vol / (turn.replace(0, np.nan) / 100.0)
    cap = close * shares
    logcap = np.log(cap.where(cap > 0))

    betas = pd.DataFrame(np.nan, index=dates, columns=list(zfactors))
    for name, z in zfactors.items():
        for dt in dates:
            y = z.loc[dt]
            x = logcap.loc[dt]
            df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
            if len(df) < 100:
                continue
            xv = df["x"].to_numpy(dtype=float)
            yv = df["y"].to_numpy(dtype=float)
            xv = xv - xv.mean()
            yv = yv - yv.mean()
            sx = xv.std(ddof=0)
            sy = yv.std(ddof=0)
            if sx <= 0 or sy <= 0:
                continue
            betas.loc[dt, name] = float((xv * yv).sum() / (len(df) * sx * sy))

    n = betas.notna().sum()
    tstat = betas.mean() / betas.std(ddof=1) * np.sqrt(n)
    return betas, tstat


# ===================== 因子平均截面相关 / Average factor correlation =====================
def compute_corr_mats(zfactors, dates):
    """逐月计算因子间截面相关，返回 {日期: ndarray}。
    Monthly cross-sectional correlation matrices of the z-scored factors.
    """
    names = list(zfactors)
    mats = {}
    for dt in dates:
        df = pd.DataFrame({n: zfactors[n].loc[dt] for n in names})
        mats[dt] = df.corr(min_periods=100).to_numpy()
    return mats, names


def average_corr(mats, dates, year, names):
    """训练窗口内的平均截面相关矩阵，对角线置 0，缺失按 0 处理。
    Average cross-sectional correlation within the training window.
    """
    start = pd.Timestamp(f"{year - TRAIN_YEARS}-01-01")
    end = pd.Timestamp(f"{year - 1}-12-31")
    arrs = [mats[d] for d in dates if (d > start) and (d <= end)]
    if not arrs:
        return np.zeros((len(names), len(names)))
    with np.errstate(invalid="ignore"):
        arr = np.nanmean(np.stack(arrs), axis=0)
    arr = np.nan_to_num(arr, nan=0.0)
    np.fill_diagonal(arr, 0.0)
    return arr


def max_abs_corr(idx, corr):
    """子集内两两相关的最大绝对值。Max absolute pairwise correlation."""
    if len(idx) < 2:
        return 0.0
    sub = corr[np.ix_(idx, idx)]
    return float(np.max(np.abs(sub - np.diag(np.diag(sub)))))


# ===================== 训练窗口诊断 / Training-window diagnostics =====================
def build_diagnostics(ic_series, panels, ls_panel, size_beta, names, year):
    """构造某一年训练窗口内的全部因子诊断与门控结果。
    Build all factor diagnostics and gate flags for one training window.
    """
    start = pd.Timestamp(f"{year - TRAIN_YEARS}-01-01")
    end = pd.Timestamp(f"{year - 1}-12-31")
    mask = lambda s: s.loc[(s.index > start) & (s.index <= end)]

    rows = []
    for name in names:
        ic = mask(ic_series[name]).dropna()
        if len(ic) < 12:
            continue
        std = ic.std(ddof=1)
        mean_ic = ic.mean()
        icir = mean_ic / std if std > 0 else 0.0
        sign = int(np.sign(mean_ic)) or 1

        top, bot = panels[name]
        long_raw = top if sign > 0 else bot
        long_ann, long_t = ann_t(mask(long_raw))

        ls = mask(ls_panel[name])
        vol = ann_vol(ls)

        sb = mask(size_beta[name]).dropna()
        size_expo = float(sb.mean()) if len(sb) >= 12 else np.nan
        size_t = (sb.mean() / sb.std(ddof=1) * np.sqrt(len(sb))
                  if len(sb) >= 12 and sb.std(ddof=1) > 0 else np.nan)

        rows.append({
            "因子": name, "平均IC": mean_ic, "ICIR": icir, "方向": sign,
            "多头年化": long_ann, "多头t": long_t,
            "多空波动": vol, "规模暴露": size_expo, "规模t": size_t,
            "训练月数": len(ic),
        })

    diag = pd.DataFrame(rows)
    if diag.empty:
        raise RuntimeError(f"{year} 年训练窗口内没有可用因子")

    vol_cap = diag["多空波动"].quantile(VOL_PCT)
    gate = ((diag["ICIR"].abs() >= ICIR_MIN) &
            (diag["多头年化"] > 0) &
            (diag["多头t"] >= LONG_T_MIN) &
            (diag["多空波动"] <= vol_cap) &
            (diag["规模暴露"].abs() <= SIZE_EXPO_MAX))
    diag["多空波动上限"] = vol_cap
    diag["门控通过"] = gate
    diag["AbsICIR"] = diag["ICIR"].abs()
    return diag.sort_values("AbsICIR", ascending=False).reset_index(drop=True)


# ===================== 损失函数与启发式搜索 / Loss & local search =====================
class YearProblem:
    """把某一年的训练窗口数据打包，供子集评估复用。
    Pack one year's training-window data for repeated subset evaluation.
    """

    def __init__(self, zfactors, returns, dates, diag, corr, names, year):
        start = pd.Timestamp(f"{year - TRAIN_YEARS}-01-01")
        end = pd.Timestamp(f"{year - 1}-12-31")
        positions = np.where((dates > start) & (dates <= end))[0]
        positions = positions[positions + 1 < len(dates)]

        self.names = names
        self.pos_index = {n: i for i, n in enumerate(names)}
        self.corr = corr
        self.year = year

        zarr = np.stack([zfactors[n].to_numpy(dtype=np.float32)[positions]
                         for n in names])
        self.Z = zarr
        self.R = returns.to_numpy(dtype=np.float32)[positions + 1]

        self.ic_mean = diag.set_index("因子")["平均IC"]
        self.icir = diag.set_index("因子")["ICIR"]
        self.sign = diag.set_index("因子")["方向"]
        self.long_t = diag.set_index("因子")["多头t"]
        self.mu_ic = float(diag["平均IC"].mean())
        self.sd_ic = float(diag["平均IC"].std(ddof=1)) or 1.0
        self.mu_icir = float(diag["ICIR"].mean())
        self.sd_icir = float(diag["ICIR"].std(ddof=1)) or 1.0

        self.diag = diag
        self.cache = {}

    def _weights(self, subset, lam):
        raw = np.abs(self.icir.loc[subset].to_numpy(dtype=float))
        total = raw.sum()
        w_icir = raw / total if total > 0 else np.full(len(subset), 1 / len(subset))
        return (1 - lam) * w_icir + lam / len(subset)

    def evaluate(self, subset, lam=LAMBDA_MAIN):
        """计算子集在训练窗口内的合成 IC、ICIR 与损失。
        Composite IC / ICIR and the loss for a factor subset.
        """
        key = (tuple(sorted(subset)), lam)
        if key in self.cache:
            return self.cache[key]

        subset = list(subset)
        idx = [self.pos_index[n] for n in subset]
        signs = self.sign.loc[subset].to_numpy(dtype=float)
        w = self._weights(subset, lam)

        score = np.tensordot(w * signs, self.Z[idx], axes=(0, 0))
        ics = []
        for i in range(score.shape[0]):
            s = score[i]
            r = self.R[i]
            m = np.isfinite(s) & np.isfinite(r)
            if m.sum() < 100:
                continue
            ics.append(_rank_ic(s[m], r[m]))
        ics = np.asarray([v for v in ics if np.isfinite(v)], dtype=float)

        if len(ics) < 12:
            metrics = {"loss": np.inf, "IC_comp": np.nan, "ICIR_comp": np.nan,
                       "max_rho": np.nan, "K": len(subset), "n_flip": np.nan}
            self.cache[key] = metrics
            return metrics

        ic_mean = float(ics.mean())
        ic_std = float(ics.std(ddof=1))
        icir_comp = ic_mean / ic_std if ic_std > 0 else 0.0
        rho = max_abs_corr(idx, self.corr)
        n_flip = int((signs < 0).sum())

        z_ic = (ic_mean - self.mu_ic) / self.sd_ic
        z_icir = (icir_comp - self.mu_icir) / self.sd_icir
        loss = (-ALPHA * z_ic - BETA * z_icir + GAMMA * rho +
                DELTA * abs(len(subset) - K_TARGET) + ETA * n_flip)

        metrics = {"loss": float(loss), "IC_comp": ic_mean,
                   "ICIR_comp": float(icir_comp), "max_rho": rho,
                   "K": len(subset), "n_flip": n_flip}
        self.cache[key] = metrics
        return metrics


def initial_set(diag, corr, names):
    """按 |ICIR| 降序贪心加入，跳过与已选因子相关性超限的因子。
    Greedy |ICIR| ordering with a collinearity check.
    """
    pos = {n: i for i, n in enumerate(names)}
    chosen = []
    for name in diag.loc[diag["门控通过"], "因子"]:
        if len(chosen) >= K_TARGET:
            break
        idx = pos[name]
        if all(abs(corr[idx, pos[c]]) <= RHO_MAX for c in chosen):
            chosen.append(name)
    return chosen


def degrade_set(diag, corr, names):
    """降级规则：先保留 ICIR 与规模约束，按多头 t 排序，共线剔除。

    严格门控不合格的年份仍要给出一个只做多组合。降级只放宽多头 t / 波动
    门控，保留 |ICIR|>=ICIR_MIN 且 |规模暴露|<=0.30 的因子；宁可少选，
    也不再塞入低 ICIR 因子。数量截断到 K_target，避免权重被稀释。

    Fallback: relax only the long-side / volatility gates, keep the ICIR and
    size constraints, rank by long-side t, remove collinear factors and cap at
    K_target. Prefer fewer factors over padding with low-ICIR names.
    """
    pos = {n: i for i, n in enumerate(names)}
    pool = diag[(diag["ICIR"].abs() >= ICIR_MIN) &
                (diag["规模暴露"].abs() <= SIZE_EXPO_MAX)]
    if len(pool) < K_MIN:
        pool = diag[diag["ICIR"].abs() >= ICIR_MIN]
    if len(pool) == 0:
        # 极端年份至少保留 |ICIR| 最高的 1 个因子，但不再塞入低 ICIR 因子
        # In an extreme year keep at least the highest-|ICIR| factor, but never
        # pad the set with low-ICIR factors.
        pool = diag.nlargest(1, "AbsICIR")
    pool = pool.sort_values(["多头t", "AbsICIR", "因子"],
                            ascending=[False, False, True])

    chosen = []
    for name in pool["因子"]:
        if len(chosen) >= K_TARGET:
            break
        idx = pos[name]
        if all(abs(corr[idx, pos[c]]) <= RHO_MAX for c in chosen):
            chosen.append(name)
    return chosen[:K_TARGET]


def propose(subset, eligible, rng):
    """随机提出一个邻居：增、删或替换。Propose one random neighbor."""
    cur = set(subset)
    outside = sorted(eligible - cur)
    if not outside:
        return None
    r = rng.random()
    if r < 0.4 and len(cur) < K_MAX:
        return tuple(sorted(cur | {outside[rng.integers(len(outside))]}))
    if r < 0.65 and len(cur) > K_MIN:
        return tuple(sorted(cur - {sorted(cur)[rng.integers(len(cur))]}))
    if len(cur) <= K_MAX and cur:
        drop = sorted(cur)[rng.integers(len(cur))]
        return tuple(sorted((cur - {drop}) | {outside[rng.integers(len(outside))]}))
    return None


def local_search(init, problem, eligible, rng):
    """严格下降的局部搜索，连续 PATIENCE 轮无改进即停。
    Strict-improvement local search with an early stop.
    """
    best = tuple(sorted(init))
    best_m = problem.evaluate(best)
    cur, cur_m = best, best_m
    no_improve = 0
    it = 0
    for it in range(1, MAX_ITER + 1):
        cand = propose(cur, eligible, rng)
        if cand is None:
            continue
        if max_abs_corr([problem.pos_index[n] for n in cand], problem.corr) > RHO_MAX:
            no_improve += 1
        else:
            m = problem.evaluate(cand)
            if m["loss"] < cur_m["loss"] - 1e-12:
                cur, cur_m = cand, m
                if m["loss"] < best_m["loss"] - 1e-12:
                    best, best_m = cand, m
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
        if no_improve >= PATIENCE:
            break
    return best, best_m, it


# ===================== 单年求解 / Solve one year =====================
def solve_year(year, diag, corr, names, problem):
    """执行门控 -> 初值 -> 启发式搜索 -> 权重，并记录入选明细。
    Gate, initialise, search and weight one year.
    """
    degraded = False
    eligible = set(diag.loc[diag["门控通过"], "因子"])
    iters = 0

    if len(eligible) >= K_MIN:
        init = initial_set(diag, corr, names)
        if len(init) < K_MIN:
            chosen = degrade_set(diag, corr, names)
            degraded = True
            best = tuple(sorted(chosen))
            best_m = problem.evaluate(best)
        else:
            best, best_m, iters = local_search(
                init, problem, eligible, np.random.default_rng(SEED + year))
    else:
        chosen = degrade_set(diag, corr, names)
        degraded = True
        best = tuple(sorted(chosen))
        best_m = problem.evaluate(best)

    subset = [n for n in names if n in set(best)]
    raw = np.abs(problem.icir.loc[subset].to_numpy(dtype=float))
    w_icir = raw / raw.sum() if raw.sum() > 0 else np.full(len(subset), 1 / len(subset))
    w_final = (1 - LAMBDA_MAIN) * w_icir + LAMBDA_MAIN / len(subset)
    long_t = problem.long_t.loc[subset].to_numpy(dtype=float)
    w_longt = np.clip(long_t, 0, None)

    picked = diag.set_index("因子").loc[subset].copy()
    picked["ICIR权重"] = w_icir
    picked["最终权重"] = w_final
    picked["多头t权重"] = (w_longt / w_longt.sum() if w_longt.sum() > 0
                          else np.full(len(subset), 1 / len(subset)))
    picked = picked.sort_values("最终权重", ascending=False).reset_index()
    picked.insert(0, "样本外年份", year)

    return {
        "year": year, "train": f"{year - TRAIN_YEARS}-{year - 1}",
        "picked": picked, "subset": subset,
        "weights": pd.Series(w_final, index=subset),
        "longt_weights": pd.Series(picked.set_index("因子")["多头t权重"]),
        "signs": problem.sign.loc[subset],
        "metrics": best_m, "degraded": degraded, "iters": iters,
    }


# ===================== 回测与导出 / Backtest & exports =====================
def make_bench_nav(index):
    """基准净值，与策略同索引。Benchmark NAV aligned to the strategy index."""
    bench = load_benchmark(pv.START, pv.END)["close"].reindex(index)
    return (1 + bench.pct_change(fill_method=None).fillna(0)).cumprod()


def export_holdings(weights, out_dir):
    """导出每月持仓：长表 CSV + 宽表 CSV + Excel。
    Export monthly holdings as long/wide CSV and Excel.
    """
    long_rows, wide = [], {}
    for dt, row in weights.iterrows():
        nz = row[row > 0].sort_values(ascending=False)
        if nz.empty:
            continue
        wide[dt] = nz.index.tolist()
        for code, w in nz.items():
            long_rows.append({"调仓日": dt.strftime("%Y-%m-%d"),
                              "股票代码": code, "权重": round(float(w), 6)})

    long_df = pd.DataFrame(long_rows)
    wide_df = pd.DataFrame.from_dict(wide, orient="index")
    wide_df.columns = [f"第{i + 1}只" for i in range(wide_df.shape[1])]
    wide_df.index.name = "调仓日"

    long_df.to_csv(out_dir / "holdings.csv", index=False, encoding="utf-8-sig")
    wide_df.to_csv(out_dir / "holdings_wide.csv", encoding="utf-8-sig")
    with pd.ExcelWriter(out_dir / "holdings.xlsx") as xw:
        long_df.to_excel(xw, sheet_name="持仓明细", index=False)
        wide_df.to_excel(xw, sheet_name="持仓宽表")
    return long_df, wide_df


def run_backtest(composite, returns, dates, top_n, out_dir, title,
                 ic=None, grp=None):
    """Top N 等权回测，输出绩效、全部结果图与持仓表。
    Equal-weight Top N backtest with the full figure set and holdings.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    holdings = np.full((len(dates), top_n), np.nan)
    for i in range(len(dates)):
        row = composite.iloc[i].dropna()
        if len(row) >= top_n:
            codes = row.nlargest(top_n).index
            holdings[i] = [composite.columns.get_loc(c) for c in codes]

    weights = stock_selection(returns.columns, holdings, dates)
    outputs = calc_nav(returns.fillna(0), weights, comsn=pv.COMSN)

    nav = outputs["nav"]
    pnl = outputs["pnl"]
    years = (nav.index[-1] - nav.index[0]).days / 365.24
    ann_ret = nav.iloc[-1] ** (1 / years) - 1
    ann_vol_ = pnl.std() * np.sqrt(12)
    perf = pd.Series({
        "累计收益": nav.iloc[-1] - 1,
        "年化收益": ann_ret,
        "年化波动": ann_vol_,
        "夏普比率": ann_ret / ann_vol_ if ann_vol_ > 0 else 0,
        "最大回撤": (nav / nav.cummax() - 1).min(),
        "年化换手": outputs["turnover"].mean() * 12,
    })
    perf.to_csv(out_dir / "performance.csv", encoding="utf-8-sig")

    export_holdings(weights, out_dir)
    rp.plot_all(outputs, make_bench_nav(composite.index), out_dir, title,
                ic=ic, grp=grp)
    return outputs, perf, weights


# ===================== 主流程 / Main flow =====================
def main():
    print(f"优化器一完整版：约束 + 启发式搜索 + λ 收缩 | 样本外 "
          f"{OOS_START_YEAR}-{OOS_END_YEAR}")
    print("=" * 72)

    print("\n[1/6] 载入数据、计算因子与预处理（约数分钟）...")
    daily = pv.load_market_data(pv.MARKET, pv.START, pv.END)
    monthly = pv.build_monthly(daily)
    dates = monthly["dates"]
    returns = monthly["returns"]
    valid = pv.filter_pool(monthly)
    print(f"  可交易股票池均值：{valid.sum(axis=1).mean():.0f} 只")
    factors = pv.compute_price_volume_factors(monthly, valid)
    zfactors = {name: preprocess(fac) for name, fac in factors.items()}
    names = select_pool(list(zfactors))
    zfactors = {n: zfactors[n] for n in names}
    print(f"  因子池：{FACTOR_POOL}，{len(names)} 个因子，"
          f"ICIR 下限 {ICIR_MIN:g}，K 目标 {K_TARGET}")

    print("[2/6] 预计算 IC、多空/多头超额、规模暴露与相关矩阵...")
    ic_series = rm.rank_ic_panel(zfactors, returns)
    panels = rt.excess_panels(zfactors, returns)
    ls_panel = {n: panels[n][0] - panels[n][1] for n in names}
    size_beta, _ = compute_size_exposure(monthly, zfactors, dates)
    corr_mats, _ = compute_corr_mats(zfactors, dates)
    print(f"  因子数 {len(names)}，月末样本 {len(dates)} 期")

    print("[3/6] 逐年训练：门控 -> 启发式搜索 -> λ 收缩...")
    fits, diag_records, summary_records = {}, [], []
    for year in range(OOS_START_YEAR, OOS_END_YEAR + 1):
        diag = build_diagnostics(ic_series, panels, ls_panel, size_beta,
                                 names, year)
        n_icir = int((diag["ICIR"].abs() >= ICIR_MIN).sum())
        n_long = int(((diag["多头年化"] > 0) & (diag["多头t"] >= LONG_T_MIN)).sum())
        n_vol = int((diag["多空波动"] <= diag["多空波动上限"]).sum())
        n_size = int((diag["规模暴露"].abs() <= SIZE_EXPO_MAX).sum())
        n_gate = int(diag["门控通过"].sum())
        print(f"\n  {year - TRAIN_YEARS}-{year - 1} 门控：ICIR {n_icir} | "
              f"多头 {n_long} | 波动 {n_vol} | 规模 {n_size} | 全部通过 {n_gate}")
        corr = average_corr(corr_mats, dates, year, names)
        problem = YearProblem(zfactors, returns, dates, diag, corr, names, year)
        fit = solve_year(year, diag, corr, names, problem)
        fits[year] = fit

        print(f"\n[{fit['train']}] -> {year}：K={len(fit['subset'])}"
              f"{'（降级）' if fit['degraded'] else ''}，"
              f"迭代 {fit['iters']}，损失 {fit['metrics']['loss']:.3f}")
        for _, r in fit["picked"].iterrows():
            print(f"  {r['因子']:<20s} 权重 {r['最终权重']:.3f} "
                  f"方向 {int(r['方向']):+d} ICIR {r['ICIR']:+.2f} "
                  f"多头t {r['多头t']:.2f}")

        rec = diag.copy()
        rec.insert(0, "样本外年份", year)
        rec.insert(1, "训练窗口", fit["train"])
        rec["是否入选"] = rec["因子"].isin(fit["subset"])
        diag_records.append(rec)

        summary_records.append({
            "样本外年份": year, "训练窗口": fit["train"],
            "K": len(fit["subset"]), "降级": fit["degraded"],
            "搜索迭代": fit["iters"], "损失": fit["metrics"]["loss"],
            "IC_comp": fit["metrics"]["IC_comp"],
            "ICIR_comp": fit["metrics"]["ICIR_comp"],
            "max_rho": fit["metrics"]["max_rho"],
            "N_flip": fit["metrics"]["n_flip"],
        })

    diag_df = pd.concat(diag_records, ignore_index=True)
    diag_df.to_csv(OUTPUT_DIR / "factor_diagnostics.csv",
                   index=False, encoding="utf-8-sig")

    selection = pd.concat([f["picked"] for f in fits.values()], ignore_index=True)
    keep = ["样本外年份", "因子", "平均IC", "ICIR", "方向", "多头年化", "多头t",
            "多空波动", "规模暴露", "规模t", "ICIR权重", "最终权重", "多头t权重"]
    selection[keep].to_csv(OUTPUT_DIR / "factor_selection.csv",
                           index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(summary_records)
    summary.to_csv(OUTPUT_DIR / "selection_summary.csv",
                   index=False, encoding="utf-8-sig")

    print("\n[4/6] 合成样本外分数（主配置 λ=0.3）...")
    comps = [rm.build_composite(zfactors, fits[y], y)
             for y in range(OOS_START_YEAR, OOS_END_YEAR + 1)]
    composite = pd.concat(comps).sort_index()
    oos_start = pd.Timestamp(f"{OOS_START_YEAR}-01-01")
    oos_end = pd.Timestamp(f"{OOS_END_YEAR}-12-31")
    composite = composite.loc[(composite.index >= oos_start) &
                              (composite.index <= oos_end)]
    composite.to_csv(OUTPUT_DIR / "composite_factor.csv", encoding="utf-8-sig")

    ret_oos = returns.loc[composite.index]
    ic_oos = rm.rank_ic_panel({"合成因子": composite}, ret_oos)["合成因子"]
    grp = rp.group_returns(composite, ret_oos, N_GROUP)
    print(f"  样本外 {len(composite)} 期，合成因子 IC 均值 {ic_oos.mean():.3f}")

    print(f"\n[5/6] 样本外回测 Top{'/'.join(map(str, TOP_N_LIST))}"
          "（含持仓导出与全部结果图）...")
    rows = {}
    for tn in TOP_N_LIST:
        out_dir = OUTPUT_DIR / f"top{tn}"
        _, perf, _ = run_backtest(
            composite, ret_oos, composite.index, tn, out_dir,
            f"优化器一 启发式 Top{tn}", ic=ic_oos, grp=grp)
        rows[f"top{tn}"] = perf
        print(f">>> top{tn}: 累计 {perf['累计收益'] * 100:.2f}% | "
              f"夏普 {perf['夏普比率']:.3f} | 回撤 {perf['最大回撤'] * 100:.2f}%")

    cmp = pd.DataFrame(rows).T
    cmp.index.name = "持仓数"
    cmp.to_csv(OUTPUT_DIR / "oos_performance.csv", encoding="utf-8-sig")

    print("\n[6/6] 权重敏感性：λ=0.2 / 0.3 / 0.5 与多头 t 权重（同一子集）...")
    main_key = f"top{TOP_N_LIST[-1]}"
    sens = {f"lambda0.3_{main_key}": cmp.loc[main_key]}
    variants = [("lambda0.2", 0.2, None), ("lambda0.5", 0.5, None),
                ("longt", None, "longt")]
    for tag, lam, mode in variants:
        comps_v = []
        for y in range(OOS_START_YEAR, OOS_END_YEAR + 1):
            fit = fits[y]
            subset = fit["subset"]
            if mode == "longt":
                w = fit["longt_weights"].reindex(subset).to_numpy()
            else:
                raw = np.abs(fit["picked"].set_index("因子").loc[subset, "ICIR"]
                             .to_numpy(dtype=float))
                w_icir = raw / raw.sum() if raw.sum() > 0 else np.full(len(subset), 1 / len(subset))
                w = (1 - lam) * w_icir + lam / len(subset)
            fit_v = dict(fit)
            fit_v["weights"] = pd.Series(w, index=subset)
            comps_v.append(rm.build_composite(zfactors, fit_v, y))
        comp_v = pd.concat(comps_v).sort_index()
        comp_v = comp_v.loc[(comp_v.index >= oos_start) & (comp_v.index <= oos_end)]
        ret_v = returns.loc[comp_v.index]
        out_dir = OUTPUT_DIR / f"{tag}_top{SENS_TOP_N}"
        _, perf_v, _ = run_backtest(
            comp_v, ret_v, comp_v.index, SENS_TOP_N, out_dir,
            f"优化器一 {tag} Top{SENS_TOP_N}")
        sens[f"{tag}_top{SENS_TOP_N}"] = perf_v
        print(f">>> {tag}: 累计 {perf_v['累计收益'] * 100:.2f}% | "
              f"夏普 {perf_v['夏普比率']:.3f}")

    sens_df = pd.DataFrame(sens).T
    sens_df.index.name = "配置"
    sens_df.to_csv(OUTPUT_DIR / "sensitivity.csv", encoding="utf-8-sig")

    # ===================== 汇总打印 / Summary =====================
    print("\n" + "=" * 72)
    print("样本外绩效（2020-2025）")
    print("=" * 72)
    show = cmp[["累计收益", "年化收益", "年化波动", "夏普比率",
                "最大回撤", "年化换手"]].apply(pd.to_numeric)
    for c in ["累计收益", "年化收益", "年化波动", "最大回撤"]:
        show[c] = (show[c] * 100).round(2)
    show[["夏普比率", "年化换手"]] = show[["夏普比率", "年化换手"]].round(3)
    print(show.to_string())

    print(f"\nλ / 权重敏感性（Top{TOP_N_LIST[-1]}）：")
    ss = sens_df[["累计收益", "年化收益", "夏普比率", "最大回撤"]].apply(pd.to_numeric)
    for c in ["累计收益", "年化收益", "最大回撤"]:
        ss[c] = (ss[c] * 100).round(2)
    ss["夏普比率"] = ss["夏普比率"].round(3)
    print(ss.to_string())

    s7_path = HERE.parent / "results" / "rolling_top" / "oos_performance.csv"
    if s7_path.exists():
        s7 = pd.read_csv(s7_path, index_col=0)
        print(f"\n与 S7 对照（{main_key}）：")
        rows_cmp = [("优化器一", cmp.loc[main_key])]
        if main_key in s7.index:
            rows_cmp.append(("S7", s7.loc[main_key]))
        for label, row in rows_cmp:
            print(f"  {label:<8s} 累计 {row['累计收益'] * 100:6.2f}% | "
                  f"年化 {row['年化收益'] * 100:5.2f}% | "
                  f"夏普 {row['夏普比率']:.3f} | 回撤 {row['最大回撤'] * 100:6.2f}%")

    print("\n逐年 K 与损失：")
    print(summary.round(4).to_string(index=False))
    print(f"\n结果已保存到：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
