# A-Share Multi-Factor Stock Selection and Market-Neutral Strategy

## 1. Overview

The strategy uses pure price-volume factors as its source of alpha. At the portfolio level, it has two branches—active management and market-neutral—which share the same factor calculations, factor tests, and composite results:

- **Active management**: long-only, retains benchmark beta, uses the CSI 300 Price Index as the benchmark, and seeks relative excess returns;
- **Market-neutral**: combines a long portfolio with short stock index futures to reduce net beta to approximately zero, retaining only factor returns.

The two branches differ only in portfolio construction and hedging, not in their factors. Factor efficacy must therefore be tested independently before the branches diverge; otherwise, it is impossible to determine whether portfolio returns come from the factors or from beta exposure.

The repository currently implements the active-management branch's data pipeline, factor tests, single-factor baseline, and rolling multi-factor model (annual factor rotation + monthly stock rotation), as well as a partial implementation of Optimizer 1 (long-side gating + long-side t weighting, designated S7; see Section 11). The full heuristic search for Optimizer 1, $λ$ weight shrinkage, and the market-neutral branch remain incomplete.

The strategy uses only back-adjusted OHLCV, turnover rates, and trading value, with no financial statement data. Factors can therefore be recomputed at each month-end using market data known at that time, with no look-ahead problem caused by financial-report disclosure lags.

Core settings:

| Item | Setting |
| --- | --- |
| Stock universe | Shanghai and Shenzhen A-shares (excluding the Beijing Stock Exchange) |
| Backtest period | 2015-01-01 ~ 2025-12-31 |
| Training window | 5 years, rolled annually |
| Out-of-sample period | 2020-2025, 6 years in total |
| Factor rotation frequency | Annual, on the first rebalancing date of each year |
| Stock rotation / rebalancing frequency | Monthly, on the last trading day of each month |
| Benchmark | CSI 300 Price Index (sh.000300) |
| Transaction cost | 0.002 one-way |
| Position direction | Long-only |
| Number of holdings per period | Baseline: 10 stocks; S7 comparisons: 10 / 50 / 100 stocks, all equally weighted, $w_{i} = \frac{1}{N}$ |

The development sequence is to complete the single-factor baseline first, then add multi-factor combination, factor selection, and capital-allocation optimization. Each step is compared with the preceding step in the same environment and retained only after its incremental contribution is confirmed. Single-factor baseline results are available in `results/baseline/`.

## 2. Data and Preprocessing

### 2.1 Data Sources

Daily market data, adjustment factors, quarterly share capital, and auxiliary data all come from Baostock:

| Data | API | Purpose |
| --- | --- | --- |
| Unadjusted daily OHLCV, turn, pctChg, isST | `query_history_k_data_plus` | Price-volume factors |
| peTTM, pbMRQ | `query_history_k_data_plus` | Valuation grouping and neutralization |
| backAdjustFactor | `query_adjust_factor` | Back-adjusted prices |
| totalShare, liqaShare | `query_profit_data` (four quarters) | Market capitalization |
| ipoDate, outDate | `query_stock_basic` | Exclusion of recently listed stocks |
| Industry classification | `query_stock_industry` | Industry neutralization |
| CSI 300 | `sh.000300` | Benchmark |

### 2.2 Back-Adjusted Prices

The entire strategy maintains only one price series, with factor calculations, return accounting, and net asset value calculations all using the same convention:

$$
P^{adj}_{i,t} = close_{i,t} \times f_{i,t}
$$

Here, $f_{i,t}$ is the back-adjustment factor returned by Baostock. Back adjustment uses the start of the sample as its base and adjusts forward from there; historical values do not change when new data are added, making rolling backtests reproducible.

Forward-adjusted and back-adjusted prices produce the same two-day returns over the same period, so there is no need to maintain a separate forward-adjusted series. What must be avoided is mixing the two price series.

Market capitalization must be calculated using the **unadjusted** closing price; otherwise, it will be inflated by the adjustment factor:

$$
MV_{i,t} = close_{i,t} \times totalShare_{i,t}
$$

### 2.3 Individual Stock Volatility

No forecasting models such as EWMA or GARCH are used. Instead, individual stock volatility is the rolling standard deviation of back-adjusted daily returns:

$$
σ_{i,t} = \sqrt{252} \times std\left(r_{i,t-19}, \dots, r_{i,t}\right)
$$

Both 20-day and 60-day windows are retained for capital allocation and risk constraints, respectively.

### 2.4 Monthly Panel

The last trading day of each month is selected to sample the daily data into a monthly panel, which provides the rebalancing decision points. Factor values are calculated at month-end, positions are held for the following month, and returns are calculated over that following month.

### 2.5 Stock Universe

At each month-end, stocks are filtered according to the following conditions:

1. Not ST (`is_st = 0`);
2. Traded on that day (`volume > 0`);
3. Listed for at least 120 trading days, excluding recently listed stocks;
4. Positive trading value (`amount > 0`, used only to confirm that the stock is not suspended);
5. ADV no lower than the 20th percentile of the current cross-section.

Condition 4 does not impose a capital-size threshold: the strategy has no initial capital setting, the backtest calculates net asset value from weights, and ADV serves only as a liquidity and capacity constraint, independent of capital size.

Capacity screening is moved forward to the data-cleaning stage rather than evaluated separately after the backtest. At each month-end, average trading value over the preceding 60 trading days is calculated as:

$$
ADV_{i,t} = \frac{1}{60}\sum_{k=1}^{60} amount_{i,t-k}
$$

Stocks whose $ADV$ falls in the bottom 20% of the current cross-section are removed. This eliminates capacity bottlenecks before factor testing and stock selection, preventing illiquid stocks from contributing spurious excess returns.

## 3. Factor Framework

### 3.1 Factor Sources

Candidate factors come from two categories of public price-volume factors:

- Huatai Financial Engineering price-volume factors: momentum, turnover-weighted momentum, exponentially decaying turnover-weighted momentum, and HAlpha;
- WorldQuant Alpha101 and the World_Quant_Alphas reproduction library: formulas related to momentum and reversal selected ex ante on the basis of economic logic, without exhaustively implementing all 101 formulas.

Both categories depend only on back-adjusted OHLC, volume, trading value, and turnover rate. The vwap and adv fields required by Alpha101 can be derived directly from local data.

**Window convention**: The time windows in the original Alpha101 formulas are measured in trading days. This project implements them on the monthly panel by shifting in months (1 month ≈ 21 trading days). This is a monthly approximation, not a day-by-day reproduction of the original formulas. The window units for this group of factors are uniformly “months”; daily and monthly conventions are not mixed.

#### Factor Lookback Windows

“Window” has three distinct meanings in this strategy, with different lengths and update frequencies that must not be conflated:

| Level | Length | Purpose | Update frequency |
| --- | --- | --- | --- |
| Factor lookback window | 1–32 months | Historical period used to calculate each factor observation; window length is fixed | Recalculated monthly; window unchanged |
| Training window | 5 years | Estimate IC / ICIR, select factors, and determine weights | Rolled once per year |
| Rebalancing period | 1 month | Rotate stocks and reallocate individual stock weights | Each month-end |

Factor lookback windows by factor family are as follows:

| Factor family | Lookback window (months) | Description |
| --- | --- | --- |
| `return_Nm` | 1 / 3 / 6 / 12 | Simple interval returns |
| `wgt_return_Nm` | 1 / 3 / 6 / 12 | Return window N + average turnover window N |
| `exp_wgt_return_Nm` | 3 / 6 / 12 | Exponentially decaying turnover weights, with decay coefficient $e^{-k/(N/4)}$ |
| `HAlpha` | 12 | Intercept from a rolling regression; output requires a full 12 observations |
| Selected `Alpha101` | 1–32 | Short end: 1–3 (Alpha12/14/15/20/33/34); medium end: 5–10 (Alpha4/6/9/13/16/18/26/38/40/53); long end: 16–32 (Alpha30/35/43) |
| Price-volume extensions | 3 / 6 | `volatility_3m/6m`, `turnover_mean_3m`, `turnover_std_3m`, `amihud_3m` |

The window rules are fixed as follows:

1. All windows are measured in months. No daily windows are used, avoiding a mixture of two time conventions on the monthly panel;
2. Windows are not extrapolated when data are insufficient: a factor value is produced only from the earliest month permitted by `min_periods`, and the corresponding return must also be available. Missing values are directly set to `NaN` and excluded from cross-sectional ranking for that period, preventing factors with incomplete windows from entering stock selection;
3. Window length is part of a factor's definition. Annual factor rotation changes only “which factors to use and their weights,” not the lookback length of individual factors; otherwise, cross-year comparisons would lose meaning.

Because the longest long-end factors require 32 months and `HAlpha` requires 12 months, 2015 data are used mainly to warm up the factors. Accordingly, the training window starts in 2015 and the out-of-sample period starts in 2020.

### 3.2 Implemented Factors

The baseline implements 38 price-volume factors; see `baseline/price_volume_baseline.py`:

| Category | Count | Factors |
| --- | ---: | --- |
| Interval returns | 4 | `return_1m` / `3m` / `6m` / `12m` |
| Turnover-weighted returns | 4 | `wgt_return_1m` / `3m` / `6m` / `12m` |
| Exponentially decaying weights | 3 | `exp_wgt_return_3m` / `6m` / `12m` |
| Benchmark regression | 1 | `HAlpha` |
| Selected Alpha101 | 21 | `Alpha4` / `6` / `9` / `12` / `13` / `14` / `15` / `16` / `18` / `20` / `23` / `26` / `30` / `33` / `34` / `35` / `38` / `40` / `43` / `53` / `101` |
| Price-volume extensions | 5 | `volatility_3m` / `6m`, `turnover_mean_3m`, `turnover_std_3m`, `amihud_3m` |

The candidate pool contains 38 factors, within the 30–40 range. `exp_wgt_return_1m` and `wgt_return_1m` have identical degenerate weights when N=1 and have therefore been deduplicated. The screening rule is `|ICIR| ≥ 0.2`, followed by truncation to 20 factors in descending order of |ICIR|.

### 3.3 Factor Preprocessing

Three cross-sectional steps are performed in the following fixed order:

1. MAD winsorization: centered on the median, values beyond median ± 3 × 1.4826 × MAD are clipped;
2. Industry neutralization: subtract the cross-sectional mean of stocks in the same industry for the current period, equivalent to taking residuals from a regression on industry dummy variables; stocks without an industry classification are assigned to the same group;
3. Z-score standardization: subtract the cross-sectional mean and divide by the cross-sectional standard deviation.

Neutralization precedes standardization to ensure consistent units when factors are combined. The implementation is in `utils/data_standardize.py`, and industry classifications come from Baostock's CSRC industry snapshot. Factor tests (IC, grouping, and long-short) and factor combination use the same preprocessed factors, eliminating the methodological split in which “raw factors are used for selection while neutralized factors are used for portfolio construction.”

$$
z_{i,k} = \frac{x_{i,k} - μ_k}{σ_k}
$$

### 3.4 Single-Factor Tests: IC and ICIR

Factor testing is a prerequisite for the entire strategy. Its purpose is to determine, before trading, whether each factor predicts next-period returns, whether its direction is positive or negative, and whether its predictions are stable. **IC is a correlation statistic, not a backtest; it involves neither trading nor costs.**

#### IC (Information Coefficient)

At each month-end $t$, within the investable universe for that period, the cross-sectional correlation between factor values $f_{i,t}$ and individual stock returns in the next period $r_{i,t+1}$ is calculated:

$$
IC_t = corr\left(f_{i,t},\ r_{i,t+1}\right)
$$

- $IC_t$ ranges from $[-1, 1]$; a larger absolute value indicates stronger explanatory power for next-period returns;
- The sign indicates factor direction: $IC > 0$ means larger factor values are associated with higher returns (positive direction), while $IC < 0$ means larger factor values are associated with lower returns (negative direction; the factor must be reversed when combined);
- In practice, a mean $|IC|$ of 0.05 is usable, while a value above 0.10 indicates a very strong factor.

To reduce the effects of extreme values and nonlinearity, **RankIC** is also calculated: both factor values and returns are converted into cross-sectional ranks before calculating correlation, using the same formula as above. RankIC is more robust to outliers and is the primary basis for factor selection.

#### ICIR (Information Coefficient Information Ratio)

Treating monthly IC over the full sample (or training window) as a time series, ICIR measures the stability of mean IC relative to its volatility:

$$
ICIR = \frac{\overline{IC}}{std(IC)} = \frac{\frac{1}{T}\sum_{t=1}^{T} IC_t}{\sqrt{\frac{1}{T-1}\sum_{t=1}^{T}(IC_t - \overline{IC})^2}}
$$

- $\mathrm{ICIR}$ accounts for both predictive power (numerator) and stability (denominator), making it a more robust measure of factor quality than $\overline{\mathrm{IC}}$ alone;
- The basic threshold for factor inclusion is:

$$
| \mathrm{ICIR}_k | \geq 0.2
$$

- The original rolling model normalized $| \mathrm{ICIR}_k |$ to determine factor weights. S7 uses ICIR only as a preliminary screening threshold; its final weights instead use the $t$-statistics of long-side excess returns. The two approaches must not be conflated.

### 3.5 Single-Factor Long-Short Diagnostics and Long-Side Excess Returns

In addition to IC/ICIR, the single-factor tests output the IC hit rate, quintile returns, monotonicity, annual performance, and turnover, as well as the long-short and long-side excess-return diagnostics described in this section. **These diagnostics determine whether factor returns come from the long side or the short side; they do not imply that the actual current portfolio takes short positions.**

At each month-end, stocks are divided into quintiles according to their preprocessed factor scores, with $Q_{1}$ the lowest group and $Q_{5}$ the highest. Factor direction is determined by the sign of mean RankIC. The static experiment `optimizer_factor_selection.py` uses full-sample RankIC, whereas `rolling_top_weighted.py` uses only RankIC from the current year's training window:

$$
s_k = \mathrm{sign}\left(\overline{\mathrm{RankIC}}_k\right) \in \{-1,+1\}
$$

The direction-adjusted single-factor long-short return is:

$$
R^{LS}_{k,t+1}
= s_k\left(\overline r_{Q_{5},t+1}-\overline r_{Q_{1},t+1}\right)
$$

A long-only portfolio cannot use the short leg, so the excess return of the long group relative to the equal-weighted return of the entire current universe is also calculated. If $s_k=+1$, the long group is $Q_{5}$; if $s_k=-1$, the long group is $Q_{1}$:

$$
Q^L_{k,t}=
\begin{cases}
Q_{5}, & s_k=+1,\\
Q_{1}, & s_k=-1,
\end{cases}
\qquad
R^L_{k,t+1}
= \overline r_{Q^L_{k,t},t+1}-\overline r_{𝓤_t,t+1}
$$

The $t$-statistic of long-side excess returns is calculated as follows; both S7's gating and its weights are derived from it:

$$
t^L_k = \frac{\overline R^L_k}{\mathrm{std}(R^L_k)}\sqrt{T}
$$

The long-short diagnostics do not deduct actual securities-borrowing costs or consider the availability of shares to borrow; they are used only for factor research. The current backtest portfolio is always long-only and holds Top N stocks at equal weights.

For this strategy, **the long-short diagnostic asks whether the factor works across both tails, while long-side excess return asks whether the tail the strategy can actually buy outperforms the universe**. Suppose a reversal factor has negative mean RankIC, so $s_k=-1$ and its investable long group is the original $Q_{1}$. If $Q_{1}$ earns $1\%$ next month, $Q_{5}$ earns $-8\%$, and the equal-weighted universe earns $5\%$, the direction-adjusted long-short return is $1\%-(-8\%)=9\%$, but long-side excess return is only $1\%-5\%=-4\%$. The factor looks strong on a long-short basis, yet all of that strength depends on declines in $Q_{5}$. A long-only portfolio cannot capture that short-leg return. If this relationship persists over the sample, S7's long-side gate will reject the factor.

In the static script, S1 adds “annualized long-short return $>0$ and long-short $t\geq2$” to the basic ICIR screen, and S2 additionally requires “annualized long-side return $>0$ and long-side $t>0$.” S4/S7 do not use long-short significance as a selection gate; they instead require “annualized long-side return $>0$ and long-side $t>1$.” S7 differs from S4 only in factor weighting. In other words, long-short diagnostics support factor research and controlled comparisons; they are not short-selling rules in the current S7 portfolio.

## 4. Optimizer 1: Factor Selection and Combination

### 4.1 Role and Target

Optimizer 1 optimizes **factors**, answering “which factors to use and how much weight to assign to each.” Its output is a selected factor set and corresponding factor weights, which are then used to combine individual stock expected-return scores. It is independent of the subsequent stock-selection and capital-allocation layers: these are three separate optimization problems with different variables and objectives:

| Optimizer | Optimization target | Question answered |
| --- | --- | --- |
| Optimizer 1 (this chapter) | Factors | Which factors to use and how much weight to assign to each |
| Stock selection (5.1) | Stocks | Which stocks to buy |
| Optimizer 2 (5.2) | Individual stock weights | How much capital to allocate to each stock |

The candidate factor pool is controlled at 30–40 factors, with no more than 20 ultimately selected, to avoid collinearity and overfitting caused by stacking weak factors.

### 4.2 Inputs and Outputs

- **Inputs**: the factor exposure matrix $Z$ (stocks × time × factors) after winsorization, standardization, and industry neutralization; monthly IC/ICIR and factor return series for each factor over the training window; and current benchmark returns;
- **Outputs**: the selected factor set $𝓚$ (containing $K=|𝓚|$ factors) and the weight $ω_k$ of each selected factor in the composite score.

### 4.3 Target Loss Function (Not Yet Implemented)

The target version optimizes the predictive power and stability of the composite factor, with penalty terms for collinearity, factor count, and direction reversals. **The current S7 does not numerically solve this loss function.**

$$
𝓛=-α\overline{\mathrm{IC}}_{\mathrm{composite}}-β\mathrm{ICIR}_{\mathrm{composite}}+γ\max_{i\lt j}|ρ_{ij}|+δ| K-K_{\mathrm{target}}|+η N_{\mathrm{flip}}
$$

The composite factor's IC in period $t$ is:

$$
\mathrm{IC}_{\mathrm{composite},t}
= \mathrm{corr}\left(S_{i,t},r_{i,t+1}\right),
\qquad
S_{i,t}=\sum_{k\in𝓚}ω_k
\mathrm{sign}(\mathrm{IC}_k)z_{i,k,t}
$$

The terms have the following meanings:

| Term | Purpose | Tuning direction |
| --- | --- | --- |
| $-\overline{\mathrm{IC}}_{\mathrm{composite}}$ | Higher mean IC of the composite factor is better (negated for inclusion in the minimization) | $α > 0$ |
| $-\mathrm{ICIR}_{\mathrm{composite}}$ | More stable composite-factor IC is better | $β > 0$ |
| $\max_{i\lt j}|ρ_{ij}|$ | Maximum pairwise correlation among distinct selected factors, penalizing collinearity; excludes self-correlation with $i=j$ | $γ > 0$ |
| $| K-K_{\mathrm{target}}|$ | Penalty for deviation of the factor count from its target (e.g., $K_{\mathrm{target}}=10$) | $δ > 0$ |
| $N_{\mathrm{flip}}$ | Number of factors requiring direction reversal, penalizing excessive reversals | $η \geq 0$ |

### 4.4 Target Constraints and Parameter Values (Mostly Not Yet Implemented)

The target version of Optimizer 1 plans to search factor sets subject to the following four constraints:

1. Number of selected factors $K\geq3$; too few factors cannot provide diversification;
2. Historical return volatility of each individual factor must not exceed a specified threshold, removing unstable factors with excessive volatility;
3. Exposures to style, size, and other factors must not exceed upper limits (Active Management V1 has no industry exposure constraint);
4. Pairwise factor correlations must not exceed a threshold; among collinear factors, only the one with higher ICIR is retained.

Specific values for constraints and penalty terms are determined by “using data-driven initial values and out-of-sample sensitivity analysis for final values”:

| Constraint / parameter | Recommended value | Basis |
| --- | --- | --- |
| Number of selected factors $K_{\mathrm{target}}$ | 10 (lower bound 3, upper bound 20) | $K<5$ provides insufficient diversification; $K>20$ makes collinearity difficult to control and produces diminishing marginal IC |
| Annualized volatility cap on single-factor long-short returns | 70th percentile of the candidate pool (approximately 25% for A-share price-volume factors) | Uses a percentile of the candidate-pool distribution to remove unstable tail factors |
| Pairwise correlation cap $\max_{i\lt j}|ρ_{ij}|$ | 0.7 | Common practice in the literature: $|ρ|>0.7$ indicates high collinearity and $>0.8$ severe collinearity |
| Style / size exposure cap | $| t|<2$ after neutralization | Uses cross-sectional regressions to test exposure significance |
| Predictive-power and stability weights $α,β$ | 1, 1 | Standardize the two terms to comparable scales first |
| Collinearity / count / reversal penalties $γ,δ,η$ | Start at 0.1–0.5 | Enable one at a time and use out-of-sample performance to assess marginal contributions |

The values follow three principles:

1. **Distributional thresholds use candidate-pool statistics**: the volatility cap is a candidate-pool percentile rather than an arbitrary value;
2. **Structural thresholds follow common practice in the literature**: collinearity at 0.7 and exposure at $| t|<2$ are commonly used standards;
3. **Final values rely on out-of-sample sensitivity analysis**: compare out-of-sample Sharpe ratios and maximum drawdowns, and choose the midpoint of a stable region rather than the best-performing point, avoiding overfitted spikes. The target grid is:

$$
K_{\mathrm{target}}\in\{5,10,15,20\},
\qquad
ρ_{\max}\in\{0.6,0.7,0.8\}
$$

The five penalty coefficients $α,β,γ,δ,η$ should not all be tuned simultaneously. The target process should first use preliminary ICIR screening and collinearity removal as the baseline, then enable constraints one at a time. The currently implemented basic ICIR threshold alone is:

$$
|\mathrm{ICIR}_k|\geq0.2
$$

**The correspondence between the current code and the above principles is as follows; design goals must not be treated as implemented features. The scripts must be distinguished here: `optimizer_factor_selection.py` is a full-sample static comparison of rules and contains no rolling retraining, whereas `rolling_top_weighted.py` integrates S7 into a 5-year training window and annual out-of-sample validation:**

| Principle / component | Current implementation status |
| --- | --- |
| Preliminary ICIR screening | Implemented: $|\mathrm{ICIR}_k|\geq0.2$; static experiments additionally compare 0.3 / 0.4 |
| Factor-count control | Partially implemented: the static script caps selection at 20 but has no minimum-of-3 constraint; the rolling version also caps selection at 20 and falls back to the top 3 by long-side $t$ when fewer than 3 qualify, relaxing the gates. Neither uses $K_{\mathrm{target}}=10$ or the $δ$ count penalty |
| 70th-percentile volatility cap | Not implemented |
| Collinearity threshold of 0.7 and $0.6/0.7/0.8$ grid | Not implemented; there is currently no collinearity removal |
| Style / size exposure $| t|<2$ | Not implemented; industry demeaning is not equivalent to an ex post exposure constraint |
| Full loss function and heuristic search | Not implemented; S7 uses fixed-rule screening and weighting |
| $K_{\mathrm{target}}$ grid | Not implemented; the existing comparisons concern ICIR thresholds and Top N holding counts, which are different |
| 5-year training window and annual rolling out-of-sample validation | Implemented in `rolling_top_weighted.py`, not in `optimizer_factor_selection.py` |

### 4.5 Target Solution Method (Not Yet Implemented)

Factor subset selection is a combinatorial optimization problem. The target version plans to use heuristic iteration:

1. **Initialization**: select the top $K_{\mathrm{target}}$ factors in descending order of $|\mathrm{ICIR}|$ over the training window as the initial set;
2. **Perturbation**: in each round, randomly attempt to add or remove one factor, or replace one factor, then recalculate the loss function;
3. **Acceptance criterion**: accept the new set if its loss is lower; otherwise, accept it with a certain probability to escape local optima;
4. **Termination**: stop when the iteration limit is reached or no improvement occurs for several consecutive rounds, then output the best factor set.

If the target solver fails or candidate factors are insufficient, the plan is to fall back to ICIR filtering and collinearity removal while retaining at least 3 factors. **The current code does not implement this full fallback process: the baseline has only ICIR filtering and count truncation, while S7 adds long-side gating and long-side $t$ weighting on top of that; neither includes collinearity removal.**

### 4.6 Current Factor Weights and Composite Score

The current code has three explicit factor-weighting schemes: equal weights for the static baseline, ICIR weights for the original rolling model, and long-side $t$ weights for S7:

$$
ω_k^{EW}=\frac{1}{K},
\qquad
ω_k^{ICIR}
=\frac{|\mathrm{ICIR}_k|}
{\sum_{j\in𝓚}|\mathrm{ICIR}_j|},
\qquad
ω_k^{S7}
=\frac{t^L_k}{\sum_{j\in𝓚}t^L_j}
$$

All three schemes share the same direction-adjustment and combination formulas:

$$
S_{i,t}
=\sum_{k\in𝓚}ω_k s_k z_{i,k,t},
\qquad
s_k=\mathrm{sign}\left(\overline{\mathrm{RankIC}}_k\right)
$$

Factors with $s_k=-1$ are first reversed so that the composite direction is consistently “higher score, higher expected return.” $S_{i,t}$ is an individual stock ranking score, not an unbiased numerical forecast of return. All weights are determined solely by training-window statistics; no AR, VAR, EWMA, GARCH, LSTM, or other models are used to forecast factor returns.

Therefore, **S7 is not a tool specifically for testing single-factor IC**: IC/RankIC are upstream diagnostics and direction inputs, while S7's task is to select multiple factors, allocate factor weights, and generate a multi-factor composite score.

### 4.7 Regularization and Overfitting Control

Current factor weights are not fitted by regression, and classical Ridge / Lasso has not yet been used. Overfitting control requires distinguishing implemented structural constraints from target-version weight shrinkage.

**Layer 1: Structural constraints (partially implemented)**

The candidate pool is fixed at 38 factors, the selected count is limited to 3–20, and annual rolling out-of-sample validation reduces overfitting. Collinearity removal and the $δ| K-K_{\mathrm{target}}|$ count penalty in the target version have not yet been implemented. These structural constraints act similarly to sparse regularization, but do not amount to actually solving an $L_1$ optimization problem.

**Layer 2: Target-version weight shrinkage (not yet implemented)**

Pure ICIR weights may concentrate on a small number of factors with coincidentally high in-sample ICIR. The target version plans to shrink weights toward equal weights:

$$
ω_k^{\mathrm{reg}}
=(1-λ)ω_k^{ICIR}+λ\frac{1}{K},
\qquad λ\in[0,1]
$$

$λ=0$ reduces to pure ICIR weights, while $λ=1$ reduces to full equal weights. The objective function consistent with the above equation is:

$$
\min_{ω}\;
(1-λ)\sum_{k\in𝓚}
\left(ω_k-ω_k^{ICIR}\right)^2
+λ\sum_{k\in𝓚}
\left(ω_k-\frac{1}{K}\right)^2
$$

The first-order condition with respect to $ω_k$ yields the shrunk weights in the preceding equation. The larger $λ$ is, the closer the weights are to equal weights, but this does not directly guarantee lower turnover. The target suggested value is 0.3, with sensitivity tests planned at 0.2 / 0.3 / 0.5; the current S7 does not yet use $λ$.

**Layer 3: Rolling retraining + out-of-sample validation (implemented)**

The current S7 determines factor sets, directions, and weights using only the training window; the test year does not participate in that year's factor fitting (see Section 4.8). Once the target version is implemented, $λ$ must likewise be selected only within the training window.

If factor weights are instead fitted directly by cross-sectional regression in the future, i.e., $\min_ω\Vert r-Zω\Vert^2$, standard Ridge or Lasso would then be used, choosing either that approach or ICIR / long-side $t$ normalization rather than stacking two sources of weights.

### 4.8 Rolling Retraining

Factor sets and weights are updated on an annual rolling basis, with a 5-year training window and a 1-year step:

| Training window | Out of sample |
| --- | --- |
| 2015–2019 | 2020 |
| 2016–2020 | 2021 |
| 2017–2021 | 2022 |
| 2018–2022 | 2023 |
| 2019–2023 | 2024 |
| 2020–2024 | 2025 |

In each round, only data within the current training window are used to repeat factor testing, selection, and weighting; the test year is entirely excluded.

### 4.9 Monthly Stock Rotation and Annual Factor Rotation

The strategy has only two decision frequencies, each responsible for one task:

| Level | Frequency | What changes | Basis |
| --- | --- | --- | --- |
| Factor layer | Annual | Factor set $𝓚$ and weights $ω_k$ | IC / ICIR over the preceding 5 years; S7 additionally uses long-side excess returns |
| Portfolio layer | Monthly | The N stocks held and individual stock weights | Current factor values × fixed weights for the year |

The specific rules are:

1. **Annual factor rotation**: on the first rebalancing date of each year, repeat factor testing using the preceding 5 years, reselect the factor set, and recalculate weights and directions. The original rolling model uses ICIR weights, while S7 uses long-side $t$ weights, all held fixed within the year;
2. **Monthly stock rotation**: at each month-end, multiply current factor values by the fixed weights for the year to obtain $S_i$, rerank, and select the top N stocks; all current versions use equal weights;
3. Within a year, therefore, “only stocks change, not factors”; both factors and stocks change only at the cross-year transition point;
4. The transition point uses only data before the transition date, and the test year does not participate in factor selection, consistent with Section 4.8.

Taking the two out-of-sample years 2020 and 2021 as examples:

| Time point | Factor set and weights | Stock selection and rebalancing |
| --- | --- | --- |
| End of 2019-12 | Complete retraining using 2015–2019 | — |
| End of 2020-01 | Switch to the factor set trained on 2015–2019 | Score using the new factor set and establish positions |
| End of 2020-02 ~ 2020-12 | Remain unchanged | Only recalculate scores and rotate stocks each month |
| End of 2021-01 | Switch after retraining on 2016–2020 | Score using the new factor set and rotate stocks |
| End of 2021-02 ~ 2021-12 | Remain unchanged | Only recalculate scores and rotate stocks each month |

The two layers are separated because IC / ICIR are low-frequency statistics: monthly factor reselection would treat noise as signal and concentrate trading at year-end, increasing turnover. In contrast, price and volume changes at the individual stock level are monthly and require monthly stock rotation. Factors determine “what to watch,” while monthly data determine “whom to select.”

Current implementation status: both frequency layers have been implemented. The script is `baseline/rolling_multifactor.py`, with outputs in `results/rolling/`.

- Annual retraining: 6 folds (2015–2019→2020, 2016–2020→2021 … 2020–2024→2025). Each fold estimates ICIR using only RankIC within the training window, screens at $|\mathrm{ICIR}|\geq0.2$, applies an upper limit of 20 and a lower limit of 3 factors, normalizes $|\mathrm{ICIR}|$ for weights, and takes the sign of mean IC for direction. Each fold selects 20 factors, with different factor sets and weights each year; see `factor_selection.csv`;
- Monthly stock rotation: at each month-end, recalculate $S_i$ using the factor set and weights determined for that year, rerank, and hold the top 10 stocks at equal weights;
- Out-of-sample 72 periods (2020-01-23 ~ 2025-12-31): cumulative return $-6.40\%$, annualized return $-1.11\%$, annualized volatility $26.38\%$, Sharpe ratio $-0.042$, maximum drawdown $-41.31\%$, and annualized turnover $10.68$x; over the same period, the CSI 300 returned $+38.82\%$ cumulatively, with a Sharpe ratio of $0.274$ and maximum drawdown of $-36.96\%$;
- The baseline script `price_volume_baseline.py` retains one-time full-sample factor selection as a static control group. After adding capacity constraints (bottom 20% ADV percentile and at least 120 trading days since listing) and unifying testing and combination on the same neutralized factors, the baseline over the full period (2015–2025) has cumulative return $-28.62\%$, annualized return $-3.04\%$, Sharpe ratio $-0.090$, and maximum drawdown $-70.58\%$, underperforming the benchmark's $+32.54\%$ over the same period. The rolling version covers only out-of-sample 2020–2025; because these periods differ, their numerical results cannot be compared directly.

## 5. Portfolio Construction

Portfolio construction proceeds serially in two steps: first determine the holdings list from the composite score (stock selection), then determine individual stock weights among the selected stocks (Optimizer 2). The two steps have independent objectives and are not merged into a single optimization problem.

### 5.1 Stock Selection: Determining Which Stocks to Buy

The sole input to stock selection is the individual stock expected-return ranking score $S_i$ output by Optimizer 1.

Monthly rebalancing rotates stocks only, not factors: the factor set and weights used during the year are determined at the annual transition point as described in Sections 4.8 / 4.9 and remain unchanged within the year. At each month-end, only current factor values and $S_i$ are recalculated. Within the current investable universe (already filtered according to Section 2.5: non-ST, not suspended, listed for at least 120 trading days, and meeting the ADV requirement):

1. Calculate the current composite score $S_i$ for each stock;
2. Sort by $S_i$ from high to low and select the top N stocks as the current holdings list;
3. Output the N stock codes and then determine weights; all current backtests use equal weights, $w_{i} = \frac{1}{N}$.

**Top N refers to the number of holdings, not the number of factors or the rolling-window length.** The baseline uses N=10; the full-sample scan compares N=10 / 20 / 50 / 100, and the S7 rolling out-of-sample test compares N=10 / 50 / 100. A smaller N makes the portfolio more concentrated and increases single-stock impact; a larger N provides more diversification but dilutes marginal signals.

### 5.2 Optimizer 2: Capital Weight Allocation (How Much to Allocate to Each Stock)

Optimizer 2 optimizes **the weights of selected stocks**, answering “how much capital to allocate to each of the N stocks.” The target plan is to use the eight weight-allocation methods from the course notes to generate individual stock weights separately within the fixed list provided by stock selection, comparing them using the same training window, covariance matrix, and out-of-sample period. This optimizer has not yet been implemented; the current S7 Top 50 / Top 100 configurations are also only equally weighted and do not constitute Optimizer 2.

#### Inputs and Constraints

- **Inputs**: back-adjusted daily return series for the N selected stocks, individual historical rolling volatility (Section 2.3), and the covariance matrix $\Sigma$ estimated over the training window;
- **Constraints**: fully invested $\sum_iw_i=1$ and long-only $w_i\geq0$; the original Top 10 design additionally imposes a 20% single-stock weight cap, i.e., no more than twice the equal weight.

#### Eight Methods

| Method | Weight rule | Description |
| --- | --- | --- |
| EW equal weight | $w_i = 1/N$ | Baseline method |
| IV inverse volatility | $w_i \propto 1/σ_i$ | Allocates more to lower-volatility stocks |
| IVar inverse variance | $w_i \propto 1/σ_i^2$ | More sensitive to volatility differences |
| GMVP global minimum variance | $\min w^\top \Sigma w$ | Minimizes portfolio variance |
| ERC equal risk contribution | Equal risk contribution across stocks, $RRC_i = 1/N$ | Uses the full covariance matrix |
| RB risk budgeting | $RRC_i = b_i$, with $b_i$ set by market capitalization or inverse volatility | ERC is the special case $b_i=1/N$ |
| MDP maximum diversification | Maximize the diversification ratio (weighted average volatility / portfolio volatility) | Converted into quadratic programming |
| MDCP maximum decorrelation | $\min w^\top C w$, where $C$ is the correlation matrix | Reduces to equal weights when $C=I$ |

ERC / RB are solved through risk-budgeting convex optimization:

$$
\min_x \ \tfrac{1}{2} x^\top \Sigma x - \sum_i b_i \log x_i,
\qquad b_i = \frac{1}{N} \ (\text{ERC})
$$

After solving, normalize $w = x / \mathbf{1}^\top x$. Constrained methods (GMVP, MDP, MDCP, RB) are all solved using `scipy.optimize.minimize` (SLSQP), and all methods share the same training-period covariance matrix to ensure comparability.

#### Selection Criteria and V1 Conventions

The selection criterion is fixed before reviewing results: over the same out-of-sample period, the method with the **highest terminal net asset value and the lowest maximum drawdown** wins.

For simplicity in the current V1 stage, all configurations use equal weights (EW), $w_{i} = \frac{1}{N}$; out-of-sample comparison of the eight methods remains to be implemented. When IV / GMVP / ERC are introduced later, the upper-level stock-selection, backtest, and attribution interfaces will remain unchanged; only the weight-generation function will be replaced.

## 6. Backtest Settings

### 6.1 Transaction Costs

The one-way fee rate is 0.002 and is deducted only during net asset value accounting:

$$
C_t = 0.002 \times TO_t
$$

### 6.2 Position Drift

No trades occur between rebalancing dates, and weights drift naturally with prices. Weights before trading in the next period are:

$$
w^-_{i,t} = \frac{w_{i,t-1}(1 + r_{i,t})}{1 + \sum_j w_{j,t-1} r_{j,t}}
$$

Trading volume and one-way turnover are calculated from post-drift weights rather than by directly subtracting target weights across two periods:

$$
\Delta w_{i,t} = w^*_{i,t} - w^-_{i,t}, \qquad
TO_t = \frac{1}{2}\sum_i |\Delta w_{i,t}|
$$

For initial position establishment, $w^- = 0$, and costs are charged on the entire portfolio. Directly subtracting the preceding period's target weights would mistakenly treat these passive weight changes as active trading, artificially inflating both turnover and costs.

### 6.3 Benchmark

The benchmark is the CSI 300 Price Index and excludes dividend reinvestment; individual stocks use back-adjusted returns and include dividend reinvestment. This is a relatively conservative convention and does not overstate excess returns.

### 6.4 Timing of Factor and Stock Rotation

The last trading day of each month is a rebalancing date, with additional factor retraining on the first rebalancing date of each year:

| Time point | Action |
| --- | --- |
| First rebalancing date of each year (2020-01, 2021-01 … 2025-01) | First retrain the factor set and weights using the preceding 5 years, then calculate current scores, then rotate stocks |
| All other month-ends | Retain the factor set and weights for the year; only recalculate scores and rotate stocks |
| Every rebalancing date | Calculate turnover and costs using post-drift weights as described in Section 6.2 |

The execution order is fixed: retrain factors → calculate current factor values → combine into $S_i$ → select stocks → allocate weights → settle costs. Retraining uses only data before the rebalancing date; market data from the current period or later are not permitted.

## 7. Return Attribution

At each month-end, standardized factor exposures are used in a cross-sectional regression on individual stock returns for the following month:

$$
r_{i,t+1} = α_t + \sum_k β_{k,t} f_{i,k,t} + ε_{i,t}
$$

Holding weights are used to obtain portfolio exposures, and monthly portfolio returns are then decomposed into an intercept term, individual factor contributions, and a residual:

$$
r^p_{t+1} = α_t + \sum_k \left(\sum_i w_{i,t} f_{i,k,t}\right)β_{k,t} + \sum_i w_{i,t}ε_{i,t}
$$

The mean and cumulative value of each component are aggregated over the full sample to determine which factors contribute most of the excess return and whether the residual share is excessive.

## 8. Controlled Experiments

All control groups use the same stock universe, fee rate, rebalancing rules, and benchmark as the main strategy.

1. Single-factor baseline: 38 price-volume factors screened under the descending |ICIR| rule, combined at equal weights, with Top 10 stocks held at equal weights;
2. Static multi-factor: fixed factor pool + ICIR weighting;
3. Rolling multi-factor: annual reselection of factors and weights (implemented; see `results/rolling/`);
4. Capital-allocation comparison: pairwise comparison of the eight methods and comparison with the equal-weight baseline;
5. Fama-French three-factor model: construct SMB and HML through cross-sorting by total market capitalization and BP, testing whether strategy excess returns are merely style exposures.

The three-factor regression takes the form:

$$
R^p_t - R^f_t = α + β_{MKT} MKT_t + β_{SMB} SMB_t + β_{HML} HML_t + ε_t
$$

If $α$ is significantly positive, the price-volume factors provide incremental returns that cannot be explained by the three factors.

## 9. Data Conventions and Known Limitations

1. **Benchmark excludes dividends**: Baostock provides only the CSI 300 Price Index; queries for the total-return index return empty. The benchmark excludes dividends while individual stocks include them, making excess returns conservative.
2. **Survivorship bias**: Baostock's daily-data API returns empty results for delisted stocks. Approximately 241 stocks delisted during 2015–2025 cannot be restored, so the actual backtest universe consists of stocks still listed, making returns slightly optimistic.
3. **Industry classification is a current snapshot**: no historical industry changes are available, causing slight biases in industry neutralization for earlier years.
4. **Share capital is a quarterly snapshot**: `totalShare` / `liqaShare` are updated quarterly and can reflect bonus shares and stock splits, but cannot be precise to the day.
5. **Beijing Stock Exchange excluded**: Baostock does not support the Beijing Stock Exchange; “all A-shares” in this document means Shanghai + Shenzhen.
6. **Index constituent weights unavailable**: $w^b$ in benchmark-relative optimization is approximated by equal weights across the tradable universe and can later be replaced with actual constituent weights.

## 10. Current Progress

| Module | Status |
| --- | --- |
| Data download and wide-table conversion | Complete: 5,368 stocks across Shanghai and Shenzhen (average of 3,783 tradable stocks at month-end) |
| Back-adjusted prices, monthly panel, and stock universe | Complete |
| 38 price-volume factors | Complete |
| Single-factor tests (IC, grouping, long-short) | Complete; long-short diagnostics in `results/baseline/longshort.csv` |
| Single-factor baseline backtest | Complete; results in `results/baseline/` |
| Monthly stock-rotation mechanism (month-end Top N reranking, equal weights) | Complete; baseline Top 10, S7 comparisons Top 10 / 50 / 100 |
| Annual factor-rotation mechanism (5-year training window, 1-year roll) | Complete; see `results/rolling/` |
| Optimizer 1: factor selection and weighting | Partially implemented: long-side gating + long-side t weighting (S7; see Section 11); full heuristic search and $λ$ shrinkage remain to be developed |
| S7 rolling out-of-sample validation | Complete; see `results/rolling_top/` |
| Optimizer 2: eight capital-weight allocation methods | To be developed; V1 uses equal weights |
| Return attribution (capacity × convention 2×2 controls) | Complete; see Section 11 and `results/ablation/` |
| Fama-French three-factor comparison | To be developed |

Baseline result interpretation: 25 of the 38 factors have positive annualized long-short returns with $t>2$, and the median annualized long-short return is 7.6%, indicating that the signals themselves are effective. However, the long-only portfolio holding 10 equally weighted stocks per period has a cumulative return of $-28.62\%$ and annualized return of $-3.04\%$ over the full period, underperforming the CSI 300's $+32.54\%$. Effective long-short results but ineffective long-side results indicate that the bottleneck lies in the long-only structure, holding concentration, and turnover, rather than factor direction. Moreover, after adding the ADV 20th-percentile and minimum-120-trading-day listing constraints, the excess returns that previously relied on illiquid small-cap stocks no longer appear, showing that the original results cannot be replicated at capacity.

## 11. Attribution and Optimization Experiments

### 11.1 Attribution of the Return Reversal: Capacity Filtering × Convention Unification 2×2 Controls

Previously, after two modifications were introduced simultaneously, the baseline changed from $+91.13\%$ to $-28.62\%$. To distinguish the primary cause, a 2×2 control experiment was run on the same data (`baseline/ablation_attribution.py`). The legacy configuration exactly reproduces $+91.13\%$, confirming the experiment's credibility:

| Configuration | Capacity filter | IC convention | Cumulative return | Annualized | Sharpe | Maximum drawdown |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| A current | On | Unified | $-28.62\%$ | $-3.04\%$ | $-0.090$ | $-70.58\%$ |
| C | On | Split | $+0.04\%$ | $+0.00\%$ | $0.000$ | $-57.84\%$ |
| B | Off | Unified | $+116.52\%$ | $+7.33\%$ | $0.200$ | $-49.11\%$ |
| D legacy | Off | Split | $+91.13\%$ | $+6.11\%$ | $0.169$ | $-48.34\%$ |

With the convention fixed, switching capacity filtering reduces returns by 91–145 percentage points; with capacity fixed, switching the convention changes returns by only $\pm25$–$29$ percentage points. **Conclusion: capacity filtering (ADV 20th percentile + listed for at least 120 days) is the overwhelming primary cause of the return reversal**, while convention unification is a second-order effect. Nearly all of the legacy version's $+91.13\%$ excess return came from illiquid small-cap and recently listed stocks. In real trading, these securities are constrained by price limits, suspensions, and market-impact costs, so their paper returns cannot be realized.

### 11.2 Partial Implementation of Optimizer 1: Long-Side Gating + Long-Side t Weighting (S7)

Selection and weighting rules are compared within the tradable universe (`baseline/optimizer_factor_selection.py`). S0–S6 are screening-rule controls. **S7 is merely an additional experiment label, defined as “S4's pure long-side gated factor set + long-side $t$ weighting”**; it is not general-purpose numerical optimization software, nor is it a seventh-generation full optimizer.

Diagnostics show that all top-20 factors have positive annualized long-short returns and $t$-statistics greater than 2, so the long-short gates in S1–S3 do not remove any further factors. However, annualized long-side excess returns are only 0.8%–4.7%, far below long-short returns of 6%–16.5%, indicating that many factors rely more heavily on the short leg. A long-only strategy should examine the long side directly, so S7 no longer requires long-short significance and instead uses:

$$
|\mathrm{ICIR}_k|\geq0.2,
\qquad
\mathrm{Ann}\left(R^L_k\right)>0,
\qquad
t^L_k\geq1
$$

The sample used for these statistics depends on the implementation: the static scan uses the full sample and is therefore an in-sample comparison, whereas the rolling version uses only the preceding 5 years for each annual fit. Both implementations use $ω_k^{S7}$ from Section 4.6 for factor weights and retain at most 20 factors. The static scan truncates by $|\mathrm{ICIR}|$ and specifies the long-side $t$ condition as $>1$; the rolling version ranks and truncates by long-side $t$-statistic, specifies the gate as $\geq1$, and additionally includes a fallback rule that selects the top 3 factors by long-side $t$ when fewer than 3 factors qualify.

Configuration × holding-count comparison (full sample in-sample, cumulative return / Sharpe ratio):

| Configuration | Top 10 | Top 20 | Top 50 | Top 100 |
| --- | --- | --- | --- | --- |
| S0 equal weight (basic ICIR screen) | $-28.6\%$ / $-0.09$ | $-6.8\%$ / $-0.02$ | $+39.8\%$ / $0.10$ | $+51.7\%$ / $0.13$ |
| S3 ICIR weighting | $-5.5\%$ / $-0.02$ | $+16.0\%$ / $0.04$ | $+57.4\%$ / $0.14$ | $+82.8\%$ / $0.20$ |
| S4 pure long-side equal weight | $-2.9\%$ / $-0.01$ | $-2.0\%$ / $-0.01$ | $+51.4\%$ / $0.13$ | $+58.0\%$ / $0.15$ |
| S7 pure long-side $t$ weighting | $+35.2\%$ / $0.08$ | $+62.1\%$ / $0.14$ | $+71.1\%$ / $0.17$ | $+88.5\%$ / $0.21$ |

The three sources of improvement are separable: **diversification** contributes the most (S0 turns positive merely by adding holdings), **long-side $t$ weighting** is second (outperforming equal weight at every holding count), and **long-side gating** is third (a positive contribution at every holding count).

### 11.3 S7 Rolling Out-of-Sample Validation (2020–2025)

The S7 rules are integrated into the rolling framework (`baseline/rolling_top_weighted.py`): each year, only the preceding 5 years are used to determine directions, calculate long-side $t$-statistics, and set gates and weights. These remain fixed within the year, for 72 out-of-sample periods in total.

There are two completely independent dimensions here:

- **Rolling**: redetermine the factor set, directions, and weights each year, and reselect stocks each month using that year's rules;
- **Top 50 / Top 100**: use the same S7 composite score and hold the 50 or 100 highest-scoring stocks, respectively, with equal weights of $1/50$ and $1/100$. These are not “50 / 100 factors,” nor are they two different optimizers.

The following table therefore changes only the holding count N to test diversification sensitivity. Top 100 is the configuration currently presented and should not be described as an optimal parameter locked in advance or as independent out-of-sample proof.

| Configuration | Cumulative return | Annualized | Sharpe | Maximum drawdown | Turnover |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original rolling (ICIR-weighted Top 10) | $-6.40\%$ | $-1.11\%$ | $-0.04$ | $-41.31\%$ | $10.68$ |
| S7 rolling Top 10 | $+14.66\%$ | $+2.33\%$ | $0.09$ | $-53.16\%$ | $10.60$ |
| S7 rolling Top 50 | $+34.29\%$ | $+5.09\%$ | $0.21$ | $-47.62\%$ | $9.65$ |
| S7 rolling Top 100 | $+50.10\%$ | $+7.08\%$ | $0.31$ | $-40.53\%$ | $9.07$ |
| CSI 300 (same period) | $+38.82\%$ | $+5.68\%$ | $0.27$ | $-36.96\%$ | — |

Out of sample, S7 Top 100 outperforms the benchmark ($+50.10\%$ vs $+38.82\%$, Sharpe ratio $0.31$ vs $0.27$). The gate contracts the factor count as market conditions change: 20 factors in 2020 → 9 in 2022 → 7 in 2024 → 5 in 2025. The core factors retained consistently are `turnover_std_3m`, `Alpha20`, `Alpha34`, `amihud_3m`, and `Alpha4`, consistent with the strongest long-side factors in the full-sample diagnostics.

Two points must be stated explicitly:

1. **Excess returns are highly concentrated in 2025**: the strategy underperforms the benchmark overall during 2020–2024, and its overtaking of the benchmark comes mainly from the rally in the second half of 2025; it does not consistently and persistently outperform (see the NAV chart in `results/rolling_top/top100/nav.png`).
2. **The drawdown during the small-cap liquidity crisis in early 2024 was deep**: price-volume factors naturally tilt toward small caps. During the February 2024 micro-cap crisis, maximum drawdown reached $-40.53\%$, deeper than the CSI 300's $-36.96\%$, making this the strategy's most vulnerable scenario.

### 11.4 Interim Conclusions

Without capacity and structural optimization, the excess returns of a pure price-volume long strategy come from untradable illiquid stocks. Within the tradable universe, **long-side gating + long-side $t$ weighting + diversification (Top 100)** can outperform the CSI 300 out of sample, but the excess returns are concentrated in 2025 and drawdowns are deep during the small-cap crisis. The full heuristic search for Optimizer 1, $λ$ weight shrinkage, and Optimizer 2's eight capital-allocation methods remain future work.
