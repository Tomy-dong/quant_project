# Changelog

**English** | [中文](CHANGELOG.zh-CN.md)

This file records important strategy, code, result and documentation changes. Its organization is inspired by [Keep a Changelog](https://keepachangelog.com/en/1.1.0/): entries summarize substantive updates by date rather than duplicating the Git commit log.

## 2026-11-26 | Main-Strategy Finalization, Expanded Results and Documentation Sync

### Update Summary

The project has moved from an early 38-factor, concentrated-holding, annually reselected model to a long-only strategy defined as **470 candidates → 40 after screening → fixed K=4 → ICIR factor weights → IVar stock weights → Top 100**. This update also adds sensitivity analysis, FF3 attribution, actual-holdings return attribution, transaction-cost and capacity analysis, and an XGBoost appendix. The Chinese and English READMEs, strategy documents, charts and holdings outputs are now aligned.

### Key Changes from Historical Versions

| Representative version | Factor and holdings configuration | Cumulative | Annualized | Sharpe | Max drawdown |
| --- | --- | ---: | ---: | ---: | ---: |
| Early 38-factor version | 38-factor pool, Top 10, GMVP | +33.01% | 4.92% | 0.149 | -38.17% |
| WorldQuant-only version | 4–5 factors, Top 30, IVar | +75.38% | 9.92% | 0.407 | -37.25% |
| Expanded-pool stage | 40-factor screened pool, Top 50, IVar | +77.27% | 10.12% | 0.500 | -38.33% |
| **Current main strategy** | **Fixed K=4, Top 100, IVar** | **+89.46%** | **11.24%** | **0.543** | **-26.43%** |

All figures use the 2020–2025 out-of-sample window and a 0.002 one-way fee. They illustrate strategy evolution; factor pools and portfolio rules differ across stages, so this is not a strict single-variable ablation. The current strategy's annualized excess return is **+8.75pp over CSI 300** and **+5.54pp over CSI 500**.

### Added

- **Expanded factor library**: 470 vectorized candidates from Alpha191, Alpha158, Alpha101 and the Tsinghua price-volume library; 40 pass the |mean IC|≥0.05 and |ICIR|≥0.5 screen.
- **Main-strategy pipeline**: `baseline/main_strategy_a.py` now produces NAV, drawdown, IC, K/TopN sensitivity, FF3 outputs and monthly Excel holdings through one workflow.
- **Sensitivity analysis**: out-of-sample scans cover K=4–20 and TopN=20–200, supporting K=4 and Top 100 as the main configuration under current constraints.
- **Actual-holdings attribution**: `baseline/attribution_main.py` decomposes returns into an intercept, four factor contributions and a residual, with contribution bars and annual exposure heatmaps.
- **Cost and capacity analysis**: `baseline/capacity_analysis.py` covers fees, square-root market impact, stock-level participation and net annualized returns at different AUM levels.
- **XGBoost appendix**: adds 40-factor nonlinear composition, raw/rank label experiments and EW/IVar/GMVP/MDCP stock-level weighting. It remains an appendix because it uses more than ten input factors.
- **Reproducible outputs**: adds CSV, PNG and Excel files for the main strategy, attribution, capacity, XGBoost and related comparisons.
- **Standalone main-strategy panels**: adds `results/main_strategy/nav_drawdown_main.png`, with cumulative NAV above and drawdown below. It contains only the fixed-K4 main strategy, CSI 300 and CSI 500 and directly matches the main result table, avoiding confusion with the XGBoost Top 30 appendix charts.
- **Combined main/appendix figure**: adds `results/main_strategy/main_vs_xgb_top30.png`, a 2×2 comparison with the fixed-K4 Top 100 main strategy on the left and four XGBoost Top 30 weighting schemes on the right. The README now uses this combined figure and labels it as a same-window, same-fee portfolio comparison rather than a controlled ablation. `baseline/plot_main_xgb_compare.py` reads saved monthly returns and does not rerun the backtests. It also exports `main_vs_xgb_top30_en.png`; the English README uses this version with fully English titles, legends, axes and footnotes.
- **Complete English figure set**: all 10 figures embedded in the English README now have dedicated English versions, covering combined NAV/drawdown, factor IC, annual IC heatmap, K/TopN sensitivity, FF3, return attribution, exposure heatmap, capacity and the Top 30 appendix. Their generators also export `*_en.png` files with English titles, legends, axes and annotations. The Chinese README and Chinese figures remain unchanged.

### Changed

- **Rolling selection replaced by a fixed factor set**: four factors selected on 2015–2019 are frozen to avoid chasing signals that decay out of sample; annual rolling selection remains a control.
- **Concentrated holdings replaced by broader diversification**: the main configuration moves from Top 10/Top 30 to Top 100. Annualized return is 11.24% versus 5.74% for Top 30, while maximum drawdown improves from -29.10% to -26.43%.
- **IVar adopted for stock weights**: inverse-variance weights are normalized within Top 100 and remain long-only and non-negative. All eight allocation methods remain available for comparison.
- **FF3 interpretation clarified**: FF3 is an attribution benchmark, not a tradable competing portfolio. With alpha=-0.10%, p=0.967 and R²=0.916, the strategy is presented as a market/small-cap style-enhanced **β+ portfolio**, not as a source of significant pure alpha.
- **Market-neutral version not adopted by default**: hedge and basis-sensitivity results are retained, but a 4% annual basis materially erodes returns, so the main strategy remains long-only active management.
- **Holdings presentation updated**: holdings-weight heatmaps are removed in favor of monthly Excel holdings; the other required charts remain and have been regenerated.

### Fixed

- Fixed the mismatch where README summaries described the new strategy while body sections still showed legacy 38-factor and Top 10/S7/Top 30 results. Both READMEs now follow the current main strategy.
- Replaced all 12 README figures with current main-strategy charts or explicitly labelled appendix charts; historical experiments are no longer mixed with current performance.
- Standardized notation: **K denotes factor count and N denotes holdings count**. “Linear N=5/N=10” in the appendix belongs to historical Top 30 controls, not the current main strategy.
- Fixed `results/final/summary.csv`, which had mistakenly reported the XGBoost Top 10 feature-importance display as the model's input factor count. Both XGBoost configurations now report **40 input factors**. `factor_sets.csv` continues to display only the ten most important features, and the counting logic in `baseline/final_report.py` is fixed to prevent regeneration from reintroducing the error.
- Clarified the raw-label mechanism: squared error gives extreme-return samples larger losses and gradients, while Top 30 selection by predicted return does not directly penalize volatility. High-score portfolios can therefore concentrate in high-beta names, but the model does not automatically prefer every high-volatility stock.
- Clarified negative factor contributions: `a158_cord20` contributes -2.64% cumulatively with t=-0.28, a small and insignificant period-specific drag rather than a calculation error.
- Unified factor preprocessing between testing and composition, and clearly separated main-strategy, rolling-control, historical and XGBoost-appendix results.

### Current Results and Risk Disclosure

- The main strategy delivers **11.24% annualized return**, **0.543 Sharpe** and **-26.43% maximum drawdown**, beating CSI 300 and CSI 500 but remaining below the 15% annualized target.
- Annualized one-way turnover is approximately **22.2x**, implying an annual fee drag of about **4.44%** at a 0.002 one-way fee.
- Under a square-root impact model with k=10bp, robust capacity is approximately **RMB 100–300 million**. Estimated net annualized return is 6.56% at RMB 300 million AUM and 2.70% at RMB 1 billion.
- FF3 alpha is insignificant, so there is no evidence of stable pure alpha. Survivorship bias, current-snapshot industry classifications, quarterly share counts and dividend-free benchmark indices remain documented limitations.

### Main Reproduction Entry Points

```bash
python baseline/main_strategy_a.py
python baseline/attribution_main.py
python baseline/capacity_analysis.py

# Appendix: 40-factor XGBoost, not the main strategy
python baseline/xgb_strategy.py
python baseline/xgb_riskweight.py
```

Key outputs are under `results/main_strategy/`, `results/attribution_main/`, `results/capacity/`, `results/final/` and `results/xgb_riskweight/`. See `docs/strategy.md` for the full methodology, historical experiments and chart index.

## 2026-11-25 | Optimizer and Strategy-Audit Framework

- Completed Optimizer 1 gates, heuristic factor selection, collinearity control and λ weight shrinkage.
- Completed all eight long-only allocation methods in Optimizer 2.
- Added FF3, return attribution, seven-sins, drawdown, scenario-stress and market-neutral feasibility tests.
- Completed an initial synchronization of bilingual strategy documentation and result files.

## 2026-11-24 | GitHub Documentation Rendering Fixes

- Fixed Markdown table and formula compatibility in GitHub previews, improving the readability of the Chinese and English strategy documents.
