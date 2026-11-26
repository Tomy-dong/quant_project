"""合并主策略与 XGBoost 附录的净值、回撤图，输出中英文两个版本。

输入为两个策略脚本已经保存的月度收益，不重新运行因子筛选或组合回测：
- results/main_strategy/monthly_pnl.csv
- results/xgb_riskweight/pnl.csv

输出：
- results/main_strategy/main_vs_xgb_top30.png（中文）
- results/main_strategy/main_vs_xgb_top30_en.png（英文）
"""
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parent.parent
MAIN_PATH = ROOT / "results" / "main_strategy" / "monthly_pnl.csv"
XGB_PATH = ROOT / "results" / "xgb_riskweight" / "pnl.csv"
OUTPUTS = {
    "zh": ROOT / "results" / "main_strategy" / "main_vs_xgb_top30.png",
    "en": ROOT / "results" / "main_strategy" / "main_vs_xgb_top30_en.png",
}
TEXT = {
    "zh": {
        "main_labels": {"固定因子(主策略)": "主策略（K=4，Top100）",
                        "沪深300": "沪深300", "中证500": "中证500"},
        "main_title": "(a) 主策略：固定 K=4 + ICIR + IVar Top100",
        "xgb_title": "(b) 附录：XGBoost v3 Top30 × 股票层风险权重",
        "nav": "累计净值（初始=1）", "drawdown": "回撤", "date": "日期",
        "title": "主策略（Top100）与 XGBoost 附录（Top30）：净值与回撤对照（2020–2025）",
        "footnote": "统一样本外区间与单边千分之二费率；左侧为主策略，右侧为附录对照，因子合成方式与持股数量不同。",
    },
    "en": {
        "main_labels": {"固定因子(主策略)": "Main strategy (K=4, Top 100)",
                        "沪深300": "CSI 300", "中证500": "CSI 500"},
        "main_title": "(a) Main: Fixed K=4 + ICIR + IVar Top 100",
        "xgb_title": "(b) Appendix: XGBoost v3 Top 30 × Stock-Level Risk Weights",
        "nav": "Cumulative NAV (initial = 1.0)", "drawdown": "Drawdown", "date": "Date",
        "title": "Main Strategy (Top 100) vs. XGBoost Appendix (Top 30): NAV and Drawdown (2020–2025)",
        "footnote": "Same out-of-sample window and 0.002 one-way fee. Left: main strategy; right: appendix. Different factor models and holdings counts.",
    },
}

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def render(main_pnl, xgb_pnl, lang):
    text = TEXT[lang]
    main_names = ["固定因子(主策略)", "沪深300", "中证500"]
    xgb_names = ["EW", "IVar", "GMVP", "MDCP"]
    main_colors = {"固定因子(主策略)": "crimson",
                   "沪深300": "steelblue", "中证500": "darkorange"}
    xgb_colors = {"EW": "#4c78a8", "IVar": "#f58518",
                  "GMVP": "#54a24b", "MDCP": "#e45756"}

    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex="col", sharey="row",
                             gridspec_kw={"height_ratios": [1.5, 1.0],
                                          "hspace": 0.10, "wspace": 0.10})
    for name in main_names:
        nav = (1 + main_pnl[name].fillna(0)).cumprod()
        dd = nav / nav.cummax() - 1
        is_main = name == "固定因子(主策略)"
        axes[0, 0].plot(nav.index, nav, label=text["main_labels"][name],
                        color=main_colors[name], lw=2.4 if is_main else 1.4)
        axes[1, 0].plot(dd.index, dd, color=main_colors[name],
                        lw=1.9 if is_main else 1.2)
    for name in xgb_names:
        nav = (1 + xgb_pnl[name].fillna(0)).cumprod()
        dd = nav / nav.cummax() - 1
        axes[0, 1].plot(nav.index, nav, label=name, color=xgb_colors[name],
                        lw=2.2 if name == "IVar" else 1.4)
        axes[1, 1].plot(dd.index, dd, color=xgb_colors[name],
                        lw=1.8 if name == "IVar" else 1.2)

    axes[0, 0].set_title(text["main_title"], fontsize=12)
    axes[0, 1].set_title(text["xgb_title"], fontsize=12)
    for col in range(2):
        axes[0, col].axhline(1.0, color="gray", lw=0.8, ls="--")
        axes[0, col].set_ylabel(text["nav"])
        axes[0, col].legend(loc="upper left", ncol=3 if col == 0 else 4, fontsize=9)
        axes[0, col].grid(alpha=0.3)
        axes[1, col].axhline(-0.4, color="gray", lw=0.8, ls="--")
        axes[1, col].set_ylabel(text["drawdown"])
        axes[1, col].set_xlabel(text["date"])
        axes[1, col].yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        axes[1, col].grid(alpha=0.3)

    fig.suptitle(text["title"], fontsize=15, y=0.98)
    fig.text(0.5, 0.015, text["footnote"], ha="center", fontsize=10)
    fig.align_ylabels(axes)
    fig.savefig(OUTPUTS[lang], dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"{lang} 合并图已保存：{OUTPUTS[lang]}")


def main():
    for path in (MAIN_PATH, XGB_PATH):
        if not path.exists():
            raise FileNotFoundError(f"缺少 {path}，请先运行对应策略脚本")
    main_pnl = pd.read_csv(MAIN_PATH, index_col=0, parse_dates=True)
    xgb_pnl = pd.read_csv(XGB_PATH, index_col=0, parse_dates=True)
    for lang in ("zh", "en"):
        render(main_pnl, xgb_pnl, lang)


if __name__ == "__main__":
    main()
