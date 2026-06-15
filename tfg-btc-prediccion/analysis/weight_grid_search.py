"""
analysis/weight_grid_search.py
─────────────────────────────────────────────────────────────────────────────
Grid search de pesos para el confluence score. Prueba 20+ combinaciones
predefinidas de pesos (cada una documentada con su rationale) y rankea
por Sharpe + return + robustez.

Output:
    results/weight_grid_search.csv
    results/weight_grid_search_top.csv (top 10)
    results/weight_grid_chart.png
"""
from __future__ import annotations

import logging
import sys
from importlib import util as _ilu
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

_spec = _ilu.spec_from_file_location("bt", ROOT / "analysis" / "backtest_confluence.py")
bt = _ilu.module_from_spec(_spec)
sys.modules["bt"] = bt
_spec.loader.exec_module(bt)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("grid")


# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURACIONES DE PESOS A PROBAR
# ═════════════════════════════════════════════════════════════════════════════
# Cada config debe sumar 1.0. Documentada con rationale.

CONFIGS = {
    # === Baseline ===
    "01_Original":           {"technical": 0.20, "ict": 0.25, "mtf": 0.20, "smart_money": 0.10, "sentiment": 0.10, "ml": 0.15},
    # Original del TFG. Balanceado, sin sesgo claro.

    # === Variaciones sobre ICT ===
    "02_ICT_x1.4":           {"technical": 0.18, "ict": 0.35, "mtf": 0.18, "smart_money": 0.09, "sentiment": 0.09, "ml": 0.11},
    "03_ICT_x1.6":           {"technical": 0.15, "ict": 0.40, "mtf": 0.17, "smart_money": 0.10, "sentiment": 0.08, "ml": 0.10},
    "04_ICT_x2":             {"technical": 0.12, "ict": 0.50, "mtf": 0.15, "smart_money": 0.08, "sentiment": 0.07, "ml": 0.08},

    # === Variaciones sobre técnico ===
    "05_Tech_Heavy":         {"technical": 0.40, "ict": 0.20, "mtf": 0.15, "smart_money": 0.10, "sentiment": 0.05, "ml": 0.10},
    "06_Tech+ICT":           {"technical": 0.30, "ict": 0.30, "mtf": 0.15, "smart_money": 0.10, "sentiment": 0.05, "ml": 0.10},

    # === Variaciones sobre MTF ===
    "07_MTF_Heavy":          {"technical": 0.15, "ict": 0.20, "mtf": 0.35, "smart_money": 0.10, "sentiment": 0.10, "ml": 0.10},
    "08_MTF+ICT":            {"technical": 0.15, "ict": 0.30, "mtf": 0.30, "smart_money": 0.10, "sentiment": 0.05, "ml": 0.10},

    # === Smart Money / Sentimiento boost ===
    "09_SM_Heavy":           {"technical": 0.15, "ict": 0.20, "mtf": 0.15, "smart_money": 0.30, "sentiment": 0.10, "ml": 0.10},
    "10_Sentiment_Heavy":    {"technical": 0.15, "ict": 0.20, "mtf": 0.15, "smart_money": 0.10, "sentiment": 0.30, "ml": 0.10},

    # === Sin ML (asume ML inválido / redistribuido) ===
    "11_No_ML":              {"technical": 0.25, "ict": 0.30, "mtf": 0.20, "smart_money": 0.12, "sentiment": 0.13, "ml": 0.00},
    "12_No_ML_ICT_Heavy":    {"technical": 0.18, "ict": 0.40, "mtf": 0.20, "smart_money": 0.11, "sentiment": 0.11, "ml": 0.00},

    # === ML Heavy ===
    "13_ML_Heavy":           {"technical": 0.15, "ict": 0.15, "mtf": 0.15, "smart_money": 0.10, "sentiment": 0.05, "ml": 0.40},

    # === Combinaciones balanceadas ===
    "14_Equal":              {"technical": 0.167, "ict": 0.167, "mtf": 0.167, "smart_money": 0.167, "sentiment": 0.166, "ml": 0.166},
    "15_Structure_Focus":    {"technical": 0.15, "ict": 0.35, "mtf": 0.30, "smart_money": 0.10, "sentiment": 0.05, "ml": 0.05},
    "16_Indicators_Focus":   {"technical": 0.30, "ict": 0.15, "mtf": 0.15, "smart_money": 0.20, "sentiment": 0.10, "ml": 0.10},

    # === Configs académicas (basadas en literatura) ===
    "17_Lopez_de_Prado_inspired": {"technical": 0.20, "ict": 0.20, "mtf": 0.25, "smart_money": 0.10, "sentiment": 0.10, "ml": 0.15},
    "18_SMC_Focus":          {"technical": 0.10, "ict": 0.40, "mtf": 0.20, "smart_money": 0.20, "sentiment": 0.05, "ml": 0.05},
    "19_Mean_Reversion":     {"technical": 0.35, "ict": 0.15, "mtf": 0.10, "smart_money": 0.15, "sentiment": 0.15, "ml": 0.10},
    "20_Trend_Following":    {"technical": 0.20, "ict": 0.25, "mtf": 0.35, "smart_money": 0.05, "sentiment": 0.05, "ml": 0.10},

    # === Extremos para validar robustez ===
    "21_Only_ICT":           {"technical": 0.05, "ict": 0.70, "mtf": 0.10, "smart_money": 0.05, "sentiment": 0.05, "ml": 0.05},
    "22_Only_Tech":          {"technical": 0.70, "ict": 0.10, "mtf": 0.05, "smart_money": 0.05, "sentiment": 0.05, "ml": 0.05},
}


def main():
    log.info("═" * 78)
    log.info(f"GRID SEARCH — {len(CONFIGS)} configuraciones de pesos | 90 días backtest")
    log.info("═" * 78)

    df = bt.load_ohlcv("4h", 90)
    df = bt.compute_indicators(df)
    df = bt.detect_obs_fvgs(df)
    df = df.dropna(subset=["rsi_14", "macd", "ema_50", "atr_14"])
    fg = bt.load_fear_greed_daily()
    daily = bt.load_daily_features()
    period_days = (df.index[-1] - df.index[0]).total_seconds() / 86400

    log.info(f"Datos: {len(df)} bars 4h | {df.index[0].date()} → {df.index[-1].date()}")
    log.info("")

    results = {}
    for name, w in CONFIGS.items():
        s = sum(w.values())
        if abs(s - 1.0) > 0.01:
            log.warning(f"  {name}: pesos no suman 1.0 ({s:.3f}), normalizo")
            w = {k: v / s for k, v in w.items()}
        try:
            trades, eq = bt.strategy_full_system(df, fg, daily, threshold=bt.THRESHOLD, weights=w)
            m = bt.compute_metrics(trades, eq, bt.INITIAL_CAPITAL, period_days)
            m["weights"] = " / ".join(f"{k[:3]}={v:.2f}" for k, v in w.items())
            results[name] = m
            log.info(f"  {name:35s} ret={m['total_return_pct']:+6.2f}% | sharpe={m['sharpe']:>5.2f} | "
                     f"trades={m['n_trades']:>3} | DD={m['max_dd_pct']:>6.2f}% | win={m['win_rate_pct']:>5.1f}%")
        except Exception as e:
            log.error(f"  {name} FAILED: {e}")
            results[name] = {"error": str(e)}

    # Save full results
    out = ROOT / "results"; out.mkdir(exist_ok=True)
    df_res = pd.DataFrame(results).T
    df_res.to_csv(out / "weight_grid_search.csv")
    log.info(f"\nSaved: {out / 'weight_grid_search.csv'}")

    # ═════════════════════════════════════════════════════════════════════════
    # ANÁLISIS Y RANKING
    # ═════════════════════════════════════════════════════════════════════════
    df_clean = df_res[df_res["n_trades"].astype(int) >= 5].copy()
    df_clean = df_clean.astype({c: float for c in ["total_return_pct","sharpe","max_dd_pct","win_rate_pct","n_trades"]})

    # Score compuesto: sharpe (50%) + return (30%) + -DD (20%)
    sh = df_clean["sharpe"]
    rt = df_clean["total_return_pct"] / 10  # escalar
    dd = -df_clean["max_dd_pct"] / 10
    df_clean["composite_score"] = (sh * 0.5 + rt * 0.3 + dd * 0.2).round(3)

    top10 = df_clean.sort_values("composite_score", ascending=False).head(10)
    top10[["composite_score", "total_return_pct", "sharpe", "max_dd_pct",
           "n_trades", "win_rate_pct", "profit_factor", "weights"]].to_csv(out / "weight_grid_search_top.csv")

    # ═════════════════════════════════════════════════════════════════════════
    # OUTPUT FINAL
    # ═════════════════════════════════════════════════════════════════════════
    print()
    print("=" * 130)
    print("TOP 10 CONFIGURACIONES (composite_score = 0.5*sharpe + 0.3*return/10 + 0.2*(-DD/10))")
    print("=" * 130)
    print(top10[["composite_score", "total_return_pct", "sharpe", "max_dd_pct",
                 "n_trades", "win_rate_pct", "profit_factor"]].to_string())
    print()

    best = top10.iloc[0]
    print("=" * 130)
    print(f"MEJOR CONFIG: {top10.index[0]}")
    print("=" * 130)
    print(f"  Composite score: {best['composite_score']}")
    print(f"  Return:          {best['total_return_pct']:+.2f}%")
    print(f"  Sharpe:          {best['sharpe']:.2f}")
    print(f"  Max DD:          {best['max_dd_pct']:.2f}%")
    print(f"  Trades:          {int(best['n_trades'])}")
    print(f"  Win rate:        {best['win_rate_pct']}%")
    print(f"  Pesos:           {best['weights']}")
    print()

    # ROBUSTEZ: configs con sharpe>0 (no solo el mejor)
    robust = df_clean[df_clean["sharpe"] > 0].sort_values("sharpe", ascending=False)
    print("=" * 130)
    print(f"CONFIGS ROBUSTAS (Sharpe > 0) — {len(robust)} de {len(df_clean)}")
    print("=" * 130)
    if len(robust) > 0:
        print(robust[["composite_score", "total_return_pct", "sharpe", "max_dd_pct",
                      "n_trades", "win_rate_pct", "weights"]].to_string())
    else:
        print("Ninguna config supera Sharpe > 0 en este periodo — sistema necesita más calibración.")
    print()


if __name__ == "__main__":
    main()
