"""
analysis/backtest_robustness.py
─────────────────────────────────────────────────────────────────────────────
Análisis de robustez del sistema completo (Full System):

  1. Sensibilidad al umbral de entrada (0.15, 0.20, 0.30, 0.40, 0.50)
  2. Performance por killzone (London / NY / fuera de killzone)
  3. Performance por régimen de mercado (bull/bear/lateral usando EMA200)
  4. Sensibilidad a pesos (ML 0.15 vs 0.30, ICT 0.25 vs 0.40)

Salida:
  results/robustness_thresholds.csv
  results/robustness_regimes.csv
  results/robustness_killzone.csv
  results/robustness_weights.csv
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from importlib import util as _ilu

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Importar módulo del backtest base (registrar en sys.modules ANTES de exec para que @dataclass funcione)
_spec = _ilu.spec_from_file_location("bt", ROOT / "analysis" / "backtest_confluence.py")
bt = _ilu.module_from_spec(_spec)
sys.modules["bt"] = bt
_spec.loader.exec_module(bt)

log = logging.getLogger("robustness")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _load_all(days=90, tf="4h"):
    df = bt.load_ohlcv(tf, days)
    df = bt.compute_indicators(df)
    df = bt.detect_obs_fvgs(df)
    df = df.dropna(subset=["rsi_14", "macd", "ema_50", "atr_14"])
    fg = bt.load_fear_greed_daily()
    daily = bt.load_daily_features()
    period_days = (df.index[-1] - df.index[0]).total_seconds() / 86400
    return df, fg, daily, period_days


# ═════════════════════════════════════════════════════════════════════════════
# 1. SENSIBILIDAD AL UMBRAL
# ═════════════════════════════════════════════════════════════════════════════
def thresholds_sweep(df, fg, daily, period_days):
    log.info("─── Sensibilidad al umbral del confluence score ───")
    rows = {}
    for th in [0.15, 0.20, 0.30, 0.40, 0.50]:
        trades, eq = bt.strategy_full_system(df, fg, daily, threshold=th)
        m = bt.compute_metrics(trades, eq, bt.INITIAL_CAPITAL, period_days)
        rows[f"th={th:.2f}"] = m
        log.info(f"  th={th:.2f} | ret={m['total_return_pct']:+.2f}% | trades={m['n_trades']:>3} "
                 f"| sharpe={m['sharpe']:>5.2f} | DD={m['max_dd_pct']:.2f}% | win={m['win_rate_pct']:.0f}%")
    return pd.DataFrame(rows).T


# ═════════════════════════════════════════════════════════════════════════════
# 2. KILLZONE FILTER
# ═════════════════════════════════════════════════════════════════════════════
def _is_killzone(ts):
    """London 07-09 UTC + NY 12-14 UTC."""
    h = ts.hour
    return (7 <= h < 9) or (12 <= h < 14)


def killzone_split(df, fg, daily, period_days):
    log.info("─── Performance dentro vs fuera de killzone ───")
    # Ejecuta Full System completo, luego segmenta los trades
    trades, eq = bt.strategy_full_system(df, fg, daily, threshold=bt.THRESHOLD)
    in_kz  = [t for t in trades if _is_killzone(t.entry_time)]
    out_kz = [t for t in trades if not _is_killzone(t.entry_time)]

    rows = {}
    for label, ts_list in [("In Killzone", in_kz), ("Out Killzone", out_kz), ("All", trades)]:
        # Equity reconstruction simplificada: sum pnls
        if not ts_list:
            rows[label] = {"n_trades": 0, "total_pnl": 0, "win_rate_pct": 0, "expectancy_usd": 0}
            continue
        pnls = [t.pnl_net for t in ts_list]
        wins = [p for p in pnls if p > 0]
        rows[label] = {
            "n_trades":       len(pnls),
            "total_pnl_usd":  round(sum(pnls), 2),
            "win_rate_pct":   round(len(wins)/len(pnls)*100, 1),
            "expectancy_usd": round(sum(pnls)/len(pnls), 2),
            "avg_win_usd":    round(sum(wins)/len(wins), 2) if wins else 0,
        }
        log.info(f"  {label:14s} | trades={len(pnls):>3} | total=${sum(pnls):+.0f} | "
                 f"win={rows[label]['win_rate_pct']:.0f}% | E[trade]=${rows[label]['expectancy_usd']:+.0f}")
    return pd.DataFrame(rows).T


# ═════════════════════════════════════════════════════════════════════════════
# 3. RÉGIMEN DE MERCADO
# ═════════════════════════════════════════════════════════════════════════════
def regime_split(df, fg, daily, period_days):
    """Bull = close > EMA200 × 1.02 | Bear = < × 0.98 | Lateral = entre medias."""
    log.info("─── Performance por régimen (EMA200) ───")
    trades, eq = bt.strategy_full_system(df, fg, daily, threshold=bt.THRESHOLD)

    # Mapear cada trade a régimen en su entry_time
    def regime_at(ts):
        idx = df.index.get_indexer([ts], method="ffill")[0]
        if idx < 0: return "unknown"
        row = df.iloc[idx]
        p = row["close"]; e = row["ema_200"]
        if pd.isna(e): return "unknown"
        if p > e * 1.02: return "bull"
        if p < e * 0.98: return "bear"
        return "lateral"

    by_reg = {"bull": [], "bear": [], "lateral": [], "unknown": []}
    for t in trades:
        by_reg[regime_at(t.entry_time)].append(t)

    rows = {}
    for reg, ts_list in by_reg.items():
        if not ts_list:
            rows[reg] = {"n_trades": 0, "total_pnl_usd": 0, "win_rate_pct": 0, "expectancy_usd": 0}
            continue
        pnls = [t.pnl_net for t in ts_list]
        wins = [p for p in pnls if p > 0]
        rows[reg] = {
            "n_trades":       len(pnls),
            "total_pnl_usd":  round(sum(pnls), 2),
            "win_rate_pct":   round(len(wins)/len(pnls)*100, 1),
            "expectancy_usd": round(sum(pnls)/len(pnls), 2),
        }
        log.info(f"  {reg:10s} | trades={len(pnls):>3} | total=${sum(pnls):+.0f} | "
                 f"win={rows[reg]['win_rate_pct']:.0f}%")
    return pd.DataFrame(rows).T


# ═════════════════════════════════════════════════════════════════════════════
# 4. SENSIBILIDAD A PESOS
# ═════════════════════════════════════════════════════════════════════════════
def weights_sweep(df, fg, daily, period_days):
    log.info("─── Sensibilidad a pesos ───")
    configs = {
        "Base (TFG)": {"technical": 0.20, "ict": 0.25, "mtf": 0.20, "smart_money": 0.10, "sentiment": 0.10, "ml": 0.15},
        "ICT Heavy":  {"technical": 0.15, "ict": 0.40, "mtf": 0.20, "smart_money": 0.10, "sentiment": 0.05, "ml": 0.10},
        "ML Heavy":   {"technical": 0.15, "ict": 0.15, "mtf": 0.20, "smart_money": 0.10, "sentiment": 0.10, "ml": 0.30},
        "Tech Heavy": {"technical": 0.40, "ict": 0.20, "mtf": 0.15, "smart_money": 0.10, "sentiment": 0.05, "ml": 0.10},
        "Equal":      {"technical": 0.17, "ict": 0.17, "mtf": 0.17, "smart_money": 0.17, "sentiment": 0.16, "ml": 0.16},
    }
    rows = {}
    for name, w in configs.items():
        trades, eq = bt.strategy_full_system(df, fg, daily, threshold=bt.THRESHOLD, weights=w)
        m = bt.compute_metrics(trades, eq, bt.INITIAL_CAPITAL, period_days)
        rows[name] = m
        log.info(f"  {name:12s} | ret={m['total_return_pct']:+.2f}% | trades={m['n_trades']:>3} "
                 f"| sharpe={m['sharpe']:>5.2f} | DD={m['max_dd_pct']:.2f}%")
    return pd.DataFrame(rows).T


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    log.info("═" * 78)
    log.info("ANÁLISIS DE ROBUSTEZ — Full System")
    log.info("═" * 78)
    df, fg, daily, period_days = _load_all(days=90, tf="4h")
    log.info(f"Datos: {len(df)} bars | {df.index[0]} → {df.index[-1]}")
    log.info("")

    out = ROOT / "results"; out.mkdir(exist_ok=True)

    th_df  = thresholds_sweep(df, fg, daily, period_days)
    th_df.to_csv(out / "robustness_thresholds.csv")

    log.info("")
    kz_df  = killzone_split(df, fg, daily, period_days)
    kz_df.to_csv(out / "robustness_killzone.csv")

    log.info("")
    reg_df = regime_split(df, fg, daily, period_days)
    reg_df.to_csv(out / "robustness_regimes.csv")

    log.info("")
    w_df   = weights_sweep(df, fg, daily, period_days)
    w_df.to_csv(out / "robustness_weights.csv")

    print()
    print("=" * 110)
    print("1) SENSIBILIDAD AL UMBRAL")
    print("=" * 110)
    print(th_df[["total_return_pct", "sharpe", "max_dd_pct", "n_trades", "win_rate_pct", "p_value_vs_0"]].to_string())

    print()
    print("=" * 110)
    print("2) IN vs OUT OF KILLZONE")
    print("=" * 110)
    print(kz_df.to_string())

    print()
    print("=" * 110)
    print("3) RÉGIMEN DE MERCADO (EMA200)")
    print("=" * 110)
    print(reg_df.to_string())

    print()
    print("=" * 110)
    print("4) SENSIBILIDAD A PESOS")
    print("=" * 110)
    print(w_df[["total_return_pct", "sharpe", "max_dd_pct", "n_trades", "win_rate_pct"]].to_string())
    print()


if __name__ == "__main__":
    main()
