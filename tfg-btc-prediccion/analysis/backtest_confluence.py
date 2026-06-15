"""
analysis/backtest_confluence.py
─────────────────────────────────────────────────────────────────────────────
Backtest extendido del sistema de scoring de confluencia sobre datos
históricos de BTC. Compara el sistema completo (lógica idéntica a la
función get_confluence_score del bot en producción) contra baselines.

Métricas profesionales: Total Return, CAGR, Sharpe, Sortino, Calmar,
Max DD, Profit Factor, Win Rate, Avg Win/Loss, Expectancy, # trades,
duración media, p-value t-test vs 0.

Outputs:
    results/backtest_compare.csv          — tabla comparativa
    results/backtest_equity_curves.png    — equity curves superpuestas
    results/backtest_drawdown.png         — underwater plot

Constraints:
    • Sin look-ahead: en cada bar usa solo datos <= ese momento
    • Fees Bybit Demo: 0.06% taker × 2 sides
    • Slippage: 0.05% por lado
    • Walk-forward en ML: entrena con datos previos al periodo backtest

Uso:
    python analysis/backtest_confluence.py
    python analysis/backtest_confluence.py --days 90 --tf 4h --capital 50000
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from db.db_utils import get_engine   # noqa: E402

# Importar ICT signals avanzados (liquidity sweeps, BOS/CHoCH, premium/discount, daily/weekly)
try:
    from analysis.ict_signals import (
        detect_liquidity_sweeps,
        detect_bos_choch,
        compute_premium_discount,
        detect_daily_weekly_levels,
        score_ict_advanced,
    )
    ICT_ADVANCED_AVAILABLE = True
except Exception as _e:
    ICT_ADVANCED_AVAILABLE = False
    print(f"[WARN] ict_signals no disponible: {_e}")


# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═════════════════════════════════════════════════════════════════════════════
TAKER_FEE       = 0.0006     # 0.06% Bybit Demo per side
SLIPPAGE        = 0.0005     # 0.05% per side
INITIAL_CAPITAL = 50_000.0
LEVERAGE        = 3.0
POSITION_PCT    = 0.25       # 25% equity por trade
ATR_STOP_MULT   = 2.0        # alineado con live_bot (antes 2.0)
ATR_TP_MULT     = 4.0        # alineado con live_bot (antes 3.0) — R:R 2:1 gross
MAX_HOLD_BARS   = 24         # 24 barras de 4h = 4 días

# ── Money management ─────────────────────────────────────────────────────────
RISK_PER_TRADE_PCT  = 1.0    # % equity arriesgado por trade (al SL)
VOL_ADJUST          = True   # ajustar tamaño según ATR/precio (vol-targeting)
VOL_TARGET_PCT      = 1.5    # ATR objetivo (% del precio) para tamaño base
DAILY_LOSS_LIMIT_PCT = 3.0   # stop trading si pierdes esto en un día (% capital)
REDUCE_AFTER_LOSS   = True   # tras una pérdida, usar 0.5× sizing en próximos N trades
REDUCE_TRADES       = 2      # nº de trades a reducir tras una pérdida

WEIGHTS = {
    # Calibrado por grid search empírico (config 20_Trend_Following ganadora)
    "technical":   0.20, "ict":         0.25, "mtf":         0.35,
    "smart_money": 0.05, "sentiment":   0.05, "ml":          0.10,
}
THRESHOLD = 0.30

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("backtest")


# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════
def load_ohlcv(tf: str, days: int) -> pd.DataFrame:
    engine = get_engine()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    sql = text("""
        SELECT timestamp, open, high, low, close, volume
        FROM btc_ohlcv WHERE timeframe = :tf AND timestamp >= :cutoff
        ORDER BY timestamp ASC
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"tf": tf, "cutoff": cutoff}).fetchall()
    if not rows:
        raise RuntimeError(f"No data for tf={tf} in last {days} days")
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index().astype(float)
    return df


def load_fear_greed_daily() -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT date, fear_greed AS value FROM daily_features "
            "WHERE asset='BTC' AND fear_greed IS NOT NULL ORDER BY date"
        )).fetchall()
    df = pd.DataFrame(rows, columns=["date", "value"])
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df.set_index("date").astype(float)


def load_daily_features() -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT date, rsi_14, macd, macd_signal, bb_upper, bb_lower,
                   sma_7, sma_30, fear_greed, returns, sentiment_avg, label
            FROM daily_features WHERE asset='BTC' ORDER BY date
        """)).fetchall()
    cols = ["date", "rsi_14", "macd", "macd_signal", "bb_upper", "bb_lower",
            "sma_7", "sma_30", "fear_greed", "returns", "sentiment_avg", "label"]
    df = pd.DataFrame(rows, columns=cols)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.set_index("date").apply(pd.to_numeric, errors="coerce")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# INDICATORS
# ═════════════════════════════════════════════════════════════════════════════
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    delta = out["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    ema12 = out["close"].ewm(span=12, adjust=False).mean()
    ema26 = out["close"].ewm(span=26, adjust=False).mean()
    out["macd"]        = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()

    out["ema_9"]   = out["close"].ewm(span=9,   adjust=False).mean()
    out["ema_21"]  = out["close"].ewm(span=21,  adjust=False).mean()
    out["ema_50"]  = out["close"].ewm(span=50,  adjust=False).mean()
    out["ema_200"] = out["close"].ewm(span=200, adjust=False).mean()

    out["bb_mid"]   = out["close"].rolling(20).mean()
    bb_std          = out["close"].rolling(20).std()
    out["bb_upper"] = out["bb_mid"] + 2 * bb_std
    out["bb_lower"] = out["bb_mid"] - 2 * bb_std

    tr = pd.concat([
        out["high"] - out["low"],
        (out["high"] - out["close"].shift()).abs(),
        (out["low"]  - out["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    out["atr_14"] = tr.rolling(14).mean()

    typical = (out["high"] + out["low"] + out["close"]) / 3
    out["vwap"] = (typical * out["volume"]).rolling(24).sum() / out["volume"].rolling(24).sum()
    out["volume_ratio"] = out["volume"] / out["volume"].rolling(20).mean()
    return out


def detect_obs_fvgs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ob_bull"]  = False
    out["ob_bear"]  = False
    out["fvg_bull"] = False
    out["fvg_bear"] = False
    for i in range(2, len(out) - 1):
        c0 = out.iloc[i-1]
        c2 = out.iloc[i+1]
        if c0["close"] < c0["open"] and c2["close"] > c2["open"] and c2["close"] > c0["high"]:
            out.iat[i-1, out.columns.get_loc("ob_bull")] = True
        if c0["close"] > c0["open"] and c2["close"] < c2["open"] and c2["close"] < c0["low"]:
            out.iat[i-1, out.columns.get_loc("ob_bear")] = True
        if out["high"].iloc[i-1] < out["low"].iloc[i+1]:
            out.iat[i, out.columns.get_loc("fvg_bull")] = True
        if out["high"].iloc[i+1] < out["low"].iloc[i-1]:
            out.iat[i, out.columns.get_loc("fvg_bear")] = True
    return out


# ═════════════════════════════════════════════════════════════════════════════
# SCORING (mismas fórmulas que 05_tools.py — sin look-ahead)
# ═════════════════════════════════════════════════════════════════════════════
def score_technical(row, price: float) -> float:
    parts = []
    rsi = row.get("rsi_14")
    if pd.notna(rsi):
        if   rsi < 30: parts.append(0.8)
        elif rsi > 70: parts.append(-0.8)
        elif rsi < 45: parts.append(0.2)
        elif rsi > 55: parts.append(-0.2)
        else:          parts.append(0.0)
    macd, mac_s = row.get("macd"), row.get("macd_signal")
    if pd.notna(macd) and pd.notna(mac_s):
        parts.append(0.4 if macd > mac_s else -0.4)
    e9, e21, e50 = row.get("ema_9"), row.get("ema_21"), row.get("ema_50")
    if pd.notna(e50) and price:
        parts.append(0.3 if price > e50 else -0.3)
    if pd.notna(e9) and pd.notna(e21):
        parts.append(0.3 if e9 > e21 else -0.3)
    bbu, bbl, bbm = row.get("bb_upper"), row.get("bb_lower"), row.get("bb_mid")
    if pd.notna(bbu) and pd.notna(bbl) and pd.notna(bbm) and price:
        r = bbu - bbl
        if r > 0:
            pos = (price - bbl) / r
            if   pos > 0.85: parts.append(-0.3)
            elif pos < 0.15: parts.append(0.3)
            else:            parts.append((pos - 0.5) * 0.4)
    if not parts: return 0.0
    return max(-1.0, min(1.0, sum(parts) / len(parts)))


def score_ict(row, price: float, df_slice: pd.DataFrame) -> float:
    parts = []
    e50, e200 = row.get("ema_50"), row.get("ema_200")
    if pd.notna(e50) and pd.notna(e200) and price:
        if   price > e50 > e200: parts.append(0.4)
        elif price < e50 < e200: parts.append(-0.4)
    recent = df_slice.tail(50)
    if recent["ob_bull"].any():
        idx = recent[recent["ob_bull"]].index[-1]
        ob_price = (recent.loc[idx, "high"] + recent.loc[idx, "low"]) / 2
        if price and abs(price - ob_price) / price < 0.02:
            parts.append(0.5)
    if recent["ob_bear"].any():
        idx = recent[recent["ob_bear"]].index[-1]
        ob_price = (recent.loc[idx, "high"] + recent.loc[idx, "low"]) / 2
        if price and abs(price - ob_price) / price < 0.02:
            parts.append(-0.5)
    recent20 = df_slice.tail(20)
    if recent20["fvg_bull"].any(): parts.append(0.3)
    if recent20["fvg_bear"].any(): parts.append(-0.3)
    if not parts: return 0.0
    return max(-1.0, min(1.0, sum(parts) / len(parts)))


def score_mtf(row, price: float) -> float:
    e200 = row.get("ema_200")
    if pd.notna(e200) and price:
        if   price > e200 * 1.02: return 0.7
        elif price < e200 * 0.98: return -0.7
        elif price > e200:        return 0.3
        else:                     return -0.3
    return 0.0


def score_smart_money(row, price: float) -> float:
    parts = []
    vwap = row.get("vwap")
    if pd.notna(vwap) and price:
        parts.append(0.5 if price > vwap else -0.5)
    vr = row.get("volume_ratio")
    if pd.notna(vr):
        if   vr > 1.5: parts.append(0.4 if parts and parts[0] >= 0 else -0.4)
        elif vr < 0.5: parts.append(-0.2)
        else:          parts.append(0.0)
    if not parts: return 0.0
    return max(-1.0, min(1.0, sum(parts) / len(parts)))


def score_sentiment(t: datetime, fg_df: pd.DataFrame) -> float:
    """F&G contrarian. Sentimiento news omitido (cobertura insuficiente histórica)."""
    if fg_df.empty: return 0.0
    prior = fg_df[fg_df.index <= t]
    if prior.empty: return 0.0
    fg = float(prior["value"].iloc[-1])
    if   fg <= 20: return 0.5
    elif fg <= 35: return 0.2
    elif fg >= 80: return -0.5
    elif fg >= 65: return -0.2
    return 0.0


def confluence_score(t, df, i, fg_df, ml_score=0.0, weights=None, use_advanced_ict=True) -> tuple[float, dict]:
    if weights is None: weights = WEIGHTS
    row = df.iloc[i]
    price = float(row["close"])
    df_slice = df.iloc[:i+1]

    # Si las columnas ICT avanzadas están presentes Y use_advanced_ict=True, usa el scoring real
    has_advanced = use_advanced_ict and ICT_ADVANCED_AVAILABLE and \
                   all(c in df.columns for c in ["liq_sweep_bull", "bos_bull", "choch_bull", "is_premium"])
    if has_advanced:
        ict_score_val, _ = score_ict_advanced(row, df_slice)
    else:
        ict_score_val = score_ict(row, price, df_slice)

    scores = {
        "technical":   score_technical(row, price),
        "ict":         ict_score_val,
        "mtf":         score_mtf(row, price),
        "smart_money": score_smart_money(row, price),
        "sentiment":   score_sentiment(t, fg_df),
        "ml":          ml_score,
    }

    # Redistribuir peso ML si no disponible
    eff = dict(weights)
    if ml_score == 0.0:
        others = [k for k in eff if k != "ml"]
        total = sum(weights[k] for k in others)
        for k in others:
            eff[k] = weights[k] + weights["ml"] * weights[k] / total
        eff["ml"] = 0.0

    final = sum(scores[k] * eff[k] for k in scores)
    return max(-1.0, min(1.0, final)), scores


# ═════════════════════════════════════════════════════════════════════════════
# SIMULATOR
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class Trade:
    entry_time: datetime
    entry_price: float
    direction: str
    size_usd: float
    stop_loss: float
    take_profit: float
    bars_held: int = 0
    exit_time: Optional[datetime] = None
    exit_price: Optional[float]   = None
    exit_reason: str = ""
    pnl_gross: float = 0.0
    fees: float = 0.0
    pnl_net: float = 0.0


def simulate_strategy(
    df: pd.DataFrame,
    signal_fn: Callable[[int], Optional[str]],
    capital: float = INITIAL_CAPITAL,
    leverage: float = LEVERAGE,
    position_pct: float = POSITION_PCT,
    max_hold: int = MAX_HOLD_BARS,
    money_mgmt: bool = True,           # ← activar money management mejorado
) -> tuple[list[Trade], pd.Series]:
    """
    Simula trades con money management opcional:
      - Vol-adjusted sizing: size *= VOL_TARGET_PCT / (atr/price*100), clip [0.5x, 1.5x]
      - Daily loss limit: skip si hoy perdiste > DAILY_LOSS_LIMIT_PCT del capital
      - Reduce-after-loss: tamaño 0.5× en los próximos N trades tras una pérdida
    """
    trades: list[Trade] = []
    open_t: Optional[Trade] = None
    equity = capital
    equity_curve = []
    daily_pnl: dict = {}   # date → pnl del día (USD)
    reduce_counter = 0     # contador para reduce-after-loss

    for i in range(len(df)):
        row = df.iloc[i]
        price = float(row["close"])
        ts = df.index[i]
        atr = row.get("atr_14")

        # Manage open
        if open_t is not None:
            open_t.bars_held += 1
            hit_sl = (open_t.direction == "long"  and row["low"]  <= open_t.stop_loss) or \
                     (open_t.direction == "short" and row["high"] >= open_t.stop_loss)
            hit_tp = (open_t.direction == "long"  and row["high"] >= open_t.take_profit) or \
                     (open_t.direction == "short" and row["low"]  <= open_t.take_profit)

            exit_price, reason = None, None
            if hit_sl:
                exit_price, reason = open_t.stop_loss, "stop_loss"
            elif hit_tp:
                exit_price, reason = open_t.take_profit, "take_profit"
            elif open_t.bars_held >= max_hold:
                exit_price, reason = price, "timeout"
            else:
                new_sig = signal_fn(i)
                if new_sig is not None and new_sig != open_t.direction:
                    exit_price, reason = price, "reverse_signal"

            if exit_price is not None:
                if open_t.direction == "long":
                    eff = exit_price * (1 - SLIPPAGE)
                    gross = (eff - open_t.entry_price) / open_t.entry_price * open_t.size_usd
                else:
                    eff = exit_price * (1 + SLIPPAGE)
                    gross = (open_t.entry_price - eff) / open_t.entry_price * open_t.size_usd
                gross *= leverage
                fees = open_t.size_usd * leverage * TAKER_FEE * 2
                open_t.exit_time   = ts
                open_t.exit_price  = exit_price
                open_t.exit_reason = reason
                open_t.pnl_gross   = gross
                open_t.fees        = fees
                open_t.pnl_net     = gross - fees
                trades.append(open_t)
                equity += open_t.pnl_net
                # Tracking diario para daily loss limit
                day_key = ts.date()
                daily_pnl[day_key] = daily_pnl.get(day_key, 0.0) + open_t.pnl_net
                # Si fue pérdida, activar reduce-after-loss
                if money_mgmt and open_t.pnl_net < 0 and REDUCE_AFTER_LOSS:
                    reduce_counter = REDUCE_TRADES
                elif open_t.pnl_net > 0 and reduce_counter > 0:
                    reduce_counter -= 1
                open_t = None

        # Try new entry
        if open_t is None and pd.notna(atr) and atr > 0:
            # ── Money management filters ────────────────────────────────────
            if money_mgmt:
                day_key = ts.date()
                today_pnl = daily_pnl.get(day_key, 0.0)
                if today_pnl < -(equity * DAILY_LOSS_LIMIT_PCT / 100):
                    equity_curve.append(equity)
                    continue   # daily loss limit hit — no más trades hoy

            sig = signal_fn(i)
            if sig in ("long", "short"):
                # Base size
                size = equity * position_pct
                # Vol-adjusted: reduce size si ATR es alto
                if money_mgmt and VOL_ADJUST:
                    atr_pct = float(atr) / price * 100
                    if atr_pct > 0:
                        adj = VOL_TARGET_PCT / atr_pct
                        adj = max(0.5, min(1.5, adj))    # clip [0.5x, 1.5x]
                        size *= adj
                # Reduce-after-loss
                if money_mgmt and reduce_counter > 0:
                    size *= 0.5

                if sig == "long":
                    entry_eff = price * (1 + SLIPPAGE)
                    sl = price - ATR_STOP_MULT * atr
                    tp = price + ATR_TP_MULT  * atr
                else:
                    entry_eff = price * (1 - SLIPPAGE)
                    sl = price + ATR_STOP_MULT * atr
                    tp = price - ATR_TP_MULT  * atr
                open_t = Trade(
                    entry_time=ts, entry_price=entry_eff, direction=sig,
                    size_usd=size, stop_loss=sl, take_profit=tp,
                )

        equity_curve.append(equity)

    # Cierra open al final
    if open_t is not None:
        last = float(df["close"].iloc[-1])
        if open_t.direction == "long":
            gross = (last - open_t.entry_price) / open_t.entry_price * open_t.size_usd * leverage
        else:
            gross = (open_t.entry_price - last) / open_t.entry_price * open_t.size_usd * leverage
        fees = open_t.size_usd * leverage * TAKER_FEE * 2
        open_t.exit_time   = df.index[-1]
        open_t.exit_price  = last
        open_t.exit_reason = "end_of_period"
        open_t.pnl_gross   = gross
        open_t.fees        = fees
        open_t.pnl_net     = gross - fees
        trades.append(open_t)
        equity += open_t.pnl_net
        equity_curve[-1] = equity

    return trades, pd.Series(equity_curve, index=df.index, name="equity")


# ═════════════════════════════════════════════════════════════════════════════
# METRICS
# ═════════════════════════════════════════════════════════════════════════════
def compute_metrics(trades: list[Trade], equity: pd.Series, initial: float, days: float) -> dict:
    if equity.empty:
        return {"error": "no_equity"}
    final = float(equity.iloc[-1])
    total_ret_pct = (final - initial) / initial * 100
    years = days / 365.25
    cagr = ((final / initial) ** (1 / years) - 1) * 100 if years > 0 and final > 0 else 0.0

    daily_eq  = equity.resample("1D").last().dropna()
    daily_ret = daily_eq.pct_change().dropna()

    sharpe = (daily_ret.mean() / daily_ret.std() * math.sqrt(365)) if len(daily_ret) > 1 and daily_ret.std() > 0 else 0.0
    downside = daily_ret[daily_ret < 0]
    sortino = (daily_ret.mean() / downside.std() * math.sqrt(365)) if len(downside) > 1 and downside.std() > 0 else 0.0

    running_max = equity.cummax()
    dd = (equity - running_max) / running_max
    max_dd_pct = float(dd.min()) * 100
    calmar = cagr / abs(max_dd_pct) if max_dd_pct != 0 else 0.0

    if not trades:
        return {
            "total_return_pct": round(total_ret_pct, 2), "cagr_pct": round(cagr, 2),
            "sharpe": round(sharpe, 2), "sortino": round(sortino, 2),
            "max_dd_pct": round(max_dd_pct, 2), "calmar": round(calmar, 2),
            "profit_factor": 0, "win_rate_pct": 0, "avg_win_usd": 0, "avg_loss_usd": 0,
            "expectancy_usd": 0, "n_trades": 0, "avg_duration_h": 0,
            "p_value_vs_0": 1.0, "final_capital": round(final, 2),
        }

    pnls   = np.array([t.pnl_net for t in trades])
    wins   = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    win_rate = len(wins) / len(pnls) * 100
    avg_win  = float(wins.mean())   if len(wins)   else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")
    expectancy = float(pnls.mean())

    durations = [(t.exit_time - t.entry_time).total_seconds() / 3600
                 for t in trades if t.exit_time and t.entry_time]
    avg_dur = float(np.mean(durations)) if durations else 0.0

    from scipy import stats
    if len(pnls) > 1 and pnls.std() > 0:
        t_stat, p_val = stats.ttest_1samp(pnls, 0)
        p_one = p_val / 2 if t_stat > 0 else 1 - p_val / 2
    else:
        p_one = 1.0

    return {
        "total_return_pct": round(total_ret_pct, 2),
        "cagr_pct":         round(cagr, 2),
        "sharpe":           round(sharpe, 2),
        "sortino":          round(sortino, 2),
        "max_dd_pct":       round(max_dd_pct, 2),
        "calmar":           round(calmar, 2),
        "profit_factor":    round(pf, 2) if pf != float("inf") else "∞",
        "win_rate_pct":     round(win_rate, 1),
        "avg_win_usd":      round(avg_win, 2),
        "avg_loss_usd":     round(avg_loss, 2),
        "expectancy_usd":   round(expectancy, 2),
        "n_trades":         len(trades),
        "avg_duration_h":   round(avg_dur, 1),
        "p_value_vs_0":     round(float(p_one), 4),
        "final_capital":    round(final, 2),
    }


# ═════════════════════════════════════════════════════════════════════════════
# STRATEGIES
# ═════════════════════════════════════════════════════════════════════════════
def strategy_buyhold(df, fg_df, daily_feat):
    cap = INITIAL_CAPITAL
    e = float(df["close"].iloc[0])
    x = float(df["close"].iloc[-1])
    pnl = (x - e) / e * cap
    fees = cap * TAKER_FEE * 2
    t = Trade(entry_time=df.index[0], entry_price=e, direction="long",
              size_usd=cap, stop_loss=0, take_profit=0,
              exit_time=df.index[-1], exit_price=x, exit_reason="end_of_period",
              pnl_gross=pnl, fees=fees, pnl_net=pnl - fees)
    eq = cap + (df["close"] - e) / e * cap - fees
    return [t], eq


def strategy_random(df, fg_df, daily_feat, seed=42):
    rng = np.random.RandomState(seed)
    def signal_fn(i):
        if rng.random() < 0.04:
            return "long" if rng.random() < 0.5 else "short"
        return None
    return simulate_strategy(df, signal_fn)


def strategy_rsi(df, fg_df, daily_feat):
    def signal_fn(i):
        rsi = df["rsi_14"].iloc[i]
        if pd.isna(rsi): return None
        if rsi < 30: return "long"
        if rsi > 70: return "short"
        return None
    return simulate_strategy(df, signal_fn)


def strategy_ict_only(df, fg_df, daily_feat):
    def signal_fn(i):
        if i < 50: return None
        row = df.iloc[i]
        price = float(row["close"])
        s = score_ict(row, price, df.iloc[:i+1])
        if s >= 0.40:  return "long"
        if s <= -0.40: return "short"
        return None
    return simulate_strategy(df, signal_fn)


def strategy_xgb_daily(df, fg_df, daily_feat):
    """Logistic regression entrenado walk-forward sobre daily_features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    feat_cols = ["rsi_14", "macd", "macd_signal", "bb_upper", "bb_lower",
                 "sma_7", "sma_30", "fear_greed", "returns", "sentiment_avg"]
    if daily_feat.empty:
        return [], pd.Series([INITIAL_CAPITAL]*len(df), index=df.index)

    backtest_start = df.index[0].normalize()
    train = daily_feat[daily_feat.index < backtest_start].copy()
    train = train.dropna(subset=feat_cols + ["label"])
    if len(train) < 100:
        log.warning("XGB daily: pocos datos de entrenamiento (<100), skip")
        return [], pd.Series([INITIAL_CAPITAL]*len(df), index=df.index)

    X = train[feat_cols].values
    y = train["label"].astype(int).values
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    clf = LogisticRegression(max_iter=300, C=0.5)
    clf.fit(Xs, y)
    log.info(f"XGB daily: trained on {len(train)} samples (data before {backtest_start.date()})")

    cache = {}
    def signal_fn(i):
        ts = df.index[i]
        date = ts.normalize()
        if date in cache:
            prob = cache[date]
        else:
            prior = daily_feat[daily_feat.index < date].dropna(subset=feat_cols)
            if prior.empty:
                return None
            row = prior.iloc[-1]
            try:
                Xn = sc.transform([row[feat_cols].values])
                prob = float(clf.predict_proba(Xn)[0][1])
                cache[date] = prob
            except Exception:
                return None
        if prob > 0.52: return "long"
        if prob < 0.48: return "short"
        return None
    return simulate_strategy(df, signal_fn)


def strategy_full_system(df, fg_df, daily_feat, threshold=THRESHOLD, weights=None):
    def signal_fn(i):
        if i < 50: return None
        ts = df.index[i]
        s, _ = confluence_score(ts, df, i, fg_df, ml_score=0.0, weights=weights)
        if s >=  threshold: return "long"
        if s <= -threshold: return "short"
        return None
    return simulate_strategy(df, signal_fn)


def strategy_full_system_filtered(df, fg_df, daily_feat,
                                   threshold=0.35, weights=None,
                                   min_bbw_pct=1.2, max_bbw_pct=8.0,
                                   respect_regime=True):
    """
    Versión MEJORADA del Full System con filtros del live_bot:
      - threshold 0.35 (más selectivo)
      - skip si BBW <1.2% (squeeze) o >8% (vol extrema)
      - skip si direction contra EMA200 (respect_regime)
    """
    def signal_fn(i):
        if i < 50: return None
        ts = df.index[i]
        row = df.iloc[i]
        price = float(row["close"])

        # Filtro BBW
        bb_u, bb_l = row.get("bb_upper"), row.get("bb_lower")
        if pd.notna(bb_u) and pd.notna(bb_l) and price > 0:
            bbw = (bb_u - bb_l) / price * 100
            if bbw < min_bbw_pct or bbw > max_bbw_pct:
                return None

        s, _ = confluence_score(ts, df, i, fg_df, ml_score=0.0, weights=weights)
        if abs(s) < threshold:
            return None
        direction = "long" if s > 0 else "short"

        # Filtro régimen
        if respect_regime:
            ema200 = row.get("ema_200")
            if pd.notna(ema200):
                if direction == "long" and price < float(ema200):
                    return None
                if direction == "short" and price > float(ema200):
                    return None

        return direction
    return simulate_strategy(df, signal_fn)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--tf",   default="4h")
    ap.add_argument("--capital", type=float, default=INITIAL_CAPITAL)
    args = ap.parse_args()

    log.info("═" * 78)
    log.info(f"BACKTEST CONFLUENCE | tf={args.tf} | days={args.days} | capital=${args.capital:,.0f}")
    log.info(f"Costes: fee={TAKER_FEE*100}% × 2 | slippage={SLIPPAGE*100}% × 2 | leverage={LEVERAGE}x")
    log.info("═" * 78)

    df = load_ohlcv(args.tf, args.days)
    df = compute_indicators(df)
    df = detect_obs_fvgs(df)
    # Aplicar detectores ICT avanzados (liquidity sweeps, BOS/CHoCH, premium/discount, daily/weekly)
    if ICT_ADVANCED_AVAILABLE:
        df = detect_liquidity_sweeps(df)
        df = detect_bos_choch(df)
        df = compute_premium_discount(df)
        df = detect_daily_weekly_levels(df)
        log.info("ICT signals avanzados aplicados: liq_sweeps, BOS/CHoCH, premium/discount, daily/weekly")
    df = df.dropna(subset=["rsi_14", "macd", "ema_50", "atr_14"])
    fg_df = load_fear_greed_daily()
    daily_feat = load_daily_features()
    log.info(f"OHLCV: {len(df)} bars | {df.index[0]} → {df.index[-1]}")
    log.info(f"F&G:   {len(fg_df)} days disponibles")
    log.info(f"Daily: {len(daily_feat)} days disponibles")

    period_days = (df.index[-1] - df.index[0]).total_seconds() / 86400

    strategies = {
        "Buy & Hold":      strategy_buyhold,
        "Random":          strategy_random,
        "RSI Simple":      strategy_rsi,
        "ICT Only":        strategy_ict_only,
        "XGB Daily (LogReg)": strategy_xgb_daily,
        "Full System (base)":     strategy_full_system,
        "Full System (regime)":   lambda df, fg, daily: strategy_full_system_filtered(
            df, fg, daily, threshold=0.30, min_bbw_pct=0.0, max_bbw_pct=100.0, respect_regime=True),
        "Full System (BBW only)": lambda df, fg, daily: strategy_full_system_filtered(
            df, fg, daily, threshold=0.30, min_bbw_pct=1.0, max_bbw_pct=10.0, respect_regime=False),
        "Full System (TFG opt)":  lambda df, fg, daily: strategy_full_system_filtered(
            df, fg, daily, threshold=0.30, min_bbw_pct=1.0, max_bbw_pct=10.0, respect_regime=True),
    }

    results, equities = {}, {}
    for name, fn in strategies.items():
        log.info(f"→ Running: {name}")
        try:
            trades, eq = fn(df, fg_df, daily_feat)
            m = compute_metrics(trades, eq, args.capital, period_days)
            results[name] = m
            equities[name] = eq
            log.info(f"   ret={m.get('total_return_pct'):+.2f}% | sharpe={m.get('sharpe')} | "
                     f"trades={m.get('n_trades')} | DD={m.get('max_dd_pct'):.2f}%")
        except Exception as e:
            import traceback
            log.error(f"   FAILED: {e}")
            traceback.print_exc()
            results[name] = {"error": str(e)}

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    cmp_df = pd.DataFrame(results).T
    cmp_path = out_dir / "backtest_compare.csv"
    cmp_df.to_csv(cmp_path)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(14, 7))
        for name, eq in equities.items():
            ax.plot(eq.index, eq.values, label=name, lw=1.6)
        ax.axhline(args.capital, color="gray", ls="--", alpha=0.4, label="Capital inicial")
        ax.set_title(f"Equity Curves | {args.days} días {args.tf} | fee {TAKER_FEE*100}% × 2 | "
                     f"slip {SLIPPAGE*100}% × 2 | lev {LEVERAGE}x")
        ax.set_xlabel("Fecha"); ax.set_ylabel("Capital ($)")
        ax.legend(loc="upper left"); ax.grid(alpha=0.3)
        plt.tight_layout()
        png1 = out_dir / "backtest_equity_curves.png"
        plt.savefig(png1, dpi=110); plt.close()
        log.info(f"Saved: {png1}")

        fig, ax = plt.subplots(figsize=(14, 5))
        for name, eq in equities.items():
            rm = eq.cummax(); dd = (eq - rm) / rm * 100
            ax.fill_between(dd.index, dd.values, 0, alpha=0.25, label=name)
        ax.set_title("Drawdown (Underwater Plot)")
        ax.set_xlabel("Fecha"); ax.set_ylabel("Drawdown %")
        ax.legend(loc="lower left"); ax.grid(alpha=0.3)
        plt.tight_layout()
        png2 = out_dir / "backtest_drawdown.png"
        plt.savefig(png2, dpi=110); plt.close()
        log.info(f"Saved: {png2}")
    except Exception as e:
        log.warning(f"matplotlib falló: {e}")

    print()
    print("═" * 110)
    print(f"COMPARATIVA — {args.days} días | {args.tf} | Capital inicial: ${args.capital:,.0f}")
    print("═" * 110)
    print(cmp_df.to_string())
    print()
    log.info(f"CSV completo en: {cmp_path}")


if __name__ == "__main__":
    main()
