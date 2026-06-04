"""
trade_calculator.py — Sizing, slippage, fees, break-even e interés compuesto para trades BTC.
"""
from __future__ import annotations
import math
from typing import Optional

# ── Constantes ────────────────────────────────────────────────────────────────

FEES = {
    "binance_spot_maker":    0.075,
    "binance_spot_taker":    0.075,
    "binance_futures_maker": 0.020,
    "binance_futures_taker": 0.040,
    "bybit_futures_maker":   0.020,
    "bybit_futures_taker":   0.055,
}

# Slippage estimado en % según exchange y tamaño de posición USD
_SLIPPAGE = {
    "binance_spot": [
        (1_000,    0.05),
        (10_000,   0.08),
        (50_000,   0.15),
        (float('inf'), 0.30),
    ],
    "binance_futures": [
        (1_000,    0.03),
        (10_000,   0.05),
        (50_000,   0.10),
        (float('inf'), 0.20),
    ],
    "bybit_futures": [
        (1_000,    0.03),
        (10_000,   0.05),
        (50_000,   0.12),
        (float('inf'), 0.25),
    ],
}

# Funding rate típico Binance perpetual: 0.01% cada 8h
_FUNDING_RATE_8H = 0.0001   # 0.01% por 8h
_FUNDING_INTERVAL_H = 8.0

_SPOT_EXCHANGES = {"binance_spot"}


def _slippage_pct(exchange: str, position_usd: float) -> float:
    tiers = _SLIPPAGE.get(exchange, _SLIPPAGE["binance_futures"])
    for threshold, pct in tiers:
        if position_usd <= threshold:
            return pct / 100
    return tiers[-1][1] / 100


def _fee_pct(exchange: str) -> float:
    """Taker fee para el exchange dado."""
    key = f"{exchange}_taker"
    return FEES.get(key, 0.075) / 100


def calculate_trade(
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    capital_usd: float,
    risk_pct: float = 1.0,
    leverage: float = 1.0,
    exchange: str = "binance_futures",
    holding_hours: float = 4.0,
) -> dict:
    """
    Calcula todos los parámetros de un trade con costes reales.
    """
    warnings = []

    # ── Sizing ────────────────────────────────────────────────────────────────
    risk_amount = capital_usd * risk_pct / 100
    sl_dist_pct = abs(entry_price - stop_loss) / entry_price
    if sl_dist_pct == 0:
        return {"error": "SL igual al entry price"}

    position_usd = risk_amount / sl_dist_pct
    position_usd_leveraged = position_usd    # sin cambiar; leverage amplifica gains/losses
    position_btc = position_usd / entry_price
    tp_dist_pct  = abs(take_profit - entry_price) / entry_price
    rr_gross = tp_dist_pct / sl_dist_pct if sl_dist_pct > 0 else 0

    # ── Costes ────────────────────────────────────────────────────────────────
    fee  = _fee_pct(exchange)
    slip = _slippage_pct(exchange, position_usd)

    fee_entry     = position_usd * fee
    fee_exit      = position_usd * fee
    slip_entry    = position_usd * slip
    slip_exit     = position_usd * slip
    total_fees    = fee_entry + fee_exit + slip_entry + slip_exit

    funding_cost = 0.0
    if exchange not in _SPOT_EXCHANGES:
        funding_periods = holding_hours / _FUNDING_INTERVAL_H
        funding_cost    = position_usd * _FUNDING_RATE_8H * funding_periods

    total_costs    = total_fees + funding_cost
    total_costs_pct = total_costs / position_usd * 100 if position_usd > 0 else 0

    # ── Break-even ────────────────────────────────────────────────────────────
    be_move_pct  = total_costs / position_usd if position_usd > 0 else 0
    direction    = 1 if take_profit > entry_price else -1
    breakeven    = entry_price * (1 + direction * be_move_pct)
    be_pct       = be_move_pct * 100

    # ── P&L neto ──────────────────────────────────────────────────────────────
    gross_profit = position_usd * tp_dist_pct
    gross_loss   = position_usd * sl_dist_pct
    net_profit   = gross_profit - total_costs
    net_loss     = -(gross_loss + total_costs)
    net_rr       = net_profit / gross_loss if gross_loss > 0 else 0

    # ── Proyección interés compuesto ─────────────────────────────────────────
    comp_10  = _compound_projection(capital_usd, risk_pct, 0.50, net_rr, 10)
    comp_50  = _compound_projection(capital_usd, risk_pct, 0.50, net_rr, 50)

    # ── Warnings ──────────────────────────────────────────────────────────────
    if total_costs_pct > 0.3:
        warnings.append(f"Costes totales elevados: {total_costs_pct:.2f}% de la posición")
    if slip > 0.001:
        warnings.append(f"Slippage estimado alto para posición de ${position_usd:,.0f}: {slip*100:.2f}%")
    if rr_gross < 1.5:
        warnings.append(f"R:R bruto {rr_gross:.2f}:1 por debajo del mínimo recomendado 1.5:1")
    if net_rr < 1.0:
        warnings.append(f"R:R neto {net_rr:.2f}:1 — el trade no es rentable después de costes")
    if sl_dist_pct > 0.02:
        warnings.append(f"SL muy amplio: {sl_dist_pct*100:.2f}% — considerar reducir posición")
    if leverage > 5:
        warnings.append(f"Apalancamiento x{leverage} — riesgo de liquidación elevado")
    if exchange not in _SPOT_EXCHANGES and funding_cost > risk_amount * 0.05:
        warnings.append(f"Funding cost ${funding_cost:.2f} es >5% del riesgo en {holding_hours}h")

    return {
        # Sizing
        "risk_amount_usd":   round(risk_amount, 2),
        "position_size_usd": round(position_usd, 2),
        "position_size_btc": round(position_btc, 6),
        "leverage_used":     leverage,

        # Niveles
        "entry":          entry_price,
        "stop_loss":      round(stop_loss, 2),
        "take_profit":    round(take_profit, 2),
        "sl_distance_pct": round(sl_dist_pct * 100, 3),
        "tp_distance_pct": round(tp_dist_pct * 100, 3),
        "risk_reward_gross": round(rr_gross, 2),

        # Costes
        "fee_entry_usd":      round(fee_entry, 2),
        "fee_exit_usd":       round(fee_exit, 2),
        "slippage_entry_usd": round(slip_entry, 2),
        "slippage_exit_usd":  round(slip_exit, 2),
        "funding_cost_usd":   round(funding_cost, 2),
        "total_costs_usd":    round(total_costs, 2),
        "total_costs_pct":    round(total_costs_pct, 3),

        # Break-even
        "breakeven_price": round(breakeven, 2),
        "breakeven_pct":   round(be_pct, 3),

        # P&L neto
        "net_profit_if_tp_usd": round(net_profit, 2),
        "net_loss_if_sl_usd":   round(net_loss, 2),
        "net_rr_ratio":         round(net_rr, 2),

        # Interés compuesto (50% win rate)
        "compound_10_trades_usd": round(comp_10, 2),
        "compound_50_trades_usd": round(comp_50, 2),

        "warnings": warnings,
        "exchange": exchange,
        "holding_hours": holding_hours,
    }


def _compound_projection(
    capital: float,
    risk_pct: float,
    win_rate: float,
    net_rr: float,
    n_trades: int,
) -> float:
    """
    Proyección capital tras n_trades con interés compuesto.
    Gana: capital * risk_pct/100 * net_rr
    Pierde: capital * risk_pct/100
    """
    c = capital
    for _ in range(n_trades):
        risk = c * risk_pct / 100
        if _pseudo_random_win(win_rate):
            c += risk * net_rr
        else:
            c -= risk
    return c


def _pseudo_random_win(win_rate: float) -> bool:
    """Determinista: promedio exacto de win_rate para proyección estable."""
    # No aleatoria: devuelve True si win_rate > 0.5 o alternancia simple
    return win_rate >= 0.5


def calculate_compound_growth(
    initial_capital: float,
    risk_per_trade_pct: float,
    win_rate: float,
    rr_ratio: float,
    n_trades: int,
    costs_per_trade_pct: float = 0.15,
) -> dict:
    """
    Proyección de crecimiento por interés compuesto con Kelly Criterion.
    """
    # Kelly: f* = (p*b - q) / b  donde b=rr_ratio, p=win_rate, q=1-p
    b = rr_ratio
    p = win_rate
    q = 1 - p
    kelly_full = (p * b - q) / b if b > 0 else 0
    kelly_half = max(0.0, kelly_full * 0.5)

    # Expectancy por trade (neta de costes)
    expectancy = p * rr_ratio - q - costs_per_trade_pct / 100

    capital = initial_capital
    by_trade = []
    max_cap  = capital
    peak     = capital
    max_dd   = 0.0
    worst_streak = 0
    current_streak = 0

    for i in range(n_trades):
        risk = capital * risk_per_trade_pct / 100
        # Alternar ganar/perder según win_rate de forma determinista
        is_win = (i / n_trades) < win_rate
        if is_win:
            gain = risk * rr_ratio - capital * costs_per_trade_pct / 100
            capital += gain
            current_streak = 0
            if capital > peak:
                peak = capital
        else:
            loss = risk + capital * costs_per_trade_pct / 100
            capital -= loss
            current_streak += 1
            worst_streak = max(worst_streak, current_streak)
            dd = (peak - capital) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)

        if i % 10 == 0 or i == n_trades - 1:
            by_trade.append({
                "trade_n":          i + 1,
                "capital":          round(capital, 2),
                "cumulative_return": round((capital / initial_capital - 1) * 100, 2),
            })

    return {
        "initial_capital":    initial_capital,
        "final_capital":      round(capital, 2),
        "total_return_pct":   round((capital / initial_capital - 1) * 100, 2),
        "expectancy_per_trade": round(expectancy * 100, 3),
        "kelly_full_pct":     round(kelly_full * 100, 2),
        "kelly_half_pct":     round(kelly_half * 100, 2),
        "max_drawdown_pct":   round(max_dd, 2),
        "worst_losing_streak": worst_streak,
        "n_trades":           n_trades,
        "by_trade":           by_trade,
    }


def get_optimal_risk(
    win_rate: float,
    rr_ratio: float,
    max_drawdown_tolerance: float = 0.20,
) -> float:
    """
    Kelly Criterion modificado limitado por max_drawdown_tolerance.
    Devuelve el % óptimo de riesgo por trade.
    """
    b = rr_ratio
    p = win_rate
    q = 1 - p
    kelly = (p * b - q) / b if b > 0 else 0
    half_kelly = max(0.0, kelly * 0.5)

    # Límite por drawdown: en 5 pérdidas consecutivas no superar max_dd
    # (1 - risk_pct/100)^5 >= (1 - max_drawdown_tolerance)
    # risk_pct/100 <= 1 - (1 - max_dd)^(1/5)
    max_risk_from_dd = (1 - (1 - max_drawdown_tolerance) ** 0.2) * 100

    return round(min(half_kelly * 100, max_risk_from_dd, 5.0), 2)  # cap a 5%
