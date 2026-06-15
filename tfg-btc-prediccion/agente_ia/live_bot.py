"""
live_bot.py — Bot de paper trading en vivo con Bybit Demo.

Uso:
  python agente_ia/live_bot.py              # arrancar bot
  python agente_ia/live_bot.py --status     # mostrar estado
  python agente_ia/live_bot.py --stop       # parar bot
  python agente_ia/live_bot.py --close-all  # cerrar todas las posiciones
  python agente_ia/live_bot.py --dry-run    # una iteración sin ejecutar órdenes reales

Doble trigger:
  1. HORARIO: al inicio de cada killzone (London 07:00-09:30, NY 13:00-14:30 UTC)
  2. ALERTAS: cuando el precio toca un OB/FVG precalculado
"""
from __future__ import annotations

# ── CRÍTICO: desactivar TF/Keras ANTES de cualquier import de HuggingFace ──
# transformers detecta tensorflow + keras 3 e intenta cargar el backend TF
# (requiere tf-keras, no instalado) → crash. El proyecto usa solo PyTorch.
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import argparse
import json
import logging
import re
import signal
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / "config" / ".env")

from sqlalchemy import text
from db.db_utils import get_engine

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).parent / "live_bot.log", encoding="utf-8"),
    ],
)

PID_FILE = Path(__file__).parent / ".live_bot.pid"

# ── Configuración ─────────────────────────────────────────────────────────────

LIVE_CONFIG = {
    "symbol":              "BTCUSDT",
    "initial_capital":     50000.0,
    "risk_per_trade_pct":  1.0,
    "leverage":            3.0,
    "max_concurrent":      1,
    "max_trades_per_day":  4,        # antes 5 — más selectivo
    "max_agent_calls_day": 5,        # antes 15 — coste gpt-4o; concentrar en momentos clave
    "skip_weekends":       True,     # no operar sábado/domingo (baja liquidez)
    "min_score_to_call":   0.30,     # ratio óptimo según backtest 90d (threshold sweet spot)
    "sl_atr_mult":         2.0,      # antes 1.5 — menos whipsaw, mejor expectancy
    "tp_atr_mult":         4.0,      # antes 3.0 — R:R 2:1 gross (probado +2.6% return vs antes)
    "max_holding_hours":   24,
    # ── Asegurar beneficio: breakeven + trailing stop (mueve el SL NATIVO) ────
    "secure_be_pct":       0.5,      # a +0.5% de beneficio → SL a breakeven (no se puede perder)
    "trail_dist_pct":      0.4,      # luego el SL sigue al precio a 0.4% por detrás del máximo
    "fee_buffer_pct":      0.12,     # colchón sobre la entrada para cubrir fees en el breakeven
    # ── Filtros para evitar entradas en mercado adverso ──────────────────────
    "min_bbw_pct":         1.0,      # BB width / price >= 1.0% (skip squeeze fuerte)
    "max_bbw_pct":         10.0,     # BB width / price <= 10% (skip vol extrema)
    "respect_regime":      True,     # solo long si price>EMA200, solo short si price<EMA200
    "cooldown_min_loss":   120,      # 2h cooldown tras stop loss (anti-tilt)
    "cooldown_min_win":    30,       # 30min cooldown tras TP
    "killzones": [
        {"name": "london_open", "start_h": 7,  "start_m": 0,  "end_h": 9,  "end_m": 30},
        {"name": "ny_open",     "start_h": 13, "start_m": 0,  "end_h": 14, "end_m": 30},
    ],
    "check_interval_sec":  60,
    "alert_regen_min":     60,
}

# ── Helpers de tiempo ─────────────────────────────────────────────────────────

def _in_killzone(now: datetime, kz: dict) -> bool:
    t = now.hour * 60 + now.minute
    s = kz["start_h"] * 60 + kz["start_m"]
    e = kz["end_h"]   * 60 + kz["end_m"]
    return s <= t < e


def _killzone_just_started(now: datetime, kz: dict, window_min: int = 5) -> bool:
    """True si estamos en los primeros `window_min` de la killzone."""
    t = now.hour * 60 + now.minute
    s = kz["start_h"] * 60 + kz["start_m"]
    return s <= t < s + window_min


# ── Bot state DB ──────────────────────────────────────────────────────────────

def _load_state(engine) -> dict:
    with engine.connect() as conn:
        row = conn.execute(text("SELECT * FROM bot_state WHERE id=1")).fetchone()
    if not row:
        return {}
    return dict(row._mapping)


def _update_state(engine, **kwargs):
    sets = ", ".join(f"{k}=:{k}" for k in kwargs)
    kwargs["id"] = 1
    with engine.begin() as conn:
        conn.execute(text(f"UPDATE bot_state SET {sets} WHERE id=:id"), kwargs)


def _get_today_agent_calls(engine) -> int:
    state = _load_state(engine)
    last_call = state.get("last_agent_call")
    today = datetime.now(timezone.utc).date()
    if last_call and hasattr(last_call, 'date') and last_call.date() == today:
        return int(state.get("agent_calls_today", 0))
    # Nuevo día: reset
    _update_state(engine, agent_calls_today=0)
    return 0


# ── Modules lazy load ─────────────────────────────────────────────────────────

def _load_tools():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "tools05", Path(__file__).parent / "05_tools.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_agent():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "agent06", Path(__file__).parent / "06_agent.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_bybit():
    from agente_ia.bybit_client import BybitDemoClient
    return BybitDemoClient()


def _load_alerts():
    from agente_ia.alert_generator import check_alerts, generate_alerts
    return check_alerts, generate_alerts


# ── Decisión de llamar al agente ──────────────────────────────────────────────

def should_call_agent(
    now: datetime,
    current_price: float,
    engine,
    config: dict,
    last_kz_called: dict,
) -> tuple[bool, str, str]:
    """
    Decide si llamar al agente AHORA.
    Returns (should_call, trigger_type, trigger_detail)
    """
    # 0. Fin de semana: no operar (sábado=5, domingo=6 en UTC)
    if config.get("skip_weekends") and now.weekday() >= 5:
        return False, "", "fin de semana (skip_weekends)"

    state = _load_state(engine)
    calls_today = int(state.get("agent_calls_today", 0))
    if calls_today >= config["max_agent_calls_day"]:
        return False, "", "max_agent_calls_day alcanzado"

    # 1. ¿Estamos al inicio de una killzone y no hemos llamado en ella?
    for kz in config["killzones"]:
        kz_key = kz["name"] + "_" + now.strftime("%Y%m%d")
        if _killzone_just_started(now, kz) and not last_kz_called.get(kz_key):
            last_kz_called[kz_key] = True
            return True, "killzone", f"Inicio killzone {kz['name']}"

    # 2. ¿El precio tocó alguna alerta?
    check_fn, _ = _load_alerts()
    triggered = check_fn(current_price, engine)
    if triggered:
        best = triggered[0]
        detail = f"Precio tocó {best['level_type']} {best['timeframe']} @ {best['level_price']:.2f}"
        return True, "level_alert", detail

    return False, "", ""


# ── Parsear decisión del agente ───────────────────────────────────────────────

def _parse_agent_decision(response: str) -> dict:
    """
    Extrae el bloque JSON de decisión del agente.
    Busca el último bloque ```json ... ``` en la respuesta.
    """
    pattern = r'```json\s*([\s\S]*?)\s*```'
    matches = re.findall(pattern, response, re.IGNORECASE)
    if matches:
        for m in reversed(matches):
            try:
                d = json.loads(m)
                if "decision" in d:
                    return d
            except Exception:
                continue

    # Fallback: buscar directamente un JSON con "decision"
    try:
        j_match = re.search(r'\{[^{}]*"decision"[^{}]*\}', response, re.DOTALL)
        if j_match:
            return json.loads(j_match.group())
    except Exception:
        pass

    # No encontrado: interpretar por palabras clave
    resp_lower = response.lower()
    if "no_trade" in resp_lower or "no trade" in resp_lower:
        return {"decision": "NO_TRADE", "reasoning": "Agente rechazó el trade"}
    if "long" in resp_lower and ("trade" in resp_lower or "abr" in resp_lower):
        return {"decision": "TRADE", "direction": "long", "confidence": 0.6,
                "reasoning": "Interpretado de respuesta (sin JSON)"}
    if "short" in resp_lower and ("trade" in resp_lower or "abr" in resp_lower):
        return {"decision": "TRADE", "direction": "short", "confidence": 0.6,
                "reasoning": "Interpretado de respuesta (sin JSON)"}
    return {"decision": "NO_TRADE", "reasoning": "Sin decisión clara en respuesta"}


# ── Gestión de posiciones abiertas ────────────────────────────────────────────

def manage_open_positions(engine, config: dict, bybit, tools_mod, dry_run: bool = False):
    """Verifica posiciones abiertas, SL/TP hit, timeout y señal contraria."""
    now = datetime.now(timezone.utc)
    with engine.connect() as conn:
        open_trades = conn.execute(text("""
            SELECT id, entry_time, entry_price, direction, stop_loss, take_profit,
                   bybit_order_id, holding_hours, size_usd
            FROM live_trades WHERE status='open'
        """)).fetchall()

    for trade in open_trades:
        tid, entry_time, entry_price, direction, sl, tp, order_id, max_hold, size_usd_trade = trade

        # Precio actual
        try:
            price_now = bybit.get_price() if bybit.is_configured() else \
                tools_mod.dispatch_tool("query_market", {"timeframe": "1h"}).get("price_now", 0)
        except Exception:
            continue

        # Calcular PnL actual
        mult = 1.0 if direction == "long" else -1.0
        pnl_pct = mult * (price_now - entry_price) / entry_price * 100

        # ── Asegurar beneficio: breakeven + trailing del SL NATIVO ────────────
        # A partir de secure_be_pct subimos el SL a breakeven; luego lo arrastramos
        # a trail_dist_pct por detrás del precio. El SL solo MEJORA (nunca empeora),
        # y se escribe en Bybit (set-trading-stop) para que proteja aunque el bot
        # se apague. Cuando el precio retroceda, el propio SL nativo cierra y asegura.
        try:
            be_pct    = config.get("secure_be_pct", 0.5)
            trail_pct = config.get("trail_dist_pct", 0.4)
            fee_buf   = config.get("fee_buffer_pct", 0.12)
            if pnl_pct >= be_pct and sl:
                if direction == "long":
                    be_price = entry_price * (1 + fee_buf / 100)
                    trail_sl = price_now * (1 - trail_pct / 100)
                    new_sl   = round(max(be_price, trail_sl, sl), 2)
                    improved = new_sl > sl + 0.01
                else:
                    be_price = entry_price * (1 - fee_buf / 100)
                    trail_sl = price_now * (1 + trail_pct / 100)
                    new_sl   = round(min(be_price, trail_sl, sl), 2)
                    improved = new_sl < sl - 0.01
                if improved:
                    ok = True
                    if not dry_run and bybit.is_configured():
                        res = bybit.update_stop_loss(new_sl)
                        ok = not res.get("error")
                    if ok:
                        with engine.begin() as conn:
                            conn.execute(text("UPDATE live_trades SET stop_loss=:sl WHERE id=:id"),
                                         {"sl": new_sl, "id": tid})
                        log.info("Trade #%d: SL %.2f -> %.2f (pnl %.2f%%, asegurando beneficio)",
                                 tid, sl, new_sl, pnl_pct)
                        sl = new_sl  # el check de abajo usa el SL ya actualizado
        except Exception as e:
            log.warning("Trade #%d: trailing SL falló: %s", tid, e)

        # ¿SL o TP hit?
        exit_reason = None
        if direction == "long":
            if price_now <= sl:
                exit_reason = "sl_hit"
            elif price_now >= tp:
                exit_reason = "tp_hit"
        else:
            if price_now >= sl:
                exit_reason = "sl_hit"
            elif price_now <= tp:
                exit_reason = "tp_hit"

        # Timeout
        entry_dt = entry_time if hasattr(entry_time, 'tzinfo') else entry_time.replace(tzinfo=timezone.utc)
        holding_h = (now - entry_dt).total_seconds() / 3600
        max_h = max_hold or config.get("max_holding_hours", 24)
        if holding_h >= max_h:
            exit_reason = "timeout"

        if exit_reason:
            if not dry_run and bybit.is_configured():
                bybit.close_position(direction)

            pnl_usd   = (price_now - entry_price) * mult
            # Bybit taker fee 0.055% × 2 sides × notional
            notional  = float(size_usd_trade or 0)
            fees_usd  = round(notional * 0.00055 * 2, 4) if notional else 0.0
            net_pnl   = round(pnl_usd - fees_usd, 4)
            with engine.begin() as conn:
                conn.execute(text("""
                    UPDATE live_trades
                    SET exit_time=:now, exit_price=:price, exit_reason=:reason,
                        pnl_usd=:pnl, pnl_pct=:pct, status='closed',
                        holding_hours=:hold, fees_usd=:fees, net_pnl_usd=:net_pnl
                    WHERE id=:id
                """), {
                    "now": now, "price": price_now, "reason": exit_reason,
                    "pnl": round(pnl_usd, 2), "pct": round(pnl_pct, 3),
                    "hold": round(holding_h, 2), "fees": fees_usd,
                    "net_pnl": net_pnl, "id": tid,
                })
                # Actualizar capital en bot_state
                conn.execute(text("""
                    UPDATE bot_state SET current_capital = current_capital + :net_pnl WHERE id=1
                """), {"net_pnl": net_pnl})
            log.info("Trade #%d cerrado: %s @ %.2f | PnL %.2f%% ($%.2f bruto, $%.2f neto) | razón: %s",
                     tid, direction, price_now, pnl_pct, pnl_usd, net_pnl, exit_reason)

    # Actualizar open_trades en bot_state
    with engine.connect() as conn:
        n_open = conn.execute(text("SELECT COUNT(*) FROM live_trades WHERE status='open'")).scalar()
    _update_state(engine, open_trades=n_open)


# ── Filtros de mercado (anti-chop, anti-tilt) ─────────────────────────────────

def _market_filters_ok(cs: dict, direction_hint: str, config: dict) -> tuple[bool, str]:
    """
    Filtros previos a llamar al agente:
      1. Bollinger Band Width dentro de banda razonable (no squeeze, no extreme vol)
      2. Regime check: dirección coherente con EMA200 si respect_regime=True
    Returns (ok, reason_if_not_ok)
    """
    price = float(cs.get("price_now") or 0)
    if price <= 0:
        return True, ""  # no podemos validar, permitir

    mods = cs.get("modules") or {}
    # Sacar indicadores técnicos del módulo technical si están
    tech_sigs = (mods.get("technical") or {}).get("signals") or []

    # Buscar BBW del módulo (signals contienen strings tipo "Precio cerca BB inferior")
    # No tenemos BBW directo, pedimos al tool query_market
    try:
        import importlib.util as _ilu
        from pathlib import Path as _P
        tools_path = _P(__file__).parent / "05_tools.py"
        spec = _ilu.spec_from_file_location("t05", tools_path)
        t = _ilu.module_from_spec(spec); spec.loader.exec_module(t)
        mkt = t.query_market(timeframe="1h", candles=2)
        ind = mkt.get("indicators") or {}
        bb_u, bb_l = ind.get("bb_upper"), ind.get("bb_lower")
        ema_200 = ind.get("ema_200") or ind.get("sma_30")
        if bb_u and bb_l and price > 0:
            bbw_pct = (float(bb_u) - float(bb_l)) / price * 100
            if bbw_pct < config.get("min_bbw_pct", 1.2):
                return False, f"BBW {bbw_pct:.2f}% < {config['min_bbw_pct']}% (squeeze/chop)"
            if bbw_pct > config.get("max_bbw_pct", 8.0):
                return False, f"BBW {bbw_pct:.2f}% > {config['max_bbw_pct']}% (volatilidad extrema)"
        if config.get("respect_regime") and ema_200:
            if direction_hint == "long" and price < float(ema_200):
                return False, f"Long contra-tendencia (precio {price:.0f} < EMA200 {float(ema_200):.0f})"
            if direction_hint == "short" and price > float(ema_200):
                return False, f"Short contra-tendencia (precio {price:.0f} > EMA200 {float(ema_200):.0f})"
    except Exception as e:
        log.debug("market_filters check no disponible: %s", e)
    return True, ""


def _in_cooldown(engine, config: dict) -> tuple[bool, str]:
    """True si estamos en cooldown desde el último trade cerrado."""
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT exit_time, exit_reason FROM live_trades
            WHERE status='closed' AND exit_time IS NOT NULL
            ORDER BY exit_time DESC LIMIT 1
        """)).fetchone()
    if not row or not row[0]:
        return False, ""
    last_exit, reason = row[0], row[1] or ""
    if not hasattr(last_exit, "tzinfo") or last_exit.tzinfo is None:
        last_exit = last_exit.replace(tzinfo=timezone.utc)
    mins_since = (datetime.now(timezone.utc) - last_exit).total_seconds() / 60
    cooldown = config["cooldown_min_loss"] if reason == "sl_hit" else config["cooldown_min_win"]
    if mins_since < cooldown:
        return True, f"cooldown {cooldown - mins_since:.0f}min restantes (último {reason})"
    return False, ""


# ── Evaluar y ejecutar trade ──────────────────────────────────────────────────

def evaluate_and_trade(
    trigger_type: str,
    trigger_detail: str,
    engine,
    config: dict,
    bybit,
    tools_mod,
    agent_mod,
    dry_run: bool = False,
) -> bool:
    """
    1. Calcula confluence score
    2. Si |score| < umbral → salir (no gastar llamada)
    3. Llama al agente LLM
    4. Parsea decisión
    5. Abre posición si TRADE
    Devuelve True si se abrió un trade.
    """
    # Verificar posiciones abiertas
    with engine.connect() as conn:
        n_open = conn.execute(text("SELECT COUNT(*) FROM live_trades WHERE status='open'")).scalar()
    if n_open >= config.get("max_concurrent", 1):
        log.info("evaluate_and_trade: max_concurrent %d alcanzado", config["max_concurrent"])
        return False

    # Verificar trades del día
    today = datetime.now(timezone.utc).date()
    with engine.connect() as conn:
        trades_today = conn.execute(text("""
            SELECT COUNT(*) FROM live_trades
            WHERE entry_time::date = :today
        """), {"today": today}).scalar()
    if trades_today >= config.get("max_trades_per_day", 5):
        log.info("evaluate_and_trade: max_trades_per_day %d alcanzado", trades_today)
        return False

    # 1. Confluence score (no gasta llamada OpenAI)
    try:
        cs = tools_mod.dispatch_tool("get_confluence_score", {"timeframe": "1h"})
    except Exception as e:
        log.error("evaluate_and_trade: confluence score error: %s", e)
        return False

    score = cs.get("final_score", 0)
    label = cs.get("label", "NEUTRAL")
    price_now = cs.get("price_now", 0)

    if abs(score) < config.get("min_score_to_call", 0.35):
        log.info("evaluate_and_trade: score %.3f < umbral %.2f — no se llama al agente",
                 score, config["min_score_to_call"])
        return False

    # ── FILTRO 1: cooldown post-trade ──
    in_cd, cd_reason = _in_cooldown(engine, config)
    if in_cd:
        log.info("evaluate_and_trade: skip — %s", cd_reason)
        return False

    # ── FILTRO 2: market filters (BBW + regime) ──
    direction_hint = "long" if score > 0 else "short"
    ok, reason = _market_filters_ok(cs, direction_hint, config)
    if not ok:
        log.info("evaluate_and_trade: skip — %s", reason)
        return False

    log.info("evaluate_and_trade: trigger=%s score=%.3f (%s) — llamando al agente",
             trigger_type, score, label)

    # 2. Llamar al agente LLM con prompt de decisión REFORZADO
    signal_snap = json.dumps(cs, default=str)[:500]
    prompt = (
        f"[BOT_DECISION_REQUEST]\n"
        f"Trigger: {trigger_detail}\n"
        f"Confluence score cuantitativo: {score:.3f} ({label})\n"
        f"Confianza: {cs.get('confidence', 0):.2f}\n"
        f"Precio actual: {price_now:.2f} USDT\n"
        f"Signal snapshot: {signal_snap}\n\n"
        f"INSTRUCCIONES CRÍTICAS (R:R objetivo 2:1, SL 2.0×ATR, TP 4.0×ATR):\n"
        f"1. Llama get_ict_context para validar estructura y nivel de entrada cercano.\n"
        f"2. Llama get_entry_zone({direction_hint!r}) para zona de entrada exacta.\n"
        f"3. RECHAZA el trade (decision=NO_TRADE) si CUALQUIERA de estos se cumple:\n"
        f"   • Estructura ICT en CONFLICTO con la dirección del score (ej. score SELL pero structure bullish recién hecho BOS)\n"
        f"   • Sin OB o FVG válido cerca (<1.5% del precio) en la dirección del trade\n"
        f"   • Precio ya cerca de SL natural (resistencia/soporte muy próxima en contra)\n"
        f"   • Confluencia < 0.40 Y killzone no activa\n"
        f"   • Range bound apretado sin breakout claro (alta probabilidad de whipsaw)\n"
        f"4. SOLO devuelve TRADE si:\n"
        f"   - Tienes nivel ICT claro en la dirección\n"
        f"   - SL 2.0×ATR no se rompe inmediatamente por estructura cercana\n"
        f"   - Hay confluencia múltiple (al menos 3 módulos del score alineados)\n"
        f"5. Sé conservador: ante duda, NO_TRADE. R:R 2:1 requiere win-rate >35% para edge.\n\n"
        f"Devuelve el bloque JSON al final con: decision, direction, confidence, reasoning."
    )

    try:
        response, _ = agent_mod.chat(prompt, history=[], verbose=False)
    except Exception as e:
        log.error("Agent call failed: %s", e)
        return False

    # 3. Parsear decisión
    decision = _parse_agent_decision(response)
    log.info("Agent decision: %s | reasoning: %s",
             decision.get("decision"), decision.get("reasoning", "")[:100])

    if decision.get("decision") != "TRADE":
        _record_agent_call(engine, trigger_type, trigger_detail, score, label,
                           response, decision, cs, price_now)
        return False

    direction = decision.get("direction", "long")

    # 4-6. Calcular params + ejecutar (Bybit o sim) + persistir — lógica
    #      compartida con el botón manual del dashboard (trade_executor).
    from agente_ia.trade_executor import execute_trade
    result = execute_trade(
        engine, bybit, tools_mod,
        direction=direction, config=config,
        trigger_type=trigger_type, trigger_detail=trigger_detail,
        confluence=cs, agent_decision_text=response,
        dry_run=dry_run,
    )
    if not result.get("ok"):
        log.error("execute_trade falló: %s", result.get("error"))
        return False

    # 7. Registrar llamada al agente + contadores
    _record_agent_call(engine, trigger_type, trigger_detail, score, label,
                       response, decision, cs, price_now, trade_opened=True)

    mode = "[DRY RUN] " if dry_run else f"[{result.get('mode','sim').upper()}] "
    log.info("%sTrade ABIERTO: #%s %s @ %.2f", mode, result.get("trade_id"),
             direction, result.get("entry_price", 0))
    return True


def _record_agent_call(engine, trigger_type, trigger_detail, score, label,
                       response, decision, cs, price_now, trade_opened=False):
    now = datetime.now(timezone.utc)
    _update_state(engine,
                  last_agent_call=now,
                  agent_calls_today=_get_today_agent_calls(engine) + 1)


def _get_session_name(dt: datetime) -> str:
    h = dt.hour
    if 0 <= h < 8:   return "asia"
    if 7 <= h < 12:  return "london"
    if 13 <= h < 22: return "ny"
    return "dead_zone"


# ── Loop principal ────────────────────────────────────────────────────────────

_running = True


def _signal_handler(sig, frame):
    global _running
    log.info("Señal recibida (%s) — parando bot...", sig)
    _running = False


def run_bot(config: dict = None, dry_run: bool = False):
    global _running
    _running = True

    if config is None:
        config = LIVE_CONFIG

    signal.signal(signal.SIGINT,  _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # PID file
    PID_FILE.write_text(str(os.getpid()))

    engine     = get_engine()
    tools_mod  = _load_tools()
    agent_mod  = _load_agent()
    bybit      = _load_bybit()
    _, gen_alerts = _load_alerts()

    mode_str = "[DRY RUN] " if dry_run else ""
    log.info("%sBot arrancando | leverage=%.1fx | max_calls=%d",
             mode_str, config["leverage"], config["max_agent_calls_day"])

    if bybit.is_configured():
        try:
            bal = bybit.get_balance()
            log.info("Balance Bybit Demo: %.2f USDT", bal)
        except Exception as e:
            log.warning("No se pudo leer balance: %s", e)
    else:
        log.warning("Bybit keys no configuradas — operando en modo simulación local")

    # Inicializar bot_state (seed de la fila id=1 si no existe — antes faltaba
    # y _update_state hacía UPDATE sin efecto)
    from agente_ia.trade_executor import ensure_bot_state
    ensure_bot_state(engine, config["initial_capital"])
    _update_state(engine,
                  is_running=True,
                  started_at=datetime.now(timezone.utc),
                  current_capital=config["initial_capital"])

    last_alert_regen = datetime.now(timezone.utc) - timedelta(hours=2)
    last_kz_called   = {}

    # Generar alertas iniciales
    try:
        n = gen_alerts(engine)
        log.info("Alertas iniciales generadas: %d", n)
    except Exception as e:
        log.warning("Error generando alertas iniciales: %s", e)

    # Loop principal
    while _running:
        try:
            now = datetime.now(timezone.utc)

            # Precio actual
            try:
                price = bybit.get_price()
            except Exception:
                mkt = tools_mod.dispatch_tool("query_market", {"timeframe": "1h"})
                price = mkt.get("price_now", 0)

            # Gestionar posiciones abiertas
            manage_open_positions(engine, config, bybit, tools_mod, dry_run)

            # ¿Llamar al agente?
            should_call, trigger, detail = should_call_agent(
                now, price, engine, config, last_kz_called
            )

            if should_call:
                evaluate_and_trade(
                    trigger, detail, engine, config,
                    bybit, tools_mod, agent_mod, dry_run
                )

            # Regenerar alertas cada alert_regen_min
            mins_since_regen = (now - last_alert_regen).total_seconds() / 60
            if mins_since_regen >= config.get("alert_regen_min", 60):
                try:
                    n = gen_alerts(engine)
                    log.info("Alertas regeneradas: %d activas", n)
                    last_alert_regen = now
                except Exception as e:
                    log.warning("Error regenerando alertas: %s", e)

            _update_state(engine, last_check=now)

        except Exception as e:
            log.error("Error en loop principal: %s — reintentando en 60s", e)
            time.sleep(60)
            continue

        # En dry-run, ejecutar solo una iteración
        if dry_run:
            log.info("[DRY RUN] Una iteración completada — saliendo")
            break

        time.sleep(config.get("check_interval_sec", 60))

    # Cleanup
    _update_state(engine, is_running=False)
    if PID_FILE.exists():
        PID_FILE.unlink()
    log.info("Bot parado correctamente.")


# ── Comandos CLI ──────────────────────────────────────────────────────────────

def cmd_status():
    engine = get_engine()
    state  = _load_state(engine)
    print("\n=== Estado del Bot ===")
    print(f"  Running:       {state.get('is_running', False)}")
    print(f"  Started:       {state.get('started_at', 'N/A')}")
    print(f"  Capital:       {state.get('current_capital', 'N/A')} USDT")
    print(f"  Total trades:  {state.get('total_trades', 0)}")
    print(f"  Open trades:   {state.get('open_trades', 0)}")
    print(f"  Agent calls today: {state.get('agent_calls_today', 0)}")
    print(f"  Last check:    {state.get('last_check', 'N/A')}")

    with engine.connect() as conn:
        open_t = conn.execute(text("""
            SELECT direction, entry_price, stop_loss, take_profit, confluence_score
            FROM live_trades WHERE status='open'
        """)).fetchall()
    if open_t:
        print("\n  Posición abierta:")
        for t in open_t:
            print(f"    {t[0].upper()} @ {t[1]:.2f} | SL {t[2]:.2f} | TP {t[3]:.2f} | score {t[4]}")


def cmd_stop():
    if PID_FILE.exists():
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, signal.SIGTERM)
        print(f"Señal SIGTERM enviada al bot (PID {pid})")
    else:
        print("PID file no encontrado — el bot puede no estar corriendo")
        engine = get_engine()
        _update_state(engine, is_running=False)


def cmd_close_all():
    """Cierra todas las posiciones abiertas en Bybit y registra el PnL realizado
    (con el tamaño REAL de Bybit) en live_trades + capital en bot_state."""
    bybit = _load_bybit()
    engine = get_engine()
    now = datetime.now(timezone.utc)

    # 1. Leer trades abiertos en DB y tamaño/posiciones reales en Bybit ANTES de cerrar
    with engine.connect() as conn:
        open_trades = conn.execute(text("""
            SELECT id, entry_price, direction, entry_time, size_btc
            FROM live_trades WHERE status='open'
        """)).fetchall()

    real_size = None
    if bybit.is_configured():
        try:
            positions = bybit.get_open_positions()
            if len(positions) == 1:
                real_size = abs(float(positions[0].get("size", 0)))
            price = bybit.get_price()
            bybit.close_all()
            print(f"Cerradas {len(positions)} posiciones en Bybit")
        except Exception as e:
            print(f"Error cerrando en Bybit: {e}")
            price = bybit.get_price()
    else:
        print("Bybit no configurado — cerrando en DB solamente")
        price = 0

    # 2. Registrar cada trade con PnL realizado
    for t in open_trades:
        tid, entry, direction, etime, size_db = t
        size = real_size if (real_size and len(open_trades) == 1) else float(size_db or 0)
        mult = 1.0 if direction == "long" else -1.0
        pnl_usd = (price - float(entry)) * mult * size if price else 0.0
        fees    = round(float(size) * float(entry) * 0.00055 * 2, 4)
        net     = round(pnl_usd - fees, 2)
        pnl_pct = mult * (price - float(entry)) / float(entry) * 100 if price and entry else 0.0
        etime_a = etime if (etime and etime.tzinfo) else (etime.replace(tzinfo=timezone.utc) if etime else now)
        hold    = round((now - etime_a).total_seconds() / 3600, 2)
        with engine.begin() as conn:
            conn.execute(text("""
                UPDATE live_trades
                SET status='closed', exit_time=:now, exit_price=:price,
                    exit_reason='manual_close', pnl_usd=:pnl, pnl_pct=:pct,
                    fees_usd=:fees, net_pnl_usd=:net, holding_hours=:hold
                WHERE id=:id
            """), {"now": now, "price": price, "pnl": round(pnl_usd, 2),
                   "pct": round(pnl_pct, 3), "fees": fees, "net": net,
                   "hold": hold, "id": tid})
            conn.execute(text("""
                UPDATE bot_state SET current_capital = COALESCE(current_capital,50000) + :net
                WHERE id=1
            """), {"net": net})
        print(f"  Trade #{tid} {direction}: exit {price} | neto ${net} ({pnl_pct:+.2f}%)")

    _update_state(engine, open_trades=0)
    print("Todas las posiciones cerradas.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Bot — Bybit Demo Paper Trading")
    parser.add_argument("--status",    action="store_true", help="Mostrar estado del bot")
    parser.add_argument("--stop",      action="store_true", help="Parar el bot")
    parser.add_argument("--close-all", action="store_true", help="Cerrar todas las posiciones")
    parser.add_argument("--dry-run",   action="store_true", help="Una iteración sin ejecutar órdenes reales")
    args = parser.parse_args()

    if args.status:
        cmd_status()
    elif args.stop:
        cmd_stop()
    elif getattr(args, "close_all", False):
        cmd_close_all()
    else:
        run_bot(dry_run=args.dry_run)
