"""
trade_executor.py — Ejecución y registro de trades, compartido por live_bot y dashboard.

Centraliza en un único sitio: calcular parámetros (si no vienen dados), abrir en
Bybit Demo (o simular en local como fallback) y persistir en `live_trades` +
`bot_state`. Así el bot automático (live_bot.py) y el botón manual del dashboard
(`/api/bot/execute`) usan exactamente la misma lógica.

Distinción real vs simulado: el `bybit_order_id` lleva prefijo `sim_` cuando no se
ejecutó en Bybit (keys ausentes o dry-run); un id sin prefijo es una orden real.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

from sqlalchemy import text

log = logging.getLogger(__name__)


def ensure_bot_state(engine, initial_capital: float = 50000.0):
    """Garantiza que exista la fila bot_state id=1 (antes faltaba → el estado
    no persistía porque _update_state hacía UPDATE sin INSERT previo)."""
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO bot_state (id, initial_capital, current_capital)
            VALUES (1, :cap, :cap)
            ON CONFLICT (id) DO NOTHING
        """), {"cap": initial_capital})


def _session_name(dt: datetime) -> str:
    h = dt.hour
    if 0 <= h < 8:   return "asia"
    if 7 <= h < 12:  return "london"
    if 13 <= h < 22: return "ny"
    return "dead_zone"


def execute_trade(engine, bybit, tools_mod, *, direction: str, config: dict,
                  trigger_type: str, trigger_detail: str,
                  confluence: dict | None = None, agent_decision_text: str = "",
                  trade_params: dict | None = None, dry_run: bool = False) -> dict:
    """
    Abre y registra un trade. Devuelve dict con el resultado.
    No lanza: ante error devuelve {'ok': False, 'error': ...}.
    """
    try:
        ensure_bot_state(engine, float(config.get("initial_capital", 50000.0)))

        capital  = float(config.get("initial_capital", 50000.0))
        risk_pct = float(config.get("risk_per_trade_pct", 1.0))
        leverage = float(config.get("leverage", 3.0))

        # 1. Parámetros del trade (si no vienen dados, calcular frescos)
        if trade_params is None:
            trade_params = tools_mod.dispatch_tool("get_trade_parameters", {
                "direction": direction, "capital_usd": capital,
                "risk_pct": risk_pct, "leverage": leverage, "holding_hours": 8.0,
            })
        if not trade_params or trade_params.get("error"):
            return {"ok": False, "error": f"trade_params: {(trade_params or {}).get('error', 'sin datos')}"}

        size_usd = float(trade_params["position_size_usd"])
        size_btc = float(trade_params["position_size_btc"])
        sl       = float(trade_params["stop_loss"])
        tp       = float(trade_params["take_profit"])
        rr       = float(trade_params.get("net_rr_ratio") or 0)
        entry    = float(trade_params.get("entry") or trade_params.get("current_price") or 0)

        # 2. Ejecutar en Bybit (real) o simular (fallback)
        order_id = "sim_" + str(int(time.time()))
        executed_real = False
        if not dry_run and bybit is not None and bybit.is_configured():
            res = bybit.open_position(direction=direction, size_usd=size_usd,
                                      leverage=leverage, stop_loss=sl, take_profit=tp)
            if res.get("error") or res.get("status") == "rejected":
                return {"ok": False, "error": f"Bybit: {res.get('error', 'rechazada')}"}
            order_id      = res.get("order_id") or order_id
            entry         = float(res.get("entry_price") or entry)
            # Tamaño REAL ejecutado en Bybit (ya dividido por leverage). Antes se
            # guardaba el size_btc sin ajustar → PnL de posición abierta inflado ~Nx.
            if res.get("size_btc"):
                size_btc = float(res["size_btc"])
            executed_real = True

        # 3. Persistir en live_trades
        now  = datetime.now(timezone.utc)
        sl_pct = abs(sl - entry) / entry * 100 if entry else 0
        tp_pct = abs(tp - entry) / entry * 100 if entry else 0
        cs_score = float((confluence or {}).get("final_score") or 0)
        cs_label = (confluence or {}).get("label", "")

        with engine.begin() as conn:
            tid = conn.execute(text("""
                INSERT INTO live_trades
                    (bybit_order_id, entry_time, entry_price, direction,
                     size_usd, size_btc, leverage,
                     stop_loss, take_profit, sl_pct, tp_pct, risk_reward,
                     trigger_type, trigger_detail,
                     confluence_score, confluence_label, agent_decision,
                     signal_snapshot, session, status)
                VALUES
                    (:order_id, :entry_time, :entry_price, :direction,
                     :size_usd, :size_btc, :leverage,
                     :sl, :tp, :sl_pct, :tp_pct, :rr,
                     :trigger_type, :trigger_detail,
                     :cs_score, :cs_label, :agent_decision,
                     CAST(:snapshot AS jsonb), :session, 'open')
                RETURNING id
            """), {
                "order_id": order_id, "entry_time": now, "entry_price": entry,
                "direction": direction, "size_usd": round(size_usd, 2),
                "size_btc": round(size_btc, 6), "leverage": leverage,
                "sl": sl, "tp": tp, "sl_pct": round(sl_pct, 3), "tp_pct": round(tp_pct, 3),
                "rr": round(rr, 2), "trigger_type": trigger_type, "trigger_detail": trigger_detail,
                "cs_score": round(cs_score, 4), "cs_label": cs_label,
                "agent_decision": (agent_decision_text or "")[:2000],
                "snapshot": json.dumps(confluence or {}, default=str)[:4000],
                "session": _session_name(now),
            }).scalar()

            # 4. Actualizar bot_state
            n_open = conn.execute(text("SELECT COUNT(*) FROM live_trades WHERE status='open'")).scalar()
            conn.execute(text("""
                UPDATE bot_state
                SET open_trades = :n_open,
                    total_trades = COALESCE(total_trades,0) + 1
                WHERE id = 1
            """), {"n_open": n_open})

        mode = "real" if executed_real else "sim"
        log.info("execute_trade: #%s %s %.4f BTC @ %.2f | SL %.2f TP %.2f RR %.2f | %s | order=%s",
                 tid, direction, size_btc, entry, sl, tp, rr, mode, order_id)
        return {
            "ok": True, "trade_id": tid, "order_id": order_id, "mode": mode,
            "direction": direction, "entry_price": entry, "stop_loss": sl,
            "take_profit": tp, "size_usd": round(size_usd, 2),
            "size_btc": round(size_btc, 6), "rr": round(rr, 2),
        }
    except Exception as e:
        log.error("execute_trade error: %s", e)
        return {"ok": False, "error": str(e)}
