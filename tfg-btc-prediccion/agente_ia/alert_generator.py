"""
alert_generator.py — Genera y comprueba alertas de nivel desde ICT (OBs, FVGs, swings).
Evita llamar al agente LLM constantemente: el bot solo llama cuando el precio toca un nivel.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from sqlalchemy import text
from db.db_utils import get_engine

log = logging.getLogger(__name__)

MAX_DISTANCE_PCT  = 3.0    # solo alertas a menos del 3% del precio
ALERT_EXPIRY_H    = 24     # alertas caducan en 24h


def _get_current_price(engine) -> float:
    """Precio de cierre más reciente de btc_ohlcv 1h."""
    sql = text("SELECT close FROM btc_ohlcv WHERE timeframe='1h' ORDER BY timestamp DESC LIMIT 1")
    with engine.connect() as conn:
        row = conn.execute(sql).fetchone()
    return float(row[0]) if row else 0.0


def _fetch_ict_levels(engine, price: float) -> list[dict]:
    """
    Extrae niveles ICT activos (OBs, FVGs, swings) de 15m, 1h y 4h.
    Filtra a MAX_DISTANCE_PCT del precio.
    """
    levels = []
    now    = datetime.now(timezone.utc)
    exp    = now + timedelta(hours=ALERT_EXPIRY_H)

    for tf in ("15m", "1h", "4h"):
        sql = text("""
            SELECT timestamp, open, high, low, close,
                   ob_bullish, ob_bearish,
                   fvg_bullish, fvg_bearish,
                   swing_high, swing_low
            FROM btc_ohlcv
            WHERE timeframe = :tf
            ORDER BY timestamp DESC
            LIMIT 300
        """)
        try:
            with engine.connect() as conn:
                df = pd.read_sql(sql, conn, params={"tf": tf})
        except Exception as e:
            log.warning("alert_generator fetch %s: %s", tf, e)
            continue

        if df.empty:
            continue

        for _, row in df.iterrows():
            h, l = float(row["high"]), float(row["low"])
            mid  = (h + l) / 2

            # OB alcista — precio de entrada = mid del OB
            if row.get("ob_bullish"):
                dist = abs(mid - price) / price * 100
                if dist <= MAX_DISTANCE_PCT:
                    levels.append({
                        "level_price":   round(mid, 2),
                        "level_type":    "ob_bullish",
                        "timeframe":     tf,
                        "direction":     "long",
                        "tolerance_pct": 0.20,
                        "expires_at":    exp,
                        "notes":         f"OB alcista {tf} {l:.0f}–{h:.0f} ({dist:.2f}% distancia)",
                    })

            # OB bajista
            if row.get("ob_bearish"):
                dist = abs(mid - price) / price * 100
                if dist <= MAX_DISTANCE_PCT:
                    levels.append({
                        "level_price":   round(mid, 2),
                        "level_type":    "ob_bearish",
                        "timeframe":     tf,
                        "direction":     "short",
                        "tolerance_pct": 0.20,
                        "expires_at":    exp,
                        "notes":         f"OB bajista {tf} {l:.0f}–{h:.0f} ({dist:.2f}% distancia)",
                    })

            # FVG alcista (gap por debajo del precio)
            if row.get("fvg_bullish"):
                dist = abs(mid - price) / price * 100
                if dist <= MAX_DISTANCE_PCT:
                    levels.append({
                        "level_price":   round(mid, 2),
                        "level_type":    "fvg_bullish",
                        "timeframe":     tf,
                        "direction":     "long",
                        "tolerance_pct": 0.25,
                        "expires_at":    exp,
                        "notes":         f"FVG alcista {tf} {l:.0f}–{h:.0f} ({dist:.2f}% distancia)",
                    })

            # FVG bajista
            if row.get("fvg_bearish"):
                dist = abs(mid - price) / price * 100
                if dist <= MAX_DISTANCE_PCT:
                    levels.append({
                        "level_price":   round(mid, 2),
                        "level_type":    "fvg_bearish",
                        "timeframe":     tf,
                        "direction":     "short",
                        "tolerance_pct": 0.25,
                        "expires_at":    exp,
                        "notes":         f"FVG bajista {tf} {l:.0f}–{h:.0f} ({dist:.2f}% distancia)",
                    })

            # Swing high — posible resistencia → short
            if row.get("swing_high"):
                dist = abs(h - price) / price * 100
                if dist <= MAX_DISTANCE_PCT:
                    levels.append({
                        "level_price":   round(h, 2),
                        "level_type":    "swing_high",
                        "timeframe":     tf,
                        "direction":     "short",
                        "tolerance_pct": 0.10,
                        "expires_at":    exp,
                        "notes":         f"Swing high {tf} {h:.0f} ({dist:.2f}% distancia)",
                    })

            # Swing low — posible soporte → long
            if row.get("swing_low"):
                dist = abs(l - price) / price * 100
                if dist <= MAX_DISTANCE_PCT:
                    levels.append({
                        "level_price":   round(l, 2),
                        "level_type":    "swing_low",
                        "timeframe":     tf,
                        "direction":     "long",
                        "tolerance_pct": 0.10,
                        "expires_at":    exp,
                        "notes":         f"Swing low {tf} {l:.0f} ({dist:.2f}% distancia)",
                    })

    return levels


def generate_alerts(engine=None) -> int:
    """
    Regenera alertas activas desde los datos ICT de btc_ohlcv.
    Limpia alertas caducadas y disparadas antes de insertar nuevas.
    Devuelve el número de alertas activas resultantes.
    """
    if engine is None:
        engine = get_engine()

    now   = datetime.now(timezone.utc)
    price = _get_current_price(engine)
    if not price:
        log.warning("generate_alerts: sin precio disponible")
        return 0

    # Limpiar alertas caducadas o ya disparadas
    with engine.begin() as conn:
        conn.execute(text("""
            DELETE FROM price_alerts
            WHERE triggered = TRUE
               OR expires_at < :now
        """), {"now": now})

    levels = _fetch_ict_levels(engine, price)
    if not levels:
        log.info("generate_alerts: sin niveles ICT dentro del rango %.1f%%", MAX_DISTANCE_PCT)
        return 0

    # Deduplicar: no insertar si ya existe una alerta del mismo tipo a menos de 0.1%
    with engine.connect() as conn:
        existing = conn.execute(text("""
            SELECT level_price, level_type FROM price_alerts
            WHERE triggered = FALSE AND expires_at > :now
        """), {"now": now}).fetchall()
    existing_set = {(float(r[0]), r[1]) for r in existing}

    inserted = 0
    with engine.begin() as conn:
        for lv in levels:
            # Evitar duplicado cercano
            is_dup = any(
                abs(lv["level_price"] - ep) / lv["level_price"] < 0.001 and lv["level_type"] == et
                for ep, et in existing_set
            )
            if is_dup:
                continue
            conn.execute(text("""
                INSERT INTO price_alerts
                    (level_price, level_type, timeframe, direction,
                     tolerance_pct, expires_at, notes)
                VALUES
                    (:level_price, :level_type, :timeframe, :direction,
                     :tolerance_pct, :expires_at, :notes)
            """), lv)
            inserted += 1

    # Contar total activas
    with engine.connect() as conn:
        total = conn.execute(text(
            "SELECT COUNT(*) FROM price_alerts WHERE triggered=FALSE AND expires_at > :now"
        ), {"now": now}).scalar()

    log.info("generate_alerts: %d nuevas, %d activas totales (precio=%.2f)", inserted, total, price)
    return int(total)


def check_alerts(current_price: float, engine=None) -> list[dict]:
    """
    Comprueba si el precio ha tocado alguna alerta activa.
    Marca las disparadas y devuelve la lista para que el bot llame al agente.
    """
    if engine is None:
        engine = get_engine()

    now = datetime.now(timezone.utc)
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT id, level_price, level_type, timeframe, direction,
                   tolerance_pct, notes
            FROM price_alerts
            WHERE triggered = FALSE AND expires_at > :now
        """), {"now": now}).fetchall()

    triggered = []
    for row in rows:
        aid, lp, lt, tf, direction, tol, notes = row
        dist = abs(current_price - lp) / lp * 100
        if dist <= tol:
            triggered.append({
                "id":          aid,
                "level_price": lp,
                "level_type":  lt,
                "timeframe":   tf,
                "direction":   direction,
                "distance_pct": round(dist, 4),
                "notes":       notes,
            })

    if triggered:
        ids = [t["id"] for t in triggered]
        with engine.begin() as conn:
            conn.execute(text("""
                UPDATE price_alerts
                SET triggered=TRUE, triggered_at=:now
                WHERE id = ANY(:ids)
            """), {"now": now, "ids": ids})
        log.info("check_alerts: %d alertas disparadas a precio=%.2f", len(triggered), current_price)

    return triggered


def get_active_alerts(engine=None) -> list[dict]:
    """Devuelve todas las alertas activas (para el dashboard)."""
    if engine is None:
        engine = get_engine()
    now = datetime.now(timezone.utc)
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT id, level_price, level_type, timeframe, direction,
                   tolerance_pct, expires_at, notes,
                   ABS(level_price - (
                       SELECT close FROM btc_ohlcv WHERE timeframe='1h'
                       ORDER BY timestamp DESC LIMIT 1
                   )) / level_price * 100 AS distance_pct
            FROM price_alerts
            WHERE triggered=FALSE AND expires_at > :now
            ORDER BY distance_pct ASC
        """), {"now": now}).fetchall()
    return [
        {
            "id":          r[0],
            "level_price": r[1],
            "level_type":  r[2],
            "timeframe":   r[3],
            "direction":   r[4],
            "tolerance_pct": r[5],
            "expires_at":  str(r[6]),
            "notes":       r[7],
            "distance_pct": round(float(r[8]), 3) if r[8] else None,
        }
        for r in rows
    ]


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    e = get_engine()
    n = generate_alerts(e)
    print(f"Alertas activas: {n}")
    alerts = get_active_alerts(e)
    print(json.dumps(alerts[:10], indent=2, default=str))
