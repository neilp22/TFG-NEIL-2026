"""
06_bias_calculator.py — Orquestador de scores y generador de contexto LLM

Uso:
  python agente_ia/news/06_bias_calculator.py
"""

import sys, importlib.util, math, logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).parent.parent.parent
_HERE = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / 'config' / '.env')

from sqlalchemy import text
from db.db_utils import get_engine

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


# ── Cargar módulos con nombres numéricos ──────────────────────────────────────

def _load_module(filename: str):
    path = _HERE / filename
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Constantes ────────────────────────────────────────────────────────────────

WEIGHTS = {
    "news":          0.20,
    "calendar":      0.25,
    "onchain":       0.25,
    "micro":         0.20,
    "institutional": 0.10,
}

LAST_HALVING = datetime(2024, 4, 20, tzinfo=timezone.utc)

TIER_WEIGHTS = {1: 1.0, 2: 0.65, 3: 0.3}

BIAS_LABELS_ES = {
    "strong_bull": "FUERTEMENTE ALCISTA",
    "bull":        "MODERADAMENTE ALCISTA",
    "neutral":     "NEUTRAL",
    "bear":        "MODERADAMENTE BAJISTA",
    "strong_bear": "FUERTEMENTE BAJISTA",
}

HALVING_PHASES_ES = {
    "early_bull":   "BULL TEMPRANO post-halving",
    "late_bull":    "BULL TARDÍO post-halving",
    "distribution": "FASE DE DISTRIBUCIÓN",
    "accumulation": "FASE DE ACUMULACIÓN",
}

IMPACT_MULT = {"high": 1.0, "medium": 0.5, "low": 0.15}


# ── Score de noticias ─────────────────────────────────────────────────────────

def calculate_news_score(engine=None, hours_back: int = 6) -> Optional[float]:
    """Lee scores precomputados por 01_news_scraper.py desde raw_texts."""
    if engine is None:
        engine = get_engine()

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_back)

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT sentiment_decay, impact_weight
            FROM raw_texts
            WHERE timestamp >= :cutoff
              AND asset = 'BTC'
              AND sentiment_decay IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT 300
        """), {"cutoff": cutoff}).fetchall()

    if not rows:
        return None

    # sentiment_decay = raw_score * impact_weight  →  weighted avg = sum(decay) / sum(weight)
    total_w  = sum(float(r.impact_weight or 0) for r in rows if r.impact_weight)
    total_d  = sum(float(r.sentiment_decay or 0) for r in rows if r.sentiment_decay is not None)

    if total_w == 0:
        valid = [float(r.sentiment_decay) for r in rows if r.sentiment_decay is not None]
        return round(sum(valid) / len(valid), 4) if valid else None

    score = total_d / total_w
    return max(-1.0, min(1.0, score))


# ── Score de calendario ───────────────────────────────────────────────────────

def calculate_calendar_score(engine=None) -> Optional[float]:
    if engine is None:
        engine = get_engine()

    now    = datetime.now(timezone.utc)
    past   = now - timedelta(hours=6)
    future = now + timedelta(hours=48)

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT event_name, impact, actual, btc_bias,
                   event_datetime,
                   EXTRACT(EPOCH FROM (event_datetime - NOW())) / 3600.0 AS hours_away
            FROM economic_calendar
            WHERE event_datetime BETWEEN :past AND :future
              AND currency IN ('USD', 'EUR', 'GBP')
              AND btc_bias IS NOT NULL
            ORDER BY event_datetime ASC
        """), {"past": past, "future": future}).fetchall()

    if not rows:
        return None

    total_weight = 0.0
    weighted_sum = 0.0

    for row in rows:
        if row.btc_bias is None:
            continue
        impact_m = IMPACT_MULT.get(row.impact, 0.15)
        hours    = float(row.hours_away or 0)

        if hours <= 0:
            # Evento pasado con actual
            time_w = 1.0 if row.actual is not None else 0.25
        else:
            # Evento futuro: anticipación
            time_w = 0.25 / (1 + hours / 24)

        contrib = float(row.btc_bias) * impact_m * time_w
        weighted_sum += contrib
        total_weight += impact_m * time_w

    if total_weight == 0:
        return None

    score = weighted_sum / total_weight
    return max(-1.0, min(1.0, score))


# ── Info de halving ───────────────────────────────────────────────────────────

def get_halving_info() -> dict:
    now        = datetime.now(timezone.utc)
    days_since = (now - LAST_HALVING).days

    if days_since < 365:
        phase = "early_bull"
        desc  = "Típicamente el período de mayor apreciación"
    elif days_since < 548:
        phase = "late_bull"
        desc  = "Bull tardío — aumenta riesgo de distribución"
    elif days_since < 730:
        phase = "distribution"
        desc  = "Fase de distribución — alta volatilidad"
    else:
        phase = "accumulation"
        desc  = "Acumulación pre-halving o bear market"

    return {
        "days_since":  days_since,
        "phase":       phase,
        "phase_es":    HALVING_PHASES_ES.get(phase, phase),
        "description": desc,
        "last_halving": LAST_HALVING.strftime("%Y-%m-%d"),
    }


# ── Label de bias ─────────────────────────────────────────────────────────────

def _bias_label(score: float) -> str:
    if score >= 0.5:  return "strong_bull"
    if score >= 0.15: return "bull"
    if score > -0.15: return "neutral"
    if score > -0.5:  return "bear"
    return "strong_bear"


# ── Snapshot completo ─────────────────────────────────────────────────────────

def build_bias_snapshot(engine=None, timeframe: str = "1h") -> dict:
    if engine is None:
        engine = get_engine()

    # Cargar módulos auxiliares
    try:
        m_onchain = _load_module("03_onchain_scraper.py")
        onchain_score = m_onchain.calculate_onchain_score(engine)
    except Exception as e:
        log.warning(f"Onchain score error: {e}")
        onchain_score = None

    try:
        m_micro = _load_module("04_micro_scraper.py")
        micro_data    = m_micro.run_all(engine)
        micro_score   = m_micro.calculate_micro_score(_raw_data=micro_data)
        fg_data       = micro_data.get("fear_greed")
        funding_data  = micro_data.get("funding")
    except Exception as e:
        log.warning(f"Micro score error: {e}")
        micro_score = None
        fg_data     = None
        funding_data = None
        micro_data   = {}

    try:
        m_etf = _load_module("05_etf_scraper.py")
        institutional_score = m_etf.calculate_institutional_score(engine)
    except Exception as e:
        log.warning(f"Institutional score error: {e}")
        institutional_score = None

    # ETF flow 5d para el resumen
    etf_flow_5d = None
    try:
        from datetime import date as ddate
        cutoff5d = ddate.today() - timedelta(days=10)
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT SUM(flow_usd) / COUNT(DISTINCT flow_date) AS avg_daily
                FROM institutional_flows
                WHERE flow_date >= :cutoff AND vehicle_type = 'spot_etf' AND asset = 'BTC'
            """), {"cutoff": cutoff5d}).fetchone()
            if row and row.avg_daily is not None:
                etf_flow_5d = round(float(row.avg_daily), 1)
    except Exception:
        pass

    news_score     = calculate_news_score(engine)
    calendar_score = calculate_calendar_score(engine)

    scores = {
        "news":          news_score,
        "calendar":      calendar_score,
        "onchain":       onchain_score,
        "micro":         micro_score,
        "institutional": institutional_score,
    }

    # Ponderación renormalizando módulos sin datos
    available = {k: v for k, v in scores.items() if v is not None}
    if available:
        total_w = sum(WEIGHTS[k] for k in available)
        bias_score = sum(v * WEIGHTS[k] / total_w for k, v in available.items())
        bias_score = max(-1.0, min(1.0, bias_score))
    else:
        bias_score = 0.0

    label = _bias_label(bias_score)

    # Top drivers
    module_scores = [(k, v) for k, v in available.items()]
    module_scores.sort(key=lambda x: abs(x[1]), reverse=True)
    top_drivers = [
        {
            "module":      k,
            "signal":      "bullish" if v > 0 else ("bearish" if v < 0 else "neutral"),
            "value":       round(v, 3),
            "description": _driver_desc(k, v, micro_data),
        }
        for k, v in module_scores[:4]
    ]

    # Calendario próximas 48h
    calendar_upcoming = []
    try:
        now     = datetime.now(timezone.utc)
        cutoff  = now + timedelta(hours=48)
        with engine.connect() as conn:
            cal_rows = conn.execute(text("""
                SELECT event_name, event_datetime, impact, btc_bias,
                       EXTRACT(EPOCH FROM (event_datetime - NOW())) / 3600.0 AS hours_away
                FROM economic_calendar
                WHERE event_datetime BETWEEN :now AND :cutoff
                  AND impact = 'high'
                ORDER BY event_datetime ASC
                LIMIT 5
            """), {"now": now, "cutoff": cutoff}).fetchall()
        calendar_upcoming = [
            {
                "event":      r.event_name,
                "datetime":   r.event_datetime.strftime("%Y-%m-%d %H:%M UTC"),
                "impact":     r.impact,
                "hours_away": round(float(r.hours_away), 1),
                "btc_bias":   r.btc_bias,
            }
            for r in cal_rows
        ]
    except Exception:
        pass

    # Contar artículos en últimas 6h
    news_count = 0
    try:
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT COUNT(*) FROM raw_texts
                WHERE timestamp >= NOW() - INTERVAL '6 hours' AND asset = 'BTC'
            """)).scalar()
            news_count = row or 0
    except Exception:
        pass

    # Precio 24h (Binance ticker)
    price_context = None
    try:
        import requests as _req
        resp = _req.get(
            "https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT",
            headers={"User-Agent": "Mozilla/5.0"}, timeout=8,
        )
        resp.raise_for_status()
        t = resp.json()
        price_context = {
            "price":           round(float(t["lastPrice"]), 2),
            "change_24h_pct":  round(float(t["priceChangePercent"]), 2),
            "change_24h_usd":  round(float(t["priceChange"]), 2),
            "high_24h":        round(float(t["highPrice"]), 2),
            "low_24h":         round(float(t["lowPrice"]), 2),
            "volume_24h_usd_b": round(float(t["quoteVolume"]) / 1e9, 2),
        }
    except Exception as e:
        log.warning(f"Price 24h ticker: {e}")

    # Top noticias por impacto (últimas 6h)
    top_news = []
    try:
        now_dt = datetime.now(timezone.utc)
        with engine.connect() as conn:
            news_rows = conn.execute(text("""
                SELECT text, source, source_tier, timestamp, sentiment_raw
                FROM raw_texts
                WHERE asset = 'BTC'
                  AND timestamp >= NOW() - INTERVAL '6 hours'
                  AND processed = TRUE
                  AND sentiment_raw IS NOT NULL
                ORDER BY ABS(sentiment_raw) DESC
                LIMIT 5
            """)).fetchall()
        for r in news_rows:
            ts = r.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            import re as _re
            clean_headline = _re.sub(r'<[^>]+>', '', r.text).strip()[:110]
            top_news.append({
                "headline":      clean_headline,
                "source":        r.source,
                "tier":          int(r.source_tier or 3),
                "hours_ago":     round((now_dt - ts).total_seconds() / 3600, 1),
                "sentiment_raw": round(float(r.sentiment_raw), 2),
            })
    except Exception as e:
        log.warning(f"Top news: {e}")

    # Indicadores técnicos del último cierre diario
    technicals = None
    try:
        with engine.connect() as conn:
            trow = conn.execute(text("""
                SELECT date, close, returns, rsi_14, macd, macd_signal,
                       bb_upper, bb_lower, sma_7, sma_30
                FROM daily_features
                WHERE asset = 'BTC'
                ORDER BY date DESC
                LIMIT 1
            """)).fetchone()
        if trow:
            technicals = {
                "date":        str(trow.date)[:10],
                "close":       round(float(trow.close), 2)       if trow.close       else None,
                "rsi_14":      round(float(trow.rsi_14), 1)      if trow.rsi_14      else None,
                "macd":        round(float(trow.macd), 4)        if trow.macd        else None,
                "macd_signal": round(float(trow.macd_signal), 4) if trow.macd_signal else None,
                "bb_upper":    round(float(trow.bb_upper))       if trow.bb_upper    else None,
                "bb_lower":    round(float(trow.bb_lower))       if trow.bb_lower    else None,
                "sma_7":       round(float(trow.sma_7))          if trow.sma_7       else None,
                "sma_30":      round(float(trow.sma_30))         if trow.sma_30      else None,
            }
    except Exception as e:
        log.warning(f"Technicals: {e}")

    snapshot = {
        "timestamp":          datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "asset":              "BTC",
        "timeframe":          timeframe,
        "scores":             scores,
        "bias_score":         round(bias_score, 3),
        "bias_label":         label,
        "top_drivers":        top_drivers,
        "halving":            get_halving_info(),
        "fear_greed":         {"value": fg_data["value"], "label": fg_data["label"]} if fg_data else None,
        "calendar_upcoming":  calendar_upcoming,
        "price_context":      price_context,
        "top_news":           top_news,
        "technicals":         technicals,
        "market_detail": {
            "long_short_ratio":  micro_data.get("long_short", {}).get("ratio")       if micro_data else None,
            "long_short_signal": micro_data.get("long_short", {}).get("signal")      if micro_data else None,
            "oi_change_24h":     micro_data.get("oi", {}).get("oi_change_24h")       if micro_data else None,
            "funding_8h_avg":    funding_data.get("funding_8h_avg")                  if funding_data else None,
        },
        "meta": {
            "news_count_6h":   news_count,
            "etf_flow_5d_avg": etf_flow_5d,
            "funding_annual":  funding_data["annual_pct"] if funding_data else None,
        },
    }

    # Persistir en bias_snapshots
    _save_snapshot(snapshot, scores, engine)

    return snapshot


def _driver_desc(module: str, value: float, micro_data: dict) -> str:
    direction = "alcista" if value > 0 else "bajista"
    desc_map = {
        "news":          f"Noticias recientes — sentimiento {direction}",
        "calendar":      f"Macro económica — sesgo {direction}",
        "onchain":       f"Métricas on-chain — señal {direction}",
        "micro":         f"Microestructura — funding/OI {direction}",
        "institutional": f"Flujos ETF institucionales — {direction}",
    }
    return desc_map.get(module, f"{module} {direction}")


def _save_snapshot(snapshot: dict, scores: dict, engine) -> None:
    import json as _json
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO bias_snapshots
                    (asset, timeframe,
                     score_news, score_calendar, score_onchain,
                     score_institutional, score_micro,
                     bias_score, bias_label, top_drivers,
                     halving_phase, computed_by)
                VALUES
                    ('BTC', :tf,
                     :news, :calendar, :onchain,
                     :inst, :micro,
                     :bias, :label, CAST(:drivers AS jsonb),
                     :phase, 'pipeline_v2')
            """), {
                "tf":       snapshot["timeframe"],
                "news":     scores.get("news"),
                "calendar": scores.get("calendar"),
                "onchain":  scores.get("onchain"),
                "inst":     scores.get("institutional"),
                "micro":    scores.get("micro"),
                "bias":     snapshot["bias_score"],
                "label":    snapshot["bias_label"],
                "drivers":  _json.dumps(snapshot.get("top_drivers", []), default=str),
                "phase":    snapshot.get("halving", {}).get("phase"),
            })
    except Exception as e:
        log.warning(f"Snapshot save error: {e}")


# ── Formato LLM ───────────────────────────────────────────────────────────────

def format_llm_context(snapshot: dict) -> str:
    scores   = snapshot["scores"]
    meta     = snapshot.get("meta", {})
    halving  = snapshot["halving"]
    fg       = snapshot.get("fear_greed")
    upcoming = snapshot.get("calendar_upcoming", [])
    drivers  = snapshot.get("top_drivers", [])
    price    = snapshot.get("price_context")
    tech     = snapshot.get("technicals")
    mkt      = snapshot.get("market_detail", {})
    top_news = snapshot.get("top_news", [])

    def score_str(key: str, label: str, pct: int, extra: str = "") -> str:
        v = scores.get(key)
        if v is None:
            return f"  {label:<15} ({pct}%): [SIN DATOS]"
        extra_part = f"  [{extra}]" if extra else ""
        return f"  {label:<15} ({pct}%): {v:+.2f}{extra_part}"

    news_extra = f"{meta.get('news_count_6h', 0)} artículos últimas 6h"
    cal_extra  = f"próximo: {upcoming[0]['event'][:28]} en {upcoming[0]['hours_away']:.0f}h" if upcoming else ""
    micro_extra_parts = []
    if meta.get("funding_annual") is not None:
        micro_extra_parts.append(f"Funding: {meta['funding_annual']:.2f}% anual")
    if fg:
        micro_extra_parts.append(f"F&G: {fg['value']}/100")
    micro_extra = " | ".join(micro_extra_parts)
    etf_extra   = f"ETF flow 5d: ${meta['etf_flow_5d_avg']:.0f}M/día" if meta.get("etf_flow_5d_avg") is not None else ""

    # Precio
    if price:
        chg_arrow = "▲" if price["change_24h_pct"] >= 0 else "▼"
        precio_line = (
            f"  Último: ${price['price']:,.0f}  |  "
            f"24h: {chg_arrow}{price['change_24h_pct']:+.2f}% (${price['change_24h_usd']:+,.0f})  |  "
            f"Vol: ${price['volume_24h_usd_b']:.1f}B\n"
            f"  High 24h: ${price['high_24h']:,.0f}  |  Low 24h: ${price['low_24h']:,.0f}"
        )
    else:
        precio_line = "  (sin datos de precio)"

    # Técnico
    if tech and tech.get("rsi_14") is not None:
        rsi = tech["rsi_14"]
        rsi_label = "sobrecomprado" if rsi > 70 else ("sobrevendido" if rsi < 30 else "neutral")
        macd_dir = ""
        if tech.get("macd") is not None and tech.get("macd_signal") is not None:
            macd_dir = "alcista" if tech["macd"] > tech["macd_signal"] else "bajista"
        sma_pos = ""
        if tech.get("sma_7") and tech.get("sma_30") and tech.get("close"):
            c = tech["close"]
            if c > tech["sma_7"] > tech["sma_30"]:
                sma_pos = "precio > SMA7 > SMA30 (tendencia alcista)"
            elif c < tech["sma_7"] < tech["sma_30"]:
                sma_pos = "precio < SMA7 < SMA30 (tendencia bajista)"
            else:
                sma_pos = f"SMA7=${tech['sma_7']:,.0f} | SMA30=${tech['sma_30']:,.0f}"
        tech_line = (
            f"  RSI(14): {rsi:.1f} [{rsi_label}]  |  "
            f"MACD: {macd_dir}  |  {sma_pos}\n"
            f"  BB Upper: ${tech['bb_upper']:,.0f}  |  BB Lower: ${tech['bb_lower']:,.0f}  "
            f"(último cierre: {tech['date']})"
        ) if tech.get("bb_upper") else f"  RSI(14): {rsi:.1f} [{rsi_label}]  |  MACD: {macd_dir}  |  {sma_pos}"
    else:
        tech_line = "  (sin datos técnicos — tabla daily_features no actualizada)"

    # Microestructura detallada
    ls_ratio  = mkt.get("long_short_ratio")
    ls_signal = mkt.get("long_short_signal", "")
    oi_chg    = mkt.get("oi_change_24h")
    mkt_lines = []
    if ls_ratio is not None:
        mkt_lines.append(f"  Long/Short ratio: {ls_ratio:.2f}  [{ls_signal} — {'exceso de longs' if ls_ratio > 1.5 else ('exceso de shorts' if ls_ratio < 0.8 else 'equilibrado')}]")
    if oi_chg is not None:
        mkt_lines.append(f"  OI cambio 24h: {oi_chg:+.2f}%  |  Funding anual: {meta.get('funding_annual', 0):.2f}%")
    if fg:
        mkt_lines.append(f"  Fear & Greed: {fg['value']}/100 — {fg['label']}")
    mkt_block = "\n".join(mkt_lines) if mkt_lines else "  (sin datos)"

    # Noticias destacadas
    if top_news:
        news_lines = []
        for n in top_news:
            sentiment_label = "▲" if n["sentiment_raw"] > 0.1 else ("▼" if n["sentiment_raw"] < -0.1 else "→")
            news_lines.append(
                f"  {sentiment_label} [{n['source']} | hace {n['hours_ago']:.1f}h | score: {n['sentiment_raw']:+.2f}]\n"
                f"    {n['headline'][:100]}"
            )
        news_block = "\n".join(news_lines)
    else:
        news_block = "  (sin noticias recientes con scoring)"

    # Drivers
    drivers_lines = "\n".join(
        f"  {i+1}. [{d['module'].upper():13}] {d['signal'].upper():7} ({d['value']:+.2f}) — {d['description']}"
        for i, d in enumerate(drivers)
    ) if drivers else "  (sin datos suficientes)"

    # Calendario
    cal_lines = "\n".join(
        f"  • {ev['datetime']} | {ev['impact'].upper():6} | "
        f"{ev['event'][:40]:40} | bias={'▲' if ev['btc_bias'] == 1 else ('▼' if ev['btc_bias'] == -1 else '—')}"
        for ev in upcoming
    ) if upcoming else "  (sin eventos high-impact en 48h)"

    label_es = BIAS_LABELS_ES.get(snapshot["bias_label"], snapshot["bias_label"])
    fg_label = fg["label"] if fg else "N/A"
    fg_value = fg["value"] if fg else "N/A"

    return f"""\
════════════════════════════════════════════════════════
BIAS SNAPSHOT — BTC — {snapshot['timestamp']} UTC
════════════════════════════════════════════════════════

SCORE FINAL: {snapshot['bias_score']:+.2f}  ({label_es})

PRECIO:
{precio_line}

SCORES POR MÓDULO:
{score_str('news',          'Noticias',       20, news_extra)}
{score_str('calendar',      'Macro/Calendar', 25, cal_extra)}
{score_str('onchain',       'On-chain',       25)}
{score_str('micro',         'Microstructure', 20, micro_extra)}
{score_str('institutional', 'Institucional',  10, etf_extra)}

TOP DRIVERS:
{drivers_lines}

MICROESTRUCTURA:
{mkt_block}

TÉCNICO (último cierre diario):
{tech_line}

NOTICIAS DESTACADAS (últimas 6h):
{news_block}

CALENDARIO PRÓXIMAS 48H:
{cal_lines}

RÉGIMEN: {halving['phase_es']} — día {halving['days_since']} desde último halving
FEAR & GREED: {fg_value}/100 — {fg_label}
════════════════════════════════════════════════════════"""


if __name__ == "__main__":
    engine   = get_engine()
    snapshot = build_bias_snapshot(engine)
    context  = format_llm_context(snapshot)
    print(context)
