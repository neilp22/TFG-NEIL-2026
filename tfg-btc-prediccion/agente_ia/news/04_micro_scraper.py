"""
04_micro_scraper.py — Microestructura de mercado: Binance Futures (público) + Fear & Greed

Uso:
  python agente_ia/news/04_micro_scraper.py
"""

import sys, logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / 'config' / '.env')

import requests
from sqlalchemy import text
from db.db_utils import get_engine

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# ── Endpoints ─────────────────────────────────────────────────────────────────

BINANCE_BASE = "https://fapi.binance.com"
FEARGREED_URL = "https://api.alternative.me/fng/?limit=7"

ENDPOINTS = {
    "funding_rate":  f"{BINANCE_BASE}/fapi/v1/fundingRate?symbol=BTCUSDT&limit=8",
    "open_interest": f"{BINANCE_BASE}/fapi/v1/openInterest?symbol=BTCUSDT",
    "oi_history":    f"{BINANCE_BASE}/futures/data/openInterestHist?symbol=BTCUSDT&period=1h&limit=24",
    "long_short":    f"{BINANCE_BASE}/futures/data/globalLongShortAccountRatio?symbol=BTCUSDT&period=1h&limit=4",
    "top_traders":   f"{BINANCE_BASE}/futures/data/topLongShortAccountRatio?symbol=BTCUSDT&period=1h&limit=4",
}

HEADERS = {"User-Agent": "Mozilla/5.0"}


def _get(url: str) -> Optional[dict | list]:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        log.warning(f"GET {url}: {e}")
        return None


# ── Funciones de señal (puras, sin DB) ───────────────────────────────────────

def funding_signal(rate: float) -> tuple[str, float]:
    annual_pct = rate * 3 * 365 * 100
    if annual_pct > 30:
        return "bearish", min(annual_pct / 50, 1.0)
    if annual_pct > 15:
        return "bearish", 0.3
    if annual_pct < -5:
        return "bullish", min(abs(annual_pct) / 20, 1.0)
    return "neutral", 0.0


def oi_signal(oi_now: float, oi_4h_ago: float, price_now: float, price_4h_ago: float) -> tuple[str, float]:
    if oi_4h_ago <= 0 or price_4h_ago <= 0:
        return "neutral", 0.0
    oi_change    = (oi_now - oi_4h_ago) / oi_4h_ago * 100
    price_change = (price_now - price_4h_ago) / price_4h_ago * 100
    if oi_change > 3 and price_change > 1:
        return "bullish", 0.6
    if oi_change > 3 and price_change < -1:
        return "bearish", 0.7
    if oi_change < -3 and price_change > 1:
        return "bullish", 0.5   # short squeeze
    return "neutral", 0.0


def long_short_signal(ratio: float) -> tuple[str, float]:
    if ratio > 1.8:
        return "bearish", 0.6
    if ratio > 1.4:
        return "bearish", 0.3
    if ratio < 0.7:
        return "bullish", 0.6
    if ratio < 0.85:
        return "bullish", 0.3
    return "neutral", 0.0


def fear_greed_signal(value: int) -> tuple[str, float]:
    if value <= 20:
        return "bullish", 0.8
    if value <= 35:
        return "bullish", 0.4
    if value >= 80:
        return "bearish", 0.8
    if value >= 65:
        return "bearish", 0.3
    return "neutral", 0.0


# ── Fetch & parse ─────────────────────────────────────────────────────────────

def fetch_funding_rate() -> Optional[dict]:
    data = _get(ENDPOINTS["funding_rate"])
    if not data or not isinstance(data, list):
        return None
    latest = data[-1]
    rate = float(latest.get("fundingRate", 0))
    sig, strength = funding_signal(rate)
    return {
        "funding_rate":   rate,
        "funding_8h_avg": sum(float(d["fundingRate"]) for d in data) / len(data),
        "annual_pct":     rate * 3 * 365 * 100,
        "signal":         sig,
        "signal_strength": strength,
    }


def fetch_open_interest() -> Optional[dict]:
    # Actual OI
    oi_now_data = _get(ENDPOINTS["open_interest"])
    if not oi_now_data:
        return None
    oi_now = float(oi_now_data.get("openInterest", 0))

    # Historical OI para calcular cambio 4h
    hist = _get(ENDPOINTS["oi_history"])
    oi_4h_ago = None
    if hist and isinstance(hist, list) and len(hist) >= 5:
        oi_4h_ago = float(hist[-5]["sumOpenInterest"]) * float(hist[-5].get("sumOpenInterestValue", 1))
        oi_now_hist = float(hist[-1]["sumOpenInterest"]) * float(hist[-1].get("sumOpenInterestValue", 1))
        oi_now = oi_now_hist  # más preciso

    return {
        "oi_usd":       oi_now,
        "oi_4h_ago":    oi_4h_ago,
        "oi_change_24h": (
            (oi_now - float(hist[0]["sumOpenInterest"]) * float(hist[0].get("sumOpenInterestValue", 1)))
            / max(float(hist[0]["sumOpenInterest"]) * float(hist[0].get("sumOpenInterestValue", 1)), 1)
            * 100
        ) if hist and len(hist) >= 24 else None,
    }


def fetch_long_short() -> Optional[dict]:
    data = _get(ENDPOINTS["long_short"])
    if not data or not isinstance(data, list):
        return None
    latest = data[-1]
    ratio = float(latest.get("longShortRatio", 1.0))
    sig, strength = long_short_signal(ratio)
    return {"ratio": ratio, "signal": sig, "signal_strength": strength}


def fetch_fear_greed() -> Optional[dict]:
    data = _get(FEARGREED_URL)
    if not data:
        return None
    items = data.get("data", [])
    if not items:
        return None
    latest = items[0]
    value = int(latest.get("value", 50))
    label = latest.get("value_classification", "Neutral")
    sig, strength = fear_greed_signal(value)
    return {"value": value, "label": label, "signal": sig, "signal_strength": strength}


def fetch_btc_price() -> Optional[float]:
    """Precio actual de BTC via Binance spot (público)."""
    try:
        resp = requests.get(
            "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT",
            headers=HEADERS, timeout=8
        )
        return float(resp.json()["price"])
    except Exception:
        return None


# ── DB insert ─────────────────────────────────────────────────────────────────

def _insert_micro(row: dict, engine) -> None:
    with engine.begin() as conn:
        try:
            conn.execute(text("""
                INSERT INTO market_microstructure
                    (timestamp, asset, exchange,
                     oi_usd, oi_change_1h, oi_change_24h,
                     funding_rate, funding_8h_avg,
                     put_call_ratio,
                     signal, source)
                VALUES
                    (:ts, 'BTC', 'binance',
                     :oi_usd, :oi_change_1h, :oi_change_24h,
                     :funding_rate, :funding_8h_avg,
                     :long_short_ratio,
                     :signal, 'binance_fapi')
            """), row)
        except Exception as e:
            log.debug(f"micro_insert: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_all(engine=None) -> dict:
    if engine is None:
        engine = get_engine()

    funding  = fetch_funding_rate()
    oi_data  = fetch_open_interest()
    ls_data  = fetch_long_short()
    fg_data  = fetch_fear_greed()
    price    = fetch_btc_price()

    log.info(f"Funding: {funding}")
    log.info(f"OI: {oi_data}")
    log.info(f"Long/Short: {ls_data}")
    log.info(f"Fear&Greed: {fg_data}")
    log.info(f"BTC price: {price}")

    db_row = {
        "ts":             datetime.now(timezone.utc),
        "oi_usd":         oi_data["oi_usd"]        if oi_data else None,
        "oi_change_1h":   None,
        "oi_change_24h":  oi_data.get("oi_change_24h") if oi_data else None,
        "funding_rate":   funding["funding_rate"]   if funding else None,
        "funding_8h_avg": funding["funding_8h_avg"] if funding else None,
        "long_short_ratio": ls_data["ratio"]        if ls_data else None,
        "signal":         funding["signal"]         if funding else None,
    }
    _insert_micro(db_row, engine)

    return {
        "funding":    funding,
        "oi":         oi_data,
        "long_short": ls_data,
        "fear_greed": fg_data,
        "price":      price,
    }


def calculate_micro_score(engine=None, _raw_data: Optional[dict] = None) -> Optional[float]:
    """
    Calcula micro score [-1, +1].
    Si se pasa _raw_data (resultado de run_all), no necesita DB ni red.
    """
    if _raw_data is None:
        # Obtener datos frescos de la API
        data = run_all(engine)
    else:
        data = _raw_data

    funding  = data.get("funding")
    ls_data  = data.get("long_short")
    fg_data  = data.get("fear_greed")
    oi_data  = data.get("oi")
    price    = data.get("price")

    signal_map = {"bullish": +1, "bearish": -1, "neutral": 0}
    weights = {"funding": 0.35, "oi": 0.25, "long_short": 0.20, "fear_greed": 0.20}

    scores = {}
    if funding:
        sig, strength = funding["signal"], funding["signal_strength"]
        scores["funding"] = signal_map[sig] * strength

    if ls_data:
        sig, strength = ls_data["signal"], ls_data["signal_strength"]
        scores["long_short"] = signal_map[sig] * strength

    if fg_data:
        sig, strength = fg_data["signal"], fg_data["signal_strength"]
        scores["fear_greed"] = signal_map[sig] * strength

    # OI signal requiere precio histórico — aproximación con datos actuales
    if oi_data and oi_data.get("oi_4h_ago") and price:
        # Sin precio 4h anterior, usamos cambio de OI como señal simple
        oi_change = oi_data.get("oi_change_24h") or 0
        if oi_change > 5:
            scores["oi"] = 0.4    # OI sube mucho — puede ser bullish o bearish
        elif oi_change < -5:
            scores["oi"] = -0.3
        else:
            scores["oi"] = 0.0

    total_weight = sum(weights[k] for k in scores)
    if total_weight == 0:
        return None

    weighted = sum(scores[k] * weights[k] for k in scores)
    score = weighted / total_weight
    return max(-1.0, min(1.0, score))


if __name__ == "__main__":
    engine = get_engine()
    data = run_all(engine)
    score = calculate_micro_score(_raw_data=data)
    print(f"\nMicro score: {score:+.3f}")
    if data.get("fear_greed"):
        fg = data["fear_greed"]
        print(f"Fear & Greed: {fg['value']}/100 — {fg['label']}")
