"""
03_onchain_scraper.py — Métricas on-chain via CryptoQuant API v1

Uso:
  python agente_ia/news/03_onchain_scraper.py
"""

import os, sys, logging, time
from datetime import datetime, timezone, timedelta, date
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

# ── Config ────────────────────────────────────────────────────────────────────

CQ_BASE  = "https://api.cryptoquant.com/v1"
CQ_KEY   = os.getenv("CRYPTOQUANT_API_KEY", "").strip()

# (metric_name, endpoint_path, params)
ENDPOINTS = [
    ("exchange_inflow",  "/btc/exchange-flows/inflow",   {"exchange": "all", "window": "day", "limit": 7}),
    ("exchange_outflow", "/btc/exchange-flows/outflow",  {"exchange": "all", "window": "day", "limit": 7}),
    ("exchange_netflow", "/btc/exchange-flows/netflow",  {"exchange": "all", "window": "day", "limit": 7}),
    ("mvrv_ratio",       "/btc/market-indicator/mvrv-ratio", {"limit": 7}),
    ("sopr",             "/btc/market-indicator/sopr",   {"limit": 7}),
    ("hash_rate",        "/btc/network-indicator/hash-rate", {"limit": 7}),
    ("exchange_reserve", "/btc/exchange-flows/reserve",  {"exchange": "all", "limit": 30}),
]

SIGNAL_RULES = {
    "exchange_netflow": {
        "bullish_threshold": -3000,
        "bearish_threshold":  5000,
    },
    "mvrv_ratio": {
        "bullish_threshold": 1.0,
        "bearish_threshold": 3.5,
    },
    "sopr": {
        "bullish_threshold": 0.98,
        "bearish_threshold": 1.05,
    },
}

# Pesos para calculate_onchain_score
METRIC_WEIGHTS = {
    "exchange_netflow": 0.35,
    "mvrv_ratio":       0.25,
    "sopr":             0.20,
    "hash_rate":        0.10,
    "exchange_reserve": 0.10,
    "n_transactions":   0.08,
}


def _cq_get(path: str, params: dict) -> Optional[list]:
    """Llama a CryptoQuant API. Devuelve la lista de data o None si falla."""
    if not CQ_KEY:
        log.debug("CRYPTOQUANT_API_KEY no configurada")
        return None
    headers = {"Authorization": f"Bearer {CQ_KEY}"}
    try:
        resp = requests.get(f"{CQ_BASE}{path}", params=params, headers=headers, timeout=15)
        if resp.status_code == 429:
            log.warning("CryptoQuant: rate limit (429) — omitiendo endpoint")
            return None
        resp.raise_for_status()
        return resp.json().get("data", [])
    except Exception as e:
        log.warning(f"CryptoQuant {path}: {e}")
        return None


def _already_fetched_today(metric_name: str, engine) -> bool:
    """True si ya hay un registro de hoy para esta métrica."""
    today = date.today().isoformat()
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT 1 FROM on_chain_metrics
            WHERE metric_name = :metric
              AND DATE(timestamp AT TIME ZONE 'UTC') = :today
            LIMIT 1
        """), {"metric": metric_name, "today": today}).fetchone()
    return row is not None


def _moving_avg(values: list[float], n: int) -> Optional[float]:
    if len(values) < n:
        return None
    return sum(values[-n:]) / n


def _apply_signal_rule(metric_name: str, value: float, all_values: list[float]) -> tuple[str, float]:
    """Devuelve (signal, strength) para una métrica dada."""
    if metric_name in SIGNAL_RULES:
        rule = SIGNAL_RULES[metric_name]
        bull_th = rule["bullish_threshold"]
        bear_th = rule["bearish_threshold"]
        if value <= bull_th:
            strength = min(abs(value - bull_th) / max(abs(bull_th), 1), 1.0)
            return "bullish", round(strength, 3)
        if value >= bear_th:
            strength = min(abs(value - bear_th) / max(abs(bear_th), 1), 1.0)
            return "bearish", round(strength, 3)
        return "neutral", 0.0

    if metric_name == "hash_rate":
        avg7 = _moving_avg(all_values, 7)
        if avg7 and avg7 > 0:
            if value > avg7 * 1.05:
                return "bullish", min((value / avg7 - 1) * 10, 1.0)
            if value < avg7 * 0.90:
                return "bearish", min((1 - value / avg7) * 10, 1.0)
        return "neutral", 0.0

    if metric_name == "exchange_reserve":
        avg30 = _moving_avg(all_values, 30)
        if avg30 and avg30 > 0:
            if value < avg30 * 0.97:
                return "bullish", min((1 - value / avg30) * 20, 1.0)
            if value > avg30 * 1.03:
                return "bearish", min((value / avg30 - 1) * 20, 1.0)
        return "neutral", 0.0

    if metric_name == "n_transactions":
        avg7 = _moving_avg(all_values, 7)
        if avg7 and avg7 > 0:
            if value > avg7 * 1.10:
                return "bullish", min((value / avg7 - 1) * 5, 1.0)
            if value < avg7 * 0.85:
                return "bearish", min((1 - value / avg7) * 5, 1.0)
        return "neutral", 0.0

    return "neutral", 0.0


def _insert_metric(metric_name: str, data: list[dict], engine) -> int:
    if not data:
        return 0

    values = [d.get("value") for d in data if d.get("value") is not None]
    latest = data[-1]
    value  = latest.get("value")
    if value is None:
        return 0

    # Calcular medias móviles
    avg7  = _moving_avg(values, 7)
    avg30 = _moving_avg(values, 30)

    # Timestamp del dato más reciente
    ts_str = latest.get("date") or latest.get("timestamp") or latest.get("start_time")
    try:
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")) if ts_str else datetime.now(timezone.utc)
    except Exception:
        ts = datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    signal, strength = _apply_signal_rule(metric_name, value, values)

    with engine.begin() as conn:
        try:
            result = conn.execute(text("""
                INSERT INTO on_chain_metrics
                    (timestamp, asset, metric_name, value, value_7d_avg, value_30d_avg,
                     signal, signal_strength, source)
                VALUES
                    (:ts, 'BTC', :metric, :value, :avg7, :avg30,
                     :signal, :strength, 'cryptoquant')
                ON CONFLICT (timestamp, asset, metric_name, source) DO NOTHING
            """), {
                "ts": ts, "metric": metric_name, "value": value,
                "avg7": avg7, "avg30": avg30,
                "signal": signal, "strength": strength,
            })
            return result.rowcount
        except Exception as e:
            log.debug(f"on_chain insert {metric_name}: {e}")
            return 0


def _run_free_sources(engine) -> dict:
    """Fetch on-chain metrics from blockchain.info (free, no API key needed)."""
    results = {}

    # Hash rate — 30 días de histórico para moving average
    try:
        resp = requests.get(
            "https://api.blockchain.info/charts/hash-rate",
            params={"timespan": "30days", "format": "json", "sampled": "false"},
            timeout=20,
        )
        resp.raise_for_status()
        pts = resp.json().get("values", [])
        if pts:
            data = [
                {"date": datetime.fromtimestamp(v["x"], tz=timezone.utc).isoformat(), "value": v["y"]}
                for v in pts
            ]
            inserted = _insert_metric("hash_rate", data, engine)
            results["hash_rate"] = f"inserted_{inserted}" if inserted else "cached"
            log.info(f"blockchain.info hash_rate: {len(data)} pts, inserted={inserted}")
    except Exception as e:
        log.warning(f"blockchain.info hash-rate: {e}")
        results["hash_rate"] = "api_error"

    # Transaction count — 14 días
    try:
        resp = requests.get(
            "https://api.blockchain.info/charts/n-transactions",
            params={"timespan": "14days", "format": "json", "sampled": "false"},
            timeout=20,
        )
        resp.raise_for_status()
        pts = resp.json().get("values", [])
        if pts:
            data = [
                {"date": datetime.fromtimestamp(v["x"], tz=timezone.utc).isoformat(), "value": v["y"]}
                for v in pts
            ]
            inserted = _insert_metric("n_transactions", data, engine)
            results["n_transactions"] = f"inserted_{inserted}" if inserted else "cached"
            log.info(f"blockchain.info n_transactions: {len(data)} pts, inserted={inserted}")
    except Exception as e:
        log.warning(f"blockchain.info n-transactions: {e}")
        results["n_transactions"] = "api_error"

    return results


def run_all(engine=None) -> dict:
    if engine is None:
        engine = get_engine()

    results = {}

    # Intentar CryptoQuant (puede dar 403 en plan gratuito)
    if CQ_KEY:
        for metric_name, path, params in ENDPOINTS:
            if _already_fetched_today(metric_name, engine):
                log.info(f"On-chain {metric_name}: ya cacheado hoy — skip")
                results[metric_name] = "cached"
                continue
            data = _cq_get(path, params)
            if data is None:
                results[metric_name] = "api_error"
                continue
            inserted = _insert_metric(metric_name, data, engine)
            results[metric_name] = "inserted" if inserted else "duplicate"
            log.info(f"On-chain {metric_name}: {len(data)} puntos, inserted={inserted}")
            time.sleep(0.5)
    else:
        log.info("CRYPTOQUANT_API_KEY no configurada — saltando CryptoQuant")

    # Siempre complementar con blockchain.info (gratis)
    free_results = _run_free_sources(engine)
    results.update(free_results)

    return results


def calculate_onchain_score(engine=None) -> Optional[float]:
    """Lee métricas recientes de on_chain_metrics y devuelve score [-1, +1]."""
    if engine is None:
        engine = get_engine()

    cutoff = datetime.now(timezone.utc) - timedelta(hours=48)

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT DISTINCT ON (metric_name)
                metric_name, value, value_7d_avg, value_30d_avg, signal, signal_strength
            FROM on_chain_metrics
            WHERE timestamp >= :cutoff AND asset = 'BTC'
            ORDER BY metric_name, timestamp DESC
        """), {"cutoff": cutoff}).fetchall()

    if not rows:
        return None

    # Reconstruir señales desde la DB (signal ya calculado)
    signal_map = {"bullish": +1, "bearish": -1, "neutral": 0}
    total_weight = 0.0
    weighted_sum = 0.0

    for row in rows:
        metric = row.metric_name
        w = METRIC_WEIGHTS.get(metric, 0.05)
        sig_val = signal_map.get(row.signal or "neutral", 0)
        strength = float(row.signal_strength or 0.0)
        weighted_sum += sig_val * strength * w
        total_weight += w

    if total_weight == 0:
        return None

    score = weighted_sum / total_weight
    return max(-1.0, min(1.0, score))


if __name__ == "__main__":
    engine = get_engine()
    result = run_all(engine)
    print("Run result:", result)
    score = calculate_onchain_score(engine)
    print(f"On-chain score: {score}")
