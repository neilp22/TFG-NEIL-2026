"""
analysis/backfill_binance.py — Descarga histórico OHLCV de Binance e inserta en btc_ohlcv.

Sin auth (endpoint público). Solo rellena open/high/low/close/volume/timestamp;
los indicadores se recalculan en el backtest desde los datos crudos.

Uso:
    python analysis/backfill_binance.py --tf 4h --days 365
    python analysis/backfill_binance.py --tf 1h --days 180
"""
from __future__ import annotations
import argparse, logging, sys, time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import requests
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).parent.parent))
from db.db_utils import get_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("backfill")

BINANCE = "https://api.binance.com/api/v3/klines"
TF_MAP = {"5m": "5m", "15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}
TF_MS  = {"5m": 5*60*1000, "15m": 15*60*1000, "1h": 60*60*1000, "4h": 4*60*60*1000, "1d": 24*60*60*1000}


def fetch_klines(symbol, tf, start_ms, limit=1000):
    r = requests.get(BINANCE, params={
        "symbol": symbol, "interval": TF_MAP[tf],
        "startTime": start_ms, "limit": limit
    }, timeout=15)
    r.raise_for_status()
    return r.json()


def backfill(symbol="BTCUSDT", tf="4h", days=365):
    engine = get_engine()
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - days * 24 * 60 * 60 * 1000
    step_ms = TF_MS[tf] * 1000   # 1000 candles per request

    log.info(f"Backfill {symbol} {tf} desde {datetime.fromtimestamp(start_ms/1000, tz=timezone.utc).date()} ({days} días)")

    inserted, updated, skipped = 0, 0, 0
    cur = start_ms
    while cur < end_ms:
        try:
            rows = fetch_klines(symbol, tf, cur, limit=1000)
        except Exception as e:
            log.error(f"Binance request fallo: {e}")
            time.sleep(5)
            continue
        if not rows:
            break

        with engine.begin() as conn:
            for k in rows:
                ts = datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc)
                o, h, l, c, v = float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])
                existing = conn.execute(text(
                    "SELECT id FROM btc_ohlcv WHERE timestamp=:ts AND timeframe=:tf"
                ), {"ts": ts, "tf": tf}).fetchone()
                if existing:
                    skipped += 1
                    continue
                conn.execute(text("""
                    INSERT INTO btc_ohlcv (timestamp, timeframe, open, high, low, close, volume)
                    VALUES (:ts, :tf, :o, :h, :l, :c, :v)
                """), {"ts": ts, "tf": tf, "o": o, "h": h, "l": l, "c": c, "v": v})
                inserted += 1

        last_ts = rows[-1][0]
        cur = last_ts + TF_MS[tf]
        log.info(f"  progreso: hasta {datetime.fromtimestamp(last_ts/1000, tz=timezone.utc).date()} | "
                 f"insertadas: {inserted} | skip: {skipped}")
        time.sleep(0.2)   # respect rate limit

    log.info(f"DONE — inserted={inserted} skipped={skipped}")
    return inserted


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--tf", default="4h", choices=list(TF_MAP.keys()))
    ap.add_argument("--days", type=int, default=365)
    args = ap.parse_args()
    backfill(args.symbol, args.tf, args.days)
