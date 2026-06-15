"""
02_price_updater.py — Actualización incremental de daily_features

Flujo:
  1. MAX(date) en daily_features WHERE asset='BTCUSDT'
  2. Si ya está al día, salir
  3. Descargar OHLCV desde Binance (+ 60 días extra para indicadores)
  4. Calcular indicadores técnicos (ta-lib con fallback manual)
  5. Obtener sentimiento de sentiment_scores para los días nuevos
  6. UPSERT en daily_features ON CONFLICT (date, asset) DO UPDATE
  7. Añadir columnas sentiment_morning y has_sentiment si no existen

Uso:
  python agente_ia/02_price_updater.py           # actualización normal
  python agente_ia/02_price_updater.py --force   # fuerza re-cálculo últimos 7 días
"""

import os, sys, logging
from datetime import datetime, timezone, timedelta, date
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / 'config' / '.env')

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import requests
from sqlalchemy import text
from db.db_utils import get_engine

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

ASSET = 'BTCUSDT'
ASSET_LABEL = 'BTC'
BINANCE_BASE = "https://api.binance.com/api/v3/klines"
INDICATOR_WARMUP_DAYS = 60  # días extra para estabilizar indicadores


# ── Binance OHLCV ────────────────────────────────────────────────────────────

def _fetch_binance_ohlcv(start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """Descarga velas diarias BTC/USDT desde Binance REST API."""
    rows = []
    current = start_dt

    while current < end_dt:
        params = {
            'symbol':    ASSET,
            'interval':  '1d',
            'startTime': int(current.timestamp() * 1000),
            'endTime':   int(min(end_dt, current + timedelta(days=1000)).timestamp() * 1000),
            'limit':     1000,
        }
        try:
            resp = requests.get(BINANCE_BASE, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log.error(f"Binance fetch error: {e}")
            break

        if not data:
            break

        for candle in data:
            open_time = datetime.fromtimestamp(candle[0] / 1000, tz=timezone.utc)
            rows.append({
                'date':   open_time.date(),
                'open':   float(candle[1]),
                'high':   float(candle[2]),
                'low':    float(candle[3]),
                'close':  float(candle[4]),
                'volume': float(candle[5]),
            })

        last_ts = datetime.fromtimestamp(data[-1][0] / 1000, tz=timezone.utc)
        current = last_ts + timedelta(days=1)

        if len(data) < 1000:
            break

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    df = df.drop_duplicates('date').sort_values('date').reset_index(drop=True)
    return df


# ── Indicadores técnicos ─────────────────────────────────────────────────────

def _calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula todos los indicadores. Intenta librería 'ta', fallback manual."""
    try:
        import ta
        df['rsi_14']       = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        macd_ind           = ta.trend.MACD(df['close'])
        df['macd']         = macd_ind.macd()
        df['macd_signal']  = macd_ind.macd_signal()
        bb                 = ta.volatility.BollingerBands(df['close'], window=20)
        df['bb_upper']     = bb.bollinger_hband()
        df['bb_lower']     = bb.bollinger_lband()
        df['sma_7']        = ta.trend.SMAIndicator(df['close'], window=7).sma_indicator()
        df['sma_30']       = ta.trend.SMAIndicator(df['close'], window=30).sma_indicator()
        log.info("Indicadores calculados con librería 'ta'")
    except ImportError:
        log.warning("Librería 'ta' no disponible — usando cálculo manual")
        df['rsi_14']      = _calc_rsi(df['close'], 14)
        ema12             = df['close'].ewm(span=12, adjust=False).mean()
        ema26             = df['close'].ewm(span=26, adjust=False).mean()
        df['macd']        = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        sma20             = df['close'].rolling(20).mean()
        std20             = df['close'].rolling(20).std()
        df['bb_upper']    = sma20 + 2 * std20
        df['bb_lower']    = sma20 - 2 * std20
        df['sma_7']       = df['close'].rolling(7).mean()
        df['sma_30']      = df['close'].rolling(30).mean()

    df['returns'] = df['close'].pct_change()
    df['label']   = (df['returns'].shift(-1) > 0).astype(int)  # label es dirección de mañana
    return df


# ── Sentimiento ───────────────────────────────────────────────────────────────

def _get_daily_sentiment(dates: list, engine) -> pd.DataFrame:
    """Agrega sentimiento por día desde raw_texts (sentiment_raw) o sentiment_scores (compound_score)."""
    if not dates:
        return pd.DataFrame(columns=['date', 'sentiment_avg', 'sentiment_std', 'sentiment_count'])

    date_list = "','".join(str(d)[:10] for d in dates)

    # Fuente primaria: raw_texts.sentiment_raw (pipeline nuevo)
    query = f"""
        SELECT
            DATE(timestamp AT TIME ZONE 'UTC')  AS date,
            AVG(sentiment_raw)                   AS sentiment_avg,
            STDDEV(sentiment_raw)                AS sentiment_std,
            COUNT(*)                             AS sentiment_count
        FROM raw_texts
        WHERE asset = 'BTC'
          AND processed = TRUE
          AND sentiment_raw IS NOT NULL
          AND DATE(timestamp AT TIME ZONE 'UTC') IN ('{date_list}')
        GROUP BY DATE(timestamp AT TIME ZONE 'UTC')
    """
    try:
        df = pd.read_sql(query, engine)
        df['date'] = pd.to_datetime(df['date'])
        if not df.empty:
            return df
    except Exception as e:
        log.debug(f"raw_texts sentiment query: {e}")

    # Fallback: sentiment_scores (pipeline antiguo FinBERT) — JOIN con raw_texts para timestamp
    query_fb = f"""
        SELECT
            DATE(rt.timestamp AT TIME ZONE 'UTC') AS date,
            AVG(ss.compound_score)                 AS sentiment_avg,
            STDDEV(ss.compound_score)              AS sentiment_std,
            COUNT(*)                               AS sentiment_count
        FROM sentiment_scores ss
        JOIN raw_texts rt ON ss.text_id = rt.id
        WHERE rt.asset = 'BTC'
          AND DATE(rt.timestamp AT TIME ZONE 'UTC') IN ('{date_list}')
        GROUP BY DATE(rt.timestamp AT TIME ZONE 'UTC')
    """
    try:
        df = pd.read_sql(query_fb, engine)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        log.warning(f"Sentiment query error: {e}")
        return pd.DataFrame(columns=['date', 'sentiment_avg', 'sentiment_std', 'sentiment_count'])


def _get_morning_sentiment(dates: list, engine) -> pd.DataFrame:
    """Sentimiento de noticias publicadas antes de las 18:00 UTC."""
    if not dates:
        return pd.DataFrame(columns=['date', 'sentiment_morning'])

    date_list = "','".join(str(d)[:10] for d in dates)
    query = f"""
        SELECT
            DATE(timestamp AT TIME ZONE 'UTC') AS date,
            AVG(sentiment_raw)                  AS sentiment_morning
        FROM raw_texts
        WHERE asset = 'BTC'
          AND processed = TRUE
          AND sentiment_raw IS NOT NULL
          AND EXTRACT(HOUR FROM timestamp AT TIME ZONE 'UTC') < 18
          AND DATE(timestamp AT TIME ZONE 'UTC') IN ('{date_list}')
        GROUP BY DATE(timestamp AT TIME ZONE 'UTC')
    """
    try:
        df = pd.read_sql(query, engine)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        log.warning(f"Morning sentiment query error: {e}")
        return pd.DataFrame(columns=['date', 'sentiment_morning'])


# ── Fear & Greed ─────────────────────────────────────────────────────────────

def _get_fear_greed(start_date: date, end_date: date) -> dict:
    """Descarga Fear & Greed Index de alternative.me para un rango de fechas."""
    days = (end_date - start_date).days + 10
    try:
        resp = requests.get(
            f"https://api.alternative.me/fng/?limit={days}&format=json",
            timeout=15
        )
        resp.raise_for_status()
        data = resp.json().get('data', [])
        result = {}
        for item in data:
            dt = datetime.fromtimestamp(int(item['timestamp']), tz=timezone.utc).date()
            if start_date <= dt <= end_date:
                result[dt] = float(item['value'])
        return result
    except Exception as e:
        log.warning(f"Fear & Greed API error: {e}")
        return {}


# ── Schema helpers ────────────────────────────────────────────────────────────

def _ensure_columns(engine):
    """Añade columnas sentiment_morning y has_sentiment si no existen."""
    alters = [
        "ALTER TABLE daily_features ADD COLUMN IF NOT EXISTS sentiment_morning FLOAT",
        "ALTER TABLE daily_features ADD COLUMN IF NOT EXISTS has_sentiment BOOLEAN DEFAULT FALSE",
    ]
    with engine.begin() as conn:
        for sql in alters:
            try:
                conn.execute(text(sql))
            except Exception as e:
                log.debug(f"Column alter (may already exist): {e}")


# ── UPSERT ────────────────────────────────────────────────────────────────────

def _upsert_daily_features(rows: list[dict], engine) -> int:
    """UPSERT en daily_features. Devuelve filas afectadas."""
    if not rows:
        return 0

    upsert_sql = text("""
        INSERT INTO daily_features (
            date, asset, close, returns, label,
            rsi_14, macd, macd_signal, bb_upper, bb_lower, sma_7, sma_30,
            sentiment_avg, sentiment_std, sentiment_count,
            sentiment_morning, has_sentiment, fear_greed
        ) VALUES (
            :date, :asset, :close, :returns, :label,
            :rsi_14, :macd, :macd_signal, :bb_upper, :bb_lower, :sma_7, :sma_30,
            :sentiment_avg, :sentiment_std, :sentiment_count,
            :sentiment_morning, :has_sentiment, :fear_greed
        )
        ON CONFLICT (date, asset) DO UPDATE SET
            close            = EXCLUDED.close,
            returns          = EXCLUDED.returns,
            label            = EXCLUDED.label,
            rsi_14           = EXCLUDED.rsi_14,
            macd             = EXCLUDED.macd,
            macd_signal      = EXCLUDED.macd_signal,
            bb_upper         = EXCLUDED.bb_upper,
            bb_lower         = EXCLUDED.bb_lower,
            sma_7            = EXCLUDED.sma_7,
            sma_30           = EXCLUDED.sma_30,
            sentiment_avg    = COALESCE(EXCLUDED.sentiment_avg, daily_features.sentiment_avg),
            sentiment_std    = COALESCE(EXCLUDED.sentiment_std, daily_features.sentiment_std),
            sentiment_count  = COALESCE(EXCLUDED.sentiment_count, daily_features.sentiment_count),
            sentiment_morning= COALESCE(EXCLUDED.sentiment_morning, daily_features.sentiment_morning),
            has_sentiment    = EXCLUDED.has_sentiment,
            fear_greed       = COALESCE(EXCLUDED.fear_greed, daily_features.fear_greed)
    """)

    affected = 0
    for row in rows:
        # Transacción individual por fila — un fallo no aborta las demás
        try:
            with engine.begin() as conn:
                result = conn.execute(upsert_sql, row)
                affected += result.rowcount
        except Exception as e:
            log.warning(f"UPSERT error for {row.get('date')}: {e}")

    return affected


# ── Función principal ─────────────────────────────────────────────────────────

def run_update(force_days: int = 0) -> dict:
    """
    Ejecuta actualización incremental de daily_features.

    Args:
        force_days: si > 0, recalcula los últimos N días aunque ya existan.
    """
    engine = get_engine()
    _ensure_columns(engine)

    # 1. MAX(date) actual
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT MAX(date) FROM daily_features WHERE asset = :asset"
        ), {'asset': ASSET_LABEL})
        max_date = result.scalar()

    today = datetime.now(timezone.utc).date()

    if max_date is None:
        log.info("No hay datos en daily_features — descargando histórico completo (365 días)")
        start_date = today - timedelta(days=365)
    elif force_days > 0:
        start_date = max_date - timedelta(days=force_days)
        log.info(f"Modo forzado: recalculando desde {start_date}")
    else:
        start_date = max_date + timedelta(days=1)

    if start_date >= today and force_days == 0:
        log.info(f"daily_features ya está al día (MAX date={max_date})")
        return {'status': 'up_to_date', 'max_date': str(max_date)}

    # 2. Descargar OHLCV con warmup para indicadores
    fetch_start = datetime.combine(
        start_date - timedelta(days=INDICATOR_WARMUP_DAYS),
        datetime.min.time(),
        tzinfo=timezone.utc,
    )
    fetch_end = datetime.combine(today, datetime.min.time(), tzinfo=timezone.utc)

    log.info(f"Descargando OHLCV Binance {fetch_start.date()} → {fetch_end.date()}")
    df = _fetch_binance_ohlcv(fetch_start, fetch_end)

    if df.empty:
        log.error("Sin datos de Binance")
        return {'status': 'error', 'error': 'No Binance data'}

    # 3. Calcular indicadores
    df = _calc_indicators(df)

    # 4. Filtrar solo los días nuevos (descartar warmup)
    df_new = df[df['date'].dt.date >= start_date].copy()
    if df_new.empty:
        log.info("Sin filas nuevas tras filtro de fecha")
        return {'status': 'up_to_date'}

    new_dates = df_new['date'].dt.date.tolist()
    log.info(f"Fechas nuevas a procesar: {len(new_dates)} días ({new_dates[0]} → {new_dates[-1]})")

    # 5. Sentimiento
    sent_df   = _get_daily_sentiment(new_dates, engine)
    morn_df   = _get_morning_sentiment(new_dates, engine)

    sent_map  = dict(zip(sent_df['date'], sent_df['sentiment_avg'])) if not sent_df.empty else {}
    std_map   = dict(zip(sent_df['date'], sent_df['sentiment_std']))  if not sent_df.empty else {}
    cnt_map   = dict(zip(sent_df['date'], sent_df['sentiment_count'])) if not sent_df.empty else {}
    morn_map  = dict(zip(morn_df['date'], morn_df['sentiment_morning'])) if not morn_df.empty else {}

    # 6. Fear & Greed
    fg_map = _get_fear_greed(new_dates[0], new_dates[-1])

    # 7. Construir filas para UPSERT
    rows = []
    for _, row in df_new.iterrows():
        d = row['date'].date()
        sentiment_avg = sent_map.get(row['date'])
        rows.append({
            'date':             d,
            'asset':            ASSET_LABEL,
            'close':            float(row['close'])  if not pd.isna(row['close']) else None,
            'returns':          float(row['returns'])       if not pd.isna(row['returns']) else None,
            'label':            int(row['label'])           if not pd.isna(row['label']) else None,
            'rsi_14':           float(row['rsi_14'])        if not pd.isna(row['rsi_14']) else None,
            'macd':             float(row['macd'])          if not pd.isna(row['macd']) else None,
            'macd_signal':      float(row['macd_signal'])   if not pd.isna(row['macd_signal']) else None,
            'bb_upper':         float(row['bb_upper'])      if not pd.isna(row['bb_upper']) else None,
            'bb_lower':         float(row['bb_lower'])      if not pd.isna(row['bb_lower']) else None,
            'sma_7':            float(row['sma_7'])         if not pd.isna(row['sma_7']) else None,
            'sma_30':           float(row['sma_30'])        if not pd.isna(row['sma_30']) else None,
            'sentiment_avg':    float(sentiment_avg)        if sentiment_avg is not None else None,
            'sentiment_std':    float(std_map.get(row['date'], np.nan)) if std_map.get(row['date']) is not None else None,
            'sentiment_count':  int(cnt_map.get(row['date'], 0)),
            'sentiment_morning': float(morn_map.get(row['date'], np.nan)) if morn_map.get(row['date']) is not None else None,
            'has_sentiment':    int(sentiment_avg is not None),
            'fear_greed':       float(fg_map.get(d, np.nan)) if fg_map.get(d) is not None else None,
        })

    inserted = _upsert_daily_features(rows, engine)
    log.info(f"UPSERT completado: {inserted} filas afectadas ({len(rows)} procesadas)")

    return {
        'status':    'updated',
        'rows_new':  len(rows),
        'inserted':  inserted,
        'date_from': str(new_dates[0]),
        'date_to':   str(new_dates[-1]),
    }


# ── Intraday OHLCV (btc_ohlcv) ───────────────────────────────────────────────

_TF_CONFIG = {
    '5m':  {'interval': '5m',  'limit': 1000},  # ~3.5 días
    '15m': {'interval': '15m', 'limit': 1000},  # ~10 días
    '1h':  {'interval': '1h',  'limit': 1000},  # ~42 días
    '4h':  {'interval': '4h',  'limit': 1000},  # ~166 días
    '1d':  {'interval': '1d',  'limit': 500},   # ~500 días
}


def _fetch_intraday(interval: str, limit: int) -> pd.DataFrame:
    """Descarga velas desde Binance REST API."""
    try:
        resp = requests.get(
            BINANCE_BASE,
            params={'symbol': ASSET, 'interval': interval, 'limit': limit},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.error(f"Binance {interval} error: {e}")
        return pd.DataFrame()

    rows = []
    for c in data:
        ts = datetime.fromtimestamp(c[0] / 1000, tz=timezone.utc)
        rows.append({
            'timestamp': ts,
            'open':   float(c[1]),
            'high':   float(c[2]),
            'low':    float(c[3]),
            'close':  float(c[4]),
            'volume': float(c[5]),
        })
    return pd.DataFrame(rows)


def update_ohlcv(engine=None, timeframes=None) -> dict:
    """
    Descarga velas desde Binance y hace upsert en btc_ohlcv.
    IMPORTANTE: upsert (no DO NOTHING) — la vela en curso ya existe en DB
    pero su close/high/low cambian hasta que cierra; sin el UPDATE el último
    precio de la DB se queda congelado y el agente ve un precio desfasado.

    timeframes: iterable opcional ('1h', '4h', ...) para limitar la actualización.
    """
    if engine is None:
        engine = get_engine()

    tf_items = [(tf, cfg) for tf, cfg in _TF_CONFIG.items()
                if timeframes is None or tf in timeframes]

    stats = {}
    for tf, cfg in tf_items:
        df = _fetch_intraday(cfg['interval'], cfg['limit'])
        if df.empty:
            log.warning(f"Sin datos Binance {tf}")
            stats[tf] = 0
            continue

        df['timeframe'] = tf
        rows = df.to_dict('records')

        upserted = 0
        sql = text("""
            INSERT INTO btc_ohlcv (timestamp, timeframe, open, high, low, close, volume)
            VALUES (:timestamp, :timeframe, :open, :high, :low, :close, :volume)
            ON CONFLICT (timestamp, timeframe) DO UPDATE SET
                open   = EXCLUDED.open,
                high   = EXCLUDED.high,
                low    = EXCLUDED.low,
                close  = EXCLUDED.close,
                volume = EXCLUDED.volume
        """)
        with engine.begin() as conn:
            for row in rows:
                try:
                    r = conn.execute(sql, row)
                    upserted += r.rowcount
                except Exception as e:
                    log.debug(f"btc_ohlcv upsert {tf} {row['timestamp']}: {e}")

        log.info(f"btc_ohlcv {tf}: {len(rows)} fetched, {upserted} upserted")
        stats[tf] = upserted

    # Calcular indicadores e ICT en las filas nuevas/recientes
    tfs_recalc = tuple(tf for tf, _ in tf_items)
    try:
        from agente_ia.indicators import recalculate_indicators
        recalculate_indicators(engine, timeframes=tfs_recalc)
    except Exception as e:
        log.warning(f"indicators recalculate: {e}")
        # Intentar import relativo
        try:
            import importlib.util, sys as _sys
            _spec = importlib.util.spec_from_file_location(
                "indicators",
                Path(__file__).parent / "indicators.py",
            )
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            _mod.recalculate_indicators(engine, timeframes=tfs_recalc)
        except Exception as e2:
            log.warning(f"indicators fallback also failed: {e2}")

    return {'status': 'updated', 'inserted': stats}


if __name__ == '__main__':
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', type=int, default=0,
                    help='Recalcular últimos N días aunque ya existan')
    ap.add_argument('--ohlcv', action='store_true', help='Actualizar btc_ohlcv (intraday)')
    args = ap.parse_args()

    if args.ohlcv:
        result = update_ohlcv()
    else:
        result = run_update(force_days=args.force)
    print(json.dumps(result, indent=2, default=str))
