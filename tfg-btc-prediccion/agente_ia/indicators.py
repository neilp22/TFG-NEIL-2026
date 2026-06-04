"""
indicators.py — Indicadores técnicos, ICT concepts y sesiones de mercado
sobre la tabla btc_ohlcv.

Funciones públicas:
  recalculate_indicators(engine, timeframes, lookback_candles)
  get_ict_snapshot(engine, timeframe, n_context) -> dict
  get_session_context(dt) -> dict
  get_technical_levels(engine, price_now, timeframes) -> list
"""

import logging
import math
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import text

log = logging.getLogger(__name__)

# ── Sesiones de mercado (UTC) ─────────────────────────────────────────────────

SESSIONS = {
    "asia":         (0,  0,  8,  0),
    "london":       (7,  0, 12,  0),
    "ny":           (13, 0, 22,  0),
    "dead_zone":    (22, 0, 24,  0),
}
KILLZONES = {
    "london_open": (7,  0,  9, 30),
    "ny_open":     (13, 0, 14, 30),
}


def _time_in_range(h, m, sh, sm, eh, em) -> bool:
    t = h * 60 + m
    s = sh * 60 + sm
    e = eh * 60 + em
    return s <= t < e


def get_session_name(dt: datetime) -> str:
    h, m = dt.hour, dt.minute
    for name, (sh, sm, eh, em) in SESSIONS.items():
        if _time_in_range(h, m, sh, sm, eh, em):
            return name
    return "asia"  # 00:00 wrap


def is_killzone(dt: datetime) -> bool:
    h, m = dt.hour, dt.minute
    for _, (sh, sm, eh, em) in KILLZONES.items():
        if _time_in_range(h, m, sh, sm, eh, em):
            return True
    return False


def minutes_to_next_killzone(dt: datetime) -> int:
    h, m = dt.hour, dt.minute
    now_min = h * 60 + m
    candidates = []
    for kz_sh, kz_sm, kz_eh, kz_em in [(7, 0, 9, 30), (13, 0, 14, 30)]:
        start = kz_sh * 60 + kz_sm
        if now_min < start:
            candidates.append(start - now_min)
        else:
            candidates.append(1440 - now_min + start)
    return min(candidates)


# ── Indicadores técnicos ──────────────────────────────────────────────────────

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift()).abs()
    lc = (df['low']  - df['close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _vwap_daily(df: pd.DataFrame) -> pd.Series:
    """VWAP reseteado cada día (UTC)."""
    df = df.copy()
    df['_day'] = df['timestamp'].dt.date
    tp = (df['high'] + df['low'] + df['close']) / 3
    df['_tv'] = tp * df['volume']
    df['_cv'] = df.groupby('_day')['volume'].cumsum()
    df['_ctv'] = df.groupby('_day')['_tv'].cumsum()
    return df['_ctv'] / df['_cv']


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula todos los indicadores técnicos sobre un DataFrame OHLCV.
    df debe tener columnas: timestamp, open, high, low, close, volume
    Devuelve df con columnas nuevas añadidas.
    """
    df = df.copy().sort_values('timestamp').reset_index(drop=True)

    # EMAs
    df['ema_9']   = _ema(df['close'], 9)
    df['ema_21']  = _ema(df['close'], 21)
    df['ema_50']  = _ema(df['close'], 50)
    df['ema_200'] = _ema(df['close'], 200)

    # RSI
    df['rsi_14'] = _rsi(df['close'], 14)

    # MACD (12,26,9)
    ema12 = _ema(df['close'], 12)
    ema26 = _ema(df['close'], 26)
    df['macd']        = ema12 - ema26
    df['macd_signal'] = _ema(df['macd'], 9)
    df['macd_hist']   = df['macd'] - df['macd_signal']

    # Bollinger Bands (20,2)
    sma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    df['bb_upper'] = sma20 + 2 * std20
    df['bb_mid']   = sma20
    df['bb_lower'] = sma20 - 2 * std20
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'].replace(0, np.nan)

    # ATR
    df['atr_14'] = _atr(df, 14)

    # VWAP (solo para 1h — para 4h/1d no tiene sentido)
    try:
        df['vwap'] = _vwap_daily(df)
    except Exception:
        df['vwap'] = np.nan

    # Volume SMA 20 y ratio
    df['volume_sma']   = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma'].replace(0, np.nan)

    # Sesión
    df['session']    = df['timestamp'].apply(get_session_name)
    df['is_killzone'] = df['timestamp'].apply(is_killzone)

    return df


# ── ICT Concepts ──────────────────────────────────────────────────────────────

def detect_swings(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """
    Swing High: high[i] > high[i-n..i-1] AND high[i] > high[i+1..i+n]
    Swing Low:  low[i]  < low[i-n..i-1]  AND low[i]  < low[i+1..i+n]
    """
    df = df.copy()
    df['swing_high'] = False
    df['swing_low']  = False

    for i in range(n, len(df) - n):
        h = df['high'].iloc[i]
        l = df['low'].iloc[i]
        if (h == df['high'].iloc[i-n:i].max() and
                h == df['high'].iloc[i+1:i+n+1].max() and
                h > df['high'].iloc[i-n:i].max()):
            pass  # rewrite below with cleaner logic
        if (h > df['high'].iloc[i-n:i].max() and
                h > df['high'].iloc[i+1:i+n+1].max()):
            df.at[df.index[i], 'swing_high'] = True
        if (l < df['low'].iloc[i-n:i].min() and
                l < df['low'].iloc[i+1:i+n+1].min()):
            df.at[df.index[i], 'swing_low'] = True

    return df


def detect_bos_choch(df: pd.DataFrame) -> pd.DataFrame:
    """
    BOS Bullish: close rompe sobre el último swing high relevante
    BOS Bearish: close rompe bajo el último swing low relevante
    CHoCH: BOS en dirección opuesta a la tendencia reciente (últimos 5 BOS)
    """
    df = df.copy()
    df['bos_bullish'] = False
    df['bos_bearish'] = False
    df['choch']       = False

    last_sh = None   # último swing high price
    last_sl = None   # último swing low price
    recent_bos = []  # 'bull' o 'bear', últimos 5

    for i in range(len(df)):
        row = df.iloc[i]
        if row.get('swing_high', False):
            last_sh = row['high']
        if row.get('swing_low', False):
            last_sl = row['low']

        c = row['close']
        if last_sh is not None and c > last_sh:
            df.at[df.index[i], 'bos_bullish'] = True
            # CHoCH si tendencia previa era bajista
            if recent_bos and recent_bos[-1] == 'bear':
                df.at[df.index[i], 'choch'] = True
            recent_bos.append('bull')
            last_sh = None  # consumido
            if len(recent_bos) > 5:
                recent_bos.pop(0)

        elif last_sl is not None and c < last_sl:
            df.at[df.index[i], 'bos_bearish'] = True
            if recent_bos and recent_bos[-1] == 'bull':
                df.at[df.index[i], 'choch'] = True
            recent_bos.append('bear')
            last_sl = None
            if len(recent_bos) > 5:
                recent_bos.pop(0)

    return df


def detect_fvg(df: pd.DataFrame) -> pd.DataFrame:
    """
    FVG Bullish:  low[i+1]  > high[i-1]  — gap alcista
    FVG Bearish:  high[i+1] < low[i-1]   — gap bajista
    Marcado en la vela central (i).
    Solo FVGs que aún no han sido mitigados (precio no volvió al gap).
    """
    df = df.copy()
    df['fvg_bullish'] = False
    df['fvg_bearish'] = False

    # Registrar gaps activos: lista de (top, bottom, direction)
    active_gaps = []
    mitigated   = set()

    for i in range(1, len(df) - 1):
        c = df['close'].iloc[i]
        h_prev = df['high'].iloc[i-1]
        l_prev = df['low'].iloc[i-1]
        h_next = df['high'].iloc[i+1]
        l_next = df['low'].iloc[i+1]

        # Verificar si algún gap activo está siendo mitigado
        new_active = []
        for gap in active_gaps:
            top, bot, direction = gap
            if direction == 'bull' and c >= bot:
                mitigated.add(id(gap))
            elif direction == 'bear' and c <= top:
                mitigated.add(id(gap))
            else:
                new_active.append(gap)
        active_gaps = new_active

        if l_next > h_prev:
            df.at[df.index[i], 'fvg_bullish'] = True
            active_gaps.append((l_next, h_prev, 'bull'))

        if h_next < l_prev:
            df.at[df.index[i], 'fvg_bearish'] = True
            active_gaps.append((l_prev, h_next, 'bear'))

    return df


def detect_order_blocks(df: pd.DataFrame, move_threshold: float = 0.007) -> pd.DataFrame:
    """
    OB Bullish: última vela bajista antes de movimiento alcista > move_threshold
    OB Bearish: última vela alcista antes de movimiento bajista > move_threshold
    Válido mientras precio no cierre más allá del OB.
    """
    df = df.copy()
    df['ob_bullish'] = False
    df['ob_bearish'] = False

    n = len(df)
    for i in range(n - 4):
        # Movimiento en las 3 velas siguientes
        future_high = df['high'].iloc[i+1:i+4].max()
        future_low  = df['low'].iloc[i+1:i+4].min()
        move_up   = (future_high - df['close'].iloc[i]) / df['close'].iloc[i]
        move_down = (df['close'].iloc[i] - future_low)  / df['close'].iloc[i]

        o = df['open'].iloc[i]
        c = df['close'].iloc[i]

        if move_up >= move_threshold and c < o:   # vela bajista antes de pump
            # Verificar que no ha sido mitigado (precio no cerró bajo el low del OB)
            ob_low = df['low'].iloc[i]
            mitigated = any(
                df['close'].iloc[j] < ob_low
                for j in range(i+1, min(i+51, n))
            )
            if not mitigated:
                df.at[df.index[i], 'ob_bullish'] = True

        if move_down >= move_threshold and c > o:  # vela alcista antes de dump
            ob_high = df['high'].iloc[i]
            mitigated = any(
                df['close'].iloc[j] > ob_high
                for j in range(i+1, min(i+51, n))
            )
            if not mitigated:
                df.at[df.index[i], 'ob_bearish'] = True

    return df


def detect_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Detecta patrones de vela simples."""
    df = df.copy()
    df['candle_pattern'] = None

    for i in range(2, len(df)):
        o = df['open'].iloc[i]
        h = df['high'].iloc[i]
        l = df['low'].iloc[i]
        c = df['close'].iloc[i]
        body = abs(c - o)
        total_range = h - l if h != l else 1e-9
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l

        pattern = None

        if body < total_range * 0.1:
            pattern = "doji"
        elif lower_wick > 2 * body and upper_wick < body:
            pattern = "hammer" if c > o else "hanging_man"
        elif upper_wick > 2 * body and lower_wick < body:
            pattern = "shooting_star" if c < o else "inverted_hammer"
        else:
            # Engulfing
            o_prev = df['open'].iloc[i-1]
            c_prev = df['close'].iloc[i-1]
            if c_prev < o_prev and c > o and c > o_prev and o < c_prev:
                pattern = "engulfing_bull"
            elif c_prev > o_prev and c < o and c < o_prev and o > c_prev:
                pattern = "engulfing_bear"
            # Morning/Evening star (3 velas)
            elif i >= 2:
                o2, c2 = df['open'].iloc[i-2], df['close'].iloc[i-2]
                o1, c1 = df['open'].iloc[i-1], df['close'].iloc[i-1]
                star_body = abs(c1 - o1) / (df['high'].iloc[i-1] - df['low'].iloc[i-1] + 1e-9)
                if c2 < o2 and star_body < 0.3 and c > o and c > (o2 + c2) / 2:
                    pattern = "morning_star"
                elif c2 > o2 and star_body < 0.3 and c < o and c < (o2 + c2) / 2:
                    pattern = "evening_star"

        if pattern:
            df.at[df.index[i], 'candle_pattern'] = pattern

    return df


def calculate_all_ict(df: pd.DataFrame, swing_n: int = 3) -> pd.DataFrame:
    """Aplica todos los conceptos ICT en orden."""
    df = detect_swings(df, n=swing_n)
    df = detect_bos_choch(df)
    df = detect_fvg(df)
    df = detect_order_blocks(df)
    df = detect_candle_patterns(df)
    return df


# ── Recalcular e insertar en DB ───────────────────────────────────────────────

def _load_ohlcv(engine, timeframe: str, limit: int = 1200) -> pd.DataFrame:
    """Carga las últimas `limit` velas de btc_ohlcv para un timeframe."""
    sql = text("""
        SELECT id, timestamp, timeframe, open, high, low, close, volume
        FROM btc_ohlcv
        WHERE timeframe = :tf
        ORDER BY timestamp DESC
        LIMIT :lim
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={'tf': timeframe, 'lim': limit})
    if df.empty:
        return df
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return df


def _update_indicators_in_db(df: pd.DataFrame, engine) -> int:
    """Actualiza las columnas de indicadores en btc_ohlcv."""
    indicator_cols = [
        'rsi_14', 'ema_9', 'ema_21', 'ema_50', 'ema_200',
        'macd', 'macd_signal', 'macd_hist',
        'bb_upper', 'bb_mid', 'bb_lower', 'bb_width',
        'atr_14', 'vwap', 'volume_sma', 'volume_ratio',
        'swing_high', 'swing_low', 'bos_bullish', 'bos_bearish', 'choch',
        'fvg_bullish', 'fvg_bearish', 'ob_bullish', 'ob_bearish',
        'session', 'is_killzone', 'candle_pattern',
    ]
    sql = text("""
        UPDATE btc_ohlcv SET
            rsi_14       = :rsi_14,
            ema_9        = :ema_9,
            ema_21       = :ema_21,
            ema_50       = :ema_50,
            ema_200      = :ema_200,
            macd         = :macd,
            macd_signal  = :macd_signal,
            macd_hist    = :macd_hist,
            bb_upper     = :bb_upper,
            bb_mid       = :bb_mid,
            bb_lower     = :bb_lower,
            bb_width     = :bb_width,
            atr_14       = :atr_14,
            vwap         = :vwap,
            volume_sma   = :volume_sma,
            volume_ratio = :volume_ratio,
            swing_high   = :swing_high,
            swing_low    = :swing_low,
            bos_bullish  = :bos_bullish,
            bos_bearish  = :bos_bearish,
            choch        = :choch,
            fvg_bullish  = :fvg_bullish,
            fvg_bearish  = :fvg_bearish,
            ob_bullish   = :ob_bullish,
            ob_bearish   = :ob_bearish,
            session      = :session,
            is_killzone  = :is_killzone,
            candle_pattern = :candle_pattern
        WHERE id = :id
    """)
    updated = 0
    with engine.begin() as conn:
        for _, row in df.iterrows():
            params = {'id': int(row['id'])}
            for col in indicator_cols:
                val = row.get(col)
                if isinstance(val, (bool, np.bool_)):
                    val = bool(val)
                elif isinstance(val, float) and math.isnan(val):
                    val = None
                elif isinstance(val, (np.integer,)):
                    val = int(val)
                elif isinstance(val, (np.floating,)):
                    val = None if math.isnan(float(val)) else float(val)
                params[col] = val
            try:
                r = conn.execute(sql, params)
                updated += r.rowcount
            except Exception as e:
                log.debug(f"UPDATE indicators id={row['id']}: {e}")
    return updated


def recalculate_indicators(engine, timeframes=('5m', '15m', '1h', '4h', '1d'), lookback=1200) -> dict:
    """
    Lee las últimas `lookback` velas de cada timeframe desde btc_ohlcv,
    recalcula todos los indicadores + ICT y actualiza las filas en DB.
    """
    results = {}
    swing_n = {'5m': 5, '15m': 4, '1h': 3, '4h': 2, '1d': 2}

    for tf in timeframes:
        df = _load_ohlcv(engine, tf, limit=lookback)
        if df.empty:
            log.warning(f"No hay datos en btc_ohlcv para {tf}")
            results[tf] = 0
            continue

        log.info(f"Calculando indicadores+ICT para {tf} ({len(df)} velas)...")
        df = calculate_indicators(df)
        df = calculate_all_ict(df, swing_n=swing_n.get(tf, 3))
        n = _update_indicators_in_db(df, engine)
        log.info(f"  {tf}: {n} filas actualizadas")
        results[tf] = n

    return results


# ── Snapshot ICT para tools ───────────────────────────────────────────────────

def get_ict_snapshot(engine, timeframe: str = '1h', n_context: int = 100) -> dict:
    """
    Devuelve snapshot ICT estructurado de las últimas n_context velas.
    Usado por la tool get_ict_context().
    """
    df = _load_ohlcv(engine, timeframe, limit=n_context + 50)
    if df.empty:
        return {}

    df = calculate_indicators(df)
    df = calculate_all_ict(df, swing_n=3 if timeframe == '1h' else 2)

    last = df.iloc[-1]
    price = float(last['close'])
    now   = pd.Timestamp(last['timestamp']).to_pydatetime().replace(tzinfo=timezone.utc)

    # Swing levels cercanos (últimas 50 velas)
    recent = df.tail(50)
    sh_rows = recent[recent['swing_high'] == True]
    sl_rows = recent[recent['swing_low']  == True]

    swing_highs = [
        {
            "price": float(r['high']),
            "distance_pct": round((float(r['high']) - price) / price * 100, 3),
            "age_candles": int(len(recent) - i - 1),
        }
        for i, (_, r) in enumerate(sh_rows.iterrows())
        if abs((float(r['high']) - price) / price) < 0.05
    ]
    swing_lows = [
        {
            "price": float(r['low']),
            "distance_pct": round((float(r['low']) - price) / price * 100, 3),
            "age_candles": int(len(recent) - i - 1),
        }
        for i, (_, r) in enumerate(sl_rows.iterrows())
        if abs((float(r['low']) - price) / price) < 0.05
    ]

    # Order Blocks activos (últimas 50 velas, no mitigados)
    ob_bull_rows = recent[recent['ob_bullish'] == True]
    ob_bear_rows = recent[recent['ob_bearish'] == True]

    def _ob_entry(r, side):
        h, l = float(r['high']), float(r['low'])
        dist = (l - price) / price * 100 if side == 'bull' else (h - price) / price * 100
        return {"high": h, "low": l, "distance_pct": round(dist, 3),
                "age_candles": int(recent.index.get_loc(r.name) if hasattr(recent.index, 'get_loc') else 0)}

    ob_bull = [_ob_entry(r, 'bull') for _, r in ob_bull_rows.iterrows()]
    ob_bear = [_ob_entry(r, 'bear') for _, r in ob_bear_rows.iterrows()]

    # FVGs no mitigados
    fvg_bull = recent[recent['fvg_bullish'] == True]
    fvg_bear = recent[recent['fvg_bearish'] == True]

    fvg_above = [
        {"top": float(r['high']), "bottom": float(r['low']),
         "distance_pct": round((float(r['low']) - price) / price * 100, 3)}
        for _, r in fvg_bear.iterrows()
        if float(r['low']) > price
    ]
    fvg_below = [
        {"top": float(r['high']), "bottom": float(r['low']),
         "distance_pct": round((float(r['high']) - price) / price * 100, 3)}
        for _, r in fvg_bull.iterrows()
        if float(r['high']) < price
    ]

    # BOS y CHoCH recientes
    recent_bos_bull = recent[recent['bos_bullish'] == True]
    recent_bos_bear = recent[recent['bos_bearish'] == True]
    recent_choch    = recent[recent['choch'] == True]

    def _last_event(rows, df_ref):
        if rows.empty:
            return {"detected": False}
        last_row = rows.iloc[-1]
        age = len(df_ref) - df_ref.index.get_loc(last_row.name) - 1 if hasattr(df_ref.index, 'get_loc') else 0
        return {"detected": True, "candles_ago": age, "price": float(last_row['close'])}

    bos_bull_last = _last_event(recent_bos_bull, recent)
    bos_bear_last = _last_event(recent_bos_bear, recent)
    choch_last    = _last_event(recent_choch, recent)

    if bos_bull_last['detected'] and bos_bear_last['detected']:
        recent_bos = {
            "direction": "bullish" if bos_bull_last['candles_ago'] < bos_bear_last['candles_ago'] else "bearish",
            "candles_ago": min(bos_bull_last['candles_ago'], bos_bear_last['candles_ago']),
            "price": bos_bull_last['price'] if bos_bull_last['candles_ago'] < bos_bear_last['candles_ago'] else bos_bear_last['price'],
        }
    elif bos_bull_last['detected']:
        recent_bos = {"direction": "bullish", **bos_bull_last}
    elif bos_bear_last['detected']:
        recent_bos = {"direction": "bearish", **bos_bear_last}
    else:
        recent_bos = {"direction": "none", "candles_ago": None, "price": None}

    # Market structure
    bull_count = len(recent[recent['bos_bullish'] == True])
    bear_count = len(recent[recent['bos_bearish'] == True])
    if bull_count > bear_count + 1:
        structure = "bullish"
    elif bear_count > bull_count + 1:
        structure = "bearish"
    else:
        structure = "ranging"

    return {
        "session":              last.get('session', 'unknown'),
        "is_killzone":          bool(last.get('is_killzone', False)),
        "minutes_to_next_killzone": minutes_to_next_killzone(now),
        "swing_highs_nearby":  swing_highs[:5],
        "swing_lows_nearby":   swing_lows[:5],
        "active_ob_bullish":   ob_bull[-3:],
        "active_ob_bearish":   ob_bear[-3:],
        "unmitigated_fvg_above": fvg_above[:3],
        "unmitigated_fvg_below": fvg_below[:3],
        "recent_bos":          recent_bos,
        "recent_choch":        {"detected": choch_last['detected'],
                                "candles_ago": choch_last.get('candles_ago'),
                                "direction": "bullish" if choch_last['detected'] and recent_bos['direction'] == 'bullish' else "bearish" if choch_last['detected'] else None},
        "market_structure":    structure,
        "current_candle_pattern": last.get('candle_pattern'),
    }


def get_session_context(engine, dt: datetime = None) -> dict:
    """Contexto de sesión actual con estadísticas históricas desde btc_ohlcv."""
    if dt is None:
        dt = datetime.now(timezone.utc)

    session = get_session_name(dt)
    kz      = is_killzone(dt)
    min_next = minutes_to_next_killzone(dt)

    # HOD/LOD de hoy UTC
    today_start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    sql_hod = text("""
        SELECT MAX(high) AS hod, MIN(low) AS lod, MAX(close) AS last_close
        FROM btc_ohlcv
        WHERE timeframe = '1h'
          AND timestamp >= :start AND timestamp <= :now
    """)
    hod = lod = last_close = None
    try:
        with engine.connect() as conn:
            r = conn.execute(sql_hod, {'start': today_start, 'now': dt}).fetchone()
            if r:
                hod, lod, last_close = r
    except Exception:
        pass

    range_today_pct = ((hod - lod) / lod * 100) if hod and lod and lod > 0 else None

    # ATR promedio diario (últimos 30 días) desde daily_features
    sql_atr = text("""
        SELECT AVG(ABS(returns)) * 100 AS avg_daily_range_pct
        FROM daily_features
        WHERE asset = 'BTC'
          AND date >= :cutoff
    """)
    avg_daily_range_pct = None
    try:
        with engine.connect() as conn:
            cutoff = (dt - timedelta(days=30)).date()
            r = conn.execute(sql_atr, {'cutoff': cutoff}).fetchone()
            if r and r[0]:
                avg_daily_range_pct = float(r[0])
    except Exception:
        pass

    range_completion = (range_today_pct / avg_daily_range_pct * 100) if (range_today_pct and avg_daily_range_pct) else None

    # Bias histórico del día de la semana
    day_name = dt.strftime('%A')
    sql_dow = text("""
        SELECT AVG(returns) AS avg_ret
        FROM daily_features
        WHERE asset = 'BTC'
          AND EXTRACT(DOW FROM date) = :dow
          AND date >= :cutoff
    """)
    day_bias = "neutral"
    try:
        with engine.connect() as conn:
            cutoff2 = (dt - timedelta(days=365)).date()
            r = conn.execute(sql_dow, {'dow': dt.weekday(), 'cutoff': cutoff2}).fetchone()
            if r and r[0] is not None:
                avg_ret = float(r[0])
                day_bias = "bullish" if avg_ret > 0.003 else "bearish" if avg_ret < -0.003 else "neutral"
    except Exception:
        pass

    session_hours = {
        "asia":      "00:00–08:00 UTC",
        "london":    "07:00–12:00 UTC",
        "ny":        "13:00–22:00 UTC",
        "dead_zone": "22:00–00:00 UTC",
    }

    return {
        "current_session":          session,
        "session_hours":            session_hours.get(session, ""),
        "is_killzone":              kz,
        "minutes_to_next_killzone": min_next,
        "day_of_week":              day_name,
        "historical_day_bias":      day_bias,
        "high_of_day":              float(hod) if hod else None,
        "low_of_day":               float(lod) if lod else None,
        "range_so_far_pct":         round(range_today_pct, 3) if range_today_pct else None,
        "avg_daily_range_pct_30d":  round(avg_daily_range_pct, 3) if avg_daily_range_pct else None,
        "range_completion_pct":     round(range_completion, 1) if range_completion else None,
    }


def get_technical_levels(engine, price_now: float, timeframes=('1h', '4h')) -> list:
    """
    Devuelve lista de niveles técnicos relevantes ordenados por distancia al precio.
    """
    levels = []

    for tf in timeframes:
        df = _load_ohlcv(engine, tf, limit=300)
        if df.empty:
            continue
        df = calculate_indicators(df)
        df = calculate_all_ict(df, swing_n=3 if tf == '1h' else 2)

        last = df.iloc[-1]

        # EMAs
        for ema_col, label in [('ema_9', 'EMA9'), ('ema_21', 'EMA21'), ('ema_50', 'EMA50'), ('ema_200', 'EMA200')]:
            val = last.get(ema_col)
            if val and not math.isnan(float(val)):
                dist = (float(val) - price_now) / price_now * 100
                levels.append({
                    "price": round(float(val), 2),
                    "type": "ema",
                    "timeframe": tf,
                    "distance_pct": round(dist, 3),
                    "strength": "strong" if ema_col in ('ema_50', 'ema_200') else "medium",
                    "description": f"{label} {tf}",
                })

        # VWAP
        vwap = last.get('vwap')
        if vwap and not math.isnan(float(vwap)):
            dist = (float(vwap) - price_now) / price_now * 100
            levels.append({
                "price": round(float(vwap), 2),
                "type": "vwap",
                "timeframe": tf,
                "distance_pct": round(dist, 3),
                "strength": "strong",
                "description": f"VWAP {tf}",
            })

        # BB
        for bb_col, lbl in [('bb_upper', 'BB Upper'), ('bb_lower', 'BB Lower')]:
            val = last.get(bb_col)
            if val and not math.isnan(float(val)):
                dist = (float(val) - price_now) / price_now * 100
                levels.append({
                    "price": round(float(val), 2),
                    "type": "bb_upper" if 'upper' in bb_col else "bb_lower",
                    "timeframe": tf,
                    "distance_pct": round(dist, 3),
                    "strength": "medium",
                    "description": f"{lbl} {tf}",
                })

        # Swing Highs/Lows (últimas 30 velas)
        recent = df.tail(30)
        for _, r in recent[recent['swing_high'] == True].iterrows():
            p = float(r['high'])
            dist = (p - price_now) / price_now * 100
            if abs(dist) < 5:
                levels.append({
                    "price": round(p, 2),
                    "type": "swing_high",
                    "timeframe": tf,
                    "distance_pct": round(dist, 3),
                    "strength": "strong",
                    "description": f"Swing High {tf}",
                })
        for _, r in recent[recent['swing_low'] == True].iterrows():
            p = float(r['low'])
            dist = (p - price_now) / price_now * 100
            if abs(dist) < 5:
                levels.append({
                    "price": round(p, 2),
                    "type": "swing_low",
                    "timeframe": tf,
                    "distance_pct": round(dist, 3),
                    "strength": "strong",
                    "description": f"Swing Low {tf}",
                })

        # OB activos
        for _, r in recent[recent['ob_bullish'] == True].iterrows():
            p = float(r['low'])
            dist = (p - price_now) / price_now * 100
            levels.append({
                "price": round(p, 2),
                "type": "ob_bullish",
                "timeframe": tf,
                "distance_pct": round(dist, 3),
                "strength": "strong",
                "description": f"OB Bullish {tf}",
            })
        for _, r in recent[recent['ob_bearish'] == True].iterrows():
            p = float(r['high'])
            dist = (p - price_now) / price_now * 100
            levels.append({
                "price": round(p, 2),
                "type": "ob_bearish",
                "timeframe": tf,
                "distance_pct": round(dist, 3),
                "strength": "strong",
                "description": f"OB Bearish {tf}",
            })

    # Ordenar por distancia absoluta
    levels.sort(key=lambda x: abs(x['distance_pct']))

    # Resistencia y soporte más cercanos
    above = [l for l in levels if l['distance_pct'] > 0]
    below = [l for l in levels if l['distance_pct'] < 0]
    nearest_resistance = above[0] if above else None
    nearest_support    = below[0] if below else None

    rr_long  = (above[0]['distance_pct'] / abs(below[0]['distance_pct'])) if above and below else None
    rr_short = (abs(below[0]['distance_pct']) / above[0]['distance_pct'])  if above and below else None

    return {
        "price_now": price_now,
        "levels": levels[:20],  # top 20 más cercanos
        "nearest_resistance": nearest_resistance,
        "nearest_support": nearest_support,
        "risk_reward_long":  round(rr_long, 2)  if rr_long else None,
        "risk_reward_short": round(rr_short, 2) if rr_short else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# AUCTION MARKET THEORY — Volume Profile, CVD, Wyckoff
# ─────────────────────────────────────────────────────────────────────────────

def calculate_volume_profile(df: pd.DataFrame, n_bars: int = 24, bins: int = 50) -> dict:
    """
    Volume Profile de las últimas n_bars velas.
    Devuelve POC (Point of Control), VAH/VAL (70% del volumen), y el perfil completo.
    """
    recent = df.tail(n_bars).copy()
    if recent.empty or 'volume' not in recent.columns:
        return {"error": "datos insuficientes"}

    lo = float(recent['low'].min())
    hi = float(recent['high'].max())
    if hi <= lo:
        return {"error": "rango inválido"}

    bin_size = (hi - lo) / bins
    profile = [0.0] * bins

    for _, row in recent.iterrows():
        c_lo, c_hi, vol = float(row['low']), float(row['high']), float(row.get('volume', 0))
        if vol <= 0 or c_hi <= c_lo:
            continue
        c_range = c_hi - c_lo
        for i in range(bins):
            bin_lo = lo + i * bin_size
            bin_hi = bin_lo + bin_size
            overlap = max(0.0, min(c_hi, bin_hi) - max(c_lo, bin_lo))
            profile[i] += vol * (overlap / c_range)

    total_vol = sum(profile)
    if total_vol == 0:
        return {"error": "sin volumen"}

    poc_idx   = profile.index(max(profile))
    poc_price = round(lo + (poc_idx + 0.5) * bin_size, 2)

    # Value Area = 70% del volumen alrededor del POC
    target = total_vol * 0.70
    accumulated = profile[poc_idx]
    low_idx = high_idx = poc_idx
    while accumulated < target and (low_idx > 0 or high_idx < bins - 1):
        add_low  = profile[low_idx - 1]  if low_idx > 0        else -1
        add_high = profile[high_idx + 1] if high_idx < bins - 1 else -1
        if add_high >= add_low:
            high_idx += 1
            accumulated += profile[high_idx]
        else:
            low_idx -= 1
            accumulated += profile[low_idx]

    vah = round(lo + (high_idx + 1) * bin_size, 2)
    val = round(lo + low_idx * bin_size, 2)

    profile_out = [
        {"price": round(lo + (i + 0.5) * bin_size, 2), "volume": round(profile[i], 4)}
        for i in range(bins)
    ]

    return {
        "poc": poc_price,
        "vah": vah,
        "val": val,
        "total_volume": round(total_vol, 4),
        "n_bars": n_bars,
        "profile": profile_out,
    }


def calculate_cvd(df: pd.DataFrame) -> pd.Series:
    """
    Cumulative Volume Delta aproximado.
    Velas alcistas (close>open) → volumen positivo, bajistas → negativo.
    Divergencia CVD vs precio señala reversiones tempranas.
    """
    df = df.copy()
    df['delta'] = df.apply(
        lambda r: float(r['volume']) if float(r['close']) >= float(r['open'])
                  else -float(r['volume']),
        axis=1
    )
    return df['delta'].cumsum()


def calculate_wyckoff_phase(df_4h: pd.DataFrame, lookback: int = 50) -> dict:
    """
    Detecta la fase Wyckoff actual sobre las últimas `lookback` velas 4h.
    Fases: accumulation, markup, distribution, markdown, unknown.
    """
    recent = df_4h.tail(lookback).copy()
    if len(recent) < 20:
        return {"phase": "unknown", "confidence": 0.0}

    closes  = recent['close'].values.astype(float)
    highs   = recent['high'].values.astype(float)
    lows    = recent['low'].values.astype(float)
    volumes = recent['volume'].values.astype(float) if 'volume' in recent.columns else None

    # Tendencia general (primeras vs últimas 10 velas)
    trend_up   = closes[-10:].mean() > closes[:10].mean() * 1.01
    trend_down = closes[-10:].mean() < closes[:10].mean() * 0.99
    trend_flat = not trend_up and not trend_down

    # Rango de las últimas 20 velas
    hi20 = highs[-20:].max()
    lo20 = lows[-20:].min()
    range_pct = (hi20 - lo20) / lo20 * 100 if lo20 > 0 else 0

    # Volumen decreciente en rango lateral (señal de accumulation/distribution)
    vol_trend = "flat"
    if volumes is not None and len(volumes) >= 20:
        v_early = volumes[-20:-10].mean()
        v_late  = volumes[-10:].mean()
        if v_late < v_early * 0.85:
            vol_trend = "decreasing"
        elif v_late > v_early * 1.15:
            vol_trend = "increasing"

    # Spring: mínimo reciente que rompe soporte pero cierra por encima del rango
    spring = False
    if len(lows) >= 10:
        range_low = lows[-10:-1].min()
        last_low, last_close = lows[-1], closes[-1]
        if last_low < range_low * 0.998 and last_close > range_low:
            spring = True

    # Upthrust: máximo reciente que rompe resistencia pero cierra por debajo del rango
    upthrust = False
    if len(highs) >= 10:
        range_high = highs[-10:-1].max()
        last_high, last_close = highs[-1], closes[-1]
        if last_high > range_high * 1.002 and last_close < range_high:
            upthrust = True

    # Effort vs Result: vela grande que cierra cerca del open = debilidad oculta
    avg_range = (highs - lows).mean()
    last_range = highs[-1] - lows[-1]
    last_body  = abs(closes[-1] - recent['open'].values[-1])
    effort_vs_result = "neutral"
    if last_range > avg_range * 1.8 and last_body < last_range * 0.25:
        effort_vs_result = "high_effort_low_result"

    # Determinar fase
    if trend_up and vol_trend == "increasing":
        phase, confidence = "markup", 0.70
    elif trend_down and vol_trend == "increasing":
        phase, confidence = "markdown", 0.70
    elif trend_flat and vol_trend == "decreasing" and spring:
        phase, confidence = "accumulation", 0.80
    elif trend_flat and vol_trend == "decreasing" and upthrust:
        phase, confidence = "distribution", 0.80
    elif trend_flat and vol_trend == "decreasing":
        phase, confidence = "accumulation", 0.45
    elif trend_up and vol_trend == "decreasing":
        phase, confidence = "distribution", 0.55
    elif trend_down and vol_trend == "decreasing":
        phase, confidence = "markup", 0.40  # posible cambio
    else:
        phase, confidence = "unknown", 0.20

    return {
        "phase":             phase,
        "confidence":        round(confidence, 2),
        "spring_detected":   spring,
        "upthrust_detected": upthrust,
        "effort_vs_result":  effort_vs_result,
        "range_pct_20bars":  round(range_pct, 2),
        "vol_trend":         vol_trend,
        "trend_direction":   "up" if trend_up else "down" if trend_down else "flat",
    }


def get_auction_theory_snapshot(engine, n_bars_vp: int = 24) -> dict:
    """
    Snapshot completo de Auction Market Theory para el dashboard:
    Volume Profile (1h), CVD (1h), Wyckoff (4h).
    """
    result = {}

    try:
        df_1h = _load_ohlcv(engine, '1h', limit=max(n_bars_vp + 10, 100))
        if not df_1h.empty:
            result['volume_profile'] = calculate_volume_profile(df_1h, n_bars=n_bars_vp)
            cvd = calculate_cvd(df_1h)
            result['cvd'] = {
                "current": round(float(cvd.iloc[-1]), 4) if len(cvd) > 0 else None,
                "prev_5":  round(float(cvd.iloc[-5]), 4) if len(cvd) >= 5 else None,
                "trend":   "rising"  if len(cvd) >= 5 and cvd.iloc[-1] > cvd.iloc[-5] else
                           "falling" if len(cvd) >= 5 and cvd.iloc[-1] < cvd.iloc[-5] else "flat",
                "divergence": None,
            }
            # CVD divergence: precio sube pero CVD baja → distribución oculta
            if len(df_1h) >= 10:
                price_trend = df_1h['close'].iloc[-1] > df_1h['close'].iloc[-10]
                cvd_trend   = cvd.iloc[-1] > cvd.iloc[-10]
                if price_trend and not cvd_trend:
                    result['cvd']['divergence'] = "bearish_divergence"
                elif not price_trend and cvd_trend:
                    result['cvd']['divergence'] = "bullish_divergence"
    except Exception as e:
        result['volume_profile'] = {"error": str(e)}
        result['cvd'] = {"error": str(e)}

    try:
        df_4h = _load_ohlcv(engine, '4h', limit=60)
        if not df_4h.empty:
            result['wyckoff'] = calculate_wyckoff_phase(df_4h, lookback=50)
    except Exception as e:
        result['wyckoff'] = {"error": str(e)}

    return result
