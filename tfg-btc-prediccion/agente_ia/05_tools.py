"""
05_tools.py — 9 herramientas del agente IA financiero BTC

Herramientas:
  1. query_market            — precio + indicadores intraday desde btc_ohlcv
  2. run_ml_prediction       — ensemble model con ICT features
  3. rag_search              — búsqueda semántica en noticias (pgvector)
  4. get_sentiment           — agregado de sentimiento FinBERT reciente
  5. get_fear_greed          — Fear & Greed Index histórico
  6. get_ict_context         — conceptos ICT (OBs, FVGs, BOS, CHoCH, sesión)
  7. get_session_stats       — estadísticas de sesión con histórico
  8. get_technical_levels    — niveles técnicos cercanos ordenados por distancia
  9. get_multi_timeframe_bias — confluencia de bias entre 1h, 4h, 1d

Usado por 06_agent.py para el tool-calling loop de OpenAI.
"""

import json
import logging
import math
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / 'config' / '.env')

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import requests
from sqlalchemy import text

from db.db_utils import get_engine

log = logging.getLogger(__name__)

_HERE   = Path(__file__).parent
_ROOT   = _HERE.parent
_MODELS = _ROOT / 'models' / 'saved'


def _load_indicators():
    """Import indicators module con fallback."""
    try:
        from agente_ia import indicators as _ind
        return _ind
    except ImportError:
        import importlib.util
        spec = importlib.util.spec_from_file_location("indicators", _HERE / "indicators.py")
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


# ── Definiciones de herramientas (OpenAI function calling) ────────────────────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "query_market",
            "strict": False,
            "description": (
                "Devuelve precio actual de Bitcoin, cambios 1h/4h/24h, "
                "indicadores técnicos (RSI, MACD, BB, EMAs, VWAP, ATR) "
                "y las últimas N velas desde btc_ohlcv. "
                "Fallback a daily_features si no hay datos intraday."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "timeframe": {
                        "type": "string",
                        "description": "Timeframe: '1h', '4h' o '1d'. Default: '1h'.",
                    },
                    "candles": {
                        "type": "integer",
                        "description": "Número de velas a devolver en history (1-100). Default: 20.",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_ml_prediction",
            "strict": False,
            "description": (
                "Ejecuta el modelo ensemble (GBM+RF+Logistic) entrenado con "
                "features ICT intraday. Devuelve dirección bullish/bearish, "
                "probabilidad y confianza. Si AUC < 0.55, devuelve valid=false."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "strict": False,
            "description": (
                "Busca en el corpus de noticias financieras usando similitud semántica "
                "(pgvector). Devuelve las noticias más relevantes para una consulta, "
                "con filtro temporal para evitar look-ahead bias."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Pregunta o tema a buscar en las noticias.",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Número de noticias a devolver (1-10). Default: 5.",
                    },
                    "before_date": {
                        "type": "string",
                        "description": "Filtro temporal: solo noticias anteriores (YYYY-MM-DD).",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sentiment",
            "strict": False,
            "description": (
                "Devuelve el sentimiento FinBERT agregado de noticias BTC "
                "para los últimos N días."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "Número de días a analizar (1-30). Default: 7.",
                    }
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_fear_greed",
            "strict": False,
            "description": (
                "Devuelve el Fear & Greed Index de Bitcoin para los últimos N días. "
                "0-25: Miedo extremo, 25-50: Miedo, 50-75: Codicia, 75-100: Codicia extrema."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "Número de días históricos (1-30). Default: 7.",
                    }
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_ict_context",
            "strict": False,
            "description": (
                "Devuelve análisis ICT (Inner Circle Trader) completo: "
                "sesión actual, killzones, swing highs/lows, Order Blocks activos, "
                "Fair Value Gaps no mitigados, BOS y CHoCH recientes, "
                "estructura de mercado y patrón de vela actual."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "timeframe": {
                        "type": "string",
                        "description": "Timeframe a analizar: '1h' o '4h'. Default: '1h'.",
                    }
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_session_stats",
            "strict": False,
            "description": (
                "Devuelve estadísticas de sesión de mercado: sesión actual (Asia/London/NY), "
                "killzone, high/low del día, rango recorrido vs histórico, "
                "bias histórico del día de la semana."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_technical_levels",
            "strict": False,
            "description": (
                "Devuelve todos los niveles técnicos relevantes cercanos al precio actual: "
                "EMAs, VWAP, BB, swing highs/lows, Order Blocks, FVGs. "
                "Ordenados por distancia. Incluye risk/reward para long y short."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_multi_timeframe_bias",
            "strict": False,
            "description": (
                "Analiza la confluencia de bias entre 1h, 4h y 1d. "
                "Determina si los timeframes están alineados y si hay condiciones "
                "para operar. Incluye resumen textual de la situación."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
        },
    },
]


# ── Tool 1: query_market ──────────────────────────────────────────────────────

def query_market(timeframe: str = '1h', candles: int = 20) -> dict:
    """Precio actual + indicadores técnicos desde btc_ohlcv."""
    timeframe = timeframe if timeframe in ('1h', '4h', '1d') else '1h'
    candles   = max(1, min(candles, 100))
    engine    = get_engine()

    # Intentar btc_ohlcv primero
    sql = text("""
        SELECT timestamp, open, high, low, close, volume,
               rsi_14, ema_9, ema_21, ema_50, ema_200,
               macd, macd_signal, macd_hist,
               bb_upper, bb_mid, bb_lower, bb_width,
               atr_14, vwap, volume_ratio, session, is_killzone
        FROM btc_ohlcv
        WHERE timeframe = :tf
        ORDER BY timestamp DESC
        LIMIT :n
    """)

    def _safe(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        if isinstance(v, (np.floating, np.integer)):
            v = float(v)
        return round(v, 4) if isinstance(v, float) else v

    try:
        with engine.connect() as conn:
            rows = conn.execute(sql, {'tf': timeframe, 'n': candles + 5}).fetchall()
        if rows:
            cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'rsi_14', 'ema_9', 'ema_21', 'ema_50', 'ema_200',
                    'macd', 'macd_signal', 'macd_hist',
                    'bb_upper', 'bb_mid', 'bb_lower', 'bb_width',
                    'atr_14', 'vwap', 'volume_ratio', 'session', 'is_killzone']
            df = pd.DataFrame(rows, columns=cols).sort_values('timestamp').tail(candles)
            last = df.iloc[-1]
            price = float(last['close'])

            # Cambios porcentuales
            n1h  = 1 if timeframe == '1h' else (4 if timeframe == '4h' else 1)
            n4h  = 4 if timeframe == '1h' else (1 if timeframe == '4h' else 1)
            n24h = 24 if timeframe == '1h' else 6

            def _pct_change(n):
                if len(df) <= n:
                    return None
                prev = float(df.iloc[-(n+1)]['close'])
                return round((price - prev) / prev * 100, 3) if prev else None

            # EMA trend
            e9, e21, e50 = last.get('ema_9'), last.get('ema_21'), last.get('ema_50')
            def _ema_trend(e_fast, e_slow):
                if e_fast is None or e_slow is None:
                    return 'unknown'
                return 'bullish' if float(e_fast) > float(e_slow) else 'bearish'

            # MACD
            macd_h = last.get('macd_hist')
            macd_sig = 'bullish' if macd_h and float(macd_h) > 0 else 'bearish' if macd_h else 'neutral'

            # BB position
            bb_u = last.get('bb_upper')
            bb_l = last.get('bb_lower')
            bb_pos = None
            if bb_u and bb_l and float(bb_u) != float(bb_l):
                bb_pos = round((price - float(bb_l)) / (float(bb_u) - float(bb_l)), 3)

            history = [
                {
                    'timestamp': str(r['timestamp']),
                    'open':  _safe(r['open']),
                    'high':  _safe(r['high']),
                    'low':   _safe(r['low']),
                    'close': _safe(r['close']),
                    'volume': _safe(r['volume']),
                    'rsi':   _safe(r['rsi_14']),
                }
                for _, r in df.iterrows()
            ]

            return {
                'source':       'btc_ohlcv',
                'timeframe':    timeframe,
                'timestamp':    str(last['timestamp']),
                'price_now':    price,
                f'change_{timeframe}': _pct_change(n1h),
                'change_4h':    _pct_change(n4h),
                'change_24h':   _pct_change(n24h),
                'indicators': {
                    'rsi_14':        _safe(last.get('rsi_14')),
                    'rsi_signal':    _interpret_rsi(last.get('rsi_14')),
                    'ema_9':         _safe(e9),
                    'ema_21':        _safe(e21),
                    'ema_50':        _safe(e50),
                    'ema_200':       _safe(last.get('ema_200')),
                    'ema_trend_1h':  _ema_trend(e9, e21),
                    'ema_trend_4h':  _ema_trend(e21, e50),
                    'macd':          _safe(last.get('macd')),
                    'macd_signal':   _safe(last.get('macd_signal')),
                    'macd_hist':     _safe(macd_h),
                    'macd_bias':     macd_sig,
                    'bb_upper':      _safe(bb_u),
                    'bb_lower':      _safe(bb_l),
                    'bb_position':   bb_pos,
                    'bb_width':      _safe(last.get('bb_width')),
                    'atr_14':        _safe(last.get('atr_14')),
                    'atr_normalized': round(float(last['atr_14']) / price, 5) if last.get('atr_14') and price else None,
                    'vwap':          _safe(last.get('vwap')),
                    'volume_ratio':  _safe(last.get('volume_ratio')),
                    'volume_spike':  bool(last.get('volume_ratio') and float(last['volume_ratio']) > 1.5),
                },
                'session':      last.get('session'),
                'is_killzone':  bool(last.get('is_killzone')),
                'history':      history,
            }
    except Exception as e:
        log.warning(f"btc_ohlcv query failed: {e} — fallback a daily_features")

    # Fallback: daily_features
    return _query_market_daily(candles)


def _query_market_daily(days: int = 7) -> dict:
    engine = get_engine()
    sql = text("""
        SELECT date, close, returns, rsi_14, macd, macd_signal,
               bb_upper, bb_lower, sma_7, sma_30, sentiment_avg, fear_greed
        FROM daily_features WHERE asset='BTC' ORDER BY date DESC LIMIT :d
    """)
    try:
        with engine.connect() as conn:
            rows = conn.execute(sql, {'d': days}).fetchall()
        df = pd.DataFrame(rows, columns=['date','close','returns','rsi_14','macd','macd_signal',
                                          'bb_upper','bb_lower','sma_7','sma_30','sentiment_avg','fear_greed'])
        df = df.sort_values('date')
        last = df.iloc[-1]
        def _s(v):
            if v is None or (isinstance(v, float) and math.isnan(v)): return None
            return round(float(v), 4) if isinstance(v, (int, float)) else str(v)
        return {
            'source':    'daily_features (fallback)',
            'date':      str(last['date']),
            'price_now': _s(last['close']),
            'indicators': {
                'rsi_14': _s(last['rsi_14']),
                'macd':   _s(last['macd']),
                'bb_upper': _s(last['bb_upper']),
                'bb_lower': _s(last['bb_lower']),
                'sma_7':  _s(last['sma_7']),
                'sma_30': _s(last['sma_30']),
            },
        }
    except Exception as e:
        return {'error': str(e)}


def _interpret_rsi(rsi):
    if rsi is None or (isinstance(rsi, float) and math.isnan(rsi)):
        return 'unavailable'
    rsi = float(rsi)
    if rsi >= 75:   return 'extremely_overbought'
    if rsi >= 65:   return 'overbought'
    if rsi <= 25:   return 'extremely_oversold'
    if rsi <= 35:   return 'oversold'
    return 'neutral'


# ── Tool 2: run_ml_prediction ─────────────────────────────────────────────────

def run_ml_prediction() -> dict:
    """Predicción usando ensemble model o fallback a XGBoost diario."""
    # Intentar modelo v2 (intraday ensemble)
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("train_model", _HERE / "train_model.py")
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        result = mod.predict_current()
        if result.get('valid') or result.get('reason') in ('model_not_trained', 'model_not_significant'):
            return result
    except Exception as e:
        log.warning(f"Ensemble predict failed: {e}")

    # Fallback: XGBoost diario (modelo original)
    return _run_xgboost_daily()


def _run_xgboost_daily() -> dict:
    engine = get_engine()
    sql = text("""
        SELECT date, close, rsi_14, macd, macd_signal, bb_upper, bb_lower,
               sma_7, sma_30, fear_greed, returns, sentiment_avg
        FROM daily_features WHERE asset='BTC'
        ORDER BY date DESC LIMIT 1
    """)
    try:
        with engine.connect() as conn:
            row = conn.execute(sql).fetchone()
    except Exception as e:
        return {'error': str(e)}

    if not row:
        return {'prediction': None, 'reason': 'no_data', 'valid': False}

    features = {
        'rsi_14': row[2], 'macd': row[3], 'macd_signal': row[4],
        'bb_upper': row[5], 'bb_lower': row[6], 'sma_7': row[7],
        'sma_30': row[8], 'fear_greed': row[9], 'returns': row[10],
        'sentiment_avg': float(row[11]) if row[11] is not None else 0.0,
    }

    xgb_path = _MODELS / 'xgboost_best.pkl'
    prob_up   = None
    if xgb_path.exists():
        try:
            import pickle
            with open(xgb_path, 'rb') as f:
                model = pickle.load(f)
            feat_order = ['rsi_14','macd','macd_signal','bb_upper','bb_lower',
                          'sma_7','sma_30','fear_greed','returns','sentiment_avg']
            X = pd.DataFrame([{k: features.get(k) or 0.0 for k in feat_order}])
            prob_up = float(model.predict_proba(X)[0][1])
        except Exception as e:
            log.warning(f"XGBoost pkl load: {e}")

    if prob_up is None:
        rsi  = features.get('rsi_14') or 50.0
        sent = features.get('sentiment_avg') or 0.0
        prob_up = max(0.35, min(0.65, 0.50 + sent * 0.05 + (50 - rsi) * 0.001))

    direction  = "bullish" if prob_up > 0.5 else "bearish"
    confidence = "high" if abs(prob_up - 0.5) > 0.15 else "medium" if abs(prob_up - 0.5) > 0.08 else "low"

    return {
        'model':       'XGBoost diario (AUC 0.528, no significativo)',
        'direction':   direction,
        'probability': round(prob_up, 4),
        'confidence':  confidence,
        'model_auc':   0.528,
        'valid':       False,
        'note':        'AUC no significativo (p=0.48). Modelo v2 aún no entrenado o inválido.',
    }


# ── Tool 3: rag_search ────────────────────────────────────────────────────────

def rag_search(query: str, k: int = 5, before_date: str = None) -> dict:
    k = max(1, min(k, 10))
    engine = get_engine()

    # Intento semántico
    results = None
    try:
        try:
            from agente_ia.setup_rag import semantic_search
        except ImportError:
            sys.path.insert(0, str(_HERE))
            from setup_rag import semantic_search  # type: ignore
        results = semantic_search(query, engine, k=k, before_date=before_date)
    except Exception:
        pass  # caer en fallback textual

    if results:
        return {'query': query, 'k': len(results), 'before_date': before_date,
                'method': 'semantic (pgvector)', 'results': results}

    # Fallback: búsqueda textual LIKE
    like_query = f"%{query.lower()[:50]}%"
    sql = text("""
        SELECT text, source, timestamp FROM raw_texts
        WHERE asset='BTC' AND LOWER(text) LIKE :q
        ORDER BY timestamp DESC LIMIT :k
    """)
    try:
        with engine.connect() as conn:
            rows = conn.execute(sql, {'q': like_query, 'k': k}).fetchall()
        fallback_results = [{'text': r[0][:200], 'source': r[1], 'timestamp': str(r[2]), 'similarity': None} for r in rows]
        return {'query': query, 'method': 'text_fallback (pgvector no disponible)',
                'results': fallback_results}
    except Exception as e:
        return {'error': str(e)}


# ── Tool 4: get_sentiment ─────────────────────────────────────────────────────

def get_sentiment(days: int = 7) -> dict:
    days   = max(1, min(days, 30))
    engine = get_engine()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).date()

    # Primario: raw_texts.sentiment_raw
    sql = text("""
        SELECT DATE(timestamp AT TIME ZONE 'UTC') AS date,
               AVG(sentiment_raw) AS mean_sentiment,
               STDDEV(sentiment_raw) AS std_sentiment,
               COUNT(*) AS article_count
        FROM raw_texts
        WHERE asset='BTC' AND processed=TRUE AND sentiment_raw IS NOT NULL
          AND DATE(timestamp AT TIME ZONE 'UTC') >= :cutoff
        GROUP BY DATE(timestamp AT TIME ZONE 'UTC')
        ORDER BY date DESC
    """)
    try:
        with engine.connect() as conn:
            rows = conn.execute(sql, {'cutoff': str(cutoff)}).fetchall()
        if rows:
            daily = [{'date': str(r[0]),
                      'mean_sentiment': round(float(r[1]), 4) if r[1] else None,
                      'std_sentiment':  round(float(r[2]), 4) if r[2] else None,
                      'article_count':  int(r[3])} for r in rows]
            return _build_sentiment_response(daily)
    except Exception as e:
        log.debug(f"raw_texts sentiment: {e}")

    # Fallback: daily_features
    sql2 = text("""
        SELECT date, sentiment_avg, sentiment_std, sentiment_count
        FROM daily_features WHERE asset='BTC' AND date >= :cutoff ORDER BY date DESC
    """)
    try:
        with engine.connect() as conn:
            rows = conn.execute(sql2, {'cutoff': str(cutoff)}).fetchall()
        daily = [{'date': str(r[0]),
                  'mean_sentiment': round(float(r[1]), 4) if r[1] else None,
                  'std_sentiment':  round(float(r[2]), 4) if r[2] else None,
                  'article_count':  int(r[3]) if r[3] else 0} for r in rows]
        return _build_sentiment_response(daily)
    except Exception as e:
        return {'error': str(e)}


def _build_sentiment_response(daily: list) -> dict:
    if not daily:
        return {'daily': [], 'summary': 'No sentiment data available'}
    scores = [d['mean_sentiment'] for d in daily if d['mean_sentiment'] is not None]
    avg = round(sum(scores) / len(scores), 4) if scores else None
    label = 'neutro'
    if avg is not None:
        label = 'positivo' if avg > 0.1 else 'negativo' if avg < -0.1 else 'neutro'
    return {
        'days_with_data': len(daily),
        'period_avg':     avg,
        'sentiment_label': label,
        'daily':          daily,
        'scale':          'FinBERT: -1 (muy negativo) a +1 (muy positivo)',
    }


# ── Tool 5: get_fear_greed ────────────────────────────────────────────────────

def get_fear_greed(days: int = 7) -> dict:
    days = max(1, min(days, 30))
    try:
        resp = requests.get(f"https://api.alternative.me/fng/?limit={days}&format=json", timeout=15)
        resp.raise_for_status()
        data = resp.json().get('data', [])
        if not data:
            raise ValueError("empty")
        entries = [{'date': datetime.fromtimestamp(int(d['timestamp']), tz=timezone.utc).strftime('%Y-%m-%d'),
                    'value': int(d['value']),
                    'classification': d.get('value_classification', '')} for d in data]
        avg = round(sum(e['value'] for e in entries) / len(entries), 1)
        return {'latest': entries[0], 'period_avg': avg, 'history': entries}
    except Exception as e:
        log.warning(f"F&G API: {e}")
        return _fear_greed_from_db(days)


def _fear_greed_from_db(days: int) -> dict:
    engine = get_engine()
    sql = text("SELECT date, fear_greed FROM daily_features WHERE asset='BTC' AND fear_greed IS NOT NULL ORDER BY date DESC LIMIT :d")
    try:
        with engine.connect() as conn:
            rows = conn.execute(sql, {'d': days}).fetchall()
        entries = [{'date': str(r[0]), 'value': round(float(r[1]), 1)} for r in rows]
        avg = round(sum(e['value'] for e in entries) / len(entries), 1) if entries else None
        return {'source': 'daily_features', 'latest': entries[0] if entries else None,
                'period_avg': avg, 'history': entries}
    except Exception as e:
        return {'error': str(e)}


# ── Tool 6: get_ict_context ───────────────────────────────────────────────────

def get_ict_context(timeframe: str = '1h') -> dict:
    """Snapshot ICT completo: sesión, OBs, FVGs, BOS, CHoCH, estructura."""
    timeframe = timeframe if timeframe in ('1h', '4h') else '1h'
    engine = get_engine()
    try:
        ind = _load_indicators()
        return ind.get_ict_snapshot(engine, timeframe=timeframe, n_context=100)
    except Exception as e:
        log.error(f"get_ict_context: {e}")
        return {'error': str(e), 'timeframe': timeframe}


# ── Tool 7: get_session_stats ─────────────────────────────────────────────────

def get_session_stats() -> dict:
    """Estadísticas de la sesión actual con contexto histórico."""
    engine = get_engine()
    try:
        ind = _load_indicators()
        return ind.get_session_context(engine, dt=datetime.now(timezone.utc))
    except Exception as e:
        log.error(f"get_session_stats: {e}")
        return {'error': str(e)}


# ── Tool 8: get_technical_levels ──────────────────────────────────────────────

def get_technical_levels() -> dict:
    """Niveles técnicos cercanos ordenados por distancia al precio."""
    engine = get_engine()
    try:
        # Precio actual desde btc_ohlcv o daily_features
        price = _get_current_price(engine)
        if price is None:
            return {'error': 'Cannot determine current price'}
        ind = _load_indicators()
        return ind.get_technical_levels(engine, price_now=price, timeframes=('1h', '4h'))
    except Exception as e:
        log.error(f"get_technical_levels: {e}")
        return {'error': str(e)}


def _get_current_price(engine) -> float | None:
    for tf in ('1h', '4h', '1d'):
        try:
            with engine.connect() as conn:
                r = conn.execute(text(
                    "SELECT close FROM btc_ohlcv WHERE timeframe=:tf ORDER BY timestamp DESC LIMIT 1"
                ), {'tf': tf}).fetchone()
            if r:
                return float(r[0])
        except Exception:
            pass
    try:
        with engine.connect() as conn:
            r = conn.execute(text("SELECT close FROM daily_features WHERE asset='BTC' ORDER BY date DESC LIMIT 1")).fetchone()
        return float(r[0]) if r else None
    except Exception:
        return None


# ── Tool 9: get_multi_timeframe_bias ─────────────────────────────────────────

def get_multi_timeframe_bias() -> dict:
    """Confluencia de bias entre 1h, 4h y 1d."""
    engine = get_engine()
    ind    = _load_indicators()
    biases = {}

    for tf in ('1h', '4h', '1d'):
        try:
            df = ind._load_ohlcv(engine, tf, limit=200)
            if df.empty:
                biases[tf] = {'direction': 'unknown', 'strength': 0.0, 'reasons': []}
                continue
            df = ind.calculate_indicators(df)
            df = ind.calculate_all_ict(df, swing_n=3 if tf == '1h' else 2)
            last = df.iloc[-1]

            reasons = []
            score   = 0.0

            # EMA alignment
            e9, e21, e50 = last.get('ema_9'), last.get('ema_21'), last.get('ema_50')
            price = float(last['close'])
            if e9 and e21:
                if float(e9) > float(e21):
                    score += 0.2; reasons.append("EMA9 > EMA21 (alcista)")
                else:
                    score -= 0.2; reasons.append("EMA9 < EMA21 (bajista)")
            if e21 and e50:
                if float(e21) > float(e50):
                    score += 0.15; reasons.append("EMA21 > EMA50 (alcista)")
                else:
                    score -= 0.15; reasons.append("EMA21 < EMA50 (bajista)")

            # RSI
            rsi = last.get('rsi_14')
            if rsi:
                rsi = float(rsi)
                if rsi > 55:
                    score += 0.1; reasons.append(f"RSI {rsi:.1f} (momentum alcista)")
                elif rsi < 45:
                    score -= 0.1; reasons.append(f"RSI {rsi:.1f} (momentum bajista)")

            # MACD hist
            macd_h = last.get('macd_hist')
            if macd_h:
                if float(macd_h) > 0:
                    score += 0.1; reasons.append("MACD hist positivo")
                else:
                    score -= 0.1; reasons.append("MACD hist negativo")

            # BOS recientes (últimas 10 velas)
            recent10 = df.tail(10)
            bull_bos = int(recent10.get('bos_bullish', pd.Series(dtype=bool)).sum())
            bear_bos = int(recent10.get('bos_bearish', pd.Series(dtype=bool)).sum())
            if bull_bos > bear_bos:
                score += 0.2; reasons.append(f"{bull_bos} BOS alcistas recientes")
            elif bear_bos > bull_bos:
                score -= 0.2; reasons.append(f"{bear_bos} BOS bajistas recientes")

            # CHoCH reciente
            if recent10.get('choch', pd.Series(dtype=bool)).any():
                # CHoCH rompe la tendencia — reduce certeza
                score *= 0.6
                reasons.append("CHoCH reciente — posible cambio de tendencia")

            direction = "bullish" if score > 0.1 else "bearish" if score < -0.1 else "neutral"
            strength  = round(abs(score), 3)
            biases[tf] = {'direction': direction, 'strength': strength, 'reasons': reasons}

        except Exception as e:
            log.warning(f"MTF bias {tf}: {e}")
            biases[tf] = {'direction': 'unknown', 'strength': 0.0, 'reasons': [str(e)]}

    # Confluencia
    dirs = [b['direction'] for b in biases.values() if b['direction'] != 'unknown']
    bull_count = dirs.count('bullish')
    bear_count = dirs.count('bearish')
    neut_count = dirs.count('neutral')

    if bull_count >= 2:
        confluence_dir = 'bullish'
    elif bear_count >= 2:
        confluence_dir = 'bearish'
    else:
        confluence_dir = 'neutral'

    aligned = bull_count == 3 or bear_count == 3
    conf_score = round(max(bull_count, bear_count) / max(len(dirs), 1), 2)
    trade_allowed = conf_score >= 0.67 and confluence_dir != 'neutral'

    # Resumen textual
    summaries = {
        ('bullish', 'bullish', 'bullish'): "Todos los TF alineados alcistas — bias fuerte",
        ('bearish', 'bearish', 'bearish'): "Todos los TF alineados bajistas — bias fuerte",
    }
    bias_tuple = tuple(biases.get(tf, {}).get('direction', 'unknown') for tf in ('1h', '4h', '1d'))
    summary = summaries.get(bias_tuple)
    if not summary:
        if confluence_dir == 'bullish':
            summary = f"4h y 1d alcistas ({bull_count}/3 TFs) — esperar confirmación en 1h" if biases.get('1h', {}).get('direction') != 'bullish' else "Bias alcista — 1h en confluencia"
        elif confluence_dir == 'bearish':
            summary = f"Bias bajista ({bear_count}/3 TFs)"
        else:
            summary = "TFs mixtos — sin confluencia clara, no operar"

    return {
        'bias_1h':  biases.get('1h', {}),
        'bias_4h':  biases.get('4h', {}),
        'bias_1d':  biases.get('1d', {}),
        'confluence': {
            'direction':    confluence_dir,
            'score':        conf_score,
            'aligned':      aligned,
            'trade_allowed': trade_allowed,
        },
        'summary': summary,
    }


# ── Dispatcher ────────────────────────────────────────────────────────────────

TOOLS_MAP = {
    'query_market':             query_market,
    'run_ml_prediction':        run_ml_prediction,
    'rag_search':               rag_search,
    'get_sentiment':            get_sentiment,
    'get_fear_greed':           get_fear_greed,
    'get_ict_context':          get_ict_context,
    'get_session_stats':        get_session_stats,
    'get_technical_levels':     get_technical_levels,
    'get_multi_timeframe_bias': get_multi_timeframe_bias,
}


def dispatch_tool(name: str, arguments: dict) -> dict:
    fn = TOOLS_MAP.get(name)
    if fn is None:
        return {'error': f'Tool not found: {name}'}
    try:
        return fn(**arguments)
    except Exception as e:
        log.error(f"Tool {name} error: {e}")
        return {'error': str(e)}


# ── Tool 10: get_volume_profile ───────────────────────────────────────────────

TOOL_DEFINITIONS.append({
    "type": "function",
    "function": {
        "name": "get_volume_profile",
        "strict": False,
        "description": (
            "Devuelve perfil de volumen histórico por hora y sesión: "
            "volumen medio de esta hora vs otras horas, sesión más activa, "
            "y si el volumen actual es suficiente para confirmar un setup."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
})


def get_volume_profile() -> dict:
    """Perfil de volumen por hora del día y sesión, comparado con media histórica."""
    engine = get_engine()
    import pandas as pd
    from sqlalchemy import text as _text

    sql = _text("""
        SELECT EXTRACT(HOUR FROM timestamp AT TIME ZONE 'UTC') AS hour,
               AVG(volume) AS avg_vol,
               COUNT(*) AS n
        FROM btc_ohlcv
        WHERE timeframe = '1h'
          AND timestamp >= NOW() - INTERVAL '60 days'
        GROUP BY 1
        ORDER BY 1
    """)
    try:
        with engine.connect() as conn:
            rows = conn.execute(sql).fetchall()
        hour_avg = {int(r[0]): float(r[1]) for r in rows}

        # Volumen actual (última vela 1h)
        sql2 = _text("SELECT volume, EXTRACT(HOUR FROM timestamp AT TIME ZONE 'UTC') FROM btc_ohlcv WHERE timeframe='1h' ORDER BY timestamp DESC LIMIT 1")
        with engine.connect() as conn:
            row = conn.execute(sql2).fetchone()
        cur_vol = float(row[0]) if row else None
        cur_hour = int(row[1]) if row else datetime.now(timezone.utc).hour

        avg_this_hour = hour_avg.get(cur_hour, 1)
        vol_ratio_vs_hour = (cur_vol / avg_this_hour) if cur_vol and avg_this_hour else None

        # Horas killzone
        kz_hours = list(range(7, 10)) + list(range(13, 15))
        kz_avg = sum(hour_avg.get(h, 0) for h in kz_hours) / max(len(kz_hours), 1)
        all_avg = sum(hour_avg.values()) / max(len(hour_avg), 1)

        # Ranking de volumen por hora (top 5)
        sorted_hours = sorted(hour_avg.items(), key=lambda x: x[1], reverse=True)
        top_hours = [{'hour_utc': h, 'avg_volume': round(v, 2)} for h, v in sorted_hours[:5]]

        # ¿Confirma el volumen actual para un trade?
        confirm = None
        if vol_ratio_vs_hour is not None:
            if vol_ratio_vs_hour >= 1.2:
                confirm = 'strong_confirm'
            elif vol_ratio_vs_hour >= 0.8:
                confirm = 'neutral'
            else:
                confirm = 'weak_no_confirm'

        return {
            'current_volume':       round(cur_vol, 2) if cur_vol else None,
            'avg_volume_this_hour': round(avg_this_hour, 2),
            'vol_ratio_vs_hour_avg': round(vol_ratio_vs_hour, 3) if vol_ratio_vs_hour else None,
            'trade_volume_confirmation': confirm,
            'killzone_avg_volume':  round(kz_avg, 2),
            'session_avg_volume':   round(all_avg, 2),
            'top_volume_hours_utc': top_hours,
            'note': 'vol_ratio_vs_hour_avg > 1.0 confirma el movimiento; < 0.7 = trampa potencial',
        }
    except Exception as e:
        return {'error': str(e)}


TOOLS_MAP['get_volume_profile'] = get_volume_profile


# ── Tool 11: get_trade_parameters ─────────────────────────────────────────────

TOOL_DEFINITIONS.append({
    "type": "function",
    "function": {
        "name": "get_trade_parameters",
        "strict": False,
        "description": (
            "Calcula los parámetros exactos de un trade BTC: sizing de posición, "
            "stop loss basado en ATR, take profit basado en niveles técnicos, "
            "costes reales (fees + slippage + funding), break-even y proyección "
            "de interés compuesto. Úsala cuando el usuario pida niveles de entrada, "
            "cuánto arriesgar, o cómo gestionar el riesgo del trade."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["long", "short"],
                    "description": "Dirección del trade",
                },
                "capital_usd": {
                    "type": "number",
                    "description": "Capital total disponible en USD (default: 10000)",
                },
                "risk_pct": {
                    "type": "number",
                    "description": "% del capital a arriesgar por trade (default: 1.0)",
                },
                "leverage": {
                    "type": "number",
                    "description": "Apalancamiento (1 = sin apalancamiento)",
                },
                "exchange": {
                    "type": "string",
                    "enum": ["binance_spot", "binance_futures", "bybit_futures"],
                    "description": "Exchange a usar",
                },
                "holding_hours": {
                    "type": "number",
                    "description": "Horas estimadas de duración del trade",
                },
                "entry_override": {
                    "type": "number",
                    "description": "Precio de entrada manual (opcional, si no se usa el precio live)",
                },
                "sl_override": {
                    "type": "number",
                    "description": "Stop loss manual (opcional)",
                },
                "tp_override": {
                    "type": "number",
                    "description": "Take profit manual (opcional)",
                },
            },
            "required": ["direction", "capital_usd"],
            "additionalProperties": False,
        },
    },
})


def get_trade_parameters(
    direction: str = "long",
    capital_usd: float = 10000,
    risk_pct: float = 1.0,
    leverage: float = 1.0,
    exchange: str = "binance_futures",
    holding_hours: float = 4.0,
    entry_override: float = None,
    sl_override: float = None,
    tp_override: float = None,
) -> dict:
    """
    Calcula sizing, SL, TP, costes reales y proyección de interés compuesto.
    Si no se especifican entry/sl/tp, los calcula automáticamente desde
    el precio live y el ATR 1h.
    """
    try:
        import importlib.util as _ilu
        from pathlib import Path as _P
        import sys as _sys

        # Cargar trade_calculator
        _root = _P(__file__).parent.parent
        tc_path = _P(__file__).parent / "trade_calculator.py"
        spec = _ilu.spec_from_file_location("trade_calculator", tc_path)
        tc = _ilu.module_from_spec(spec)
        spec.loader.exec_module(tc)

        # Obtener precio y ATR actuales
        mkt = query_market(timeframe='1h', candles=1)
        price_now = mkt.get('price_now') or 0
        atr = (mkt.get('indicators') or {}).get('atr_14') or (price_now * 0.005)

        entry = entry_override or price_now
        if not entry:
            return {'error': 'No se pudo obtener precio actual'}

        # SL y TP automáticos basados en ATR y dirección
        if direction == 'long':
            sl = sl_override or round(entry - 1.5 * atr, 2)
            tp = tp_override or round(entry + 2.5 * atr, 2)
        else:
            sl = sl_override or round(entry + 1.5 * atr, 2)
            tp = tp_override or round(entry - 2.5 * atr, 2)

        result = tc.calculate_trade(
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            capital_usd=capital_usd,
            risk_pct=risk_pct,
            leverage=leverage,
            exchange=exchange,
            holding_hours=holding_hours,
        )

        # Añadir info extra
        result['direction'] = direction
        result['current_price'] = price_now
        result['atr_1h'] = round(atr, 2)
        result['sl_method'] = 'manual' if sl_override else 'ATR 1.5x'
        result['tp_method'] = 'manual' if tp_override else 'ATR 2.5x'

        # Optimal risk
        optimal_risk = tc.get_optimal_risk(win_rate=0.50, rr_ratio=result.get('risk_reward_gross', 1.5))
        result['optimal_risk_pct'] = optimal_risk

        return result
    except Exception as e:
        return {'error': str(e)}


TOOLS_MAP['get_trade_parameters'] = get_trade_parameters


# ── Tool 12: get_confluence_score ─────────────────────────────────────────────

TOOL_DEFINITIONS.append({
    "type": "function",
    "function": {
        "name": "get_confluence_score",
        "strict": False,
        "description": (
            "Devuelve el score de confluencia CUANTITATIVO ya calculado combinando "
            "todos los módulos: técnico (RSI/MACD/EMA/BB), ICT (OB/FVG/BOS/CHoCH), "
            "multi-timeframe (1h/4h/1d alignment), smart_money (VWAP/volume/CVD), "
            "sentimiento (news+F&G) y ML (ensemble model). "
            "Score final: -1 (strong bear) a +1 (strong bull). "
            "USA SIEMPRE ESTA TOOL PRIMERO antes de cualquier recomendación de trade. "
            "El score es matemático — tu trabajo es explicarlo. Solo ajusta si hay razón "
            "de peso que el cálculo no captura (evento macro, conflicto fuerte), "
            "y justifica explícitamente el ajuste."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "timeframe": {
                    "type": "string",
                    "enum": ["1h", "4h"],
                    "description": "Timeframe principal para ICT. Default '1h'.",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
})

CONFLUENCE_WEIGHTS = {
    "technical":   0.20,
    "ict":         0.25,
    "mtf":         0.20,
    "smart_money": 0.10,
    "sentiment":   0.10,
    "ml":          0.15,
}


def _score_technical(mkt: dict) -> tuple[float, list[str]]:
    """Score técnico -1..+1 desde indicadores de query_market."""
    ind  = mkt.get("indicators") or {}
    price = mkt.get("price_now") or 0
    signals = []
    score_parts = []

    rsi = ind.get("rsi_14")
    if rsi is not None:
        if rsi < 30:
            score_parts.append(0.8)
            signals.append(f"RSI {rsi:.0f} sobrevendido (bullish)")
        elif rsi > 70:
            score_parts.append(-0.8)
            signals.append(f"RSI {rsi:.0f} sobrecomprado (bearish)")
        elif rsi < 45:
            score_parts.append(0.2)
            signals.append(f"RSI {rsi:.0f} zona baja")
        elif rsi > 55:
            score_parts.append(-0.2)
            signals.append(f"RSI {rsi:.0f} zona alta")
        else:
            score_parts.append(0.0)
            signals.append(f"RSI {rsi:.0f} neutral")

    macd = ind.get("macd")
    macd_sig = ind.get("macd_signal")
    if macd is not None and macd_sig is not None:
        if macd > macd_sig:
            score_parts.append(0.4)
            signals.append("MACD alcista (sobre señal)")
        else:
            score_parts.append(-0.4)
            signals.append("MACD bajista (bajo señal)")

    ema_9  = ind.get("ema_9")
    ema_21 = ind.get("ema_21")
    ema_50 = ind.get("ema_50")
    if price and ema_50:
        if price > ema_50:
            score_parts.append(0.3)
            signals.append(f"Precio sobre EMA50 ({ema_50:.0f})")
        else:
            score_parts.append(-0.3)
            signals.append(f"Precio bajo EMA50 ({ema_50:.0f})")
    if ema_9 and ema_21:
        if ema_9 > ema_21:
            score_parts.append(0.3)
            signals.append("EMA9 > EMA21 (tendencia alcista corto plazo)")
        else:
            score_parts.append(-0.3)
            signals.append("EMA9 < EMA21 (tendencia bajista corto plazo)")

    bb_upper = ind.get("bb_upper")
    bb_lower = ind.get("bb_lower")
    bb_mid   = ind.get("bb_mid")
    if price and bb_upper and bb_lower and bb_mid:
        bb_range = bb_upper - bb_lower
        bb_pos   = (price - bb_lower) / bb_range if bb_range > 0 else 0.5
        if bb_pos > 0.85:
            score_parts.append(-0.3)
            signals.append("Precio cerca BB superior (reversión posible)")
        elif bb_pos < 0.15:
            score_parts.append(0.3)
            signals.append("Precio cerca BB inferior (rebote posible)")
        else:
            score_parts.append((bb_pos - 0.5) * 0.4)

    if not score_parts:
        return 0.0, ["Sin indicadores técnicos disponibles"]
    return max(-1.0, min(1.0, sum(score_parts) / len(score_parts))), signals


def _score_ict(ict: dict, price: float) -> tuple[float, list[str]]:
    """Score ICT -1..+1 desde get_ict_context."""
    score_parts = []
    signals = []

    # Estructura de mercado
    ms = ict.get("market_structure", "")
    if ms == "bullish":
        score_parts.append(0.4)
        signals.append("Estructura de mercado alcista")
    elif ms == "bearish":
        score_parts.append(-0.4)
        signals.append("Estructura de mercado bajista")

    # BOS
    bos = ict.get("recent_bos") or {}
    if isinstance(bos, dict) and bos.get("detected"):
        d = bos.get("direction", "")
        if d == "bullish":
            score_parts.append(0.7)
            signals.append(f"BOS alcista hace {bos.get('candles_ago','?')} velas")
        elif d == "bearish":
            score_parts.append(-0.7)
            signals.append(f"BOS bajista hace {bos.get('candles_ago','?')} velas")

    # CHoCH
    choch = ict.get("recent_choch") or {}
    if isinstance(choch, dict) and choch.get("detected"):
        d = choch.get("direction", "")
        if d == "bullish":
            score_parts.append(0.5)
            signals.append("CHoCH alcista — cambio de carácter")
        elif d == "bearish":
            score_parts.append(-0.5)
            signals.append("CHoCH bajista — cambio de carácter")

    # OBs cercanos (<2%)
    obs_bull = ict.get("active_ob_bullish") or []
    obs_bear = ict.get("active_ob_bearish") or []
    close_bull = [o for o in obs_bull if abs(o.get("distance_pct", 999)) < 2.0]
    close_bear = [o for o in obs_bear if abs(o.get("distance_pct", 999)) < 2.0]
    if close_bull:
        score_parts.append(0.5)
        signals.append(f"OB alcista cercano ({close_bull[0].get('low',0):.0f}–{close_bull[0].get('high',0):.0f})")
    if close_bear:
        score_parts.append(-0.5)
        signals.append(f"OB bajista cercano ({close_bear[0].get('low',0):.0f}–{close_bear[0].get('high',0):.0f})")

    # FVGs cercanos (<2%)
    for fvg in (ict.get("unmitigated_fvg_below") or []):
        if abs(fvg.get("distance_pct", 999)) < 2.0:
            score_parts.append(0.3)
            signals.append(f"FVG alcista bajo precio ({fvg.get('bottom',0):.0f}–{fvg.get('top',0):.0f})")
    for fvg in (ict.get("unmitigated_fvg_above") or []):
        if abs(fvg.get("distance_pct", 999)) < 2.0:
            score_parts.append(-0.3)
            signals.append(f"FVG bajista sobre precio ({fvg.get('bottom',0):.0f}–{fvg.get('top',0):.0f})")

    # Killzone
    if ict.get("is_killzone"):
        signals.append(f"En killzone ({ict.get('session','')}) — alta probabilidad de movimiento")

    if not score_parts:
        return 0.0, [f"Estructura {ms or 'desconocida'} — sin señales ICT activas <2%"]
    return max(-1.0, min(1.0, sum(score_parts) / len(score_parts))), signals


def _score_mtf(mtf: dict) -> tuple[float, list[str]]:
    """Score MTF -1..+1 desde get_multi_timeframe_bias."""
    conf = mtf.get("confluence") or {}
    direction = conf.get("direction", "neutral")
    conf_score = float(conf.get("score", 0))
    aligned    = int(conf.get("aligned", 0))
    signals    = []

    individual = []
    for tf in ("1h", "4h", "1d"):
        b = mtf.get(f"bias_{tf}") or {}
        bias_d = b.get("direction", "neutral")  # key is "direction" not "bias_direction"
        signals.append(f"{tf}: {bias_d} (strength {b.get('strength', 0):.2f})")
        if bias_d == "bullish":
            individual.append(1.0)
        elif bias_d == "bearish":
            individual.append(-1.0)
        else:
            individual.append(0.0)

    score = sum(individual) / len(individual) if individual else 0.0
    aligned_bool = conf.get("aligned", False)
    if aligned_bool or aligned >= 2:
        signals.append(f"Todos los TF alineados {direction}")
    return max(-1.0, min(1.0, score)), signals


def _score_smart_money(vol_profile: dict, mkt: dict) -> tuple[float, list[str]]:
    """Score smart money -1..+1 desde VWAP, volume profile y volumen."""
    score_parts = []
    signals     = []
    price       = mkt.get("price_now") or 0
    ind         = mkt.get("indicators") or {}

    vwap = ind.get("vwap")
    if vwap and price:
        if price > vwap:
            score_parts.append(0.5)
            signals.append(f"Precio sobre VWAP ({vwap:.0f}) — sesión alcista")
        else:
            score_parts.append(-0.5)
            signals.append(f"Precio bajo VWAP ({vwap:.0f}) — sesión bajista")

    vol_confirm = vol_profile.get("trade_volume_confirmation")
    vol_ratio   = vol_profile.get("vol_ratio_vs_hour_avg")
    if vol_confirm == "strong_confirm":
        score_parts.append(0.4)
        signals.append(f"Volumen alto (ratio {vol_ratio:.2f}x) — confirma dirección")
    elif vol_confirm == "weak_no_confirm":
        score_parts.append(-0.2)
        signals.append(f"Volumen bajo (ratio {vol_ratio:.2f}x) — movimiento sin convicción")
    else:
        score_parts.append(0.0)
        signals.append(f"Volumen neutral (ratio {vol_ratio:.2f}x)" if vol_ratio else "Volumen: sin datos")

    if not score_parts:
        return 0.0, ["Sin datos de smart money"]
    return max(-1.0, min(1.0, sum(score_parts) / len(score_parts))), signals


def _score_sentiment_module(sent: dict, fg: dict) -> tuple[float, list[str]]:
    """Score sentimiento -1..+1 desde news + Fear & Greed."""
    score_parts = []
    signals     = []

    # News score — key is "period_avg" in get_sentiment()
    ns = sent.get("period_avg") if sent else None
    if ns is not None:
        score_parts.append(max(-1.0, min(1.0, float(ns) * 2)))
        signals.append(f"Sentimiento noticias: {float(ns):+.3f} ({sent.get('sentiment_label','?')})")

    # Fear & Greed (contrarian para extremos)
    fg_val = fg.get("value") if fg else None
    if fg_val is not None:
        fg_val = float(fg_val)
        fg_label = fg.get("classification", fg.get("label", ""))
        if fg_val <= 20:
            score_parts.append(0.5)
            signals.append(f"F&G {fg_val:.0f} Extreme Fear — contrarian alcista")
        elif fg_val <= 35:
            score_parts.append(0.2)
            signals.append(f"F&G {fg_val:.0f} Fear — ligeramente bullish contrarian")
        elif fg_val >= 80:
            score_parts.append(-0.5)
            signals.append(f"F&G {fg_val:.0f} Extreme Greed — contrarian bajista")
        elif fg_val >= 65:
            score_parts.append(-0.2)
            signals.append(f"F&G {fg_val:.0f} Greed — ligeramente bearish contrarian")
        else:
            score_parts.append(0.0)
            signals.append(f"F&G {fg_val:.0f} {fg_label} — neutral")

    if not score_parts:
        return 0.0, ["Sin datos de sentimiento"]
    return max(-1.0, min(1.0, sum(score_parts) / len(score_parts))), signals


def _score_ml_module(ml: dict) -> tuple[float, float, list[str]]:
    """Score ML -1..+1 + peso efectivo. Descarta si inválido o proba ~0.5."""
    signals = []
    prob = ml.get("probability")
    valid = ml.get("valid", False)
    auc   = ml.get("model_auc", 0)

    if not valid or auc < 0.55:
        signals.append(f"Modelo AUC {auc:.3f} <0.55 — peso ignorado")
        return 0.0, 0.0, signals

    if prob is None:
        return 0.0, 0.0, ["Sin predicción ML"]

    if 0.45 <= prob <= 0.55:
        signals.append(f"ML proba {prob:.3f} — sin señal clara (redistribuido)")
        return 0.0, 0.0, signals

    direction_score = (prob - 0.5) * 2  # -1..+1
    signals.append(f"ML proba {prob:.3f} — {'alcista' if prob > 0.5 else 'bajista'} (AUC {auc:.3f})")
    return max(-1.0, min(1.0, direction_score)), CONFLUENCE_WEIGHTS["ml"], signals


def get_confluence_score(timeframe: str = "1h") -> dict:
    """
    Calcula el score de confluencia ponderado de todos los módulos.
    Redistribuye el peso del ML si no es válido.
    """
    import json as _json
    from datetime import datetime as _dt, timezone as _tz

    try:
        mkt      = query_market(timeframe=timeframe, candles=5)
        price    = mkt.get("price_now") or 0
        ict      = get_ict_context(timeframe=timeframe)
        mtf      = get_multi_timeframe_bias()
        vol_prof = get_volume_profile()
        sent     = get_sentiment(days=1)
        fg_data  = get_fear_greed(days=1)
        ml_data  = run_ml_prediction()

        fg = None
        if isinstance(fg_data, dict):
            hist = fg_data.get("history") or []
            fg   = hist[0] if hist else fg_data

        # Scores por módulo
        tech_score, tech_sigs    = _score_technical(mkt)
        ict_score,  ict_sigs     = _score_ict(ict, price)
        mtf_score,  mtf_sigs     = _score_mtf(mtf)
        sm_score,   sm_sigs      = _score_smart_money(vol_prof, mkt)
        sent_score, sent_sigs    = _score_sentiment_module(sent, fg)
        ml_score, ml_w_eff, ml_sigs = _score_ml_module(ml_data)

        # Pesos efectivos (redistribuir ML si inválido)
        base_w = dict(CONFLUENCE_WEIGHTS)
        base_w["ml"] = ml_w_eff
        if ml_w_eff == 0.0:
            # Redistribuir 0.15 entre los otros módulos proporcionalmente
            other_keys = [k for k in base_w if k != "ml"]
            total_other = sum(CONFLUENCE_WEIGHTS[k] for k in other_keys)
            for k in other_keys:
                base_w[k] = CONFLUENCE_WEIGHTS[k] + (CONFLUENCE_WEIGHTS["ml"] * CONFLUENCE_WEIGHTS[k] / total_other)

        # Score final ponderado
        scores = {
            "technical":   tech_score,
            "ict":         ict_score,
            "mtf":         mtf_score,
            "smart_money": sm_score,
            "sentiment":   sent_score,
            "ml":          ml_score,
        }
        raw_score = sum(scores[k] * base_w[k] for k in scores)
        raw_score = max(-1.0, min(1.0, raw_score))

        # Label
        if raw_score >= 0.60:
            label = "STRONG_BUY"
        elif raw_score >= 0.30:
            label = "BUY"
        elif raw_score <= -0.60:
            label = "STRONG_SELL"
        elif raw_score <= -0.30:
            label = "SELL"
        else:
            label = "NEUTRAL"

        # Confianza: % de módulos activos que apuntan en la misma dirección
        active_scores = [(k, scores[k]) for k in scores if base_w[k] > 0 and scores[k] != 0.0]
        target_sign   = 1 if raw_score >= 0 else -1
        aligned_n     = sum(1 for _, v in active_scores if (v > 0) == (target_sign > 0))
        total_active  = len(active_scores) if active_scores else 1
        confidence    = aligned_n / total_active

        # Conflicto: si sentimiento apunta contra los otros 4 módulos fuertes
        main_dir  = 1 if (tech_score + ict_score + mtf_score) > 0 else -1
        conflict  = bool(sent_score * main_dir < -0.2 and abs(sent_score) > 0.2)
        conflict_note = ""
        if conflict:
            if sent_score > 0:
                conflict_note = "Sentimiento positivo en conflicto con estructura bajista técnica/ICT."
            else:
                conflict_note = "Sentimiento negativo (Extreme Fear contrarian) en conflicto con estructura alcista."

        modules_out = {
            k: {
                "score":        round(scores[k], 3),
                "weight":       round(base_w[k], 3),
                "contribution": round(scores[k] * base_w[k], 4),
                "signals":      [tech_sigs, ict_sigs, mtf_sigs, sm_sigs, sent_sigs, ml_sigs][
                    list(scores.keys()).index(k)
                ],
            }
            for k in scores
        }

        return {
            "timeframe":       timeframe,
            "timestamp":       _dt.now(_tz.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "price_now":       price,
            "modules":         modules_out,
            "raw_score":       round(raw_score, 3),
            "final_score":     round(raw_score, 3),
            "label":           label,
            "confidence":      round(confidence, 2),
            "conflict":        conflict,
            "conflict_note":   conflict_note,
            "aligned_modules": aligned_n,
            "total_modules":   total_active,
            "tradeable":       confidence >= 0.5 and not (conflict and abs(sent_score) > 0.4),
            "ml_weight_used":  round(ml_w_eff, 3),
        }
    except Exception as e:
        log.error(f"get_confluence_score error: {e}")
        return {"error": str(e)}


TOOLS_MAP['get_confluence_score'] = get_confluence_score


# ── Tool 13: get_entry_zone ───────────────────────────────────────────────────

TOOL_DEFINITIONS.append({
    "type": "function",
    "function": {
        "name": "get_entry_zone",
        "strict": False,
        "description": (
            "Busca zonas de entrada precisas en timeframes menores (5m/15m) "
            "alineadas con la dirección del bias mayor. Devuelve OBs y FVGs de 5m/15m "
            "cercanos al precio actual (<0.8%) que sirven como puntos de entrada exactos. "
            "Úsala cuando ya tienes dirección clara del confluence score y necesitas "
            "el punto de entrada óptimo. "
            "Si no hay zona válida <0.8%, devuelve el nivel más cercano y la distancia."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["long", "short"],
                    "description": "Dirección del trade prevista",
                },
            },
            "required": ["direction"],
            "additionalProperties": False,
        },
    },
})


def get_entry_zone(direction: str = "long") -> dict:
    """
    Busca en 5m y 15m OBs y FVGs en la dirección dada, cercanos al precio actual.
    """
    engine = get_engine()
    try:
        # Precio actual
        mkt   = query_market(timeframe='1h', candles=1)
        price = mkt.get("price_now") or 0
        if not price:
            return {"error": "Sin precio actual"}

        zones = []
        for tf in ("5m", "15m"):
            sql = text("""
                SELECT timestamp, open, high, low, close,
                       ob_bullish, ob_bearish, fvg_bullish, fvg_bearish
                FROM btc_ohlcv
                WHERE timeframe = :tf
                ORDER BY timestamp DESC
                LIMIT 200
            """)
            with engine.connect() as conn:
                df = pd.read_sql(sql, conn, params={"tf": tf})

            if df.empty:
                continue

            for _, row in df.iterrows():
                h, l = float(row["high"]), float(row["low"])
                mid  = (h + l) / 2
                dist = abs(mid - price) / price * 100

                if direction == "long":
                    if row.get("ob_bullish") or row.get("fvg_bullish"):
                        zone_type = "OB" if row.get("ob_bullish") else "FVG"
                        zones.append({
                            "timeframe":    tf,
                            "type":         zone_type,
                            "side":         "bullish",
                            "high":         round(h, 2),
                            "low":          round(l, 2),
                            "mid":          round(mid, 2),
                            "distance_pct": round(dist, 3),
                            "timestamp":    str(row["timestamp"]),
                        })
                else:
                    if row.get("ob_bearish") or row.get("fvg_bearish"):
                        zone_type = "OB" if row.get("ob_bearish") else "FVG"
                        zones.append({
                            "timeframe":    tf,
                            "type":         zone_type,
                            "side":         "bearish",
                            "high":         round(h, 2),
                            "low":          round(l, 2),
                            "mid":          round(mid, 2),
                            "distance_pct": round(dist, 3),
                            "timestamp":    str(row["timestamp"]),
                        })

        # Ordenar por distancia
        zones.sort(key=lambda x: x["distance_pct"])

        close_zones = [z for z in zones if z["distance_pct"] <= 0.8]
        best_entry  = zones[0] if zones else None
        has_immediate = len(close_zones) > 0

        return {
            "direction":         direction,
            "price_now":         price,
            "immediate_zones":   close_zones[:3],
            "best_entry":        best_entry,
            "has_immediate_zone": has_immediate,
            "all_zones_near":    zones[:6],
            "note": (
                "Zona inmediata disponible (<0.8%) — entrada en mercado o límite en zona."
                if has_immediate else
                f"Sin zona <0.8% del precio. Zona más cercana: "
                f"{best_entry['type']} {best_entry['timeframe']} a {best_entry['distance_pct']:.2f}% "
                f"— esperar retest en {best_entry['low']:.0f}–{best_entry['high']:.0f}."
                if best_entry else "Sin zonas disponibles en 5m/15m."
            ),
        }
    except Exception as e:
        log.error(f"get_entry_zone error: {e}")
        return {"error": str(e)}


TOOLS_MAP['get_entry_zone'] = get_entry_zone


if __name__ == '__main__':
    import json as _json
    logging.basicConfig(level=logging.INFO)

    tests = [
        ('query_market',   {'timeframe': '1h', 'candles': 5}),
        ('get_fear_greed', {'days': 3}),
        ('get_sentiment',  {'days': 3}),
        ('get_session_stats', {}),
        ('get_ict_context', {'timeframe': '1h'}),
        ('get_technical_levels', {}),
        ('get_multi_timeframe_bias', {}),
        ('run_ml_prediction', {}),
    ]
    for name, args in tests:
        print(f"\n{'='*50}")
        print(f"=== {name} ===")
        result = dispatch_tool(name, args)
        print(_json.dumps(result, indent=2, default=str)[:800])
