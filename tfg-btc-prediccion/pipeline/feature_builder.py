# pipeline/feature_builder.py
import pandas as pd
import numpy as np
import os, sys
from sqlalchemy import text
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from db.db_utils import get_engine
 
# ── Indicadores técnicos manuales (sin ta-lib) ───────────────────────
 
def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta  = series.diff()
    gain   = delta.clip(lower=0)
    loss   = (-delta).clip(lower=0)
    avg_g  = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_l  = loss.ewm(com=period - 1, min_periods=period).mean()
    rs     = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))
 
def calc_macd(series: pd.Series,
              fast=12, slow=26, signal=9):
    ema_fast   = series.ewm(span=fast, adjust=False).mean()
    ema_slow   = series.ewm(span=slow, adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line
 
def calc_bollinger(series: pd.Series, period: int = 20, std: float = 2.0):
    sma    = series.rolling(period).mean()
    stddev = series.rolling(period).std()
    return sma + std * stddev, sma - std * stddev
 
# ── Migración de columnas de sentimiento separadas ───────────────────

def ensure_sentiment_columns(engine):
    """
    Añade a daily_features las columnas de sentimiento separadas por fuente.
    Seguro de ejecutar múltiples veces (IF NOT EXISTS).
    """
    new_cols = {
        'sentiment_finbert': 'NUMERIC(8,4)',
        'sentiment_kaggle':  'NUMERIC(8,4)',
        'n_finbert':         'INTEGER',
        'n_kaggle':          'INTEGER',
        'has_sentiment':     'SMALLINT',
        # Corte horario: solo noticias publicadas antes de las 18:00 UTC
        # (información anterior al cierre BTC, potencialmente predictiva)
        'sentiment_morning': 'NUMERIC(8,4)',
        'n_morning':         'INTEGER',
    }
    with engine.begin() as conn:
        for col, dtype in new_cols.items():
            conn.execute(text(f"""
                ALTER TABLE daily_features
                ADD COLUMN IF NOT EXISTS {col} {dtype}
            """))
    print(f'Columnas de sentimiento verificadas: {list(new_cols.keys())}')


# ── Carga de datos ────────────────────────────────────────────────────

def load_price_data(asset: str = 'BTC') -> pd.DataFrame:
    """Carga los datos de precio diario ordenados por fecha."""
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT timestamp::date AS date, open, high, low, close, volume
            FROM price_data
            WHERE asset = :asset AND timeframe = '1d'
            ORDER BY timestamp ASC
        """), conn, params={'asset': asset})
    df['date'] = pd.to_datetime(df['date']).dt.date
    return df
 
def load_sentiment_by_day(asset: str = 'BTC') -> pd.DataFrame:
    """
    Agrega scores de sentimiento por día separando las dos fuentes incompatibles:
      - ProsusAI/finbert : distribución continua [-0.97, +0.95], media +0.05
      - kaggle_precomputed: distribución trimodal {-1, 0, +1}, media -0.28

    Mezclar ambas en un único AVG produce un feature contaminado.
    sentiment_finbert es la fuente principal para los modelos ML.
    """
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT
                rt.timestamp::date AS date,

                -- FinBERT completo (todas las horas del día T)
                AVG(ss.compound_score)
                    FILTER (WHERE ss.model_used = 'ProsusAI/finbert')
                    AS sentiment_finbert,

                -- Kaggle precomputado (etiquetas humanas {-1, 0, +1})
                AVG(ss.compound_score)
                    FILTER (WHERE ss.model_used = 'kaggle_precomputed')
                    AS sentiment_kaggle,

                -- FinBERT matutino: solo noticias publicadas antes de las 18:00 UTC.
                -- Noticias de 18:00-23:59 son contemporáneas al cierre BTC (reactivas).
                -- Noticias sin hora exacta (timestamp = 00:00:00) se incluyen aquí
                -- porque representan items de fecha-solamente, no cierre-de-día.
                AVG(ss.compound_score)
                    FILTER (
                        WHERE ss.model_used = 'ProsusAI/finbert'
                        AND EXTRACT(HOUR FROM rt.timestamp AT TIME ZONE 'UTC') < 18
                    ) AS sentiment_morning,

                -- Conteos por fuente y ventana horaria
                COUNT(*) FILTER (WHERE ss.model_used = 'ProsusAI/finbert')   AS n_finbert,
                COUNT(*) FILTER (WHERE ss.model_used = 'kaggle_precomputed') AS n_kaggle,
                COUNT(*) FILTER (
                    WHERE ss.model_used = 'ProsusAI/finbert'
                    AND EXTRACT(HOUR FROM rt.timestamp AT TIME ZONE 'UTC') < 18
                ) AS n_morning,
                COUNT(*) AS n_total,

                -- Retener avg/std globales para compatibilidad con columnas existentes
                AVG(ss.compound_score)    AS sentiment_avg,
                STDDEV(ss.compound_score) AS sentiment_std,
                COUNT(ss.id)              AS sentiment_count

            FROM sentiment_scores ss
            JOIN raw_texts rt ON rt.id = ss.text_id
            WHERE rt.asset = :asset
              AND rt.timestamp IS NOT NULL
            GROUP BY rt.timestamp::date
            ORDER BY date ASC
        """), conn, params={'asset': asset})
    df['date'] = pd.to_datetime(df['date']).dt.date
    return df
 
# ── Cálculo y guardado de features ──────────────────────────────────
 
def build_features(asset: str = 'BTC'):
    """Calcula todos los features y actualiza daily_features."""
    engine = get_engine()

    # Migrar columnas nuevas si no existen
    ensure_sentiment_columns(engine)

    # Cargar datos de precio
    df = load_price_data(asset)
    if df.empty:
        print(f'Sin datos de precio para {asset}')
        return

    # Calcular retornos logarítmicos
    df['returns'] = np.log(df['close'] / df['close'].shift(1))

    # Variable objetivo: 1 si el precio sube mañana, 0 si baja
    # IMPORTANTE: shift(-1) para predecir el día siguiente
    df['label'] = (df['returns'].shift(-1) > 0).astype('Int64')

    # Indicadores técnicos
    df['rsi_14']     = calc_rsi(df['close'], 14).round(2)
    df['macd'], df['macd_signal'] = calc_macd(df['close'])
    df['macd']       = df['macd'].round(6)
    df['macd_signal']= df['macd_signal'].round(6)
    df['bb_upper'], df['bb_lower'] = calc_bollinger(df['close'])
    df['sma_7']      = df['close'].rolling(7).mean()
    df['sma_30']     = df['close'].rolling(30).mean()

    # Unir con datos de sentimiento (separados por fuente)
    sentiment = load_sentiment_by_day(asset)
    df = df.merge(sentiment, on='date', how='left')

    # has_sentiment: 1 si hay al menos un texto de FinBERT ese día, 0 si no
    # Distingue "sentimiento neutro real" de "sin datos" (ambos quedan en 0.0 sin este flag)
    df['has_sentiment'] = df['n_finbert'].fillna(0).gt(0).astype(int)

    # Insertar/actualizar en daily_features
    inserted = 0
    with engine.begin() as conn:
        for _, row in df.iterrows():
            # Convertir NaN a None para que PostgreSQL los acepte como NULL
            r = {k: (None if pd.isna(v) else v) for k, v in row.items()}
            r['asset'] = asset
            conn.execute(text("""
                INSERT INTO daily_features
                    (date, asset, close, returns, label,
                     rsi_14, macd, macd_signal, bb_upper, bb_lower,
                     sma_7, sma_30,
                     sentiment_avg, sentiment_std, sentiment_count,
                     sentiment_finbert, sentiment_kaggle,
                     sentiment_morning,
                     n_finbert, n_kaggle, n_morning, has_sentiment,
                     updated_at)
                VALUES
                    (:date, :asset, :close, :returns, :label,
                     :rsi_14, :macd, :macd_signal, :bb_upper, :bb_lower,
                     :sma_7, :sma_30,
                     :sentiment_avg, :sentiment_std, :sentiment_count,
                     :sentiment_finbert, :sentiment_kaggle,
                     :sentiment_morning,
                     :n_finbert, :n_kaggle, :n_morning, :has_sentiment,
                     NOW())
                ON CONFLICT (date, asset) DO UPDATE SET
                    close              = EXCLUDED.close,
                    returns            = EXCLUDED.returns,
                    label              = EXCLUDED.label,
                    rsi_14             = EXCLUDED.rsi_14,
                    macd               = EXCLUDED.macd,
                    macd_signal        = EXCLUDED.macd_signal,
                    bb_upper           = EXCLUDED.bb_upper,
                    bb_lower           = EXCLUDED.bb_lower,
                    sma_7              = EXCLUDED.sma_7,
                    sma_30             = EXCLUDED.sma_30,
                    sentiment_avg      = EXCLUDED.sentiment_avg,
                    sentiment_std      = EXCLUDED.sentiment_std,
                    sentiment_count    = EXCLUDED.sentiment_count,
                    sentiment_finbert  = EXCLUDED.sentiment_finbert,
                    sentiment_kaggle   = EXCLUDED.sentiment_kaggle,
                    sentiment_morning  = EXCLUDED.sentiment_morning,
                    n_finbert          = EXCLUDED.n_finbert,
                    n_kaggle           = EXCLUDED.n_kaggle,
                    n_morning          = EXCLUDED.n_morning,
                    has_sentiment      = EXCLUDED.has_sentiment,
                    updated_at         = NOW()
            """), r)
            inserted += 1
    print(f'daily_features actualizado: {inserted} filas para {asset}')
 
if __name__ == '__main__':
    build_features('BTC')

