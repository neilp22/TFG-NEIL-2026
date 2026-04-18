# pipeline/feature_builder_hourly.py
import pandas as pd
import numpy as np
import os, sys
from sqlalchemy import text
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from db.db_utils import get_engine
from pipeline.indicators import (
    calc_rsi, calc_macd, calc_bollinger,
    calc_atr, calc_obv, calc_stochastic, calc_williams_r,
    calc_order_blocks, calc_fair_value_gaps, calc_volume_profile
)

def load_hourly_price(asset='BTC') -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT timestamp, open, high, low, close, volume
            FROM price_data
            WHERE asset=:asset AND timeframe='1h'
            ORDER BY timestamp ASC"""), conn, params={'asset': asset})
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return df.set_index('timestamp')

def load_sentiment_by_hour(asset='BTC') -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT
                date_trunc('hour', rt.timestamp) AS hour,
                AVG(ss.compound_score)  AS sentiment_avg,
                COUNT(ss.id)            AS sentiment_count
            FROM sentiment_scores ss
            JOIN raw_texts rt ON rt.id = ss.text_id
            WHERE rt.asset = :asset
            GROUP BY date_trunc('hour', rt.timestamp)
            ORDER BY hour ASC"""), conn, params={'asset': asset})
    df['hour'] = pd.to_datetime(df['hour'], utc=True)
    return df.set_index('hour')

def build_hourly_features(asset='BTC'):
    print(f'Cargando datos de precio horario para {asset}...')
    df = load_hourly_price(asset)
    print(f'  {len(df)} velas cargadas')

    # Retornos y label
    df['returns'] = np.log(df['close'] / df['close'].shift(1))
    df['label']   = (df['returns'].shift(-1) > 0).astype('Int64')

    # Indicadores clasicos
    df['rsi_14']               = calc_rsi(df['close'], 14).round(2)
    df['macd'], df['macd_signal'] = calc_macd(df['close'])
    df['bb_upper'], df['bb_lower'] = calc_bollinger(df['close'])
    df['sma_7']  = df['close'].rolling(7).mean()
    df['sma_30'] = df['close'].rolling(30).mean()

    # Indicadores quant adicionales
    df['atr_14']    = calc_atr(df['high'], df['low'], df['close'], 14)
    df['obv']       = calc_obv(df['close'], df['volume'])
    df['stoch_k']   = calc_stochastic(df['high'], df['low'], df['close'], 14).round(2)
    df['williams_r']= calc_williams_r(df['high'], df['low'], df['close'], 14).round(2)

    # ICT: Order Blocks y Fair Value Gaps
    df['ob_bullish'], df['ob_bearish'] = calc_order_blocks(
        df['open'], df['close'], df['returns'], threshold=0.003
    )
    df['fvg_up'], df['fvg_down'] = calc_fair_value_gaps(df['high'], df['low'])

    # Volume Profile (ventana 24h)
    print('  Calculando Volume Profile (puede tardar ~3 min)...')
    df['poc'], df['va_high'], df['va_low'] = calc_volume_profile(
        df['high'], df['low'], df['close'], df['volume'], window=24
    )

    # Unir sentimiento horario
    sentiment = load_sentiment_by_hour(asset)
    df = df.join(sentiment, how='left')
    df['sentiment_avg']   = df['sentiment_avg'].fillna(0.0)
    df['sentiment_count'] = df['sentiment_count'].fillna(0).astype(int)

    # Eliminar primeras filas con NaN por las ventanas rodantes
    df = df.dropna(subset=['rsi_14', 'macd', 'sma_30', 'atr_14'])
    df = df.iloc[:-1]  # eliminar ultima fila (label no disponible)

    # Insertar en hourly_features
    engine = get_engine(); inserted = 0
    with engine.begin() as conn:
        for ts, row in df.iterrows():
            r = {k: (None if (isinstance(v, float) and np.isnan(v)) else
                     (bool(v) if isinstance(v, (bool, np.bool_)) else v))
                 for k, v in row.items()}
            r['asset'] = asset
            r['timestamp'] = ts
            conn.execute(text("""
                INSERT INTO hourly_features
                (timestamp,asset,close,returns,label,
                 rsi_14,macd,macd_signal,bb_upper,bb_lower,sma_7,sma_30,
                 atr_14,obv,stoch_k,williams_r,
                 ob_bullish,ob_bearish,fvg_up,fvg_down,
                 poc,va_high,va_low,
                 sentiment_avg,sentiment_count,updated_at)
                VALUES
                (:timestamp,:asset,:close,:returns,:label,
                 :rsi_14,:macd,:macd_signal,:bb_upper,:bb_lower,:sma_7,:sma_30,
                 :atr_14,:obv,:stoch_k,:williams_r,
                 :ob_bullish,:ob_bearish,:fvg_up,:fvg_down,
                 :poc,:va_high,:va_low,
                 :sentiment_avg,:sentiment_count,NOW())
                ON CONFLICT (timestamp, asset) DO UPDATE SET
                  close=EXCLUDED.close, returns=EXCLUDED.returns,
                  label=EXCLUDED.label, rsi_14=EXCLUDED.rsi_14,
                  macd=EXCLUDED.macd, updated_at=NOW()"""), r)
            inserted += 1
    print(f'hourly_features actualizado: {inserted} filas para {asset}')

if __name__ == '__main__':
    build_hourly_features('BTC')
