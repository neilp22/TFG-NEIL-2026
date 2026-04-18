# pipeline/binance_futures_fetcher.py
import requests
import pandas as pd
from datetime import datetime, timezone
from sqlalchemy import text
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from db.db_utils import get_engine

BINANCE_FAPI = 'https://fapi.binance.com'

def fetch_funding_rate(symbol='BTCUSDT', limit=1000) -> pd.DataFrame:
    """Descarga el historial de Funding Rate de Binance Futures."""
    url    = f'{BINANCE_FAPI}/fapi/v1/fundingRate'
    all_data = []
    end_time = None

    while True:
        params = {'symbol': symbol, 'limit': limit}
        if end_time: params['endTime'] = end_time
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        if not data: break
        all_data.extend(data)
        end_time = data[0]['fundingTime'] - 1
        if len(data) < limit: break
        # Funding Rate empieza en sept 2019 para BTC
        first_ts = data[0]['fundingTime'] / 1000
        if datetime.fromtimestamp(first_ts, tz=timezone.utc).year < 2020:
            break

    df = pd.DataFrame(all_data)
    df['timestamp'] = pd.to_datetime(df['fundingTime'], unit='ms', utc=True)
    df['funding_rate'] = df['fundingRate'].astype(float)
    return df[['timestamp', 'funding_rate']].sort_values('timestamp')

def fetch_open_interest(symbol='BTCUSDT', period='1h', limit=500) -> pd.DataFrame:
    """Descarga el historial de Open Interest de Binance Futures."""
    url    = f'{BINANCE_FAPI}/futures/data/openInterestHist'
    params = {'symbol': symbol, 'period': period, 'limit': limit}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    df['timestamp']     = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df['open_interest'] = df['sumOpenInterestValue'].astype(float)
    return df[['timestamp', 'open_interest']].sort_values('timestamp')

def upsert_funding_and_oi(asset='BTC'):
    """Inserta funding_rate y open_interest en hourly_features."""
    engine = get_engine()

    print('  Descargando Funding Rate...')
    df_fr = fetch_funding_rate()
    print(f'  {len(df_fr)} registros de funding rate descargados')

    print('  Descargando Open Interest...')
    df_oi = fetch_open_interest()
    print(f'  {len(df_oi)} registros de open interest descargados')

    # El funding rate se paga cada 8h; forward-fill para rellenar las horas intermedias
    with engine.begin() as conn:
        for _, row in df_fr.iterrows():
            conn.execute(text("""
                UPDATE hourly_features
                SET funding_rate = :fr, updated_at = NOW()
                WHERE asset = :asset
                AND timestamp >= :ts
                AND timestamp < :ts_next"""),
                {'fr': row['funding_rate'], 'asset': asset,
                 'ts': row['timestamp'],
                 'ts_next': row['timestamp'] + pd.Timedelta(hours=8)})

        for _, row in df_oi.iterrows():
            conn.execute(text("""
                UPDATE hourly_features
                SET open_interest = :oi, updated_at = NOW()
                WHERE asset = :asset
                AND timestamp = :ts"""),
                {'oi': row['open_interest'], 'asset': asset,
                 'ts': row['timestamp']})

    print('  Funding Rate y Open Interest actualizados en hourly_features.')

if __name__ == '__main__':
    upsert_funding_and_oi('BTC')
