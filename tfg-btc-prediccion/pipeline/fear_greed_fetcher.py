# pipeline/fear_greed_fetcher.py
import requests
import pandas as pd
from sqlalchemy import text
from datetime import date
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from db.db_utils import get_engine
 
FNG_URL = 'https://api.alternative.me/fng/?limit=0&format=json'
 
def fetch_fear_greed_history():
    """Descarga todo el histórico del Fear & Greed Index."""
    resp = requests.get(FNG_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()['data']
    df = pd.DataFrame(data)[['value', 'timestamp']]
    df['date']        = pd.to_datetime(df['timestamp'].astype(int), unit='s').dt.date
    df['fear_greed']  = df['value'].astype(int)
    df['asset']       = 'BTC'
    return df[['date', 'asset', 'fear_greed']].sort_values('date')
 
def upsert_fear_greed(df):
    """
    Inserta o actualiza el fear_greed en daily_features.
    Crea la fila si no existe; solo actualiza fear_greed si ya existe.
    """
    engine = get_engine()
    count = 0
    with engine.begin() as conn:
        for _, row in df.iterrows():
            conn.execute(text("""
                INSERT INTO daily_features (date, asset, fear_greed)
                VALUES (:date, :asset, :fear_greed)
                ON CONFLICT (date, asset)
                DO UPDATE SET fear_greed = EXCLUDED.fear_greed,
                              updated_at = NOW()
            """), row.to_dict())
            count += 1
    return count
 
if __name__ == '__main__':
    print('Descargando Fear & Greed Index histórico...')
    df = fetch_fear_greed_history()
    print(f'  Descargados {len(df)} valores ({df.date.min()} – {df.date.max()})')
    n = upsert_fear_greed(df)
    print(f'  Insertados/actualizados {n} registros en daily_features.')

