# pipeline/price_fetcher.py
import time
import pandas as pd
from datetime import datetime, timezone
from binance.client import Client
from sqlalchemy import text
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from db.db_utils import get_engine
 
# Binance no requiere clave para datos históricos públicos
client = Client()
 
INTERVALS = {
    '1d':  Client.KLINE_INTERVAL_1DAY,
    '1h':  Client.KLINE_INTERVAL_1HOUR,
}
 
def fetch_klines(symbol: str, interval: str, start: str, end: str = None):
    """
    Descarga todas las velas OHLCV para un símbolo e intervalo.
    Pagina automáticamente de 1000 en 1000.
    """
    all_klines = []
    while True:
        klines = client.get_historical_klines(
            symbol, INTERVALS[interval],
            start_str=start,
            end_str=end,
            limit=1000
        )
        if not klines:
            break
        all_klines.extend(klines)
        # La última vela descargada se usa como nuevo inicio
        last_ts = klines[-1][0]
        start = str(last_ts + 1)  # +1 ms para no duplicar
        time.sleep(0.3)  # respetar rate limit
        if len(klines) < 1000:
            break  # ya no hay más datos
    return all_klines
 
def klines_to_df(klines, asset: str, timeframe: str):
    """Convierte la respuesta de Binance a un DataFrame limpio."""
    df = pd.DataFrame(klines, columns=[
        'timestamp','open','high','low','close','volume',
        'close_time','qav','trades','tbbav','tbqav','ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df['asset']     = asset
    df['timeframe'] = timeframe
    for col in ['open','high','low','close','volume']:
        df[col] = df[col].astype(float)
    return df[['timestamp','asset','timeframe','open','high','low','close','volume']]
 
def insert_price_data(df):
    """Inserta el DataFrame en price_data. Ignora duplicados."""
    engine = get_engine()
    rows_inserted = 0
    with engine.begin() as conn:
        for _, row in df.iterrows():
            conn.execute(text("""
                INSERT INTO price_data (timestamp, asset, timeframe, open, high, low, close, volume)
                VALUES (:timestamp, :asset, :timeframe, :open, :high, :low, :close, :volume)
                ON CONFLICT (timestamp, asset, timeframe) DO NOTHING
            """), row.to_dict())
            rows_inserted += 1
    return rows_inserted
 
def load_historical(asset='BTC', symbol='BTCUSDT',
                    start='1 Jan, 2019', intervals=None):
    """Carga completa histórica para un activo."""
    if intervals is None:
        intervals = ['1d', '1h']
 
    for tf in intervals:
        print(f'Descargando {asset} {tf} desde {start}...')
        klines = fetch_klines(symbol, tf, start)
        if not klines:
            print(f'  Sin datos para {tf}')
            continue
        df = klines_to_df(klines, asset, tf)
        n = insert_price_data(df)
        print(f'  Insertadas {n} filas en price_data ({tf})')
 
if __name__ == '__main__':
    # Carga histórica inicial: BTC diario y horario desde 2019
    load_historical(
        asset='BTC',
        symbol='BTCUSDT',
        start='1 Jan, 2019',
        intervals=['1d', '1h']
    )
    print('Carga histórica de precio completada.')
