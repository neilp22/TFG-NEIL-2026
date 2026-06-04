# pipeline/load_bitcoin_news_sentiments.py
# Carga bitcoin-news-sentiments.csv en raw_texts para procesamiento con FinBERT
# Columnas: Headlines | Date | Sentiments | Source
# Uso: python pipeline/load_bitcoin_news_sentiments.py

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from sqlalchemy import text
from db.db_utils import get_engine

FILE_PATH = '/Users/neilpradasmartinez/Downloads/bitcoin-news-sentiments.csv'

# Keywords para asegurar relevancia a BTC
BTC_KEYWORDS = {
    'bitcoin', 'btc', 'crypto', 'cryptocurrency', 'blockchain',
    'satoshi', 'halving', 'coinbase', 'binance', 'altcoin',
    'bullish', 'bearish', 'defi', 'hodl', 'mining',
}

def is_relevant(text: str) -> bool:
    return any(kw in str(text).lower() for kw in BTC_KEYWORDS)

def insert_texts(posts: list) -> int:
    if not posts:
        return 0
    engine = get_engine()
    inserted = 0
    with engine.begin() as conn:
        for p in posts:
            res = conn.execute(text("""
                INSERT INTO raw_texts
                    (timestamp, asset, source, text, url, processed)
                VALUES
                    (:timestamp, :asset, :source, :text, :url, :processed)
                ON CONFLICT DO NOTHING
            """), p)
            inserted += res.rowcount
    return inserted

def load():
    print("="*55)
    print("CARGANDO: bitcoin-news-sentiments.csv → raw_texts")
    print("="*55)

    df = pd.read_csv(FILE_PATH)
    print(f"  Filas totales en CSV: {len(df)}")

    # Parsear fechas
    df['_ts'] = pd.to_datetime(df['Date'], errors='coerce', utc=True)
    df = df.dropna(subset=['_ts', 'Headlines'])
    print(f"  Filas con fecha válida: {len(df)}")
    print(f"  Período: {df['_ts'].min().date()} → {df['_ts'].max().date()}")

    # Filtrar por relevancia BTC
    mask = df['Headlines'].apply(is_relevant)
    df_btc = df[mask].copy()
    print(f"  Filas relevantes a BTC: {len(df_btc)} "
          f"({len(df_btc)/len(df)*100:.1f}%)")

    # Preparar posts para inserción
    posts = []
    for _, row in df_btc.iterrows():
        headline = str(row['Headlines']).strip()[:800]
        url = str(row['Source']).strip() if pd.notna(row['Source']) else ''

        posts.append({
            'timestamp': row['_ts'],
            'asset':     'BTC',
            'source':    'kaggle_btc_news_sentiments',
            'text':      headline,
            'url':       url[:500],
            'processed': False,  # FinBERT lo procesará después
        })

    print(f"\n  Insertando {len(posts)} titulares en raw_texts...")
    n = insert_texts(posts)
    print(f"  ✅ Nuevos insertados: {n} "
          f"({len(posts)-n} ya existían)")

    # Resumen por año
    df_btc['year'] = df_btc['_ts'].dt.year
    print(f"\n  Distribución por año:")
    for year, count in df_btc.groupby('year').size().items():
        print(f"    {year}: {count} titulares")

    return n


if __name__ == '__main__':
    n = load()
    print(f"\n{'='*55}")
    print(f"Total insertado: {n} titulares en raw_texts")
    print(f"\nSiguientes pasos:")
    print(f"  1. python pipeline/sentiment_processor.py")
    print(f"  2. python pipeline/feature_builder.py")
    print(f"  3. python analysis/coverage_analysis.py")