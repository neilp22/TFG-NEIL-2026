# pipeline/load_news_csv.py
# Carga /Users/neilpradasmartinez/Desktop/news.csv en raw_texts
# Columnas: DATETIME | HEADLINE | SUMMARY | SOURCE | URL | CATEGORIES | TAGS
# Combina HEADLINE + SUMMARY para dar más contexto a FinBERT
# Uso: python pipeline/load_news_csv.py

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from sqlalchemy import text
from db.db_utils import get_engine

FILE_PATH = '/Users/neilpradasmartinez/Desktop/news.csv'

BTC_KEYWORDS = {
    'bitcoin', 'btc', 'crypto', 'cryptocurrency', 'blockchain',
    'satoshi', 'halving', 'coinbase', 'binance', 'altcoin',
    'bullish', 'bearish', 'defi', 'hodl', 'mining', 'ethereum',
    'digital asset', 'web3', 'nft', 'stablecoin', 'rally',
    'crash', 'dump', 'pump', 'hodl', 'whale',
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
    print("CARGANDO: news.csv → raw_texts")
    print("="*55)

    df = pd.read_csv(FILE_PATH, low_memory=False)
    print(f"  Filas totales: {len(df)}")

    # Parsear fechas
    df['_ts'] = pd.to_datetime(df['DATETIME'], errors='coerce', utc=True)
    df = df.dropna(subset=['_ts', 'HEADLINE'])
    print(f"  Filas con fecha válida: {len(df)}")
    print(f"  Período: {df['_ts'].min().date()} → {df['_ts'].max().date()}")

    # Combinar HEADLINE + SUMMARY para más contexto a FinBERT
    # Limitamos el summary a 300 chars para no saturar el modelo (max 512 tokens)
    def build_text(row):
        headline = str(row['HEADLINE']).strip()
        summary  = str(row['SUMMARY']).strip() if pd.notna(row['SUMMARY']) else ''
        # Limpiar summary muy largo
        if len(summary) > 300:
            summary = summary[:300] + '...'
        if summary and summary != 'nan':
            return f"{headline}. {summary}"
        return headline

    df['_text'] = df.apply(build_text, axis=1)
    df['_text'] = df['_text'].str[:800]  # límite de seguridad

    # Filtrar relevantes a BTC
    mask = df['_text'].apply(is_relevant)
    df_btc = df[mask].copy()
    print(f"  Relevantes a BTC: {len(df_btc)} ({len(df_btc)/len(df)*100:.1f}%)")

    if df_btc.empty:
        print("  ⚠ No hay noticias relevantes a BTC")
        return 0

    # Preparar posts
    posts = []
    for _, row in df_btc.iterrows():
        url = str(row.get('URL', '')).strip()
        if url == 'nan':
            url = ''

        posts.append({
            'timestamp': row['_ts'],
            'asset':     'BTC',
            'source':    'news_csv_2025',
            'text':      row['_text'],
            'url':       url[:500],
            'processed': False,  # FinBERT lo procesará
        })

    print(f"\n  Insertando {len(posts)} noticias en raw_texts...")
    n = insert_texts(posts)
    print(f"  ✅ Nuevas insertadas:  {n}")
    print(f"  ↩ Ya existían:        {len(posts) - n}")

    # Resumen por año/mes
    df_btc['year']  = df_btc['_ts'].dt.year
    df_btc['month'] = df_btc['_ts'].dt.month
    print(f"\n  Distribución por año:")
    for year, count in df_btc.groupby('year').size().items():
        print(f"    {int(year)}: {count} noticias")

    return n


if __name__ == '__main__':
    n = load()
    print(f"\n{'='*55}")
    print(f"Total insertado: {n} noticias en raw_texts")
    print(f"\nSiguientes pasos:")
    print(f"  1. python pipeline/sentiment_processor.py")
    print(f"  2. python pipeline/feature_builder.py")
    print(f"  3. python analysis/coverage_analysis.py")