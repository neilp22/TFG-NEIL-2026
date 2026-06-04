# pipeline/load_gdelt_bigquery.py
# Carga los CSVs exportados desde BigQuery (GDELT) en raw_texts
# Columnas: fecha_raw | domain | url | title | V2Tone
# Uso: python pipeline/load_gdelt_bigquery.py

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import re
from datetime import timezone
from sqlalchemy import text
from db.db_utils import get_engine

# ── Archivos exportados desde BigQuery ───────────────────────────────────────
FILES = [
    'data/raw/bquxjob_6654261d_19dacfc1ecc.csv',   # 2026
]

BTC_KEYWORDS = {
    'bitcoin', 'btc', 'crypto', 'cryptocurrency', 'blockchain',
    'satoshi', 'halving', 'coinbase', 'binance', 'altcoin',
    'bullish', 'bearish', 'defi', 'hodl', 'mining',
    'digital asset', 'web3', 'stablecoin', 'rally', 'crash',
}

def is_relevant(text: str) -> bool:
    return any(kw in str(text).lower() for kw in BTC_KEYWORDS)

def clean_html(text: str) -> str:
    """Limpia entidades HTML del título."""
    import html
    text = html.unescape(str(text))
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_fecha(fecha_raw: str) -> pd.Timestamp:
    """Convierte YYYYMMDDHHMMSS a timestamp UTC."""
    try:
        s = str(int(fecha_raw))
        return pd.Timestamp(
            year=int(s[0:4]), month=int(s[4:6]), day=int(s[6:8]),
            hour=int(s[8:10]), minute=int(s[10:12]),
            tz='UTC'
        )
    except Exception:
        return pd.Timestamp.now(tz='UTC')

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

def load_file(filepath: str) -> int:
    fname = os.path.basename(filepath)
    print(f"\n{'─'*50}")
    print(f"Cargando: {fname}")

    if not os.path.exists(filepath):
        print(f"  ⚠ Archivo no encontrado: {filepath}")
        return 0

    df = pd.read_csv(filepath, low_memory=False)
    print(f"  Filas totales: {len(df)}")

    # Limpiar títulos con entidades HTML
    df['title'] = df['title'].fillna('').apply(clean_html)

    # Filtrar filas sin título
    df = df[df['title'].str.len() > 10].copy()
    print(f"  Con título válido: {len(df)}")

    # Filtrar relevantes a BTC
    mask = df['title'].apply(is_relevant)
    df_btc = df[mask].copy()
    print(f"  Relevantes a BTC: {len(df_btc)} ({len(df_btc)/len(df)*100:.1f}%)")

    if df_btc.empty:
        print("  ⚠ Sin noticias relevantes")
        return 0

    # Parsear fechas
    df_btc['_ts'] = df_btc['fecha_raw'].apply(parse_fecha)

    # Distribución por año
    years = df_btc['_ts'].dt.year.value_counts().sort_index()
    for year, count in years.items():
        print(f"    {year}: {count} noticias")

    # Construir posts
    posts = []
    for _, row in df_btc.iterrows():
        title = row['title'][:800]
        url   = str(row.get('url', ''))[:500]
        if url == 'nan':
            url = ''

        posts.append({
            'timestamp': row['_ts'],
            'asset':     'BTC',
            'source':    f"gdelt_{row['_ts'].year}",
            'text':      title,
            'url':       url,
            'processed': False,  # FinBERT lo procesará
        })

    n = insert_texts(posts)
    print(f"  ✅ Insertadas: {n} nuevas  |  Ya existían: {len(posts)-n}")
    return n


def load_all():
    print("="*55)
    print("CARGANDO CSVs DE BIGQUERY (GDELT) → raw_texts")
    print("="*55)

    total = 0
    for filepath in FILES:
        total += load_file(filepath)

    print(f"\n{'='*55}")
    print(f"TOTAL INSERTADO: {total} noticias en raw_texts")
    print(f"{'='*55}")
    print("\nSiguientes pasos:")
    print("  1. python pipeline/sentiment_processor.py")
    print("  2. python pipeline/feature_builder.py")
    print("  3. python analysis/coverage_analysis.py")
    return total


if __name__ == '__main__':
    load_all()