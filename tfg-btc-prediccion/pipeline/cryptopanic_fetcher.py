# pipeline/kaggle_loader.py
# Descarga datasets de Kaggle e inserta los titulares en raw_texts
# Uso: python pipeline/kaggle_loader.py

import kagglehub
import os
import sys
import pandas as pd
from datetime import timezone
from sqlalchemy import text

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from db.db_utils import get_engine

# ── Datasets a descargar ──────────────────────────────────────────────────────
DATASETS = [
    {
        'slug': 'imadallal/sentiment-analysis-of-bitcoin-news-2021-2024',
        'source': 'kaggle_btc_news_2021_2024',
    },
    {
        'slug': 'aaroncbastian/crypto-news-headlines-and-market-prices-by-date',
        'source': 'kaggle_crypto_news_2024',
    },
]

# ── Keywords para filtrar ─────────────────────────────────────────────────────
BTC_KEYWORDS = {
    'bitcoin', 'btc', 'crypto', 'cryptocurrency', 'blockchain',
    'satoshi', 'halving', 'coinbase', 'binance', 'altcoin',
    'bullish', 'bearish', 'defi', 'digital asset',
}

def is_relevant(text: str) -> bool:
    return any(kw in str(text).lower() for kw in BTC_KEYWORDS)


def find_csv(folder: str) -> str:
    """Encuentra el primer CSV en la carpeta descargada."""
    for fname in os.listdir(folder):
        if fname.endswith('.csv'):
            return os.path.join(folder, fname)
    raise FileNotFoundError(f"No se encontró ningún CSV en {folder}")


def detect_columns(df: pd.DataFrame) -> dict:
    """
    Detecta automáticamente las columnas relevantes del CSV.
    Devuelve un dict con las claves: timestamp, text, url
    """
    cols = {c.lower().strip(): c for c in df.columns}

    # Columna de fecha
    ts_col = None
    for candidate in ['date', 'datetime', 'published_at', 'timestamp',
                       'published', 'time', 'created_at', 'publishedat']:
        if candidate in cols:
            ts_col = cols[candidate]
            break

    # Columna de texto/título
    text_col = None
    for candidate in ['title', 'headline', 'text', 'news', 'content',
                       'description', 'summary', 'article']:
        if candidate in cols:
            text_col = cols[candidate]
            break

    # Columna de URL
    url_col = None
    for candidate in ['url', 'link', 'source_url', 'article_url']:
        if candidate in cols:
            url_col = cols[candidate]
            break

    return {
        'ts_col': ts_col,
        'text_col': text_col,
        'url_col': url_col,
    }


def insert_texts(posts: list) -> int:
    """Inserta en raw_texts evitando duplicados."""
    if not posts:
        return 0
    engine = get_engine()
    inserted = 0
    with engine.begin() as conn:
        for p in posts:
            res = conn.execute(text("""
                INSERT INTO raw_texts (timestamp, asset, source, text, url, processed)
                VALUES (:timestamp, :asset, :source, :text, :url, :processed)
                ON CONFLICT DO NOTHING
            """), p)
            inserted += res.rowcount
    return inserted


def load_dataframe(df: pd.DataFrame, source: str, asset: str = 'BTC') -> int:
    """Procesa un DataFrame y lo inserta en raw_texts."""

    print(f"\n  Columnas disponibles: {list(df.columns)}")
    resolved = detect_columns(df)
    print(f"  Detectado → fecha: '{resolved['ts_col']}' | "
          f"texto: '{resolved['text_col']}' | url: '{resolved['url_col']}'")

    if not resolved['text_col']:
        print("  ⚠ No se encontró columna de texto. Columnas disponibles:")
        for c in df.columns:
            print(f"     - {c}: {df[c].iloc[0] if len(df) > 0 else 'vacío'}")
        return 0

    posts = []
    skipped = 0

    for _, row in df.iterrows():
        # Texto
        raw_text = str(row.get(resolved['text_col'], '')).strip()
        if not raw_text or raw_text == 'nan':
            skipped += 1
            continue

        # Filtro de relevancia
        if not is_relevant(raw_text):
            skipped += 1
            continue

        # Timestamp
        ts = pd.Timestamp.now(tz='UTC')
        if resolved['ts_col']:
            try:
                parsed = pd.to_datetime(row[resolved['ts_col']], errors='coerce', utc=True)
                if pd.notna(parsed):
                    ts = parsed
            except Exception:
                pass

        # URL
        url = ''
        if resolved['url_col']:
            url = str(row.get(resolved['url_col'], ''))[:500]
            if url == 'nan':
                url = ''

        posts.append({
            'timestamp': ts,
            'asset': asset,
            'source': source,
            'text': raw_text[:800],
            'url': url,
            'processed': False,
        })

    print(f"  Filas totales: {len(df)} | "
          f"Relevantes: {len(posts)} | Descartadas: {skipped}")

    n = insert_texts(posts)
    print(f"  ✅ Insertadas en BD: {n} nuevas")
    return n


def process_dataset(slug: str, source: str, asset: str = 'BTC') -> int:
    """Descarga un dataset de Kaggle y lo carga en raw_texts."""
    print(f"\n{'='*55}")
    print(f"Descargando: {slug}")
    print('='*55)

    try:
        path = kagglehub.dataset_download(slug)
        print(f"  Descargado en: {path}")
    except Exception as e:
        print(f"  ⚠ Error descargando {slug}: {e}")
        return 0

    try:
        csv_path = find_csv(path)
        print(f"  CSV encontrado: {os.path.basename(csv_path)}")
    except FileNotFoundError as e:
        # Buscar en subcarpetas
        total = 0
        for root, dirs, files in os.walk(path):
            for fname in files:
                if fname.endswith('.csv'):
                    fpath = os.path.join(root, fname)
                    print(f"\n  Procesando: {fname}")
                    df = pd.read_csv(fpath, low_memory=False)
                    total += load_dataframe(df, source=source, asset=asset)
        return total

    df = pd.read_csv(csv_path, low_memory=False)
    return load_dataframe(df, source=source, asset=asset)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Carga datasets de Kaggle en raw_texts')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['all', '2021_2024', '2024_2026'],
                        help='Qué dataset cargar (default: all)')
    args = parser.parse_args()

    total = 0

    if args.dataset in ('all', '2021_2024'):
        total += process_dataset(
            slug='imadallal/sentiment-analysis-of-bitcoin-news-2021-2024',
            source='kaggle_btc_2021_2024',
        )

    if args.dataset in ('all', '2024_2026'):
        total += process_dataset(
            slug='aaroncbastian/crypto-news-headlines-and-market-prices-by-date',
            source='kaggle_crypto_headlines',
        )

    print(f"\n{'='*55}")
    print(f"TOTAL INSERTADO: {total} titulares en raw_texts")
    print('='*55)
    print("\nSiguiente paso:")
    print("  python db/verify_texts.py")
    print("  python pipeline/sentiment_processor.py")