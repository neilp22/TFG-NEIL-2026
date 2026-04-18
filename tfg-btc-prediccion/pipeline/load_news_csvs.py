# pipeline/load_news_csvs.py
# Carga los dos datasets en la base de datos correctamente:
#
# bitcoin_sentiments_21_24.csv → raw_texts + sentiment_scores (ya tiene score calculado)
# crypto_sentiment_prediction_dataset.csv → NO tiene titulares, se ignora para raw_texts
#
# Uso: python pipeline/load_news_csvs.py

import pandas as pd
import os
import sys
from datetime import timezone
from sqlalchemy import text

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from db.db_utils import get_engine

# ── Rutas de los archivos ─────────────────────────────────────────────────────
FILE_2021_2024 = "/Users/neilpradasmartinez/Desktop/TFG CODIGO/newd/bitcoin_sentiments_21_24.csv"
FILE_2025      = "/Users/neilpradasmartinez/Desktop/TFG CODIGO/newd/crypto_sentiment_prediction_dataset.csv"

# ── Keywords para filtrar que sea relevante a BTC ────────────────────────────
BTC_KEYWORDS = {
    'bitcoin', 'btc', 'crypto', 'cryptocurrency', 'blockchain',
    'satoshi', 'halving', 'coinbase', 'binance', 'altcoin',
    'bullish', 'bearish', 'defi', 'digital asset', 'ethereum',
    'usd', 'market', 'price', 'rally', 'crash', 'dump', 'pump',
}

def is_relevant(text: str) -> bool:
    return any(kw in str(text).lower() for kw in BTC_KEYWORDS)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET 1: bitcoin_sentiments_21_24.csv
# Columnas: Date | Short Description | Accurate Sentiments
# → Insertar en raw_texts Y en sentiment_scores directamente
#   (no hace falta pasar por FinBERT, ya tiene el score)
# ─────────────────────────────────────────────────────────────────────────────

def load_2021_2024():
    print("\n" + "="*55)
    print("CARGANDO: bitcoin_sentiments_21_24.csv (2021-2024)")
    print("="*55)

    df = pd.read_csv(FILE_2021_2024)
    print(f"  Filas totales: {len(df)}")

    # Parsear fecha
    df['_ts'] = pd.to_datetime(df['Date'], errors='coerce', utc=True)
    df = df.dropna(subset=['_ts', 'Short Description'])

    # Filtrar solo relevantes a BTC
    mask = df['Short Description'].apply(is_relevant)
    df = df[mask].copy()
    print(f"  Filas relevantes a BTC: {len(df)}")

    engine = get_engine()
    inserted_texts = 0
    inserted_scores = 0

    with engine.begin() as conn:
        for _, row in df.iterrows():
            text_val = str(row['Short Description']).strip()[:800]
            ts = row['_ts']

            # Intentar parsear el score (puede ser float entre 0-1 o -1 a 1)
            try:
                raw_score = float(row['Accurate Sentiments'])
            except Exception:
                raw_score = 0.0

            # El score parece ser positivo [0,1] → convertir a compound [-1,1]
            # Si el score > 0.5 → positivo, si < 0.5 → negativo
            # compound = (score - 0.5) * 2  → mapea [0,1] a [-1,1]
            if 0 <= raw_score <= 1:
                compound = round((raw_score - 0.5) * 2, 4)
                score_positive = round(raw_score, 4)
                score_negative = round(1 - raw_score, 4)
                score_neutral = 0.0
            else:
                # Ya está en rango [-1, 1]
                compound = round(raw_score, 4)
                score_positive = round(max(raw_score, 0), 4)
                score_negative = round(max(-raw_score, 0), 4)
                score_neutral = round(1 - abs(raw_score), 4)

            # 1. Insertar en raw_texts con processed=TRUE (ya tenemos el score)
            res = conn.execute(text("""
                INSERT INTO raw_texts
                    (timestamp, asset, source, text, url, processed)
                VALUES
                    (:timestamp, :asset, :source, :text, :url, :processed)
                ON CONFLICT DO NOTHING
                RETURNING id
            """), {
                'timestamp': ts,
                'asset': 'BTC',
                'source': 'kaggle_btc_2021_2024',
                'text': text_val,
                'url': '',
                'processed': True,  # ya tiene score, no necesita FinBERT
            })

            row_id = res.fetchone()
            if row_id:
                inserted_texts += 1
                text_id = row_id[0]

                # 2. Insertar score directamente en sentiment_scores
                conn.execute(text("""
                    INSERT INTO sentiment_scores
                        (text_id, score_positive, score_negative,
                         score_neutral, compound_score, model_used)
                    VALUES
                        (:text_id, :score_positive, :score_negative,
                         :score_neutral, :compound_score, :model_used)
                    ON CONFLICT DO NOTHING
                """), {
                    'text_id': text_id,
                    'score_positive': score_positive,
                    'score_negative': score_negative,
                    'score_neutral': score_neutral,
                    'compound_score': compound,
                    'model_used': 'kaggle_precomputed',
                })
                inserted_scores += 1

    print(f"  ✅ raw_texts insertados:      {inserted_texts}")
    print(f"  ✅ sentiment_scores insertados: {inserted_scores}")
    return inserted_texts


# ─────────────────────────────────────────────────────────────────────────────
# DATASET 2: crypto_sentiment_prediction_dataset.csv
# Columnas: timestamp | cryptocurrency | news_sentiment_score |
#           social_sentiment_score | fear_greed_index | ...
# → No tiene titulares de texto → NO va a raw_texts
# → Tiene fear_greed_index y news_sentiment_score → los usamos para
#   rellenar daily_features directamente para BTC
# ─────────────────────────────────────────────────────────────────────────────

def load_2025_features():
    print("\n" + "="*55)
    print("CARGANDO: crypto_sentiment_prediction_dataset.csv (2025)")
    print("  → No tiene titulares. Cargando fear_greed + sentiment")
    print("    directamente en daily_features para BTC")
    print("="*55)

    df = pd.read_csv(FILE_2025)
    print(f"  Filas totales: {len(df)}")

    # Filtrar solo Bitcoin
    btc_mask = df['cryptocurrency'].str.lower().isin(['bitcoin', 'btc'])
    df_btc = df[btc_mask].copy()
    print(f"  Filas de Bitcoin: {len(df_btc)}")

    if df_btc.empty:
        print("  ⚠ No hay filas de Bitcoin en este dataset.")
        print("  Cryptos disponibles:", df['cryptocurrency'].unique()[:10])
        return 0

    df_btc['_date'] = pd.to_datetime(df_btc['timestamp'], errors='coerce').dt.date
    df_btc = df_btc.dropna(subset=['_date'])

    # Agregar por día (puede haber múltiples filas por día)
    df_daily = df_btc.groupby('_date').agg({
        'news_sentiment_score':   'mean',
        'social_sentiment_score': 'mean',
        'fear_greed_index':       'mean',
        'rsi_technical_indicator':'mean',
    }).reset_index()

    engine = get_engine()
    updated = 0

    with engine.begin() as conn:
        for _, row in df_daily.iterrows():
            conn.execute(text("""
                INSERT INTO daily_features (date, asset, fear_greed, updated_at)
                VALUES (:date, 'BTC', :fear_greed, NOW())
                ON CONFLICT (date, asset) DO UPDATE SET
                    fear_greed = EXCLUDED.fear_greed,
                    updated_at = NOW()
            """), {
                'date': row['_date'],
                'fear_greed': int(row['fear_greed_index'])
                    if pd.notna(row['fear_greed_index']) else None,
            })
            updated += 1

    print(f"  ✅ daily_features actualizados: {updated} días con fear_greed")
    return updated


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Dataset 1: titulares 2021-2024 → raw_texts + sentiment_scores
    n1 = load_2021_2024()

    # Dataset 2: features 2025 → daily_features (fear_greed)
    n2 = load_2025_features()

    print(f"\n{'='*55}")
    print("RESUMEN FINAL")
    print(f"{'='*55}")
    print(f"  Titulares 2021-2024 insertados: {n1}")
    print(f"  Días 2025 actualizados en daily_features: {n2}")
    print(f"\nSiguiente paso:")
    print(f"  python db/verify_texts.py")
    print(f"  python pipeline/feature_builder.py")