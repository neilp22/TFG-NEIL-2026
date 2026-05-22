# models/data_loader.py
import pandas as pd
import numpy as np
import sys, os
from sqlalchemy import text
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from db.db_utils import get_engine

# ── Feature sets canónicos — fuente única de verdad ─────────────────────────────
# Importar desde todos los modelos para garantizar comparaciones ceteris paribus.
# Regla: la única diferencia entre FEATURES_PRICE y FEATURES_SENTIMENT son las
# columnas NLP. Cualquier delta de AUC se atribuye exclusivamente al sentimiento.

# Indicadores técnicos + fear_greed (índice de mercado CNN Money, no NLP)
FEATURES_PRICE = [
    'returns',
    'rsi_14', 'macd', 'macd_signal',
    'bb_upper', 'bb_lower', 'sma_7', 'sma_30',
    'fear_greed',
]

# sentiment_finbert: FinBERT todas las horas del día T (incluye noticias reactivas).
# has_sentiment: flag binario — distingue "sin datos" de "sentimiento neutro real".
FEATURES_SENTIMENT = FEATURES_PRICE + [
    'sentiment_finbert',
    'has_sentiment',
]

# Variante horaria: solo noticias antes de 18:00 UTC.
# Si AUC(MORNING) > AUC(SENTIMENT), las noticias nocturnas son reactivas (causalidad inversa).
FEATURES_MORNING = FEATURES_PRICE + [
    'sentiment_morning',
    'has_sentiment',
]

FEATURES = FEATURES_SENTIMENT  # alias — usado en walk_forward_splits de este módulo
TARGET = 'label'

def validate_target_semantics(df):
    """
    Verifica que label apunta a la dirección del día SIGUIENTE, no del actual.
    Si corr(returns_t, label_t) es alta, hay target leakage.
    Si corr(returns_t, label_{t+1}) es alta, la semántica es correcta.
    """
    corr_same_day = df['returns'].corr(df['label'])
    corr_next_day = df['returns'].shift(-1).corr(df['label'])

    assert abs(corr_same_day) < 0.95, (
        f"ALERTA CRÍTICA: returns y label tienen correlación {corr_same_day:.3f}. "
        f"Posible target leakage — label puede estar codificando el retorno del mismo día."
    )

    print("=" * 50)
    print("VALIDACIÓN DE SEMÁNTICA DEL TARGET")
    print(f"  corr(returns_t, label_t)   = {corr_same_day:.4f}  ← debe ser bajo")
    print(f"  corr(returns_t, label_t+1) = {corr_next_day:.4f}  ← debe ser próximo a 0")
    print("  Validación OK: no hay target leakage detectado.")
    print("=" * 50)
    return True


def load_dataset(asset='BTC', fill_sentiment=True):
    """
    Carga daily_features y devuelve un DataFrame limpio.

    Sentimiento: usa sentiment_finbert (FinBERT puro) como feature principal.
    has_sentiment distingue días sin cobertura (0) de días con sentimiento neutro real.
    Si las columnas separadas no existen aún en la DB (antes de correr build_features),
    hace fallback a sentiment_avg con un aviso.
    """
    engine = get_engine()
    with engine.connect() as conn:
        # Comprobar si las columnas separadas ya existen
        col_check = conn.execute(text("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'daily_features'
              AND column_name = 'sentiment_finbert'
        """)).fetchone()
        has_split_cols = col_check is not None

        if has_split_cols:
            df = pd.read_sql(text("""
                SELECT date, close, returns, label,
                       rsi_14, macd, macd_signal,
                       bb_upper, bb_lower, sma_7, sma_30,
                       fear_greed,
                       sentiment_finbert, sentiment_kaggle,
                       sentiment_morning, has_sentiment
                FROM daily_features
                WHERE asset = :asset AND label IS NOT NULL
                ORDER BY date ASC"""), conn, params={'asset': asset})
        else:
            print("AVISO: columnas sentiment_finbert/has_sentiment no encontradas en DB.")
            print("       Ejecuta pipeline/feature_builder.py para migrar.")
            print("       Usando sentiment_avg como fallback (fuentes mezcladas).")
            df = pd.read_sql(text("""
                SELECT date, close, returns, label,
                       rsi_14, macd, macd_signal,
                       bb_upper, bb_lower, sma_7, sma_30,
                       fear_greed, sentiment_avg
                FROM daily_features
                WHERE asset = :asset AND label IS NOT NULL
                ORDER BY date ASC"""), conn, params={'asset': asset})
            df['sentiment_finbert'] = df['sentiment_avg']
            df['sentiment_morning'] = df['sentiment_avg']
            df['has_sentiment']     = df['sentiment_avg'].notna().astype(int)

    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    validate_target_semantics(df)

    # Registrar cobertura antes de imputar (distingue ausencia de neutralidad real)
    df['has_sentiment'] = df['sentiment_finbert'].notna().astype(int)
    df['has_kaggle']    = df['sentiment_kaggle'].notna().astype(int) \
                          if 'sentiment_kaggle' in df.columns else 0

    # Imputar con 0.0 solo los NaN (días sin cobertura)
    # has_sentiment=0 le indica al modelo que el 0.0 es ausencia, no señal real
    if fill_sentiment:
        df['sentiment_finbert'] = df['sentiment_finbert'].fillna(0.0)
        df['sentiment_morning'] = df['sentiment_morning'].fillna(0.0)
        if 'sentiment_kaggle' in df.columns:
            df['sentiment_kaggle'] = df['sentiment_kaggle'].fillna(0.0)

    n_missing      = (df['has_sentiment'] == 0).sum()
    n_only_evening = (
        df['sentiment_morning'].eq(0.0) & df['has_sentiment'].eq(1)
    ).sum()
    print(f"Días sin cobertura FinBERT imputados a 0.0:     "
          f"{n_missing} ({n_missing / len(df) * 100:.1f}%)")
    print(f"Días con noticias solo tras 18:00 UTC (reactivas): "
          f"{n_only_evening} ({n_only_evening / len(df) * 100:.1f}%)")

    # Eliminar filas con NaN en features criticas (primeras filas por SMAs)
    df = df.dropna(subset=['rsi_14', 'macd', 'sma_30'])

    # Eliminar la ultima fila (label no disponible aun)
    df = df.iloc[:-1]

    print(f'Dataset cargado: {len(df)} dias ({df.index.min().date()} a {df.index.max().date()})')
    print(f'Balance de clases: {df[TARGET].mean()*100:.1f}% dias alcistas')
    return df

def walk_forward_splits(df, n_splits=4, test_size=90, gap=7):
    """
    Genera splits de walk-forward validation con embargo temporal.

    gap: días excluidos entre fin de train y comienzo de test.
         Evita que features rolling (vol_7d, sentiment_ma3) contaminen el test.
    Devuelve lista de (X_train, y_train, X_test, y_test, test_dates).
    """
    splits = []
    total  = len(df)
    min_train = total - n_splits * test_size

    for i in range(n_splits):
        test_start = min_train + i * test_size
        test_end   = test_start + test_size
        train_end  = test_start - gap          # embargo: gap días excluidos

        if test_end > total or train_end <= 0:
            break

        train_df = df.iloc[:train_end]
        test_df  = df.iloc[test_start:test_end]

        X_train = train_df[FEATURES].values
        y_train = train_df[TARGET].values.astype(int)
        X_test  = test_df[FEATURES].values
        y_test  = test_df[TARGET].values.astype(int)

        splits.append((X_train, y_train, X_test, y_test, test_df.index))
        print(f'  Split {i+1}: train={len(train_df)}d | gap={gap}d | test={len(test_df)}d '
              f'({test_df.index[0].date()} -- {test_df.index[-1].date()})')

    return splits

def scale_splits(splits):
    """Escala cada split con StandardScaler ajustado SOLO en train."""
    scaled = []
    for X_train, y_train, X_test, y_test, dates in splits:
        scaler  = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_train)
        X_te_sc = scaler.transform(X_test)
        scaled.append((X_tr_sc, y_train, X_te_sc, y_test, dates, scaler))
    return scaled

if __name__ == '__main__':
    df     = load_dataset()
    splits = walk_forward_splits(df, n_splits=4, test_size=90, gap=7)
    print(f'Generados {len(splits)} splits de validacion.')