# models/data_loader.py
import pandas as pd
import numpy as np
import sys, os
from sqlalchemy import text
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from db.db_utils import get_engine

# Features que usaran todos los modelos
FEATURES = [
    'rsi_14', 'macd', 'macd_signal',
    'bb_upper', 'bb_lower', 'sma_7', 'sma_30',
    'fear_greed', 'returns',
    'sentiment_avg'   # puede tener NULLs -- se imputa con 0
]
TARGET = 'label'

def load_dataset(asset='BTC', fill_sentiment=True):
    """Carga daily_features y devuelve un DataFrame limpio."""
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT date, close, returns, label,
                   rsi_14, macd, macd_signal,
                   bb_upper, bb_lower, sma_7, sma_30,
                   fear_greed, sentiment_avg
            FROM daily_features
            WHERE asset = :asset AND label IS NOT NULL
            ORDER BY date ASC"""), conn, params={'asset': asset})
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    # Imputar sentiment_avg: 0 significa neutro (ausencia de datos)
    if fill_sentiment:
        df['sentiment_avg'] = df['sentiment_avg'].fillna(0.0)

    # Eliminar filas con NaN en features criticas (primeras filas por SMAs)
    df = df.dropna(subset=['rsi_14', 'macd', 'sma_30'])

    # Eliminar la ultima fila (label no disponible aun)
    df = df.iloc[:-1]

    print(f'Dataset cargado: {len(df)} dias ({df.index.min().date()} a {df.index.max().date()})')
    print(f'Balance de clases: {df[TARGET].mean()*100:.1f}% dias alcistas')
    return df

def walk_forward_splits(df, n_splits=5, test_size=60):
    """
    Genera splits de walk-forward validation.
    test_size: numero de dias en cada ventana de test (60 = ~2 meses).
    Devuelve lista de (X_train, y_train, X_test, y_test, test_dates).
    """
    splits = []
    total  = len(df)
    # El primer train usa todos los datos menos n_splits*test_size
    min_train = total - n_splits * test_size

    for i in range(n_splits):
        train_end = min_train + i * test_size
        test_end  = train_end + test_size
        if test_end > total: break

        train_df = df.iloc[:train_end]
        test_df  = df.iloc[train_end:test_end]

        X_train = train_df[FEATURES].values
        y_train = train_df[TARGET].values.astype(int)
        X_test  = test_df[FEATURES].values
        y_test  = test_df[TARGET].values.astype(int)

        splits.append((X_train, y_train, X_test, y_test, test_df.index))
        print(f'  Split {i+1}: train={len(train_df)}d | test={len(test_df)}d '
              f'({test_df.index[0].date()} -- {test_df.index[-1].date()})')

    return splits  # fuera del for

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
    splits = walk_forward_splits(df, n_splits=5, test_size=60)
    print(f'Generados {len(splits)} splits de validacion.')