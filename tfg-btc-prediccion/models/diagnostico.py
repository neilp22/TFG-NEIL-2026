# models/diagnostico.py
# Diagnóstico del dataset y mejora del modelo XGBoost
# Uso: python models/diagnostico.py

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from sqlalchemy import text
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score,
                             roc_auc_score, confusion_matrix,
                             classification_report)
from sklearn.utils.class_weight import compute_sample_weight
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
except ImportError:
    os.system("pip install xgboost")
    from xgboost import XGBClassifier

from db.db_utils import get_engine

# ── Configuración ─────────────────────────────────────────────────────────────
WINDOW_TEST_DAYS = 60
N_SPLITS         = 5
RANDOM_STATE     = 42

FEATURES_PRICE = [
    'returns', 'rsi_14', 'macd', 'bb_upper', 'bb_lower', 'sma_7', 'sma_30'
]
FEATURES_SENTIMENT = FEATURES_PRICE + ['sentiment_avg', 'fear_greed']


def load_data() -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT date, close, returns, label,
                   rsi_14, macd, bb_upper, bb_lower,
                   sma_7, sma_30, sentiment_avg, fear_greed
            FROM daily_features
            WHERE asset = 'BTC'
              AND label IS NOT NULL
            ORDER BY date ASC
        """), conn)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    return df


def prepare_features(df, feature_cols):
    df = df.copy()
    if 'sentiment_avg' in df.columns:
        df['sentiment_avg'] = df['sentiment_avg'].fillna(0)
    if 'fear_greed' in df.columns:
        df['fear_greed'] = df['fear_greed'].fillna(df['fear_greed'].median())
    price_features = [f for f in feature_cols if f in FEATURES_PRICE]
    df = df.dropna(subset=price_features)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PASO 1: DIAGNÓSTICO DEL DATASET
# ─────────────────────────────────────────────────────────────────────────────

def diagnostico(df):
    print("\n" + "="*55)
    print("DIAGNÓSTICO DEL DATASET")
    print("="*55)

    label_valid = df['label'].dropna()
    n_up   = int((label_valid == 1).sum())
    n_down = int((label_valid == 0).sum())
    total  = len(label_valid)
    pct_up = n_up / total * 100

    print(f"\n  Total días con label: {total}")
    print(f"  Días alcistas (1):    {n_up} ({pct_up:.1f}%)")
    print(f"  Días bajistas (0):    {n_down} ({100-pct_up:.1f}%)")

    if pct_up > 55:
        print(f"\n  ⚠ DESBALANCE: {pct_up:.1f}% alcistas")
        print("    → Aplicaremos class_weight balanceado")
        balanced = True
    elif pct_up < 45:
        print(f"\n  ⚠ DESBALANCE: solo {pct_up:.1f}% alcistas")
        print("    → Aplicaremos class_weight balanceado")
        balanced = True
    else:
        print(f"\n  ✅ Dataset balanceado ({pct_up:.1f}% alcistas)")
        balanced = False

    print(f"\n  Días con sentiment_avg: "
          f"{df['sentiment_avg'].notna().sum()} "
          f"({df['sentiment_avg'].notna().mean()*100:.1f}%)")
    print(f"  Días con fear_greed:    "
          f"{df['fear_greed'].notna().sum()} "
          f"({df['fear_greed'].notna().mean()*100:.1f}%)")

    # Correlaciones
    print("\n  Correlaciones con label:")
    features_check = ['returns', 'rsi_14', 'macd', 'sentiment_avg', 'fear_greed']
    df_check = df.copy()
    df_check['sentiment_avg'] = df_check['sentiment_avg'].fillna(0)
    df_check['fear_greed']    = df_check['fear_greed'].fillna(df_check['fear_greed'].median())
    for f in features_check:
        if f in df_check.columns:
            corr = df_check[f].corr(df_check['label'])
            print(f"    {f:<20} {corr:+.4f}")

    return balanced


# ─────────────────────────────────────────────────────────────────────────────
# PASO 2: MODELO MEJORADO CON CLASS WEIGHT
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_splits(df, n_splits, test_days):
    n = len(df)
    splits = []
    for i in range(n_splits):
        test_end   = n - i * test_days
        test_start = test_end - test_days
        train_end  = test_start
        if train_end < 100:
            break
        splits.append({
            'train':     df.iloc[:train_end],
            'test':      df.iloc[test_start:test_end],
            'split_num': n_splits - i,
        })
    return list(reversed(splits))


def evaluate_balanced(feature_cols, df, label='modelo'):
    print(f"\n{'─'*55}")
    print(f"MODELO MEJORADO: {label.upper()}")
    print(f"Features: {feature_cols}")
    print('─'*55)

    df_clean = prepare_features(df, feature_cols)
    splits   = walk_forward_splits(df_clean, N_SPLITS, WINDOW_TEST_DAYS)

    all_metrics = []

    for split in splits:
        X_train = split['train'][feature_cols].values
        y_train = split['train']['label'].values.astype(int)
        X_test  = split['test'][feature_cols].values
        y_test  = split['test']['label'].values.astype(int)

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        # Calcular scale_pos_weight para balancear clases
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        model = XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,  # balanceo de clases
            random_state=RANDOM_STATE,
            eval_metric='logloss',
            verbosity=0,
            reg_alpha=0.1,    # L1 regularización
            reg_lambda=1.0,   # L2 regularización
        )
        model.fit(X_train, y_train)

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)
        cm  = confusion_matrix(y_test, y_pred)

        metrics = {
            'split':      split['split_num'],
            'accuracy':   round(acc, 4),
            'f1':         round(f1, 4),
            'auc_roc':    round(auc, 4),
            'n_train':    len(y_train),
            'n_test':     len(y_test),
            'test_start': split['test'].index[0].date(),
            'test_end':   split['test'].index[-1].date(),
            'pred_up':    int(y_pred.sum()),
            'pred_down':  int((y_pred == 0).sum()),
        }
        all_metrics.append(metrics)

        print(f"  Split {split['split_num']} "
              f"[{metrics['test_start']} → {metrics['test_end']}] | "
              f"Acc: {acc:.3f} | F1: {f1:.3f} | AUC: {auc:.3f} | "
              f"Pred↑:{metrics['pred_up']} ↓:{metrics['pred_down']}")

    metrics_df = pd.DataFrame(all_metrics)
    summary = {
        'label':        label,
        'splits':       all_metrics,
        'acc_mean':     metrics_df['accuracy'].mean(),
        'acc_std':      metrics_df['accuracy'].std(),
        'f1_mean':      metrics_df['f1'].mean(),
        'f1_std':       metrics_df['f1'].std(),
        'auc_mean':     metrics_df['auc_roc'].mean(),
        'auc_std':      metrics_df['auc_roc'].std(),
    }

    print(f"\n  MEDIA → "
          f"Acc: {summary['acc_mean']:.3f} ± {summary['acc_std']:.3f} | "
          f"F1: {summary['f1_mean']:.3f} ± {summary['f1_std']:.3f} | "
          f"AUC: {summary['auc_mean']:.3f} ± {summary['auc_std']:.3f}")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Cargando datos...")
    df = load_data()
    print(f"Dataset: {len(df)} días | "
          f"{df.index.min().date()} → {df.index.max().date()}")

    # Paso 1: Diagnóstico
    balanced = diagnostico(df)

    # Paso 2: Modelos mejorados con balanceo
    r_price = evaluate_balanced(FEATURES_PRICE,     df, 'solo precio')
    r_sent  = evaluate_balanced(FEATURES_SENTIMENT, df, 'precio + sentiment')

    # Resumen final
    print("\n" + "="*55)
    print("RESUMEN COMPARATIVO (modelo mejorado)")
    print("="*55)
    print(f"{'Métrica':<15} {'Solo precio':>18} {'+ Sentiment':>18} {'Mejora':>8}")
    print("─"*62)
    for m, label in [('acc', 'Accuracy'), ('f1', 'F1-Score'), ('auc', 'AUC-ROC')]:
        p_m  = r_price[f'{m}_mean']
        p_s  = r_price[f'{m}_std']
        s_m  = r_sent[f'{m}_mean']
        s_s  = r_sent[f'{m}_std']
        diff = s_m - p_m
        sign = '+' if diff >= 0 else ''
        print(f"{label:<15} {p_m:.3f} ± {p_s:.3f}   "
              f"{s_m:.3f} ± {s_s:.3f}   ({sign}{diff:.3f})")

    # Guardar métricas
    os.makedirs('results', exist_ok=True)
    rows = []
    for s in r_price['splits']:
        rows.append({**s, 'model': 'price_only'})
    for s in r_sent['splits']:
        rows.append({**s, 'model': 'price_sentiment'})
    pd.DataFrame(rows).to_csv('results/xgboost_mejorado.csv', index=False)

    print("\n✅ Guardado en results/xgboost_mejorado.csv")
    print("\nInterpretación para el informe:")
    auc_diff = r_sent['auc_mean'] - r_price['auc_mean']
    if auc_diff > 0:
        print(f"  → El sentiment mejora el AUC en {auc_diff:+.3f}")
        print("    Evidencia de señal predictiva en el análisis de noticias")
    else:
        print(f"  → El sentiment no mejora el AUC ({auc_diff:+.3f})")
        print("    Consistente con la hipótesis de mercado eficiente")
        print("    (resultado válido y defendible en el TFG)")