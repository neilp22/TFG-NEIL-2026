# models/arima_model.py
# ARIMA baseline con walk-forward validation (5 splits x 60 dias)
# Uso: python models/arima_model.py

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sqlalchemy import text
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    os.system("pip install statsmodels")
    from statsmodels.tsa.arima.model import ARIMA

from db.db_utils import get_engine

# ── Configuración ─────────────────────────────────────────────────────────────
WINDOW_TEST_DAYS = 60
N_SPLITS         = 5
ARIMA_ORDER      = (2, 0, 2)

plt.rcParams['figure.facecolor'] = '#0d1117'
plt.rcParams['axes.facecolor']   = '#161b22'
plt.rcParams['axes.edgecolor']   = '#30363d'
plt.rcParams['text.color']       = '#e6edf3'
plt.rcParams['axes.labelcolor']  = '#e6edf3'
plt.rcParams['xtick.color']      = '#8b949e'
plt.rcParams['ytick.color']      = '#8b949e'
plt.rcParams['grid.color']       = '#21262d'
plt.rcParams['font.family']      = 'monospace'

os.makedirs('results', exist_ok=True)
os.makedirs('notebooks/figures', exist_ok=True)

ACCENT = '#f7931a'
GREEN  = '#3fb950'
RED    = '#f85149'
BLUE   = '#58a6ff'


def load_data():
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT date, returns, label
            FROM daily_features
            WHERE asset = 'BTC'
              AND label IS NOT NULL
              AND returns IS NOT NULL
            ORDER BY date ASC
        """), conn)
    df['date'] = pd.to_datetime(df['date'])
    return df.set_index('date')


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


def evaluate_arima(df):
    """
    ARIMA walk-forward validation.
    En cada paso del test, re-entrena con todos los datos anteriores
    y predice el siguiente retorno. Si predice >0 → label=1, si no → label=0.
    """
    print(f"\n{'─'*55}")
    print("MODELO: ARIMA (2,0,2) — Baseline")
    print(f"{'─'*55}")

    splits = walk_forward_splits(df, N_SPLITS, WINDOW_TEST_DAYS)
    all_metrics = []

    for split in splits:
        train = split['train']['returns'].values
        test  = split['test']

        y_pred  = []
        y_proba = []
        y_true  = test['label'].values.astype(int)

        # Walk-forward: predecir un paso a la vez
        history = list(train)

        for i in range(len(test)):
            try:
                model = ARIMA(history, order=ARIMA_ORDER)
                fit   = model.fit()
                # Predicción del siguiente retorno
                forecast = fit.forecast(steps=1)[0]
                # Convertir a probabilidad via sigmoid suavizado
                prob = 1 / (1 + np.exp(-forecast * 100))
                pred = 1 if forecast > 0 else 0
            except Exception:
                prob = 0.5
                pred = 1  # fallback alcista

            y_pred.append(pred)
            y_proba.append(prob)
            # Añadir el retorno real al histórico
            history.append(test['returns'].iloc[i])

        acc = accuracy_score(y_true, y_pred)
        f1  = f1_score(y_true, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_true, y_proba)
        except Exception:
            auc = 0.5

        metrics = {
            'split':      split['split_num'],
            'accuracy':   round(acc, 4),
            'f1':         round(f1, 4),
            'auc_roc':    round(auc, 4),
            'n_train':    len(train),
            'n_test':     len(y_true),
            'test_start': split['test'].index[0].date(),
            'test_end':   split['test'].index[-1].date(),
            'model':      'arima',
        }
        all_metrics.append(metrics)

        print(f"  Split {split['split_num']} "
              f"[{metrics['test_start']} → {metrics['test_end']}] | "
              f"Train: {len(train):>4}d | "
              f"Acc: {acc:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}")

    mdf = pd.DataFrame(all_metrics)
    summary = {
        'model':    'ARIMA(2,0,2)',
        'splits':   all_metrics,
        'acc_mean': mdf['accuracy'].mean(),
        'acc_std':  mdf['accuracy'].std(),
        'f1_mean':  mdf['f1'].mean(),
        'f1_std':   mdf['f1'].std(),
        'auc_mean': mdf['auc_roc'].mean(),
        'auc_std':  mdf['auc_roc'].std(),
    }

    print(f"\n  MEDIA → "
          f"Acc: {summary['acc_mean']:.3f} ± {summary['acc_std']:.3f} | "
          f"F1: {summary['f1_mean']:.3f} ± {summary['f1_std']:.3f} | "
          f"AUC: {summary['auc_mean']:.3f} ± {summary['auc_std']:.3f}")

    return summary, pd.DataFrame(all_metrics)


if __name__ == '__main__':
    print("="*55)
    print("ARIMA — WALK-FORWARD VALIDATION")
    print(f"Splits: {N_SPLITS} | Test: {WINDOW_TEST_DAYS} días")
    print("Nota: puede tardar 5-10 minutos (re-entrena en cada paso)")
    print("="*55)

    df = load_data()
    print(f"\nDataset: {len(df)} días | "
          f"{df.index.min().date()} → {df.index.max().date()}")

    summary, metrics_df = evaluate_arima(df)

    metrics_df.to_csv('results/arima_metrics.csv', index=False)
    print(f"\n✅ Guardado en results/arima_metrics.csv")

    print("\n" + "="*55)
    print("RESUMEN ARIMA")
    print("="*55)
    print(f"  Accuracy: {summary['acc_mean']:.3f} ± {summary['acc_std']:.3f}")
    print(f"  F1-Score: {summary['f1_mean']:.3f} ± {summary['f1_std']:.3f}")
    print(f"  AUC-ROC:  {summary['auc_mean']:.3f} ± {summary['auc_std']:.3f}")
    print(f"\n  Interpretación:")
    if summary['auc_mean'] < 0.52:
        print("  ARIMA no supera el baseline aleatorio.")
        print("  Confirma que los retornos no tienen autocorrelación lineal")
        print("  explotable — consistente con la hipótesis de mercado eficiente.")
    else:
        print(f"  ARIMA muestra AUC > 0.52, indica autocorrelación lineal leve.")