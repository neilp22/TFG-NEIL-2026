# models/xgboost_model.py
# XGBoost con walk-forward validation (5 splits)
# Para el Informe Seguimiento 1 del TFG
# Uso: python models/xgboost_model.py

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import text
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score,
                             roc_auc_score, classification_report)
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
except ImportError:
    print("Instalando xgboost...")
    os.system("pip install xgboost")
    from xgboost import XGBClassifier

from db.db_utils import get_engine

# ── Configuración ─────────────────────────────────────────────────────────────
WINDOW_TEST_DAYS = 60      # días en cada ventana de test
N_SPLITS         = 5       # número de ventanas walk-forward
RANDOM_STATE     = 42

# Features con y sin sentiment (para el análisis de ablación)
FEATURES_PRICE = [
    'returns', 'rsi_14', 'macd', 'bb_upper', 'bb_lower', 'sma_7', 'sma_30'
]
FEATURES_SENTIMENT = FEATURES_PRICE + ['sentiment_avg', 'fear_greed']

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


# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────────────────────────────────────

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


def prepare_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Prepara el dataframe: elimina NaN en features clave, rellena sentiment."""
    df = df.copy()

    # Rellenar sentiment con 0 si no hay datos (días sin noticias)
    if 'sentiment_avg' in df.columns:
        df['sentiment_avg'] = df['sentiment_avg'].fillna(0)
    if 'fear_greed' in df.columns:
        df['fear_greed'] = df['fear_greed'].fillna(df['fear_greed'].median())

    # Eliminar filas sin features de precio (primeros días sin SMA, RSI, etc.)
    price_features = [f for f in feature_cols if f in FEATURES_PRICE]
    df = df.dropna(subset=price_features)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_splits(df: pd.DataFrame, n_splits: int, test_days: int):
    """
    Genera índices de train/test para walk-forward validation.
    Cada split usa todo el histórico anterior como train.
    """
    n = len(df)
    splits = []

    for i in range(n_splits):
        test_end   = n - i * test_days
        test_start = test_end - test_days
        train_end  = test_start

        if train_end < 100:  # mínimo 100 días de train
            break

        splits.append({
            'train': df.iloc[:train_end],
            'test':  df.iloc[test_start:test_end],
            'split_num': n_splits - i,
        })

    return list(reversed(splits))


def evaluate_model(feature_cols: list, df: pd.DataFrame,
                   label: str = 'con sentiment') -> dict:
    """
    Entrena y evalúa XGBoost con walk-forward validation.
    Devuelve métricas agregadas.
    """
    print(f"\n{'─'*50}")
    print(f"Modelo {label.upper()}")
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print('─'*50)

    df_clean = prepare_features(df, feature_cols)
    splits   = walk_forward_splits(df_clean, N_SPLITS, WINDOW_TEST_DAYS)

    all_metrics = []
    all_preds   = []

    for split in splits:
        X_train = split['train'][feature_cols].values
        y_train = split['train']['label'].values.astype(int)
        X_test  = split['test'][feature_cols].values
        y_test  = split['test']['label'].values.astype(int)

        # Escalar
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        # Modelo
        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            eval_metric='logloss',
            verbosity=0,
        )
        model.fit(X_train, y_train)

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)

        metrics = {
            'split':    split['split_num'],
            'accuracy': round(acc, 4),
            'f1':       round(f1, 4),
            'auc_roc':  round(auc, 4),
            'n_train':  len(y_train),
            'n_test':   len(y_test),
            'test_start': split['test'].index[0].date(),
            'test_end':   split['test'].index[-1].date(),
        }
        all_metrics.append(metrics)
        all_preds.append((split['test'].index, y_test, y_pred, y_proba))

        print(f"  Split {split['split_num']} "
              f"[{metrics['test_start']} → {metrics['test_end']}] | "
              f"Train: {len(y_train):>4}d | "
              f"Acc: {acc:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}")

    # Métricas agregadas
    metrics_df = pd.DataFrame(all_metrics)
    summary = {
        'label':        label,
        'feature_cols': feature_cols,
        'splits':       all_metrics,
        'predictions':  all_preds,
        'acc_mean':     metrics_df['accuracy'].mean(),
        'acc_std':      metrics_df['accuracy'].std(),
        'f1_mean':      metrics_df['f1'].mean(),
        'f1_std':       metrics_df['f1'].std(),
        'auc_mean':     metrics_df['auc_roc'].mean(),
        'auc_std':      metrics_df['auc_roc'].std(),
    }

    print(f"\n  MEDIA  → Acc: {summary['acc_mean']:.3f} ± {summary['acc_std']:.3f} | "
          f"F1: {summary['f1_mean']:.3f} ± {summary['f1_std']:.3f} | "
          f"AUC: {summary['auc_mean']:.3f} ± {summary['auc_std']:.3f}")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICAS DE RESULTADOS
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(results_price: dict, results_sentiment: dict):
    """Genera gráficas comparativas para el informe."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('XGBoost — Walk-Forward Validation (5 splits × 60 días)',
                 fontsize=13, color='#e6edf3')

    metrics   = ['accuracy', 'f1', 'auc_roc']
    titles    = ['Accuracy', 'F1-Score', 'AUC-ROC']
    baselines = [0.50, 0.50, 0.50]
    colors_p  = '#58a6ff'
    colors_s  = '#f7931a'

    for ax, metric, title, baseline in zip(axes, metrics, titles, baselines):
        splits = list(range(1, N_SPLITS + 1))

        vals_p = [s[metric] for s in results_price['splits']]
        vals_s = [s[metric] for s in results_sentiment['splits']]

        x = np.arange(len(splits))
        width = 0.35

        bars_p = ax.bar(x - width/2, vals_p, width, label='Solo precio',
                        color=colors_p, alpha=0.8)
        bars_s = ax.bar(x + width/2, vals_s, width, label='Precio + Sentiment',
                        color=colors_s, alpha=0.8)

        ax.axhline(baseline, color='#f85149', linestyle='--',
                   linewidth=0.8, alpha=0.6, label='Baseline (0.50)')
        ax.axhline(0.53, color='#3fb950', linestyle=':',
                   linewidth=0.8, alpha=0.6, label='Objetivo TFG (0.53)')

        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([f'S{i}' for i in splits], fontsize=9)
        ax.set_ylim(0.3, 0.75)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=7, framealpha=0.3)

        # Valores encima de las barras
        for bar in bars_p:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{bar.get_height():.2f}', ha='center', va='bottom',
                    fontsize=7, color='#e6edf3')
        for bar in bars_s:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{bar.get_height():.2f}', ha='center', va='bottom',
                    fontsize=7, color='#e6edf3')

    plt.tight_layout()
    plt.savefig('notebooks/figures/05_xgboost_resultados.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  → Guardada: notebooks/figures/05_xgboost_resultados.png")


def plot_feature_importance(df: pd.DataFrame, feature_cols: list):
    """Importancia de features del último modelo entrenado."""
    df_clean = prepare_features(df, feature_cols)
    X = df_clean[feature_cols].values
    y = df_clean['label'].values.astype(int)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        random_state=RANDOM_STATE, eval_metric='logloss', verbosity=0,
    )
    model.fit(X, y)

    importance = pd.Series(model.feature_importances_, index=feature_cols)
    importance = importance.sort_values()

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle('Importancia de Features — XGBoost (modelo completo)',
                 fontsize=12, color='#e6edf3')

    colors = ['#f7931a' if 'sentiment' in f or 'fear' in f else '#58a6ff'
              for f in importance.index]
    bars = ax.barh(importance.index, importance.values,
                   color=colors, alpha=0.85, height=0.6)

    for bar, val in zip(bars, importance.values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9, color='#e6edf3')

    ax.set_xlabel('Feature Importance (F-score)', fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')

    from matplotlib.patches import Patch
    legend = [Patch(color='#f7931a', label='Sentiment features'),
              Patch(color='#58a6ff', label='Precio/técnicos')]
    ax.legend(handles=legend, fontsize=9, framealpha=0.3)

    plt.tight_layout()
    plt.savefig('notebooks/figures/06_feature_importance.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  → Guardada: notebooks/figures/06_feature_importance.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("="*55)
    print("XGBOOST — WALK-FORWARD VALIDATION")
    print(f"Splits: {N_SPLITS} | Test window: {WINDOW_TEST_DAYS} días")
    print("="*55)

    # Cargar datos
    print("\nCargando daily_features desde PostgreSQL...")
    df = load_data()
    print(f"Dataset: {len(df)} días | "
          f"{df.index.min().date()} → {df.index.max().date()}")
    print(f"Días alcistas: {df['label'].mean()*100:.1f}%")

    # Modelo 1: Solo precio
    results_price = evaluate_model(FEATURES_PRICE, df, label='solo precio')

    # Modelo 2: Precio + Sentiment
    results_sent = evaluate_model(FEATURES_SENTIMENT, df,
                                  label='precio + sentiment')

    # Gráficas
    print("\nGenerando gráficas...")
    plot_results(results_price, results_sent)
    plot_feature_importance(df, FEATURES_SENTIMENT)

    # Guardar métricas en CSV
    rows = []
    for split in results_price['splits']:
        rows.append({**split, 'model': 'price_only'})
    for split in results_sent['splits']:
        rows.append({**split, 'model': 'price_sentiment'})

    pd.DataFrame(rows).to_csv('results/xgboost_metrics.csv', index=False)
    print("  → Guardada: results/xgboost_metrics.csv")

    # Resumen final
    print("\n" + "="*55)
    print("RESUMEN COMPARATIVO")
    print("="*55)
    print(f"{'Métrica':<15} {'Solo precio':>15} {'+ Sentiment':>15}")
    print("─"*47)
    for m, label in [('acc', 'Accuracy'), ('f1', 'F1-Score'), ('auc', 'AUC-ROC')]:
        p_m = results_price[f'{m}_mean']
        p_s = results_price[f'{m}_std']
        s_m = results_sent[f'{m}_mean']
        s_s = results_sent[f'{m}_std']
        diff = s_m - p_m
        sign = '+' if diff >= 0 else ''
        print(f"{label:<15} {p_m:.3f} ± {p_s:.3f}   "
              f"{s_m:.3f} ± {s_s:.3f}   ({sign}{diff:.3f})")

    print("\n✅ Resultados guardados en results/xgboost_metrics.csv")
    print("✅ Gráficas guardadas en notebooks/figures/")
    print("\nEste output es directamente citable en el Informe Seguimiento 1")