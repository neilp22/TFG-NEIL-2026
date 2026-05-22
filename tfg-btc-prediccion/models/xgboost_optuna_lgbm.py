"""
models/xgboost_optuna_lgbm.py
===============================
XGBoost con búsqueda bayesiana de hiperparámetros (Optuna) y LightGBM,
ambos con features extendidas y walk-forward validation (5 splits × 60 días).

Features nuevas respecto al baseline:
  - vol_7d       : volatilidad realizada 7 días (std de log-returns)
  - sentiment_ma3: media móvil 3 días del sentimiento FinBERT
  - bb_width     : ancho relativo de las Bandas de Bollinger
  - rsi_change   : variación diaria del RSI
  - vol_rel      : volumen relativo vs media 30d (JOIN con price_data)

Guarda en results/:
  - xgboost_optuna_lgbm_metrics.csv  (mismo formato que xgboost_metrics.csv)
  - ml_predictions.csv               (predicciones del mejor modelo, para backtest)
  - xgboost_optuna_predictions.csv
  - lgbm_predictions.csv

Uso:
  cd tfg-btc-prediccion
  python models/xgboost_optuna_lgbm.py

Neil Pradas · TFG UAB 2025/26
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sqlalchemy import text
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("Instalando optuna...")
    os.system("pip install optuna")
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    from xgboost import XGBClassifier
except ImportError:
    os.system("pip install xgboost")
    from xgboost import XGBClassifier

try:
    import lightgbm as lgb
except ImportError:
    print("Instalando lightgbm...")
    os.system("pip install lightgbm")
    import lightgbm as lgb

from db.db_utils import get_engine
from data_loader import FEATURES_PRICE, FEATURES_SENTIMENT

os.makedirs('results', exist_ok=True)

# ── Configuración ──────────────────────────────────────────────────────────────
WINDOW_TEST_DAYS = 90
N_SPLITS         = 4
GAP_DAYS         = 7       # embargo temporal entre train y test
RANDOM_STATE     = 42
OPTUNA_TRIALS    = 30   # trials por split (~2-4 min total)

# FEATURES_PRICE / FEATURES_SENTIMENT importados desde data_loader (fuente única de verdad).
# FEATURES_EXTENDED añade features engineered sobre FEATURES_SENTIMENT:
#   vol_7d, sentiment_ma3, bb_width, rsi_change, vol_rel requieren load_data_extended().
FEATURES_EXTENDED = FEATURES_SENTIMENT + [
    'vol_7d',
    'sentiment_ma3',
    'bb_width',
    'rsi_change',
    'vol_rel',
]


# ── Carga de datos con features extendidas ─────────────────────────────────────

def load_data_extended() -> pd.DataFrame:
    """
    Carga daily_features + JOIN con price_data para volumen.
    Calcula las features nuevas en Python sobre los datos cargados.
    """
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT
                df.date,
                df.close, df.returns, df.label,
                df.rsi_14, df.macd, df.macd_signal,
                df.bb_upper, df.bb_lower, df.sma_7, df.sma_30,
                df.sentiment_finbert, df.has_sentiment, df.fear_greed,
                p.volume
            FROM daily_features df
            LEFT JOIN (
                SELECT timestamp::date AS date, asset, volume
                FROM price_data
                WHERE asset = 'BTC' AND timeframe = '1d'
            ) p ON p.date = df.date AND p.asset = df.asset
            WHERE df.asset = 'BTC'
              AND df.label IS NOT NULL
            ORDER BY df.date ASC
        """), conn)

    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    # Registrar cobertura antes de imputar
    df['has_sentiment']    = df['sentiment_finbert'].notna().astype(int)
    df['sentiment_finbert'] = df['sentiment_finbert'].fillna(0.0)
    # fear_greed: NO imputar aquí — la mediana se calcula por fold en el loop
    # de splits para evitar leakage estadístico del test hacia el train.

    n_missing = (df['has_sentiment'] == 0).sum()
    print(f"Días sin cobertura FinBERT imputados a 0.0: "
          f"{n_missing} ({n_missing / len(df) * 100:.1f}%)")

    # Eliminar filas sin indicadores técnicos (primeras N filas)
    df = df.dropna(subset=['rsi_14', 'macd', 'sma_30'])

    # ── Features nuevas ──────────────────────────────────────────────────────
    df['vol_7d']        = df['returns'].rolling(7).std()
    df['sentiment_ma3'] = df['sentiment_finbert'].rolling(3).mean()
    df['bb_width']      = (df['bb_upper'] - df['bb_lower']) / df['close'].replace(0, np.nan)
    df['rsi_change']    = df['rsi_14'].diff().fillna(0)

    if 'volume' in df.columns and df['volume'].notna().sum() > 50:
        vol_ma30       = df['volume'].rolling(30).mean()
        df['vol_rel']  = df['volume'] / vol_ma30.replace(0, np.nan)
        df['vol_rel']  = df['vol_rel'].fillna(1.0)
    else:
        df['vol_rel'] = df['bb_width']  # proxy: volatilidad implícita

    # Eliminar filas sin vol_7d (necesita 7 retornos previos)
    df = df.dropna(subset=['vol_7d', 'bb_width'])

    # Quitar última fila: su label apunta al día siguiente (fuera del dataset)
    df = df.iloc[:-1]

    return df


# ── Walk-forward splits ────────────────────────────────────────────────────────

def walk_forward_splits(df: pd.DataFrame, n_splits: int, test_days: int, gap: int = 7) -> list:
    n = len(df)
    min_train = n - n_splits * test_days
    splits = []

    for i in range(n_splits):
        test_start = min_train + i * test_days
        test_end   = test_start + test_days
        train_end  = test_start - gap      # embargo: gap días excluidos

        if test_end > n or train_end <= 0:
            break
        splits.append({
            'train':     df.iloc[:train_end],
            'test':      df.iloc[test_start:test_end],
            'split_num': i + 1,
        })
    return splits


# ── Optuna: optimización de hiperparámetros XGBoost ───────────────────────────

def _optuna_objective(trial, X_train, y_train):
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 100, 500),
        'max_depth':         trial.suggest_int('max_depth', 3, 8),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight':  trial.suggest_int('min_child_weight', 1, 10),
        'gamma':             trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha':         trial.suggest_float('reg_alpha', 0.0, 2.0),
        'reg_lambda':        trial.suggest_float('reg_lambda', 0.5, 5.0),
    }
    # Validación temporal: 70% train, 30% val (sin data leakage)
    split_idx = int(len(X_train) * 0.7)
    X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
    y_tr, y_val = y_train[:split_idx], y_train[split_idx:]

    if len(np.unique(y_val)) < 2:
        return 0.5  # devolver baseline si solo hay una clase

    model = XGBClassifier(
        **params,
        random_state=RANDOM_STATE,
        eval_metric='logloss',
        verbosity=0,
    )
    model.fit(X_tr, y_tr)
    return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])


def optimize_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                     n_trials: int = OPTUNA_TRIALS) -> dict:
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(
        lambda trial: _optuna_objective(trial, X_train, y_train),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    return study.best_params


# ── Evaluación: XGBoost + Optuna ──────────────────────────────────────────────

def evaluate_xgboost_optuna(df: pd.DataFrame,
                             feature_cols: list) -> dict:
    print(f"\n{'─'*60}")
    print(f"XGBoost + Optuna  ({OPTUNA_TRIALS} trials/split)")
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print('─'*60)

    splits      = walk_forward_splits(df, N_SPLITS, WINDOW_TEST_DAYS, GAP_DAYS)
    all_metrics = []
    all_preds   = []

    for split in splits:
        train_df = split['train'].copy()
        test_df  = split['test'].copy()

        if 'fear_greed' in feature_cols:
            fg_median = train_df['fear_greed'].median()
            train_df['fear_greed'] = train_df['fear_greed'].fillna(fg_median)
            test_df['fear_greed']  = test_df['fear_greed'].fillna(fg_median)

        X_train = train_df[feature_cols].values
        y_train = train_df['label'].values.astype(int)
        X_test  = test_df[feature_cols].values
        y_test  = test_df['label'].values.astype(int)

        scaler     = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        print(f"  Split {split['split_num']}: buscando hiperparámetros...", end=' ', flush=True)
        best_params = optimize_xgboost(X_train_sc, y_train)

        model = XGBClassifier(
            **best_params,
            random_state=RANDOM_STATE,
            eval_metric='logloss',
            verbosity=0,
        )
        model.fit(X_train_sc, y_train)

        y_pred  = model.predict(X_test_sc)
        y_proba = model.predict_proba(X_test_sc)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)

        test_start = split['test'].index[0].date()
        test_end   = split['test'].index[-1].date()

        print(f"Acc: {acc:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}  "
              f"[lr={best_params['learning_rate']:.3f}, depth={best_params['max_depth']}]")

        all_metrics.append({
            'split':      split['split_num'],
            'accuracy':   round(acc, 4),
            'f1':         round(f1, 4),
            'auc_roc':    round(auc, 4),
            'n_train':    len(y_train),
            'n_test':     len(y_test),
            'test_start': test_start,
            'test_end':   test_end,
            'model':      'xgboost_optuna',
        })

        pred_df = split['test'][['close', 'returns', 'rsi_14', 'label']].copy()
        pred_df['prob_up'] = y_proba
        pred_df['pred']    = y_pred
        all_preds.append(pred_df)

    metrics_df = pd.DataFrame(all_metrics)
    summary = {
        'label':       'xgboost_optuna',
        'splits':      all_metrics,
        'predictions': pd.concat(all_preds),
        'acc_mean':    metrics_df['accuracy'].mean(),
        'acc_std':     metrics_df['accuracy'].std(),
        'f1_mean':     metrics_df['f1'].mean(),
        'f1_std':      metrics_df['f1'].std(),
        'auc_mean':    metrics_df['auc_roc'].mean(),
        'auc_std':     metrics_df['auc_roc'].std(),
    }
    print(f"\n  MEDIA → Acc: {summary['acc_mean']:.3f} ± {summary['acc_std']:.3f} | "
          f"F1: {summary['f1_mean']:.3f} ± {summary['f1_std']:.3f} | "
          f"AUC: {summary['auc_mean']:.3f} ± {summary['auc_std']:.3f}")
    return summary


# ── Evaluación: LightGBM ──────────────────────────────────────────────────────

def evaluate_lightgbm(df: pd.DataFrame,
                      feature_cols: list) -> dict:
    print(f"\n{'─'*60}")
    print("LightGBM (hiperparámetros por defecto robustos)")
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print('─'*60)

    splits      = walk_forward_splits(df, N_SPLITS, WINDOW_TEST_DAYS, GAP_DAYS)
    all_metrics = []
    all_preds   = []

    for split in splits:
        train_df = split['train'].copy()
        test_df  = split['test'].copy()

        if 'fear_greed' in feature_cols:
            fg_median = train_df['fear_greed'].median()
            train_df['fear_greed'] = train_df['fear_greed'].fillna(fg_median)
            test_df['fear_greed']  = test_df['fear_greed'].fillna(fg_median)

        X_train = train_df[feature_cols].values
        y_train = train_df['label'].values.astype(int)
        X_test  = test_df[feature_cols].values
        y_test  = test_df['label'].values.astype(int)

        scaler     = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=10,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            verbosity=-1,
            force_row_wise=True,
        )
        model.fit(X_train_sc, y_train)

        y_pred  = model.predict(X_test_sc)
        y_proba = model.predict_proba(X_test_sc)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)

        test_start = split['test'].index[0].date()
        test_end   = split['test'].index[-1].date()

        print(f"  Split {split['split_num']} "
              f"[{test_start} → {test_end}] | "
              f"Acc: {acc:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}")

        all_metrics.append({
            'split':      split['split_num'],
            'accuracy':   round(acc, 4),
            'f1':         round(f1, 4),
            'auc_roc':    round(auc, 4),
            'n_train':    len(y_train),
            'n_test':     len(y_test),
            'test_start': test_start,
            'test_end':   test_end,
            'model':      'lightgbm',
        })

        pred_df = split['test'][['close', 'returns', 'rsi_14', 'label']].copy()
        pred_df['prob_up'] = y_proba
        pred_df['pred']    = y_pred
        all_preds.append(pred_df)

    metrics_df = pd.DataFrame(all_metrics)
    summary = {
        'label':       'lightgbm',
        'splits':      all_metrics,
        'predictions': pd.concat(all_preds),
        'acc_mean':    metrics_df['accuracy'].mean(),
        'acc_std':     metrics_df['accuracy'].std(),
        'f1_mean':     metrics_df['f1'].mean(),
        'f1_std':      metrics_df['f1'].std(),
        'auc_mean':    metrics_df['auc_roc'].mean(),
        'auc_std':     metrics_df['auc_roc'].std(),
    }
    print(f"\n  MEDIA → Acc: {summary['acc_mean']:.3f} ± {summary['acc_std']:.3f} | "
          f"F1: {summary['f1_mean']:.3f} ± {summary['f1_std']:.3f} | "
          f"AUC: {summary['auc_mean']:.3f} ± {summary['auc_std']:.3f}")
    return summary


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 62)
    print("MODELOS MEJORADOS: XGBoost + Optuna  &  LightGBM")
    print(f"Splits: {N_SPLITS} × {WINDOW_TEST_DAYS} días  |  "
          f"Optuna trials: {OPTUNA_TRIALS}/split")
    print("=" * 62)

    print("\nCargando datos con features extendidas desde PostgreSQL...")
    df = load_data_extended()
    print(f"Dataset: {len(df)} días  |  "
          f"{df.index.min().date()} → {df.index.max().date()}")
    print(f"Días alcistas: {df['label'].mean()*100:.1f}%")

    # Solo usar features que existen en el DataFrame
    feat_cols = [f for f in FEATURES_EXTENDED if f in df.columns]
    missing   = [f for f in FEATURES_EXTENDED if f not in df.columns]
    if missing:
        print(f"  ⚠ Features no disponibles (se omiten): {missing}")

    # Detectar duplicación silenciosa: cuando no hay volumen real, vol_rel == bb_width.
    # Incluir ambas infla el número de features y distorsiona la importancia.
    if 'vol_rel' in feat_cols and 'bb_width' in feat_cols:
        if df['vol_rel'].equals(df['bb_width']):
            feat_cols = [f for f in feat_cols if f != 'vol_rel']
            print("WARNING: vol_rel == bb_width (sin datos de volumen real). "
                  "vol_rel eliminada de features para evitar duplicación.")

    print(f"Features activas ({len(feat_cols)}): {feat_cols}")

    # ── 1. XGBoost + Optuna ────────────────────────────────────────────
    results_xgb = evaluate_xgboost_optuna(df, feat_cols)

    # ── 2. LightGBM ───────────────────────────────────────────────────
    results_lgb = evaluate_lightgbm(df, feat_cols)

    # ── Guardar métricas ───────────────────────────────────────────────
    rows = results_xgb['splits'] + results_lgb['splits']
    pd.DataFrame(rows).to_csv('results/xgboost_optuna_lgbm_metrics.csv', index=False)
    print("\n✅ Métricas → results/xgboost_optuna_lgbm_metrics.csv")

    # ── Guardar predicciones ───────────────────────────────────────────
    results_xgb['predictions'].to_csv('results/xgboost_optuna_predictions.csv')
    results_lgb['predictions'].to_csv('results/lgbm_predictions.csv')

    # El mejor modelo en AUC es el que se usa para el backtest de umbrales
    best = results_xgb if results_xgb['auc_mean'] >= results_lgb['auc_mean'] else results_lgb
    best['predictions'].to_csv('results/ml_predictions.csv')
    print(f"✅ Predicciones del mejor modelo ({best['label']}) → results/ml_predictions.csv")
    print("   (estas predicciones son la entrada del backtest con umbrales)")

    # ── Tabla comparativa final ────────────────────────────────────────
    print("\n" + "=" * 68)
    print("RESUMEN COMPARATIVO")
    print("=" * 68)
    print(f"{'Modelo':<32} {'Accuracy':>14} {'F1-Score':>12} {'AUC-ROC':>12}")
    print("─" * 68)

    for s in [results_xgb, results_lgb]:
        print(f"  {s['label']:<30} "
              f"{s['acc_mean']:.3f}±{s['acc_std']:.3f}  "
              f"{s['f1_mean']:.3f}±{s['f1_std']:.3f}  "
              f"{s['auc_mean']:.3f}±{s['auc_std']:.3f}")

    print(f"  {'XGBoost baseline (precio+sent)':<30} "
          f"{'0.477±0.063':>14}  {'0.470±0.166':>12}  {'0.528±0.077':>12}  ← anterior")
    print("─" * 68)

    delta_xgb = results_xgb['auc_mean'] - 0.528
    delta_lgb = results_lgb['auc_mean'] - 0.528
    sign_x = '+' if delta_xgb >= 0 else ''
    sign_l = '+' if delta_lgb >= 0 else ''
    print(f"  Δ AUC vs baseline:  XGBoost+Optuna {sign_x}{delta_xgb:.3f}  |  "
          f"LightGBM {sign_l}{delta_lgb:.3f}")

    print("\n✅ Script completado. Siguiente paso:")
    print("   python pipeline/trading_strategy_backtest.py --ml-threshold")
