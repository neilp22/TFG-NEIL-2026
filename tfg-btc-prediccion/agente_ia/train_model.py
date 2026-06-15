"""
train_model.py — Entrena modelo ensemble sobre datos intraday 1h + 4h
con features técnicas + ICT + sesión.

Uso:
  python agente_ia/train_model.py
  python agente_ia/train_model.py --target-hours 4  # horizonte de predicción
"""

import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / 'config' / '.env')

from db.db_utils import get_engine
from agente_ia.indicators import (
    calculate_indicators, calculate_all_ict,
    get_session_name, is_killzone,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

MODELS_DIR = _ROOT / 'models' / 'saved'
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ── Feature engineering ────────────────────────────────────────────────────────

def build_feature_matrix(df_1h: pd.DataFrame, df_4h: pd.DataFrame,
                         target_hours: int = 4) -> pd.DataFrame:
    """
    Combina 1h + 4h en una feature matrix lista para sklearn.
    Target: retorno > 0.5% en las próximas target_hours → 1, resto → 0.
    """
    df = df_1h.copy().sort_values('timestamp').reset_index(drop=True)

    # Calcular indicadores si no están presentes
    if 'ema_9' not in df.columns:
        df = calculate_indicators(df)
    if 'swing_high' not in df.columns:
        df = calculate_all_ict(df, swing_n=3)

    # Merge 4h features (buscar la vela 4h vigente para cada vela 1h)
    df_4h = df_4h.copy().sort_values('timestamp')
    if 'ema_9' not in df_4h.columns:
        df_4h = calculate_indicators(df_4h)
    if 'swing_high' not in df_4h.columns:
        df_4h = calculate_all_ict(df_4h, swing_n=2)

    df_4h = df_4h.rename(columns={c: f'{c}_4h' for c in df_4h.columns if c not in ('timestamp',)})
    df = pd.merge_asof(df.sort_values('timestamp'), df_4h.sort_values('timestamp'),
                       on='timestamp', direction='backward')

    # Target: retorno acumulado en las próximas target_hours velas
    future_returns = df['close'].shift(-target_hours) / df['close'] - 1
    THRESHOLD = 0.005  # 0.5%
    df['target'] = (future_returns > THRESHOLD).astype(int)

    # Feature: posición dentro de las BB (0=bajo, 1=alto)
    bb_range = (df['bb_upper'] - df['bb_lower']).replace(0, np.nan)
    df['bb_position']    = (df['close'] - df['bb_lower']) / bb_range

    # Features
    feature_cols = []

    # Precio relativo a EMAs
    for col in ['ema_9', 'ema_21', 'ema_50']:
        df[f'close_vs_{col}'] = (df['close'] - df[col]) / df[col].replace(0, np.nan)
        feature_cols.append(f'close_vs_{col}')

    vwap_col = 'vwap' if 'vwap' in df.columns else None
    if vwap_col and df[vwap_col].notna().sum() > 100:
        df['close_vs_vwap'] = (df['close'] - df[vwap_col]) / df[vwap_col].replace(0, np.nan)
        feature_cols.append('close_vs_vwap')

    feature_cols += ['bb_position', 'rsi_14', 'macd_hist', 'atr_14', 'bb_width', 'volume_ratio']

    # Momentum MACD
    df['macd_cross'] = np.where(df['macd'] > df['macd_signal'], 1, -1)
    feature_cols.append('macd_cross')

    # ATR normalizado
    df['atr_normalized'] = df['atr_14'] / df['close'].replace(0, np.nan)
    feature_cols.append('atr_normalized')

    # ICT (booleanos → int) — SHIFT para eliminar look-ahead bias
    # detect_order_blocks usa hasta +50 barras futuras → shift OB por 50
    # detect_swings/bos/fvg usan +3 barras futuras → shift por 3
    # (FIX leakage detectado en auditoría 2026-06-07)
    ICT_OB_LOOKAHEAD = 50      # detect_order_blocks mira hasta +50 barras futuras
    ICT_SWING_LOOKAHEAD = 3    # detect_swings/bos/fvg mira +3 barras

    for col in ['ob_bullish', 'ob_bearish']:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(int).shift(ICT_OB_LOOKAHEAD).fillna(0).astype(int)
            feature_cols.append(col)
    for col in ['swing_high', 'swing_low', 'bos_bullish', 'bos_bearish', 'choch',
                'fvg_bullish', 'fvg_bearish']:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(int).shift(ICT_SWING_LOOKAHEAD).fillna(0).astype(int)
            feature_cols.append(col)

    # ICT recientes (ventana móvil 3-5 velas) — calculadas DESPUÉS del shift
    for col in ['bos_bullish', 'bos_bearish', 'choch', 'fvg_bullish', 'fvg_bearish']:
        if col in df.columns:
            df[f'{col}_3'] = df[col].rolling(3).sum().fillna(0)
            df[f'{col}_5'] = df[col].rolling(5).sum().fillna(0)
            feature_cols += [f'{col}_3', f'{col}_5']

    # OB cercano (precio dentro de ±1% del OB) — sobre el ob_bullish YA shifteado
    if 'ob_bullish' in df.columns:
        df['near_ob_bull'] = df['ob_bullish'].rolling(5).sum().fillna(0)
        df['near_ob_bear'] = df['ob_bearish'].rolling(5).sum().fillna(0)
        feature_cols += ['near_ob_bull', 'near_ob_bear']

    # Sesión
    df['is_killzone_int'] = df.get('is_killzone', pd.Series(False, index=df.index)).fillna(False).astype(int)
    feature_cols.append('is_killzone_int')

    session_map = {'asia': 0, 'london': 1, 'ny': 2, 'dead_zone': 3}
    if 'session' in df.columns:
        df['session_enc'] = df['session'].map(session_map).fillna(0)
    else:
        df['session_enc'] = df['timestamp'].apply(get_session_name).map(session_map).fillna(0)
    feature_cols.append('session_enc')

    # Encoding cíclico hora y día de la semana
    df['hour']     = pd.to_datetime(df['timestamp']).dt.hour
    df['weekday']  = pd.to_datetime(df['timestamp']).dt.weekday
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin']  = np.sin(2 * np.pi * df['weekday'] / 7)
    df['day_cos']  = np.cos(2 * np.pi * df['weekday'] / 7)
    feature_cols  += ['hour_sin', 'hour_cos', 'day_sin', 'day_cos']

    # Multi-timeframe features (4h)
    for col4 in ['rsi_14_4h', 'macd_hist_4h', 'bb_position']:
        col4_full = col4 if col4 in df.columns else None
        if col4_full and df[col4_full].notna().sum() > 100:
            feature_cols.append(col4_full)

    # EMA trend 4h
    if 'ema_9_4h' in df.columns and 'ema_21_4h' in df.columns:
        df['trend_4h'] = np.where(df['ema_9_4h'] > df['ema_21_4h'], 1, -1)
        feature_cols.append('trend_4h')

    # RSI divergencia entre timeframes
    if 'rsi_14_4h' in df.columns:
        df['rsi_divergence'] = df['rsi_14'] - df['rsi_14_4h']
        feature_cols.append('rsi_divergence')

    # OB 4h cerca del precio — TAMBIÉN shifteado (4h × 50 ≈ 200h lookahead → shift por 13 barras 4h ≈ 50 barras 1h)
    if 'ob_bullish_4h' in df.columns:
        OB_4H_SHIFT = 13     # 13 barras 4h ≈ 52h, suficiente para 1h target +4h
        df['ob_bull_4h_int'] = df['ob_bullish_4h'].fillna(False).astype(int).shift(OB_4H_SHIFT).fillna(0).astype(int)
        df['ob_bear_4h_int'] = df['ob_bearish_4h'].fillna(False).astype(int).shift(OB_4H_SHIFT).fillna(0).astype(int) if 'ob_bearish_4h' in df.columns else 0
        feature_cols += ['ob_bull_4h_int', 'ob_bear_4h_int']

    # Eliminar duplicados
    feature_cols = list(dict.fromkeys(feature_cols))
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Eliminar filas con target NaN (últimas target_hours velas)
    df = df.dropna(subset=['target'])

    # Rellenar NaN en features con 0
    df[feature_cols] = df[feature_cols].fillna(0)

    result = df[['timestamp', 'close', 'target'] + feature_cols].copy()
    result.attrs['feature_cols'] = feature_cols
    return result


# ── Entrenamiento ──────────────────────────────────────────────────────────────

def train(target_hours: int = 4) -> dict:
    engine = get_engine()

    log.info("Cargando datos de btc_ohlcv...")
    from sqlalchemy import text

    sql_1h = text("SELECT * FROM btc_ohlcv WHERE timeframe='1h' ORDER BY timestamp")
    sql_4h = text("SELECT * FROM btc_ohlcv WHERE timeframe='4h' ORDER BY timestamp")

    with engine.connect() as conn:
        df_1h = pd.read_sql(sql_1h, conn)
        df_4h = pd.read_sql(sql_4h, conn)

    if df_1h.empty:
        log.error("Sin datos 1h en btc_ohlcv. Ejecutar primero python agente_ia/02_price_updater.py --ohlcv")
        return {'valid': False, 'error': 'no_data'}

    df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'], utc=True)
    df_4h['timestamp'] = pd.to_datetime(df_4h['timestamp'], utc=True)

    log.info(f"1h: {len(df_1h)} velas | 4h: {len(df_4h)} velas")

    log.info("Construyendo feature matrix...")
    df = build_feature_matrix(df_1h, df_4h, target_hours=target_hours)
    feature_cols = df.attrs['feature_cols']

    X = df[feature_cols].values
    y = df['target'].values

    log.info(f"Dataset: {len(X)} muestras | {len(feature_cols)} features | target_hours={target_hours}")
    log.info(f"Balance clases: {y.mean():.3f} (1=sube, 0=no sube)")

    # TimeSeriesSplit con GAP = target_hours para purgar leakage del target
    # (último sample de train tiene target calculado con close[+target_hours]
    #  que cae dentro del test set sin gap)
    tscv  = TimeSeriesSplit(n_splits=5, gap=target_hours)

    from sklearn.utils.class_weight import compute_sample_weight

    models_def = {
        "gbm":   GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                             learning_rate=0.05, subsample=0.8,
                                             random_state=42),
        "rf":    RandomForestClassifier(n_estimators=200, max_depth=6,
                                         min_samples_leaf=20,
                                         class_weight='balanced',     # ← fix balance (0.166 ratio)
                                         random_state=42),
        "logit": LogisticRegression(C=0.1, max_iter=1000,
                                    class_weight='balanced',          # ← fix balance
                                    random_state=42),
    }

    fold_aucs   = []
    fold_precs  = []
    fold_recalls = []
    fold_f1s    = []

    log.info(f"Validación TimeSeriesSplit (5 folds, gap={target_hours})...")
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Escalar — scaler NUEVO por fold (evita contaminación entre folds)
        scaler_fold = StandardScaler()
        X_tr_s = scaler_fold.fit_transform(X_tr)
        X_te_s = scaler_fold.transform(X_te)

        # Sample weights para GBM (que no acepta class_weight directamente)
        sw_tr = compute_sample_weight('balanced', y_tr)

        fold_preds = []
        for name, m in models_def.items():
            if name == 'gbm':
                m.fit(X_tr_s, y_tr, sample_weight=sw_tr)
            else:
                m.fit(X_tr_s, y_tr)
            fold_preds.append(m.predict_proba(X_te_s)[:, 1])

        avg_proba = np.mean(fold_preds, axis=0)
        auc = roc_auc_score(y_te, avg_proba) if len(np.unique(y_te)) > 1 else 0.5
        pred_labels = (avg_proba > 0.5).astype(int)
        prec = precision_score(y_te, pred_labels, zero_division=0)
        rec  = recall_score(y_te, pred_labels, zero_division=0)
        f1   = f1_score(y_te, pred_labels, zero_division=0)

        fold_aucs.append(auc)
        fold_precs.append(prec)
        fold_recalls.append(rec)
        fold_f1s.append(f1)
        log.info(f"  Fold {fold+1}: AUC={auc:.4f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}")

    mean_auc = float(np.mean(fold_aucs))
    std_auc  = float(np.std(fold_aucs))
    log.info(f"AUC medio: {mean_auc:.4f} ± {std_auc:.4f}")

    valid = mean_auc >= 0.55

    # Entrenar modelo final sobre todos los datos — scaler NUEVO para producción
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    sw_all = compute_sample_weight('balanced', y)
    final_models = {}
    for name, m in models_def.items():
        if name == 'gbm':
            m.fit(X_s, y, sample_weight=sw_all)
        else:
            m.fit(X_s, y)
        final_models[name] = m

    # Feature importance del GBM
    feature_importance = {
        name: float(imp)
        for name, imp in zip(feature_cols, final_models['gbm'].feature_importances_)
    }
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]

    # Guardar
    date_str = datetime.now(timezone.utc).strftime('%Y%m%d')
    model_path   = MODELS_DIR / f'btc_ensemble_v2_{date_str}.pkl'
    scaler_path  = MODELS_DIR / f'btc_scaler_v2_{date_str}.pkl'
    metrics_path = MODELS_DIR / 'metrics_v2.json'
    fi_path      = MODELS_DIR / 'feature_importance_v2.json'
    latest_path  = MODELS_DIR / 'btc_ensemble_latest.pkl'
    latest_scaler_path = MODELS_DIR / 'btc_scaler_latest.pkl'

    artifact = {
        'models': final_models,
        'feature_cols': feature_cols,
        'target_hours': target_hours,
        'valid': valid,
        'mean_auc': mean_auc,
    }
    joblib.dump(artifact, model_path)
    joblib.dump(scaler, scaler_path)

    if valid:
        joblib.dump(artifact, latest_path)
        joblib.dump(scaler, latest_scaler_path)
        log.info(f"Modelo guardado en {latest_path}")
    else:
        log.warning(f"AUC {mean_auc:.4f} < 0.55 — modelo guardado con valid=False (no se usará como latest)")

    metrics = {
        'auc_mean':      mean_auc,
        'auc_std':       std_auc,
        'auc_per_fold':  fold_aucs,
        'precision_mean': float(np.mean(fold_precs)),
        'recall_mean':    float(np.mean(fold_recalls)),
        'f1_mean':        float(np.mean(fold_f1s)),
        'n_samples':      len(X),
        'n_features':     len(feature_cols),
        'target_hours':   target_hours,
        'valid':          valid,
        'trained_at':     datetime.now(timezone.utc).isoformat(),
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    with open(fi_path, 'w') as f:
        json.dump(dict(top_features), f, indent=2)

    log.info(f"Métricas guardadas en {metrics_path}")

    return metrics


def predict_current(engine=None) -> dict:
    """
    Genera predicción usando el modelo latest sobre el estado actual del mercado.
    Devuelve dict con direction, probability, confidence, model_auc, top_features, valid.
    """
    latest_path = MODELS_DIR / 'btc_ensemble_latest.pkl'
    scaler_path = MODELS_DIR / 'btc_scaler_latest.pkl'
    metrics_path = MODELS_DIR / 'metrics_v2.json'

    if not latest_path.exists():
        return {'prediction': None, 'reason': 'model_not_trained', 'valid': False}

    artifact = joblib.load(latest_path)
    scaler   = joblib.load(scaler_path) if scaler_path.exists() else None

    if not artifact.get('valid', False):
        return {'prediction': None, 'reason': 'model_not_significant', 'valid': False}

    model_auc = artifact.get('mean_auc', 0)
    if model_auc < 0.55:
        return {'prediction': None, 'reason': 'model_not_significant',
                'model_auc': model_auc, 'valid': False}

    if engine is None:
        engine = get_engine()

    from sqlalchemy import text
    with engine.connect() as conn:
        df_1h = pd.read_sql(text("SELECT * FROM btc_ohlcv WHERE timeframe='1h' ORDER BY timestamp DESC LIMIT 300"), conn)
        df_4h = pd.read_sql(text("SELECT * FROM btc_ohlcv WHERE timeframe='4h' ORDER BY timestamp DESC LIMIT 100"), conn)

    df_1h['timestamp'] = pd.to_datetime(df_1h['timestamp'], utc=True)
    df_4h['timestamp'] = pd.to_datetime(df_4h['timestamp'], utc=True)
    df_1h = df_1h.sort_values('timestamp').reset_index(drop=True)
    df_4h = df_4h.sort_values('timestamp').reset_index(drop=True)

    df = build_feature_matrix(df_1h, df_4h, target_hours=artifact.get('target_hours', 4))
    feature_cols = artifact['feature_cols']

    # Solo la última fila
    last_row = df.tail(1)
    present_cols = [c for c in feature_cols if c in last_row.columns]
    missing_cols = [c for c in feature_cols if c not in last_row.columns]

    X = last_row[present_cols].values
    if missing_cols:
        X = np.hstack([X, np.zeros((1, len(missing_cols)))])

    if scaler:
        X = scaler.transform(X)

    probas = []
    for name, m in artifact['models'].items():
        try:
            probas.append(m.predict_proba(X)[0, 1])
        except Exception:
            pass

    if not probas:
        return {'prediction': None, 'reason': 'prediction_error', 'valid': False}

    avg_proba = float(np.mean(probas))
    confidence = "high" if avg_proba > 0.65 or avg_proba < 0.35 else \
                 "medium" if avg_proba > 0.58 or avg_proba < 0.42 else "low"
    direction = "bullish" if avg_proba > 0.5 else "bearish"

    # Feature importance
    fi_path = MODELS_DIR / 'feature_importance_v2.json'
    top_features = []
    if fi_path.exists():
        with open(fi_path) as f:
            fi = json.load(f)
        top_features = [{"name": k, "importance": v} for k, v in list(fi.items())[:5]]

    return {
        "direction":   direction,
        "probability": round(avg_proba, 4),
        "confidence":  confidence,
        "model_auc":   round(model_auc, 4),
        "top_features": top_features,
        "valid":       True,
    }


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--target-hours', type=int, default=4)
    ap.add_argument('--predict', action='store_true', help='Solo predecir, no entrenar')
    args = ap.parse_args()

    if args.predict:
        result = predict_current()
    else:
        result = train(target_hours=args.target_hours)
    print(json.dumps(result, indent=2, default=str))
