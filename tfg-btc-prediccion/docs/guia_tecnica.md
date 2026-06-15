# TFG — Predicción de Bitcoin con ML y Sentimiento (UAB 2025/26)

## Objetivo
Predecir la dirección diaria del precio de Bitcoin (sube/baja, clasificación binaria) combinando indicadores técnicos con sentimiento FinBERT. Evaluación con walk-forward validation (5 splits × 60 días de test).

## Regla principal
**No tocar la DB ni el pipeline de ingesta de datos.** Solo trabajar en modelos (`models/`), backtest (`pipeline/trading_strategy_backtest.py`) y análisis (`analysis/`).

---

## Stack
- **Python 3.11+**
- **PostgreSQL local** — DB: `btc_tfg`, user: `tfg_user`, pw: `tfg_password_2026`
- Conexión: `db/db_utils.py` → `get_engine()`
- **Librerías principales:** xgboost, lightgbm, scikit-learn, pytorch (LSTM), statsmodels (ARIMA), optuna, pandas, sqlalchemy

---

## Esquema de base de datos

### `daily_features` (tabla maestra)
| Columna | Descripción |
|---------|-------------|
| date, asset | índice temporal + activo (ej. 'BTC') |
| close, returns | precio cierre y retorno diario |
| label | 1 si subió, 0 si bajó (target) |
| rsi_14, macd, macd_signal | indicadores técnicos |
| bb_upper, bb_lower | Bollinger Bands |
| sma_7, sma_30 | medias móviles |
| sentiment_avg, sentiment_std, sentiment_count | agregado FinBERT diario |
| fear_greed | índice Fear & Greed (0-100) |

### `price_data`
Contiene OHLCV incluyendo `volume`. El campo `volume` **no está** en `daily_features` — hacer JOIN con `price_data` si se necesita.

---

## Features estándar (`models/data_loader.py`)
```python
FEATURES = [
    'rsi_14', 'macd', 'macd_signal',
    'bb_upper', 'bb_lower', 'sma_7', 'sma_30',
    'fear_greed', 'returns', 'sentiment_avg'
]
TARGET = 'label'
```
`sentiment_avg` puede tener NULLs — se imputa con 0.0 (neutro).

---

## Resultados actuales (walk-forward 5 splits × 60 días)

| Modelo | AUC medio | Accuracy | F1 |
|--------|-----------|----------|----|
| XGBoost precio+sentiment | **0.528 ± 0.077** | 0.477 | 0.470 |
| XGBoost solo precio | 0.517 | 0.467 | 0.467 |
| XGBoost+Optuna | ~0.513 | ~0.507 | ~0.559 |
| LightGBM | ~0.490 | ~0.480 | ~0.472 |
| Random Forest | ~0.520 | 0.467 | 0.470 |
| LSTM | 0.499 | 0.507 | 0.381 (inestable) |
| ARIMA | 0.487 | 0.477 | 0.573 |

---

## Estructura de archivos clave

```
tfg-btc-prediccion/
├── db/
│   ├── db_utils.py              # get_engine() — conexión PostgreSQL
│   └── create_tables.py
├── models/
│   ├── data_loader.py           # FEATURES, load_dataset()
│   ├── xgboost_model.py         # modelo principal (precio+sentiment)
│   ├── xgboost_optuna_lgbm.py   # XGBoost+Optuna y LightGBM con features extendidas
│   ├── tree_model.py            # Random Forest (⚠️ ver bugs)
│   ├── lstm_model.py            # LSTM (PyTorch)
│   ├── arima_model.py           # ARIMA baseline
│   ├── comparativa_modelos.py   # genera tabla comparativa
│   └── saved/
│       └── lstm_best.pt         # mejor checkpoint LSTM
├── pipeline/
│   ├── trading_strategy_backtest.py  # backtest con umbrales ML
│   ├── feature_builder.py
│   └── sentiment_processor.py
├── analysis/
│   └── coverage_analysis.py
├── results/
│   ├── comparativa_final.csv
│   ├── xgboost_optuna_lgbm_metrics.csv
│   ├── ml_predictions.csv
│   └── backtest_threshold_analysis.csv
└── notebooks/
    └── 02_news_sentiment_eda.ipynb
```

---

## Features extendidas (xgboost_optuna_lgbm.py)
Además de las estándar: `vol_7d`, `sentiment_ma3`, `bb_width`, `rsi_change`, `vol_rel` (requiere JOIN con `price_data`).

## Backtest (trading_strategy_backtest.py)
- Umbrales ML: 0.55 / 0.60 / 0.65 / 0.70
- Filtro RSI: solo opera si RSI ∈ [35, 65]
- Ejecución: `python pipeline/trading_strategy_backtest.py --start 2022-01-01 --end 2024-01-01`

---

## Bugs conocidos
- **`tree_model.py`** — código suelto en líneas 85-120 fuera del bloque `__main__`: se ejecuta al importar el módulo.
- **`trading_strategy_backtest.py`** — originalmente cargaba `sentiment_7d` y `sentiment_momentum` del SQL (no existen en el schema). Corregido: se calculan desde `sentiment_avg` con rolling window.
