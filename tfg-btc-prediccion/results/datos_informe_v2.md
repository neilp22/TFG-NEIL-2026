# DATOS INFORME v2 — Seguimiento 2 (Predicción BTC)

> Generado automáticamente el 2026-05-23.

## 1. Corpus de noticias

**Total textos:** 103,005 | **Fuentes:** 27 | **Timestamps fantasma (00:00):** 449

### 1.1 Distribución por fuente

| Fuente | N textos | Fecha min | Fecha max |
|--------|----------|-----------|-----------|
| `kaggle_btc_news_sentiments` | 40,820 | 2011-06-16 | 2024-03-15 |
| `news_csv_2025` | 32,220 | 2025-07-17 | 2025-11-23 |
| `kaggle_btc_2021_2024` | 10,081 | 2021-11-05 | 2024-09-12 |
| `gdelt_2025` | 9,325 | 2025-01-01 | 2025-12-31 |
| `gdelt_2024` | 7,791 | 2024-05-09 | 2025-01-01 |
| `gdelt_2026` | 2,303 | 2026-01-01 | 2026-03-03 |
| `decrypt` | 85 | 2025-12-29 | 2026-04-17 |
| `utoday` | 57 | 2026-04-12 | 2026-04-17 |
| `rss_decrypt` | 43 | 2025-12-29 | 2026-03-05 |
| `cointelegraph` | 41 | 2026-04-17 | 2026-04-17 |
| `coindesk` | 34 | 2026-04-16 | 2026-04-17 |
| `cryptopotato` | 29 | 2026-04-15 | 2026-04-17 |
| `theblock` | 20 | 2026-04-16 | 2026-04-17 |
| `cryptonews` | 20 | 2026-04-15 | 2026-04-17 |
| `rss_coindesk` | 18 | 2026-03-05 | 2026-03-05 |
| `rss_cointelegraph` | 16 | 2026-03-05 | 2026-03-05 |
| `bitcoinist` | 16 | 2026-04-17 | 2026-04-17 |
| `zycrypto` | 11 | 2026-04-16 | 2026-04-17 |
| `beincrypto` | 11 | 2026-04-17 | 2026-04-17 |
| `rss_bitcoinmagazine` | 10 | 2026-03-04 | 2026-03-05 |
| `rss_newsbtc` | 10 | 2026-03-05 | 2026-03-06 |
| `bitcoin_magazine` | 10 | 2026-04-16 | 2026-04-17 |
| `newsbtc` | 9 | 2026-04-17 | 2026-04-17 |
| `cryptoslate` | 8 | 2026-04-16 | 2026-04-17 |
| `rss_cryptoslate` | 8 | 2026-03-04 | 2026-03-05 |
| `ambcrypto` | 6 | 2026-04-17 | 2026-04-17 |
| `gdelt_2023` | 3 | 2023-06-03 | 2023-06-03 |

### 1.2 Modelos de sentimiento aplicados

| Modelo | N scores | Media | Std | Min | Max |
|--------|----------|-------|-----|-----|-----|
| `ProsusAI/finbert` | 92,921 | 0.0519 | 0.5743 | -0.9694 | 0.9476 |
| `kaggle_precomputed` | 10,081 | -0.2822 | 0.9353 | -1.0 | 0.9995 |

### 1.3 Distribución horaria de publicación (UTC)

| Hora | N | Hora | N | Hora | N |
|------|---|------|---|------|---|
| **00h** | 1,927 | **01h** | 19,303 | **02h** | 25,690 |
| **03h** | 1,326 | **04h** | 1,355 | **05h** | 1,608 |
| **06h** | 1,792 | **07h** | 1,897 | **08h** | 2,255 |
| **09h** | 2,303 | **10h** | 2,666 | **11h** | 3,003 |
| **12h** | 3,228 | **13h** | 3,229 | **14h** | 3,601 |
| **15h** | 4,070 | **16h** | 3,543 | **17h** | 3,466 |
| **18h** | 3,391 | **19h** | 2,898 | **20h** | 2,898 |
| **21h** | 2,544 | **22h** | 2,721 | **23h** | 2,291 |


## 2. Dataset (daily_features)

| Parámetro | Valor |
|-----------|-------|
| Observaciones | 2,592 |
| Período | 2019-01-30 → 2026-03-05 |
| Alcistas (label=1) | 1,324 (51.08%) |
| Bajistas (label=0) | 1,268 |
| Cobertura FinBERT | 2,511 días (96.9%) |
| Días sin cobertura (imputados 0.0) | 81 |
| sentiment\_finbert (media ± std) | 0.018 ± 0.1946 |
| fear\_greed (media) | 48.59 |
| Returns (log, media ± std) | 0.001172 ± 0.033387 |
| Volatilidad anualizada BTC | **53.0%** |
| corr(returns\_t, label\_t) | -0.0725 |
| Target leakage | **NO ✓** |


## 3. Resultados modelos — walk-forward validation

| Modelo | N splits | AUC medio | Std | IC 95% | p-val vs 0.5 | Sig? | Acc | F1 |
|--------|----------|-----------|-----|--------|--------------|------|-----|----|
| **XGB precio** | 4 | 0.5169 | 0.0331 | [0.4844, 0.5494] | 0.3781 | NO | 0.5167 | 0.4559 |
| **XGB+FinBERT** | 4 | 0.4948 | 0.0122 | [0.4828, 0.5068] | 0.4546 | NO | 0.4972 | 0.4344 |
| **XGB+Morning** | 4 | 0.5282 | 0.0417 | [0.4874, 0.5691] | 0.2706 | NO | 0.5028 | 0.4324 |
| **XGB+Optuna** | 4 | 0.5322 | 0.0495 | [0.4837, 0.5807] | 0.2865 | NO | 0.5028 | 0.5325 |
| **LightGBM** | 4 | 0.5199 | 0.0192 | [0.5011, 0.5387] | 0.1312 | NO | 0.5139 | 0.5419 |
| **LSTM** | 5 | 0.4985 | 0.0829 | [0.4258, 0.5712] | — | NO | 0.5067 | 0.3911 |
| **ARIMA** | 5 | 0.4853 | 0.0176 | [0.4699, 0.5007] | — | NO | 0.4733 | 0.5923 |

*Walk-forward: 4 splits × 90 días de test, gap=7 días. Período: 2025-03-11 → 2026-03-05.*


## 4. Tests estadísticos

### 4.1 Tests vs azar (H0: AUC = 0.5)

| Modelo | AUC | IC 95% | t-stat | p-valor | Sig? |
|--------|-----|--------|--------|---------|------|
| XGB precio | 0.5169 | [0.4844, 0.5494] | 1.0198 | 0.3829 | NO |
| XGB+FinBERT | 0.4948 | [0.4828, 0.5068] | -0.8516 | 0.4570 | NO |
| XGB+Morning | 0.5282 | [0.4874, 0.5691] | 1.3541 | 0.2687 | NO |
| XGB+Optuna | 0.5322 | [0.4837, 0.5807] | 1.3016 | 0.2840 | NO |
| LightGBM | 0.5199 | [0.5011, 0.5387] | 2.0731 | 0.1299 | NO |
| LSTM | 0.4985 | [0.4258, 0.5712] | -0.0394 | 0.9705 | NO |
| ARIMA | 0.4853 | [0.4699, 0.5007] | -1.8681 | 0.1351 | NO |

*Con n=4 splits el poder estadístico es ≈20% para ΔAUC=0.05. p>0.05 es esperado incluso con mejora real.*

### 4.2 Comparaciones pareadas vs XGB precio (t-test pareado)

| Comparación | ΔAUC | t-stat | p-valor | Sig? |
|-------------|------|--------|---------|------|
| XGB+FinBERT vs XGB precio | -0.0221 | -1.1588 | 0.3304 | NO |
| XGB+Morning vs XGB precio | +0.0113 | 0.3175 | 0.7717 | NO |
| XGB+Optuna vs XGB precio | +0.0153 | 0.4941 | 0.6552 | NO |
| LightGBM vs XGB precio | +0.0030 | 0.1945 | 0.8582 | NO |

**XGB+Morning vs XGB+FinBERT completo:** ΔAUC=+0.0334, splits donde Morning > Full=3/4, p=0.2029 → No significativo

### 4.3 Mann-Whitney: sentimiento en alcistas vs bajistas

| Grupo | N | Media sentiment\_finbert |
|-------|---|--------------------------|
| Alcistas (label=1) | 1,324 | 0.0147 |
| Bajistas (label=0) | 1,268 | 0.0214 |

**U=821,316.5, p=0.342** → No significativo (esperado — sentimiento no discrimina clases)


## 5. Backtest — XGBoost+Optuna (con costes de transacción 0.1%)

**Período:** 2025-03-11 → 2026-03-05 | **N días:** 360

### 5.1 Resultados principales (threshold=0.50)

| Métrica | Estrategia ML | Buy & Hold |
|---------|--------------|------------|
| Retorno total | **-13.26%** | -17.83% |
| Sharpe Ratio | **-0.166** | -0.187 |
| Max Drawdown | -48.71% | -51.71% |
| Capital final (desde 10.000) | 8,674.33 | 8,216.86 |
| Días en mercado | 56.9% (205 días) | 100% |
| N operaciones (buy+sell) | 69 | — |

### 5.2 Sensibilidad al threshold

| Threshold | % en mercado | Accuracy % | Retorno % | Sharpe | Max DD % |
|-----------|-------------|------------|-----------|--------|----------|
| 0.5 | 56.9 | — | -13.26 | -0.166 | -48.71 |
| 0.55 | 16.7 | — | -10.46 | -0.216 | -32.33 |
| 0.6 | 8.3 | — | -19.53 | -0.734 | -25.60 |
| 0.65 | 6.1 | — | -24.01 | -1.199 | -26.14 |
| 0.7 | 2.2 | — | -17.81 | -0.967 | -19.75 |

## 6. Lag analysis — correlación sentimiento vs label

*(N=2225 días de train — solo primer fold para evitar data snooping)*

| Lag (días) | Correlación |
|------------|-------------|
| 0 | -0.0155 |
| 1 | -0.0115 |
| 2 | -0.0019 |
| 3 | -0.0130 |
| 4 | 0.0132 |
| 5 | 0.0224 |
| 6 | 0.0006 |
| 7 | -0.0155 |
| 8 | 0.0439 |
| 9 | -0.0163 |
| 10 | -0.0067 |

**Mejor lag:** 8 días (r=0.0439)


## 7. Correlación de features con target

| Rank | Feature | Correlación |
|------|---------|-------------|
| 1 | `returns` | -0.0725 |
| 2 | `bb_lower` | -0.0303 |
| 3 | `sma_7` | -0.0296 |
| 4 | `sma_30` | -0.0295 |
| 5 | `bb_upper` | -0.0286 |
| 6 | `fear_greed` | +0.0246 |
| 7 | `sentiment_morning` | -0.0185 |
| 8 | `sentiment_finbert` | -0.0172 |
| 9 | `macd_signal` | +0.0165 |
| 10 | `macd` | +0.0063 |
| 11 | `rsi_14` | -0.0042 |
| 12 | `has_sentiment` | -0.0028 |

## 8. Resumen ejecutivo — Datos clave para el informe

### Corpus
- **103,005 textos** en 27 fuentes (2011–2026)
- FinBERT (ProsusAI): **92,921 scores**, media=0.0519 (distribución continua [-1, +1])
- Kaggle precomputado: **10,081 scores**, media=-0.2822 (trimodal {-1, 0, +1})
- Timestamps fantasma (00:00:00 UTC): 449 — tratados como fecha-solamente

### Dataset
- **2,592 observaciones** diarias (2019-01-30 → 2026-03-05)
- Balance clases: **51.08% alcistas** — casi balanceado
- Cobertura FinBERT: **96.9%** (81 días imputados a 0.0)
- Volatilidad anualizada BTC: **53.0%**
- corr(returns\_t, label\_t) = **-0.0725** → sin target leakage ✓

### Resultados modelos
- **Mejor modelo:** XGB+Optuna AUC=0.5322 ± 0.0495
- **FinBERT completo vs solo precio:** ΔAUC=-0.0221 (sentimiento reactivo — noticias del día son posteriores al movimiento)
- **Morning (<18h) vs FinBERT completo:** ΔAUC=+0.0334 → noticias tardías son reactivas (causalidad inversa)
- **Ningún modelo supera el azar estadísticamente** (todos p > 0.05) → consistente con EMH
- Feature importance sentiment\_avg = **0.000** → XGBoost descarta el sentimiento como feature

### Tests estadísticos
- Mann-Whitney (alcistas vs bajistas): U=821,316, **p=0.342** → no sig.
- Media sentimiento alcistas=0.0147, bajistas=0.0214 → distribuciones prácticamente idénticas
- Morning vs Full FinBERT: ΔAUC=+0.0334, p=0.2029 → no sig.
- Poder estadístico con n=4 splits ≈ 18-20% para ΔAUC=0.05 → ausencia de sig. no descarta mejora real

### Backtest
- Período: 2025-03-11 → 2026-03-05 (360 días)
- Estrategia ML @ 0.50: retorno=-13.26%, Sharpe=-0.166, MaxDD=-48.7%
- Buy & Hold: retorno=-17.83%, Sharpe=-0.187, MaxDD=-51.7%
- Todos los thresholds generan retornos negativos → el modelo no supera el mercado
- Mejor Sharpe: threshold=0.50 (-0.166) con 56.9% exposición

### Figuras generadas
| Archivo | Descripción |
|---------|-------------|
| `results/figures/fig6_equity_real.pdf` | Curva de equity estrategia ML vs Buy & Hold |
| `results/figures/fig8_drawdown.pdf` | Drawdown comparado |
| `results/figures/fig9_threshold_sensitivity.pdf` | Sharpe/retorno/exposición por threshold |
| `results/figures/fig10_corr_features.pdf` | Correlación de features con label |
| `results/figures/fig11_lag_analysis.pdf` | Lag analysis del sentimiento |