# DATOS INFORME — Seguimiento 2 (Predicción BTC con ML y Sentimiento)

> Extraído automáticamente el 2026-05-23. No modificar manualmente.


## 1. Corpus de noticias

**Total textos indexados:** 103,005

### 1.1 Distribución por fuente

| Fuente | N textos | Fecha min | Fecha max |
|--------|----------|-----------|-----------|
| kaggle_btc_news_sentiments | 40,820 | 2011-06-16 | 2024-03-15 |
| news_csv_2025 | 32,220 | 2025-07-17 | 2025-11-23 |
| kaggle_btc_2021_2024 | 10,081 | 2021-11-05 | 2024-09-12 |
| gdelt_2025 | 9,325 | 2025-01-01 | 2025-12-31 |
| gdelt_2024 | 7,791 | 2024-05-09 | 2025-01-01 |
| gdelt_2026 | 2,303 | 2026-01-01 | 2026-03-03 |
| decrypt | 85 | 2025-12-29 | 2026-04-17 |
| utoday | 57 | 2026-04-12 | 2026-04-17 |
| rss_decrypt | 43 | 2025-12-29 | 2026-03-05 |
| cointelegraph | 41 | 2026-04-17 | 2026-04-17 |
| coindesk | 34 | 2026-04-16 | 2026-04-17 |
| cryptopotato | 29 | 2026-04-15 | 2026-04-17 |
| theblock | 20 | 2026-04-16 | 2026-04-17 |
| cryptonews | 20 | 2026-04-15 | 2026-04-17 |
| rss_coindesk | 18 | 2026-03-05 | 2026-03-05 |
| rss_cointelegraph | 16 | 2026-03-05 | 2026-03-05 |
| bitcoinist | 16 | 2026-04-17 | 2026-04-17 |
| zycrypto | 11 | 2026-04-16 | 2026-04-17 |
| beincrypto | 11 | 2026-04-17 | 2026-04-17 |
| rss_bitcoinmagazine | 10 | 2026-03-04 | 2026-03-05 |
| rss_newsbtc | 10 | 2026-03-05 | 2026-03-06 |
| bitcoin_magazine | 10 | 2026-04-16 | 2026-04-17 |
| newsbtc | 9 | 2026-04-17 | 2026-04-17 |
| cryptoslate | 8 | 2026-04-16 | 2026-04-17 |
| rss_cryptoslate | 8 | 2026-03-04 | 2026-03-05 |
| ambcrypto | 6 | 2026-04-17 | 2026-04-17 |
| gdelt_2023 | 3 | 2023-06-03 | 2023-06-03 |

### 1.2 Modelos de sentimiento

| Modelo | N | Media score | Std | Min | Max |
|--------|---|-------------|-----|-----|-----|
| ProsusAI/finbert | 92,921 | 0.0519 | 0.5743 | -0.9694 | 0.9476 |
| kaggle_precomputed | 10,081 | -0.2822 | 0.9353 | -1.0 | 0.9995 |

### 1.3 Cobertura temporal

- Días con noticias en raw\_texts: **4,106**
- Días en daily\_features: **2,622**
- Cobertura: **156.6%** (>100% porque raw\_texts cubre fuera del rango de precio)
- Timestamps fantasma (exactamente 00:00:00 UTC): **304**

### 1.4 Distribución horaria (justificación corte 18:00 UTC)

| Hora UTC | N textos |
|----------|----------|
| 00h | 1,927 |
| 01h | 19,303 |
| 02h | 25,690 |
| 03h | 1,326 |
| 04h | 1,355 |
| 05h | 1,608 |
| 06h | 1,792 |
| 07h | 1,897 |
| 08h | 2,255 |
| 09h | 2,303 |
| 10h | 2,666 |
| 11h | 3,003 |
| 12h | 3,228 |
| 13h | 3,229 |
| 14h | 3,601 |
| 15h | 4,070 |
| 16h | 3,543 |
| 17h | 3,466 |
| 18h | 3,391 |
| 19h | 2,898 |
| 20h | 2,898 |
| 21h | 2,544 |
| 22h | 2,721 |
| 23h | 2,291 |


## 2. Dataset final (daily_features)

| Parámetro | Valor |
|-----------|-------|
| Observaciones totales | 2,592 |
| Período | 2019-01-30 → 2026-03-05 |
| Días alcistas (label=1) | 1,324 (51.08%) |
| Días bajistas (label=0) | 1,268 |
| Días con cobertura FinBERT | 2,511 (96.9%) |
| Días sin cobertura (imputados a 0.0) | 81 |

### 2.1 Estadísticas de features clave

| Feature | Media | Std |
|---------|-------|-----|
| sentiment\_finbert | 0.018 | 0.1946 |
| fear\_greed | 48.59 | 21.93 |
| returns (log) | 0.001172 | 0.033387 |
| Volatilidad anualizada BTC | 53.0% | — |

### 2.2 Validación de target leakage

- corr(returns\_t, label\_t) = **-0.0725** ← debe ser bajo (sin leakage si < 0.95)
- corr(returns\_t+1, label\_t) = **0.6562** ← correlación interna sign(x)~x, esperada
- Leakage detectado: **NO ✓**


## 3. Resultados modelos (walk-forward validation)

| Modelo | AUC medio | Std | IC 95% | p-val vs 0.5 | Sig? | Acc | F1 | AUC por split |
|--------|-----------|-----|--------|-------------|------|-----|----|---------------|
| XGB precio | 0.5170 | 0.0330 | [0.4847, 0.5493] | 0.3781 | NO | 0.5167 | 0.4559 | [0.524, 0.4696, 0.527, 0.547] |
| XGB+FinBERT | 0.4948 | 0.0123 | [0.4827, 0.5068] | 0.4546 | NO | 0.4972 | 0.4344 | [0.51, 0.4943, 0.4948, 0.4801] |
| XGB+Morning | 0.5282 | 0.0419 | [0.4871, 0.5694] | 0.2706 | NO | 0.5028 | 0.4324 | [0.535, 0.5817, 0.4824, 0.5138] |
| XGB+Optuna | 0.5320 | 0.0495 | [0.4835, 0.5805] | 0.2865 | NO | 0.5028 | 0.5325 | [0.593, 0.5353, 0.472, 0.5285] |
| LightGBM | 0.5197 | 0.0192 | [0.501, 0.5385] | 0.1312 | NO | 0.5139 | 0.5419 | [0.5325, 0.5007, 0.5398, 0.5065] |
| LSTM | 0.4985 | 0.0829 | [?, ?] | ? | ? | 0.5067 | 0.3911 | [0.4548, 0.5714, 0.6027, 0.4152, 0.4486] |
| ARIMA | 0.4853 | 0.0176 | [?, ?] | ? | ? | 0.4733 | 0.5923 | [0.5124, 0.49, 0.4654, 0.4766, 0.4821] |


## 4. Tests estadísticos

### 4.1 Tests vs azar (H0: AUC = 0.5)

| Modelo | n splits | AUC medio | IC 95% | t-stat | p-value | Significativo |
|--------|----------|-----------|--------|--------|---------|---------------|
| XGB precio | 4 | 0.5169 | [0.4844, 0.5494] | 1.0198 | 0.3829 | NO |
| XGB+FinBERT | 4 | 0.4948 | [0.4828, 0.5068] | -0.8516 | 0.4570 | NO |
| XGB+Morning | 4 | 0.5282 | [0.4874, 0.5691] | 1.3541 | 0.2687 | NO |
| XGB+Optuna | 4 | 0.5322 | [0.4837, 0.5807] | 1.3016 | 0.2840 | NO |
| LightGBM | 4 | 0.5199 | [0.5011, 0.5387] | 2.0731 | 0.1299 | NO |
| LSTM | 5 | 0.4985 | [0.4258, 0.5712] | -0.0394 | 0.9705 | NO |
| ARIMA | 5 | 0.4853 | [0.4699, 0.5007] | -1.8681 | 0.1351 | NO |

### 4.2 Comparaciones pareadas (t-test pareado vs XGB precio)

| Comparación | ΔAUC | t-stat | p-value | Significativo |
|-------------|------|--------|---------|---------------|
| XGB+Morning vs XGB precio | +0.0113 | 0.3175 | 0.7717 | NO |
| XGB+FinBERT vs XGB precio | -0.0221 | -1.1588 | 0.3304 | NO |
| XGB+Optuna vs XGB precio | +0.0153 | 0.4941 | 0.6552 | NO |
| LightGBM vs XGB precio | +0.0030 | 0.1945 | 0.8582 | NO |

### 4.3 Mann-Whitney: sentimiento en días alcistas vs bajistas

| Grupo | N | Media sentiment_finbert |
|-------|---|-------------------------|
| Alcistas (label=1) | 1324 | 0.0147 |
| Bajistas (label=0) | 1268 | 0.0214 |

**U-stat = 821316.5, p-value = 0.342** → No significativo


## 5. Backtest con umbrales ML

**Período de test:** 360 días (predicciones LightGBM concatenadas)

**Buy & Hold:** retorno=-17.83%, Sharpe=-0.187, Max DD=-51.71%

| Threshold | Cobertura % | Accuracy % | Retorno total % | Sharpe | Max Drawdown % |
|-----------|-------------|------------|-----------------|--------|----------------|
| 0.5 | 56.9 | 50.28 | -0.44 | 0.147 | -58.72 |
| 0.55 | 16.7 | 51.67 | -5.12 | -0.041 | -36.36 |
| 0.6 | 8.3 | 50.56 | -17.08 | -0.62 | -27.41 |
| 0.65 | 6.1 | 50.56 | -21.7 | -1.06 | -27.43 |
| 0.7 | 2.2 | 50.56 | -16.82 | -0.905 | -20.16 |


## 6. Lag analysis — correlación sentimiento vs target

*(Calculado sobre 2225 días de train del primer fold — sin data snooping)*

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


## 7. Correlación de features con target (label)

| Feature | Correlación con label |
|---------|----------------------|
| returns | -0.0725 |
| bb_lower | -0.0303 |
| sma_7 | -0.0296 |
| sma_30 | -0.0295 |
| bb_upper | -0.0286 |
| fear_greed | 0.0246 |
| sentiment_morning | -0.0185 |
| sentiment_finbert | -0.0172 |
| macd_signal | 0.0165 |
| macd | 0.0063 |
| rsi_14 | -0.0042 |
| has_sentiment | -0.0028 |


## 8. Feature importance (XGBoost — último modelo)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | sma_30 | 0.116787 |
| 2 | sma_7 | 0.112524 |
| 3 | macd_signal | 0.112377 |
| 4 | returns | 0.112232 |
| 5 | bb_lower | 0.111262 |
| 6 | rsi_14 | 0.110060 |
| 7 | bb_upper | 0.109667 |
| 8 | macd | 0.108394 |
| 9 | fear_greed | 0.106698 |
| 10 | sentiment_avg | 0.000000 |


## DATOS CLAVE PARA EL INFORME

### Corpus
- **103,005 textos** indexados de **27 fuentes** (2011-2026)
- FinBERT (ProsusAI): **92,921 scores**, media=0.0519
- Kaggle precomputado: **10,081 scores**, media=-0.2822 (distribución trimodal {-1,0,+1})
- Timestamps fantasma (00:00:00 UTC): **304** — tratados como fecha-solamente, incluidos en sentiment\_morning

### Dataset
- **2592 observaciones diarias** (2019-01-30 → 2026-03-05)
- Balance clases: 51.08% alcistas — dataset casi balanceado
- Cobertura FinBERT: **96.9%** de días (solo 81 imputados a 0.0)
- Volatilidad anualizada BTC: **53.0%**
- corr(returns\_t, label\_t) = **-0.0725** → sin target leakage

### Resultados modelos
- **Mejor modelo (AUC):** XGB+Optuna = 0.5320 ± 0.0495
- **Sentimiento FinBERT completo (XGB+FinBERT):** AUC=0.4948 — *peor* que solo precio (AUC=0.517)
- **Sentimiento matutino (XGB+Morning):** AUC=0.5282 — mejor que FinBERT completo (+0.0334)
- ΔAUC Morning vs Full FinBERT: **+0.0334** → noticias tardías son reactivas (causalidad inversa)
- Ningún modelo supera el azar estadísticamente (todos p > 0.05) → consistente con EMH
- LSTM inestable: F1=0.3911 (modo predice siempre alcista en splits 2 y 4)

### Tests estadísticos
- Mann-Whitney sentimiento alcistas vs bajistas: U=821316.5, **p=0.342** → sin diferencia significativa
- Media sentiment alcistas=0.0147, bajistas=0.0214 (similar → sentimiento no discrimina)
- Poder estadístico con n=4 splits: ~18-20% para detectar ΔAUC=0.05 → p>0.05 esperado incluso con mejora real

### Backtest
- Buy & Hold (período test): retorno=-17.83%, Sharpe=-0.187
- Todos los thresholds ML generan retornos negativos → el modelo no puede usarse para trading rentable
- Mayor cobertura (threshold=0.50): 56.9% operaciones, Sharpe=0.147

### Lag analysis
- Mayor correlación sentimiento-label en lag **8 días** (r=0.0439) → correlación marginal
- Lag 0 (contemporáneo): r=-0.0155 — confirmado que sentimiento del día T no predice T+1

### Feature importance
- Las 8 features técnicas tienen importancias casi idénticas (~11% cada una → XGBoost no discrimina)
- sentiment\_avg = **0.000000** (importancia nula) → el modelo la descarta completamente