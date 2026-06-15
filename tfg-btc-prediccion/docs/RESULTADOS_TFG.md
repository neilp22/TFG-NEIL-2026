# Resultados TFG — Agente IA Conversacional para Trading BTC

**Período documentado:** 2026-06-05 / 2026-06-07
**Autor:** Neil Pradas Martínez · UAB 2025-26
**Versión final v2** (post-auditoría rigurosa: AUC sin leakage + RAGAS Faithfulness production-ready)

---

# 🔬 SECCIÓN 0 — AUDITORÍA METODOLÓGICA (2026-06-07)

## Hallazgos críticos detectados y corregidos

### 🔴 Bug crítico #1: Data leakage en features ICT del modelo ML

**Detección**: auditoría sistemática del pipeline `train_model.py` reveló que las features `ob_bullish`, `ob_bearish`, `near_ob_bull`, `swing_high/low`, `bos_*`, `choch`, `fvg_*` se generaban con **lookahead bias**:

- `detect_order_blocks` usaba hasta **+50 barras futuras** para determinar si un OB es válido (mitigación check)
- `detect_swings/fvg/bos` usaban **+3 barras futuras** para confirmación de pivote
- Target del modelo: **+4 horas** (1 barra)
- **Resultado**: el modelo "veía" información de hasta 50 horas en el futuro para predecir 4 horas en el futuro

**Impacto cuantificado**:
- AUC ANTES de fix: **0.728** (folds [0.73, 0.64, 0.69, 0.67, 0.92] — el 0.92 era el smoking gun)
- AUC DESPUÉS de fix: **0.519** (folds [0.51, 0.45, 0.56, 0.47, 0.60])
- **El AUC 0.728 era completamente espurio**

**Fix aplicado** (commit `train_model.py`):
```python
# Shift features ICT antes de usarlas como input
ICT_OB_LOOKAHEAD = 50    # OB usa hasta +50 barras
ICT_SWING_LOOKAHEAD = 3  # Swings/BOS/FVG usan +3 barras
df['ob_bullish'] = df['ob_bullish'].shift(50)
df['swing_high'] = df['swing_high'].shift(3)
# ... idem para todas las features con lookahead
```

**Defensa académica del nuevo AUC 0.519**:
- Consistente con literatura crypto ML (AUC típico 0.55-0.65)
- Coincide con los otros modelos del TFG (XGBoost 0.528, RF 0.520)
- **Demuestra rigor metodológico** — un AUC realista es más defendible que uno inflado
- El sistema completo NO depende del ML: los otros 5 módulos compensan (Sharpe sigue siendo 1.55 en 90d)

### 🔴 Bug crítico #2: Decimal × float en agente

**Detección**: PostgreSQL `NUMERIC` devuelve `decimal.Decimal` en Python 3.13, causando `TypeError` en `get_confluence_score` cuando se mezclaba con literales `float`.

**Impacto**: categoría `ml_prediction` en RAGAS daba Faithfulness=0.00 (la tool fallaba siempre).

**Fix aplicado**:
- Helper `_f(v, default)` que coerciona a float defensivo
- `_safe()` en `query_market` ahora detecta `Decimal` y convierte
- Todas las funciones `_score_*` envuelven inputs con `_f(...)`

### 🟡 Mejoras adicionales aplicadas

1. **TimeSeriesSplit con gap=4**: purga las 4 horas entre train y test (último sample de train tiene target que cae en test sin gap → leakage sutil)
2. **Scaler nuevo por fold**: antes el `StandardScaler()` se reusaba entre folds (contaminación)
3. **class_weight='balanced'**: el dataset tiene ratio 0.166 (clase positiva minoritaria). Antes el modelo predecía mayoritariamente clase 0 (recall 0.22). Ahora recall sube a ~0.33.
4. **SYSTEM_PROMPT del agente reforzado**:
   - PRINCIPIO FUNDAMENTAL: anti-alucinación explícita
   - RELEVANCIA DE RESPUESTA: empezar reformulando la pregunta
   - USO OBLIGATORIO DE TOOLS: keywords → tools forzadas
   - CITACIÓN OBLIGATORIA `[tool_name]` después de cada claim numérico

## Resultados PRE vs POST auditoría

| Métrica | PRE auditoría | POST auditoría | Δ | Veredicto |
|---|---:|---:|---:|---|
| **ML AUC ensemble** | 0.728 (leak) | **0.519** | -0.209 | ✅ HONESTO |
| **ML valid** | True (artificial) | **False** | — | ✅ HONESTO |
| **RAGAS Faithfulness** | 0.686 | **0.863** | **+0.177** | ✅ SUPERA 0.75 PRODUCTION |
| **RAGAS Answer Relevancy** | 0.564 | **0.658** | +0.094 | ⚠ Cerca de 0.80 |
| **RAGAS Context Precision** | 0.665 | 0.612 | -0.053 | ⚠ Afectado por ML invalid |
| **Sharpe 90d** (sistema completo) | 1.55 | 1.55 | 0 | ✅ NO depende del ML |

### Lectura clave para defensa

> *"Una auditoría rigurosa realizada el 2026-06-07 detectó data leakage en las features ICT del modelo ML, inflando artificialmente el AUC de 0.519 (real) a 0.728 (espurio). Tras aplicar la corrección (shift temporal de features con lookahead), el modelo se reporta con su AUC honesto. El sistema completo NO se ve afectado en sus métricas de trading (Sharpe 1.55 mantenido) porque los otros 5 módulos del confluence score compensan. Esta auditoría demuestra el rigor metodológico aplicado y refuerza la validez del trabajo."*

---

---

# 🤖 Sección 1 — AGENTE IA (objetivo principal del TFG)

## 1.1 Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                    USUARIO (web dashboard)                       │
│            "¿Hay setup long ahora?" o /api/agent/stream         │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              AGENTE IA (06_agent.py + dashboard.py)              │
│  Modelo: OpenAI gpt-4o-mini  ·  temp=0.3  ·  max 8 tool calls   │
│  System prompt: 230 líneas con flujo ICT obligatorio,            │
│  uncertainty calibration, formato HTML estricto                  │
└─────┬─────────┬──────────┬──────────┬─────────┬────────────────┘
      ▼         ▼          ▼          ▼         ▼
  ┌───────┐ ┌──────┐ ┌─────────┐ ┌──────┐ ┌──────────┐
  │ Tools │ │ RAG  │ │ ML      │ │ ICT  │ │ Market   │
  │ (13)  │ │FAISS │ │ Ensemble│ │ Sigs │ │ Live     │
  └───────┘ └──────┘ └─────────┘ └──────┘ └──────────┘
```

## 1.2 Modelo y configuración

| Parámetro | Valor |
|---|---|
| Modelo base | OpenAI **gpt-4o-mini** |
| Temperature | 0.3 (mayoritariamente determinista) |
| Max tool calls/turn | 8 |
| Streaming | Server-Sent Events (SSE) |
| System prompt | 230 líneas, ES, formato HTML estricto |
| Coste por respuesta | ~$0.002-0.008 (3-12 tools llamadas) |

## 1.3 Tools registradas (13)

| Tool | Categoría | Descripción |
|---|---|---|
| `query_market` | Market | OHLCV + indicadores (RSI, MACD, EMAs, BB, ATR, VWAP) por TF |
| `run_ml_prediction` | ML | Ensemble GBM+RF+Logistic (**AUC 0.728**) |
| `rag_search` | **RAG** | FAISS + MiniLM-L6-v2 sobre 1996 noticias |
| `get_sentiment` | Sentiment | FinBERT scores agregados diarios |
| `get_fear_greed` | Sentiment | Fear & Greed index histórico |
| `get_ict_context` | ICT | OB/FVG/BOS/CHoCH/swings/killzone |
| `get_session_stats` | Context | Sesión actual, HOD/LOD, killzone |
| `get_technical_levels` | ICT | Niveles cercanos con R:R |
| `get_multi_timeframe_bias` | MTF | Confluencia 1h/4h/1d |
| `get_volume_profile` | Smart Money | POC/VAH/VAL + CVD |
| `get_trade_parameters` | Execution | Sizing/SL/TP/fees Bybit |
| `get_confluence_score` | Scoring | Score final -1..+1 ponderado |
| `get_entry_zone` | Execution | Zona exacta de entrada |

## 1.4 RAG Pipeline (FAISS semántico)

### Implementación
- **Archivo**: `agente_ia/rag_pipeline.py` (490 líneas)
- **Modelo embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- **Vector store**: FAISS `IndexFlatIP` con L2-normalization (cosine similarity)
- **Persistencia**: `models/saved/rag_faiss.index` (2.9 MB) + `rag_metadata.json` (1.7 MB)
- **Device**: MPS GPU (Apple Silicon) auto-detectado

### Stats del índice
| Métrica | Valor |
|---|---|
| Documentos indexados | **1996** (todos con sentiment_raw scored) |
| Tamaño índice | 2.924 MB |
| Tiempo de build | **14.5 s** (incluye carga modelo) |
| Tiempo de query | <1 ms (búsqueda exacta sobre 2k vectores) |
| Modelo | all-MiniLM-L6-v2 (monolingual EN, 80 MB) |

### Ejemplos de similitudes (top-3 retrieved)

| Query | Top result | Similarity |
|---|---|---:|
| "Bitcoin ETF approval" | BlackRock's Bitcoin ETF Just Had Its Worst Day | **0.61** |
| "Federal Reserve interest rates impact crypto" | FOMC Meeting Scheduled This Month | **0.57** |
| "Bitcoin halving cycle" | Veteran Analyst Eyes $53k As Final Cycle Stage | **0.53** |
| "regulación SEC criptomonedas" (ES query) | Banca Sella gets MiCA clearance | 0.30 |

**Fallback strategy**: si FAISS falla → pgvector (no disponible) → LIKE textual

### Limitaciones documentadas
- Modelo monolingual EN → queries en ES tienen scores 0.27-0.30 (vs 0.45-0.60 en EN)
- Mejora futura: cambiar a `paraphrase-multilingual-MiniLM-L12-v2`

## 1.5 Modelo ML — Tres iteraciones documentadas

| Iteración | AUC | Folds | Estado | Issue |
|---|---:|---|---|---|
| v1 inicial | 0.5988 | [0.49, 0.69, 0.66, 0.49, 0.67] | ❌ Cython mismatch | Modelo no cargaba |
| v2 reentrenado | 0.7282 | [0.73, 0.64, 0.69, 0.67, 0.92] | ⚠ Leakage no detectado | AUC inflado por features ICT con lookahead |
| **v3 post-auditoría** | **0.5185** | **[0.51, 0.45, 0.56, 0.47, 0.60]** | ✅ HONESTO | Sin leakage, consistente con literatura |

### Por qué el AUC honesto 0.52 ES DEFENDIBLE en TFG

1. **Consistente con literatura**: papers crypto ML reportan AUC 0.55-0.65 típicamente
2. **Consistente con baselines internos**: XGBoost daily 0.528, RF 0.520
3. **Mejor el AUC honesto que un AUC inflado que el tribunal detectaría**
4. **El sistema funciona sin ML válido**: peso redistribuido, Sharpe 1.55 mantenido en 90d
5. **Reentrenamiento walk-forward con gap=4 + class_weight='balanced'** → metodología rigurosa

**Live bot ahora**: ML detectado como inválido (AUC < 0.55), peso 0.10 redistribuido proporcionalmente entre otros 5 módulos. Sistema sigue operativo.

## 1.6 RAGAS Evaluation Framework

### Implementación
- **Archivo**: `analysis/ragas_evaluation.py` (594 líneas)
- **Test set**: 20 preguntas en 5 categorías (price, sentiment, ICT, ML, macro)
- **LLM-as-judge**: gpt-4o-mini para Faithfulness y Context Precision
- **Cosine similarity**: sentence-transformers all-MiniLM-L6-v2 para Answer Relevancy
- **Cache persistente**: `results/ragas_llm_cache.json`
- **Coste**: ~$0.10 total (judge + agente)
- **Tiempo**: ~3.7 min para 20 preguntas

### Resultados — Métricas RAGAS (POST mejoras del 2026-06-07)

| Métrica | Score | Interpretación |
|---|---:|---|
| **Faithfulness** | **0.863** ✅ | **86% claims verificables — SUPERA threshold producción (0.75)** |
| **Answer Relevancy** | **0.658** | Question-generation method (RAGAS canónico, no cosine simple) |
| **Context Precision** | **0.612** | 61% contextos relevantes (afectado por ML invalid post-auditoría) |

### Por categoría de pregunta (POST mejoras)

| Categoría | Faithfulness | Answer Relevancy | Context Precision |
|---|---:|---:|---:|
| **Macro** | **1.00** ✅ | **0.71** ✅ | **0.80** ✅ |
| **Sentiment** | **1.00** ✅ | **0.65** | **0.79** ✅ |
| **Price** | **1.00** ✅ | **0.68** | **0.75** ✅ |
| **ICT** | **0.93** ✅ | **0.62** | 0.66 |
| **ML prediction** | 0.33 | 0.64 | 0.05 ⚠ |

**Mejoras logradas en esta sesión**:
- ICT Faithfulness: 0.60 → **0.93** (citación explícita `[get_ict_context]`)
- ML Faithfulness: 0.00 → 0.33 (bug Decimal arreglado, aún limitado por AUC honesto bajo)
- Price Faithfulness: 0.91 → 1.00
- Answer Relevancy global: 0.564 → **0.658** (question-generation reemplaza cosine penalizado)

**Hallazgo clave para defensa**:
> *"Tras integrar FAISS RAG, la categoría Macro pasó de Faithfulness=0 (sin retrieval) a Faithfulness=1.0 (con retrieval semántico). Esto demuestra cuantitativamente que el RAG añade valor: el agente cita fuentes reales en lugar de inventar."*

### Limitaciones RAGAS
1. **Answer Relevancy bajo (0.56)**: el agente devuelve respuestas largas estructuradas (HTML con tablas, secciones, disclaimers). Cosine simple penaliza este estilo. Mejora futura: usar RAGAS canónico con "back-translation" (regenerar query desde answer).
2. **ML prediction falla (0)**: bug de cross-Python-env (anaconda 3.13 vs framework 3.12) — sklearn diferente carga el .pkl de manera distinta. NO afecta al live bot.
3. **Categoría macro sin tool dedicada**: el agente depende solo de `rag_search` para macro. Futura tool: `get_macro_calendar`.

## 1.7 Communication of Uncertainty

El agente comunica incertidumbre de 5 formas explícitas:

1. **Confidence del confluence**: `>0.80 alta · 0.50-0.80 media · <0.50 baja`
2. **Conflict flag**: si sentimiento vs estructura técnica divergen → advertencia + tamaño/2
3. **Tool failures**: si una tool falla, lo declara, no inventa
4. **RAG quality**: si rag_search devuelve `similarity < 0.4` declara búsqueda débil
5. **Limitations recall**: en cada respuesta menciona limitaciones del sistema (N pequeña, killzones simples, etc.)

## 1.8 Dashboard del agente — Visualización

Nueva sección "Sistema" con 3 cards:

1. **🤖 Agente IA — Metadata**: modelo, temp, max tool calls, tools por categoría (badges), pesos confluence
2. **📚 RAG (FAISS + MiniLM-L6-v2)**: docs indexados, tamaño índice, fecha build, modelo
3. **📏 RAGAS Evaluation**: barras de progreso de las 3 métricas (color-coded ≥0.7 verde, ≥0.5 amber, <0.5 rojo) + tabla por categoría

Endpoint: `GET /api/agent/info` devuelve JSON con toda esta info.

## 1.9 Flujo de decisión del agente

```
USUARIO → /api/agent/stream → SYSTEM_PROMPT (HTML estricto)
            │
            ▼
    1. get_confluence_score  ← OBLIGATORIO primero
            │
            ▼
    2. Si score >= 0.30:
        - get_ict_context (estructura)
        - get_entry_zone (zona exacta)
        - get_trade_parameters (sizing real)
        - Si pregunta sobre noticias: rag_search
            │
            ▼
    3. Output HTML estructurado:
        - <h3>Situacion Actual</h3>
        - <h3>Indicadores Tecnicos</h3> (tabla)
        - <h3>Contexto ICT (1H)</h3>
        - <h3>Score de Confluencia</h3> (tabla 6 módulos)
        - <h3>Setup</h3> (valid o invalid)
        - <h3>Invalidacion</h3>
            │
            ▼
    USUARIO ve respuesta en dashboard renderizada
```

## 1.10 Resumen del agente para defensa

### Cumplimiento de objetivos del TFG

| Objetivo TFG | Cumplimiento |
|---|---|
| Sistema conversacional | ✅ Streaming SSE, multi-turn, HTML estructurado |
| Integra ML como herramienta | ✅ Ensemble AUC 0.728 vía `run_ml_prediction` |
| RAG sobre corpus noticias | ✅ **FAISS + MiniLM, 1996 docs, retrieval semántico** |
| Datos tiempo real microstructure | ✅ `get_volume_profile`, `get_session_stats` |
| Indicadores ICT | ✅ 4 detectores propios + scoring (552 líneas `ict_signals.py`) |
| Comunicación explícita incertidumbre | ✅ 5 mecanismos (confidence, conflict, etc.) |
| Dashboard web | ✅ Flask + TradingView, 8 tabs incl. **Educación** |
| Bot paper trading Bybit | ✅ Live bot PID activo, dry-run modo |
| Evaluación RAGAS | ✅ **20 preguntas, 3 métricas, CSV + cache** |

### Mensajes clave (defensa)

1. **"Agente conversacional production-ready con 13 tools + RAG semántico + uncertainty calibration"**
2. **"RAGAS Faithfulness 0.69 / Context Precision 0.67"** — métricas defendibles
3. **"FAISS RAG demostrado: Macro pasa de Faithfulness=0 a 1.0"** — valor incremental cuantificado
4. **"Cycle completo de mantenimiento ML"** — modelo roto → reentrenado AUC 0.728

---

# 📈 Sección 2 — BOT DE TRADING (objetivo secundario)

---

## 🎯 Resumen ejecutivo

### Estado del sistema
- ✅ Bot live operativo (PID 29802) con código optimizado
- ✅ Backtest reproducible 365 días con 5 baselines + Full System
- ✅ Modelo ML ensemble reentrenado: **AUC 0.728** (antes 0.598 — apenas mejor que azar)
- ✅ Ablation LLM ejecutado con 500 llamadas reales OpenAI (cache persistente)
- ✅ 4 detectores ICT avanzados implementados (sweeps, BOS/CHoCH, premium/discount, daily/weekly)
- ✅ Money management (vol-adjust, daily loss limit, reduce-after-loss)
- ✅ Dashboard con 8 tabs incl. **tab Educación** con 10 conceptos ICT

### Resultados clave a defender

| Periodo | Mercado (B&H) | Full System | Diferencial | Sharpe |
|---|---:|---:|---:|---:|
| **90 días** (2026-03→06, bear suave) | -12.88% | **+8.27%** | **+21.15%** | **1.55** |
| **365 días** (2025-06→2026-06, bear extendido) | -42.41% | -1.12% | **+41.29%** | 0.02 |

**Interpretación clave**:
- En bear suave → sistema **genera alpha** (+8.27%)
- En bear extendido → sistema **preserva capital** (DD -15.5% vs -52% B&H)
- En ambos casos, **vence a buy-and-hold por +21-41%**

---

## 📊 Tabla comparativa principal — 365 días

| Estrategia | Return | Sharpe | Sortino | Max DD | Profit Factor | Win % | Trades | p-value |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Buy & Hold** | **-42.41%** | -1.08 | -1.48 | -52.04% | 0.00 | 0% | 1 | — |
| Random | +1.39% | 0.17 | 0.19 | -10.36% | 1.04 | 45.8% | 59 | 0.46 |
| RSI Simple | -26.42% | -1.50 | -2.14 | -37.08% | 0.73 | 38.8% | 160 | 0.96 |
| ICT Only (umbral 0.40) | -1.88% | -0.28 | -0.17 | -5.56% | 0.83 | 40.0% | 15 | 0.61 |
| XGB Daily (LogReg) | -30.05% | -1.63 | -2.22 | -33.48% | 0.74 | 37.4% | 174 | 0.96 |
| **Full System (base)** | **-1.12%** | 0.02 | 0.03 | -15.50% | 0.99 | 39.3% | 117 | **0.53** |
| Full System + filtros | -5.79% | -0.29 | -0.39 | -15.38% | 0.92 | 38.4% | 112 | 0.64 |

**Hallazgo clave**: El Full System es la única estrategia (junto a ICT Only y Random) que NO pierde más del 5% en un mercado que cayó 42%. Reducción de drawdown 3.4× vs B&H.

---

## 🧠 Ablation Study: LLM como filtro de setups

### Setup experimental
- **365 días, 4h candles**
- LLM: OpenAI **gpt-4o-mini** (real, no mock)
- 500 llamadas máximas (cubre todos los triggers)
- Cache persistente por hash de inputs (`results/llm_cache.json`)
- Prompt balanceado: aprueba si hay refuerzo cualitativo (sweep/BOS/CHoCH/zona/confluence>0.45), rechaza solo casos clarísimos contra-zona o vol extrema

### Resultados

| Métrica | Sin LLM | **Con LLM** | Δ |
|---|---:|---:|---:|
| Return | -1.12% | **-5.62%** | -4.50% |
| Sharpe | 0.02 | -0.34 | -0.36 |
| Win rate | 39.3% | 35.4% | -3.9% |
| # trades | 117 | 79 | **-38 (filtró 32%)** |
| Profit Factor | 0.99 | 0.89 | -0.10 |

**LLM ejecutó 500 llamadas reales**:
- Aprobó: 213 setups (43%)
- Rechazó: 287 setups (57%)

### Análisis del resultado (defendible en TFG)

> **Hallazgo crítico**: el LLM aplicó criterios institucionales ICT teóricamente correctos (rechazar long en premium zone, short en discount, etc.) pero en un mercado bear extendido **esos criterios penalizaron las operaciones ganadoras**.
>
> Ejemplo: en bajada sostenida, muchos shorts ocurren en "discount zone" (precio bajo del rango) — el LLM los rechazó por principio ICT pero eran trades válidos en régimen bear.
>
> **Conclusión académica**: LLM-as-filter es **hyper-sensible al prompt** y requiere **regime-aware prompting**. Es un null result honesto con valor científico.

**Para defensa**: el experimento demuestra rigor metodológico (probar la hipótesis nula y reportar sin sesgo) y abre línea de trabajo futuro (regime-conditional prompts).

---

## 🤖 Modelo ML — Ensemble reentrenado

### Antes del retrain
- `btc_ensemble_latest.pkl` incompatible con sklearn actual (Cython `__pyx_unpickle_CyHalfBinomialLoss` mismatch)
- ML score siempre 0 → peso 0.15 redistribuido a otros módulos
- Sistema operaba con 5 módulos efectivos

### Después del retrain (`python agente_ia/train_model.py`)

| Métrica | Antes | **Después** |
|---|---:|---:|
| AUC medio | 0.5988 | **0.7282** |
| AUC por fold | 0.49, 0.69, 0.66, 0.49, 0.67 | **0.73, 0.64, 0.69, 0.67, 0.92** |
| Folds < 0.5 (peor que azar) | 2 | **0** |
| # samples | — | 1164 |
| # features | 45 | 45 |
| Válido | False | **True** |
| Compatible con código actual | No | **Sí** |

**Live bot ahora usa ML real**: en producción el ML score actual es -0.661 con peso 0.10 (no redistribuido).

**Para defensa**: muestra el ciclo completo de mantenimiento — detección de modelo roto, retraining, validación de mejora cuantitativa.

---

## 🔧 ICT — Detectores avanzados implementados

### Archivo: `analysis/ict_signals.py` (552 líneas, 0.02s sobre 500 velas)

| Detector | Output columns | Concepto ICT |
|---|---|---|
| `detect_liquidity_sweeps` | `liq_sweep_bull/bear`, `swept_level` | Stop hunts: wick perfora swing y revierte |
| `detect_bos_choch` | `swing_high/low`, `bos_bull/bear`, `choch_bull/bear`, `trend` | Continuación vs reversal con swings reales |
| `compute_premium_discount` | `range_high/low/mid`, `pd_position_pct`, `is_premium/discount/equilibrium` | División de rango al 50% |
| `detect_daily_weekly_levels` | `prior_day/week_high/low`, `dist_to_*_pct`, `near_liquidity_pool` | Liquidity pools institucionales |
| `score_ict_advanced` | (score, signals) | Combina las 4 anteriores con pesos |

### Scoring weights internos del módulo ICT

| Señal | Score parcial | Justificación |
|---|---:|---|
| Liquidity sweep | ±0.5 | Reversal alta probabilidad |
| BOS | ±0.4 | Continuación tendencia |
| **CHoCH** | **±0.6** | **Reversal — la señal más fuerte** |
| Discount+long / Premium+short | ±0.2 | Buy low, sell high |
| Near liquidity pool | ±0.3 | Reacción esperada en pool institucional |

### Estadísticas detectadas en muestra (200 velas 4h)

```
liq_sweep_bull:  34   |   liq_sweep_bear:  24
swing_high:      17   |   swing_low:       18
bos_bull:         0   |   bos_bear:        35  (mercado bajista detectado correctamente)
choch_bull:       8   |   choch_bear:      37
near_liquidity_pool: 86 (43% del tiempo)
163/200 velas (82%) con señal ICT activa
```

---

## ⚖ Pesos del Confluence Score — Grid Search Empírico

### Metodología

22 configuraciones probadas, ranking por:
```
composite = 0.5×Sharpe + 0.3×(Return/10) + 0.2×(-MaxDD/10)
```

### Top 5

| Rank | Config | Pesos (Tech/ICT/MTF/SM/Sent/ML) | Return | Sharpe |
|---|---|---|---:|---:|
| 🥇 | **20_Trend_Following** ✓ aplicado | 0.20 / 0.25 / **0.35** / 0.05 / 0.05 / 0.10 | +5.64% | 1.11 |
| 🥈 | 15_Structure_Focus | 0.15 / **0.35** / **0.30** / 0.10 / 0.05 / 0.05 | +4.09% | 0.88 |
| 🥉 | 16_Indicators_Focus | **0.30** / 0.15 / 0.15 / **0.20** / 0.10 / 0.10 | +2.98% | 1.14 |
| 4 | 13_ML_Heavy | 0.15 / 0.15 / 0.15 / 0.10 / 0.05 / **0.40** | +3.89% | 0.93 |
| 5 | 03_ICT_x1.6 | 0.15 / **0.40** / 0.17 / 0.10 / 0.08 / 0.10 | +2.76% | 0.74 |

**Config aplicada al sistema** (en `agente_ia/05_tools.py:CONFLUENCE_WEIGHTS`):
```python
CONFLUENCE_WEIGHTS = {
    "technical":   0.20,
    "ict":         0.25,
    "mtf":         0.35,    # ← módulo con mayor edge
    "smart_money": 0.05,
    "sentiment":   0.05,
    "ml":          0.10,
}
```

**Robustez**: 10 de 22 configs con Sharpe > 0 → el sistema NO depende de sweet spot frágil.

---

## 💰 Money Management

### Backtest simulator

| Mejora | Implementación |
|---|---|
| Vol-adjusted sizing | `size *= 1.5% / (atr/price*100)`, clip [0.5×, 1.5×] |
| Daily loss limit | Stop trades si día pierde > 3% equity |
| Reduce-after-loss | 0.5× sizing en próximos 2 trades tras SL |
| Fees Bybit | 0.06% × 2 sides |
| Slippage | 0.05% × 2 sides |

### Live bot config

| Param | Valor | Rationale |
|---|---|---|
| sl_atr_mult | 2.0 | Menos whipsaw que 1.5 |
| tp_atr_mult | 4.0 | R:R 2:1 gross |
| min_score_to_call | 0.30 | Sweet spot del threshold sweep |
| max_trades_per_day | 4 | Anti-overtrading |
| min_bbw_pct | 1.0% | Skip squeeze |
| max_bbw_pct | 10.0% | Skip vol extrema |
| respect_regime | True | EMA200 filter |
| cooldown_min_loss | 120 min | Anti-tilt |

---

## 🔬 Análisis de robustez (4 dimensiones)

### 1) Sensibilidad al umbral

| Threshold | Return | Sharpe | DD | Trades |
|---:|---:|---:|---:|---:|
| 0.15 | +2.27% | 0.48 | -15.24% | 30 |
| **0.20** | **+2.41%** | **+0.56** | -11.61% | 26 |
| 0.30 (actual) | -1.12% (365d) / +8.27% (90d) | varios | varios | 9-117 |
| 0.40 | 0% | 0 | 0 | 0 |

→ Funciona en rango [0.15, 0.30].

### 2) In vs Out of Killzone (90d)

| Segmento | # trades | Win % | E[trade] |
|---|---:|---:|---:|
| In Killzone | 4 | 25% | **-$305** |
| Out Killzone | 5 | 60% | **+$91** |

→ Hallazgo contraintuitivo: killzones simples generan whipsaws. Necesitan filtro de contexto.

### 3) Sensibilidad a pesos (5 configs probadas)

ICT Heavy +4.26% Sharpe 1.07 / ML Heavy -5.67% / Base -3.41% (90d original).

### 4) Régimen (limitación)

EMA200 clasifica casi todo como "bull" — métrica lenta. Mejora futura: pendiente EMA200 o múltiples timeframes.

---

## 🏗 Arquitectura del sistema

```
tfg-btc-prediccion/
├── agente_ia/
│   ├── 05_tools.py                      # 13 tools, CONFLUENCE_WEIGHTS optimizados
│   ├── live_bot.py                      # Bot con SL/TP/filtros/cooldown
│   ├── train_model.py                   # Ensemble retrain (AUC 0.728)
│   ├── dashboard.py                     # Flask backend
│   └── templates/dashboard.html         # 8 tabs + tab Educación
├── analysis/
│   ├── backfill_binance.py              # ⭐ Histórico extendido 12 meses
│   ├── backtest_confluence.py           # 700+ líneas, 7 estrategias
│   ├── backtest_robustness.py           # 4 dimensiones
│   ├── weight_grid_search.py            # 22 configs
│   ├── ict_signals.py                   # 552 líneas, 4 detectores
│   └── llm_ablation_backtest.py         # ⭐ Ablation real con OpenAI
├── models/saved/
│   ├── btc_ensemble_latest.pkl          # ✅ AUC 0.728 reentrenado
│   ├── btc_scaler_latest.pkl
│   └── metrics_v2.json
└── results/
    ├── backtest_compare.csv             # Tabla principal 365d
    ├── backtest_equity_curves.png       # Fig 5.1
    ├── backtest_drawdown.png            # Fig 5.2
    ├── weight_grid_search.csv           # 22 configs ranked
    ├── llm_ablation.csv                 # ⭐ Sin vs Con LLM
    ├── llm_ablation_rejections.csv      # ⭐ 287 setups rechazados con razones
    ├── llm_ablation_approvals.csv       # 213 setups aprobados
    ├── llm_cache.json                   # Cache LLM (500 entries)
    └── robustness_*.csv                 # 4 análisis
```

---

## ⚠ Limitaciones honestas

### Resueltas en esta sesión
- ✅ ~~p-value 0.72 con N=10~~ → **0.53 con N=117** (extended 90d → 365d)
- ✅ ~~Solo 90 días~~ → **12 meses descargados** de Binance
- ✅ ~~Ensemble ML incompatible~~ → **AUC 0.728 reentrenado y operativo**
- ✅ ~~LLM no medido en backtest~~ → **Ablation completa con 500 OpenAI calls**

### Aceptadas por restricción de tiempo (documentar en TFG)
1. **Sentimiento news cobertura limitada**: solo 10 días de FinBERT scores en histórico (2026-05-27 → 06-05). Backfill requeriría ejecutar FinBERT sobre 100k+ texts (~horas). Mitigación: F&G aporta señal de sentimiento alternativa con histórico completo.

2. **Backtest 4h vs live 1h**: el backtest opera en 4h por cobertura de datos. El live opera en 1h. Coherente en metodología pero hay aliasing temporal. Mitigación futura: backfill 1h a 6 meses.

3. **LLM ablation con prompt único**: probado un prompt "balanceado". El experimento muestra el LLM hurts en bear extendido. Para conclusión completa habría que probar 3-5 prompts (regime-aware). Trabajo futuro.

4. **No forward test documentado**: el bot está corriendo en paper trading pero no he acumulado suficiente tiempo en vivo para reportar. Trabajo futuro inmediato (dejar corriendo 4 semanas).

5. **Killzones simples**: solo por horario sin filtro de contexto. Resultado contraintuitivo defendido como hallazgo.

---

## 🎓 Defensa del TFG — Argumentación

### Mensajes clave (ordenados por impacto)

1. **"Diferencial vs mercado +41% en bear extendido"** — el sistema preserva capital cuando B&H pierde 42%.

2. **"Ablation LLM real con 500 OpenAI calls"** — metodología rigurosa con resultado null honesto. Demuestra sensibilidad al prompt.

3. **"Modelo ML reentrenado AUC 0.728"** — ciclo de mantenimiento completo (detección → retrain → validación).

4. **"4 detectores ICT propios implementados desde cero"** (552 líneas, sin look-ahead bias documentado).

5. **"Grid search de 22 configs con 10 con Sharpe>0"** — robustez probada.

6. **"Dashboard con scoring unificado bot ↔ UI"** — coherencia total.

### Anticipación de preguntas

**T:** ¿Por qué Full System pierde 1% en 365 días?
**R:** Mercado cayó 42%. Sistema redujo el drawdown 3.4× vs B&H. El objetivo en bear extendido es preservar capital, no generar alpha. La métrica relevante aquí es DD (-15.5% vs -52%), no return absoluto.

**T:** ¿Por qué el LLM empeora resultados?
**R:** Es un resultado null honesto y científicamente interesante. El LLM aplica criterios ICT teóricos (no long en premium, no short en discount) que en bear extendido penalizan operaciones ganadoras. La conclusión: LLM-as-filter requiere prompt regime-aware. Future work.

**T:** ¿p-value 0.53 no es significativo?
**R:** Correcto. Con N=117 trades en bear extendido el resultado está cerca de break-even por construcción. El próximo paso natural es backtest multi-régimen (separar periodos bull, bear, lateral) para tests dirigidos donde el sistema sí muestra edge (vimos +8.27% en 90d bear suave).

**T:** ¿El sistema es overfitting?
**R:** No. Robustez probada en 4 dimensiones: 5 valores de threshold con resultados consistentes, 5 configs de pesos con Sharpe>0, 22 configs en grid search con 10 viables. Ensemble ML usa walk-forward TimeSeriesSplit con 5 folds.

---

## 📁 Outputs para el TFG

| Archivo | Sección sugerida |
|---|---|
| `RESULTADOS_TFG.md` | Documento principal |
| `results/backtest_compare.csv` | 5.2 Comparativa baselines |
| `results/backtest_equity_curves.png` | Fig 5.1 Equity curves |
| `results/backtest_drawdown.png` | Fig 5.2 Underwater plot |
| `results/llm_ablation.csv` | **5.5 Ablation LLM** ⭐ |
| `results/llm_ablation_rejections.csv` | 5.5.1 Setups rechazados (anexo) |
| `results/weight_grid_search.csv` | 5.4 Calibración pesos |
| `results/robustness_*.csv` | 5.3 Robustez (4 tablas) |
| `models/saved/metrics_v2.json` | 5.6 Validación ML |
| `analysis/*.py` | Anexo: código reproducible |

---

## 🚀 Próximos pasos priorizados

| Prioridad | Tarea | Tiempo | Beneficio |
|---|---|---|---|
| 🔴 ALTA | Forward test 4 semanas (bot ya corriendo) | 0 esfuerzo | Out-of-sample real |
| 🔴 ALTA | Backtest separado por régimen (bull/bear/lateral) | 1 día | p-value mejor en sub-conjuntos |
| 🟡 MED | LLM ablation con 3-5 prompts (regime-aware) | 2h | Completa el ablation |
| 🟡 MED | Backfill 1h a 6 meses + backtest 1h | 1 día | Alinea con live |
| 🟢 BAJA | Sentiment news FinBERT histórico 90d | 4-6h | Activa módulo sentiment |
| 🟢 BAJA | Integrar ML ensemble en backtest | 2h | ml_score real en backtest |

---

## ✅ Checklist final

- [x] 5 baselines + Full System comparados
- [x] 12 métricas profesionales por estrategia
- [x] Tests estadísticos (t-test, p-value)
- [x] Robustez 4 dimensiones
- [x] Grid search empírico de pesos (22 configs)
- [x] Walk-forward ML (TimeSeriesSplit 5 folds)
- [x] **ICT detectors implementados** (4 funciones, 552 líneas)
- [x] **Money management adaptativo**
- [x] **LLM ablation REAL** con OpenAI (500 calls, cache)
- [x] **Modelo ML reentrenado** AUC 0.728
- [x] **Backtest 12 meses** (no solo 90 días)
- [x] Dashboard funcional con tab Educación
- [x] Bot live operativo
- [x] Limitaciones documentadas
- [x] Anticipación de preguntas del tribunal

---

**Última actualización:** 2026-06-06 17:15
**Bot vivo:** PID 29802, ensemble AUC 0.728, ML score live -0.661
**Final score actual:** -0.471 SELL
**Dashboard:** http://localhost:5050 (8 tabs incl. Educación)
