# Predicció de la direcció diària de Bitcoin amb *machine learning* i anàlisi de sentiment

> **TFG — Grau en Enginyeria de Dades · Universitat Autònoma de Barcelona (UAB), curs 2025/26**
> Autor: **Neil Pradas Martínez**

Un resultat nul honest i un sistema multi-agent d'IA construït sobre la seva infraestructura.

---

## 1. Resum

Aquest treball s'estructura en **dues fases encadenades**:

1. **Fase 1 — Investigació predictiva.** Investiga si un sistema d'aprenentatge automàtic
   enriquit amb anàlisi de sentiment de notícies financeres pot predir la direcció diària
   del preu de Bitcoin amb significança estadística. Sobre infraestructura pròpia
   (PostgreSQL, **105.125 notícies** i **65.466 registres OHLCV**) s'avaluen quatre famílies
   de models amb *walk-forward validation*. Després de successives auditories del *pipeline*
   (que van corregir *data leakage*, mostres de test insuficients i biaixos d'avaluació), el
   millor resultat honest és un **AUC ≈ 0,52** (*p* = 0,287; IC95% [0,484, 0,581]),
   **indistingible de l'atzar** i coherent amb l'eficiència de mercat semi-forta.

2. **Fase 2 — Sistema multi-agent d'IA.** Acceptat que els models no són predictors fiables,
   la infraestructura es reaprofita com a base d'un agent d'IA sobre **GPT-4o** que tracta el
   model com una **font de senyal feble** (pes 10%) entre d'altres. El disseny és un equip de
   tres rols (**Analista · Risk Manager · Decision Maker**) sota el principi
   **«el codi verifica, el LLM opina»**, amb la decisió final presa per un **motor de regles
   determinista**. La qualitat del component RAG es valida amb RAGAS
   (*Faithfulness* **0,863**).

**Conclusió clau:** en *machine learning* financer, la dificultat real no és entrenar models
—qualsevol obté un AUC de 0,73— sinó assegurar-se que el número reportat és real. Un *null
finding* verificat val més que un positiu que ningú no ha contrastat.

---

## 2. Resultats principals

### Models ML — *walk-forward* (4 splits × 90 dies, gap = 7)

| Model | AUC | IC95% | *p* | Significatiu |
|---|---|---|---|---|
| XGBoost + Optuna | **0,532** | [0,484, 0,581] | 0,287 | no |
| XGBoost + Morning | 0,528 | [0,487, 0,569] | 0,271 | no |
| LightGBM | 0,520 | [0,501, 0,539] | 0,130 | no |
| XGBoost preu | 0,517 | [0,484, 0,549] | 0,378 | no |
| XGBoost + FinBERT | 0,495 | [0,483, 0,507] | 0,455 | no |
| LSTM | 0,499 | [0,426, 0,571] | 0,971 | no |
| ARIMA | 0,485 | [0,470, 0,501] | 0,135 | no |

**Cap model supera l'atzar amb significança** (α = 0,05): tots els intervals de confiança
inclouen el 0,5.

### Dos resultats originals que expliquen el *null finding*

- **Auditoria de *leakage*:** una *feature* ICT (`detect_order_blocks`) usava +50 barres
  futures (*forward window*). En corregir-ho, l'AUC aparent va caure de **0,728 → 0,519**.
- **Causalitat inversa del sentiment:** el sentiment **matinal** (< 18:00 UTC) supera el
  complet en 3 de 4 splits (ΔAUC mitjà **+0,033**). Les notícies de la tarda *descriuen* el
  moviment, no l'*anticipen*.

### Sistema multi-agent

- **RAGAS *Faithfulness* = 0,863** (per sobre del llindar industrial 0,75).
- *Backtest* del model com a estratègia: el **Sharpe empitjora** en augmentar el llindar de
  confiança (−0,166 → −1,199) → signatura d'un model sense *edge* predictiu.

---

## 3. Estructura del repositori

```
TFG CODIGO/
├── README.md                  ← aquest fitxer
├── requirements.txt           ← (es troba dins de tfg-btc-prediccion/)
├── tfg-btc-prediccion/        ← CODI del projecte
│   ├── db/                     # connexió i esquema PostgreSQL
│   │   ├── db_utils.py         #   get_engine()
│   │   └── create_tables.py
│   ├── pipeline/               # ingesta de dades i feature engineering
│   │   ├── scheduler.py        #   pipeline diari (APScheduler)
│   │   ├── price_fetcher.py / binance_futures_fetcher.py
│   │   ├── rss_fetcher.py / cryptopanic_fetcher.py / load_gdelt_bigquery.py
│   │   ├── sentiment_processor.py     # FinBERT
│   │   ├── feature_builder.py
│   │   └── trading_strategy_backtest.py
│   ├── models/                 # models ML i tests estadístics
│   │   ├── data_loader.py      #   FEATURES, load_dataset()
│   │   ├── xgboost_model.py / xgboost_optuna_lgbm.py
│   │   ├── lstm_model.py / arima_model.py / tree_model.py
│   │   ├── statistical_tests.py
│   │   └── comparativa_modelos.py
│   ├── agente_ia/              # Fase 2 — sistema multi-agent
│   │   ├── 06_agent.py         #   Analista (GPT-4o + 13 eines)
│   │   ├── critic_agent.py     #   Risk Manager
│   │   ├── decision_engine.py  #   Decision Maker (motor de regles)
│   │   ├── rag_pipeline.py     #   RAG amb FAISS
│   │   ├── 03_setup_rag.py     #   construcció de l'índex FAISS
│   │   ├── 05_tools.py         #   13 eines + Confluence Score
│   │   ├── trade_executor.py / bybit_client.py / live_bot.py
│   │   └── 07_app.py / dashboard.py   # dashboard Flask
│   ├── analysis/               # avaluacions Fase 2
│   │   ├── ragas_evaluation.py
│   │   ├── llm_ablation_backtest.py
│   │   └── weight_grid_search.py
│   ├── models/saved/           # models entrenats + índex FAISS
│   ├── results/                # CSV i figures de resultats
│   ├── config/                 # .env (NO versionat) — veure .env.example
│   └── docs/                   # documentació tècnica
│       ├── guia_tecnica.md     #   stack, esquema BD, features
│       ├── RESULTADOS_TFG.md
│       ├── AUDITORIA_FINAL.md
│       ├── CAMBIOS_AGENTE_MULTIAGENTE.md
│       └── DASHBOARD_COMPONENTS.md
├── docs/                       # ENTREGABLES acadèmics
│   ├── memoria/article/        #   article IEEE (LaTeX + PDF)
│   ├── memoria/dossier/        #   figures del dossier
│   └── informes/               #   informes i diari del TFG
└── datasets/                   # dades pesades (NO versionades; es regeneren)
```

---

## 4. Stack tecnològic

- **Python 3.11+**
- **PostgreSQL** (12 taules) — connexió via `db/db_utils.py → get_engine()`
- **ML:** scikit-learn, XGBoost, LightGBM, PyTorch (LSTM), statsmodels (ARIMA), Optuna
- **NLP / sentiment:** `ProsusAI/finbert` (Transformers, PyTorch)
- **RAG:** FAISS + `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Agent:** OpenAI GPT-4o (*function calling* + *streaming*)
- **Execució:** Bybit Demo (*paper trading*)
- **Dashboard:** Flask + TradingView Lightweight Charts

Tot l'*stack* és *open-source*; l'única despesa és l'API d'OpenAI (≈ 30 € en tot el projecte).

---

## 5. Posada en marxa

### 5.1 Requisits previs

- Python 3.11+
- PostgreSQL en execució local
- (Opcional, Fase 2) Claus d'API: OpenAI i Bybit Demo

### 5.2 Instal·lació

```bash
git clone https://github.com/neilpradas/TFG-NEIL-2026.git
cd "TFG-NEIL-2026/tfg-btc-prediccion"

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 5.3 Configuració de secrets

Copia `config/.env.example` a `config/.env` i emplena els valors. El fitxer `.env` està
ignorat per git i **mai** s'ha de versionar.

```bash
cp config/.env.example config/.env
```

### 5.4 Base de dades

```bash
python db/create_tables.py        # crea l'esquema
python pipeline/scheduler.py      # executa el pipeline diari (ingesta + features)
```

---

## 6. Com reproduir els resultats

> Executa des de l'arrel del codi (`tfg-btc-prediccion/`).

### Fase 1 — Models i validació

```bash
python models/xgboost_model.py            # XGBoost (preu + sentiment)
python models/xgboost_optuna_lgbm.py      # XGBoost+Optuna i LightGBM (features esteses)
python models/lstm_model.py               # LSTM (PyTorch)
python models/arima_model.py              # ARIMA baseline
python models/comparativa_modelos.py      # taula comparativa final
```

### Backtest de l'estratègia

```bash
python pipeline/trading_strategy_backtest.py --start 2022-01-01 --end 2024-01-01
```

### Fase 2 — Sistema multi-agent

```bash
python agente_ia/03_setup_rag.py          # construeix l'índex FAISS (RAG)
python agente_ia/07_app.py                # dashboard web (Flask)
python agente_ia/live_bot.py              # bot de paper trading (Bybit Demo)
```

### Avaluacions Fase 2

```bash
python analysis/ragas_evaluation.py       # RAGAS (Faithfulness, etc.)
python analysis/llm_ablation_backtest.py  # ablació del filtre LLM
python analysis/weight_grid_search.py     # pesos del Confluence Score
```

---

## 7. Esquema de dades (resum)

Taula mestra **`daily_features`** (índex `date`, `asset`):

| Grup | Columnes |
|---|---|
| Preu | `close`, `returns`, `label` (1 = puja, 0 = baixa) |
| Tècnics | `rsi_14`, `macd`, `macd_signal`, `bb_upper`, `bb_lower`, `sma_7`, `sma_30` |
| Sentiment | `sentiment_avg`, `sentiment_std`, `sentiment_count` |
| Macro | `fear_greed` (0–100) |

El **volum** viu a `price_data` (OHLCV) i s'incorpora via *join* quan cal.

**Features estàndard** (`models/data_loader.py`):

```python
FEATURES = ['rsi_14', 'macd', 'macd_signal', 'bb_upper', 'bb_lower',
            'sma_7', 'sma_30', 'fear_greed', 'returns', 'sentiment_avg']
TARGET = 'label'
```

---

## 8. Metodologia (principis)

- **Reproductibilitat:** tot el codi i la BD es poden recrear des de zero.
- **Honestitat estadística:** cap resultat es reporta sense interval de confiança i test de
  significança (t-test, *bootstrap*, Wilcoxon, Mann-Whitney).
- **Validació:** *walk-forward* (4 × 90 dies) amb *embargo* de 7 dies entre *train* i *test*;
  `StandardScaler` ajustat només sobre el *train* de cada *split*.
- **Auditoria contínua:** cada resultat «sospitosament bo» es revisava buscant l'error que el
  podia explicar (10 correccions aplicades).

---

## 9. Sistema multi-agent

```
Analista (GPT-4o)  ──▶  Risk Manager  ──▶  Decision Maker  ──▶  Execució
13 eines + RAG          verif. codi +       motor de regles      Bybit Demo
(FAISS, 1.996 docs)     contraarg. LLM      GO / NO-GO           (SL/TP)
```

- **Confluence Score** [−1, +1] amb 6 mòduls ponderats per *grid search*: MTF (0,35),
  ICT (0,25), Technical (0,20), **ML (0,10)**, Smart Money (0,05), Sentiment (0,05).
- **Principi «el codi verifica, el LLM opina»:** tot allò comprovable es resol en codi; el
  LLM només aporta el contraargument adversarial i la lectura qualitativa.

---

## 10. Limitacions

- No s'ha pogut **validar estadísticament la rendibilitat** del sistema final.
- *Backtestejar* un agent basat en LLM introdueix soroll *look-ahead* (coneixement paramètric
  del període) impossible d'eliminar retrospectivament → l'única avaluació neta és la
  **prospectiva** (*paper trading* en viu).
- Potència estadística limitada (4 splits) en la fase predictiva.

---

## 11. Documentació addicional

- **Article IEEE complet:** [`docs/memoria/article/article.pdf`](docs/memoria/article/article.pdf)
- **Guia tècnica:** [`tfg-btc-prediccion/docs/guia_tecnica.md`](tfg-btc-prediccion/docs/guia_tecnica.md)
- **Resultats detallats:** [`tfg-btc-prediccion/docs/RESULTADOS_TFG.md`](tfg-btc-prediccion/docs/RESULTADOS_TFG.md)
- **Auditoria:** [`tfg-btc-prediccion/docs/AUDITORIA_FINAL.md`](tfg-btc-prediccion/docs/AUDITORIA_FINAL.md)

---

## 12. Llicència i autoria

Treball de Fi de Grau acadèmic. © 2026 Neil Pradas Martínez — UAB.
Ús educatiu i de recerca. **Cap part d'aquest projecte constitueix consell financer.**
