# Registro de cambios — Sistema multi-agente + ejecución Bybit + noticias

> Documento de trabajo con todos los cambios de código. Recoge qué se cambió, dónde y por qué.
>
> Fecha inicio: 2026-06-13 · Autor: Neil + asistente
>
> **ACTUALIZACIÓN 2026-06-14:** el `main.tex` (notebooks/dossier_cambios) YA ha sido
> reescrito y reorganizado para reflejar todo esto. Compila correctamente (87 pág.).
> Cambios en el documento:
> - Reorganización: capítulo de Sentiment movido ANTES de ML (flujo datos→sentiment→ML→agente→eval→conclusiones).
> - Capítulo del agente reescrito como sistema multi-agente (Analista+Risk Manager+Decision Maker), gpt-4o, motor determinista; nueva sección "Evolució a un sistema multi-agent".
> - Sección del bot: ejecución real Bybit, trade real ejecutado, break-even+trailing, límites (5/día, sin findes).
> - Abstract (CA+EN) actualizado a multi-agente.
> - Conclusiones: nueva sección "Sobre el sistema d'IA: què funciona i què encara no es pot afirmar" (honesto: funciona a nivel ingeniería y verificado, pero sin significancia estadística por ventana corta + imposibilidad de backtest sin contaminación del LLM).

---

## Contexto / motivación

El sistema pasa de **un solo agente conversacional** (descrito en `main.tex`) a un
**equipo multi-agente determinista** porque da mejores resultados y es más defendible:

1. **Analista** (`06_agent.py`, intacto) — GPT-4o-mini con function calling, 13 tools, RAG.
2. **Risk Manager** (`critic_agent.py`) — auditor adversarial del análisis.
3. **Decision Maker** (`decision_engine.py`, NUEVO) — fusiona analista + risk manager +
   confluence score en una decisión **GO/NO-GO determinista** (motor de reglas como
   fuente de verdad; una 3ª llamada LLM a temp=0 solo redacta, sin poder contradecir
   al motor).

Además se corrige el flujo de **ejecución en Bybit Demo** (antes simulaba en local por
keys vacías) y la **actualización de noticias** en el dashboard.

> ⚠️ **Pendiente TFG:** actualizar `main.tex` (y memoria del proyecto) para reflejar el
> sistema multi-agente. Hoy el documento dice "un solo agente".

---

## Bugs / hallazgos detectados durante el análisis (estado PRE-cambios)

| # | Hallazgo | Evidencia | Impacto |
|---|----------|-----------|---------|
| 1 | RAG no inicializaba | `transformers` 4.51 + TF 2.20 + Keras 3 sin `tf-keras` | RAG caído |
| 2 | Keys Bybit vacías | `BYBIT_DEMO_API_KEY/SECRET=""` en `config/.env` | Bot simulaba en local; trade #4 = `dry_run_...` |
| 3 | Noticias "no se actualizan" | `/api/system/transparency` ordena por influencia, no fecha; `loadNoticias()` sin auto-refresh ni botón | Artículos viejos fijos arriba |
| 4 | `bot_state` sin fila id=1 | `_update_state` hace UPDATE sin INSERT previo | Estado del bot no persiste |
| 5 | Agente y crítico en cajas separadas | Sin decisión final fusionada ni botón de ejecución | — |

---

## Cambios aplicados

### 0. Fix RAG (Keras 3 / tf-keras)  ✅
- **Archivos:** `agente_ia/dashboard.py`, `agente_ia/live_bot.py`
- Se fija `USE_TF=0` / `USE_TORCH=1` (+ `TRANSFORMERS_NO_TF/FLAX`) como **primeras líneas**
  del proceso, antes de cualquier import de HuggingFace. El proyecto usa solo PyTorch, así
  que se desactiva el backend TensorFlow que provocaba el crash.
- Verificado: `semantic_search` devuelve embeddings 384-dim y resultados FAISS.

### A. Noticias — refresco + orden por fecha + scrape on-demand  ✅
- **Backend** (`dashboard.py`):
  - `GET /api/news/latest?order=recent|influence&filter=all|pos|neg&limit&hours` — lee
    `raw_texts` con orden parametrizable (fecha por defecto). Antes el listado salía de
    `/api/system/transparency` ordenado SIEMPRE por influencia (causa del bug "no se
    actualizan").
  - `POST /api/news/scrape` — lanza `run_full_pipeline()` en hilo de fondo (no bloquea),
    con lock anti-concurrencia. `GET /api/news/scrape/status` para polling.
  - Añadido `request` al import global de Flask (antes se importaba local en cada vista).
- **Frontend** (`templates/dashboard.html`):
  - Toggle de orden **Recientes / Influyentes** (default Recientes).
  - Botones **↻ Actualizar** (recarga instantánea) y **⟳ Scrapear ahora** (dispara scrapers
    + spinner + polling + recarga).
  - Auto-refresco cada 60s si la pestaña de noticias está activa.
- **Verificado:** la query ordena por `timestamp DESC` y devuelve las noticias de los
  últimos minutos.

### B. Ejecución real Bybit Demo + fix registro  ✅
- Keys Bybit Demo añadidas y verificadas (balance 165.887 USDT, precio OK, sin posiciones).
- **NUEVO `trade_executor.py`** — ejecutor compartido por `live_bot.py` y el dashboard:
  - `execute_trade(...)`: calcula params (si no vienen), abre en Bybit (real) o simula
    (fallback con prefijo `sim_`), persiste en `live_trades` y actualiza `bot_state`.
  - `ensure_bot_state()`: **arregla el bug #4** (la fila `bot_state id=1` no existía →
    `_update_state` hacía UPDATE sin efecto). Ahora `INSERT ... ON CONFLICT DO NOTHING`.
  - Distinción real/simulado: `bybit_order_id` con prefijo `sim_` = no tocó Bybit.
- **`live_bot.py`**: `evaluate_and_trade` ahora delega los pasos 4-7 en `trade_executor`
  (DRY, una sola lógica de ejecución). Siembra `bot_state` al arrancar.
- **`dashboard.py`**: `POST /api/bot/execute` — botón manual "Ejecutar en Bybit Demo".
  Gating en SERVIDOR: solo ejecuta si la última decisión fue GO y no caducó (<300s);
  recalcula params frescos; usa `trade_executor`. Verificado: sin decisión → 409.

### C. Decisión final fusionada (Decision Maker determinista)  ✅
- **NUEVO `decision_engine.py`**:
  - `decide(confluence, critic, trade_params)` — **motor de reglas determinista** (fuente
    de verdad). NO_GO si: score neutro · |score|<0.30 · conflict=true · dead zone
    (22:00–00:00 UTC) · crítico=REJECT · R:R<1.5. Avisos (no bloquean): crítico=CAUTION,
    cifras sin cita, confianza<0.50. Umbrales centralizados en `RULES`.
  - `synthesize(...)` — **3ª llamada LLM (Decision Maker, temp=0)** que redacta el
    veredicto; el GO/NO_GO se le pasa decidido y NO puede cambiarlo. Fallback textual
    determinista si falla. Verificado con 3 casos (GO / REJECT→NO_GO / débil+conflict→NO_GO).
- **`dashboard.py` `/api/agent/stream`**: tras analista + crítico, calcula la decisión,
  la guarda en servidor (`_last_decision`, para gatear ejecución) y emite evento SSE
  `decision`.
- **Frontend** (`dashboard.html`): banner final **GO/NO_GO** con niveles (entry/SL/TP/RR)
  y botón **▶ Ejecutar en Bybit Demo** (deshabilitado si NO-GO).

---

### D. Afinado del Risk Manager — falsos positivos  ✅
- **`critic_agent.py`** (`CRITIC_SYSTEM_PROMPT`): el crítico inventaba problemas:
  - Marcaba como "confusión" que el precio del analista COINCIDIERA con el de
    referencia (siendo idéntico → es lo correcto).
  - Marcaba como "incoherencia" que el analista decidiera NO operar pese a un score
    BUY/SELL (cuando declinar por R:R bajo / volumen débil es conservador y correcto;
    el Decision Maker daba NO_GO igualmente).
- Cambios en el prompt:
  - El precio de referencia ES el precio actual: solo señalar discrepancias REALES
    (números distintos), nunca coincidencias.
  - CITAS: marcar solo cifras NUEVAS sin fuente que no coincidan con el ground truth.
  - COHERENCIA: incoherente SOLO si propone trade en dirección OPUESTA al score;
    decidir esperar es válido.
  - Sección "QUÉ NO ES UN PROBLEMA" + instrucción de devolver `problemas: []` y usar
    APPROVE cuando no haya problemas reales (no CAUTION por costumbre).
- **Verificado** con el caso real del usuario (precio coincidente + "no operar" con
  score BUY): antes CAUTION con 3 problemas falsos → ahora **APPROVE, problemas: []**.
- Nota de despliegue: el dashboard cachea `critic_agent` en `sys.modules` (vía `_load`),
  por lo que requiere **reinicio** para aplicar el prompt nuevo.

### D bis. Risk Manager — falsos positivos persistentes (refuerzo)  ✅
- El prompt solo no bastaba: gpt-4o-mini seguía generando "el precio X es diferente
  del de referencia Y" con X≈Y (incluso REJECT 100%). Causa: el modelo compara
  mecánicamente cada precio con la referencia, aunque por diseño SIEMPRE coinciden.
- **Doble defensa en `critic_agent.py`**:
  1. Prompt: prohibición explícita de comparar precios ("PROHIBIDO comparar precios…
     es un FALSO POSITIVO automático"); se elimina el check de niveles vs referencia.
  2. **Filtro determinista en código** (`_is_false_price_problem`): descarta cualquier
     `problema` que mencione "referencia" o que afirme una "diferencia" cuyas cifras de
     precio (>1000) sean ~iguales (<0.05%). Si tras filtrar no queda ningún problema y
     el veredicto era REJECT/CAUTION, se **degrada a APPROVE** (no puede haber veredicto
     adverso sin problema real).
- **Verificado**: filtro unitario (descarta los 2 problemas del usuario, conserva
  volumen/dirección-opuesta/R:R) + review completa con LLM → CAUTION con 1 problema
  legítimo (volumen), 0 falsos positivos de precio.

### E. Rediseño profesional del Risk Manager (separación verificación/opinión)  ✅
Motivo: gpt-4o-mini seguía inventando falsos positivos pese a prompt + filtros
(cambió "diferente de referencia" → "(debería ser $X)", tautología con cifras
iguales). Causa raíz: pedíamos a un LLM flojo que hiciera verificación determinista
de números/citas Y juicio cualitativo a la vez. Decisión del usuario: gpt-4o en
crítico y decision maker; verificación de precio determinista; citas híbridas.

Rediseño de `critic_agent.py` (separación estricta):
- **VERIFICACIÓN → 100% código** (`_deterministic_problems`): veredicto
  (APPROVE/CAUTION/REJECT) y `problemas` salen SOLO de reglas sobre el ground truth
  (conflict, |score|<0.30, dead zone 22-00 UTC, R:R<1.5, uso de ML como señal) +
  **verificación de citas híbrida** (`_current_price_mismatch`): solo se comprueba
  el precio actual citado vs el de mercado (lo único con ground truth claro); SL/TP/
  niveles/volumen NO se policían (son precios legítimamente distintos). Resultado:
  imposible generar el falso positivo tautológico.
- **OPINIÓN → LLM gpt-4o** (`_llm_opinion`): SOLO redacta el contraargumento
  adversarial + nivel de riesgo (bajo/medio/alto). Tiene PROHIBIDO verificar/citar
  números. Si falla, el veredicto determinista se mantiene.
- `review_analysis(...)` nueva firma: acepta `trade_params=` y `model='gpt-4o'`.
- `decision_engine.synthesize`: modelo por defecto → **gpt-4o**.
- `dashboard.py` stream: calcula confluence + trade_params ANTES del crítico y se los
  pasa; render del crítico muestra badge de riesgo y ya no muestra el aviso de "cifras
  sin cita" (redundante; el mismatch real, si lo hay, aparece como problema).
- **Verificado**: caso del usuario (precio coincidente + volumen débil) → problemas []
  + APPROVE; precio realmente mal citado ($63.000 vs $64.266) → lo detecta; review con
  gpt-4o → contraargumento real sobre volumen/estructura, 0 ruido numérico.

### F. Todos los modelos de chat → gpt-4o  ✅
- Cambiado a `gpt-4o` en: `06_agent.py` (chat y chat_stream, default → lo usa también
  el live_bot vía chat()), `dashboard.py` (stream del analista + metadata UI),
  `ask_trade.py`. Crítico y Decision Maker ya estaban en gpt-4o.
- **Coste medido (gpt-4o, $2.50/1M in · $10/1M out):**
  - Analista, llamada ligera medida: 14.302 in + 513 out ≈ **$0.041**.
  - Análisis completo en dashboard (analista 4-6 tools + crítico + decision):
    **≈ $0.06–0.14** por análisis (con gpt-4o-mini era ~$0.005–0.01).
  - Live bot: 1 llamada analista por decisión, máx 15/día → hasta ~$0.6–1.5/día.

### G. Límites del bot + fix tamaño real Bybit  ✅
- `live_bot.py LIVE_CONFIG`: `max_agent_calls_day` 15 → **5**; nuevo `skip_weekends: True`
  (guard en `should_call_agent`: sábado/domingo UTC no llama al agente).
- **Primer trade REAL en Bybit** (#5, order UUID): LONG 64.213,7 → cerrado manualmente a
  64.587,9 = **neto +$131,14 (+0,58%)**. bot_state capital 50.000 → 50.131,14.
- **Bug corregido (`trade_executor.py`)**: guardaba `size_btc` sin dividir por leverage
  (1,294) mientras Bybit ejecutaba 0,432 (÷3) → el PnL de posición abierta se mostraba
  ~3× inflado en el dashboard. Ahora, en ejecución real, se guarda el `size_btc` real que
  devuelve Bybit.

### H. Asegurar beneficio (breakeven + trailing) + botón cerrar + bot pausado  ✅
- **Bot pausado** (a petición del usuario, para refinar sin gastar API). Reactivar:
  pestaña Bot → "Iniciar Bot" o `python agente_ia/live_bot.py`.
- **Breakeven + trailing del SL NATIVO** (`live_bot.manage_open_positions` + config
  `secure_be_pct=0.5`, `trail_dist_pct=0.4`, `fee_buffer_pct=0.12`):
  - A +0.5% de beneficio → SL a breakeven (entry+fees): la posición ya no puede perder.
  - Luego el SL trepa a 0.4% por detrás del precio; solo MEJORA, nunca empeora.
  - Se escribe en Bybit vía `bybit_client.update_stop_loss()` (set-trading-stop), así
    protege **aunque el bot se apague**. Al retroceder, el SL nativo cierra y asegura.
  - Simulado: entry 64000 → asegura ~+1.31% en un movimiento a +1.72% con retroceso.
- **Botón "Cerrar posiciones"** en la pestaña Bot (usa `/api/bot/close-all`).
- **`cmd_close_all` reescrito**: ahora registra PnL realizado con el tamaño REAL de
  Bybit + actualiza capital (antes solo marcaba 'closed' sin PnL). Ya no detiene el bot
  al cerrar (cerrar ≠ parar).

## Arquitectura final del equipo multi-agente (para el TFG)

```
Usuario / trigger
      │
      ▼
[1] ANALISTA (06_agent.py) ── 13 tools + RAG ──▶ análisis ICT (prosa, streaming)
      │
      ▼
[2] RISK MANAGER (critic_agent.py) ── audita ──▶ veredicto APPROVE/CAUTION/REJECT
      │
      ▼
[3] DECISION MAKER (decision_engine.py)
      ├─ decide()      → motor de reglas DETERMINISTA → GO / NO_GO  (fuente de verdad)
      └─ synthesize()  → LLM temp=0 redacta el veredicto (no puede cambiar el GO/NO_GO)
      │
      ▼
   Botón "Ejecutar" (gateado: solo si GO) ──▶ trade_executor.execute_trade()
                                               ├─ Bybit Demo (real) o sim (fallback)
                                               └─ registra en live_trades + bot_state
```

## Pendiente / acciones manuales
- **Reiniciar** `dashboard.py` y `live_bot.py` (ambos corren con código viejo cacheado;
  los cambios y las keys de Bybit solo se aplican al reiniciar).
- El **trade #4** (`dry_run_1781366734`) sigue abierto como simulación local (pre-keys).
  Decidir: dejarlo, o cerrarlo (`python agente_ia/live_bot.py --close-all`) y empezar
  limpio con ejecución real.
- **Actualizar `main.tex`** con esta arquitectura multi-agente (hoy dice "un solo agente").
