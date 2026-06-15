"""
dashboard.py — Servidor Flask del dashboard BTC
Uso: python agente_ia/dashboard.py  →  http://localhost:5050
"""
# ── CRÍTICO: desactivar TF/Keras ANTES de cualquier import de HuggingFace ──
# transformers detecta tensorflow 2.20 (+ keras 3) e intenta cargar el backend
# TF, que requiere tf-keras (no instalado) → crash y "RAG no inicializado".
# El proyecto usa solo PyTorch, así que se desactiva TF a nivel de proceso.
import os as _os
_os.environ.setdefault("USE_TF", "0")
_os.environ.setdefault("USE_TORCH", "1")
_os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
_os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import importlib.util, json, logging, sys, threading, time
from datetime import datetime, timezone
from pathlib import Path

import requests as _req
from flask import Flask, jsonify, request, Response, render_template, stream_with_context

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(_ROOT / 'config' / '.env')

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)

app = Flask(__name__, template_folder=str(Path(__file__).parent / 'templates'))


def _load(name, filepath):
    # Reutilizar si ya está cargado: 06_agent importa tools.py, que comparte
    # la misma instancia "tools05" — así el precio live sembrado aquí es el
    # mismo que ven las tools del agente.
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_tools    = _load("tools05",      Path(__file__).parent / "05_tools.py")
_agent    = _load("agent06",      Path(__file__).parent / "06_agent.py")
_pipeline = _load("run_pipeline", _ROOT / "agente_ia" / "news" / "run_pipeline.py")

# ── System prompt ICT completo ─────────────────────────────────────────────────
_agent.SYSTEM_PROMPT = """Eres un trader profesional especializado en Bitcoin intraday usando metodología ICT.

═══════════════════════════════════════════════════════
PRINCIPIO FUNDAMENTAL (CRÍTICO)
═══════════════════════════════════════════════════════
NO inventes números. NO afirmes nada sin tool support.
Si una tool no devuelve un dato, di "datos insuficientes para [X]".
Toda cifra concreta DEBE provenir de una tool o de rag_search.

RELEVANCIA DE RESPUESTA (obligatorio):
Comienza tu respuesta SIEMPRE con una frase que repita o reformule
la pregunta del usuario. Ejemplos:
  Pregunta: "¿Hay setup long ahora?"
  Inicio:   "<h3>Setup LONG en BTC ahora</h3><p>Analizo si hay setup long válido..."
  Pregunta: "¿Cómo está el sentimiento?"
  Inicio:   "<h3>Sentimiento de noticias BTC</h3><p>Reviso el sentimiento FinBERT reciente..."
  Pregunta: "¿Qué predice el modelo ML?"
  Inicio:   "<h3>Predicción del modelo ML</h3><p>Consulto el ensemble ML para BTC..."

USO OBLIGATORIO DE TOOLS:
- Si la pregunta menciona "noticias", "noticia", "ETF", "regulación", "Fed", "macro",
  "halving", "evento", "tipo de interés", "inflación", "CPI", "DXY", "dólar",
  "BlackRock", "institucional", "adopción", "hackeo", "exploit", "exchange"
  → DEBES llamar rag_search PRIMERO (antes de cualquier otra tool).
- Si la pregunta menciona "ML", "modelo", "predicción", "probabilidad", "ensemble",
  "AUC", "confianza estadística"
  → DEBES llamar run_ml_prediction Y get_confluence_score.
- Si la pregunta menciona "setup", "entrada", "stop", "TP", "target", "sizing"
  → DEBES llamar get_confluence_score, get_entry_zone y get_trade_parameters.
- Si la pregunta menciona "sentimiento", "FinBERT", "fear greed"
  → DEBES llamar get_sentiment Y get_fear_greed.

═══════════════════════════════════════════════════════
SESIONES DE MERCADO (UTC)
═══════════════════════════════════════════════════════
Asia:          00:00–08:00  → Rango bajo, evitar salvo noticias
London Open:   07:00–09:30  ★ KILLZONE 1 — mayor probabilidad
London Close:  11:00–12:00  → Trampas y reversals frecuentes
NY Open:       13:00–14:30  ★ KILLZONE 2 — mayor volumen del día
NY Session:    13:00–22:00  → Operativo con filtros
Dead Zone:     22:00–00:00  ✗ NUNCA operar

═══════════════════════════════════════════════════════
MÉTODO DE TRADE (ICT + Volumen + Confluencia)
═══════════════════════════════════════════════════════

PASO 1 — HTF Bias (4h y 1d, llamar get_multi_timeframe_bias)
  BULLISH: EMA9>EMA21>EMA50, BOS alcistas recientes, precio sobre VWAP
  BEARISH: EMA9<EMA21<EMA50, BOS bajistas recientes, precio bajo VWAP
  Si 4h y 1d divergen → RANGING, reducir tamaño o no operar

PASO 2 — LTF Structure (1h, llamar get_ict_context)
  La estructura 1h debe CONFIRMAR el HTF bias:
  - CHoCH 1h en dirección del HTF bias = señal de entrada próxima
  - BOS 1h en dirección del HTF bias = setup activo
  - Si estructura 1h opuesta al HTF = esperar

PASO 3 — Zona de entrada (llamar get_technical_levels)
  LONG: precio retrocede a OB bullish o FVG bullish (por debajo del precio)
    → Entry: 50% del OB o top del FVG
  SHORT: precio retrocede a OB bearish o FVG bearish (por encima del precio)
    → Entry: 50% del OB o bottom del FVG

PASO 4 — Confirmación de volumen (llamar get_volume_profile)
  vol_ratio_vs_hour_avg > 1.0  → confirma el movimiento
  vol_ratio_vs_hour_avg < 0.7  → trampa, NO entrar

PASO 5 — Risk Management
  SL: bajo el low del OB bullish (longs) / sobre el high del OB bearish (shorts)
  SL máximo: 1.5% del precio
  R:R mínimo: 1.5:1
  T1: siguiente FVG o swing level (≥1:1)
  T2: HTF swing high/low (≥2:1)

═══════════════════════════════════════════════════════
SCORING DE CONFLUENCIA (-1 a +1) — llamar get_confluence_score
═══════════════════════════════════════════════════════
SIEMPRE llamar get_confluence_score ANTES de dar cualquier setup.
El score es matemático (-1 bear fuerte, 0 neutral, +1 bull fuerte).
Pesos OPTIMIZADOS por grid search empírico (Trend Following config ganadora):
  Técnico 20% · ICT 25% · MTF 35% · SmartMoney 5% · Sentimiento 5% · ML 10%

Score ≥ +0.60 → SETUP FUERTE — tamaño máximo
Score +0.30 a +0.59 → SETUP VÁLIDO — entrar con gestión estándar
Score +0.10 a +0.29 → SETUP DÉBIL — reducir tamaño o esperar confirmación
Score -0.09 a +0.09 → NO TRADE — neutral, falta confluencia
Score ≤ -0.10 → NO TRADE (bias bajista — sólo shorts si se confirma)

Interpretar campo "conflict": si true, hay conflicto sentimiento vs estructura — añadir advertencia.

═══════════════════════════════════════════════════════
INDICADORES TÉCNICOS — INTERPRETACIÓN
═══════════════════════════════════════════════════════
RSI 1h:
  >75: extremo sobrecomprado, evitar longs, buscar short en OB/FVG
  65-75: sobrecomprado, caution en longs
  35-65: zona neutral, operar según estructura
  25-35: sobrevendido, caution en shorts
  <25: extremo sobrevendido, evitar shorts, buscar long en OB/FVG

MACD Hist:
  Cruzando 0 al alza + volumen = momentum alcista confirmado
  Cruzando 0 a la baja + volumen = momentum bajista confirmado

BB Position (0-1):
  >0.85: precio cerca de BB upper → posible rechazo bajista
  <0.15: precio cerca de BB lower → posible rebote alcista

Volumen ratio:
  >1.5: spike de volumen → confirma breakout o reversión
  <0.8: sin convicción → desconfiar del movimiento

═══════════════════════════════════════════════════════
MODELO ML (ensemble GBM+RF+Logistic intradía, post-auditoría de leakage)
═══════════════════════════════════════════════════════
AUC honesto walk-forward: 0.5185 ± 0.055 (5 folds) — NO supera el azar con
significancia estadística. Una versión previa reportaba AUC 0.728, pero era
look-ahead bias (order blocks calculados con velas futuras) y fue DESCARTADA.
El modelo desplegado está marcado valid=false: run_ml_prediction devolverá
"model_not_significant". En ese caso NO uses ML como señal, decláralo
explícitamente ("ML no significativo, excluido del análisis") y basa el
análisis en estructura, volumen y sentimiento.
Si en el futuro valid=true:
  probability < 0.35: señal bajista fuerte | 0.35-0.50: bajista moderado
  probability 0.50-0.65: alcista moderado  | > 0.65: alcista fuerte

═══════════════════════════════════════════════════════
COMUNICACIÓN DE INCERTIDUMBRE (obligatorio)
═══════════════════════════════════════════════════════
En cada respuesta debes hacer EXPLÍCITA tu incertidumbre:

1. Confianza del setup (campo "confidence" en get_confluence_score):
   - >0.80 → "Alta confianza: módulos altamente alineados"
   - 0.50–0.80 → "Confianza media: mayoría alineada, algunas discrepancias"
   - <0.50 → "Baja confianza: módulos divididos, posible whipsaw"

2. Si conflict=true en confluence_score → SIEMPRE advertir y reducir tamaño sugerido a la mitad

3. Si una tool falla o devuelve datos insuficientes → declararlo, no inventar

4. Si rag_search no devuelve resultados relevantes → decir "sin contexto news reciente sobre este tema"

5. Limitaciones del sistema (recordar):
   - Backtest 90d: +8.27% Sharpe 1.55. Backtest 365d: -1.12% (mercado -42%). N pequeña.
   - Killzones simples sin filtro de contexto (whipsaws documentados)
   - ML post-auditoría: AUC 0.518 (no significativo, valid=false). El AUC 0.728
     previo era leakage y fue descartado — no lo cites como vigente.

═══════════════════════════════════════════════════════
USO DE RAG (rag_search) — corpus de noticias
═══════════════════════════════════════════════════════
LLAMA rag_search en estos casos:
- Pregunta sobre noticias específicas o eventos recientes
- Necesitas context para interpretar movimientos de precio inusuales
- Pregunta sobre regulación, ETF, Fed, halving, etc.

CITA siempre fuentes recuperadas con su source y timestamp.
Si rag_search devuelve resultados con similarity < 0.4, declara que la búsqueda fue débil.

NOTA: el corpus tiene 1996 noticias indexadas con FAISS (cosine similarity).

═══════════════════════════════════════════════════════
REGLAS ABSOLUTAS
═══════════════════════════════════════════════════════
✗ NO operar en dead zone (22:00–00:00 UTC)
✗ NO operar sábado/domingo con volume_ratio < 0.5
✗ NO entrar si R:R < 1.5:1
✗ NO entrar si vol_ratio_vs_hour < 0.7
✗ NO entrar 30min antes/después de eventos macro HIGH impact
✓ SIEMPRE calcular el score de confluencia antes de dar setup

═══════════════════════════════════════════════════════
HERRAMIENTAS DISPONIBLES (llamar TODAS)
═══════════════════════════════════════════════════════
1.  query_market(timeframe)     → precio, RSI, MACD, BB, EMAs, VWAP, ATR
2.  run_ml_prediction           → dirección ML con probabilidad
3.  rag_search(query)           → noticias relevantes
4.  get_sentiment               → FinBERT sentiment días recientes
5.  get_fear_greed              → Fear & Greed
6.  get_ict_context(timeframe)  → OBs, FVGs, BOS, CHoCH, swings, estructura
7.  get_session_stats           → sesión, killzone, HOD/LOD, rango
8.  get_technical_levels        → niveles cercanos ordenados con R:R
9.  get_multi_timeframe_bias    → confluencia 1h/4h/1d
10. get_volume_profile          → confirmación de volumen por hora y sesión
11. get_trade_parameters        → sizing, fees, slippage, break-even, compounding
12. get_confluence_score        → score cuantitativo final -1..+1 (LLAMAR SIEMPRE)
13. get_entry_zone(direction)   → zona exacta de entrada con OB/FVG más cercanos

═══════════════════════════════════════════════════════
CITACIÓN OBLIGATORIA — para RAGAS Faithfulness
═══════════════════════════════════════════════════════
DESPUÉS de cada claim numérico o factual, escribe entre paréntesis la tool
o source de donde sale el dato. Ejemplos:
  - "RSI 1h en 67.3 [query_market]"
  - "Sentiment 7d: +0.12 [get_sentiment]"
  - "Fear &amp; Greed 42 — Neutral [get_fear_greed]"
  - "Confluence score +0.34 BUY [get_confluence_score]"
  - "ML probability 0.58 bullish [run_ml_prediction]"
  - "ETF outflows $4B [rag_search: newsbtc, 2026-05-30]"
  - "OB bullish 67200–67450 a 0.4% [get_ict_context]"

Si una afirmación NO viene de una tool ni de rag_search:
  → escribe "(sin fuente directa)" o reformula con "típicamente" /
    "en general" / "históricamente" para que el judge LLM sepa que
    es conocimiento general, no un claim verificable.

REGLA ANTI-ALUCINACIÓN: si una tool devuelve error o datos vacíos,
NO inventes el dato — escribe "datos insuficientes para [campo]
([tool] sin respuesta)". Esto es preferible a un número inventado.

Esto es CRÍTICO para que el judge LLM pueda verificar tus claims
contra los contextos recuperados.

═══════════════════════════════════════════════════════
FORMATO DE RESPUESTA (HTML estricto, sin emojis)
═══════════════════════════════════════════════════════

Usa EXACTAMENTE estas etiquetas HTML. Sin emojis. Sin texto plano fuera de etiquetas.

<h3>Situacion Actual</h3>
<p>Precio, sesión, estructura 1h/4h/1d en 2-3 líneas. Usa <span class="bull">alcista</span> o <span class="bear">bajista</span>.</p>

<h3>Indicadores Tecnicos</h3>
<table>
<tr><th>TF</th><th>RSI</th><th>MACD hist</th><th>BB pos</th><th>EMA trend</th><th>Vol ratio</th><th>VWAP</th></tr>
<tr><td>1H</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td></tr>
<tr><td>4H</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td></tr>
</table>

<h3>Contexto ICT (1H)</h3>
<p>Market structure | BOS/CHoCH reciente | OBs activos (precio y distancia) | FVGs sin mitigar.</p>

<h3>Perfil de Volumen</h3>
<p>POC, VAH, VAL | CVD divergencias | Wyckoff fase estimada.</p>

<h3>Sentimiento y Macro</h3>
<p>Fear &amp; Greed: XX (clasificacion). FinBERT 7d: +/-X.XX. Bias institucional. Eventos macro proximos.</p>

<h3>Modelo ML</h3>
<p>Direccion: LONG/SHORT. Probabilidad: X.XX. Top features que activaron la señal.</p>

<h3>Score de Confluencia</h3>
<p>Score: <strong>±X.XXX</strong> — LABEL (get_confluence_score). Confianza: XX%. Modulos alineados: X/6.</p>
<table>
<tr><th>Modulo</th><th>Score</th><th>Peso</th><th>Contribucion</th></tr>
<tr><td>Tecnico</td><td>±X.XXX</td><td>20%</td><td>±X.XXX</td></tr>
<tr><td>ICT</td><td>±X.XXX</td><td>25%</td><td>±X.XXX</td></tr>
<tr><td>MTF</td><td>±X.XXX</td><td>35%</td><td>±X.XXX</td></tr>
<tr><td>SmartMoney</td><td>±X.XXX</td><td>5%</td><td>±X.XXX</td></tr>
<tr><td>Sentimiento</td><td>±X.XXX</td><td>5%</td><td>±X.XXX</td></tr>
<tr><td>ML</td><td>±X.XXX</td><td>10%</td><td>±X.XXX</td></tr>
</table>

<h3>Setup</h3>
<div class="setup-box valid">
  <p><strong>VALIDO — Score +X.XX (LABEL)</strong></p>
  <table>
  <tr><td>Direccion</td><td><span class="bull">LONG</span></td></tr>
  <tr><td>Entry</td><td>$XX,XXX — condicion exacta</td></tr>
  <tr><td>Stop Loss</td><td><span class="bear">$XX,XXX</span> — razon</td></tr>
  <tr><td>Target 1</td><td><span class="bull">$XX,XXX</span> — +X.XX% | R:R X:1</td></tr>
  <tr><td>Target 2</td><td><span class="bull">$XX,XXX</span> — +X.XX% | R:R X:1</td></tr>
  <tr><td>Size (1% riesgo)</td><td>$XXX en BTC</td></tr>
  <tr><td>Break-even</td><td>$XX,XXX — fees totales $X.XX</td></tr>
  <tr><td>Timing</td><td>killzone + condicion</td></tr>
  </table>
</div>

O si no hay setup:
<div class="setup-box invalid">
  <p><strong>NO VALIDO — Score ±X.XX (LABEL)</strong></p>
  <p>Razon principal. Que esperar para que se active.</p>
</div>

<h3>Invalidacion</h3>
<ul>
<li>Condicion 1 con precio exacto</li>
<li>Condicion 2 con precio exacto</li>
<li>Condicion 3 con precio exacto</li>
</ul>

REGLAS DE FORMATO:
- NUNCA uses emojis
- NUNCA escribas texto plano fuera de etiquetas HTML
- Usa <span class="bull"> para valores alcistas y <span class="bear"> para bajistas
- Todas las tablas deben tener <th> en la primera fila
- El div setup-box debe tener clase "valid" o "invalid" segun corresponda
- Responde en español
"""

from db.db_utils import get_engine
from sqlalchemy import text

BINANCE_REST = "https://api.binance.com/api/v3"
BINANCE_WS   = "wss://stream.binance.com:9443/ws/btcusdt@ticker"


# ══════════════════════════════════════════════════════════════════════════════
# PRECIO EN TIEMPO REAL — WebSocket Binance
# ══════════════════════════════════════════════════════════════════════════════

_live_price = {
    "price":      None,
    "change_24h": None,
    "high_24h":   None,
    "low_24h":    None,
    "volume_24h": None,
    "bid":        None,
    "ask":        None,
    "spread":     None,
    "timestamp":  None,
    "source":     "initializing",
    "connected":  False,
}
_price_lock = threading.Lock()


def _on_ws_message(ws, message):
    try:
        d = json.loads(message)
        bid = float(d.get("b", 0) or 0)
        ask = float(d.get("a", 0) or 0)
        px  = float(d.get("c", 0) or 0)
        # Sembrar el precio en las tools del agente — misma fuente para todos
        if px > 0:
            try:
                _tools.set_live_price(px, "binance_ws")
            except Exception:
                pass
        with _price_lock:
            _live_price.update({
                "price":      px,
                "change_24h": float(d.get("P", 0) or 0),
                "high_24h":   float(d.get("h", 0) or 0),
                "low_24h":    float(d.get("l", 0) or 0),
                "volume_24h": float(d.get("v", 0) or 0),
                "bid":        bid,
                "ask":        ask,
                "spread":     round(ask - bid, 2) if ask and bid else None,
                "timestamp":  d.get("E"),
                "source":     "binance_ws",
                "connected":  True,
            })
    except Exception as e:
        log.warning("WS message parse error: %s", e)


def _on_ws_error(ws, error):
    log.warning("WS error: %s", error)
    with _price_lock:
        _live_price["connected"] = False


def _on_ws_close(ws, *args):
    with _price_lock:
        _live_price["connected"] = False
    log.info("WS closed, reconnecting in 3s...")
    threading.Timer(3.0, _start_price_ws).start()


def _on_ws_open(ws):
    log.info("Binance WS connected")
    with _price_lock:
        _live_price["connected"] = True


def _start_price_ws():
    try:
        import websocket as ws_lib
        wsa = ws_lib.WebSocketApp(
            BINANCE_WS,
            on_message=_on_ws_message,
            on_error=_on_ws_error,
            on_close=_on_ws_close,
            on_open=_on_ws_open,
        )
        wsa.run_forever(ping_interval=20, ping_timeout=10)
    except Exception as e:
        log.error("WS start failed: %s", e)
        threading.Timer(5.0, _start_price_ws).start()


# Arrancar WebSocket en hilo daemon
_ws_thread = threading.Thread(target=_start_price_ws, daemon=True, name="btc-ws")
_ws_thread.start()


# ── REST fallback para obtener bid/ask ────────────────────────────────────────
def _fetch_book_ticker():
    try:
        r = _req.get(f"{BINANCE_REST}/ticker/bookTicker",
                     params={"symbol": "BTCUSDT"}, timeout=3)
        d = r.json()
        bid = float(d["bidPrice"])
        ask = float(d["askPrice"])
        with _price_lock:
            if not _live_price["bid"]:
                _live_price["bid"]    = bid
                _live_price["ask"]    = ask
                _live_price["spread"] = round(ask - bid, 2)
    except Exception:
        pass

threading.Thread(target=_fetch_book_ticker, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
# APSCHEDULER — Actualizar btc_ohlcv cada 15 min
# ══════════════════════════════════════════════════════════════════════════════

_ohlcv_update_lock = threading.Lock()
_last_ohlcv_update = {}  # tf -> epoch del último update OK


def _update_ohlcv_job(timeframes=None, min_interval_s=0):
    """
    Descarga las últimas velas de Binance y actualiza btc_ohlcv.
    timeframes: limitar a esos TFs (None = todos).
    min_interval_s: si todos los TFs pedidos se actualizaron hace menos de
    este tiempo, no hace nada (evita updates redundantes al llamar al agente).
    """
    tfs = tuple(timeframes) if timeframes else ('5m', '15m', '1h', '4h', '1d')
    now = datetime.now(timezone.utc).timestamp()
    if min_interval_s and all(
        now - _last_ohlcv_update.get(tf, 0) < min_interval_s for tf in tfs
    ):
        return {'status': 'skipped_fresh'}

    if not _ohlcv_update_lock.acquire(timeout=60):
        return {'status': 'busy'}
    try:
        mod = _load("price_updater02", Path(__file__).parent / "02_price_updater.py")
        result = mod.update_ohlcv(timeframes=tfs)
        done = datetime.now(timezone.utc).timestamp()
        for tf in tfs:
            _last_ohlcv_update[tf] = done
        log.info("OHLCV update job %s: %s", tfs, result)
        return result
    except Exception as e:
        log.error("OHLCV update job failed: %s", e)
        return {'status': 'error', 'error': str(e)}
    finally:
        _ohlcv_update_lock.release()


def _update_5m_job():
    """Actualiza solo 5m y 15m cada 5 minutos para entradas precisas."""
    _update_ohlcv_job(timeframes=('5m', '15m'))


try:
    from apscheduler.schedulers.background import BackgroundScheduler
    _scheduler = BackgroundScheduler()
    _scheduler.add_job(_update_ohlcv_job, 'interval', minutes=15, id='ohlcv_update')
    _scheduler.add_job(_update_5m_job,    'interval', minutes=5,  id='ohlcv_5m_update')
    _scheduler.start()
    log.info("APScheduler iniciado — 1h/4h/1d cada 15 min, 5m/15m cada 5 min")
except Exception as e:
    log.warning("APScheduler no disponible: %s", e)
    _scheduler = None


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/api/live')
def api_live():
    with _price_lock:
        data = dict(_live_price)

    if not data.get("price"):
        # Fallback REST si el WS aún no conectó
        try:
            r = _req.get(f"{BINANCE_REST}/ticker/24hr",
                         params={"symbol": "BTCUSDT"}, timeout=4)
            d = r.json()
            book = _req.get(f"{BINANCE_REST}/ticker/bookTicker",
                            params={"symbol": "BTCUSDT"}, timeout=3).json()
            data = {
                "price":      float(d["lastPrice"]),
                "change_24h": float(d["priceChangePercent"]),
                "high_24h":   float(d["highPrice"]),
                "low_24h":    float(d["lowPrice"]),
                "volume_24h": float(d["volume"]),
                "bid":        float(book["bidPrice"]),
                "ask":        float(book["askPrice"]),
                "spread":     round(float(book["askPrice"]) - float(book["bidPrice"]), 2),
                "source":     "binance_rest_fallback",
                "connected":  False,
            }
        except Exception as e:
            return jsonify({"error": str(e)}), 503

    # Compatibilidad con frontend (change_pct alias)
    data["change_pct"]  = data.get("change_24h")
    data["volume_usdt"] = data.get("volume_24h")
    data["ts"] = datetime.now(timezone.utc).isoformat()
    return jsonify(data)


@app.route('/')
def index():
    return render_template('dashboard.html')


@app.route('/api/market')
def api_market():
    data = {}
    for tf in ('1h', '4h', '1d'):
        data[tf] = _tools.dispatch_tool('query_market', {'timeframe': tf, 'candles': 2})
    return jsonify(data)


@app.route('/api/ohlcv/<timeframe>')
def api_ohlcv(timeframe):
    tf = timeframe if timeframe in ('1h', '4h', '1d') else '1h'
    limit = {'1h': 120, '4h': 100, '1d': 90}.get(tf, 100)
    engine = get_engine()
    sql = text("SELECT timestamp, open, high, low, close, volume FROM btc_ohlcv "
               "WHERE timeframe=:tf ORDER BY timestamp DESC LIMIT :n")
    with engine.connect() as conn:
        rows = conn.execute(sql, {'tf': tf, 'n': limit}).fetchall()
    candles = [{'time': int(r[0].timestamp()), 'open': float(r[1]), 'high': float(r[2]),
                'low': float(r[3]), 'close': float(r[4]), 'volume': float(r[5])}
               for r in reversed(rows)]
    return jsonify(candles)


@app.route('/api/indicators')
def api_indicators():
    engine = get_engine()
    result = {}
    for tf in ('1h', '4h'):
        sql = text("""
            SELECT timestamp, close, rsi_14, macd, macd_signal, macd_hist,
                   bb_upper, bb_lower, bb_mid, ema_9, ema_21, ema_50, ema_200,
                   atr_14, volume, volume_ratio, vwap, swing_high, swing_low,
                   bos_bullish, bos_bearish, ob_bullish, ob_bearish,
                   fvg_bullish, fvg_bearish, session, is_killzone
            FROM btc_ohlcv WHERE timeframe=:tf ORDER BY timestamp DESC LIMIT 80
        """)
        with engine.connect() as conn:
            rows = conn.execute(sql, {'tf': tf}).fetchall()
        cols = ['timestamp','close','rsi_14','macd','macd_signal','macd_hist',
                'bb_upper','bb_lower','bb_mid','ema_9','ema_21','ema_50','ema_200',
                'atr_14','volume','volume_ratio','vwap','swing_high','swing_low',
                'bos_bullish','bos_bearish','ob_bullish','ob_bearish',
                'fvg_bullish','fvg_bearish','session','is_killzone']
        bool_cols = {'swing_high','swing_low','bos_bullish','bos_bearish',
                     'ob_bullish','ob_bearish','fvg_bullish','fvg_bearish','is_killzone'}
        result[tf] = [
            {c: (int(r[i].timestamp()) if i == 0
                 else bool(r[i]) if c in bool_cols
                 else str(r[i])  if c == 'session'
                 else float(r[i]) if r[i] is not None else None)
             for i, c in enumerate(cols)}
            for r in reversed(rows)
        ]
    return jsonify(result)


@app.route('/api/ict')
def api_ict():
    return jsonify({
        '1h': _tools.dispatch_tool('get_ict_context', {'timeframe': '1h'}),
        '4h': _tools.dispatch_tool('get_ict_context', {'timeframe': '4h'}),
    })


@app.route('/api/session')
def api_session():
    return jsonify(_tools.dispatch_tool('get_session_stats', {}))


@app.route('/api/levels')
def api_levels():
    return jsonify(_tools.dispatch_tool('get_technical_levels', {}))


@app.route('/api/mtf')
def api_mtf():
    return jsonify(_tools.dispatch_tool('get_multi_timeframe_bias', {}))


@app.route('/api/sentiment')
def api_sentiment():
    return jsonify(_tools.dispatch_tool('get_sentiment', {'days': 7}))


@app.route('/api/fear_greed')
def api_fear_greed():
    return jsonify(_tools.dispatch_tool('get_fear_greed', {'days': 14}))


@app.route('/api/volume_profile')
def api_volume_profile():
    return jsonify(_tools.dispatch_tool('get_volume_profile', {}))


@app.route('/api/snapshot')
def api_snapshot():
    try:
        ctx = _pipeline.get_llm_context(max_age_hours=1)
    except Exception as e:
        ctx = f'Error: {e}'
    return jsonify({'text': ctx})


@app.route('/api/ml')
def api_ml():
    return jsonify(_tools.dispatch_tool('run_ml_prediction', {}))


@app.route('/api/price/update')
def api_price_update():
    """Fuerza actualización manual de btc_ohlcv desde Binance."""
    try:
        _update_ohlcv_job()
        return jsonify({'status': 'ok', 'ts': datetime.now(timezone.utc).isoformat()})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500


# ── Auction Market Theory (Volume Profile, CVD, Wyckoff) ─────────────────────
@app.route('/api/auction')
def api_auction():
    try:
        import importlib.util as _ilu
        spec = _ilu.spec_from_file_location(
            "indicators_mod", Path(__file__).parent / "indicators.py"
        )
        ind_mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(ind_mod)
        engine = get_engine()
        result = ind_mod.get_auction_theory_snapshot(engine, n_bars_vp=24)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Signals Breakdown ─────────────────────────────────────────────────────────
@app.route('/api/signals/breakdown')
def api_signals_breakdown():
    """
    Estado actual de TODAS las señales del sistema con scores ponderados.
    Adaptador fino sobre get_confluence_score (misma fuente que usa el bot)
    para garantizar paridad bot ↔ dashboard.
    """
    try:
        # ── ÚNICA FUENTE DE VERDAD — misma función que el bot ──────────────────
        cs = _tools.dispatch_tool('get_confluence_score', {'timeframe': '1h'})
        if not isinstance(cs, dict) or 'error' in cs:
            return jsonify({'error': (cs or {}).get('error', 'confluence_failed')}), 500

        modules = cs.get('modules') or {}
        # Renombrar 'ml' → 'ml_model' para que el JS (MODULE_LABELS) lo entienda
        modules_remapped = {
            'technical':   modules.get('technical', {}),
            'ict':         modules.get('ict', {}),
            'mtf':         modules.get('mtf', {}),
            'smart_money': modules.get('smart_money', {}),
            'sentiment':   modules.get('sentiment', {}),
            'ml_model':    modules.get('ml', {}),
        }

        def _s(mod):
            return float((modules_remapped.get(mod) or {}).get('score') or 0)
        def _w(mod):
            return float((modules_remapped.get(mod) or {}).get('weight') or 0)
        def _sigs(mod):
            return (modules_remapped.get(mod) or {}).get('signals') or []

        scores_by_module = {k: round(_s(k), 3) for k in modules_remapped}
        weights_used     = {k: round(_w(k), 3) for k in modules_remapped}

        final_score = float(cs.get('final_score') or cs.get('raw_score') or 0)
        label       = cs.get('label', 'NEUTRAL')
        confidence  = float(cs.get('confidence') or 0)

        # Datos extra para detalle del dashboard (no afectan al scoring)
        with _price_lock:
            price = _live_price.get('price')
        price = price or float(cs.get('price_now') or 0)

        return jsonify({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'price':     price,
            'timeframe': cs.get('timeframe', '1h'),
            # Por módulo — incluye score, weight, contribution y signals
            'technical':   {'score': _s('technical'),   'weight': _w('technical'),   'signals': _sigs('technical')},
            'ict':         {'score': _s('ict'),         'weight': _w('ict'),         'signals': _sigs('ict')},
            'mtf':         {'score': _s('mtf'),         'weight': _w('mtf'),         'signals': _sigs('mtf')},
            'smart_money': {'score': _s('smart_money'), 'weight': _w('smart_money'), 'signals': _sigs('smart_money')},
            'sentiment':   {'score': _s('sentiment'),   'weight': _w('sentiment'),   'signals': _sigs('sentiment')},
            'ml_model':    {'score': _s('ml_model'),    'weight': _w('ml_model'),    'signals': _sigs('ml_model')},
            'final': {
                'score':            round(final_score, 4),
                'label':            label,
                'confidence':       round(confidence, 3),
                'weights_used':     weights_used,
                'scores_by_module': scores_by_module,
                'conflict':         bool(cs.get('conflict', False)),
                'conflict_note':    cs.get('conflict_note', ''),
            },
        })
    except Exception as e:
        log.error("signals/breakdown error: %s", e)
        return jsonify({"error": str(e)}), 500


# ── Trade Calculator ──────────────────────────────────────────────────────────
@app.route('/api/trade/calculate', methods=['POST', 'GET'])
def api_trade_calculate():
    """
    POST body o GET params: entry, stop_loss, take_profit, capital, risk_pct, leverage, exchange, hours
    """
    try:
        if _req.request and False:  # placeholder
            pass
        from flask import request as _flask_req
        if _flask_req.method == 'POST':
            body = _flask_req.get_json(force=True) or {}
        else:
            body = _flask_req.args.to_dict()

        entry       = float(body.get('entry', 0))
        stop_loss   = float(body.get('stop_loss', 0))
        take_profit = float(body.get('take_profit', 0))
        capital     = float(body.get('capital', 10000))
        risk_pct    = float(body.get('risk_pct', 1.0))
        leverage    = float(body.get('leverage', 1.0))
        exchange    = str(body.get('exchange', 'binance_futures'))
        hours       = float(body.get('hours', 4.0))

        if not entry or not stop_loss or not take_profit:
            # Autocompletar con precio live + ATR
            with _price_lock:
                price_now = _live_price.get('price')
            mkt = _tools.dispatch_tool('query_market', {'timeframe': '1h', 'candles': 1})
            price_now = price_now or mkt.get('price_now', 0)
            atr = (mkt.get('indicators') or {}).get('atr_14') or price_now * 0.005
            if not entry:      entry       = price_now
            if not stop_loss:  stop_loss   = price_now - 1.5 * atr
            if not take_profit: take_profit = price_now + 2.5 * atr

        from agente_ia.trade_calculator import calculate_trade
        result = calculate_trade(
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            capital_usd=capital,
            risk_pct=risk_pct,
            leverage=leverage,
            exchange=exchange,
            holding_hours=hours,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/compound')
def api_compound():
    """Proyección de interés compuesto."""
    try:
        from flask import request as _r
        from agente_ia.trade_calculator import calculate_compound_growth, get_optimal_risk
        capital  = float(_r.args.get('capital', 10000))
        risk_pct = float(_r.args.get('risk_pct', 1.0))
        wr       = float(_r.args.get('win_rate', 0.50))
        rr       = float(_r.args.get('rr', 1.8))
        n        = int(_r.args.get('n_trades', 50))
        optimal  = get_optimal_risk(wr, rr)
        proj     = calculate_compound_growth(capital, risk_pct, wr, rr, n)
        proj['optimal_risk_pct'] = optimal
        return jsonify(proj)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/confluence')
def api_confluence():
    """Score de confluencia ponderado calculado por el agente (get_confluence_score)."""
    try:
        from flask import request as _r
        tf = _r.args.get('timeframe', '1h')
        result = _tools.dispatch_tool('get_confluence_score', {'timeframe': tf})
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════
# BOT ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

def _bot_engine():
    return get_engine()


def _load_bot_modules():
    import importlib.util as _ilu
    spec = _ilu.spec_from_file_location("live_bot", Path(__file__).parent / "live_bot.py")
    mod  = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@app.route('/api/bot/status')
def api_bot_status():
    try:
        from sqlalchemy import text as _t
        e = _bot_engine()
        with e.connect() as conn:
            state = conn.execute(_t("SELECT * FROM bot_state WHERE id=1")).fetchone()
            open_t = conn.execute(_t("""
                SELECT id, direction, entry_price, stop_loss, take_profit,
                       confluence_score, entry_time, trigger_type, size_btc
                FROM live_trades WHERE status='open'
                ORDER BY entry_time DESC
            """)).fetchall()

        state_d = dict(state._mapping) if state else {}

        # ── FIX: detectar bot vivo por proceso (no solo por bot_state DB) ─────
        is_running_db = bool(state_d.get("is_running", False))
        is_running_proc = False
        try:
            import psutil
            for proc in psutil.process_iter(['cmdline']):
                cmd = " ".join(proc.info.get('cmdline') or [])
                if 'live_bot.py' in cmd:
                    is_running_proc = True
                    break
        except Exception:
            pass
        is_running = is_running_db or is_running_proc

        # Capital fallback (si bot_state está vacío, calcular desde trades)
        capital_initial = float(state_d.get("initial_capital") or 50000)
        capital_current = state_d.get("current_capital")
        if capital_current is None:
            with e.connect() as conn:
                pnl_total = conn.execute(_t(
                    "SELECT COALESCE(SUM(net_pnl_usd), 0) FROM live_trades WHERE status='closed'"
                )).scalar() or 0
            capital_current = capital_initial + float(pnl_total)

        # PnL no realizado
        try:
            from agente_ia.bybit_client import BybitDemoClient
            bc = BybitDemoClient()
            unrealized_pnl = bc.get_unrealized_pnl() if bc.is_configured() else 0
            bybit_price = bc.get_price()
        except Exception:
            unrealized_pnl = 0
            bybit_price = None

        # Fallback de precio para PnL si Bybit no devuelve
        if not bybit_price:
            with _price_lock:
                bybit_price = _live_price.get("price")

        positions = []
        for t in open_t:
            tid, direction, entry_price, sl, tp, conf, entry_time, trigger, size_btc = t
            price_now = bybit_price or 0
            mult = 1 if direction == 'long' else -1
            # PnL = (price_now - entry) × size_btc × mult
            sz = float(size_btc or 0.001)
            upnl = mult * (float(price_now) - float(entry_price)) * sz if price_now else 0
            positions.append({
                "id": tid, "direction": direction,
                "entry_price": float(entry_price), "stop_loss": float(sl), "take_profit": float(tp),
                "confluence_score": float(conf or 0), "entry_time": str(entry_time),
                "trigger_type": trigger, "unrealized_pnl": round(upnl, 2),
            })

        return jsonify({
            "is_running":        is_running,
            "is_running_source": "process" if is_running_proc and not is_running_db else "db",
            "started_at":        str(state_d.get("started_at", "")),
            "initial_capital":   capital_initial,
            "current_capital":   round(float(capital_current), 2),
            "total_trades":      state_d.get("total_trades", 0),
            "open_trades":       len(positions),
            "agent_calls_today": state_d.get("agent_calls_today", 0),
            "last_check":        str(state_d.get("last_check", "")),
            "open_positions":    positions,
            "bybit_price":       bybit_price,
            "unrealized_pnl":    unrealized_pnl,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/bot/trades')
def api_bot_trades():
    try:
        from sqlalchemy import text as _t
        from flask import request as _r
        e = _bot_engine()
        limit = int(_r.args.get("limit", 50))
        with e.connect() as conn:
            rows = conn.execute(_t("""
                SELECT id, direction, entry_time, entry_price, exit_price,
                       stop_loss, take_profit, size_usd, leverage,
                       pnl_usd, pnl_pct, net_pnl_usd, fees_usd,
                       status, exit_reason, trigger_type, trigger_detail,
                       confluence_score, confluence_label, session, holding_hours
                FROM live_trades
                ORDER BY entry_time DESC
                LIMIT :lim
            """), {"lim": limit}).fetchall()
        return jsonify([dict(r._mapping) for r in rows])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/bot/alerts')
def api_bot_alerts():
    """
    Devuelve alertas activas dedup por zona de precio (±$50) y ordenadas
    por distancia al precio actual. Solo top 12 más relevantes.
    """
    try:
        from agente_ia.alert_generator import get_active_alerts
        alerts = get_active_alerts(_bot_engine())
        total_raw = len(alerts)

        # Precio actual para ordenar por distancia
        with _price_lock:
            cur_price = _live_price.get("price")
        if not cur_price:
            return jsonify({"alerts": alerts[:12], "count": len(alerts), "total_raw": total_raw})

        # Dedup por bucket de $50 + tipo + dirección
        seen = {}
        for a in alerts:
            try:
                price = float(a.get("level_price") or 0)
                bucket = round(price / 50) * 50
                key = (bucket, a.get("level_type"), a.get("direction"))
                dist = abs(price - float(cur_price)) / float(cur_price) * 100
                if key not in seen or dist < seen[key].get("_dist", 999):
                    a["_dist"] = dist
                    a["distance_pct"] = round(dist, 3)
                    seen[key] = a
            except Exception:
                continue

        # Ordenar por distancia ascendente, top 12
        dedup = sorted(seen.values(), key=lambda x: x.get("_dist", 999))[:12]
        for a in dedup:
            a.pop("_dist", None)

        return jsonify({
            "alerts": dedup,
            "count": len(dedup),
            "total_raw": total_raw,
            "dedup_ratio": f"{len(dedup)}/{total_raw}",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/bot/metrics')
def api_bot_metrics():
    try:
        from sqlalchemy import text as _t
        e = _bot_engine()
        with e.connect() as conn:
            state = conn.execute(_t("SELECT * FROM bot_state WHERE id=1")).fetchone()
            closed = conn.execute(_t("""
                SELECT net_pnl_usd, pnl_usd, fees_usd, trigger_type, session,
                       entry_time, exit_time, holding_hours, direction,
                       confluence_score, entry_price, exit_price
                FROM live_trades WHERE status='closed'
                ORDER BY entry_time
            """)).fetchall()
            n_open = conn.execute(_t("SELECT COUNT(*) FROM live_trades WHERE status='open'")).scalar()
            n_total = conn.execute(_t("SELECT COUNT(*) FROM live_trades")).scalar()

        state_d  = dict(state._mapping) if state else {}
        init_cap = float(state_d.get("initial_capital", 50000))
        cur_cap  = float(state_d.get("current_capital") or init_cap)

        trades = [dict(r._mapping) for r in closed]
        wins   = [t for t in trades if (t.get("net_pnl_usd") or 0) > 0]
        losses = [t for t in trades if (t.get("net_pnl_usd") or 0) <= 0]

        total_profit = sum(t["net_pnl_usd"] for t in wins  if t.get("net_pnl_usd"))
        total_loss   = abs(sum(t["net_pnl_usd"] for t in losses if t.get("net_pnl_usd")))
        profit_factor = round(total_profit / total_loss, 3) if total_loss > 0 else 0

        # Max drawdown
        running = init_cap
        peak    = init_cap
        max_dd  = 0.0
        for t in trades:
            running += (t.get("net_pnl_usd") or 0)
            if running > peak:
                peak = running
            dd = (peak - running) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)

        # Por trigger
        by_trigger = {}
        for trig in ("killzone", "level_alert", "manual"):
            trig_trades = [t for t in trades if t.get("trigger_type") == trig]
            trig_wins   = [t for t in trig_trades if (t.get("net_pnl_usd") or 0) > 0]
            by_trigger[trig] = {
                "trades": len(trig_trades),
                "wins":   len(trig_wins),
                "win_rate": round(len(trig_wins) / len(trig_trades) * 100, 1) if trig_trades else 0,
                "pnl":    round(sum((t.get("net_pnl_usd") or 0) for t in trig_trades), 2),
            }

        # Por sesión
        by_session = {}
        for sess in ("london", "ny", "asia"):
            sess_t = [t for t in trades if t.get("session") == sess]
            sess_w = [t for t in sess_t if (t.get("net_pnl_usd") or 0) > 0]
            by_session[sess] = {
                "trades":   len(sess_t),
                "win_rate": round(len(sess_w) / len(sess_t) * 100, 1) if sess_t else 0,
                "pnl":      round(sum((t.get("net_pnl_usd") or 0) for t in sess_t), 2),
            }

        # Equity curve (capital acumulado por trade)
        equity = []
        running = init_cap
        for t in trades:
            running += (t.get("net_pnl_usd") or 0)
            equity.append({
                "time": str(t.get("exit_time", "")),
                "capital": round(running, 2),
            })

        # Días corriendo
        started = state_d.get("started_at")
        days_running = 0.0
        if started:
            try:
                now = datetime.now(timezone.utc)
                if hasattr(started, 'tzinfo') and started.tzinfo:
                    days_running = (now - started).total_seconds() / 86400
                else:
                    days_running = (now - started.replace(tzinfo=timezone.utc)).total_seconds() / 86400
            except Exception:
                pass

        best  = max(trades, key=lambda t: t.get("net_pnl_usd", 0), default=None)
        worst = min(trades, key=lambda t: t.get("net_pnl_usd", 0), default=None)

        return jsonify({
            "capital_initial":   init_cap,
            "capital_current":   round(cur_cap, 2),
            "total_return_pct":  round((cur_cap / init_cap - 1) * 100, 3) if init_cap else 0,
            "total_trades":      int(n_total or 0),
            "open_trades":       int(n_open or 0),
            "closed_trades":     len(trades),
            "win_rate":          round(len(wins) / len(trades) * 100, 1) if trades else 0,
            "profit_factor":     profit_factor,
            "avg_win_usd":       round(total_profit / len(wins), 2) if wins else 0,
            "avg_loss_usd":      round(-total_loss / len(losses), 2) if losses else 0,
            "max_drawdown_pct":  round(max_dd, 2),
            "total_fees_usd":    round(sum(t.get("fees_usd", 0) for t in trades if t.get("fees_usd")), 2),
            "best_trade":        best,
            "worst_trade":       worst,
            "by_trigger":        by_trigger,
            "by_session":        by_session,
            "agent_calls_today": state_d.get("agent_calls_today", 0),
            "days_running":      round(days_running, 2),
            "equity_curve":      equity,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/bot/start', methods=['POST'])
def api_bot_start():
    try:
        import subprocess, sys as _sys, psutil
        from flask import request as _req
        # Check if already running
        for proc in psutil.process_iter(['cmdline']):
            try:
                cmd = ' '.join(proc.info['cmdline'] or [])
                if 'live_bot.py' in cmd:
                    return jsonify({"status": "already_running", "pid": proc.pid})
            except Exception:
                pass
        bot_py = str(Path(__file__).parent / "live_bot.py")
        dry = _req.args.get("dry_run", "false").lower() == "true"
        args = [_sys.executable, bot_py]
        if dry:
            args.append("--dry-run")
        proc = subprocess.Popen(args, cwd=str(Path(__file__).parent.parent))
        return jsonify({"status": "started", "pid": proc.pid, "dry_run": dry})
    except ImportError:
        # psutil not available — start without check
        import subprocess, sys as _sys
        from flask import request as _req
        bot_py = str(Path(__file__).parent / "live_bot.py")
        dry = _req.args.get("dry_run", "false").lower() == "true"
        args = [_sys.executable, bot_py]
        if dry:
            args.append("--dry-run")
        proc = subprocess.Popen(args, cwd=str(Path(__file__).parent.parent))
        return jsonify({"status": "started", "pid": proc.pid, "dry_run": dry})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/bot/stop', methods=['POST'])
def api_bot_stop():
    try:
        _bot = _load_bot_modules()
        _bot.cmd_stop()
        return jsonify({"status": "stopped"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/bot/close-all', methods=['POST'])
def api_bot_close_all():
    try:
        _bot = _load_bot_modules()
        _bot.cmd_close_all()
        return jsonify({"status": "all_closed"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _binance_klines(symbol, interval, limit):
    """Fetch OHLCV from Binance REST. Returns list of [ts_ms, o, h, l, c, v]."""
    tf_map = {'15m': '15m', '1h': '1h', '4h': '4h', '1d': '1d', '5m': '5m'}
    r = _req.get(
        f"{BINANCE_REST}/klines",
        params={'symbol': symbol, 'interval': tf_map.get(interval, interval), 'limit': min(limit, 1000)},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def _compute_ict_signals(candles):
    """
    OB, FVG, swings y killzones sobre la lista de velas del chart, con la MISMA
    lógica que agente_ia/indicators.py (detect_order_blocks / detect_fvg), para
    que lo que se dibuja coincida con lo que ven las tools del agente:
      - OB bullish: vela bajista antes de impulso ≥0.7% en las 3 velas siguientes;
        invalidado si un close posterior cierra bajo su low (espejo para bearish).
      - FVG: gap de 3 velas; invalidado si una vela posterior cruza el punto
        medio del gap (consequent encroachment).
    Devuelve zonas con high/low reales para dibujar rectángulos, no líneas.
    """
    ob_levels, fvg_levels, swing_levels, killzone_markers = [], [], [], []
    n = len(candles)
    if n < 5:
        return ob_levels, fvg_levels, swing_levels, killzone_markers

    from datetime import datetime as _dt, timezone as _tz

    highs  = [c['high']  for c in candles]
    lows   = [c['low']   for c in candles]
    closes = [c['close'] for c in candles]

    MOVE_TH = 0.007  # mismo umbral que indicators.detect_order_blocks

    for i in range(n):
        c_ = candles[i]
        ts = c_['time']
        o, h, l, cl = c_['open'], c_['high'], c_['low'], c_['close']

        # ── Swings (2 vecinos por lado, menos ruido que 1) ────────────────────
        if 2 <= i < n - 2:
            if h > max(highs[i-2:i]) and h > max(highs[i+1:i+3]):
                swing_levels.append({'time': ts, 'price': h, 'type': 'swing_high'})
            if l < min(lows[i-2:i]) and l < min(lows[i+1:i+3]):
                swing_levels.append({'time': ts, 'price': l, 'type': 'swing_low'})

        # ── Order Blocks ──────────────────────────────────────────────────────
        if i < n - 4:
            future_high = max(highs[i+1:i+4])
            future_low  = min(lows[i+1:i+4])
            move_up   = (future_high - cl) / cl
            move_down = (cl - future_low) / cl

            if move_up >= MOVE_TH and cl < o:
                if not any(closes[j] < l for j in range(i+1, n)):
                    ob_levels.append({'time': ts, 'type': 'ob_bull',
                                      'high': h, 'low': l,
                                      'price': round((h + l) / 2, 2)})
            if move_down >= MOVE_TH and cl > o:
                if not any(closes[j] > h for j in range(i+1, n)):
                    ob_levels.append({'time': ts, 'type': 'ob_bear',
                                      'high': h, 'low': l,
                                      'price': round((h + l) / 2, 2)})

        # ── Fair Value Gaps ───────────────────────────────────────────────────
        if 1 <= i < n - 1:
            # Bullish: gap entre high[i-1] y low[i+1]
            if lows[i+1] > highs[i-1]:
                top, bottom = lows[i+1], highs[i-1]
                mid = (top + bottom) / 2
                if not any(lows[j] <= mid for j in range(i+2, n)):
                    fvg_levels.append({'time': ts, 'type': 'fvg_bull',
                                       'high': top, 'low': bottom,
                                       'price': round(mid, 2)})
            # Bearish: gap entre low[i-1] y high[i+1]
            if highs[i+1] < lows[i-1]:
                top, bottom = lows[i-1], highs[i+1]
                mid = (top + bottom) / 2
                if not any(highs[j] >= mid for j in range(i+2, n)):
                    fvg_levels.append({'time': ts, 'type': 'fvg_bear',
                                       'high': top, 'low': bottom,
                                       'price': round(mid, 2)})

        # ── Killzones (mismas ventanas que indicators.KILLZONES) ──────────────
        # London open 07:00–09:30 UTC · NY open 13:00–14:30 UTC
        t = _dt.fromtimestamp(ts, tz=_tz.utc)
        mins = t.hour * 60 + t.minute
        if (7 * 60 <= mins < 9 * 60 + 30) or (13 * 60 <= mins < 14 * 60 + 30):
            killzone_markers.append({'time': ts})

    return ob_levels, fvg_levels, swing_levels, killzone_markers


@app.route('/api/chart/<timeframe>')
def api_chart(timeframe):
    """Candles from Binance + ICT levels computed on-the-fly + active trade."""
    tf = timeframe if timeframe in ('5m', '15m', '1h', '4h', '1d') else '1h'
    limit = {'5m': 300, '15m': 800, '1h': 720, '4h': 360, '1d': 500}.get(tf, 300)

    try:
        klines = _binance_klines('BTCUSDT', tf, limit)
        candles = []
        for k in klines:
            candles.append({
                'time':   int(k[0]) // 1000,
                'open':   float(k[1]),
                'high':   float(k[2]),
                'low':    float(k[3]),
                'close':  float(k[4]),
                'volume': float(k[5]),
            })

        ob_levels, fvg_levels, swing_levels, killzone_markers = _compute_ict_signals(candles)

        # Active trade from DB (best-effort)
        active_trade = None
        try:
            engine = get_engine()
            with engine.connect() as conn:
                tr = conn.execute(text("""
                    SELECT id, direction, entry_price, stop_loss, take_profit,
                           entry_time, confluence_score
                    FROM live_trades WHERE status='open'
                    ORDER BY entry_time DESC LIMIT 1
                """)).fetchone()
            if tr:
                active_trade = {
                    'id': tr[0], 'direction': tr[1],
                    'entry_price': float(tr[2] or 0), 'stop_loss': float(tr[3] or 0),
                    'take_profit': float(tr[4] or 0),
                    'entry_time': int(tr[5].timestamp()) if tr[5] else None,
                    'confluence_score': float(tr[6]) if tr[6] else None,
                }
        except Exception:
            pass

        # News markers from transparency API (best-effort, avoid extra DB call)
        news_markers = []

        return jsonify({
            'timeframe':        tf,
            'candles':          candles,
            'ob_levels':        ob_levels[-12:],
            'fvg_levels':       fvg_levels[-8:],
            'swing_levels':     swing_levels[-30:],
            'killzone_markers': killzone_markers,
            'news_markers':     news_markers,
            'active_trade':     active_trade,
        })
    except Exception as e:
        log.error("api_chart error: %s", e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/agent/info')
def api_agent_info():
    """Metadata del agente IA: modelo, tools, RAG stats, RAGAS scores."""
    import json as _json
    out = {
        "model": "gpt-4o",
        "temperature": 0.3,
        "max_tool_calls_per_turn": 8,
        "system_prompt_lines": 230,
        "tools": [
            {"name": "query_market",          "category": "data",       "desc": "OHLCV + indicadores"},
            {"name": "run_ml_prediction",     "category": "ml",         "desc": "Ensemble GBM+RF+Logistic (AUC 0.518 post-auditoría, no significativo)"},
            {"name": "rag_search",            "category": "rag",        "desc": "FAISS + MiniLM-L6-v2 (1996 docs)"},
            {"name": "get_sentiment",         "category": "sentiment",  "desc": "FinBERT scores agregados"},
            {"name": "get_fear_greed",        "category": "sentiment",  "desc": "F&G index"},
            {"name": "get_ict_context",       "category": "ict",        "desc": "OB/FVG/BOS/CHoCH/swings"},
            {"name": "get_session_stats",     "category": "context",    "desc": "Killzones, HOD/LOD"},
            {"name": "get_technical_levels",  "category": "ict",        "desc": "Niveles con R:R"},
            {"name": "get_multi_timeframe_bias","category": "mtf",      "desc": "Confluencia 1h/4h/1d"},
            {"name": "get_volume_profile",    "category": "smart_money","desc": "POC/VAH/VAL"},
            {"name": "get_trade_parameters",  "category": "execution",  "desc": "Sizing + SL/TP + fees"},
            {"name": "get_confluence_score",  "category": "scoring",    "desc": "Score final -1..+1 ponderado"},
            {"name": "get_entry_zone",        "category": "execution",  "desc": "Zona exacta de entrada"},
        ],
    }

    # RAG stats (FAISS index)
    try:
        from agente_ia.rag_pipeline import get_index_stats
        rag_stats = get_index_stats()
    except Exception as e:
        rag_stats = {"error": str(e), "available": False}
    out["rag"] = rag_stats

    # RAGAS scores (si existen)
    ragas_csv = _ROOT / "results" / "ragas_evaluation.csv"
    if ragas_csv.exists():
        try:
            import pandas as _pd
            df = _pd.read_csv(ragas_csv)
            out["ragas"] = {
                "n_questions": len(df),
                "faithfulness":      round(float(df["faithfulness"].mean()), 3),
                "answer_relevancy":  round(float(df["answer_relevancy"].mean()), 3),
                "context_precision": round(float(df["context_precision"].mean()), 3),
                "by_category": df.groupby("category")[["faithfulness", "answer_relevancy", "context_precision"]].mean().round(3).reset_index().to_dict(orient="records"),
                "evaluated_at": ragas_csv.stat().st_mtime,
            }
        except Exception as e:
            out["ragas"] = {"error": str(e)}
    else:
        out["ragas"] = {"available": False, "note": "Run analysis/ragas_evaluation.py to generate"}

    # ML model info
    metrics_path = _ROOT / "models" / "saved" / "metrics_v2.json"
    if metrics_path.exists():
        try:
            out["ml_model"] = _json.loads(metrics_path.read_text())
        except Exception:
            pass

    # Confluence weights actuales (misma instancia de tools que usa el agente)
    try:
        out["confluence_weights"] = dict(_tools.CONFLUENCE_WEIGHTS)
    except Exception:
        out["confluence_weights"] = None

    return jsonify(out)


@app.route('/api/system/transparency')
def api_system_transparency():
    """Transparencia total: qué datos usa la IA y cómo los procesa."""
    import json as _json
    from pathlib import Path as _Path

    engine = get_engine()
    out = {}

    with engine.connect() as c:

        # ── SENTIMIENTO: artículos por fuente (últimas 24h) ───────────────────
        try:
            rows = c.execute(text("""
                SELECT r.source, r.source_tier,
                       COUNT(*) as total,
                       ROUND(AVG(r.sentiment_raw)::numeric, 4)   as avg_raw,
                       ROUND(AVG(r.sentiment_decay)::numeric, 4) as avg_decay,
                       MAX(r.timestamp) as ultimo,
                       COALESCE(w.weight, 0.3) as tier_weight
                FROM raw_texts r
                LEFT JOIN source_weights w ON w.source = r.source
                WHERE r.timestamp > NOW() - INTERVAL '24 hours'
                  AND r.sentiment_raw IS NOT NULL
                GROUP BY r.source, r.source_tier, w.weight
                ORDER BY total DESC
            """)).fetchall()
            out['by_source'] = [dict(r._mapping) for r in rows]
        except Exception as e:
            out['by_source'] = []; log.warning("transparency by_source: %s", e)

        # ── TOP artículos más influyentes (últimas 48h) ───────────────────────
        try:
            rows = c.execute(text("""
                SELECT r.source, r.source_tier,
                       LEFT(r.text, 120) as preview,
                       r.url,
                       ROUND(r.sentiment_raw::numeric, 4)   as sentiment_raw,
                       ROUND(r.sentiment_decay::numeric, 4) as sentiment_decay,
                       r.timestamp,
                       COALESCE(w.weight, 0.3) as tier_weight,
                       ROUND((ABS(r.sentiment_decay) * COALESCE(w.weight, 0.3))::numeric, 4) as influence
                FROM raw_texts r
                LEFT JOIN source_weights w ON w.source = r.source
                WHERE r.timestamp > NOW() - INTERVAL '48 hours'
                  AND r.sentiment_raw IS NOT NULL
                ORDER BY influence DESC
                LIMIT 8
            """)).fetchall()
            out['top_articles'] = [dict(r._mapping) for r in rows]
        except Exception as e:
            out['top_articles'] = []; log.warning("transparency top_articles: %s", e)

        # ── NARRATIVAS detectadas (keywords últimas 48h) ──────────────────────
        try:
            row = c.execute(text("""
                SELECT
                  SUM(CASE WHEN text ILIKE ANY(ARRAY['%blackrock%','%etf%','%institutional%','%inflow%','%outflow%']) THEN 1 ELSE 0 END) as institutional,
                  SUM(CASE WHEN text ILIKE ANY(ARRAY['%fed%','%fomc%','%cpi%','%inflation%','%rate hike%','%rate cut%']) THEN 1 ELSE 0 END) as macro,
                  SUM(CASE WHEN text ILIKE ANY(ARRAY['%sec%','%regulation%','%ban%','%lawsuit%','%compliance%']) THEN 1 ELSE 0 END) as regulatory,
                  SUM(CASE WHEN text ILIKE ANY(ARRAY['%breakout%','%support%','%resistance%','%rally%','%crash%','%dump%']) THEN 1 ELSE 0 END) as technical,
                  SUM(CASE WHEN text ILIKE ANY(ARRAY['%halving%','%mining%','%whale%','%exchange%','%blockchain%','%on-chain%']) THEN 1 ELSE 0 END) as onchain,
                  COUNT(*) as total
                FROM raw_texts
                WHERE timestamp > NOW() - INTERVAL '48 hours'
            """)).fetchone()
            out['narratives'] = dict(row._mapping) if row else {}
        except Exception as e:
            out['narratives'] = {}; log.warning("transparency narratives: %s", e)

        # ── Artículos sin procesar ────────────────────────────────────────────
        try:
            out['unprocessed_24h'] = int(c.execute(text(
                "SELECT COUNT(*) FROM raw_texts WHERE processed=false AND timestamp > NOW()-INTERVAL '24 hours'"
            )).scalar() or 0)
            out['total_24h'] = int(c.execute(text(
                "SELECT COUNT(*) FROM raw_texts WHERE timestamp > NOW()-INTERVAL '24 hours'"
            )).scalar() or 0)
        except Exception as e:
            out['unprocessed_24h'] = 0; out['total_24h'] = 0

        # ── Source weights ────────────────────────────────────────────────────
        try:
            rows = c.execute(text(
                "SELECT source, tier, weight, notes FROM source_weights WHERE active=true ORDER BY tier, weight DESC"
            )).fetchall()
            out['source_weights'] = [dict(r._mapping) for r in rows]
        except Exception as e:
            out['source_weights'] = []

        # ── CALENDARIO MACRO ──────────────────────────────────────────────────
        try:
            rows = c.execute(text("""
                SELECT event_name, event_datetime, country, impact,
                       forecast, actual, previous, surprise_dir,
                       btc_bias, bias_reason
                FROM economic_calendar
                WHERE event_datetime BETWEEN NOW() - INTERVAL '24 hours'
                                         AND NOW() + INTERVAL '48 hours'
                ORDER BY event_datetime ASC
                LIMIT 15
            """)).fetchall()
            out['calendar_events'] = [dict(r._mapping) for r in rows]
        except Exception as e:
            out['calendar_events'] = []

        # ── ON-CHAIN METRICS ──────────────────────────────────────────────────
        try:
            rows = c.execute(text("""
                SELECT metric_name,
                       ROUND(value::numeric, 4) as value,
                       ROUND(value_7d_avg::numeric, 4) as value_7d_avg,
                       signal, ROUND(signal_strength::numeric, 4) as signal_strength,
                       source, timestamp
                FROM on_chain_metrics
                ORDER BY timestamp DESC, metric_name
            """)).fetchall()
            # Deduplicate: keep latest per metric
            seen = set(); deduped = []
            for r in rows:
                d = dict(r._mapping)
                if d['metric_name'] not in seen:
                    seen.add(d['metric_name']); deduped.append(d)
            out['onchain'] = deduped
        except Exception as e:
            out['onchain'] = []

        # ── ETF FLOWS (últimos 7 días) ─────────────────────────────────────────
        try:
            rows = c.execute(text("""
                SELECT vehicle, flow_date,
                       ROUND(flow_usd::numeric, 1) as flow_usd,
                       consecutive_days_inflow, consecutive_days_outflow
                FROM institutional_flows
                WHERE flow_date >= CURRENT_DATE - 7
                ORDER BY flow_date DESC, ABS(COALESCE(flow_usd,0)) DESC
                LIMIT 35
            """)).fetchall()
            out['etf_flows'] = [dict(r._mapping) for r in rows]
        except Exception as e:
            out['etf_flows'] = []

        # ── ETF resumen (5d net) ───────────────────────────────────────────────
        try:
            row = c.execute(text("""
                SELECT
                  ROUND(SUM(COALESCE(flow_usd,0))::numeric, 1) as net_5d,
                  COUNT(DISTINCT flow_date) as days,
                  SUM(CASE WHEN COALESCE(flow_usd,0) > 0 THEN 1 ELSE 0 END) as positive_days
                FROM institutional_flows
                WHERE flow_date >= CURRENT_DATE - 5
            """)).fetchone()
            out['etf_summary'] = dict(row._mapping) if row else {}
        except Exception as e:
            out['etf_summary'] = {}

        # ── MICROSTRUCTURE ────────────────────────────────────────────────────
        try:
            row = c.execute(text("""
                SELECT oi_usd, oi_change_1h, oi_change_24h,
                       funding_rate, funding_8h_avg,
                       liq_long_1h_usd, liq_short_1h_usd,
                       put_call_ratio, iv_25d_skew,
                       cme_gap_open, cme_gap_pct, cme_gap_filled,
                       signal, exchange, timestamp
                FROM market_microstructure
                ORDER BY timestamp DESC LIMIT 1
            """)).fetchone()
            out['microstructure'] = dict(row._mapping) if row else {}
        except Exception as e:
            out['microstructure'] = {}

        # ── BIAS SNAPSHOT ─────────────────────────────────────────────────────
        try:
            row = c.execute(text("""
                SELECT timestamp, bias_score, bias_label, regime, halving_phase,
                       score_news, score_calendar, score_onchain,
                       score_institutional, score_micro,
                       weight_news, weight_calendar, weight_onchain,
                       weight_institutional, weight_micro,
                       top_drivers
                FROM bias_snapshots
                ORDER BY timestamp DESC LIMIT 1
            """)).fetchone()
            out['bias_snapshot'] = dict(row._mapping) if row else None
        except Exception as e:
            out['bias_snapshot'] = None

    # ── MODELO ML ─────────────────────────────────────────────────────────────
    try:
        meta_path = _Path("models/saved/metrics_v2.json")
        fi_path   = _Path("models/saved/feature_importance_v2.json")
        model_meta = {}
        feature_importance = {}
        if meta_path.exists():
            with open(meta_path) as f: model_meta = _json.load(f)
        if fi_path.exists():
            with open(fi_path) as f: feature_importance = _json.load(f)
        top_features = sorted(feature_importance.items(), key=lambda x: -x[1])[:10]
        folds = model_meta.get('auc_per_fold', [])
        weak_folds = sum(1 for f in folds if f < 0.5)
        out['ml_model'] = {
            'auc_mean':     round(model_meta.get('auc_mean', 0), 4),
            'auc_std':      round(model_meta.get('auc_std', 0), 4),
            'auc_per_fold': [round(f, 4) for f in folds],
            'weak_folds':   weak_folds,
            'n_features':   model_meta.get('n_features', 0),
            'valid':        model_meta.get('valid', False),
            'trained_at':   model_meta.get('trained_at', ''),
            'top_features': [{'feature': k, 'importance': round(v, 4)} for k, v in top_features],
        }
    except Exception as e:
        out['ml_model'] = {}; log.warning("transparency ml_model: %s", e)

    # ── FinBERT check ─────────────────────────────────────────────────────────
    try:
        import sys as _sys
        from pathlib import Path as _P
        _sys.path.insert(0, str(_P(__file__).parent / 'news'))
        from agente_ia.news.run_pipeline import _scraper_module  # type: ignore
        out['finbert_active'] = True
    except Exception:
        try:
            import torch  # noqa
            out['finbert_active'] = True
        except Exception:
            out['finbert_active'] = False

    # ── Pesos confluencia (única fuente: CONFLUENCE_WEIGHTS de las tools) ────
    try:
        out['confluence_weights'] = dict(_tools.CONFLUENCE_WEIGHTS)
    except Exception:
        out['confluence_weights'] = None

    out['decay_formula'] = {
        'generic':       'λ=0.08 — vida media 8.7h',
        'etf_inst':      'λ=0.025 — vida media 27.7h (impacto duradero)',
        'regulatory':    'λ=0.04 — vida media 17.3h',
    }

    out['pipeline_status'] = {
        'news_scraper':    'RSS + scraping cada 1h',
        'calendar_scraper':'cada 4h + actuals cada 1h',
        'onchain_scraper': 'cada 6h (CryptoQuant)',
        'micro_scraper':   'cada 15min (Binance futuros)',
        'etf_scraper':     'diario (Farside)',
        'bias_calculator': 'cada 1h, cache 2h',
        'price_updater':   '5m/15m cada 5min · 1h/4h/1d cada 15min',
    }

    out['timestamp'] = datetime.now(timezone.utc).isoformat()
    return jsonify(out)


# ── Noticias: listado parametrizable + scrape on-demand ─────────────────────
# El listado de la pestaña "noticias" antes salía de /api/system/transparency,
# que ordena SIEMPRE por influencia → los artículos viejos influyentes se
# quedaban fijos arriba y parecía que las noticias no se actualizaban. Estos
# endpoints permiten ordenar por fecha (recientes) y refrescar a demanda.

@app.route('/api/news/latest')
def api_news_latest():
    """Artículos recientes de raw_texts. Params: order=recent|influence,
    filter=all|pos|neg, limit=N, hours=N. Ordena por fecha por defecto."""
    order  = request.args.get('order', 'recent')
    filt   = request.args.get('filter', 'all')
    limit  = max(1, min(int(request.args.get('limit', 30) or 30), 100))
    hours  = max(1, min(int(request.args.get('hours', 48) or 48), 168))

    order_sql = "r.timestamp DESC" if order == 'recent' else "influence DESC"
    filt_sql  = ""
    if filt == 'pos':
        filt_sql = "AND r.sentiment_decay > 0.05"
    elif filt == 'neg':
        filt_sql = "AND r.sentiment_decay < -0.05"

    try:
        engine = get_engine()
        with engine.connect() as c:
            rows = c.execute(text(f"""
                SELECT r.source, r.source_tier,
                       LEFT(r.text, 160) as preview, r.url,
                       ROUND(r.sentiment_raw::numeric, 4)   as sentiment_raw,
                       ROUND(r.sentiment_decay::numeric, 4) as sentiment_decay,
                       r.timestamp,
                       COALESCE(w.weight, 0.3) as tier_weight,
                       ROUND((ABS(r.sentiment_decay) * COALESCE(w.weight, 0.3))::numeric, 4) as influence
                FROM raw_texts r
                LEFT JOIN source_weights w ON w.source = r.source
                WHERE r.timestamp > NOW() - (:hours || ' hours')::interval
                  AND r.sentiment_raw IS NOT NULL
                  {filt_sql}
                ORDER BY {order_sql}
                LIMIT :limit
            """), {"hours": hours, "limit": limit}).fetchall()
            total24 = int(c.execute(text(
                "SELECT COUNT(*) FROM raw_texts WHERE timestamp > NOW()-INTERVAL '24 hours'"
            )).scalar() or 0)
        return jsonify({
            "articles":  [dict(r._mapping) for r in rows],
            "order":     order,
            "filter":    filt,
            "total_24h": total24,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as e:
        log.warning("api_news_latest: %s", e)
        return jsonify({"error": str(e), "articles": []}), 500


# Estado del scrape en background (un solo job a la vez)
_news_scrape = {"running": False, "started_at": None, "finished_at": None,
                "result": None, "error": None}
_news_scrape_lock = threading.Lock()

# Última decisión GO/NO_GO emitida (para gatear /api/bot/execute en servidor)
_last_decision: dict | None = None
_decision_lock = threading.Lock()


def _run_news_scrape_job():
    global _news_scrape
    try:
        ctx = _pipeline.run_full_pipeline(days_back=1)
        with _news_scrape_lock:
            _news_scrape["result"] = (ctx[:200] if isinstance(ctx, str) else "ok")
            _news_scrape["error"]  = None
    except Exception as e:
        log.warning("news scrape job failed: %s", e)
        with _news_scrape_lock:
            _news_scrape["error"] = str(e)
    finally:
        with _news_scrape_lock:
            _news_scrape["running"]     = False
            _news_scrape["finished_at"] = datetime.now(timezone.utc).isoformat()


@app.route('/api/news/scrape', methods=['POST'])
def api_news_scrape():
    """Dispara el pipeline de scrapers en background (no bloquea)."""
    with _news_scrape_lock:
        if _news_scrape["running"]:
            return jsonify({"started": False, "reason": "scrape ya en curso"}), 409
        _news_scrape.update({"running": True, "error": None, "result": None,
                             "started_at": datetime.now(timezone.utc).isoformat(),
                             "finished_at": None})
    threading.Thread(target=_run_news_scrape_job, daemon=True).start()
    return jsonify({"started": True})


@app.route('/api/news/scrape/status')
def api_news_scrape_status():
    with _news_scrape_lock:
        return jsonify(dict(_news_scrape))


# ── Ejecución manual del trade (botón "Ejecutar en Bybit") ──────────────────
# Gating en SERVIDOR: solo ejecuta si la última decisión del Decision Maker fue
# GO y no está caducada. No se fía del cliente. Re-calcula SL/TP frescos.
_EXECUTE_MAX_AGE_S = 300  # la decisión caduca a los 5 min

@app.route('/api/bot/execute', methods=['POST'])
def api_bot_execute():
    with _decision_lock:
        dec = dict(_last_decision) if _last_decision else None
    if not dec:
        return jsonify({"ok": False, "error": "No hay decisión: ejecuta primero el análisis del agente."}), 409
    age = time.time() - dec.get("created_at", 0)
    if age > _EXECUTE_MAX_AGE_S:
        return jsonify({"ok": False, "error": f"Decisión caducada ({age:.0f}s > {_EXECUTE_MAX_AGE_S}s). Re-ejecuta el análisis."}), 409
    if dec.get("decision") != "GO":
        return jsonify({"ok": False, "error": "Decisión NO_GO: ejecución bloqueada.",
                        "reasons": dec.get("reasons", [])}), 403
    direction = dec.get("direction")
    if direction not in ("long", "short"):
        return jsonify({"ok": False, "error": "Dirección inválida en la decisión."}), 400

    try:
        executor = _load("trade_executor", Path(__file__).parent / "trade_executor.py")
        from agente_ia.bybit_client import BybitDemoClient
        bybit = BybitDemoClient()
        engine = get_engine()
        cfg = {"initial_capital": 50000.0, "risk_per_trade_pct": 1.0, "leverage": 3.0}
        cs = _tools.dispatch_tool('get_confluence_score', {'timeframe': '1h'})
        result = executor.execute_trade(
            engine, bybit, _tools,
            direction=direction, config=cfg,
            trigger_type="manual", trigger_detail="Botón dashboard (decisión GO)",
            confluence=cs, agent_decision_text=dec.get("verdict_text", ""),
        )
        status = 200 if result.get("ok") else 500
        return jsonify(result), status
    except Exception as e:
        log.error("api_bot_execute: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route('/api/agent/stream')
def api_agent_stream():
    # 1. Refrescar OHLCV de los TFs que usa el agente (1h/4h/1d), con guard
    #    de 60s para no repetir el trabajo si se llama dos veces seguidas.
    try:
        _update_ohlcv_job(timeframes=('1h', '4h', '1d'), min_interval_s=60)
    except Exception as e:
        log.warning("agent_stream: OHLCV update failed: %s", e)

    # 2. Precio de referencia: el MISMO que muestra el dashboard (WebSocket
    #    Binance), con fallback REST. Se siembra en las tools para que
    #    query_market/get_confluence_score/get_entry_zone usen ese número.
    with _price_lock:
        lp  = _live_price.get('price')
        ch  = _live_price.get('change_24h')
        h24 = _live_price.get('high_24h')
        l24 = _live_price.get('low_24h')
    if not lp:
        try:
            ld = _req.get(f'{BINANCE_REST}/ticker/24hr',
                          params={'symbol': 'BTCUSDT'}, timeout=4).json()
            lp, ch  = float(ld['lastPrice']), float(ld['priceChangePercent'])
            h24, l24 = float(ld['highPrice']), float(ld['lowPrice'])
        except Exception as e:
            log.warning("agent_stream: live price fetch failed: %s", e)

    live_price_line = ''
    if lp:
        try:
            _tools.set_live_price(lp, "dashboard_agent_stream")
        except Exception:
            pass
        live_price_line = (
            f"PRECIO ACTUAL BTC/USDT (Binance, verificado ahora): ${lp:,.2f} USD\n"
            + (f"Cambio 24h: {ch:+.2f}%  |  High 24h: ${h24:,.2f}  |  Low 24h: ${l24:,.2f}\n"
               if ch is not None and h24 and l24 else "")
            + "Las tools (query_market, get_confluence_score, get_entry_zone) usan este "
              "mismo precio como price_now, así que sus distancias, SL y TP ya son "
              "coherentes con él.\n"
        )

    # 3. Contexto pipeline
    try:
        snapshot_ctx = _pipeline.get_llm_context(max_age_hours=1)
    except Exception:
        snapshot_ctx = '[snapshot no disponible]'

    history = [
        {'role': 'user',
         'content': f'{live_price_line}\nSnapshot noticias y bias del pipeline:\n\n{snapshot_ctx}'},
        {'role': 'assistant',
         'content': 'Entendido. Precio verificado. Voy a usar todas las herramientas para el análisis ICT completo.'},
    ]
    query = ("Analiza el mercado BTC ahora mismo con TODAS las herramientas disponibles. "
             "Sigue el método ICT del system prompt paso a paso. "
             "Llama get_confluence_score para obtener el score cuantitativo, "
             "get_entry_zone para la zona de entrada exacta, "
             "y get_trade_parameters para los parámetros del trade. "
             "El precio de referencia es el que te di al inicio — úsalo para todos los cálculos de distancia, SL y TP. "
             "Dame el setup completo o la condición exacta a esperar.")

    ref_price = lp

    def generate():
        # Meta inicial: el frontend muestra el precio de referencia del análisis
        yield f"data: {json.dumps({'meta': {'ref_price': ref_price, 'ts': datetime.now(timezone.utc).isoformat()}})}\n\n"
        analysis_parts = []
        try:
            for token in _agent.chat_stream(query, history=history, model='gpt-4o'):
                analysis_parts.append(token)
                yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

        # ── Contexto cuantitativo compartido (confluence + trade params) ──────
        cs = None
        tp_params = None
        if analysis_parts:
            try:
                cs = _tools.dispatch_tool('get_confluence_score', {'timeframe': '1h'})
                _score = float((cs or {}).get('final_score') or 0)
                _dir = 'long' if _score > 0 else 'short' if _score < 0 else None
                if _dir:
                    tp_params = _tools.dispatch_tool('get_trade_parameters', {
                        'direction': _dir, 'capital_usd': 50000.0,
                        'risk_pct': 1.0, 'leverage': 3.0, 'holding_hours': 8.0,
                    })
            except Exception as e:
                log.warning("agent_stream: contexto cuantitativo falló: %s", e)

        # ── Agente 2: Risk Manager (verdict/problemas deterministas + LLM op) ─
        review = None
        if analysis_parts and cs is not None:
            try:
                critic = _load("critic_agent", Path(__file__).parent / "critic_agent.py")
                review = critic.review_analysis(''.join(analysis_parts), confluence=cs,
                                                ref_price=ref_price, trade_params=tp_params)
                yield f"data: {json.dumps({'critic': review})}\n\n"
            except Exception as e:
                log.warning("critic review failed: %s", e)

        # ── Agente 3: Decision Maker determinista (GO/NO_GO final) ────────────
        if analysis_parts and cs is not None:
            try:
                de = _load("decision_engine", Path(__file__).parent / "decision_engine.py")
                decision = de.decide(cs, review, tp_params)
                decision['verdict_text'] = de.synthesize(decision, ''.join(analysis_parts), review)
                with _decision_lock:
                    global _last_decision
                    _last_decision = {**decision, "created_at": time.time()}
                yield f"data: {json.dumps({'decision': decision})}\n\n"
            except Exception as e:
                log.warning("decision engine failed: %s", e)
        yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


if __name__ == '__main__':
    print("=" * 52)
    print("  BTC Dashboard → http://localhost:5050")
    print("  WebSocket Binance: arrancando...")
    print("  APScheduler OHLCV update: cada 15 min")
    print("=" * 52)
    app.run(host='0.0.0.0', port=5050, debug=False, threaded=True)
