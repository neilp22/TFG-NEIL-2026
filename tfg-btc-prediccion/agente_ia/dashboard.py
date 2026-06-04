"""
dashboard.py — Servidor Flask del dashboard BTC
Uso: python agente_ia/dashboard.py  →  http://localhost:5050
"""
import importlib.util, json, logging, sys, threading
from datetime import datetime, timezone
from pathlib import Path

import requests as _req
from flask import Flask, jsonify, Response, render_template, stream_with_context

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(_ROOT / 'config' / '.env')

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)

app = Flask(__name__, template_folder=str(Path(__file__).parent / 'templates'))


def _load(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_tools    = _load("tools05",      Path(__file__).parent / "05_tools.py")
_agent    = _load("agent06",      Path(__file__).parent / "06_agent.py")
_pipeline = _load("run_pipeline", _ROOT / "agente_ia" / "news" / "run_pipeline.py")

# ── System prompt ICT completo ─────────────────────────────────────────────────
_agent.SYSTEM_PROMPT = """Eres un trader profesional especializado en Bitcoin intraday usando metodología ICT.

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
SCORING DE CONFLUENCIA (0–10)
═══════════════════════════════════════════════════════
+2  HTF (4h+1d) alineados con la dirección
+2  En killzone o <30 min para killzone
+2  OB o FVG válido como zona de entrada
+1  Volumen confirma (vol_ratio > 1.0)
+1  RSI no sobreextendido (<70 longs, >30 shorts)
+1  CHoCH o BOS reciente confirma dirección
+1  Patrón de vela en la zona de entrada (engulfing, hammer, etc)

Score  0–4 → NO TRADE — falta confluencia
Score  5–6 → SETUP DÉBIL — reducir tamaño o esperar confirmación
Score  7–8 → SETUP VÁLIDO — entrar con gestión estándar
Score  9–10 → SETUP FUERTE — tamaño máximo

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
MODELO ML (ensemble GBM+RF+Logistic, AUC 0.5988)
═══════════════════════════════════════════════════════
probability < 0.35: señal bajista fuerte (alta confianza)
probability 0.35-0.50: bajista moderado
probability 0.50-0.65: alcista moderado
probability > 0.65: señal alcista fuerte (alta confianza)
valid=false: ignorar ML, usar solo análisis técnico

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
1. query_market(timeframe)     → precio, RSI, MACD, BB, EMAs, VWAP, ATR
2. run_ml_prediction           → dirección ML con probabilidad
3. rag_search(query)           → noticias relevantes
4. get_sentiment               → FinBERT sentiment días recientes
5. get_fear_greed              → Fear & Greed
6. get_ict_context(timeframe)  → OBs, FVGs, BOS, CHoCH, swings, estructura
7. get_session_stats           → sesión, killzone, HOD/LOD, rango
8. get_technical_levels        → niveles cercanos ordenados con R:R
9. get_multi_timeframe_bias    → confluencia 1h/4h/1d
10. get_volume_profile         → confirmación de volumen por hora y sesión
11. get_trade_parameters       → sizing, fees, slippage, break-even, compounding

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
<p>Desglose punto por punto. Total X/10.</p>
<table>
<tr><th>Criterio</th><th>Cumple</th><th>Puntos</th></tr>
<tr><td>HTF alineado</td><td>SI/NO</td><td>+X</td></tr>
</table>

<h3>Setup</h3>
<div class="setup-box valid">
  <p><strong>VALIDO — Score X/10</strong></p>
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
  <p><strong>NO VALIDO — Score X/10</strong></p>
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
        with _price_lock:
            _live_price.update({
                "price":      float(d.get("c", 0) or 0),
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

def _update_ohlcv_job():
    """Descarga las últimas velas de Binance y actualiza btc_ohlcv (todos los TFs)."""
    try:
        import importlib.util as _ilu
        spec = _ilu.spec_from_file_location(
            "price_updater02",
            Path(__file__).parent / "02_price_updater.py"
        )
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod, "update_ohlcv"):
            result = mod.update_ohlcv()
            log.info("OHLCV update job: %s", result)
        else:
            log.warning("02_price_updater.py no tiene update_ohlcv()")
    except Exception as e:
        log.error("OHLCV update job failed: %s", e)


def _update_5m_job():
    """Actualiza solo 5m y 15m cada 5 minutos para entradas precisas."""
    try:
        import importlib.util as _ilu
        spec = _ilu.spec_from_file_location(
            "price_updater02",
            Path(__file__).parent / "02_price_updater.py"
        )
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # Patch temporal: solo actualizar 5m y 15m
        original_tf = None
        if hasattr(mod, "_TF_CONFIG"):
            original_tf = mod._TF_CONFIG.copy()
            mod._TF_CONFIG = {k: v for k, v in mod._TF_CONFIG.items() if k in ("5m", "15m")}
        if hasattr(mod, "update_ohlcv"):
            mod.update_ohlcv()
        if original_tf is not None:
            mod._TF_CONFIG = original_tf
    except Exception as e:
        log.error("5m update job failed: %s", e)


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
    """
    try:
        with _price_lock:
            price = _live_price.get("price")

        # ── Técnico 1h ────────────────────────────────────────────────────────
        mkt_1h = _tools.dispatch_tool('query_market', {'timeframe': '1h', 'candles': 2})
        mkt_4h = _tools.dispatch_tool('query_market', {'timeframe': '4h', 'candles': 2})
        price  = price or mkt_1h.get('price_now')

        ind_1h = mkt_1h.get('indicators', {})
        ind_4h = mkt_4h.get('indicators', {})

        rsi_1h = ind_1h.get('rsi_14')
        rsi_4h = ind_4h.get('rsi_14')
        e9_1h  = ind_1h.get('ema_9')
        e21_1h = ind_1h.get('ema_21')
        e50_1h = ind_1h.get('ema_50')
        macd_h = ind_1h.get('macd_hist')
        vr_1h  = ind_1h.get('volume_ratio')
        vwap   = ind_1h.get('vwap')

        ema_trend_1h = ("bullish" if e9_1h and e21_1h and e50_1h and e9_1h > e21_1h > e50_1h
                        else "bearish" if e9_1h and e21_1h and e50_1h and e9_1h < e21_1h < e50_1h
                        else "neutral")
        rsi_signal  = ("oversold" if rsi_1h and rsi_1h < 30
                       else "overbought" if rsi_1h and rsi_1h > 70
                       else "neutral")
        macd_signal = ("bullish" if macd_h and macd_h > 0 else
                       "bearish" if macd_h and macd_h < 0 else "neutral")
        bb_upper = ind_1h.get('bb_upper')
        bb_lower = ind_1h.get('bb_lower')
        bb_pos   = ((price - bb_lower) / (bb_upper - bb_lower)
                    if price and bb_upper and bb_lower and bb_upper != bb_lower else None)
        vol_spike = bool(vr_1h and vr_1h > 1.5)
        vwap_pos  = ("above" if price and vwap and price > vwap
                     else "below" if price and vwap and price < vwap
                     else "at")

        # Score técnico: -1 a +1
        tech_pts = 0.0
        tech_max = 5.0
        if ema_trend_1h == "bullish":  tech_pts += 1
        elif ema_trend_1h == "bearish": tech_pts -= 1
        if macd_signal == "bullish":   tech_pts += 1
        elif macd_signal == "bearish":  tech_pts -= 1
        if rsi_signal == "oversold":    tech_pts += 0.5
        elif rsi_signal == "overbought": tech_pts -= 0.5
        if vwap_pos == "above":          tech_pts += 0.5
        elif vwap_pos == "below":        tech_pts -= 0.5
        if vol_spike:                    tech_pts += 1 if tech_pts >= 0 else -1
        tech_score = round(tech_pts / tech_max, 3)

        # ── ICT ───────────────────────────────────────────────────────────────
        ict = _tools.dispatch_tool('get_ict_context', {'timeframe': '1h'})
        sess = _tools.dispatch_tool('get_session_stats', {})
        mtf  = _tools.dispatch_tool('get_multi_timeframe_bias', {})

        ms          = ict.get('market_structure', 'ranging')
        is_kz       = bool(ict.get('is_killzone'))
        ob_bull_near = bool(ict.get('active_ob_bullish'))
        ob_bear_near = bool(ict.get('active_ob_bearish'))
        fvg_above    = bool(ict.get('unmitigated_fvg_above'))
        fvg_below    = bool(ict.get('unmitigated_fvg_below'))
        bos_recent   = bool(ict.get('recent_bos', {}).get('direction', 'none') != 'none')
        choch_recent = bool(ict.get('recent_choch', {}).get('detected'))

        ict_pts = 0.0
        ict_max = 4.0
        if ms == "bullish":   ict_pts += 1
        elif ms == "bearish":  ict_pts -= 1
        if is_kz:              ict_pts += (0.5 if ict_pts >= 0 else -0.5)
        if bos_recent or choch_recent: ict_pts += (0.5 if ms == "bullish" else -0.5)
        if ob_bull_near and ms == "bullish": ict_pts += 1
        if ob_bear_near and ms == "bearish": ict_pts -= 1
        ict_score = round(ict_pts / ict_max, 3)

        # ── MTF confluencia ────────────────────────────────────────────────────
        conf       = mtf.get('confluence', {})
        mtf_dir    = conf.get('direction', 'neutral')
        trade_ok   = bool(conf.get('trade_allowed'))
        mtf_score_raw = conf.get('score', 0.5)
        mtf_score = round((mtf_score_raw - 0.5) * 2, 3)  # centrar en 0

        # ── Smart Money (VWAP + Volume Profile approx) ────────────────────────
        vp = _tools.dispatch_tool('get_volume_profile', {})
        sm_score = 0.0
        if vwap_pos == "above": sm_score += 0.3
        elif vwap_pos == "below": sm_score -= 0.3
        if vp.get('trade_volume_confirmation') == 'strong_confirm':
            sm_score += 0.4 if sm_score >= 0 else -0.4
        elif vp.get('trade_volume_confirmation') == 'weak_no_confirm':
            sm_score *= 0.5
        sm_score = round(max(-1.0, min(1.0, sm_score)), 3)

        # ── Sentimiento ───────────────────────────────────────────────────────
        sent   = _tools.dispatch_tool('get_sentiment', {'days': 3})
        fg     = _tools.dispatch_tool('get_fear_greed', {'days': 1})
        sent_v = sent.get('period_avg', 0) or 0
        fg_v   = (fg.get('latest', {}) or {}).get('value', 50) or 50
        # Normalizar sentimiento
        sent_score = round(
            sent_v * 0.5 + ((fg_v - 50) / 50) * 0.5,
            3
        )
        sent_score = max(-1.0, min(1.0, sent_score))

        # ── ML ────────────────────────────────────────────────────────────────
        ml = _tools.dispatch_tool('run_ml_prediction', {})
        ml_prob  = ml.get('probability')
        ml_valid = bool(ml.get('valid'))
        if ml_valid and ml_prob is not None:
            ml_score = round((ml_prob - 0.5) * 2, 3)   # 0.5 → 0, 0.65 → +0.3, 0.35 → -0.3
        else:
            ml_score = 0.0

        # ── Score final ponderado ─────────────────────────────────────────────
        weights = {
            "technical": 0.25,
            "ict":        0.30,
            "mtf":        0.20,
            "smart_money": 0.10,
            "sentiment":  0.05,
            "ml_model":   0.10,
        }
        scores = {
            "technical":  tech_score,
            "ict":        ict_score,
            "mtf":        mtf_score,
            "smart_money": sm_score,
            "sentiment":  sent_score,
            "ml_model":   ml_score,
        }
        final_score = round(sum(scores[k] * weights[k] for k in weights), 4)
        signals_active = sum(1 for s in scores.values() if abs(s) > 0.15)
        aligned = sum(1 for s in scores.values() if s > 0.1) >= 4 or \
                  sum(1 for s in scores.values() if s < -0.1) >= 4

        if final_score > 0.35:   label = "STRONG BUY"
        elif final_score > 0.15: label = "BUY"
        elif final_score < -0.35: label = "STRONG SELL"
        elif final_score < -0.15: label = "SELL"
        else:                     label = "NEUTRAL"

        return jsonify({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "price": price,
            "technical": {
                "ema_trend_1h":  ema_trend_1h,
                "ema_trend_4h":  ("bullish" if ind_4h.get('ema_9') and ind_4h.get('ema_21')
                                   and ind_4h['ema_9'] > ind_4h['ema_21'] else "bearish"),
                "rsi_1h":        rsi_1h,
                "rsi_4h":        rsi_4h,
                "rsi_signal":    rsi_signal,
                "macd_signal":   macd_signal,
                "bb_position":   round(bb_pos, 3) if bb_pos is not None else None,
                "vwap_position": vwap_pos,
                "volume_spike":  vol_spike,
                "volume_ratio":  vr_1h,
                "score":         tech_score,
            },
            "ict": {
                "session":          sess.get('current_session'),
                "is_killzone":      is_kz,
                "market_structure": ms,
                "ob_bullish_near":  ob_bull_near,
                "ob_bearish_near":  ob_bear_near,
                "fvg_above":        fvg_above,
                "fvg_below":        fvg_below,
                "bos_recent":       bos_recent,
                "choch_recent":     choch_recent,
                "minutes_to_kz":    ict.get('minutes_to_next_killzone'),
                "score":            ict_score,
            },
            "smart_money": {
                "vwap_position":    vwap_pos,
                "vwap":             vwap,
                "volume_confirmation": vp.get('trade_volume_confirmation'),
                "vol_ratio":        vp.get('vol_ratio_vs_hour_avg'),
                "score":            sm_score,
            },
            "sentiment": {
                "news_score":         round(sent_v, 4),
                "sentiment_label":    sent.get('sentiment_label'),
                "fear_greed":         fg_v,
                "fear_greed_label":   (fg.get('latest') or {}).get('classification'),
                "dominant_narrative": sent.get('dominant_narrative', '—'),
                "score":              sent_score,
            },
            "ml_model": {
                "prediction":  ml.get('direction'),
                "probability": ml_prob,
                "auc":         ml.get('auc'),
                "valid":       ml_valid,
                "top_features": ml.get('top_features', []),
                "score":        ml_score,
            },
            "mtf": {
                "direction":     mtf_dir,
                "trade_allowed": trade_ok,
                "score_raw":     mtf_score_raw,
                "score":         mtf_score,
            },
            "final": {
                "score":          final_score,
                "label":          label,
                "confidence":     round(min(1.0, abs(final_score) * 2), 3),
                "weights_used":   weights,
                "scores_by_module": scores,
                "signals_active": signals_active,
                "aligned":        aligned,
                "trade_allowed":  trade_ok,
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
                       confluence_score, entry_time, trigger_type
                FROM live_trades WHERE status='open'
                ORDER BY entry_time DESC
            """)).fetchall()

        state_d = dict(state._mapping) if state else {}

        # PnL no realizado
        try:
            from agente_ia.bybit_client import BybitDemoClient
            bc = BybitDemoClient()
            unrealized_pnl = bc.get_unrealized_pnl() if bc.is_configured() else 0
            bybit_price = bc.get_price()
        except Exception:
            unrealized_pnl = 0
            bybit_price = None

        positions = []
        for t in open_t:
            price_now = bybit_price or 0
            mult = 1 if t[1] == 'long' else -1
            upnl = mult * (price_now - t[2]) * 0.001 if price_now else 0
            positions.append({
                "id": t[0], "direction": t[1],
                "entry_price": t[2], "stop_loss": t[3], "take_profit": t[4],
                "confluence_score": t[5], "entry_time": str(t[6]),
                "trigger_type": t[7], "unrealized_pnl": round(upnl, 2),
            })

        return jsonify({
            "is_running":       state_d.get("is_running", False),
            "started_at":       str(state_d.get("started_at", "")),
            "initial_capital":  state_d.get("initial_capital", 50000),
            "current_capital":  state_d.get("current_capital"),
            "total_trades":     state_d.get("total_trades", 0),
            "open_trades":      state_d.get("open_trades", 0),
            "agent_calls_today": state_d.get("agent_calls_today", 0),
            "last_check":       str(state_d.get("last_check", "")),
            "open_positions":   positions,
            "bybit_price":      bybit_price,
            "unrealized_pnl":   unrealized_pnl,
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
    try:
        from agente_ia.alert_generator import get_active_alerts
        alerts = get_active_alerts(_bot_engine())
        return jsonify({"alerts": alerts, "count": len(alerts)})
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


@app.route('/api/chart/<timeframe>')
def api_chart(timeframe):
    """
    Candles + ICT levels + killzone markers + news markers + active trade.
    Used by TradingView Lightweight Charts on the dashboard.
    """
    tf = timeframe if timeframe in ('5m', '15m', '1h', '4h', '1d') else '1h'
    limit = {'5m': 200, '15m': 200, '1h': 168, '4h': 120, '1d': 90}.get(tf, 168)
    engine = get_engine()
    try:
        # ── Candles ───────────────────────────────────────────────────────────
        sql = text("""
            SELECT timestamp, open, high, low, close, volume,
                   ob_bullish, ob_bearish, fvg_bullish, fvg_bearish,
                   swing_high, swing_low, is_killzone, session,
                   bos_bullish, bos_bearish
            FROM btc_ohlcv
            WHERE timeframe=:tf
            ORDER BY timestamp DESC LIMIT :n
        """)
        with engine.connect() as conn:
            rows = conn.execute(sql, {'tf': tf, 'n': limit}).fetchall()
        rows = list(reversed(rows))

        candles, ob_levels, fvg_levels, swing_levels, killzone_markers = [], [], [], [], []

        for r in rows:
            ts = int(r[0].timestamp())
            o, h, l, c, v = float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])
            candles.append({'time': ts, 'open': o, 'high': h, 'low': l, 'close': c, 'volume': v})

            mid = round((h + l) / 2, 2)
            if r[6]:  # ob_bullish
                ob_levels.append({'time': ts, 'price': mid, 'type': 'ob_bull',
                                  'high': h, 'low': l, 'label': f'OB+ {tf}'})
            if r[7]:  # ob_bearish
                ob_levels.append({'time': ts, 'price': mid, 'type': 'ob_bear',
                                  'high': h, 'low': l, 'label': f'OB- {tf}'})
            if r[8]:  # fvg_bullish
                fvg_levels.append({'time': ts, 'price': mid, 'type': 'fvg_bull',
                                   'high': h, 'low': l, 'label': f'FVG+ {tf}'})
            if r[9]:  # fvg_bearish
                fvg_levels.append({'time': ts, 'price': mid, 'type': 'fvg_bear',
                                   'high': h, 'low': l, 'label': f'FVG- {tf}'})
            if r[10]:  # swing_high
                swing_levels.append({'time': ts, 'price': h, 'type': 'swing_high'})
            if r[11]:  # swing_low
                swing_levels.append({'time': ts, 'price': l, 'type': 'swing_low'})
            if r[12]:  # is_killzone
                killzone_markers.append({'time': ts, 'session': str(r[13] or '')})

        # ── News markers (últimas 48h) ─────────────────────────────────────────
        news_markers = []
        try:
            with engine.connect() as conn:
                nrows = conn.execute(text("""
                    SELECT published_at, title, sentiment_score
                    FROM news_articles
                    WHERE published_at >= NOW() - INTERVAL '48 hours'
                      AND ABS(sentiment_score) > 0.2
                    ORDER BY published_at DESC LIMIT 30
                """)).fetchall()
            for nr in nrows:
                if nr[0]:
                    news_markers.append({
                        'time':  int(nr[0].timestamp()),
                        'title': (nr[1] or '')[:60],
                        'score': round(float(nr[2] or 0), 3),
                    })
        except Exception:
            pass

        # ── Active trade ──────────────────────────────────────────────────────
        active_trade = None
        try:
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
                    'entry_price': float(tr[2]), 'stop_loss': float(tr[3]),
                    'take_profit': float(tr[4]),
                    'entry_time': int(tr[5].timestamp()) if tr[5] else None,
                    'confluence_score': float(tr[6]) if tr[6] else None,
                }
        except Exception:
            pass

        # Keep only last 5 OBs/FVGs (most recent = most relevant)
        ob_levels  = ob_levels[-10:]
        fvg_levels = fvg_levels[-10:]

        return jsonify({
            'timeframe':        tf,
            'candles':          candles,
            'ob_levels':        ob_levels,
            'fvg_levels':       fvg_levels,
            'swing_levels':     swing_levels[-20:],
            'killzone_markers': killzone_markers,
            'news_markers':     news_markers,
            'active_trade':     active_trade,
        })
    except Exception as e:
        log.error("api_chart error: %s", e)
        return jsonify({'error': str(e)}), 500


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

    # ── Pesos confluencia ─────────────────────────────────────────────────────
    out['confluence_weights'] = {
        'technical':   0.20,
        'ict':         0.25,
        'mtf':         0.20,
        'smart_money': 0.10,
        'sentiment':   0.10,
        'ml':          0.15,
    }

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


@app.route('/api/agent/stream')
def api_agent_stream():
    # 1. Forzar actualización OHLCV para que el agente tenga precios actuales
    try:
        _update_ohlcv_job()
    except Exception as e:
        log.warning("agent_stream: OHLCV update failed: %s", e)

    # 2. Precio live desde Binance (fuente de verdad)
    live_price_line = ''
    try:
        live_r = _req.get('https://api.binance.com/api/v3/ticker/24hr',
                          params={'symbol': 'BTCUSDT'}, timeout=4)
        ld = live_r.json()
        lp = float(ld['lastPrice'])
        ch = float(ld['priceChangePercent'])
        h24 = float(ld['highPrice'])
        l24 = float(ld['lowPrice'])
        live_price_line = (
            f"PRECIO ACTUAL BTC/USDT (Binance, verificado ahora): ${lp:,.2f} USD\n"
            f"Cambio 24h: {ch:+.2f}%  |  High 24h: ${h24:,.2f}  |  Low 24h: ${l24:,.2f}\n"
            f"IMPORTANTE: Usa este precio como referencia absoluta. "
            f"Si query_market devuelve un precio diferente, el precio correcto es ${lp:,.2f}.\n"
        )
    except Exception as e:
        log.warning("agent_stream: live price fetch failed: %s", e)

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
             "Sigue el método ICT del system prompt paso a paso, calcula el score de confluencia "
             "y dame el setup completo (o la condición exacta a esperar). "
             "El precio de referencia es el que te di al inicio — úsalo para todos los cálculos de distancia, SL y TP. "
             "Incluye los parámetros exactos del trade con get_trade_parameters.")

    def generate():
        try:
            for token in _agent.chat_stream(query, history=history, model='gpt-4o-mini'):
                yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
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
