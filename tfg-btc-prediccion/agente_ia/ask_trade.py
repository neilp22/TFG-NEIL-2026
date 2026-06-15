"""
ask_trade.py — Análisis completo BTC: snapshot de noticias + 9 tools + trade
Uso: python agente_ia/ask_trade.py
"""
import importlib.util, sys, logging
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(_ROOT / 'config' / '.env')

logging.basicConfig(level=logging.WARNING, format='%(asctime)s [%(levelname)s] %(message)s')

# ── Cargar agente (06_agent.py — no tocamos el archivo) ───────────────────────
_aspec = importlib.util.spec_from_file_location("agent06", Path(__file__).parent / "06_agent.py")
agent_mod = importlib.util.module_from_spec(_aspec)
_aspec.loader.exec_module(agent_mod)

# ── System prompt con 9 tools + ICT + estructura de respuesta ─────────────────
agent_mod.SYSTEM_PROMPT = """Eres un trader y analista técnico especializado en Bitcoin intraday.

HERRAMIENTAS DISPONIBLES (úsalas todas para el análisis):
1. query_market          — precio, RSI, MACD, BB, EMAs, VWAP, ATR (1h/4h/1d)
2. run_ml_prediction     — ensemble model ICT (AUC 0.5988, válido si valid=true)
3. rag_search            — búsqueda semántica en noticias
4. get_sentiment         — sentimiento FinBERT agregado de noticias BTC
5. get_fear_greed        — Fear & Greed Index
6. get_ict_context       — ICT: Order Blocks, FVGs, BOS, CHoCH, swings, estructura
7. get_session_stats     — sesión actual, killzones, HOD/LOD, rango vs histórico
8. get_technical_levels  — niveles técnicos cercanos con R:R long/short
9. get_multi_timeframe_bias — confluencia 1h/4h/1d, trade_allowed

REGLAS:
- NO operar en dead_zone (22:00-00:00 UTC) ni fin de semana sin volumen
- Killzone (London 07:00-09:30 / NY 13:00-14:30 UTC) = máxima probabilidad
- Setup válido = confluencia TF + nivel ICT + volumen + killzone
- Si trade_allowed=false: no hay trade, explicar qué esperar

FORMATO DE RESPUESTA (obligatorio, en este orden):
━━━ SITUACIÓN ACTUAL ━━━
Precio, sesión, estructura de mercado en 1-2 líneas

━━━ INDICADORES TÉCNICOS ━━━
1h: RSI | MACD | BB position | EMA trend | Volume ratio
4h: RSI | MACD | EMA trend
1d: RSI | EMA trend
Interpretación breve de cada uno

━━━ CONTEXTO ICT ━━━
Market structure | BOS/CHoCH recientes | Patrón vela
Order Blocks activos | FVGs sin mitigar
Sesión + killzone

━━━ SESIÓN Y TIMING ━━━
Sesión actual | HOD/LOD | Rango vs histórico | Próxima killzone

━━━ SENTIMIENTO Y MACRO ━━━
Fear & Greed | Sentimiento noticias | Resumen del bias de noticias (del contexto previo)

━━━ MODELO ML ━━━
Dirección | Probabilidad | Confianza | Top features

━━━ NIVELES CLAVE ━━━
Resistencia/soporte más cercanos con precio exacto, tipo y distancia %
R:R long | R:R short

━━━ SETUP ━━━
[VÁLIDO / NO VÁLIDO — razón]
Si válido:
  Dirección: LONG / SHORT
  Entry: precio exacto + condición de entrada
  Stop Loss: precio exacto + razón (qué nivel invalida)
  Target 1: precio + % de ganancia
  Target 2: precio + % de ganancia
  R:R: ratio
  Timing: cuándo ejecutar (killzone, condición)
Si no válido:
  Qué condición esperar + precio de referencia + cuándo volver a mirar

━━━ SEÑALES DE INVALIDACIÓN ━━━
2-3 condiciones específicas que cancelan el setup

Responde en español. Sé preciso con los precios y porcentajes.
"""

# ── Obtener snapshot de noticias/bias antes de la llamada LLM ────────────────
def _get_news_snapshot() -> str:
    try:
        _pspec = importlib.util.spec_from_file_location(
            "run_pipeline", _ROOT / "agente_ia" / "news" / "run_pipeline.py"
        )
        pipeline = importlib.util.module_from_spec(_pspec)
        _pspec.loader.exec_module(pipeline)
        ctx = pipeline.get_llm_context(max_age_hours=2)
        return ctx
    except Exception as e:
        return f"[Snapshot de noticias no disponible: {e}]"


QUERY = """Con todos los datos disponibles (incluyendo el snapshot de noticias y bias que tienes en el contexto),
dame el análisis completo del mercado BTC ahora mismo siguiendo el formato exacto del system prompt.
Muestra todos los indicadores, niveles y el setup del próximo trade (o qué esperar si no hay setup válido)."""


if __name__ == '__main__':
    SEP = "═" * 62

    # ── PASO 1: Snapshot de noticias (impreso antes de la LLM) ───────────────
    print(f"\n{SEP}")
    print("  OBTENIENDO SNAPSHOT DE NOTICIAS Y BIAS BTC...")
    print(SEP)
    news_ctx = _get_news_snapshot()
    print(news_ctx)

    # ── PASO 2: Llamada al agente con el snapshot como contexto previo ────────
    print(f"\n{SEP}")
    print("  ANALIZANDO CON AGENTE IA (GPT-4o-mini + 9 tools)...")
    print(SEP + "\n")

    # Inyectar el snapshot como mensaje de sistema adicional
    history = [
        {
            "role": "user",
            "content": f"Contexto de noticias y bias de mercado (pipeline interno):\n\n{news_ctx}"
        },
        {
            "role": "assistant",
            "content": "Entendido. Tengo el snapshot de noticias y bias. Ahora consultaré las herramientas de mercado para completar el análisis."
        }
    ]

    for token in agent_mod.chat_stream(QUERY, history=history, model="gpt-4o"):
        print(token, end='', flush=True)

    print(f"\n{SEP}\n")
