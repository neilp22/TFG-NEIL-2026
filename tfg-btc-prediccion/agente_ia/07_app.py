"""
07_app.py — Demo Streamlit del Agente de Investigación Financiera BTC

Uso:
  streamlit run agente_ia/07_app.py
  streamlit run agente_ia/07_app.py --server.port 8502
"""

import os, sys, json
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / 'config' / '.env')

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

# ── Configuración de página ───────────────────────────────────────────────────

st.set_page_config(
    page_title="Agente BTC — TFG",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("₿ Agente BTC")
    st.caption("TFG — UAB 2025/26")
    st.markdown("---")

    st.subheader("Preguntas de ejemplo")
    EXAMPLES = [
        "¿Cuál es el precio actual de Bitcoin?",
        "¿Qué dicen los indicadores técnicos hoy?",
        "¿Qué probabilidad da el modelo ML de subida?",
        "¿Cómo está el sentimiento de las últimas noticias?",
        "¿Cuál es el Fear & Greed Index hoy?",
        "Busca noticias sobre regulación de ETFs de Bitcoin",
        "Dame un resumen completo del mercado BTC",
    ]

    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True, key=f"ex_{ex[:20]}"):
            st.session_state['pending_query'] = ex

    st.markdown("---")

    with st.expander("Configuración del modelo"):
        model_choice = st.selectbox(
            "Modelo OpenAI",
            ["gpt-4o-mini", "gpt-4o"],
            index=0,
        )
        verbose_mode = st.checkbox("Modo detallado (mostrar tool calls)", value=False)

    st.markdown("---")

    st.caption(
        "**Aviso:** Este agente es un prototipo académico. "
        "Las predicciones ML tienen AUC ≈ 0.53 (no significativo). "
        "No constituye asesoramiento financiero."
    )

    api_key_ok = bool(os.getenv('OPENAI_API_KEY', ''))
    if api_key_ok:
        st.success("OpenAI API: configurada")
    else:
        st.error("OPENAI_API_KEY no configurada")

# ── Estado de sesión ──────────────────────────────────────────────────────────

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'history' not in st.session_state:
    st.session_state.history = []

# ── Header ────────────────────────────────────────────────────────────────────

st.title("Agente de Investigación Financiera BTC")
st.markdown(
    "Consulta datos de mercado, indicadores técnicos, sentimiento de noticias "
    "y predicciones ML sobre Bitcoin en lenguaje natural."
)

# ── Panel rápido de datos ─────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def _get_quick_data():
    try:
        from tools import query_market, get_fear_greed, get_sentiment
        market = query_market(days=1)
        fg     = get_fear_greed(days=1)
        sent   = get_sentiment(days=3)
        return market, fg, sent
    except Exception as e:
        return None, None, None


col1, col2, col3, col4 = st.columns(4)

market_data, fg_data, sent_data = _get_quick_data()

with col1:
    if market_data and not market_data.get('error'):
        price = market_data.get('price_usd')
        ret   = market_data.get('daily_return')
        delta_str = f"{ret*100:+.2f}%" if ret is not None else None
        st.metric("Bitcoin (BTC/USD)", f"${price:,.0f}" if price else "N/A", delta=delta_str)
    else:
        st.metric("Bitcoin (BTC/USD)", "N/A")

with col2:
    if market_data and not market_data.get('error'):
        rsi = market_data.get('indicators', {}).get('rsi_14')
        rsi_sig = market_data.get('rsi_signal', '')
        st.metric("RSI (14)", f"{rsi:.1f}" if rsi else "N/A", delta=rsi_sig)
    else:
        st.metric("RSI (14)", "N/A")

with col3:
    if fg_data and not fg_data.get('error') and fg_data.get('latest'):
        fg_val  = fg_data['latest'].get('value', 'N/A')
        fg_cls  = fg_data['latest'].get('classification', '')
        st.metric("Fear & Greed", str(fg_val), delta=fg_cls)
    else:
        st.metric("Fear & Greed", "N/A")

with col4:
    if sent_data and not sent_data.get('error'):
        avg = sent_data.get('period_avg')
        lbl = sent_data.get('sentiment_label', '')
        st.metric("Sentimiento (3d)", f"{avg:+.3f}" if avg is not None else "N/A", delta=lbl)
    else:
        st.metric("Sentimiento (3d)", "N/A")

st.markdown("---")

# ── Historial de chat ─────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# ── Input del usuario ─────────────────────────────────────────────────────────

pending = st.session_state.pop('pending_query', None)
user_input = st.chat_input("Pregunta sobre Bitcoin...") or pending

if user_input:
    # Mostrar mensaje del usuario
    st.session_state.messages.append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)

    # Generar respuesta del agente
    with st.chat_message('assistant'):
        if not api_key_ok:
            st.error(
                "OPENAI_API_KEY no configurada. "
                "Añádela en `config/.env` y reinicia la app."
            )
        else:
            try:
                from agent import chat_stream  # type: ignore

                placeholder = st.empty()
                full_response = ""
                tool_calls_shown = []

                # Mostrar indicador de carga durante tool calls
                with st.spinner("Consultando herramientas..."):
                    for token in chat_stream(
                        user_input,
                        history=st.session_state.history,
                        model=model_choice,
                    ):
                        full_response += token
                        placeholder.markdown(full_response + "▌")

                placeholder.markdown(full_response)
                st.session_state.messages.append(
                    {'role': 'assistant', 'content': full_response}
                )
                # No actualizamos history aquí para simplicidad del demo

            except ImportError as e:
                st.error(f"Error importando el agente: {e}")
            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Error del agente: {e}")
                if verbose_mode:
                    st.exception(e)

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    f"TFG — Predicción de Bitcoin con ML y Sentimiento | UAB 2025/26 | "
    f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}"
)
