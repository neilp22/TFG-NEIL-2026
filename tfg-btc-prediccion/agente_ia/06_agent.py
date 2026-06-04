"""
06_agent.py — Agente de Investigación Financiera BTC (GPT-4o-mini)

Loop principal:
  1. Recibe pregunta del usuario
  2. GPT-4o-mini decide qué herramienta llamar
  3. Ejecuta la herramienta y devuelve resultado
  4. GPT-4o-mini genera respuesta final con los datos

Uso:
  python agente_ia/06_agent.py                   # chat interactivo
  python agente_ia/06_agent.py --test            # prueba con preguntas de ejemplo
  python agente_ia/06_agent.py --query "¿Cómo está Bitcoin hoy?"
"""

import os, sys, json, logging
from pathlib import Path
from typing import Generator

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / 'config' / '.env')

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

try:
    from openai import OpenAI
    _openai_available = True
except ImportError:
    _openai_available = False
    log.warning("openai no instalado. Ejecuta: pip install openai>=1.30.0")

from tools import TOOL_DEFINITIONS, dispatch_tool  # type: ignore


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Eres un asistente de investigación financiera especializado en Bitcoin.
Tu función es analizar datos de mercado, noticias, sentimiento y predicciones ML
para ayudar al usuario a entender el estado actual del mercado BTC.

REGLAS IMPORTANTES:
1. Siempre usa las herramientas disponibles para obtener datos actuales antes de responder.
2. NUNCA des consejos de inversión definitivos ni recomendaciones de compra/venta.
3. Siempre menciona las limitaciones del modelo ML (AUC ~0.53, no estadísticamente significativo).
4. Presenta los datos con contexto: qué significan los valores, no solo los números.
5. Si no tienes datos suficientes, dilo claramente.
6. Responde en el mismo idioma que el usuario (español o inglés).

FORMATO DE RESPUESTA:
- Comienza con un resumen ejecutivo (2-3 frases)
- Presenta los datos clave en formato estructurado
- Termina con una nota de cautela sobre limitaciones

SOBRE EL MODELO ML:
- XGBoost entrenado con indicadores técnicos + sentimiento FinBERT
- Walk-forward validation: AUC medio = 0.528 (IC 95%: [0.451, 0.605])
- p-valor vs azar = 0.48 → NO estadísticamente significativo
- Úsalo como un dato más, no como señal definitiva.

FLUJO OBLIGATORIO DE DECISIÓN:
1. Llama SIEMPRE a get_confluence_score() PRIMERO. Ese score ya está calculado
   matemáticamente con pesos fijos. NO recalcules tú el score.
2. Tu trabajo es EXPLICAR ese score e interpretar el desglose por módulo.
3. Solo puedes AJUSTAR la dirección final si hay una razón de peso que el cálculo
   no captura (ej: evento macro en <2h, conflicto fuerte entre módulos, noticia
   de última hora). Si ajustas, DEBES justificarlo explícitamente.
4. Para CUALQUIER trade, llama a get_trade_parameters() para obtener el sizing,
   stop loss, take profit y costes REALES. NUNCA inventes los números de posición,
   break-even o R:R — vienen de esa tool.
5. Si get_confluence_score devuelve conflict=true, advierte al usuario y reduce
   la confianza de tu recomendación.
6. Para entradas precisas (niveles exactos de entrada), llama a get_entry_zone()
   con la dirección del trade.

REGLA DE ENTRADAS:
- Si el OB/FVG de entrada está a más de 1.5% del precio actual, indícalo como
  "entrada en retest (orden límite)" y distínguelo del precio actual claramente.
- Si no hay OB/FVG válido cerca (<1.5%), di "sin entrada de alta probabilidad ahora,
  esperar a que el precio alcance [nivel]".

MODO BOT: Si tu mensaje incluye "[BOT_DECISION_REQUEST]", al final de tu análisis
devuelve OBLIGATORIAMENTE un bloque JSON con tu decisión:
```json
{
  "decision": "TRADE" | "NO_TRADE",
  "direction": "long" | "short" | null,
  "confidence": 0.0-1.0,
  "reasoning": "una frase clara de por qué",
  "entry_type": "market" | "limit",
  "limit_price": null
}
```
Solo recomienda TRADE si: confluence score y tu análisis coinciden, hay setup ICT
válido (OB/FVG cercano), y no hay evento macro de alto riesgo en <2h.
Si el confluence score es fuerte pero el setup no es limpio, di NO_TRADE y explica.
"""

MAX_TOOL_CALLS = 8  # Máximo de iteraciones de tool-calling por turno


# ── Cliente OpenAI ────────────────────────────────────────────────────────────

def _get_client():
    api_key = os.getenv('OPENAI_API_KEY', '')
    if not api_key:
        raise ValueError("OPENAI_API_KEY no configurado en config/.env")
    if not _openai_available:
        raise ImportError("openai package no instalado")
    return OpenAI(api_key=api_key)


# ── Chat function ─────────────────────────────────────────────────────────────

def chat(
    user_message: str,
    history: list = None,
    model: str = "gpt-4o-mini",
    verbose: bool = False,
) -> tuple[str, list]:
    """
    Procesa un mensaje del usuario con el agente.

    Args:
        user_message: Pregunta o instrucción del usuario
        history:      Historial de mensajes previos (list de dicts OpenAI)
        model:        Modelo OpenAI a usar
        verbose:      Si True, imprime las llamadas a herramientas

    Returns:
        (respuesta_final, historial_actualizado)
    """
    client = _get_client()

    messages = list(history or [])
    if not any(m.get('role') == 'system' for m in messages):
        messages.insert(0, {'role': 'system', 'content': SYSTEM_PROMPT})

    messages.append({'role': 'user', 'content': user_message})

    tool_calls_count = 0

    while tool_calls_count < MAX_TOOL_CALLS:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
            temperature=0.3,
        )

        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        # Si no hay tool calls, el agente ha terminado
        if not msg.tool_calls:
            break

        # Ejecutar cada tool call
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            if verbose:
                log.info(f"Tool call: {fn_name}({json.dumps(args, ensure_ascii=False)})")

            result = dispatch_tool(fn_name, args)
            result_str = json.dumps(result, ensure_ascii=False, default=str)

            if verbose:
                log.info(f"Tool result ({fn_name}): {result_str[:200]}...")

            messages.append({
                'role':         'tool',
                'tool_call_id': tc.id,
                'content':      result_str,
            })

        tool_calls_count += len(msg.tool_calls)

    # Extraer respuesta final
    final_msg = next(
        (m for m in reversed(messages) if isinstance(m, dict) and m.get('role') == 'assistant'),
        None
    )
    if final_msg is None and hasattr(msg, 'content'):
        final_msg = {'content': msg.content}

    final_text = (final_msg or {}).get('content') or "Sin respuesta generada."

    return final_text, messages


def chat_stream(
    user_message: str,
    history: list = None,
    model: str = "gpt-4o-mini",
) -> Generator[str, None, list]:
    """
    Versión streaming de chat(). Yielda tokens de la respuesta final.
    Uso: for token in chat_stream(msg): print(token, end='', flush=True)
    """
    client = _get_client()

    messages = list(history or [])
    if not any(m.get('role') == 'system' for m in messages):
        messages.insert(0, {'role': 'system', 'content': SYSTEM_PROMPT})

    messages.append({'role': 'user', 'content': user_message})

    tool_calls_count = 0

    # Fase 1: tool calls (no streaming)
    while tool_calls_count < MAX_TOOL_CALLS:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
            temperature=0.3,
        )
        msg = response.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            # Respuesta final — re-generar en streaming
            messages_for_stream = messages[:-1]  # quitar último mensaje assistant
            break

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            result = dispatch_tool(fn_name, args)
            messages.append({
                'role':         'tool',
                'tool_call_id': tc.id,
                'content':      json.dumps(result, ensure_ascii=False, default=str),
            })

        tool_calls_count += len(msg.tool_calls)
    else:
        messages_for_stream = messages

    # Fase 2: streaming de la respuesta final
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        temperature=0.3,
    )
    full_text = ""
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            full_text += delta.content
            yield delta.content

    messages.append({'role': 'assistant', 'content': full_text})
    return messages


# ── CLI interactivo ───────────────────────────────────────────────────────────

TEST_QUERIES = [
    "¿Cuál es el precio actual de Bitcoin y qué dicen los indicadores técnicos?",
    "¿Qué probabilidad da el modelo ML de que Bitcoin suba mañana?",
    "¿Cuál es el sentimiento de las noticias de los últimos 7 días?",
    "¿Cómo está el Fear & Greed Index hoy?",
    "Busca noticias sobre regulación de Bitcoin y ETFs.",
]


def run_interactive():
    """Bucle de chat interactivo en terminal."""
    print("=== Agente de Investigación Financiera BTC ===")
    print("Escribe 'salir' o 'exit' para terminar.\n")

    history = []
    while True:
        try:
            user_input = input("Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nHasta luego.")
            break

        if not user_input:
            continue
        if user_input.lower() in ('salir', 'exit', 'quit'):
            print("Hasta luego.")
            break

        try:
            print("Agente: ", end='', flush=True)
            for token in chat_stream(user_input, history):
                print(token, end='', flush=True)
            print()
        except Exception as e:
            print(f"\n[Error]: {e}")


def run_test():
    """Ejecuta preguntas de prueba y muestra respuestas."""
    print("=== Test del Agente (5 preguntas) ===\n")
    history = []
    for i, q in enumerate(TEST_QUERIES, 1):
        print(f"[{i}/{len(TEST_QUERIES)}] {q}")
        try:
            resp, history = chat(q, history, verbose=True)
            print(f"Respuesta: {resp[:300]}...\n" if len(resp) > 300 else f"Respuesta: {resp}\n")
        except Exception as e:
            print(f"[ERROR]: {e}\n")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--test',  action='store_true', help='Ejecutar preguntas de prueba')
    ap.add_argument('--query', type=str, default='',  help='Pregunta directa')
    args = ap.parse_args()

    if args.test:
        run_test()
    elif args.query:
        resp, _ = chat(args.query, verbose=True)
        print(resp)
    else:
        run_interactive()
