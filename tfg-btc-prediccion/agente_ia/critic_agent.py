"""
critic_agent.py — Risk Manager del equipo multi-agente (rediseño profesional).

Separación estricta de responsabilidades para que sea fiable en condiciones reales:

  • VERIFICACIÓN  → 100% en CÓDIGO determinista (`_deterministic_problems`).
    El veredicto (APPROVE/CAUTION/REJECT) y la lista de `problemas` NUNCA los
    decide el LLM: salen de reglas sobre el ground truth (confluence score, R:R,
    dead zone, conflict) + una verificación de citas HÍBRIDA (solo se comprueba lo
    que tiene ground truth real: el precio actual citado vs el de mercado). Así no
    hay falsos positivos tautológicos ("el precio es X (debería ser X)").

  • OPINIÓN      → LLM (gpt-4o) SOLO redacta el contraargumento adversarial y una
    nota de riesgo cualitativa. Tiene PROHIBIDO verificar números, citas o precios.
    Es donde un risk manager real aporta valor: decir por qué te puedes equivocar.

Si el LLM falla, el veredicto determinista (lo importante) se mantiene igual.
"""
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / 'config' / '.env')

log = logging.getLogger(__name__)

# Umbrales (alineados con decision_engine.RULES para coherencia entre agentes)
MIN_ABS_SCORE = 0.30
MIN_RR        = 1.5
DEAD_ZONE_H   = (22, 23)


# ── Utilidades de parseo ────────────────────────────────────────────────────

def _nums(s: str) -> list:
    out = []
    for tok in re.findall(r'[0-9][0-9.,]*', s or ''):
        t = tok.replace(',', '').rstrip('.')
        try:
            out.append(float(t))
        except ValueError:
            pass
    return out


def _f(x, d=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def _uses_ml_as_signal(text: str) -> bool:
    """True solo si el analista usa el ML como señal FUERTE y SIN el aviso de que
    no es significativo. Conservador (requiere las 3 condiciones) → casi no
    genera falsos positivos."""
    low = (text or '').lower()
    ml     = ('modelo ml', 'machine learning', 'xgboost', 'predicción ml', 'predicción del modelo',
              'el modelo predice', 'run_ml_prediction', 'según el ml', 'el ml ')
    strong = ('confirma', 'valida', 'señal fuerte', 'alta probabilidad', 'indica claramente',
              'garantiza', 'asegura')
    caveat = ('no significativ', 'no es significativ', 'auc', 'limitaci', 'poco fiable',
              'no estadís', 'un dato más')
    return (any(t in low for t in ml) and any(s in low for s in strong)
            and not any(c in low for c in caveat))


def _current_price_mismatch(text: str, price_now: float):
    """Verificación de cita HÍBRIDA: comprueba SOLO el precio actual que cita el
    analista contra el de mercado (lo único con ground truth claro). Devuelve un
    texto de problema si difieren >0.3%, o None. No toca SL/TP/niveles (son
    precios legítimamente distintos)."""
    if not price_now:
        return None
    m = re.search(
        r'(precio actual|precio de mercado|precio spot|current price)[^\d$]{0,25}\$?\s*([\d][\d.,]*)',
        text or '', re.IGNORECASE)
    if not m:
        return None
    vals = _nums('$' + m.group(2))
    if not vals:
        return None
    v = vals[0]
    if v > 1000 and abs(v - price_now) / price_now > 0.003:
        return (f'El precio actual citado (${v:,.0f}) no coincide con el de mercado '
                f'(${price_now:,.0f})')
    return None


# ── Capa determinista: verdict + problemas (NUNCA los decide el LLM) ─────────

def _deterministic_problems(confluence: dict, trade_params: dict | None,
                            analysis_text: str, ref_price: float | None,
                            now: datetime) -> list:
    """Devuelve [{'text': str, 'hard': bool}]. 'hard' = motivo de veto (REJECT)."""
    problems = []
    score      = _f(confluence.get('final_score'))
    conflict   = bool(confluence.get('conflict'))
    price_now  = _f(confluence.get('price_now')) or _f(ref_price)

    if conflict:
        note = confluence.get('conflict_note') or 'módulos en conflicto'
        problems.append({'text': f'Conflicto de módulos no resuelto: {note}', 'hard': True})
    if abs(score) < MIN_ABS_SCORE:
        problems.append({'text': f'Score de confluencia débil ({score:+.3f}; |x| < {MIN_ABS_SCORE}): '
                                 f'señal poco fiable para operar', 'hard': True})
    if now.hour in DEAD_ZONE_H:
        problems.append({'text': 'Dead zone (22:00–00:00 UTC): liquidez baja, evitar entradas',
                         'hard': True})
    if trade_params and not trade_params.get('error'):
        rr = _f(trade_params.get('net_rr_ratio') or trade_params.get('risk_reward_gross'))
        if rr and rr < MIN_RR:
            problems.append({'text': f'R:R neto {rr:.2f} por debajo del mínimo {MIN_RR}:1',
                             'hard': True})
    if _uses_ml_as_signal(analysis_text):
        problems.append({'text': 'Usa el modelo ML como señal válida (está marcado NO '
                                 'significativo, AUC≈0.53)', 'hard': False})
    mismatch = _current_price_mismatch(analysis_text, price_now)
    if mismatch:
        problems.append({'text': mismatch, 'hard': False})
    return problems


def _verdict_from(problems: list) -> str:
    if any(p['hard'] for p in problems):
        return 'REJECT'
    return 'CAUTION' if problems else 'APPROVE'


# ── Capa LLM: SOLO contraargumento adversarial (gpt-4o) ─────────────────────

_OPINION_SYSTEM = """Eres el RISK MANAGER de una mesa de trading de BTC. La verificación
numérica y de reglas YA está hecha por código; NO es tu trabajo. Tu ÚNICO trabajo es el
juicio adversarial cualitativo.

PROHIBIDO ABSOLUTAMENTE:
- Verificar, comparar o citar números, precios, niveles o "el precio debería ser X".
- Hablar de citas o fuentes de cifras.
- Repetir los datos del análisis.

TU TAREA:
- Formula el MEJOR caso EN CONTRA de la conclusión del analista (si propone long,
  argumenta el short; si dice esperar, argumenta la oportunidad perdida, y viceversa).
- Razona sobre el mercado: estructura, contexto, qué podría salir mal, qué no está
  considerando. En 2-4 frases, en español, concreto y útil.

Responde SOLO con JSON válido:
{
  "contraargumento": "el mejor caso en contra, 2-4 frases",
  "riesgo": "bajo" | "medio" | "alto",
  "nota": "una frase de cierre con tu lectura de riesgo cualitativa"
}"""


def _llm_opinion(analysis_text: str, confluence: dict, model: str) -> dict:
    try:
        api_key = os.getenv('OPENAI_API_KEY', '')
        if not api_key:
            return {}
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        compact = {
            'label':      confluence.get('label'),
            'score':      confluence.get('final_score'),
            'confidence': confluence.get('confidence'),
            'conflict':   confluence.get('conflict'),
        }
        user = (f"Confluence (contexto, NO lo cites): {json.dumps(compact, ensure_ascii=False)}\n\n"
                f"ANÁLISIS DEL ANALISTA:\n{(analysis_text or '')[:8000]}\n\n"
                f"Da tu contraargumento adversarial.")
        resp = client.chat.completions.create(
            model=model,
            messages=[{'role': 'system', 'content': _OPINION_SYSTEM},
                      {'role': 'user', 'content': user}],
            temperature=0.3,
            response_format={'type': 'json_object'},
            timeout=45,
        )
        data = json.loads(resp.choices[0].message.content)
        return {
            'contraargumento': str(data.get('contraargumento') or '')[:600],
            'riesgo':          str(data.get('riesgo') or '').lower(),
            'nota':            str(data.get('nota') or '')[:300],
        }
    except Exception as e:
        log.warning("critic_agent._llm_opinion: %s", e)
        return {}


# ── API pública ─────────────────────────────────────────────────────────────

def review_analysis(analysis_html: str, confluence: dict = None,
                    ref_price: float = None, trade_params: dict = None,
                    model: str = 'gpt-4o', now_utc: datetime = None) -> dict:
    """
    Audita el análisis. Veredicto y problemas son DETERMINISTAS (código); el LLM
    (gpt-4o) solo añade el contraargumento. Nunca lanza.
    """
    try:
        if not confluence or confluence.get('error'):
            return {'error': 'sin confluence score para auditar'}
        now = now_utc or datetime.now(timezone.utc)
        analysis = (analysis_html or '')

        # 1. Verificación determinista
        problems = _deterministic_problems(confluence, trade_params, analysis, ref_price, now)
        verdict  = _verdict_from(problems)
        problemas_txt = [p['text'] for p in problems]
        citas_ok = not any('no coincide con el de mercado' in p['text'] for p in problems)

        # 2. Opinión cualitativa (gpt-4o) — solo si hay análisis que valorar
        op = _llm_opinion(analysis, confluence, model) if analysis.strip() else {}

        # 3. Resumen determinista (no depende del LLM)
        if verdict == 'APPROVE':
            resumen = 'Conclusión coherente con los datos y las reglas; sin incumplimientos.'
        elif verdict == 'CAUTION':
            resumen = 'Válido pero con avisos: ' + '; '.join(problemas_txt) + '.'
        else:
            hard = [p['text'] for p in problems if p['hard']]
            resumen = 'Riesgo elevado / no operar: ' + '; '.join(hard) + '.'
        if op.get('nota'):
            resumen = (resumen + ' ' + op['nota'])[:600]

        # confianza: alta cuando el veredicto es determinista y claro
        confidence = 0.9 if verdict in ('APPROVE', 'REJECT') else 0.7

        return {
            'verdict':         verdict,
            'confidence':      confidence,
            'citas_ok':        citas_ok,
            'problemas':       problemas_txt,
            'contraargumento': op.get('contraargumento', ''),
            'riesgo':          op.get('riesgo', ''),
            'resumen':         resumen,
            'model':           model,
        }
    except Exception as e:
        log.warning("critic_agent.review_analysis: %s", e)
        return {'error': str(e)}
