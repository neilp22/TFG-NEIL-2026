"""
decision_engine.py — Decision Maker determinista del equipo multi-agente.

Fusiona las tres fuentes en UNA decisión final GO / NO_GO:
  1. Confluence score  (verdad cuantitativa, pesos fijos — get_confluence_score)
  2. Risk Manager      (veredicto adversarial — critic_agent.review_analysis)
  3. Trade parameters  (R:R, SL, TP, sizing reales — get_trade_parameters)

La decisión la fija un MOTOR DE REGLAS determinista (`decide`), que es la fuente de
verdad: mismos inputs → misma salida, sin azar, defendible en la memoria del TFG.

`synthesize` añade una 3ª llamada LLM (temp=0) que SOLO redacta el veredicto en
lenguaje natural; el GO/NO_GO se le pasa ya decidido y no puede cambiarlo. Si la
llamada falla, hay un fallback textual determinista.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / 'config' / '.env')

log = logging.getLogger(__name__)

# ── Umbrales de las reglas (centralizados para citar en la memoria) ──────────
RULES = {
    "min_abs_score":   0.30,   # |confluence final_score| mínimo para operar
    "min_rr":          1.5,    # R:R neto mínimo
    "dead_zone_hours": (22, 23),  # horas UTC en las que no se opera (22:00–00:00)
    "min_confidence":  0.50,   # por debajo → solo aviso (no bloquea)
}


def _f(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def decide(confluence: dict, critic: dict | None,
           trade_params: dict | None, now_utc: datetime | None = None) -> dict:
    """
    Motor de reglas determinista. Devuelve la decisión final GO/NO_GO con las
    razones (qué reglas se incumplen) y los avisos (caución sin bloquear).
    """
    now = now_utc or datetime.now(timezone.utc)
    reasons:  list[str] = []   # incumplimientos → NO_GO
    cautions: list[str] = []   # avisos → GO con precaución

    if not confluence or confluence.get("error"):
        return {
            "decision": "NO_GO", "direction": None, "score": 0.0,
            "reasons": ["Sin confluence score disponible"], "cautions": [],
            "rules": RULES,
        }

    score      = _f(confluence.get("final_score"))
    label      = confluence.get("label", "NEUTRAL")
    conflict   = bool(confluence.get("conflict"))
    confidence = _f(confluence.get("confidence"))
    price      = _f(confluence.get("price_now"))
    direction  = "long" if score > 0 else "short" if score < 0 else None

    # ── Reglas duras (cualquiera → NO_GO) ────────────────────────────────────
    if direction is None:
        reasons.append("Score neutro (0): sin dirección")
    if abs(score) < RULES["min_abs_score"]:
        reasons.append(f"Score débil |{score:.3f}| < {RULES['min_abs_score']}")
    if conflict:
        note = confluence.get("conflict_note") or "conflicto entre módulos"
        reasons.append(f"Conflicto de módulos: {note}")
    if now.hour in RULES["dead_zone_hours"]:
        reasons.append("Dead zone (22:00–00:00 UTC): liquidez baja")

    crit_verdict = (critic or {}).get("verdict")
    if crit_verdict == "REJECT":
        prob = "; ".join((critic or {}).get("problemas", [])[:2])
        reasons.append(f"Risk Manager: REJECT{(' — ' + prob) if prob else ''}")

    rr = None
    if trade_params and not trade_params.get("error"):
        rr = _f(trade_params.get("net_rr_ratio") or trade_params.get("risk_reward_gross"))
        if rr and rr < RULES["min_rr"]:
            reasons.append(f"R:R neto {rr:.2f} < {RULES['min_rr']}")

    # ── Avisos (no bloquean) ─────────────────────────────────────────────────
    if crit_verdict == "CAUTION":
        cautions.append("Risk Manager emite caución")
    if critic and critic.get("citas_ok") is False:
        cautions.append("El análisis tiene cifras sin cita de tool")
    if confidence and confidence < RULES["min_confidence"]:
        cautions.append(f"Confianza de confluencia baja ({confidence:.2f})")

    decision = "NO_GO" if reasons else "GO"

    out = {
        "decision":   decision,
        "direction":  direction,
        "score":      round(score, 3),
        "label":      label,
        "confidence": round(confidence, 2),
        "conflict":   conflict,
        "rr":         round(rr, 2) if rr else None,
        "price":      price,
        "reasons":    reasons,
        "cautions":   cautions,
        "critic_verdict": crit_verdict,
        "rules":      RULES,
        "ts":         now.isoformat(),
    }

    # Niveles de ejecución (si hay trade_params válidos y dirección)
    if trade_params and not trade_params.get("error") and direction:
        out.update({
            "entry":       trade_params.get("entry") or trade_params.get("current_price") or price,
            "stop_loss":   trade_params.get("stop_loss"),
            "take_profit": trade_params.get("take_profit"),
            "size_usd":    trade_params.get("position_size_usd"),
            "size_btc":    trade_params.get("position_size_btc"),
        })
    return out


def _fallback_verdict(d: dict) -> str:
    """Veredicto textual determinista (si la llamada LLM falla)."""
    if d["decision"] == "GO":
        base = (f"GO {(d.get('direction') or '').upper()}: confluencia {d.get('label')} "
                f"({d.get('score')}), R:R {d.get('rr')}.")
        if d.get("cautions"):
            base += " Avisos: " + "; ".join(d["cautions"]) + "."
        return base
    return "NO-GO: " + "; ".join(d.get("reasons", ["sin condiciones de entrada"])) + "."


def synthesize(decision_obj: dict, analysis_text: str = "",
               critic: dict | None = None, model: str = "gpt-4o") -> str:
    """
    3ª llamada (Decision Maker LLM, temp=0). Redacta el veredicto final en
    español integrando analista + risk manager. El GO/NO_GO se le pasa YA
    decidido por `decide` y NO puede cambiarlo. Determinista (temp=0) y
    fail-safe (fallback textual si falla).
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            return _fallback_verdict(decision_obj)
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        system = (
            "Eres el JEFE DE MESA (Decision Maker) de un equipo de trading de BTC. "
            "La decisión final GO/NO_GO YA ha sido tomada por un motor de reglas "
            "determinista y es INMUTABLE: NO la cambies, NO la contradigas. Tu único "
            "trabajo es redactar el veredicto final en 2-4 frases en español, "
            "integrando el análisis del analista y la auditoría del Risk Manager, y "
            "explicando con claridad POR QUÉ la decisión es la que es. Si es GO, indica "
            "dirección y niveles; si es NO_GO, deja claro qué falla. Sé conciso y directo."
        )
        user = (
            f"DECISIÓN DEL MOTOR (inmutable): {decision_obj.get('decision')}\n"
            f"Dirección: {decision_obj.get('direction')}\n"
            f"Confluence: {decision_obj.get('label')} ({decision_obj.get('score')}), "
            f"confianza {decision_obj.get('confidence')}, conflict={decision_obj.get('conflict')}\n"
            f"R:R: {decision_obj.get('rr')} | Entry {decision_obj.get('entry')} "
            f"SL {decision_obj.get('stop_loss')} TP {decision_obj.get('take_profit')}\n"
            f"Razones NO_GO: {decision_obj.get('reasons')}\n"
            f"Avisos: {decision_obj.get('cautions')}\n"
            f"Risk Manager: {(critic or {}).get('verdict')} — "
            f"{(critic or {}).get('resumen','')}\n\n"
            f"ANÁLISIS DEL ANALISTA (extracto):\n{(analysis_text or '')[:3000]}\n\n"
            f"Redacta el veredicto final."
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.0,
            timeout=30,
        )
        txt = (resp.choices[0].message.content or "").strip()
        return txt or _fallback_verdict(decision_obj)
    except Exception as e:
        log.warning("decision_engine.synthesize: %s", e)
        return _fallback_verdict(decision_obj)
