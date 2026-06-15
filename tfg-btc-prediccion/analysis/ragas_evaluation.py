"""
analysis/ragas_evaluation.py
============================

Framework de evaluación tipo RAGAS para el agente conversacional BTC
(`agente_ia/06_agent.py`). Mide tres métricas inspiradas en RAGAS:

  1. Faithfulness        -> LLM-as-judge (gpt-4o-mini): % de claims
                            verificables en los contextos recuperados.
  2. Answer Relevancy    -> cosine-sim entre embeddings (all-MiniLM-L6-v2)
                            de la pregunta y la respuesta del agente.
  3. Context Precision   -> LLM-as-judge: % de contextos recuperados
                            que son útiles para responder a la pregunta.

Notas
-----
*   El "contexto" recuperado se extrae de las tools que llama el agente
    (en particular `rag_search`). Si el agente no llama a ninguna tool
    relevante de contexto, las métricas de contexto se omiten (NaN).
*   Si `agente_ia/rag_pipeline.py` (FAISS) no existe, no es bloqueante
    -- el agente seguirá usando `rag_search` por pgvector / fallback
    textual y la evaluación funciona igual; se anota como limitación.
*   El LLM-as-judge usa gpt-4o-mini con caché en disco
    (`results/ragas_llm_cache.json`) para evitar recomputar entre runs.

Uso
---
    python analysis/ragas_evaluation.py [--n 20]
"""

from __future__ import annotations

# IMPORTANT: disable TF backend before any HuggingFace import (Keras 3
# incompatibility on this machine). Sin esto, sentence-transformers
# explota al importarse.
import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import sys
import json
import time
import hashlib
import logging
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# ── Paths y .env ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_PATH   = PROJECT_ROOT / "agente_ia" / "06_agent.py"
RESULTS_DIR  = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
CSV_OUT      = RESULTS_DIR / "ragas_evaluation.csv"
CACHE_PATH   = RESULTS_DIR / "ragas_llm_cache.json"

load_dotenv(PROJECT_ROOT / "config" / ".env")

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "agente_ia"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("ragas_eval")

# Silenciar el resto (httpx, openai, urllib3, etc.) para no ensuciar el log.
for noisy in ("httpx", "openai", "urllib3", "sentence_transformers",
              "transformers", "huggingface_hub"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


# ── Coste estimado (gpt-4o-mini) ──────────────────────────────────────────────
# Tarifa: $0.150 / 1M input tokens, $0.600 / 1M output tokens
PRICE_IN_PER_TOK  = 0.150 / 1_000_000
PRICE_OUT_PER_TOK = 0.600 / 1_000_000
COST_ALERT_USD    = 3.0   # avisar si superamos esto

_total_cost_usd = 0.0
_total_in_tok   = 0
_total_out_tok  = 0


def _track_usage(usage) -> None:
    """Actualiza el coste estimado total a partir de un objeto usage de OpenAI."""
    global _total_cost_usd, _total_in_tok, _total_out_tok
    if usage is None:
        return
    in_tok  = getattr(usage, "prompt_tokens", 0) or 0
    out_tok = getattr(usage, "completion_tokens", 0) or 0
    _total_in_tok  += in_tok
    _total_out_tok += out_tok
    _total_cost_usd += in_tok * PRICE_IN_PER_TOK + out_tok * PRICE_OUT_PER_TOK
    if _total_cost_usd > COST_ALERT_USD:
        log.warning(f"[COSTE] Superado umbral ${COST_ALERT_USD}: actual ${_total_cost_usd:.2f}")


# ── Carga del agente (importlib, por prefijo numérico) ────────────────────────

def _load_agent_module():
    spec = importlib.util.spec_from_file_location("agent_06", AGENT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Carga del cliente OpenAI compartido (para LLM-as-judge) ───────────────────

def _get_openai_client():
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY no configurado en config/.env")
    return OpenAI(api_key=api_key)


# ── Cache en disco para LLM-as-judge ──────────────────────────────────────────

def _load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            log.warning("Cache ragas corrupta, reiniciando.")
    return {}


def _save_cache(cache: dict) -> None:
    try:
        CACHE_PATH.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        log.warning(f"No se pudo guardar cache: {e}")


def _cache_key(kind: str, *parts: str) -> str:
    h = hashlib.sha256()
    h.update(kind.encode())
    for p in parts:
        h.update(b"||")
        h.update((p or "").encode("utf-8", errors="replace"))
    return f"{kind}:{h.hexdigest()[:24]}"


_CACHE: dict = _load_cache()


# ── Embeddings (cache en proceso) ─────────────────────────────────────────────

_embedder = None
_emb_cache: dict[str, np.ndarray] = {}


def _get_embedder():
    global _embedder
    if _embedder is None:
        log.info("Cargando sentence-transformers/all-MiniLM-L6-v2 ...")
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedder


def _embed(text: str) -> np.ndarray:
    text = (text or "").strip() or " "
    if text in _emb_cache:
        return _emb_cache[text]
    vec = _get_embedder().encode([text], normalize_embeddings=True)[0]
    _emb_cache[text] = vec
    return vec


# ── Test set ──────────────────────────────────────────────────────────────────

def build_test_set() -> list[dict]:
    """Devuelve una lista de preguntas de prueba con su categoría."""
    return [
        # ── price (4) ─────────────────────────────────────────────────────
        {"question": "¿Cuál es el precio actual de Bitcoin y cómo ha variado en las últimas 24 horas?",
         "expected_topics": ["precio", "cambio %", "BTC"], "category": "price"},
        {"question": "¿Qué tendencia muestran las EMAs en el timeframe de 1 hora para BTC?",
         "expected_topics": ["EMA", "tendencia", "1h"], "category": "price"},
        {"question": "¿En qué zona de las Bollinger Bands se encuentra BTC ahora mismo?",
         "expected_topics": ["bollinger", "bb_upper", "bb_lower"], "category": "price"},
        {"question": "¿Cómo está el RSI actual de Bitcoin y qué implica?",
         "expected_topics": ["RSI", "sobrecompra", "sobreventa"], "category": "price"},

        # ── sentiment (4) ─────────────────────────────────────────────────
        {"question": "¿Cómo está el sentimiento de las noticias de Bitcoin en los últimos 3 días?",
         "expected_topics": ["sentimiento", "FinBERT", "noticias"], "category": "sentiment"},
        {"question": "¿Cuál es el Fear and Greed Index actual de Bitcoin?",
         "expected_topics": ["fear", "greed", "índice"], "category": "sentiment"},
        {"question": "¿Ha mejorado el sentimiento de las noticias respecto a la semana pasada?",
         "expected_topics": ["sentimiento", "evolución", "comparación"], "category": "sentiment"},
        {"question": "¿Cuántas noticias se han procesado hoy y cuál es el score medio?",
         "expected_topics": ["número de noticias", "media", "FinBERT"], "category": "sentiment"},

        # ── ict (5) ───────────────────────────────────────────────────────
        {"question": "¿Hay algún Order Block relevante cercano al precio actual?",
         "expected_topics": ["order block", "OB", "distancia"], "category": "ict"},
        {"question": "¿Estamos actualmente dentro de una killzone ICT?",
         "expected_topics": ["killzone", "sesión", "ICT"], "category": "ict"},
        {"question": "¿Hay Fair Value Gaps no mitigados en BTC 1h?",
         "expected_topics": ["FVG", "fair value gap", "mitigado"], "category": "ict"},
        {"question": "¿Cuál es el bias multitimeframe combinando 1h, 4h y 1d?",
         "expected_topics": ["multitimeframe", "bias", "1h", "4h", "1d"], "category": "ict"},
        {"question": "¿Dónde está la zona de entrada ICT para un long y cuál sería el stop loss?",
         "expected_topics": ["entrada", "stop", "long"], "category": "ict"},

        # ── macro (3) ─────────────────────────────────────────────────────
        {"question": "¿Cómo podría afectar una subida de tipos de la Fed al precio de Bitcoin?",
         "expected_topics": ["Fed", "tipos", "macro"], "category": "macro"},
        {"question": "¿Qué impacto tiene el dólar fuerte (DXY) sobre Bitcoin?",
         "expected_topics": ["DXY", "dólar", "correlación"], "category": "macro"},
        {"question": "¿Qué eventos macro relevantes podrían mover Bitcoin esta semana?",
         "expected_topics": ["eventos", "macro", "calendario"], "category": "macro"},

        # ── ml_prediction (4) ─────────────────────────────────────────────
        {"question": "¿Qué predice el modelo ensemble para Bitcoin en las próximas horas?",
         "expected_topics": ["modelo", "ensemble", "predicción", "probabilidad"],
         "category": "ml_prediction"},
        {"question": "¿Cuál es la confianza del modelo ML actualmente y es estadísticamente significativa?",
         "expected_topics": ["AUC", "confianza", "significancia"], "category": "ml_prediction"},
        {"question": "¿Cuál es el confluence score actual de BTC y qué módulos lo dominan?",
         "expected_topics": ["confluence", "score", "módulos"], "category": "ml_prediction"},
        {"question": "Dame los parámetros de trade (entry, SL, TP, sizing) para un long ahora mismo.",
         "expected_topics": ["entry", "stop", "take profit", "sizing"], "category": "ml_prediction"},

        # ── rag_news (5) ──────────────────────────────────────────────────
        {"question": "Busca noticias recientes sobre regulación crypto y ETFs de Bitcoin.",
         "expected_topics": ["regulación", "ETF", "SEC"], "category": "rag_news"},
        {"question": "¿Qué se ha dicho últimamente sobre el halving de Bitcoin?",
         "expected_topics": ["halving", "minería", "recompensa"], "category": "rag_news"},
        {"question": "¿Hay noticias sobre la adopción institucional de Bitcoin?",
         "expected_topics": ["institucional", "adopción", "BlackRock"], "category": "rag_news"},
        {"question": "Busca noticias sobre hackeos o exploits en exchanges crypto.",
         "expected_topics": ["hackeo", "exploit", "exchange"], "category": "rag_news"},
        {"question": "¿Qué noticias hay sobre la macro de Bitcoin y la inflación?",
         "expected_topics": ["inflación", "macro", "CPI"], "category": "rag_news"},
    ]


# ── Ejecutar el agente y capturar contextos / tools ───────────────────────────

# Tools cuyas salidas tratamos como "contextos recuperados" (no solo rag_search,
# también las tools que devuelven evidencia de datos). Esto sigue el espíritu
# de RAGAS: los "contexts" son la evidencia que el sistema usa para responder.
CONTEXT_TOOLS_PRIMARY = {"rag_search"}
CONTEXT_TOOLS_AUX = {
    "query_market", "get_sentiment", "get_fear_greed", "get_ict_context",
    "get_session_stats", "get_technical_levels", "get_multi_timeframe_bias",
    "get_volume_profile", "get_confluence_score", "get_entry_zone",
    "get_trade_parameters", "run_ml_prediction",
}


def _extract_rag_snippets(tool_result_str: str, max_chars: int = 500) -> list[str]:
    """Extrae fragmentos de texto del resultado de rag_search."""
    try:
        obj = json.loads(tool_result_str)
    except Exception:
        return [tool_result_str[:max_chars]]
    snippets: list[str] = []
    if isinstance(obj, dict):
        results = obj.get("results") or []
        for r in results:
            if isinstance(r, dict):
                txt = r.get("text") or r.get("content") or ""
                src = r.get("source", "")
                ts  = r.get("timestamp", "")
                if txt:
                    snippets.append(f"[{src} {ts}] {txt[:max_chars]}")
    if not snippets:
        snippets.append(json.dumps(obj, ensure_ascii=False)[:max_chars])
    return snippets


def _summarize_tool_result(name: str, tool_result_str: str, max_chars: int = 400) -> str:
    """Resumen breve de un tool result no-rag para tratarlo como contexto auxiliar."""
    s = tool_result_str.strip()
    if len(s) > max_chars:
        s = s[:max_chars] + "..."
    return f"[{name}] {s}"


def run_agent_on_question(question: str, agent_mod, timeout_s: float = 90.0) -> dict:
    """Ejecuta el agente sobre una pregunta y captura tools + contextos.

    No usa timeouts duros (SIGALRM no es seguro multi-thread); se confía en
    el cap `MAX_TOOL_CALLS=8` del propio agente para limitar duración.
    """
    t0 = time.time()
    record = {
        "question":            question,
        "answer":              "",
        "retrieved_contexts":  [],
        "rag_contexts":        [],   # solo rag_search
        "tools_used":          [],
        "duration_s":          None,
        "error":               None,
    }
    try:
        answer, history = agent_mod.chat(question, history=None, verbose=False)
        record["answer"] = answer or ""

        # Extraer tools usadas + contextos a partir del historial
        # El historial tiene msgs role=assistant con `tool_calls` (lista) y
        # msgs role=tool con `content` (la salida JSON de la tool).
        pending_calls: dict[str, str] = {}  # tool_call_id -> nombre
        for m in history:
            if not isinstance(m, dict):
                continue
            if m.get("role") == "assistant":
                for tc in (m.get("tool_calls") or []):
                    fn = (tc.get("function") or {}).get("name") or tc.get("name") or "?"
                    tcid = tc.get("id") or ""
                    if tcid:
                        pending_calls[tcid] = fn
                    record["tools_used"].append(fn)
            elif m.get("role") == "tool":
                tcid = m.get("tool_call_id") or ""
                fn = pending_calls.get(tcid, "?")
                content = m.get("content") or ""
                if not isinstance(content, str):
                    content = json.dumps(content, ensure_ascii=False, default=str)
                if fn in CONTEXT_TOOLS_PRIMARY:
                    snips = _extract_rag_snippets(content)
                    record["rag_contexts"].extend(snips)
                    record["retrieved_contexts"].extend(snips)
                elif fn in CONTEXT_TOOLS_AUX:
                    record["retrieved_contexts"].append(_summarize_tool_result(fn, content))

    except Exception as e:
        record["error"] = f"{type(e).__name__}: {e}"
        log.warning(f"Agente falló en '{question[:60]}...': {record['error']}")

    record["duration_s"] = round(time.time() - t0, 2)
    return record


# ── LLM-as-judge helpers ──────────────────────────────────────────────────────

def _llm_json(client, system: str, user: str, model: str = "gpt-4o-mini",
              max_tokens: int = 250) -> dict | None:
    """Llama al LLM pidiendo JSON. Devuelve dict o None si parseo falla."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=max_tokens,
        )
        _track_usage(resp.usage)
        content = resp.choices[0].message.content or "{}"
        return json.loads(content)
    except json.JSONDecodeError:
        log.warning("LLM judge: JSON inválido")
        return None
    except Exception as e:
        log.warning(f"LLM judge falló: {e}")
        return None


def evaluate_faithfulness(question: str, answer: str, contexts: list[str],
                          client=None) -> float:
    """LLM-as-judge: ¿qué fracción de claims de la respuesta están soportados
    por los contextos? Score [0,1]. Si no hay contextos -> NaN.
    """
    if not contexts or not (answer or "").strip():
        return float("nan")

    ckey = _cache_key("faith", question, answer, "\n---\n".join(contexts))
    if ckey in _CACHE:
        return float(_CACHE[ckey])

    if client is None:
        client = _get_openai_client()

    ctx_blob = "\n\n".join(f"[CTX {i+1}] {c}" for i, c in enumerate(contexts[:12]))
    sys_msg = (
        "You are an evaluator. Decide if every factual claim in the ANSWER "
        "is supported by the provided CONTEXTS. Ignore disclaimers, opinions, "
        "and stylistic phrases. A claim is 'supported' if a reasonable reader "
        "would find evidence for it in the contexts. Output strict JSON only."
    )
    user_msg = (
        f"QUESTION:\n{question}\n\n"
        f"ANSWER:\n{answer}\n\n"
        f"CONTEXTS:\n{ctx_blob}\n\n"
        "Return JSON: {\"total_claims\": int, \"supported_claims\": int, "
        "\"score\": float between 0 and 1, \"reason\": str}. "
        "Score = supported_claims / max(total_claims,1)."
    )
    obj = _llm_json(client, sys_msg, user_msg)
    if obj is None:
        return float("nan")
    score = obj.get("score")
    if score is None:
        tot = max(int(obj.get("total_claims") or 0), 0)
        sup = max(int(obj.get("supported_claims") or 0), 0)
        score = (sup / tot) if tot > 0 else float("nan")
    try:
        score = max(0.0, min(1.0, float(score)))
    except Exception:
        return float("nan")
    _CACHE[ckey] = score
    return score


def evaluate_answer_relevancy(question: str, answer: str, client=None,
                              n_generated: int = 3) -> float:
    """Answer Relevancy (RAGAS canónico) por question generation.

    En lugar de calcular cosine(question, answer) directamente — que
    penaliza respuestas largas/estructuradas — pedimos a un LLM que
    GENERE `n_generated` preguntas que pudieron haber producido esta
    `answer`. Luego promediamos las similitudes entre la pregunta
    original y cada pregunta generada.

    Esto desacopla el score del estilo/longitud de la respuesta y mide
    de verdad si la respuesta es RELEVANTE para la pregunta.

    Score [0,1]. Si la respuesta está vacía -> 0.
    """
    if not (answer or "").strip():
        return 0.0

    # Truncar answer si es muy larga (HTML del agente puede ser >2000 toks)
    answer_for_llm = answer.strip()
    if len(answer_for_llm) > 4000:
        answer_for_llm = answer_for_llm[:4000] + "..."

    ckey = _cache_key("ansrel_qgen", question, answer_for_llm, str(n_generated))
    if ckey in _CACHE:
        return float(_CACHE[ckey])

    if client is None:
        try:
            client = _get_openai_client()
        except Exception:
            # Fallback al método cosine simple si no hay cliente disponible
            q_vec = _embed(question)
            a_vec = _embed(answer)
            return max(0.0, min(1.0, float(np.dot(q_vec, a_vec))))

    sys_msg = (
        "You generate plausible USER questions that would lead to the given "
        "assistant ANSWER. The questions must be short (under 25 words), in "
        "the same language as the answer, and reflect what the user actually "
        "wanted to know. Output strict JSON only."
    )
    user_msg = (
        f"ANSWER:\n{answer_for_llm}\n\n"
        f"Generate {n_generated} plausible user questions that this answer "
        f"directly addresses. Return strict JSON:\n"
        f"{{\"questions\": [\"q1\", \"q2\", \"q3\"]}}"
    )
    obj = _llm_json(client, sys_msg, user_msg, max_tokens=300)
    if obj is None or not isinstance(obj.get("questions"), list):
        # Fallback al método cosine simple
        q_vec = _embed(question)
        a_vec = _embed(answer)
        sim = max(0.0, min(1.0, float(np.dot(q_vec, a_vec))))
        _CACHE[ckey] = sim
        return sim

    generated = [str(q).strip() for q in obj["questions"] if str(q).strip()]
    if not generated:
        q_vec = _embed(question)
        a_vec = _embed(answer)
        sim = max(0.0, min(1.0, float(np.dot(q_vec, a_vec))))
        _CACHE[ckey] = sim
        return sim

    q_vec = _embed(question)
    sims  = []
    for gq in generated[:n_generated]:
        g_vec = _embed(gq)
        sims.append(float(np.dot(q_vec, g_vec)))

    score = max(0.0, min(1.0, float(np.mean(sims)) if sims else 0.0))
    _CACHE[ckey] = score
    return score


def evaluate_context_precision(question: str, contexts: list[str],
                               client=None) -> float:
    """LLM-as-judge: % de contextos que son útiles para responder la pregunta."""
    if not contexts:
        return float("nan")

    ckey = _cache_key("ctxprec", question, "\n---\n".join(contexts))
    if ckey in _CACHE:
        return float(_CACHE[ckey])

    if client is None:
        client = _get_openai_client()

    ctx_blob = "\n\n".join(f"[CTX {i+1}] {c}" for i, c in enumerate(contexts[:12]))
    sys_msg = (
        "You are an evaluator of retrieval quality. For each CONTEXT, decide "
        "whether it contains information useful to answer the QUESTION. "
        "Output strict JSON only."
    )
    user_msg = (
        f"QUESTION:\n{question}\n\n"
        f"CONTEXTS (n={min(len(contexts), 12)}):\n{ctx_blob}\n\n"
        "Return JSON: {\"per_context_useful\": [0 or 1, ...], "
        "\"score\": float between 0 and 1, \"reason\": str}. "
        "Score = mean(per_context_useful)."
    )
    obj = _llm_json(client, sys_msg, user_msg, max_tokens=300)
    if obj is None:
        return float("nan")
    score = obj.get("score")
    if score is None:
        flags = obj.get("per_context_useful") or []
        score = (sum(int(bool(x)) for x in flags) / len(flags)) if flags else float("nan")
    try:
        score = max(0.0, min(1.0, float(score)))
    except Exception:
        return float("nan")
    _CACHE[ckey] = score
    return score


# ── Loop principal ────────────────────────────────────────────────────────────

def run_evaluation(n_questions: int = 20) -> pd.DataFrame:
    test_set = build_test_set()
    test_set = test_set[: max(1, n_questions)]
    log.info(f"Evaluando {len(test_set)} preguntas sobre el agente BTC...")

    # Detectar si rag_pipeline existe; si no, anotamos limitación.
    rag_pipeline_exists = (PROJECT_ROOT / "agente_ia" / "rag_pipeline.py").exists()
    if not rag_pipeline_exists:
        log.warning("agente_ia/rag_pipeline.py no existe -> sin FAISS dedicado. "
                    "Se usa el rag_search del agente (pgvector / fallback textual).")

    agent_mod = _load_agent_module()
    client    = _get_openai_client()

    # Pre-cargar embedder para evitar latencia en la primera medición
    _get_embedder()

    rows: list[dict] = []
    t_global = time.time()
    for i, item in enumerate(test_set, 1):
        q   = item["question"]
        cat = item["category"]
        log.info(f"[{i}/{len(test_set)}] ({cat}) {q}")

        rec = run_agent_on_question(q, agent_mod)
        answer    = rec["answer"]
        contexts  = rec["retrieved_contexts"]
        rag_only  = rec["rag_contexts"]
        tools     = rec["tools_used"]
        dur       = rec["duration_s"]

        # Métricas
        try:
            faith = evaluate_faithfulness(q, answer, contexts, client=client)
        except Exception as e:
            log.warning(f"faithfulness falló: {e}")
            faith = float("nan")
        try:
            ans_rel = evaluate_answer_relevancy(q, answer, client=client)
        except Exception as e:
            log.warning(f"answer_relevancy falló: {e}")
            ans_rel = float("nan")
        try:
            ctx_prec = evaluate_context_precision(q, contexts, client=client)
        except Exception as e:
            log.warning(f"context_precision falló: {e}")
            ctx_prec = float("nan")

        rows.append({
            "question":           q,
            "category":           cat,
            "faithfulness":       round(faith, 4)    if pd.notna(faith)    else np.nan,
            "answer_relevancy":   round(ans_rel, 4)  if pd.notna(ans_rel)  else np.nan,
            "context_precision":  round(ctx_prec, 4) if pd.notna(ctx_prec) else np.nan,
            "tools_used":         ",".join(tools)     if tools else "",
            "n_contexts":         len(contexts),
            "n_rag_contexts":     len(rag_only),
            "duration_s":         dur,
            "answer_preview":     (answer or "")[:200].replace("\n", " "),
            "error":              rec.get("error") or "",
        })

        # Persistir cache tras cada pregunta (resiliencia ante ctrl-c)
        _save_cache(_CACHE)

    elapsed = time.time() - t_global
    df = pd.DataFrame(rows)
    df.to_csv(CSV_OUT, index=False, encoding="utf-8")

    # Resumen
    print("\n" + "=" * 60)
    print(" RAGAS Summary")
    print("=" * 60)
    print(df[["faithfulness", "answer_relevancy", "context_precision"]].mean().round(4))
    print("\nPor categoría:")
    print(
        df.groupby("category")[
            ["faithfulness", "answer_relevancy", "context_precision"]
        ].mean().round(4)
    )
    print("\nDuración media por pregunta:",
          f"{df['duration_s'].mean():.1f}s" if len(df) else "n/a")
    print(f"Tiempo total: {elapsed:.1f}s")
    print(f"Coste estimado OpenAI (judge + agente parcial*): "
          f"${_total_cost_usd:.4f} "
          f"(in={_total_in_tok}, out={_total_out_tok} tokens)")
    print("  *solo contabiliza llamadas del LLM-as-judge; el agente "
          "tiene su propio uso de tokens.")
    print(f"CSV guardado en: {CSV_OUT}")
    print(f"Cache LLM:        {CACHE_PATH}")
    if not rag_pipeline_exists:
        print("\n[Limitación] No se encontró agente_ia/rag_pipeline.py (FAISS). "
              "Las métricas de contexto usan los outputs de rag_search "
              "(pgvector / fallback) y de las tools auxiliares.")

    return df


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20,
                    help="Número de preguntas a evaluar (default: 20).")
    args = ap.parse_args()

    df = run_evaluation(n_questions=args.n)
    print("\n=== RAGAS Summary (final) ===")
    print(df[["faithfulness", "answer_relevancy", "context_precision"]].mean().round(4))
    print("\nPor categoría:")
    print(
        df.groupby("category")[
            ["faithfulness", "answer_relevancy", "context_precision"]
        ].mean().round(4)
    )
