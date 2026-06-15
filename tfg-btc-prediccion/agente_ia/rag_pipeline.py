"""
rag_pipeline.py — RAG semántico standalone con FAISS

Pipeline RAG self-contained para el agente IA del TFG BTC. Indexa noticias de
`raw_texts` (filtrando sentiment_raw IS NOT NULL) con SentenceTransformer
all-MiniLM-L6-v2 (384-dim) y las almacena en un índice FAISS IndexFlatIP con
embeddings L2-normalizados (=> cosine similarity).

NO usa pgvector (no disponible en la DB local). Persistencia en disco:
    models/saved/rag_faiss.index   — índice FAISS
    models/saved/rag_metadata.json — metadata paralela + info de build

API pública:
    build_index(force_rebuild: bool = False) -> dict
    semantic_search(query, k=5, before_date=None, min_similarity=0.0) -> list[dict]
    get_index_stats() -> dict

Uso:
    python agente_ia/rag_pipeline.py   # build + self-test
"""

from __future__ import annotations

# ── CRITICAL: disable TF/Keras import BEFORE any HuggingFace import ─────────
# Fix Keras 3 / tf-keras conflict cuando Python tiene tensorflow instalado
import os as _os
_os.environ["TRANSFORMERS_NO_TF"] = "1"
_os.environ["USE_TF"] = "0"
_os.environ["TRANSFORMERS_NO_FLAX"] = "1"

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

# ── Project paths ────────────────────────────────────────────────────────────
_THIS_DIR    = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# ── Logging ──────────────────────────────────────────────────────────────────
log = logging.getLogger("rag_pipeline")
if not log.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

# ── Dependencies (fail fast con mensaje claro) ───────────────────────────────
try:
    import numpy as np
except ImportError:
    print("[FATAL] numpy no instalado. Ejecuta: pip install numpy", file=sys.stderr)
    sys.exit(1)

try:
    import faiss  # type: ignore
except ImportError:
    print(
        "[FATAL] faiss no instalado. Ejecuta: pip install faiss-cpu",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:
    print(
        "[FATAL] sentence-transformers no instalado.\n"
        "        Ejecuta: pip install sentence-transformers",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    from sqlalchemy import text
except ImportError:
    print("[FATAL] sqlalchemy no instalado. Ejecuta: pip install sqlalchemy", file=sys.stderr)
    sys.exit(1)

try:
    from db.db_utils import get_engine
except ImportError as e:
    print(f"[FATAL] No se puede importar db.db_utils: {e}", file=sys.stderr)
    sys.exit(1)


# ── Constantes ───────────────────────────────────────────────────────────────
MODEL_NAME      = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384
BATCH_SIZE      = 32
MAX_CHARS       = 2000     # truncado conservador (~512 tokens para MiniLM)
MIN_TEXT_LEN    = 20       # textos < 20 chars se descartan
DEDUP_PREFIX    = 100      # primeros N chars para dedup cuando no hay URL

_SAVED_DIR      = _PROJECT_ROOT / "models" / "saved"
INDEX_PATH      = _SAVED_DIR / "rag_faiss.index"
METADATA_PATH   = _SAVED_DIR / "rag_metadata.json"


# ── Modelo cacheado ──────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Carga (una sola vez) el modelo de embeddings."""
    log.info(f"Cargando modelo {MODEL_NAME} ...")
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME)
    log.info(f"Modelo cargado en {time.time() - t0:.1f}s")
    return model


# ── Helpers internos ─────────────────────────────────────────────────────────
def _ensure_saved_dir() -> None:
    _SAVED_DIR.mkdir(parents=True, exist_ok=True)


def _fetch_corpus() -> list[dict]:
    """
    Lee de raw_texts los registros con sentiment_raw NOT NULL.
    Devuelve lista de dicts con keys: id, text, source, timestamp, url, sentiment.
    Aplica filtrado por longitud y dedup (URL > prefijo 100 chars).
    """
    try:
        engine = get_engine()
    except Exception as e:
        log.error(f"No se pudo obtener engine: {e}")
        raise

    sql = text(
        """
        SELECT id, timestamp, source, text, url, sentiment_raw
        FROM raw_texts
        WHERE sentiment_raw IS NOT NULL
          AND text IS NOT NULL
        ORDER BY timestamp ASC
        """
    )

    rows: list[Any] = []
    with engine.connect() as conn:
        rows = conn.execute(sql).fetchall()

    log.info(f"Filas brutas con sentiment_raw NOT NULL: {len(rows)}")

    corpus: list[dict] = []
    seen_keys: set[str] = set()
    skipped_short = 0
    skipped_dup = 0

    for r in rows:
        rid, ts, src, txt, url, sent = r[0], r[1], r[2], r[3], r[4], r[5]
        if txt is None:
            continue
        txt_clean = txt.strip()
        if len(txt_clean) < MIN_TEXT_LEN:
            skipped_short += 1
            continue

        # Dedup key: prefer URL, fallback prefix
        key = url.strip() if url and url.strip() else txt_clean[:DEDUP_PREFIX]
        if key in seen_keys:
            skipped_dup += 1
            continue
        seen_keys.add(key)

        # Trunc text para embedding (no muta el texto guardado en metadata)
        corpus.append(
            {
                "id":        int(rid),
                "text":      txt_clean,
                "text_emb":  txt_clean[:MAX_CHARS],
                "source":    str(src) if src is not None else "",
                "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                "url":       str(url) if url else "",
                "sentiment": float(sent) if sent is not None else None,
            }
        )

    log.info(
        f"Corpus filtrado: {len(corpus)} docs "
        f"(skipped_short={skipped_short}, skipped_dup={skipped_dup})"
    )
    return corpus


def _embed_corpus(corpus: list[dict]) -> np.ndarray:
    """Genera embeddings normalizados L2 (float32) para el corpus."""
    model = _get_model()
    texts = [d["text_emb"] for d in corpus]
    log.info(f"Generando embeddings (batch={BATCH_SIZE}, n={len(texts)})...")

    t0 = time.time()
    embs = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine via inner product
    )
    log.info(f"Embeddings generados en {time.time() - t0:.1f}s — shape={embs.shape}")

    if embs.dtype != np.float32:
        embs = embs.astype(np.float32)
    return embs


def _build_faiss_index(embs: np.ndarray) -> faiss.Index:
    """IndexFlatIP sobre embeddings L2-normalizados (= cosine similarity)."""
    dim = embs.shape[1]
    if dim != EMBEDDING_DIM:
        log.warning(f"Dimensión inesperada: {dim} (esperado {EMBEDDING_DIM})")
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    log.info(f"Índice FAISS IndexFlatIP construido — ntotal={index.ntotal}")
    return index


def _save(index: faiss.Index, corpus: list[dict]) -> dict:
    """Persiste índice y metadata. Devuelve stats."""
    _ensure_saved_dir()

    faiss.write_index(index, str(INDEX_PATH))
    size_mb = INDEX_PATH.stat().st_size / (1024 * 1024)

    # Metadata: solo lo necesario para search (sin text_emb redundante)
    meta_docs = [
        {
            "id":        d["id"],
            "text":      d["text"],
            "source":    d["source"],
            "timestamp": d["timestamp"],
            "url":       d["url"],
            "sentiment": d["sentiment"],
        }
        for d in corpus
    ]
    payload = {
        "model_name":    MODEL_NAME,
        "embedding_dim": EMBEDDING_DIM,
        "n_docs":        len(meta_docs),
        "build_date":    datetime.now(timezone.utc).isoformat(),
        "index_path":    str(INDEX_PATH),
        "docs":          meta_docs,
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    log.info(f"Índice guardado en {INDEX_PATH} ({size_mb:.2f} MB)")
    log.info(f"Metadata guardada en {METADATA_PATH} ({len(meta_docs)} docs)")
    return {
        "n_docs":        len(meta_docs),
        "index_size_mb": round(size_mb, 3),
        "build_date":    payload["build_date"],
        "model_name":    MODEL_NAME,
        "index_path":    str(INDEX_PATH),
        "metadata_path": str(METADATA_PATH),
    }


@lru_cache(maxsize=1)
def _load_index_and_metadata() -> tuple[faiss.Index | None, dict | None]:
    """Carga (cached) índice FAISS y metadata desde disco."""
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        return None, None
    try:
        index = faiss.read_index(str(INDEX_PATH))
    except Exception as e:
        log.error(f"Error cargando índice FAISS: {e}")
        return None, None
    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        log.error(f"Error cargando metadata: {e}")
        return None, None
    return index, meta


def _invalidate_cache() -> None:
    _load_index_and_metadata.cache_clear()


# ── API pública ──────────────────────────────────────────────────────────────
def build_index(force_rebuild: bool = False) -> dict:
    """
    Genera embeddings de todos los textos con sentiment_raw NOT NULL y persiste
    índice FAISS + metadata JSON.

    Args:
        force_rebuild: si False y el índice ya existe en disco, no rehace nada.

    Returns:
        dict con stats: {n_docs, index_size_mb, build_date, model_name, ...}
        o {error: "..."} si falla.
    """
    t_build = time.time()

    if not force_rebuild and INDEX_PATH.exists() and METADATA_PATH.exists():
        log.info("Índice ya existe — usa force_rebuild=True para regenerar")
        stats = get_index_stats()
        stats["skipped"] = True
        return stats

    try:
        corpus = _fetch_corpus()
    except Exception as e:
        return {"error": f"DB fetch failed: {e}"}

    if not corpus:
        return {"error": "Corpus vacío (no hay sentiment_raw NOT NULL)"}

    try:
        embs = _embed_corpus(corpus)
    except Exception as e:
        log.exception("Error generando embeddings")
        return {"error": f"Embedding failed: {e}"}

    try:
        index = _build_faiss_index(embs)
    except Exception as e:
        log.exception("Error construyendo índice FAISS")
        return {"error": f"FAISS build failed: {e}"}

    try:
        stats = _save(index, corpus)
    except Exception as e:
        log.exception("Error guardando índice")
        return {"error": f"Save failed: {e}"}

    _invalidate_cache()
    stats["build_seconds"] = round(time.time() - t_build, 2)
    stats["skipped"] = False
    return stats


def semantic_search(
    query: str,
    k: int = 5,
    before_date: str | None = None,
    min_similarity: float = 0.0,
) -> list[dict]:
    """
    Búsqueda semántica top-k sobre el índice FAISS.

    Args:
        query:          texto natural de búsqueda
        k:              número de resultados a devolver
        before_date:    ISO date (YYYY-MM-DD); si se pasa, filtra timestamp < before_date
        min_similarity: descarta hits con similarity < min_similarity

    Returns:
        Lista de dicts: {text, source, timestamp, similarity, url, sentiment, id}
        Lista vacía si el índice no existe o query inválida.
    """
    if not query or not query.strip():
        return []

    index, meta = _load_index_and_metadata()
    if index is None or meta is None:
        log.error("Índice no encontrado en disco. Ejecuta build_index() primero.")
        return []

    docs: list[dict] = meta.get("docs", [])
    if not docs:
        return []

    # Si filtramos por fecha, sobre-recuperamos para tener margen
    overfetch = max(k * 5, 50) if before_date else k

    try:
        model = _get_model()
        q_emb = model.encode(
            [query.strip()],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
    except Exception as e:
        log.error(f"Error generando embedding de query: {e}")
        return []

    try:
        sims, idxs = index.search(q_emb, min(overfetch, index.ntotal))
    except Exception as e:
        log.error(f"Error búsqueda FAISS: {e}")
        return []

    results: list[dict] = []
    for sim, idx in zip(sims[0], idxs[0]):
        if idx < 0 or idx >= len(docs):
            continue
        sim_f = float(sim)
        if sim_f < min_similarity:
            continue
        d = docs[idx]
        if before_date and d.get("timestamp"):
            # Comparación lexicográfica funciona con ISO8601
            if d["timestamp"] >= before_date:
                continue
        results.append(
            {
                "id":         d.get("id"),
                "text":       d.get("text", ""),
                "source":     d.get("source", ""),
                "timestamp":  d.get("timestamp", ""),
                "url":        d.get("url", ""),
                "sentiment":  d.get("sentiment"),
                "similarity": sim_f,
            }
        )
        if len(results) >= k:
            break

    return results


def get_index_stats() -> dict:
    """
    Devuelve stats del índice persistido:
    {n_docs, index_size_mb, build_date, model_name}
    """
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        return {
            "n_docs":        0,
            "index_size_mb": 0.0,
            "build_date":    None,
            "model_name":    MODEL_NAME,
            "exists":        False,
        }
    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        return {"error": f"Cannot read metadata: {e}"}

    size_mb = INDEX_PATH.stat().st_size / (1024 * 1024)
    return {
        "n_docs":        meta.get("n_docs", len(meta.get("docs", []))),
        "index_size_mb": round(size_mb, 3),
        "build_date":    meta.get("build_date"),
        "model_name":    meta.get("model_name", MODEL_NAME),
        "exists":        True,
    }


# ── Self-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print(" RAG PIPELINE — Self-test")
    print("=" * 70)

    # 1) Build (o reuse)
    t0 = time.time()
    stats = build_index(force_rebuild=False)
    dt = time.time() - t0
    print(f"\n[build_index] ({dt:.1f}s)")
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    if "error" in stats:
        print("\nBuild falló. Abortando self-test.")
        sys.exit(1)

    # 2) Test queries
    test_queries = [
        "Bitcoin ETF approval",
        "Federal Reserve interest rates impact crypto",
        "Bitcoin halving cycle",
        "regulacion SEC criptomonedas",
    ]

    print("\n" + "=" * 70)
    print(" Test queries (top-3 cada una)")
    print("=" * 70)

    for q in test_queries:
        results = semantic_search(q, k=3)
        print(f"\n[QUERY] {q}")
        if not results:
            print("  (sin resultados)")
            continue
        assert isinstance(results, list), "semantic_search debe devolver list"
        assert all("similarity" in r for r in results), "falta key 'similarity'"
        for r in results:
            src = (r.get("source") or "")[:15].ljust(15)
            sim = r["similarity"]
            preview = (r.get("text") or "").replace("\n", " ")[:90]
            print(f"  sim={sim:+.3f} | {src} | {preview}")

    # 3) Stats finales
    print("\n" + "=" * 70)
    print(" get_index_stats()")
    print("=" * 70)
    print(json.dumps(get_index_stats(), indent=2, ensure_ascii=False))
