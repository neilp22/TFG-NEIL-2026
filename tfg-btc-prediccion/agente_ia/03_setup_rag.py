"""
03_setup_rag.py — Configuración RAG con pgvector

Modos:
  --setup-only   Solo instala extensión y crea tabla/índice (sin datos)
  --local        Genera embeddings en local (CPU, lento) desde raw_texts
  --import PATH  Importa embeddings precalculados desde CSV (recomendado)

Uso:
  python agente_ia/03_setup_rag.py --setup-only
  python agente_ia/03_setup_rag.py --import embeddings_output.csv
  python agente_ia/03_setup_rag.py --local --limit 1000

CSV esperado para --import:
  Columnas: raw_text_id, embedding (lista JSON de 384 floats)
"""

import os, sys, logging, json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / 'config' / '.env')

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sqlalchemy import text
from db.db_utils import get_engine

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

EMBEDDING_DIM = 384
BATCH_SIZE    = 256


# ── Setup pgvector ────────────────────────────────────────────────────────────

def setup_pgvector(engine) -> bool:
    """
    Instala extensión vector, crea tabla news_embeddings e índice HNSW.
    Devuelve True si todo OK, False si pgvector no está disponible.
    """
    with engine.begin() as conn:
        # 1. Intentar instalar extensión
        try:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            log.info("Extensión pgvector instalada/verificada")
        except Exception as e:
            log.error(
                f"pgvector no disponible: {e}\n"
                "Instala con: apt install postgresql-16-pgvector   (Linux)\n"
                "             brew install pgvector                  (macOS)\n"
                "O usa una imagen Docker: ankane/pgvector"
            )
            return False

        # 2. Crear tabla news_embeddings
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS news_embeddings (
                id           SERIAL PRIMARY KEY,
                raw_text_id  INTEGER REFERENCES raw_texts(id) ON DELETE CASCADE,
                embedding    vector({EMBEDDING_DIM}),
                created_at   TIMESTAMPTZ DEFAULT NOW()
            )
        """))
        log.info("Tabla news_embeddings verificada")

        # 3. Índice HNSW para búsqueda semántica rápida
        try:
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_news_emb_hnsw
                ON news_embeddings
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
            """))
            log.info("Índice HNSW creado/verificado")
        except Exception as e:
            log.warning(f"No se pudo crear índice HNSW (pgvector < 0.5?): {e}")
            # Fallback: índice IVFFlat
            try:
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_news_emb_ivfflat
                    ON news_embeddings
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                """))
                log.info("Índice IVFFlat creado como fallback")
            except Exception as e2:
                log.warning(f"IVFFlat también falló: {e2}. Búsqueda usará fuerza bruta.")

    return True


# ── Embeddings locales ────────────────────────────────────────────────────────

def _load_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        log.info("SentenceTransformer cargado (all-MiniLM-L6-v2, 384-dim)")
        return model
    except ImportError:
        log.error("sentence-transformers no instalado. Ejecuta: pip install sentence-transformers")
        return None


def generate_local_embeddings(engine, limit: int = 0) -> dict:
    """Genera embeddings localmente para raw_texts sin embedding."""
    model = _load_sentence_transformer()
    if model is None:
        return {'error': 'sentence-transformers not installed'}

    # Textos sin embedding
    limit_clause = f"LIMIT {limit}" if limit > 0 else ""
    query = f"""
        SELECT rt.id, rt.text, rt.timestamp
        FROM raw_texts rt
        LEFT JOIN news_embeddings ne ON ne.raw_text_id = rt.id
        WHERE ne.id IS NULL
        ORDER BY rt.timestamp DESC
        {limit_clause}
    """
    try:
        df = pd.read_sql(query, engine)
    except Exception as e:
        log.error(f"Error leyendo raw_texts: {e}")
        return {'error': str(e)}

    if df.empty:
        log.info("No hay textos pendientes de embedding")
        return {'inserted': 0}

    log.info(f"Generando embeddings para {len(df)} textos...")
    texts = df['text'].tolist()
    inserted = 0

    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i + BATCH_SIZE]
        batch_ids   = df['id'].iloc[i:i + BATCH_SIZE].tolist()

        try:
            embeddings = model.encode(batch_texts, batch_size=32, show_progress_bar=False)
        except Exception as e:
            log.error(f"Error generando embeddings batch {i}: {e}")
            continue

        with engine.begin() as conn:
            for raw_id, emb in zip(batch_ids, embeddings):
                try:
                    conn.execute(text("""
                        INSERT INTO news_embeddings (raw_text_id, embedding)
                        VALUES (:rid, :emb)
                        ON CONFLICT DO NOTHING
                    """), {'rid': int(raw_id), 'emb': emb.tolist()})
                    inserted += 1
                except Exception as e:
                    log.debug(f"Insert embedding error (id={raw_id}): {e}")

        log.info(f"  Batch {i//BATCH_SIZE + 1}: {min(i+BATCH_SIZE, len(texts))}/{len(texts)}")

    log.info(f"Embeddings generados: {inserted} insertados")
    return {'processed': len(df), 'inserted': inserted}


# ── Import desde CSV ──────────────────────────────────────────────────────────

def import_embeddings_csv(csv_path: str, engine) -> dict:
    """
    Importa embeddings precalculados desde CSV.
    Columnas esperadas: raw_text_id, embedding (JSON string de lista)
    """
    path = Path(csv_path)
    if not path.exists():
        log.error(f"Archivo no encontrado: {csv_path}")
        return {'error': f'File not found: {csv_path}'}

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        log.error(f"Error leyendo CSV: {e}")
        return {'error': str(e)}

    required = {'raw_text_id', 'embedding'}
    if not required.issubset(df.columns):
        log.error(f"CSV debe tener columnas: {required}. Tiene: {set(df.columns)}")
        return {'error': f'Missing columns: {required - set(df.columns)}'}

    log.info(f"Importando {len(df)} embeddings desde {csv_path}")
    inserted = 0
    errors   = 0

    with engine.begin() as conn:
        for _, row in df.iterrows():
            try:
                emb = json.loads(row['embedding']) if isinstance(row['embedding'], str) else row['embedding']
                if len(emb) != EMBEDDING_DIM:
                    log.warning(f"Embedding dim={len(emb)}, esperado {EMBEDDING_DIM} — omitiendo raw_text_id={row['raw_text_id']}")
                    errors += 1
                    continue
                conn.execute(text("""
                    INSERT INTO news_embeddings (raw_text_id, embedding)
                    VALUES (:rid, :emb)
                    ON CONFLICT DO NOTHING
                """), {'rid': int(row['raw_text_id']), 'emb': emb})
                inserted += 1
            except Exception as e:
                log.debug(f"Import error (row {_}): {e}")
                errors += 1

    log.info(f"Import completado: {inserted} insertados, {errors} errores")
    return {'total': len(df), 'inserted': inserted, 'errors': errors}


# ── Búsqueda semántica (usada por 05_tools.py) ───────────────────────────────

def semantic_search(query: str, engine, k: int = 5, before_date: str = None) -> list[dict]:
    """
    Busca las k noticias más similares usando pgvector.

    Args:
        query:       Texto de búsqueda
        engine:      SQLAlchemy engine
        k:           Número de resultados
        before_date: Filtro temporal (ISO date) para evitar look-ahead bias

    Returns:
        Lista de dicts con keys: text, source, timestamp, similarity
    """
    model = _load_sentence_transformer()
    if model is None:
        return []

    try:
        query_emb = model.encode(query).tolist()
    except Exception as e:
        log.error(f"Error generando embedding de query: {e}")
        return []

    date_filter = f"AND rt.timestamp < '{before_date}'" if before_date else ""

    sql = text(f"""
        SELECT
            rt.text,
            rt.source,
            rt.timestamp,
            1 - (ne.embedding <=> :emb::vector) AS similarity
        FROM news_embeddings ne
        JOIN raw_texts rt ON rt.id = ne.raw_text_id
        WHERE rt.asset = 'BTC'
        {date_filter}
        ORDER BY ne.embedding <=> :emb::vector
        LIMIT :k
    """)

    try:
        with engine.connect() as conn:
            rows = conn.execute(sql, {'emb': str(query_emb), 'k': k}).fetchall()
        return [
            {
                'text':       r[0],
                'source':     r[1],
                'timestamp':  str(r[2]),
                'similarity': float(r[3]) if r[3] is not None else 0.0,
            }
            for r in rows
        ]
    except Exception as e:
        log.error(f"Semantic search error: {e}")
        return []


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--setup-only', action='store_true',
                    help='Solo instala extensión y crea tablas/índices')
    ap.add_argument('--local', action='store_true',
                    help='Genera embeddings en local (CPU)')
    ap.add_argument('--limit', type=int, default=0,
                    help='Máximo de textos a procesar con --local (0=todos)')
    ap.add_argument('--import', dest='import_csv', type=str, default='',
                    help='CSV con embeddings precalculados')
    args = ap.parse_args()

    engine = get_engine()
    ok = setup_pgvector(engine)

    if not ok:
        log.error("Abortando: pgvector no disponible")
        sys.exit(1)

    if args.setup_only:
        log.info("Setup completado (--setup-only)")
        sys.exit(0)

    if args.import_csv:
        result = import_embeddings_csv(args.import_csv, engine)
        print(json.dumps(result, indent=2))
    elif args.local:
        result = generate_local_embeddings(engine, limit=args.limit)
        print(json.dumps(result, indent=2))
    else:
        log.info("Usa --setup-only, --local, o --import PATH")
        ap.print_help()
