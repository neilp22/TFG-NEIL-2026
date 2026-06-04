"""
08_quick_test.py — Test de $0: verifica configuración sin llamar a OpenAI

Checks:
  1. Variables de entorno esenciales
  2. Conexión a PostgreSQL
  3. Schema de daily_features (columnas esperadas)
  4. Herramientas: query_market, get_sentiment, get_fear_greed, run_ml_prediction
  5. RAG: pgvector disponible / fallback a búsqueda textual
  6. Simulación del loop del agente (sin OpenAI)

Salida:
  ✅ OK   — componente funcionando
  ⚠️  WARN — degradado pero funcional (ej. pgvector no disponible)
  ❌ FAIL — bloquea funcionalidad crítica

Uso:
  python agente_ia/08_quick_test.py
  python agente_ia/08_quick_test.py --verbose
"""

import os, sys, json, traceback, argparse
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / 'config' / '.env')

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

VERBOSE = False
results = []


def check(name: str, fn, warn_ok: bool = False):
    """Ejecuta un check y registra el resultado."""
    try:
        msg = fn()
        status = '✅'
        results.append((status, name, msg or ''))
        print(f"  {status} {name}: {msg or 'OK'}")
        return True
    except Warning as e:
        status = '⚠️ '
        results.append((status, name, str(e)))
        print(f"  {status} {name}: {e}")
        return warn_ok
    except Exception as e:
        status = '❌'
        results.append((status, name, str(e)))
        print(f"  {status} {name}: {e}")
        if VERBOSE:
            traceback.print_exc()
        return False


# ── 1. Variables de entorno ───────────────────────────────────────────────────

print("\n[1] Variables de entorno")

def _check_db_env():
    missing = [k for k in ['DB_HOST', 'DB_NAME', 'DB_USER', 'DB_PASSWORD'] if not os.getenv(k)]
    if missing:
        raise Exception(f"Faltan: {missing}")
    return f"DB={os.getenv('DB_NAME')}@{os.getenv('DB_HOST')}"

def _check_openai_env():
    key = os.getenv('OPENAI_API_KEY', '')
    if not key:
        raise Warning("OPENAI_API_KEY no configurada — agente no funcionará sin ella")
    return f"sk-...{key[-4:]}" if len(key) > 8 else "configurada"

def _check_cryptopanic_env():
    key = os.getenv('CRYPTOPANIC_KEY', os.getenv('CRYPTOPANIC_API_KEY', ''))
    if not key:
        raise Warning("CRYPTOPANIC_KEY no configurada — CryptoPanic usará API pública (limitada)")
    return "configurada"

check("DB env vars",         _check_db_env)
check("OPENAI_API_KEY",      _check_openai_env,      warn_ok=True)
check("CRYPTOPANIC_KEY",     _check_cryptopanic_env, warn_ok=True)


# ── 2. Conexión PostgreSQL ────────────────────────────────────────────────────

print("\n[2] Conexión PostgreSQL")

engine = None

def _check_db_connect():
    global engine
    from db.db_utils import get_engine
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        ts = conn.execute(text("SELECT NOW()")).scalar()
    return f"conectado — {ts}"

check("PostgreSQL connect", _check_db_connect)


# ── 3. Schema daily_features ──────────────────────────────────────────────────

print("\n[3] Schema daily_features")

REQUIRED_COLUMNS = [
    'date', 'asset', 'close', 'returns', 'label',
    'rsi_14', 'macd', 'macd_signal', 'bb_upper', 'bb_lower',
    'sma_7', 'sma_30', 'sentiment_avg', 'fear_greed',
]

def _check_schema():
    if engine is None:
        raise Exception("No hay conexión DB")
    from sqlalchemy import text
    with engine.connect() as conn:
        cols = conn.execute(text("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'daily_features' AND table_schema = 'public'
        """)).fetchall()
    col_names = {r[0] for r in cols}
    missing = [c for c in REQUIRED_COLUMNS if c not in col_names]
    if missing:
        raise Exception(f"Columnas faltantes: {missing}")
    extra = [c for c in ['sentiment_morning', 'has_sentiment'] if c not in col_names]
    if extra:
        raise Warning(f"Columnas opcionales no encontradas (se crearán): {extra}")
    return f"{len(col_names)} columnas presentes"

def _check_data_count():
    if engine is None:
        raise Exception("No hay conexión DB")
    from sqlalchemy import text
    with engine.connect() as conn:
        n = conn.execute(text(
            "SELECT COUNT(*) FROM daily_features WHERE asset = 'BTC'"
        )).scalar()
        max_date = conn.execute(text(
            "SELECT MAX(date) FROM daily_features WHERE asset = 'BTC'"
        )).scalar()
    if n == 0:
        raise Exception("No hay datos en daily_features")
    return f"{n} filas, última fecha: {max_date}"

check("Schema columns",  _check_schema,     warn_ok=True)
check("Data available",  _check_data_count)


# ── 4. Herramientas ───────────────────────────────────────────────────────────

print("\n[4] Herramientas del agente")

def _check_query_market():
    from tools import query_market
    result = query_market(days=3)
    if result.get('error'):
        raise Exception(result['error'])
    price = result.get('price_usd')
    return f"precio={price}, RSI={result.get('indicators', {}).get('rsi_14')}"

def _check_get_sentiment():
    from tools import get_sentiment
    result = get_sentiment(days=7)
    if result.get('error'):
        raise Exception(result['error'])
    return f"label={result.get('sentiment_label')}, días={result.get('days_with_data')}"

def _check_get_fear_greed():
    from tools import get_fear_greed
    result = get_fear_greed(days=3)
    if result.get('error'):
        raise Exception(result['error'])
    latest = result.get('latest', {})
    return f"valor={latest.get('value')}, clasif={latest.get('classification', 'N/A')}"

def _check_run_ml_prediction():
    from tools import run_ml_prediction
    result = run_ml_prediction()
    if result.get('error'):
        raise Exception(result['error'])
    model_loaded = result.get('model_loaded', False)
    prob = result.get('prob_up')
    if not model_loaded:
        raise Warning(
            f"Modelo pkl no encontrado — usando heurística. "
            f"prob_up={prob} (no es predicción real del modelo XGBoost)"
        )
    return f"prob_up={prob}, modelo={result.get('model_context', {}).get('type', 'N/A')}"

check("query_market",      _check_query_market)
check("get_sentiment",     _check_get_sentiment)
check("get_fear_greed",    _check_get_fear_greed)
check("run_ml_prediction", _check_run_ml_prediction, warn_ok=True)


# ── 5. RAG / pgvector ─────────────────────────────────────────────────────────

print("\n[5] RAG (pgvector)")

def _check_pgvector_extension():
    if engine is None:
        raise Exception("No hay conexión DB")
    from sqlalchemy import text
    try:
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        return "extensión vector disponible"
    except Exception as e:
        raise Warning(
            f"pgvector no disponible ({e}). "
            "RAG usará búsqueda textual como fallback. "
            "Para instalar: brew install pgvector (macOS) o apt install postgresql-16-pgvector"
        )

def _check_news_embeddings_table():
    if engine is None:
        raise Exception("No hay conexión DB")
    from sqlalchemy import text
    try:
        with engine.connect() as conn:
            n = conn.execute(text(
                "SELECT COUNT(*) FROM news_embeddings"
            )).scalar()
        if n == 0:
            raise Warning(
                "Tabla news_embeddings vacía — RAG no devolverá resultados semánticos. "
                "Ejecuta: python agente_ia/03_setup_rag.py --local"
            )
        return f"{n} embeddings almacenados"
    except Exception as e:
        if "does not exist" in str(e):
            raise Warning(
                "Tabla news_embeddings no existe — ejecuta: "
                "python agente_ia/03_setup_rag.py --setup-only"
            )
        raise

def _check_rag_search():
    from tools import rag_search
    result = rag_search("bitcoin price regulation", k=3)
    if result.get('error'):
        raise Warning(f"RAG search degradado: {result['error']}")
    method = result.get('method', 'unknown')
    n = result.get('k', len(result.get('results', [])))
    return f"método={method}, resultados={n}"

check("pgvector extension",    _check_pgvector_extension,    warn_ok=True)
check("news_embeddings table", _check_news_embeddings_table, warn_ok=True)
check("rag_search (fallback)", _check_rag_search,            warn_ok=True)


# ── 6. Simulación loop agente (sin OpenAI) ────────────────────────────────────

print("\n[6] Simulación del loop del agente")

def _check_tool_definitions():
    from tools import TOOL_DEFINITIONS, TOOLS_MAP
    if len(TOOL_DEFINITIONS) != 5:
        raise Exception(f"Se esperan 5 herramientas, hay {len(TOOL_DEFINITIONS)}")
    names_def = {t['function']['name'] for t in TOOL_DEFINITIONS}
    names_map = set(TOOLS_MAP.keys())
    if names_def != names_map:
        raise Exception(f"Mismatch: defs={names_def}, map={names_map}")
    return f"5 herramientas: {', '.join(sorted(names_def))}"

def _check_dispatch():
    from tools import dispatch_tool
    result = dispatch_tool('query_market', {'days': 1})
    if result.get('error'):
        raise Exception(f"dispatch_tool('query_market') error: {result['error']}")
    return "dispatch_tool funcional"

def _check_agent_import():
    try:
        import importlib
        agent_mod = importlib.import_module('agent')
        if not hasattr(agent_mod, 'chat'):
            raise Exception("Función 'chat' no encontrada en agent.py")
        if not hasattr(agent_mod, 'SYSTEM_PROMPT'):
            raise Exception("SYSTEM_PROMPT no encontrado en agent.py")
        return "módulo agent importado, chat() disponible"
    except ImportError as e:
        raise Warning(f"No se pudo importar agent.py: {e}")

check("TOOL_DEFINITIONS",  _check_tool_definitions)
check("dispatch_tool",     _check_dispatch)
check("agent.py import",   _check_agent_import, warn_ok=True)


# ── Resumen ───────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("RESUMEN")
print("="*60)

ok_count   = sum(1 for r in results if r[0] == '✅')
warn_count = sum(1 for r in results if r[0] == '⚠️ ')
fail_count = sum(1 for r in results if r[0] == '❌')
total      = len(results)

print(f"  ✅ OK:    {ok_count}/{total}")
print(f"  ⚠️  WARN: {warn_count}/{total}")
print(f"  ❌ FAIL:  {fail_count}/{total}")

if fail_count == 0 and warn_count == 0:
    print("\n  Todo OK — el agente está listo para ejecutarse.")
elif fail_count == 0:
    print("\n  Funcional con advertencias — algunas características degradadas.")
    print("  El agente puede ejecutarse pero con funcionalidad reducida.")
else:
    print(f"\n  {fail_count} checks críticos fallaron — revisar antes de ejecutar.")
    for status, name, msg in results:
        if status == '❌':
            print(f"    • {name}: {msg}")

print()
sys.exit(0 if fail_count == 0 else 1)
