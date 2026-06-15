# db/db_utils.py
import logging
import os
import threading
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import pandas as pd

load_dotenv(Path(__file__).parent.parent / 'config' / '.env')

log = logging.getLogger(__name__)

# Engine singleton por proceso. Crear un engine nuevo (con su pool) en cada
# llamada filtra conexiones hasta agotar max_connections de Postgres, y
# entonces get_engine() caía en silencio al fallback SQLite (btc_agent.db),
# rompiendo todas las tools que esperan btc_ohlcv.
_ENGINE = None
_ENGINE_LOCK = threading.Lock()


def get_engine():
    """
    Devuelve un engine de SQLAlchemy (singleton por proceso).
    Prioridad: DATABASE_URL > vars individuales PG > SQLite fallback.
    """
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = _create_engine()
        return _ENGINE


def _create_engine():
    # Opción 1: DATABASE_URL explícita
    database_url = os.getenv('DATABASE_URL', '').strip()
    if database_url:
        return create_engine(database_url, pool_pre_ping=True)

    # Opción 2: Variables PostgreSQL individuales
    db_host = os.getenv('DB_HOST', '').strip()
    db_name = os.getenv('DB_NAME', '').strip()
    db_user = os.getenv('DB_USER', '').strip()
    db_pass = os.getenv('DB_PASSWORD', '').strip()
    db_port = os.getenv('DB_PORT', '5432').strip()

    if db_host and db_name and db_user:
        url = f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
        try:
            engine = create_engine(
                url,
                pool_pre_ping=True,
                pool_size=5,
                max_overflow=5,
                pool_recycle=1800,
                connect_args={"connect_timeout": 5},
            )
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return engine
        except Exception as e:
            log.error("PostgreSQL no disponible (%s) — usando fallback SQLite. "
                      "Las tools que dependen de btc_ohlcv fallarán.", e)

    # Opción 3: SQLite local como fallback
    db_path = Path(__file__).parent.parent / 'btc_agent.db'
    log.warning("get_engine: usando SQLite fallback en %s", db_path)
    return create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})


def test_connection():
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text('SELECT version()' if 'postgresql' in engine.url.drivername else 'SELECT sqlite_version()'))
        print('Conexión OK:', result.fetchone()[0])


if __name__ == '__main__':
    test_connection()
