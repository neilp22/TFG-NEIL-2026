# db/db_utils.py
import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import pandas as pd

load_dotenv(Path(__file__).parent.parent / 'config' / '.env')


def get_engine():
    """
    Devuelve un engine de SQLAlchemy.
    Prioridad: DATABASE_URL > vars individuales PG > SQLite fallback.
    """
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
                connect_args={"connect_timeout": 5},
            )
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return engine
        except Exception:
            pass

    # Opción 3: SQLite local como fallback
    db_path = Path(__file__).parent.parent / 'btc_agent.db'
    return create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})


def test_connection():
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text('SELECT version()' if 'postgresql' in engine.url.drivername else 'SELECT sqlite_version()'))
        print('Conexión OK:', result.fetchone()[0])


if __name__ == '__main__':
    test_connection()
