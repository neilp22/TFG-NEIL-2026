# db/db_utils.py
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import pandas as pd
 
# Cargar variables del .env
load_dotenv('config/.env')
 
def get_engine():
    """Devuelve un engine de SQLAlchemy conectado a btc_tfg."""
    url = (
        f"postgresql+psycopg2://"
        f"{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}"
        f"/{os.getenv('DB_NAME')}"
    )
    return create_engine(url, pool_pre_ping=True)
 
def test_connection():
    """Comprueba que la conexión funciona correctamente."""
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text('SELECT version()'))
        print('Conexión OK:', result.fetchone()[0])
 
if __name__ == '__main__':
    test_connection()

