# db/create_tables.py
from db_utils import get_engine
from sqlalchemy import text
import os
 
def create_tables():
    engine = get_engine()
 
    # Leer el archivo SQL
    schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')
    with open(schema_path, 'r') as f:
        sql = f.read()
 
    # Ejecutar cada sentencia SQL separada por ';'
    with engine.begin() as conn:
        for statement in sql.split(';'):
            stmt = statement.strip()
            if stmt:  # ignorar sentencias vacías
                conn.execute(text(stmt))
 
    print('Tablas creadas correctamente.')
 
    # Verificar que las tablas existen
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' ORDER BY table_name"
        ))
        tablas = [row[0] for row in result]
        print('Tablas en la BD:', tablas)
 
if __name__ == '__main__':
    create_tables()

