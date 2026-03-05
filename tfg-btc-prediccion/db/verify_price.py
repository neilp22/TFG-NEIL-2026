from db_utils import get_engine
from sqlalchemy import text
 
engine = get_engine()
with engine.connect() as conn:
    # Contar filas por timeframe
    r = conn.execute(text(
        "SELECT asset, timeframe, COUNT(*) as n, "
        "MIN(timestamp)::date as desde, MAX(timestamp)::date as hasta "
        "FROM price_data GROUP BY asset, timeframe ORDER BY timeframe"
    ))
    for row in r:
        print(row)
