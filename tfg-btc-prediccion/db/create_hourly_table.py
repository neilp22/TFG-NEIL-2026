
# db/create_hourly_table.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from db.db_utils import get_engine
from sqlalchemy import text

def create_hourly_features():
    engine = get_engine()
    sql = """
    CREATE TABLE IF NOT EXISTS hourly_features (
        timestamp     TIMESTAMPTZ  NOT NULL,
        asset         VARCHAR(10)  NOT NULL,
        close         NUMERIC(18,8),
        returns       NUMERIC(10,6),
        label         SMALLINT,
        rsi_14        NUMERIC(6,2),
        macd          NUMERIC(10,6),
        macd_signal   NUMERIC(10,6),
        bb_upper      NUMERIC(18,8),
        bb_lower      NUMERIC(18,8),
        sma_7         NUMERIC(18,8),
        sma_30        NUMERIC(18,8),
        atr_14        NUMERIC(18,8),
        obv           NUMERIC(24,4),
        stoch_k       NUMERIC(6,2),
        williams_r    NUMERIC(6,2),
        ob_bullish    BOOLEAN DEFAULT FALSE,
        ob_bearish    BOOLEAN DEFAULT FALSE,
        fvg_up        BOOLEAN DEFAULT FALSE,
        fvg_down      BOOLEAN DEFAULT FALSE,
        poc           NUMERIC(18,8),
        va_high       NUMERIC(18,8),
        va_low        NUMERIC(18,8),
        funding_rate  NUMERIC(10,8),
        open_interest NUMERIC(24,4),
        sentiment_avg   NUMERIC(5,4),
        sentiment_count INTEGER,
        google_trends   SMALLINT,
        updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        PRIMARY KEY (timestamp, asset)
    );
    CREATE INDEX IF NOT EXISTS idx_hf_asset_ts ON hourly_features (asset, timestamp DESC)
    """
    with engine.begin() as conn:
        for stmt in sql.split(';'):
            s = stmt.strip()
            if s:
                conn.execute(text(s))
    print('Tabla hourly_features creada correctamente.')

if __name__ == '__main__':
    create_hourly_features()