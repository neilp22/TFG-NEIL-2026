-- db/schema.sql
-- Esquema completo de la base de datos btc_tfg
 
-- ================================================
-- TABLA 1: price_data
-- Datos OHLCV de todos los activos
-- ================================================
CREATE TABLE IF NOT EXISTS price_data (
    timestamp   TIMESTAMPTZ   NOT NULL,
    asset       VARCHAR(10)   NOT NULL,
    timeframe   VARCHAR(5)    NOT NULL,
    open        NUMERIC(18,8) NOT NULL,
    high        NUMERIC(18,8) NOT NULL,
    low         NUMERIC(18,8) NOT NULL,
    close       NUMERIC(18,8) NOT NULL,
    volume      NUMERIC(24,4) NOT NULL,
    PRIMARY KEY (timestamp, asset, timeframe)
);
 
CREATE INDEX IF NOT EXISTS idx_price_asset_ts
    ON price_data (asset, timestamp DESC);
 
-- ================================================
-- TABLA 2: raw_texts
-- Textos crudos para análisis de sentimiento
-- ================================================
CREATE TABLE IF NOT EXISTS raw_texts (
    id          BIGSERIAL     PRIMARY KEY,
    timestamp   TIMESTAMPTZ   NOT NULL,
    asset       VARCHAR(10)   NOT NULL,
    source      VARCHAR(30)   NOT NULL,
    text        TEXT          NOT NULL,
    url         TEXT,
    processed   BOOLEAN       NOT NULL DEFAULT FALSE,
    created_at  TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);
 
CREATE INDEX IF NOT EXISTS idx_texts_asset_ts
    ON raw_texts (asset, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_texts_processed
    ON raw_texts (processed) WHERE processed = FALSE;
 
-- ================================================
-- TABLA 3: sentiment_scores
-- Scores de FinBERT por texto
-- ================================================
CREATE TABLE IF NOT EXISTS sentiment_scores (
    id              BIGSERIAL     PRIMARY KEY,
    text_id         BIGINT        NOT NULL REFERENCES raw_texts(id),
    score_positive  NUMERIC(5,4)  NOT NULL,
    score_negative  NUMERIC(5,4)  NOT NULL,
    score_neutral   NUMERIC(5,4)  NOT NULL,
    compound_score  NUMERIC(5,4)  NOT NULL,
    model_used      VARCHAR(50)   NOT NULL DEFAULT 'finbert-tone',
    processed_at    TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);
 
CREATE INDEX IF NOT EXISTS idx_scores_text_id
    ON sentiment_scores (text_id);
 
-- ================================================
-- TABLA 4: daily_features
-- Tabla maestra para el pipeline de ML
-- ================================================
CREATE TABLE IF NOT EXISTS daily_features (
    date            DATE          NOT NULL,
    asset           VARCHAR(10)   NOT NULL,
    close           NUMERIC(18,8),
    returns         NUMERIC(10,6),
    label           SMALLINT,
    rsi_14          NUMERIC(6,2),
    macd            NUMERIC(10,6),
    macd_signal     NUMERIC(10,6),
    bb_upper        NUMERIC(18,8),
    bb_lower        NUMERIC(18,8),
    sma_7           NUMERIC(18,8),
    sma_30          NUMERIC(18,8),
    sentiment_avg   NUMERIC(5,4),
    sentiment_std   NUMERIC(5,4),
    sentiment_count INTEGER,
    fear_greed      SMALLINT,
    updated_at      TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    PRIMARY KEY (date, asset)
);
 
CREATE INDEX IF NOT EXISTS idx_features_asset_date
    ON daily_features (asset, date DESC);

