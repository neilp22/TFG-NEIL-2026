-- ============================================================
--  BTC AGENT — Schema completo v2
--  Módulos: news, calendar, on-chain, institutional, microstructure
-- ============================================================

-- ── Extensiones ─────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================
--  MÓDULO 1 — raw_texts  (ya existente, ampliado)
-- ============================================================
CREATE TABLE IF NOT EXISTS raw_texts (
    id          BIGSERIAL PRIMARY KEY,
    timestamp   TIMESTAMPTZ NOT NULL,
    asset       VARCHAR(10)  NOT NULL DEFAULT 'BTC',
    source      VARCHAR(80)  NOT NULL,
    source_tier SMALLINT     NOT NULL DEFAULT 3,
    -- 1=tier1 (Reuters/Bloomberg), 2=tier2 (CoinDesk/TheBlock), 3=tier3 (blogs)
    text        TEXT         NOT NULL,
    url         TEXT,
    processed   BOOLEAN      NOT NULL DEFAULT FALSE,
    -- Scoring pre-LLM (se rellena en pipeline de NLP)
    sentiment_raw   FLOAT,          -- score crudo del modelo NLP (-1 a +1)
    sentiment_decay FLOAT,          -- score × decay temporal
    impact_weight   FLOAT,          -- multiplicador según source_tier + keywords
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_raw_texts_ts        ON raw_texts (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_raw_texts_processed ON raw_texts (processed) WHERE processed = FALSE;
CREATE INDEX IF NOT EXISTS idx_raw_texts_url       ON raw_texts (url) WHERE url IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_raw_texts_asset_ts  ON raw_texts (asset, timestamp DESC);


-- ============================================================
--  MÓDULO 2 — economic_calendar
-- ============================================================
CREATE TABLE IF NOT EXISTS economic_calendar (
    id               BIGSERIAL PRIMARY KEY,
    event_datetime   TIMESTAMPTZ  NOT NULL,       -- hora exacta UTC del evento
    event_name       VARCHAR(200) NOT NULL,
    country          CHAR(2)      NOT NULL,        -- ISO 3166-1 alpha-2: US, EU, GB...
    currency         CHAR(3)      NOT NULL,        -- USD, EUR, GBP...
    impact           VARCHAR(10)  NOT NULL,        -- 'high' | 'medium' | 'low'
    -- Valores del consenso
    forecast         FLOAT,
    previous         FLOAT,
    actual           FLOAT,
    unit             VARCHAR(20),                  -- '%', 'K', 'B', 'index'...
    -- Señal derivada (calculada automáticamente)
    surprise         FLOAT,                        -- actual - forecast
    surprise_pct     FLOAT,                        -- (actual - forecast) / |forecast| * 100
    surprise_dir     VARCHAR(10),                  -- 'beat' | 'miss' | 'inline' | NULL (pendiente)
    btc_bias         SMALLINT,                     -- +1 bullish, -1 bearish, 0 neutral (regla determinista)
    bias_reason      TEXT,                         -- descripción legible de la regla aplicada
    -- Control
    source           VARCHAR(50)  NOT NULL DEFAULT 'investing_com',
    fetched_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (event_datetime, event_name, country)
);

CREATE INDEX IF NOT EXISTS idx_cal_datetime ON economic_calendar (event_datetime DESC);
CREATE INDEX IF NOT EXISTS idx_cal_impact   ON economic_calendar (impact, event_datetime DESC);
CREATE INDEX IF NOT EXISTS idx_cal_country  ON economic_calendar (country, event_datetime DESC);
CREATE INDEX IF NOT EXISTS idx_cal_pending  ON economic_calendar (actual)
    WHERE actual IS NULL AND surprise_dir IS NULL;


-- ============================================================
--  MÓDULO 3 — on_chain_metrics
-- ============================================================
CREATE TABLE IF NOT EXISTS on_chain_metrics (
    id              BIGSERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL,
    asset           VARCHAR(10) NOT NULL DEFAULT 'BTC',
    metric_name     VARCHAR(80) NOT NULL,
    -- Ejemplos: 'exchange_netflow', 'mvrv_z', 'sopr', 'hash_rate',
    --           'funding_rate_binance', 'active_addresses', 'nvt_ratio',
    --           'miner_reserve', 'stablecoin_supply_ratio'
    value           FLOAT       NOT NULL,
    value_7d_avg    FLOAT,                      -- media móvil 7 días (calculada al insertar)
    value_30d_avg   FLOAT,
    -- Señal derivada
    signal          VARCHAR(10),                -- 'bullish' | 'bearish' | 'neutral'
    signal_strength FLOAT,                      -- 0.0 a 1.0
    source          VARCHAR(50) NOT NULL,       -- 'glassnode', 'cryptoquant', 'coinglass', etc.
    fetched_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (timestamp, asset, metric_name, source)
);

CREATE INDEX IF NOT EXISTS idx_onchain_ts     ON on_chain_metrics (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_onchain_metric ON on_chain_metrics (metric_name, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_onchain_asset  ON on_chain_metrics (asset, metric_name, timestamp DESC);


-- ============================================================
--  MÓDULO 4 — institutional_flows
-- ============================================================
CREATE TABLE IF NOT EXISTS institutional_flows (
    id              BIGSERIAL PRIMARY KEY,
    flow_date       DATE        NOT NULL,
    asset           VARCHAR(10) NOT NULL DEFAULT 'BTC',
    vehicle         VARCHAR(50) NOT NULL,
    -- Ejemplos: 'IBIT', 'FBTC', 'GBTC', 'ARKB', 'BITB', 'MSTR_purchase'
    vehicle_type    VARCHAR(20) NOT NULL,        -- 'spot_etf' | 'futures_etf' | 'corporate'
    flow_usd        FLOAT,                       -- USD millones (negativo = outflow)
    flow_btc        FLOAT,                       -- BTC equivalentes
    aum_usd         FLOAT,                       -- AUM total del vehículo ese día
    -- Señal
    consecutive_days_inflow  SMALLINT,           -- racha de días positivos (calculado)
    consecutive_days_outflow SMALLINT,
    source          VARCHAR(50) NOT NULL DEFAULT 'farside',
    fetched_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (flow_date, vehicle)
);

CREATE INDEX IF NOT EXISTS idx_inst_date    ON institutional_flows (flow_date DESC);
CREATE INDEX IF NOT EXISTS idx_inst_vehicle ON institutional_flows (vehicle, flow_date DESC);

-- Vista agregada diaria (útil para el bias vector)
CREATE OR REPLACE VIEW v_etf_daily_total AS
SELECT
    flow_date,
    asset,
    SUM(flow_usd)                                      AS total_flow_usd,
    SUM(flow_btc)                                      AS total_flow_btc,
    COUNT(*) FILTER (WHERE flow_usd > 0)               AS vehicles_inflow,
    COUNT(*) FILTER (WHERE flow_usd < 0)               AS vehicles_outflow,
    CASE
        WHEN SUM(flow_usd) >  50 THEN 'bullish'
        WHEN SUM(flow_usd) < -50 THEN 'bearish'
        ELSE 'neutral'
    END AS daily_bias
FROM institutional_flows
WHERE vehicle_type = 'spot_etf'
GROUP BY flow_date, asset;


-- ============================================================
--  MÓDULO 5 — market_microstructure
-- ============================================================
CREATE TABLE IF NOT EXISTS market_microstructure (
    id              BIGSERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL,
    asset           VARCHAR(10) NOT NULL DEFAULT 'BTC',
    exchange        VARCHAR(30),                 -- 'binance', 'bybit', 'okx', 'deribit'
    -- Open Interest
    oi_usd          FLOAT,                       -- OI total en USD
    oi_change_1h    FLOAT,                       -- variación 1h en %
    oi_change_24h   FLOAT,
    -- Funding
    funding_rate    FLOAT,                       -- tasa actual (positivo = longs pagan)
    funding_8h_avg  FLOAT,
    -- Liquidaciones
    liq_long_1h_usd  FLOAT,
    liq_short_1h_usd FLOAT,
    -- Opciones (Deribit)
    put_call_ratio   FLOAT,
    iv_25d_skew      FLOAT,                      -- 25-delta skew (negativo = put premium)
    max_pain_price   FLOAT,
    -- CME
    cme_gap_open     FLOAT,                      -- precio apertura CME lunes
    cme_gap_close    FLOAT,                      -- precio cierre CME viernes
    cme_gap_pct      FLOAT,                      -- gap en %
    cme_gap_filled   BOOLEAN,
    -- Señal
    signal           VARCHAR(10),
    source           VARCHAR(30) NOT NULL DEFAULT 'coinglass',
    fetched_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_micro_ts       ON market_microstructure (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_micro_exchange ON market_microstructure (exchange, timestamp DESC);


-- ============================================================
--  MÓDULO 6 — bias_snapshots  (output del pipeline)
-- ============================================================
-- Esta tabla es el resultado final. El LLM la lee, no la genera.
CREATE TABLE IF NOT EXISTS bias_snapshots (
    id              BIGSERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    asset           VARCHAR(10) NOT NULL DEFAULT 'BTC',
    timeframe       VARCHAR(10) NOT NULL,         -- '1h' | '4h' | '1d'

    -- Scores por módulo (-1.0 a +1.0)
    score_news      FLOAT,
    score_calendar  FLOAT,
    score_onchain   FLOAT,
    score_institutional FLOAT,
    score_micro     FLOAT,

    -- Pesos aplicados (configurables, suman 1.0)
    weight_news         FLOAT NOT NULL DEFAULT 0.20,
    weight_calendar     FLOAT NOT NULL DEFAULT 0.25,
    weight_onchain      FLOAT NOT NULL DEFAULT 0.25,
    weight_institutional FLOAT NOT NULL DEFAULT 0.20,
    weight_micro        FLOAT NOT NULL DEFAULT 0.10,

    -- Score final ponderado
    bias_score      FLOAT NOT NULL,               -- [-1, +1]
    bias_label      VARCHAR(20) NOT NULL,
    -- 'strong_bull' | 'bull' | 'neutral' | 'bear' | 'strong_bear'

    -- Contexto para el LLM
    top_drivers     JSONB,                        -- [{module, signal, weight, description}]
    regime          VARCHAR(20),                  -- 'bull_market' | 'bear_market' | 'accumulation' | 'distribution'
    halving_phase   VARCHAR(20),                  -- 'pre_halving' | 'post_halving_bull' | 'bear' | 'accumulation'

    computed_by     VARCHAR(30) NOT NULL DEFAULT 'pipeline_v1'
);

CREATE INDEX IF NOT EXISTS idx_bias_ts    ON bias_snapshots (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_bias_asset ON bias_snapshots (asset, timeframe, timestamp DESC);


-- ============================================================
--  MÓDULO 7 — source_weights  (configuración, no datos)
-- ============================================================
CREATE TABLE IF NOT EXISTS source_weights (
    source      VARCHAR(80) PRIMARY KEY,
    tier        SMALLINT    NOT NULL,    -- 1, 2, 3
    weight      FLOAT       NOT NULL,   -- multiplicador base
    active      BOOLEAN     NOT NULL DEFAULT TRUE,
    notes       TEXT
);

INSERT INTO source_weights (source, tier, weight, notes) VALUES
    ('reuters',         1, 1.0,  'Tier 1 — wire service'),
    ('bloomberg',       1, 1.0,  'Tier 1 — wire service'),
    ('wsj',             1, 0.9,  'Tier 1 — financial press'),
    ('ft',              1, 0.9,  'Tier 1 — financial press'),
    ('coindesk',        2, 0.7,  'Tier 2 — crypto specialist'),
    ('theblock',        2, 0.7,  'Tier 2 — crypto specialist'),
    ('cointelegraph',   2, 0.6,  'Tier 2 — crypto media'),
    ('decrypt',         2, 0.6,  'Tier 2 — crypto media'),
    ('cryptopanic',     2, 0.5,  'Tier 2 — aggregator'),
    ('reddit_bitcoin',  3, 0.3,  'Tier 3 — social'),
    ('reddit_cryptocurrency', 3, 0.25, 'Tier 3 — social'),
    ('newsbtc',         3, 0.2,  'Tier 3 — blog'),
    ('bitcoinist',      3, 0.2,  'Tier 3 — blog'),
    ('cryptonews',      3, 0.2,  'Tier 3 — blog'),
    ('google_btc_en',   3, 0.15, 'Tier 3 — aggregated search'),
    ('google_btc_es',   3, 0.10, 'Tier 3 — aggregated search')
ON CONFLICT (source) DO NOTHING;
