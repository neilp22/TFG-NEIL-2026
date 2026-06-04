-- schema_bot.sql — Tablas para el bot de paper trading Bybit Demo

CREATE TABLE IF NOT EXISTS live_trades (
    id              BIGSERIAL PRIMARY KEY,
    bybit_order_id  VARCHAR(50),
    entry_time      TIMESTAMPTZ NOT NULL,
    entry_price     FLOAT NOT NULL,
    direction       VARCHAR(5) NOT NULL,
    size_usd        FLOAT NOT NULL,
    size_btc        FLOAT NOT NULL,
    leverage        FLOAT NOT NULL DEFAULT 1.0,
    stop_loss       FLOAT NOT NULL,
    take_profit     FLOAT NOT NULL,
    sl_pct          FLOAT,
    tp_pct          FLOAT,
    risk_reward     FLOAT,
    trigger_type    VARCHAR(20),
    trigger_detail  TEXT,
    confluence_score FLOAT,
    confluence_label VARCHAR(20),
    agent_decision  TEXT,
    signal_snapshot JSONB,
    session         VARCHAR(20),
    exit_time       TIMESTAMPTZ,
    exit_price      FLOAT,
    exit_reason     VARCHAR(20),
    pnl_usd         FLOAT,
    pnl_pct         FLOAT,
    fees_usd        FLOAT,
    funding_usd     FLOAT,
    net_pnl_usd     FLOAT,
    holding_hours   FLOAT,
    status          VARCHAR(15) DEFAULT 'open',
    capital_after   FLOAT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_live_status ON live_trades (status, entry_time DESC);
CREATE INDEX IF NOT EXISTS idx_live_open   ON live_trades (status) WHERE status = 'open';

CREATE TABLE IF NOT EXISTS price_alerts (
    id              BIGSERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    level_price     FLOAT NOT NULL,
    level_type      VARCHAR(30) NOT NULL,
    timeframe       VARCHAR(5),
    direction       VARCHAR(5),
    tolerance_pct   FLOAT DEFAULT 0.15,
    expires_at      TIMESTAMPTZ,
    triggered       BOOLEAN DEFAULT FALSE,
    triggered_at    TIMESTAMPTZ,
    notes           TEXT
);
CREATE INDEX IF NOT EXISTS idx_alerts_active ON price_alerts (triggered, expires_at)
    WHERE triggered = FALSE;

CREATE TABLE IF NOT EXISTS bot_state (
    id              INT PRIMARY KEY DEFAULT 1,
    is_running      BOOLEAN DEFAULT FALSE,
    started_at      TIMESTAMPTZ,
    initial_capital FLOAT DEFAULT 50000,
    current_capital FLOAT,
    last_check      TIMESTAMPTZ,
    last_agent_call TIMESTAMPTZ,
    agent_calls_today INT DEFAULT 0,
    total_trades    INT DEFAULT 0,
    open_trades     INT DEFAULT 0,
    config          JSONB,
    CONSTRAINT single_row CHECK (id = 1)
);
INSERT INTO bot_state (id) VALUES (1) ON CONFLICT (id) DO NOTHING;
