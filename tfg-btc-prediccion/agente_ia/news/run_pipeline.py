"""
run_pipeline.py — Orquestador principal del pipeline de bias BTC

Uso:
  python agente_ia/news/run_pipeline.py           # pipeline completo
  python agente_ia/news/run_pipeline.py --context # solo mostrar contexto LLM
"""

import sys, importlib.util, logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
_HERE = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / 'config' / '.env')

from sqlalchemy import text
from db.db_utils import get_engine

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


def _load(filename: str):
    path = _HERE / filename
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_full_pipeline(days_back: int = 1) -> str:
    """
    Ejecuta el pipeline completo en orden:
    1. news RSS
    2. calendar scraper (próximos 7 días)
    3. update_actuals (rellenar datos reales publicados)
    4. on-chain scraper
    5. micro scraper
    6. ETF scraper
    7. bias snapshot + formato LLM

    Retorna el contexto formateado para el LLM.
    """
    engine = get_engine()
    log.info("══ PIPELINE START ══════════════════════════════")

    # 0 — Actualizar daily_features (OHLCV + indicadores técnicos)
    log.info("Paso 0/7 — Price updater (daily_features)")
    try:
        spec = importlib.util.spec_from_file_location(
            "price_updater", _ROOT / "agente_ia" / "02_price_updater.py"
        )
        m_price = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m_price)
        upd = m_price.run_update()
        log.info(f"  daily_features: {upd.get('status')} — {upd.get('rows_new', 0)} filas nuevas hasta {upd.get('date_to', 'N/A')}")
    except Exception as e:
        log.warning(f"  Price updater FAILED: {e}")

    # 1 — Noticias RSS
    log.info("Paso 1/7 — News RSS")
    m_news = None
    try:
        m_news = _load("01_news_scraper.py")
        news_result = m_news.scrape_rss(days_back=days_back)
        log.info(f"  News: {news_result.get('total_inserted', 0)} artículos nuevos")
    except Exception as e:
        log.warning(f"  News FAILED: {e}")

    # 1b — Scoring de artículos (actualiza sentiment_decay en raw_texts)
    log.info("Paso 1b/7 — Scoring artículos")
    try:
        if m_news is None:
            m_news = _load("01_news_scraper.py")
        ctx = m_news.build_news_context(hours_back=6, engine=engine)
        log.info(f"  Scored: {ctx['aggregate']['article_count']} artículos en últimas 6h")
    except Exception as e:
        log.warning(f"  Scoring FAILED: {e}")

    # 2 — Calendar (próximos 7 días)
    log.info("Paso 2/7 — Calendar scraper")
    try:
        m_cal = _load("02_calendar_scraper.py")
        cal_result = m_cal.run_all(days_ahead=7)
        log.info(f"  Calendar: {cal_result.get('inserted', 0)} eventos")
    except Exception as e:
        log.warning(f"  Calendar FAILED: {e}")

    # 3 — Update actuals
    log.info("Paso 3/7 — Update actuals")
    try:
        updated = m_cal.update_actuals(engine)
        log.info(f"  Actuals updated: {updated}")
    except Exception as e:
        log.warning(f"  Update actuals FAILED: {e}")

    # 4 — On-chain
    log.info("Paso 4/7 — On-chain scraper")
    try:
        m_onchain = _load("03_onchain_scraper.py")
        oc_result = m_onchain.run_all(engine)
        log.info(f"  On-chain: {oc_result}")
    except Exception as e:
        log.warning(f"  On-chain FAILED: {e}")

    # 5 — Micro
    log.info("Paso 5/7 — Micro scraper")
    try:
        m_micro = _load("04_micro_scraper.py")
        micro_result = m_micro.run_all(engine)
        log.info(f"  Micro: funding={micro_result.get('funding', {}).get('annual_pct', 'N/A'):.2f}%" if micro_result.get('funding') else "  Micro: ok")
    except Exception as e:
        log.warning(f"  Micro FAILED: {e}")

    # 6 — ETF
    log.info("Paso 6/7 — ETF scraper")
    try:
        m_etf = _load("05_etf_scraper.py")
        etf_result = m_etf.run_all(engine)
        log.info(f"  ETF: {etf_result.get('records', 0)} registros")
    except Exception as e:
        log.warning(f"  ETF FAILED: {e}")

    # 7 — Bias snapshot
    log.info("Paso 7/7 — Bias calculator")
    try:
        m_bias = _load("06_bias_calculator.py")
        snapshot = m_bias.build_bias_snapshot(engine)
        context  = m_bias.format_llm_context(snapshot)
        log.info(f"  Bias: {snapshot['bias_score']:+.3f} ({snapshot['bias_label']})")
        log.info("══ PIPELINE COMPLETE ════════════════════════════")
        return context
    except Exception as e:
        log.error(f"  Bias FAILED: {e}")
        log.info("══ PIPELINE COMPLETE (con errores) ═════════════")
        return f"[ERROR en bias calculator: {e}]"


def get_llm_context(max_age_hours: int = 2) -> str:
    """
    Retorna el último bias_snapshot formateado.
    Si tiene más de max_age_hours, recalcula automáticamente.
    Esta es la función que llama el agente LLM.
    """
    engine = get_engine()

    # Verificar si hay snapshot reciente
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    try:
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT bias_score, bias_label, score_news, score_calendar,
                       score_onchain, score_institutional, score_micro,
                       timestamp
                FROM bias_snapshots
                WHERE asset = 'BTC' AND timestamp >= :cutoff
                ORDER BY timestamp DESC
                LIMIT 1
            """), {"cutoff": cutoff}).fetchone()
    except Exception:
        row = None

    if row:
        log.info(f"Usando snapshot cacheado: {row.timestamp} — {row.bias_label}")
        # Re-generar el formato con los datos del snapshot (sin recalcular)
        try:
            m_bias = _load("06_bias_calculator.py")
            # Build un snapshot minimal desde la DB
            snapshot = {
                "timestamp":   row.timestamp.strftime("%Y-%m-%d %H:%M"),
                "asset":       "BTC",
                "timeframe":   "1h",
                "scores": {
                    "news":          row.score_news,
                    "calendar":      row.score_calendar,
                    "onchain":       row.score_onchain,
                    "micro":         row.score_micro,
                    "institutional": row.score_institutional,
                },
                "bias_score":         row.bias_score,
                "bias_label":         row.bias_label,
                "top_drivers":        [],
                "halving":            m_bias.get_halving_info(),
                "fear_greed":         None,
                "calendar_upcoming":  [],
                "meta":               {"news_count_6h": 0, "etf_flow_5d_avg": None, "funding_annual": None},
            }
            return m_bias.format_llm_context(snapshot)
        except Exception:
            pass

    log.info("Snapshot desactualizado o ausente — ejecutando pipeline completo")
    return run_full_pipeline()


def mock_llm_context() -> str:
    """Genera un contexto LLM de ejemplo sin acceder a DB ni red. Útil para depurar el formato."""
    m_bias = _load("06_bias_calculator.py")
    snapshot = {
        "timestamp":   datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "asset":       "BTC",
        "timeframe":   "1h",
        "scores": {
            "news":          0.32,
            "calendar":      -0.15,
            "onchain":       None,
            "micro":         0.21,
            "institutional": 0.45,
        },
        "bias_score":  0.21,
        "bias_label":  "bull",
        "top_drivers": [
            {"module": "institutional", "signal": "bullish", "value": 0.45, "description": "Flujos ETF institucionales — alcista"},
            {"module": "news",          "signal": "bullish", "value": 0.32, "description": "Noticias recientes — sentimiento alcista"},
            {"module": "micro",         "signal": "bullish", "value": 0.21, "description": "Microestructura — funding/OI alcista"},
            {"module": "calendar",      "signal": "bearish", "value":-0.15, "description": "Macro económica — sesgo bajista"},
        ],
        "halving":           m_bias.get_halving_info(),
        "fear_greed":        {"value": 62, "label": "Greed"},
        "calendar_upcoming": [
            {"event": "US CPI (YoY)", "datetime": "2026-05-29 12:30 UTC",
             "impact": "high", "hours_away": 14.5, "btc_bias": -1},
        ],
        "meta": {
            "news_count_6h":  23,
            "etf_flow_5d_avg": 312.4,
            "funding_annual":  18.6,
        },
    }
    return m_bias.format_llm_context(snapshot)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", action="store_true", help="Solo mostrar contexto LLM (sin re-scraping si hay cache)")
    ap.add_argument("--mock",    action="store_true", help="Mostrar contexto LLM de ejemplo (sin DB ni red)")
    ap.add_argument("--days",    type=int, default=1)
    args = ap.parse_args()

    if args.mock:
        print(mock_llm_context())
    elif args.context:
        print(get_llm_context())
    else:
        context = run_full_pipeline(days_back=args.days)
        print(context)
