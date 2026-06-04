# pipeline/scheduler.py
import logging, sys, os
from datetime import datetime, timezone
from apscheduler.schedulers.blocking import BlockingScheduler
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler('pipeline_log.txt'), logging.StreamHandler()]
)
log = logging.getLogger(__name__)

def run_daily_pipeline():
    log.info('=' * 50)
    log.info(f'Iniciando pipeline diario: {datetime.now(timezone.utc)}')
    errors = []

    # PASO 1: Precio del dia
    try:
        from pipeline.price_fetcher import load_historical
        from datetime import date, timedelta
        yesterday = (date.today() - timedelta(days=1)).strftime('%-d %b, %Y')
        load_historical(asset='BTC', symbol='BTCUSDT',
                        start=yesterday, intervals=['1d'])
        log.info('Precio: OK')
    except Exception as e:
        log.error(f'Precio: FALLO -- {e}'); errors.append('price')

    # PASO 2: Noticias de CryptoPanic
    try:
        from pipeline.cryptopanic_fetcher import fetch_today
        n = fetch_today(asset='BTC')
        log.info(f'CryptoPanic: {n} noticias nuevas')
    except Exception as e:
        log.error(f'CryptoPanic: FALLO -- {e}'); errors.append('cryptopanic')

    # PASO 3 (opcional): Reddit PRAW si los credenciales estan disponibles
    if os.getenv('REDDIT_CLIENT_ID',''):
        try:
            from pipeline.text_scraper import scrape_all
            scrape_all(limit_per_sub=100)
            log.info('Reddit PRAW: OK')
        except Exception as e:
            log.error(f'Reddit PRAW: FALLO -- {e}'); errors.append('reddit')
    else:
        log.info('Reddit PRAW: omitido (sin credenciales)')

    # PASO 4: Fear & Greed Index
    try:
        from pipeline.fear_greed_fetcher import fetch_fear_greed_history, upsert_fear_greed
        df = fetch_fear_greed_history()
        upsert_fear_greed(df)
        log.info('Fear & Greed: OK')
    except Exception as e:
        log.error(f'Fear & Greed: FALLO -- {e}'); errors.append('fear_greed')

    # PASO 5: Procesar textos con FinBERT
    try:
        from pipeline.sentiment_processor import process_all_texts
        n = process_all_texts(batch_size=200)
        log.info(f'FinBERT: {n} textos procesados')
    except Exception as e:
        log.error(f'FinBERT: FALLO -- {e}'); errors.append('finbert')

    # PASO 6: Reconstruir daily_features
    try:
        from pipeline.feature_builder import build_features
        build_features('BTC')
        log.info('Feature builder: OK')
    except Exception as e:
        log.error(f'Feature builder: FALLO -- {e}'); errors.append('features')

    if errors: log.warning(f'Pipeline completado con errores en: {errors}')
    else:       log.info('Pipeline completado sin errores.')

def _load_agente_module(filename):
    """Carga un módulo de agente_ia con nombre que empieza por dígito."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        filename.replace('.py', ''),
        os.path.join(os.path.dirname(__file__), '..', 'agente_ia', filename)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_news_scraper_v2():
    """Scraper de noticias agente_ia (RSS + Reddit + CryptoPanic) — cada 4h."""
    log.info("Iniciando news_scraper_v2 (agente_ia)")
    try:
        scraper = _load_agente_module('01_news_scraper.py')
        result = scraper.run_all(days_back=1)
        log.info(f"news_scraper_v2: {result.get('total_nuevos', 0)} nuevos artículos")
    except Exception as e:
        log.error(f"news_scraper_v2: FALLO -- {e}")


def run_price_updater_v2():
    """Actualización incremental de daily_features vía Binance — 00:20 UTC."""
    log.info("Iniciando price_updater_v2 (agente_ia)")
    try:
        updater = _load_agente_module('02_price_updater.py')
        result = updater.run_update()
        log.info(f"price_updater_v2: {result}")
    except Exception as e:
        log.error(f"price_updater_v2: FALLO -- {e}")


if __name__ == '__main__':
    run_daily_pipeline()  # Prueba inmediata al arrancar
    scheduler = BlockingScheduler(timezone='UTC')

    # Jobs originales del pipeline de ingesta
    scheduler.add_job(run_daily_pipeline, 'cron', hour=0, minute=10,
                      id='daily_pipeline', name='Pipeline diario')

    # Jobs agente_ia — NO reemplazan los originales, son adicionales
    scheduler.add_job(run_news_scraper_v2, 'cron', hour='*/4', minute=15,
                      id='news_scraper_v2', name='Scraper noticias (agente_ia, cada 4h)')
    scheduler.add_job(run_price_updater_v2, 'cron', hour=0, minute=20,
                      id='price_updater_v2', name='Price updater (agente_ia, 00:20 UTC)')

    print('Scheduler activo.')
    print('  00:10 UTC — Pipeline diario (existente)')
    print('  00:20 UTC — Price updater v2 (agente_ia)')
    print('  Cada 4h   — News scraper v2 (agente_ia)')
    scheduler.start()
