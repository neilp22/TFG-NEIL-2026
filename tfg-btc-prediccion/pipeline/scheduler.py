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

if __name__ == '__main__':
    run_daily_pipeline()  # Prueba inmediata al arrancar
    scheduler = BlockingScheduler(timezone='UTC')
    scheduler.add_job(run_daily_pipeline, 'cron', hour=0, minute=10)
    print('Scheduler activo. Proxima ejecucion: manana a las 00:10 UTC')
    scheduler.start()
