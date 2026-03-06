# pipeline/rss_fetcher.py
import feedparser
import time
import os, sys
from datetime import datetime, timezone
from sqlalchemy import text
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from db.db_utils import get_engine

# Feeds RSS de fuentes cripto de alta calidad - sin clave, sin registro
RSS_FEEDS = {
    'coindesk':        'https://www.coindesk.com/arc/outboundfeeds/rss/',
    'cointelegraph':   'https://cointelegraph.com/rss',
    'bitcoinmagazine': 'https://bitcoinmagazine.com/.rss/full/',
    'decrypt':         'https://decrypt.co/feed',
    'cryptoslate':     'https://cryptoslate.com/feed/',
    'newsbtc':         'https://www.newsbtc.com/feed/',
}

# Keywords para filtrar solo artículos relevantes a BTC
BTC_KEYWORDS = {
    'bitcoin', 'btc', 'satoshi', 'crypto', 'blockchain',
    'halving', 'mining', 'hodl', 'coinbase', 'binance'
}

def is_btc_relevant(title: str, summary: str = '') -> bool:
    text = (title + ' ' + summary).lower()
    return any(kw in text for kw in BTC_KEYWORDS)

def parse_feed(source: str, url: str, asset: str = 'BTC') -> list:
    feed = feedparser.parse(url)
    posts = []
    for entry in feed.entries:
        title   = entry.get('title', '').strip()
        summary = entry.get('summary', '')[:300]
        if not title or not is_btc_relevant(title, summary):
            continue
        # Parsear fecha
        if hasattr(entry, 'published_parsed') and entry.published_parsed:
            ts = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        else:
            ts = datetime.now(timezone.utc)
        text = title
        if summary:
            # Limpiar HTML básico del summary
            import re
            clean = re.sub(r'<[^>]+>', '', summary).strip()
            if clean:
                text += ' ' + clean
        posts.append({
            'timestamp': ts,
            'asset':     asset,
            'source':    f'rss_{source}',
            'text':      text[:800],
            'url':       entry.get('link', ''),
            'processed': False
        })
    return posts

def insert_texts(posts: list) -> int:
    if not posts:
        return 0
    engine = get_engine()
    inserted = 0
    with engine.begin() as conn:
        for p in posts:
            res = conn.execute(text("""
                INSERT INTO raw_texts (timestamp, asset, source, text, url, processed)
                VALUES (:timestamp, :asset, :source, :text, :url, :processed)
                ON CONFLICT DO NOTHING
            """), p)
            inserted += res.rowcount
    return inserted

def scrape_all_feeds(asset: str = 'BTC') -> int:
    total = 0
    for source, url in RSS_FEEDS.items():
        print(f'  Scrapeando {source}...', end=' ')
        try:
            posts = parse_feed(source, url, asset)
            n     = insert_texts(posts)
            total += n
            print(f'{len(posts)} artículos encontrados, {n} nuevos insertados')
        except Exception as e:
            print(f'ERROR: {e}')
        time.sleep(1)
    return total

if __name__ == '__main__':
    # Instalar feedparser si no lo tienes
    # pip install feedparser
    print('Scrapeando feeds RSS de noticias cripto...')
    total = scrape_all_feeds()
    print(f'\nTotal insertado: {total} artículos en raw_texts')