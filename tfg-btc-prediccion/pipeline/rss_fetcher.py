# pipeline/rss_fetcher.py
# Descarga noticias de BTC desde feeds RSS públicos (sin API key)
# Fuentes: CoinDesk, CoinTelegraph, Bitcoin Magazine, Decrypt, The Block, Bitcoinist

import feedparser
import time
import os
import sys
from datetime import datetime, timezone
from sqlalchemy import text

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from db.db_utils import get_engine

# ── Feeds RSS de BTC (todos gratuitos, sin registro) ─────────────────────────
RSS_FEEDS = [
    {
        'url': 'https://www.coindesk.com/arc/outboundfeeds/rss/',
        'source': 'coindesk'
    },
    {
        'url': 'https://cointelegraph.com/rss',
        'source': 'cointelegraph'
    },
    {
        'url': 'https://bitcoinmagazine.com/.rss/full/',
        'source': 'bitcoin_magazine'
    },
    {
        'url': 'https://decrypt.co/feed',
        'source': 'decrypt'
    },
    {
        'url': 'https://www.theblock.co/rss.xml',
        'source': 'theblock'
    },
    {
        'url': 'https://bitcoinist.com/feed/',
        'source': 'bitcoinist'
    },
]

# Keywords para filtrar noticias relevantes a BTC
BTC_KEYWORDS = {
    'bitcoin', 'btc', 'crypto', 'cryptocurrency', 'blockchain',
    'satoshi', 'halving', 'mining', 'binance', 'coinbase',
    'altcoin', 'defi', 'hodl', 'bull', 'bear', 'rally', 'crash'
}


def is_btc_relevant(text: str) -> bool:
    """Comprueba si el texto contiene al menos una keyword de BTC."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in BTC_KEYWORDS)


def parse_date(entry) -> datetime:
    """Extrae el timestamp del entry RSS con fallback a ahora."""
    if hasattr(entry, 'published_parsed') and entry.published_parsed:
        try:
            return datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        except Exception:
            pass
    if hasattr(entry, 'updated_parsed') and entry.updated_parsed:
        try:
            return datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
        except Exception:
            pass
    return datetime.now(timezone.utc)


def fetch_feed(feed_config: dict, asset: str = 'BTC') -> list:
    """Descarga y parsea un feed RSS, filtrando por relevancia BTC."""
    posts = []
    try:
        feed = feedparser.parse(feed_config['url'])
        for entry in feed.entries:
            title = entry.get('title', '').strip()
            summary = entry.get('summary', '').strip()

            # Texto completo: título + resumen
            full_text = f"{title}. {summary}" if summary else title

            # Filtrar solo noticias relevantes a BTC
            if not is_btc_relevant(full_text):
                continue

            # Limpiar HTML básico del summary
            import re
            full_text = re.sub(r'<[^>]+>', ' ', full_text)
            full_text = re.sub(r'\s+', ' ', full_text).strip()

            # Limitar longitud
            full_text = full_text[:800]

            if not full_text:
                continue

            ts = parse_date(entry)
            url = entry.get('link', '')

            posts.append({
                'timestamp': ts,
                'asset': asset,
                'source': feed_config['source'],
                'text': full_text,
                'url': url,
                'processed': False
            })

    except Exception as e:
        print(f"  ⚠ Error en feed {feed_config['source']}: {e}")

    return posts


def insert_texts(posts: list) -> int:
    """Inserta textos en raw_texts. Evita duplicados por URL."""
    if not posts:
        return 0

    engine = get_engine()
    inserted = 0

    with engine.begin() as conn:
        for p in posts:
            res = conn.execute(text("""
                INSERT INTO raw_texts
                    (timestamp, asset, source, text, url, processed)
                VALUES
                    (:timestamp, :asset, :source, :text, :url, :processed)
                ON CONFLICT DO NOTHING
            """), p)
            inserted += res.rowcount

    return inserted


def fetch_all_feeds(asset: str = 'BTC') -> int:
    """Descarga todos los feeds RSS e inserta en raw_texts."""
    total = 0
    for feed_config in RSS_FEEDS:
        print(f"  Descargando {feed_config['source']}...")
        posts = fetch_feed(feed_config, asset=asset)
        n = insert_texts(posts)
        total += n
        print(f"    → {len(posts)} noticias encontradas, {n} nuevas insertadas")
        time.sleep(1)  # Respetar servidores RSS

    return total


if __name__ == '__main__':
    print("Descargando noticias BTC desde feeds RSS...")
    total = fetch_all_feeds()
    print(f"\nTotal insertado: {total} noticias en raw_texts")
    print("Verifica con: SELECT source, COUNT(*) FROM raw_texts GROUP BY source;")