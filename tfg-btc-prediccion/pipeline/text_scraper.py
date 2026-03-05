# pipeline/text_scraper.py
import praw
import pandas as pd
from datetime import datetime, timezone, timedelta
from sqlalchemy import text
import os, sys
from dotenv import load_dotenv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from db.db_utils import get_engine
load_dotenv('config/.env')
 
# Subreddits a monitorizar
SUBREDDITS = ['Bitcoin', 'CryptoCurrency', 'CryptoMarkets']
 
def get_reddit_client():
    return praw.Reddit(
        client_id=os.getenv('REDDIT_CLIENT_ID'),
        client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
        user_agent=os.getenv('REDDIT_USER_AGENT', 'tfg_scraper/1.0')
    )
 
def scrape_subreddit(reddit, subreddit_name: str, limit: int = 500):
    """Descarga los posts más recientes de un subreddit."""
    sub = reddit.subreddit(subreddit_name)
    posts = []
    for post in sub.new(limit=limit):
        # Combinar título y selftext para más contexto
        text = post.title
        if post.selftext and post.selftext not in ('[removed]', '[deleted]'):
            text += ' ' + post.selftext[:500]  # limitar longitud
        posts.append({
            'timestamp': datetime.fromtimestamp(post.created_utc, tz=timezone.utc),
            'asset': 'BTC',
            'source': f'reddit_{subreddit_name.lower()}',
            'text': text.strip(),
            'url': f'https://reddit.com{post.permalink}',
            'processed': False
        })
    return posts
 
def insert_texts(posts: list):
    """Inserta los textos en raw_texts. Evita duplicados por URL."""
    engine = get_engine()
    inserted = 0
    with engine.begin() as conn:
        for post in posts:
            result = conn.execute(text("""
                INSERT INTO raw_texts (timestamp, asset, source, text, url, processed)
                VALUES (:timestamp, :asset, :source, :text, :url, :processed)
                ON CONFLICT DO NOTHING
            """), post)
            inserted += result.rowcount
    return inserted
 
def scrape_all(limit_per_sub: int = 500):
    reddit = get_reddit_client()
    total = 0
    for sub in SUBREDDITS:
        print(f'Scrapeando r/{sub}...')
        posts = scrape_subreddit(reddit, sub, limit=limit_per_sub)
        n = insert_texts(posts)
        print(f'  Insertados {n} textos nuevos de r/{sub}')
        total += n
    print(f'Total insertado: {total} textos')
    return total
 
if __name__ == '__main__':
    scrape_all(limit_per_sub=500)
