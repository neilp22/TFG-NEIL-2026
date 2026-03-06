# pipeline/cryptopanic_fetcher.py
import requests, time, os, sys
from datetime import datetime, timezone
from sqlalchemy import text
from dotenv import load_dotenv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from db.db_utils import get_engine

load_dotenv('config/.env')
BASE_URL = 'https://cryptopanic.com/api/v1/posts/'
API_KEY  = os.getenv('CRYPTOPANIC_KEY')

def fetch_cryptopanic_page(page=1, currency='BTC'):
    params = {'auth_token': API_KEY, 'currencies': currency,
              'kind': 'news', 'page': page, 'public': 'true'}
    r = requests.get(BASE_URL, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def parse_posts(data, asset='BTC'):
    posts = []
    for item in data.get('results', []):
        title  = item.get('title', '').strip()
        source = item.get('source', {}).get('domain', '')
        text   = f'{title} [{source}]' if source else title
        if not text: continue
        published = item.get('published_at') or item.get('created_at')
        ts = datetime.fromisoformat(published.replace('Z', '+00:00'))
        posts.append({'timestamp': ts, 'asset': asset,
                      'source': 'cryptopanic', 'text': text,
                      'url': item.get('url',''), 'processed': False})
    return posts

def insert_texts(posts):
    if not posts: return 0
    engine = get_engine(); inserted = 0
    with engine.begin() as conn:
        for p in posts:
            res = conn.execute(text("""
                INSERT INTO raw_texts (timestamp,asset,source,text,url,processed)
                VALUES (:timestamp,:asset,:source,:text,:url,:processed)
                ON CONFLICT DO NOTHING"""), p)
            inserted += res.rowcount
    return inserted

def fetch_historical(pages=50, asset='BTC'):
    total = 0
    for page in range(1, pages + 1):
        try:
            data  = fetch_cryptopanic_page(page=page, currency=asset)
            posts = parse_posts(data, asset=asset)
            n     = insert_texts(posts)
            total += n
            print(f'  Pagina {page}/{pages}: {n} noticias nuevas insertadas')
            time.sleep(1.5)   # respetar rate limit plan gratuito
            if not data.get('next'): break
        except Exception as e:
            print(f'  Error en pagina {page}: {e}'); time.sleep(5)
    return total

def fetch_today(asset='BTC'):
    data  = fetch_cryptopanic_page(page=1, currency=asset)
    posts = parse_posts(data, asset=asset)
    return insert_texts(posts)

if __name__ == '__main__':
    print('Descargando noticias historicas de CryptoPanic...')
    total = fetch_historical(pages=50)
    print(f'Total insertado: {total} noticias en raw_texts')
