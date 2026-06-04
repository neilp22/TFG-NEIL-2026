import sys, os
sys.path.insert(0, '.')

from dotenv import load_dotenv
load_dotenv('config/.env')

from sqlalchemy import text
from db.db_utils import get_engine

print("=== Test conexión BD ===")
try:
    engine = get_engine()
    with engine.connect() as conn:
        ts  = conn.execute(text("SELECT NOW()")).scalar()
        cnt = conn.execute(text("SELECT COUNT(*) FROM raw_texts")).scalar()
    print(f"✅ Conectado — {ts}")
    print(f"   raw_texts: {cnt} filas actuales")
except Exception as e:
    print(f"❌ Error BD: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

print("\n=== Test columna url en raw_texts ===")
try:
    with engine.connect() as conn:
        cols = conn.execute(text("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name='raw_texts' AND table_schema='public'
        """)).fetchall()
    col_names = [r[0] for r in cols]
    print(f"Columnas: {col_names}")
    if 'url' in col_names:
        print("✅ Columna 'url' existe")
    else:
        print("⚠️  Columna 'url' NO existe — se necesita ALTER TABLE")
except Exception as e:
    print(f"❌ {e}")

print("\n=== Test scrape RSS (3 feeds, 1 día) ===")
import feedparser, calendar
from datetime import datetime, timezone, timedelta
from dateutil import parser as dateparser

FEEDS_TEST = [
    ("google_btc_en", "https://news.google.com/rss/search?q=bitcoin+cryptocurrency&hl=en-US&gl=US&ceid=US:en"),
    ("cointelegraph",  "https://cointelegraph.com/rss"),
    ("newsbtc",        "https://www.newsbtc.com/feed/"),
]

BTC_KEYWORDS = {'bitcoin','btc','crypto','cryptocurrency','blockchain','halving','binance','coinbase','defi','eth','stablecoin','etf','sec'}

def is_btc(t): return any(k in t.lower() for k in BTC_KEYWORDS)

def parse_ts(entry):
    for field in ('published_parsed','updated_parsed'):
        val = getattr(entry, field, None)
        if val:
            try: return datetime.fromtimestamp(calendar.timegm(val), tz=timezone.utc)
            except: pass
    for field in ('published','updated'):
        val = getattr(entry, field, None)
        if val:
            try:
                dt = dateparser.parse(val)
                return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            except: pass
    return datetime.now(timezone.utc)

cutoff = datetime.now(timezone.utc) - timedelta(days=1)
articles = []

for source, url in FEEDS_TEST:
    try:
        feed = feedparser.parse(url, request_headers={"User-Agent": "Mozilla/5.0"})
        for e in feed.entries:
            txt = f"{getattr(e,'title','')}. {getattr(e,'summary','')}"
            if not is_btc(txt): continue
            ts = parse_ts(e)
            if ts < cutoff: continue
            articles.append({
                'source':    source,
                'text':      txt.strip()[:4000],
                'url':       getattr(e, 'link', ''),
                'timestamp': ts,
                'asset':     'BTC',
            })
        print(f"  {source}: {len([a for a in articles if a['source']==source])} artículos recientes")
    except Exception as ex:
        print(f"  ❌ {source}: {ex}")

print(f"\nTotal artículos a insertar: {len(articles)}")

if articles:
    print("\n=== Insertando en BD ===")
    inserted = 0
    with engine.begin() as conn:
        for row in articles[:10]:  # solo primeros 10 para el test
            try:
                r = conn.execute(text("""
                    INSERT INTO raw_texts (timestamp, asset, source, text, url, processed)
                    VALUES (:ts, :asset, :source, :text, :url, false)
                    ON CONFLICT (url) DO NOTHING
                """), {'ts': row['timestamp'], 'asset': row['asset'],
                       'source': row['source'], 'text': row['text'],
                       'url': row.get('url','')})
                inserted += r.rowcount
            except Exception as e:
                print(f"  ⚠️  Insert error: {e}")

    print(f"✅ Insertados: {inserted} / Duplicados ignorados: {min(10,len(articles))-inserted}")

    print("\n=== Verificando últimas entradas en BD ===")
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT source, timestamp, LEFT(text,80) FROM raw_texts ORDER BY id DESC LIMIT 5"
        )).fetchall()
    for r in rows:
        print(f"  [{r[0]}] {r[1]} | {r[2]}")
