import sys, os, time
sys.path.insert(0, '.')

from dotenv import load_dotenv
load_dotenv('config/.env')

import feedparser
from datetime import datetime, timezone, timedelta
from dateutil import parser as dateparser

RSS_FEEDS = [
    ("coindesk",        "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("cointelegraph",   "https://cointelegraph.com/rss"),
    ("decrypt",         "https://decrypt.co/feed"),
    ("theblock",        "https://www.theblock.co/rss.xml"),
    ("bitcoinmagazine", "https://bitcoinmagazine.com/.rss/full/"),
    ("newsbtc",         "https://www.newsbtc.com/feed/"),
    ("cryptoslate",     "https://cryptoslate.com/feed/"),
    ("ambcrypto",       "https://ambcrypto.com/feed/"),
    ("beincrypto",      "https://beincrypto.com/feed/"),
    ("cryptonews",      "https://cryptonews.com/news/feed/"),
    ("cryptopotato",    "https://cryptopotato.com/feed/"),
    ("bitcoinist",      "https://bitcoinist.com/feed/"),
    ("utoday",          "https://u.today/rss"),
    ("zycrypto",        "https://zycrypto.com/feed/"),
    ("reuters_crypto",  "https://feeds.reuters.com/reuters/cryptoNews"),
    ("google_btc_en",   "https://news.google.com/rss/search?q=bitcoin+cryptocurrency&hl=en-US&gl=US&ceid=US:en"),
    ("google_btc_es",   "https://news.google.com/rss/search?q=bitcoin&hl=es&gl=ES&ceid=ES:es"),
    ("google_etf",      "https://news.google.com/rss/search?q=bitcoin+ETF+regulation&hl=en-US&gl=US&ceid=US:en"),
    ("google_macro",    "https://news.google.com/rss/search?q=bitcoin+federal+reserve+inflation&hl=en-US&gl=US&ceid=US:en"),
    ("google_adoption", "https://news.google.com/rss/search?q=bitcoin+institutional+adoption&hl=en-US&gl=US&ceid=US:en"),
]

cutoff = datetime.now(timezone.utc) - timedelta(days=3)
ok_feeds, fail_feeds = [], []

for source, url in RSS_FEEDS:
    try:
        feed = feedparser.parse(url, request_headers={"User-Agent": "Mozilla/5.0"})
        n = len(feed.entries)
        recent = 0
        for e in feed.entries:
            ts = None
            if hasattr(e, 'published_parsed') and e.published_parsed:
                try:
                    import calendar
                    ts = datetime.fromtimestamp(calendar.timegm(e.published_parsed), tz=timezone.utc)
                except: pass
            elif hasattr(e, 'published') and e.published:
                try: ts = dateparser.parse(e.published).astimezone(timezone.utc)
                except: pass
            if ts and ts > cutoff:
                recent += 1

        if n > 0:
            last_title = getattr(feed.entries[0], 'title', '')[:60] if feed.entries else ''
            print(f"✅ {source:<25} {n:3} entries | {recent:3} recientes | {last_title}")
            ok_feeds.append(source)
        else:
            status = getattr(feed, 'status', '?')
            print(f"⚠️  {source:<25}   0 entries (status={status})")
            fail_feeds.append(source)
        time.sleep(0.4)
    except Exception as ex:
        print(f"❌ {source:<25} ERROR: {str(ex)[:80]}")
        fail_feeds.append(source)
        time.sleep(0.4)

print(f"\n{'='*60}")
print(f"Feeds OK:   {len(ok_feeds)}/{len(RSS_FEEDS)}")
print(f"Feeds FAIL: {len(fail_feeds)}/{len(RSS_FEEDS)}")
if fail_feeds:
    print(f"Fallaron: {fail_feeds}")
