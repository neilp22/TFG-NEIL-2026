import sys, os, requests
sys.path.insert(0, '.')

from dotenv import load_dotenv
load_dotenv('config/.env')

from dateutil import parser as dateparser
from datetime import timezone

api_key = os.getenv("CRYPTOPANIC_KEY", os.getenv("CRYPTOPANIC_API_KEY", "")).strip()
print(f"API key: {'configurada (' + api_key[:8] + '...)' if api_key else '❌ NO configurada (modo público)'}")

url    = "https://cryptopanic.com/api/v1/posts/"
params = {"currencies": "BTC", "kind": "news", "filter": "important", "public": "true"}
if api_key:
    params["auth_token"] = api_key

try:
    resp = requests.get(url, params=params, timeout=15)
    print(f"HTTP Status: {resp.status_code}")

    if resp.ok:
        data  = resp.json()
        items = data.get("results", [])
        print(f"✅ CryptoPanic: {len(items)} artículos\n")
        for item in items[:5]:
            ts_str = item.get("published_at", "")
            try:
                ts   = dateparser.parse(ts_str).astimezone(timezone.utc)
                hora = ts.strftime("%Y-%m-%d %H:%M UTC")
            except:
                hora = ts_str
            title = item.get("title", "")[:75]
            votes = item.get("votes", {})
            print(f"  [{hora}] {title}")
            print(f"  Votos: +{votes.get('positive',0)} / -{votes.get('negative',0)}")
            print()
    else:
        print(f"❌ Error HTTP: {resp.status_code}")
        print(resp.text[:300])
except Exception as e:
    print(f"❌ Error CryptoPanic: {e}")
