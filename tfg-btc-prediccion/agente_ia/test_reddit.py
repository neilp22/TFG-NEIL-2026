import sys, os
sys.path.insert(0, '.')

from dotenv import load_dotenv
load_dotenv('config/.env')

client_id     = os.getenv("REDDIT_CLIENT_ID", "")
client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")

if not client_id:
    print("⚠️  REDDIT_CLIENT_ID no configurado en config/.env")
    print("   Para configurarlo:")
    print("   1. Ve a https://www.reddit.com/prefs/apps")
    print("   2. Click 'Create App' → tipo 'script'")
    print("   3. Nombre: btc_tfg | redirect: http://localhost:8080")
    print("   4. Copia client_id (debajo del nombre) y client_secret")
    print("   5. Ponlos en config/.env")
else:
    import praw
    try:
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=os.getenv("REDDIT_USER_AGENT", "btc_tfg/1.0"),
        )
        reddit.read_only = True
        for sub_name in ["Bitcoin", "CryptoCurrency"]:
            sub   = reddit.subreddit(sub_name)
            posts = list(sub.new(limit=5))
            print(f"✅ r/{sub_name}: {len(posts)} posts recientes")
            for p in posts[:2]:
                from datetime import datetime, timezone
                ts = datetime.fromtimestamp(p.created_utc, tz=timezone.utc)
                print(f"   [{ts.strftime('%H:%M UTC')}] {p.title[:70]}")
            print()
    except Exception as e:
        print(f"❌ Error Reddit: {e}")
