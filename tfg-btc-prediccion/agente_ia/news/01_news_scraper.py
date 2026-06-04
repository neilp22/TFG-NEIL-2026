"""
01_news_scraper.py — Scraper + scoring de noticias BTC via RSS

Uso:
  python agente_ia/news/01_news_scraper.py --days 3
  python agente_ia/news/01_news_scraper.py --hours 6
"""

import os, sys, math, logging, calendar as cal_mod, re, socket
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / 'config' / '.env')

socket.setdefaulttimeout(15)

import feedparser
import requests
from bs4 import BeautifulSoup
from dateutil import parser as dateparser
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from sqlalchemy import text
from db.db_utils import get_engine

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


# ── Fuentes RSS ───────────────────────────────────────────────────────────────

RSS_FEEDS = [
    ("reuters",      "https://feeds.reuters.com/reuters/businessNews",                                                         1),
    ("yahoo_finance","https://finance.yahoo.com/rss/topfinstories",                                                             1),
    ("yahoo_btc",    "https://finance.yahoo.com/rss/2.0/headline?s=BTC-USD&region=US&lang=en-US",                               1),
    ("ft",           "https://www.ft.com/rss/home",                                                                             1),
    ("coindesk",     "https://www.coindesk.com/arc/outboundfeeds/rss/",                                                         2),
    ("theblock",     "https://www.theblock.co/rss.xml",                                                                         2),
    ("decrypt",      "https://decrypt.co/feed",                                                                                 2),
    ("cointelegraph","https://cointelegraph.com/rss",                                                                           2),
    ("bitcoinist",   "https://bitcoinist.com/feed/",                                                                            3),
    ("newsbtc",      "https://www.newsbtc.com/feed/",                                                                           3),
    ("cryptoslate",  "https://cryptoslate.com/feed/",                                                                           3),
    ("ambcrypto",    "https://ambcrypto.com/feed/",                                                                             3),
    ("gnews_btc",    "https://news.google.com/rss/search?q=bitcoin&hl=en-US&gl=US&ceid=US:en",                                  2),
    ("gnews_macro",  "https://news.google.com/rss/search?q=bitcoin+federal+reserve+inflation&hl=en-US&gl=US&ceid=US:en",        2),
    ("gnews_etf",    "https://news.google.com/rss/search?q=bitcoin+ETF+SEC+approval&hl=en-US&gl=US&ceid=US:en",                 2),
    ("gnews_inst",   "https://news.google.com/rss/search?q=bitcoin+blackrock+institutional+buy&hl=en-US&gl=US&ceid=US:en",      2),
    ("gnews_reg",    "https://news.google.com/rss/search?q=bitcoin+regulation+law+government&hl=en-US&gl=US&ceid=US:en",        2),
    ("gnews_macro2", "https://news.google.com/rss/search?q=CPI+inflation+fed+rate+crypto&hl=en-US&gl=US&ceid=US:en",            2),
]

TIER_WEIGHTS  = {1: 1.0, 2: 0.65, 3: 0.30}
SOURCE_TIERS  = {name: tier for name, _, tier in RSS_FEEDS}

BTC_KEYWORDS = {
    'bitcoin', 'btc', 'crypto', 'cryptocurrency', 'blockchain', 'satoshi',
    'halving', 'mining', 'binance', 'coinbase', 'altcoin', 'defi', 'hodl',
    'bull', 'bear', 'rally', 'crash', 'ethereum', 'eth', 'stablecoin',
    'usdt', 'sec', 'etf', 'blackrock', 'microstrategy',
}


# ── Keywords profesionales ────────────────────────────────────────────────────

BULLISH_KEYWORDS = {
    "etf approval":      2.5, "spot etf":         2.0, "institutional buy": 2.0,
    "blackrock":         1.8, "microstrategy":    1.5, "inflow":            1.5,
    "accumulation":      1.4, "treasury":         1.3, "adoption":          1.3,
    "rate cut":          2.0, "fed pivot":        2.0, "dovish":            1.8,
    "cpi miss":          1.8, "inflation fell":   1.7, "inflation lower":   1.7,
    "below expectations":1.5, "soft landing":     1.3, "liquidity":         1.2,
    "all-time high":     2.0, "ath":              1.8, "breakout":          1.6,
    "rally":             1.4, "surge":            1.4, "soared":            1.3,
    "bull":              1.2, "support":          1.1, "recovery":          1.1,
    "rebound":           1.1, "clarity":          1.5, "approved":          1.8,
    "compliant":         1.2, "framework":        1.1, "milestone":         1.2,
    "partnership":       1.1, "upgrade":          1.1, "halving":           1.3,
    "scarcity":          1.2, "store of value":   1.3, "hodl":              1.0,
    "legal":             1.2,
}

BEARISH_KEYWORDS = {
    "ban":               2.5, "banned":           2.5, "lawsuit":           2.0,
    "sec charges":       2.2, "doj":              2.0, "arrested":          1.8,
    "fraud":             2.0, "ponzi":            2.0, "money laundering":  1.8,
    "sanctions":         1.8, "crackdown":        1.8, "rate hike":         2.0,
    "hawkish":           1.8, "cpi beat":         1.8, "inflation rose":    1.7,
    "inflation higher":  1.7, "above expectations":1.5,"recession":         1.5,
    "stagflation":       1.5, "tightening":       1.3, "crash":             2.0,
    "collapse":          2.0, "plunge":           1.8, "dump":              1.6,
    "sell-off":          1.5, "selloff":          1.5, "correction":        1.3,
    "resistance":        1.1, "bear":             1.2, "capitulation":      1.8,
    "liquidation":       1.5, "hack":             2.2, "exploit":           2.0,
    "breach":            1.8, "exchange down":    1.8, "insolvency":        2.0,
    "bankruptcy":        2.0, "outflow":          1.3, "withdrawal":        1.1,
    "bubble":            1.5, "scam":             1.8, "rug pull":          2.0,
    "exit scam":         2.0, "warning":          1.2, "concern":           1.0,
    "fear":              1.0, "uncertainty":      1.0,
}

NEGATION_WORDS = {
    "not", "no", "never", "neither", "nor", "without", "won't",
    "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't",
    "hasn't", "haven't", "hadn't", "cannot", "can't", "couldn't",
    "shouldn't", "wouldn't", "avoid", "deny", "denies", "rejected",
    "fails", "failed", "unlike", "despite", "contrary",
}

NARRATIVE_CATEGORIES = {
    "institutional": ["blackrock", "fidelity", "microstrategy", "etf", "institutional",
                      "fund", "treasury", "corporate", "saylor", "grayscale", "ibit"],
    "macro":         ["fed", "federal reserve", "fomc", "cpi", "inflation", "rate",
                      "gdp", "recession", "yield", "dollar", "dxy", "powell"],
    "regulatory":    ["sec", "cftc", "doj", "regulation", "ban", "law", "congress",
                      "senate", "bill", "compliance", "legal", "lawsuit"],
    "technical":     ["breakout", "support", "resistance", "ath", "all-time high",
                      "rally", "correction", "fibonacci", "rsi", "moving average"],
    "onchain":       ["halving", "mining", "hash rate", "wallet", "exchange", "whale",
                      "blockchain", "transaction", "mempool", "lightning"],
    "geopolitical":  ["war", "conflict", "sanction", "china", "russia", "geopolitical",
                      "country", "government", "election", "political"],
}

IMPACT_WINDOWS = {
    "etf approval":      (72, 0.025),
    "ban":               (48, 0.035),
    "banned":            (48, 0.035),
    "hack":              (24, 0.060),
    "rate cut":          (48, 0.030),
    "fomc":              (48, 0.030),
    "cpi":               (24, 0.050),
    "institutional buy": (48, 0.030),
    "ath":               (24, 0.060),
    "all-time high":     (24, 0.060),
    "lawsuit":           (36, 0.040),
}

TIER_IMPACT_HOURS = {1: 24, 2: 12, 3: 6}

_LABEL_ES = {
    "strong_bull": "BULLISH FUERTE",
    "bullish":     "BULLISH",
    "neutral":     "NEUTRAL",
    "bearish":     "BEARISH",
    "strong_bear": "BEARISH FUERTE",
    "unknown":     "N/D",
}


# ── Helpers de texto ──────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    clean = re.sub(r'<[^>]+>', ' ', text)
    return re.sub(r'\s+', ' ', clean).lower().strip()


def _check_negation(text: str, pos: int) -> bool:
    words = text[:pos].rstrip().split()[-3:]
    return any(w.rstrip(".,!?;:'\"") in NEGATION_WORDS for w in words)


def _find_signals(text: str, keywords: dict) -> list:
    signals = []
    used: list = []  # list of (start, end) already claimed
    for kw, weight in sorted(keywords.items(), key=lambda x: len(x[0]), reverse=True):
        start = 0
        while True:
            pos = text.find(kw, start)
            if pos == -1:
                break
            end = pos + len(kw)
            before_ok = pos == 0 or not text[pos - 1].isalpha()
            after_ok  = end >= len(text) or not text[end].isalpha()
            if before_ok and after_ok:
                overlap = any(s <= pos < e or s < end <= e for s, e in used)
                if not overlap:
                    signals.append({
                        "keyword": kw,
                        "weight":  weight,
                        "negated": _check_negation(text, pos),
                    })
                    used.append((pos, end))
            start = pos + 1
    return signals


def _detect_narratives(text: str) -> list:
    return [cat for cat, kws in NARRATIVE_CATEGORIES.items() if any(kw in text for kw in kws)]


def _get_impact_params(keywords: list, tier: int) -> tuple:
    best_h = TIER_IMPACT_HOURS.get(tier, 6)
    best_l = 0.120
    for kw in keywords:
        if kw in IMPACT_WINDOWS:
            h, l = IMPACT_WINDOWS[kw]
            if h > best_h:
                best_h, best_l = h, l
    return best_h, best_l


def _label(score: Optional[float]) -> str:
    if score is None:
        return "unknown"
    if score >= 0.5:   return "strong_bull"
    if score >= 0.15:  return "bullish"
    if score > -0.15:  return "neutral"
    if score > -0.5:   return "bearish"
    return "strong_bear"


# ── FinBERT (PyTorch directo, sin TF pipeline) ────────────────────────────────

USE_FINBERT = os.getenv("USE_FINBERT", "true").lower() == "true"
_FINBERT_TOKENIZER = None
_FINBERT_MODEL     = None


def _load_finbert():
    global _FINBERT_TOKENIZER, _FINBERT_MODEL
    if _FINBERT_MODEL is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch as _torch
        name = "ProsusAI/finbert"
        _FINBERT_TOKENIZER = AutoTokenizer.from_pretrained(name)
        _FINBERT_MODEL     = AutoModelForSequenceClassification.from_pretrained(name)
        _FINBERT_MODEL.eval()
    return _FINBERT_TOKENIZER, _FINBERT_MODEL


def finbert_score(text_str: str) -> float:
    """Devuelve score -1 (negativo) a +1 (positivo) usando FinBERT vía PyTorch."""
    import torch as _torch
    tok, model = _load_finbert()
    inputs = tok(text_str[:512], return_tensors="pt", truncation=True, padding=True)
    with _torch.no_grad():
        logits = model(**inputs).logits
    probs = _torch.softmax(logits, dim=1)[0]
    # FinBERT label order: 0=positive, 1=negative, 2=neutral
    return float(probs[0] - probs[1])


def _keyword_score(clean: str, bull_sigs: list, bear_sigs: list) -> float:
    """Score -1..+1 basado en keywords."""
    bull_total = sum(s["weight"] * (-0.5 if s["negated"] else 1.0) for s in bull_sigs)
    bear_total = sum(s["weight"] * (-0.5 if s["negated"] else 1.0) for s in bear_sigs)
    bull_score = max(0.0, bull_total)
    bear_score = max(0.0, bear_total)
    raw = (bull_score - bear_score) / (bull_score + bear_score + 1e-9)
    return max(-1.0, min(1.0, raw))


# ── Scoring principal ─────────────────────────────────────────────────────────

def score_article(text: str, source: str, tier: int, published_at: datetime) -> dict:
    """Analiza un artículo, devuelve score con metadatos completos."""
    clean = _clean_text(text)

    bull_sigs = _find_signals(clean, BULLISH_KEYWORDS)
    bear_sigs = _find_signals(clean, BEARISH_KEYWORDS)

    kw_raw_score = _keyword_score(clean, bull_sigs, bear_sigs)

    # FinBERT: si disponible y habilitado, mezcla 50% kw + 50% FinBERT
    scoring_method = "keyword"
    if USE_FINBERT and len(clean) > 20:
        try:
            fb_score  = finbert_score(clean)
            raw_score = 0.5 * kw_raw_score + 0.5 * fb_score
            raw_score = max(-1.0, min(1.0, raw_score))
            scoring_method = "hybrid_finbert"
        except Exception as _e:
            log.debug(f"FinBERT falló, usando keywords: {_e}")
            raw_score = kw_raw_score
    else:
        raw_score = kw_raw_score

    narratives = _detect_narratives(clean)
    all_kw     = [s["keyword"] for s in bull_sigs + bear_sigs]
    impact_h, decay_l = _get_impact_params(all_kw, tier)

    now = datetime.now(timezone.utc)
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)
    hours_since  = max(0.0, (now - published_at).total_seconds() / 3600)
    tier_w       = TIER_WEIGHTS.get(tier, 0.30)
    decay_weight = tier_w * math.exp(-decay_l * hours_since)
    sent_decay   = raw_score * decay_weight

    total_sigs   = len(bull_sigs) + len(bear_sigs)
    confidence   = min(1.0, (total_sigs / 3.0) * tier_w)

    return {
        "sentiment_raw":       round(raw_score, 4),
        "sentiment_decay":     round(sent_decay, 4),
        "bullish_signals":     bull_sigs,
        "bearish_signals":     bear_sigs,
        "narratives":          narratives,
        "impact_window_hours": impact_h,
        "decay_lambda":        decay_l,
        "hours_since":         round(hours_since, 2),
        "weight_applied":      round(decay_weight, 4),
        "confidence":          round(confidence, 3),
        "scoring_method":      scoring_method,
    }


# ── RSS scraping ──────────────────────────────────────────────────────────────

def _is_btc_relevant(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in BTC_KEYWORDS)


def _parse_timestamp(entry) -> datetime:
    for field in ('published_parsed', 'updated_parsed', 'created_parsed'):
        val = getattr(entry, field, None)
        if val:
            return datetime.fromtimestamp(cal_mod.timegm(val), tz=timezone.utc)
    for field in ('published', 'updated', 'created'):
        val = getattr(entry, field, None)
        if val:
            try:
                dt = dateparser.parse(val)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except Exception:
                pass
    return datetime.now(timezone.utc)


def _insert_texts(rows: list, engine) -> int:
    if not rows:
        return 0
    inserted = 0
    with engine.begin() as conn:
        for row in rows:
            url  = row.get('url') or ''
            tier = SOURCE_TIERS.get(row.get('source', ''), 3)
            try:
                if url:
                    result = conn.execute(text("""
                        INSERT INTO raw_texts
                            (timestamp, asset, source, source_tier, text, url, processed)
                        SELECT :ts, :asset, :source, :tier, :text, :url, false
                        WHERE NOT EXISTS (SELECT 1 FROM raw_texts WHERE url = :url)
                    """), {
                        'ts': row['timestamp'], 'asset': row.get('asset', 'BTC'),
                        'source': row['source'], 'tier': tier,
                        'text': row['text'][:4000], 'url': url,
                    })
                else:
                    result = conn.execute(text("""
                        INSERT INTO raw_texts
                            (timestamp, asset, source, source_tier, text, url, processed)
                        VALUES (:ts, :asset, :source, :tier, :text, NULL, false)
                    """), {
                        'ts': row['timestamp'], 'asset': row.get('asset', 'BTC'),
                        'source': row['source'], 'tier': tier,
                        'text': row['text'][:4000],
                    })
                inserted += result.rowcount
            except Exception as e:
                log.debug(f"Insert ({row.get('source')}): {e}")
    return inserted


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=False,
)
def _fetch_rss(source: str, url: str) -> list:
    feed = feedparser.parse(url, request_headers={'User-Agent': 'Mozilla/5.0'})
    rows = []
    for entry in feed.entries:
        title   = entry.get('title', '')
        summary = entry.get('summary', '')
        combined = f"{title}. {summary}"
        if not _is_btc_relevant(combined):
            continue
        rows.append({
            'source':    source,
            'text':      combined.strip(),
            'url':       entry.get('link', ''),
            'timestamp': _parse_timestamp(entry),
            'asset':     'BTC',
        })
    return rows


def scrape_rss(days_back: int = 3) -> dict:
    engine = get_engine()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    stats  = {}
    total  = 0

    def _fetch_one(source: str, url: str):
        try:
            rows = _fetch_rss(source, url)
            return source, [r for r in rows if r['timestamp'] >= cutoff], None
        except Exception as e:
            return source, [], str(e)

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_fetch_one, s, u): s for s, u, _ in RSS_FEEDS}
        for fut in as_completed(futures):
            source, rows, error = fut.result()
            if error:
                log.warning(f"RSS {source} falló: {error}")
                stats[source] = {'scraped': 0, 'inserted': 0, 'error': error}
            else:
                n = _insert_texts(rows, engine)
                stats[source] = {'scraped': len(rows), 'inserted': n}
                total += n
                if rows:
                    log.info(f"RSS {source}: {len(rows)} artículos, {n} nuevos")

    return {'stats': stats, 'total_inserted': total}


# ── Build context ─────────────────────────────────────────────────────────────

def _window_score(articles: list) -> Optional[float]:
    if not articles:
        return None
    total_w = sum(a["weight_applied"] for a in articles)
    if total_w == 0:
        return None
    return round(
        sum(a["sentiment_raw"] * a["weight_applied"] for a in articles) / total_w, 3
    )


def _window_dict(articles: list) -> dict:
    s = _window_score(articles)
    return {"score": s if s is not None else 0.0, "count": len(articles), "label": _label(s)}


def build_news_context(hours_back: int = 6, engine=None) -> dict:
    """Lee raw_texts, score todos los artículos, devuelve contexto completo."""
    if engine is None:
        engine = get_engine()

    now        = datetime.now(timezone.utc)
    cutoff_24h = now - timedelta(hours=24)

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT id, text, source, source_tier, timestamp, url
            FROM raw_texts
            WHERE timestamp >= :cutoff AND asset = 'BTC'
            ORDER BY timestamp DESC
            LIMIT 1000
        """), {"cutoff": cutoff_24h}).fetchall()

    if not rows:
        return _empty_context(now, hours_back)

    # Score all articles in memory
    scored   = []
    db_upd   = []
    for row in rows:
        tier = int(row.source_tier or 3)
        ts   = row.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        res = score_article(row.text, row.source, tier, ts)
        res.update({"id": row.id, "source": row.source, "tier": tier,
                    "timestamp": ts, "url": row.url or "",
                    "headline":  row.text[:120]})
        scored.append(res)
        db_upd.append((row.id, res["sentiment_raw"], res["sentiment_decay"], res["weight_applied"]))

    # Persist scores
    with engine.begin() as conn:
        for aid, raw, decay, weight in db_upd:
            try:
                conn.execute(text("""
                    UPDATE raw_texts
                    SET sentiment_raw=:r, sentiment_decay=:d, impact_weight=:w, processed=TRUE
                    WHERE id=:id
                """), {"id": aid, "r": raw, "d": decay, "w": weight})
            except Exception:
                pass

    def in_window(h: int) -> list:
        cutoff = now - timedelta(hours=h)
        return [a for a in scored if a["timestamp"] >= cutoff]

    arts_1h  = in_window(1)
    arts_3h  = in_window(3)
    arts_main = in_window(hours_back)
    arts_24h  = scored

    # Aggregate
    agg_score = _window_score(arts_main)
    avg_conf  = round(sum(a["confidence"] for a in arts_main) / max(len(arts_main), 1), 3)

    aggregate = {
        "score":         agg_score if agg_score is not None else 0.0,
        "label":         _label(agg_score),
        "confidence":    avg_conf,
        "article_count": len(arts_main),
        "tier1_count":   sum(1 for a in arts_main if a["tier"] == 1),
        "tier2_count":   sum(1 for a in arts_main if a["tier"] == 2),
        "tier3_count":   sum(1 for a in arts_main if a["tier"] == 3),
    }

    by_timewindow = {
        "last_1h":  _window_dict(arts_1h),
        "last_3h":  _window_dict(arts_3h),
        "last_6h":  _window_dict(in_window(6)),
        "last_24h": _window_dict(arts_24h),
    }

    # Narratives
    n_data = {cat: {"articles": [], "scores": []} for cat in NARRATIVE_CATEGORIES}
    for a in arts_main:
        for n in a["narratives"]:
            if n in n_data:
                n_data[n]["articles"].append(a)
                n_data[n]["scores"].append(a["sentiment_raw"])

    total_narrative = sum(len(v["articles"]) for v in n_data.values())
    narratives_out = {}
    for cat, data in n_data.items():
        cnt   = len(data["articles"])
        avg_s = round(sum(data["scores"]) / max(cnt, 1), 3) if data["scores"] else 0.0
        narratives_out[cat] = {
            "score": avg_s, "count": cnt,
            "pct":   round(cnt / max(total_narrative, 1) * 100),
            "dominant": False,
        }
    if narratives_out:
        dom = max(narratives_out, key=lambda k: narratives_out[k]["count"])
        if narratives_out[dom]["count"] > 0:
            narratives_out[dom]["dominant"] = True

    # Top articles
    top_raw = sorted(arts_main, key=lambda a: abs(a["sentiment_decay"]), reverse=True)[:5]
    top_articles = []
    for a in top_raw:
        remaining = max(0.0, a["impact_window_hours"] - a["hours_since"])
        top_articles.append({
            "source":                 a["source"],
            "tier":                   a["tier"],
            "headline":               a["headline"],
            "published_at":           a["timestamp"].strftime("%Y-%m-%dT%H:%M:%SZ"),
            "hours_since":            a["hours_since"],
            "sentiment_raw":          a["sentiment_raw"],
            "sentiment_decay":        a["sentiment_decay"],
            "impact_window_hours":    a["impact_window_hours"],
            "remaining_impact_hours": round(remaining, 1),
            "narratives":             a["narratives"],
            "bullish_signals": [
                {"keyword": s["keyword"], "weight": s["weight"]}
                for s in a["bullish_signals"] if not s["negated"]
            ],
            "bearish_signals": [
                {"keyword": s["keyword"], "weight": s["weight"]}
                for s in a["bearish_signals"] if not s["negated"]
            ],
            "url": a["url"],
        })

    # Impact decay map
    active = [a for a in arts_main if a["hours_since"] < a["impact_window_hours"]]
    rem_hrs = [a["impact_window_hours"] - a["hours_since"] for a in active]
    avg_rem = round(sum(rem_hrs) / max(len(rem_hrs), 1), 1)
    s_1h = by_timewindow["last_1h"]["score"]
    s_3h = by_timewindow["last_3h"]["score"]

    impact_decay_map = {
        "description":          "Horas restantes de impacto por artículo en ventana principal",
        "articles_still_active": len(active),
        "avg_remaining_hours":   avg_rem,
        "peak_impact_passed":    bool(s_1h < s_3h - 0.05),
    }

    # Momentum (last_1h vs 1h-2h ago)
    arts_lag = [a for a in scored
                if (now - timedelta(hours=2)) <= a["timestamp"] < (now - timedelta(hours=1))]
    score_now    = _window_score(arts_1h) if arts_1h else _window_score(arts_3h)
    score_lag    = _window_score(arts_lag)

    if score_now is None or score_lag is None:
        momentum = {"trend": "unknown", "score_1h_ago": score_lag,
                    "score_now": score_now, "delta": None, "acceleration": "unknown"}
    else:
        delta = round(score_now - score_lag, 3)
        momentum = {
            "trend":        "improving"   if delta >  0.02 else ("deteriorating" if delta < -0.02 else "stable"),
            "score_1h_ago": score_lag,
            "score_now":    score_now,
            "delta":        delta,
            "acceleration": "positive"   if delta >  0.02 else ("negative" if delta < -0.02 else "flat"),
        }

    # Signal breakdown
    bull_kw: dict = {}
    bear_kw: dict = {}
    negations = 0

    for a in arts_main:
        for s in a["bullish_signals"]:
            if s["negated"]:
                negations += 1
            else:
                e = bull_kw.setdefault(s["keyword"], {"appearances": 0, "weighted_score": 0.0})
                e["appearances"]    += 1
                e["weighted_score"]  = round(e["weighted_score"] + s["weight"], 2)
        for s in a["bearish_signals"]:
            if s["negated"]:
                negations += 1
            else:
                e = bear_kw.setdefault(s["keyword"], {"appearances": 0, "weighted_score": 0.0})
                e["appearances"]    += 1
                e["weighted_score"]  = round(e["weighted_score"] - s["weight"], 2)

    top_bull = sorted(
        [{"keyword": k, **v} for k, v in bull_kw.items()],
        key=lambda x: x["weighted_score"], reverse=True
    )[:5]
    top_bear = sorted(
        [{"keyword": k, **v} for k, v in bear_kw.items()],
        key=lambda x: x["weighted_score"]
    )[:5]

    return {
        "timestamp":        now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "window_hours":     hours_back,
        "asset":            "BTC",
        "aggregate":        aggregate,
        "by_timewindow":    by_timewindow,
        "narratives":       narratives_out,
        "top_articles":     top_articles,
        "impact_decay_map": impact_decay_map,
        "momentum":         momentum,
        "signal_breakdown": {
            "top_bullish_keywords": top_bull,
            "top_bearish_keywords": top_bear,
            "negations_detected":   negations,
        },
    }


def _empty_context(now: datetime, hours_back: int) -> dict:
    ew = {"score": 0.0, "count": 0, "label": "unknown"}
    return {
        "timestamp": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "window_hours": hours_back, "asset": "BTC",
        "aggregate": {"score": 0.0, "label": "unknown", "confidence": 0.0,
                      "article_count": 0, "tier1_count": 0, "tier2_count": 0, "tier3_count": 0},
        "by_timewindow": {k: ew.copy() for k in ["last_1h", "last_3h", "last_6h", "last_24h"]},
        "narratives":    {c: {"score": 0.0, "count": 0, "pct": 0, "dominant": False} for c in NARRATIVE_CATEGORIES},
        "top_articles": [], "impact_decay_map": {"description": "", "articles_still_active": 0,
                                                  "avg_remaining_hours": 0.0, "peak_impact_passed": False},
        "momentum": {"trend": "unknown", "score_1h_ago": None, "score_now": None,
                     "delta": None, "acceleration": "unknown"},
        "signal_breakdown": {"top_bullish_keywords": [], "top_bearish_keywords": [], "negations_detected": 0},
    }


# ── Formatters ────────────────────────────────────────────────────────────────

def format_news_json(context: dict) -> dict:
    return context


def format_news_narrative(context: dict) -> str:
    agg   = context["aggregate"]
    tw    = context["by_timewindow"]
    narr  = context["narratives"]
    top   = context["top_articles"]
    mom   = context["momentum"]
    sb    = context["signal_breakdown"]
    decay = context["impact_decay_map"]
    ts    = context["timestamp"].replace("T", " ").replace("Z", "")
    h     = context["window_hours"]

    def arrow(s_now, s_prev):
        if s_now is None or s_prev is None: return "→"
        if s_now > s_prev + 0.03: return "▲"
        if s_now < s_prev - 0.03: return "▼"
        return "→"

    a1  = arrow(tw["last_1h"]["score"],  tw["last_3h"]["score"])
    a3  = arrow(tw["last_3h"]["score"],  tw["last_6h"]["score"])
    a6  = arrow(tw["last_6h"]["score"],  tw["last_24h"]["score"])
    a24 = "→"

    # Dominant narrative
    dom_cat  = next((c for c, v in narr.items() if v["dominant"]), None)
    sec_cats = sorted(
        [(c, v) for c, v in narr.items() if not v["dominant"] and v["count"] > 0],
        key=lambda x: x[1]["count"], reverse=True
    )[:2]

    if dom_cat:
        dom_v   = narr[dom_cat]
        dom_line = f"NARRATIVA DOMINANTE: {dom_cat.upper()} ({dom_v['pct']}% cobertura, score {dom_v['score']:+.2f})"
        if sec_cats:
            sec0 = sec_cats[0]
            sec_line = f"  Secundaria: {sec0[0].upper()} ({sec0[1]['score']:+.2f}, {sec0[1]['pct']}%)"
        else:
            sec_line = ""
        counter = next((c for c, v in narr.items() if v["score"] < -0.1 and v["count"] > 0), None)
        counter_line = f"  Contrapeso: {counter.upper()} ({narr[counter]['score']:+.2f}, {narr[counter]['pct']}%)" if counter else ""
    else:
        dom_line = "NARRATIVA DOMINANTE: sin datos suficientes"
        sec_line = counter_line = ""

    # Top signals
    top_lines = []
    for i, a in enumerate(top, 1):
        bull_str = ", ".join(f"{s['keyword']}(×{1})" for s in a["bullish_signals"][:3])
        bear_str = ", ".join(f"{s['keyword']}(×{1})" for s in a["bearish_signals"][:2])
        kw_str   = " | ".join(filter(None, [bull_str, bear_str])) or "sin keywords"
        top_lines.append(
            f"  {i}. [TIER{a['tier']} | {a['source']} | hace {a['hours_since']:.1f}h] {a['headline'][:70]}\n"
            f"     Score: {a['sentiment_raw']:+.2f} | Impacto restante: ~{a['remaining_impact_hours']:.0f}h | {kw_str}"
        )

    # Keyword summary
    bull_kw_str = " | ".join(
        f"{k['keyword']}(×{k['appearances']})" for k in sb["top_bullish_keywords"][:5]
    ) or "(ninguno)"
    bear_kw_str = " | ".join(
        f"{k['keyword']}(×{k['appearances']})" for k in sb["top_bearish_keywords"][:5]
    ) or "(ninguno)"

    # Momentum line
    if mom["trend"] == "unknown":
        mom_line = "  Tendencia: DESCONOCIDA (datos insuficientes)"
    else:
        delta_str = f"{mom['delta']:+.2f}" if mom["delta"] is not None else "N/A"
        mom_line  = f"  Tendencia: {mom['trend'].upper()} ({delta_str} en última hora)"

    # Peak
    peak_str = "YA PASÓ" if decay["peak_impact_passed"] else "AÚN ACTIVO"

    SEP = "═" * 56

    return f"""\
{SEP}
SENTIMIENTO DE NOTICIAS — BTC — {ts} UTC
Ventana de análisis: últimas {h} horas
{SEP}

SCORE AGREGADO: {agg['score']:+.2f}  ({_LABEL_ES.get(agg['label'], agg['label'])}) | Confianza: {agg['confidence']*100:.0f}%
Artículos analizados: {agg['article_count']}  [Tier1: {agg['tier1_count']} | Tier2: {agg['tier2_count']} | Tier3: {agg['tier3_count']}]

EVOLUCIÓN TEMPORAL:
  Última 1h:   {tw['last_1h']['score']:+.2f} {a1} ({_LABEL_ES.get(tw['last_1h']['label'], tw['last_1h']['label'])})
  Últimas 3h:  {tw['last_3h']['score']:+.2f} {a3} ({_LABEL_ES.get(tw['last_3h']['label'], tw['last_3h']['label'])})
  Últimas 6h:  {tw['last_6h']['score']:+.2f} {a6} ({_LABEL_ES.get(tw['last_6h']['label'], tw['last_6h']['label'])})
  Últimas 24h: {tw['last_24h']['score']:+.2f} {a24} ({_LABEL_ES.get(tw['last_24h']['label'], tw['last_24h']['label'])})
{mom_line}

{dom_line}
{sec_line}
{counter_line}

TOP SEÑALES (por impacto ponderado):
{chr(10).join(top_lines) if top_lines else "  (sin artículos en ventana)"}

KEYWORDS MÁS ACTIVOS (últimas {h}h):
  Bullish: {bull_kw_str}
  Bearish: {bear_kw_str}
  Negaciones detectadas: {sb['negations_detected']} (ajustadas en scoring)

DECAY Y VIGENCIA:
  Artículos aún activos (dentro de su ventana de impacto): {decay['articles_still_active']}/{agg['article_count']}
  Impacto promedio restante: {decay['avg_remaining_hours']:.1f} horas
  Pico de impacto: {peak_str}

NOTA METODOLÓGICA:
  Scoring basado en keyword weighting con decay exponencial.
  Negaciones detectadas en ventana de ±3 palabras.
  Tier1 (Reuters, Yahoo, FT): peso 1.0 | Tier2 (CoinDesk, TheBlock): peso 0.65 | Tier3 (blogs): peso 0.30
  Sin NLP — precisión estimada ~65%. Validar con módulos on-chain y macro.
{SEP}"""


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse, json

    ap = argparse.ArgumentParser()
    ap.add_argument('--days',  type=int, default=3,  help='días atrás para scraping RSS')
    ap.add_argument('--hours', type=int, default=None, help='si se especifica: scrape + build context')
    args = ap.parse_args()

    if args.hours is not None:
        # Scrape first, then build context
        result = scrape_rss(days_back=max(1, args.hours // 24 + 1))
        log.info(f"Scraping: {result['total_inserted']} nuevos artículos")
        ctx = build_news_context(hours_back=args.hours)
        print(format_news_narrative(ctx))
        print()
        print(json.dumps(format_news_json(ctx), indent=2, ensure_ascii=False))
    else:
        result = scrape_rss(days_back=args.days)
        print(json.dumps({'total_inserted': result['total_inserted']}, indent=2))
        for src, s in result['stats'].items():
            if s.get('scraped', 0) > 0 or s.get('error'):
                print(f"  {src}: {s}")
