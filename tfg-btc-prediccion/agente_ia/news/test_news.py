"""
test_news.py — Tests para el módulo de sentimiento de noticias

Uso:
  python agente_ia/news/test_news.py
"""

import sys, importlib.util, traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / 'config' / '.env')

OK   = "✅"
FAIL = "❌"
SKIP = "⚠️ "


def _load(filename: str):
    path = Path(__file__).parent / filename
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ns = _load("01_news_scraper.py")


# ── Test 1 — Negaciones ───────────────────────────────────────────────────────

def test_negation():
    print("\n" + "=" * 62)
    print("TEST 1 — Negaciones ('bitcoin will NOT crash' → bullish/neutral)")
    print("=" * 62)
    passed = 0

    cases = [
        ("bitcoin will NOT crash today", "neutral_or_bull"),
        ("no sign of crash in the market",   "neutral_or_bull"),
        ("bitcoin surge and inflow without any crash", "bullish"),
        ("bitcoin crash confirmed",           "bearish"),
    ]

    now = datetime.now(timezone.utc)
    for text, expected in cases:
        res  = ns.score_article(text, "reuters", 1, now)
        raw  = res["sentiment_raw"]
        # For "neutral_or_bull": raw >= -0.1 ; for "bullish": raw > 0 ; "bearish": raw < 0
        if expected == "bearish":
            ok = raw < 0
        elif expected == "bullish":
            ok = raw > 0
        else:
            ok = raw >= -0.15

        status = OK if ok else FAIL
        passed += ok
        print(f"  {status} '{text[:50]}' → raw={raw:+.3f}  (esperado: {expected})")

    print(f"Test 1: {passed}/{len(cases)} pasaron")
    return passed == len(cases)


# ── Test 2 — Tier weighting ───────────────────────────────────────────────────

def test_tier_weighting():
    print("\n" + "=" * 62)
    print("TEST 2 — Tier weighting (Reuters > blog para mismo texto)")
    print("=" * 62)

    text = "Bitcoin ETF approval drives institutional inflow surge rally"
    now  = datetime.now(timezone.utc)

    t1 = ns.score_article(text, "reuters",   1, now)
    t3 = ns.score_article(text, "bitcoinist",3, now)

    ok1 = t1["weight_applied"] > t3["weight_applied"]
    ok2 = abs(t1["sentiment_decay"]) > abs(t3["sentiment_decay"])

    print(f"  {'✅' if ok1 else '❌'} Tier1 weight ({t1['weight_applied']:.4f}) > Tier3 weight ({t3['weight_applied']:.4f})")
    print(f"  {'✅' if ok2 else '❌'} Tier1 |decay| ({abs(t1['sentiment_decay']):.4f}) > Tier3 |decay| ({abs(t3['sentiment_decay']):.4f})")

    passed = ok1 and ok2
    print(f"Test 2: {'2/2' if passed else '0-1/2'} pasaron")
    return passed


# ── Test 3 — Decay temporal ───────────────────────────────────────────────────

def test_temporal_decay():
    print("\n" + "=" * 62)
    print("TEST 3 — Decay temporal (artículo 1h vs 5h → diferente decay)")
    print("=" * 62)

    text = "Bitcoin ETF inflow record BlackRock adoption rally surge"
    now  = datetime.now(timezone.utc)

    r1h = ns.score_article(text, "reuters", 1, now - timedelta(hours=1))
    r5h = ns.score_article(text, "reuters", 1, now - timedelta(hours=5))

    ok1 = r1h["weight_applied"] > r5h["weight_applied"]
    ok2 = abs(r1h["sentiment_decay"]) > abs(r5h["sentiment_decay"])
    ok3 = r1h["hours_since"] < r5h["hours_since"]

    print(f"  {'✅' if ok1 else '❌'} weight_1h ({r1h['weight_applied']:.4f}) > weight_5h ({r5h['weight_applied']:.4f})")
    print(f"  {'✅' if ok2 else '❌'} |decay_1h| ({abs(r1h['sentiment_decay']):.4f}) > |decay_5h| ({abs(r5h['sentiment_decay']):.4f})")
    print(f"  {'✅' if ok3 else '❌'} hours_since: 1h={r1h['hours_since']:.1f}  5h={r5h['hours_since']:.1f}")

    passed = ok1 and ok2 and ok3
    print(f"Test 3: {sum([ok1, ok2, ok3])}/3 pasaron")
    return passed


# ── Test 4 — Detección de narrativa ──────────────────────────────────────────

def test_narrative_detection():
    print("\n" + "=" * 62)
    print("TEST 4 — Narrative detection (BlackRock ETF → 'institutional')")
    print("=" * 62)
    passed = 0

    cases = [
        ("BlackRock ETF institutional fund inflow grayscale",  "institutional"),
        ("Fed CPI inflation rate FOMC recession yield",        "macro"),
        ("SEC regulation ban congress compliance lawsuit",     "regulatory"),
        ("halving mining blockchain whale transaction",        "onchain"),
        ("war conflict china russia geopolitical election",    "geopolitical"),
    ]
    now = datetime.now(timezone.utc)
    for text, expected_cat in cases:
        res  = ns.score_article(text, "coindesk", 2, now)
        narr = res["narratives"]
        ok   = expected_cat in narr
        passed += ok
        print(f"  {'✅' if ok else '❌'} '{text[:45]}' → narratives={narr}  (esperado: {expected_cat})")

    print(f"Test 4: {passed}/{len(cases)} pasaron")
    return passed == len(cases)


# ── Test 5 — Impact window ────────────────────────────────────────────────────

def test_impact_window():
    print("\n" + "=" * 62)
    print("TEST 5 — Impact window (etf approval → 72h, hack → 24h)")
    print("=" * 62)
    passed = 0

    cases = [
        ("SEC approved bitcoin spot etf approval for BlackRock", 72, "etf approval"),
        ("Exchange hack exploit breach funds stolen",           24, "hack"),
        ("Bitcoin all-time high ath breakout",                  24, "all-time high"),
        ("Fed rate cut dovish pivot expected",                  48, "rate cut"),
    ]
    now = datetime.now(timezone.utc)
    for text, expected_h, label in cases:
        res = ns.score_article(text, "reuters", 1, now)
        ok  = res["impact_window_hours"] == expected_h
        passed += ok
        print(f"  {'✅' if ok else '❌'} [{label}] impact={res['impact_window_hours']}h  (esperado: {expected_h}h)")

    print(f"Test 5: {passed}/{len(cases)} pasaron")
    return passed == len(cases)


# ── Test 6 — RSS fetch real ───────────────────────────────────────────────────

def test_rss_fetch():
    print("\n" + "=" * 62)
    print("TEST 6 — RSS fetch real (CoinDesk → ≥5 artículos)")
    print("=" * 62)

    try:
        rows = ns._fetch_rss("coindesk", "https://www.coindesk.com/arc/outboundfeeds/rss/")
        ok   = len(rows) >= 5
        print(f"  {'✅' if ok else '❌'} CoinDesk: {len(rows)} artículos (necesitamos ≥5)")
        if rows:
            print(f"  Muestra: '{rows[0]['text'][:80]}...'")
        print(f"Test 6: {'1/1' if ok else '0/1'} pasaron")
        return ok
    except Exception as e:
        print(f"  {FAIL} Error: {e}")
        print("Test 6: 0/1 pasaron")
        return False


# ── Test 7 — Score agregado ───────────────────────────────────────────────────

def test_build_context():
    print("\n" + "=" * 62)
    print("TEST 7 — build_news_context() devuelve dict con todas las claves")
    print("=" * 62)

    try:
        from db.db_utils import get_engine
        engine = get_engine()
        ctx    = ns.build_news_context(hours_back=6, engine=engine)

        required_keys = [
            "timestamp", "window_hours", "asset", "aggregate",
            "by_timewindow", "narratives", "top_articles",
            "impact_decay_map", "momentum", "signal_breakdown",
        ]
        aggregate_keys = ["score", "label", "confidence", "article_count",
                          "tier1_count", "tier2_count", "tier3_count"]
        timewindow_keys = ["last_1h", "last_3h", "last_6h", "last_24h"]
        narrative_keys  = ["institutional", "macro", "regulatory",
                           "technical", "onchain", "geopolitical"]

        passed = 0
        total  = 0

        for k in required_keys:
            ok = k in ctx
            passed += ok; total += 1
            print(f"  {'✅' if ok else '❌'} clave raíz '{k}' presente")

        for k in aggregate_keys:
            ok = k in ctx.get("aggregate", {})
            passed += ok; total += 1
            if not ok:
                print(f"  {FAIL} 'aggregate.{k}' falta")

        for k in timewindow_keys:
            ok = k in ctx.get("by_timewindow", {})
            passed += ok; total += 1
            if not ok:
                print(f"  {FAIL} 'by_timewindow.{k}' falta")

        for k in narrative_keys:
            ok = k in ctx.get("narratives", {})
            passed += ok; total += 1
            if not ok:
                print(f"  {FAIL} 'narratives.{k}' falta")

        print(f"\n  aggregate.score      = {ctx['aggregate']['score']}")
        print(f"  aggregate.article_count = {ctx['aggregate']['article_count']}")
        print(f"Test 7: {passed}/{total} claves ok")
        return passed == total

    except Exception as e:
        print(f"  {FAIL} Excepción: {e}")
        traceback.print_exc()
        print("Test 7: 0 pasaron")
        return False


# ── Test 8 — Output doble ─────────────────────────────────────────────────────

def test_output_formats():
    print("\n" + "=" * 62)
    print("TEST 8 — Output doble (narrative string + JSON dict)")
    print("=" * 62)

    try:
        from db.db_utils import get_engine
        import json

        engine = get_engine()
        ctx    = ns.build_news_context(hours_back=6, engine=engine)

        narrative = ns.format_news_narrative(ctx)
        json_out  = ns.format_news_json(ctx)

        ok1 = isinstance(narrative, str) and "═" in narrative
        ok2 = isinstance(json_out, dict) and "aggregate" in json_out

        # Must be JSON-serializable
        try:
            json.dumps(json_out, ensure_ascii=False)
            ok3 = True
        except Exception:
            ok3 = False

        ok4 = "SENTIMIENTO DE NOTICIAS" in narrative
        ok5 = "signal_breakdown" in json_out

        print(f"  {'✅' if ok1 else '❌'} narrative es str con '═' (len={len(narrative)})")
        print(f"  {'✅' if ok2 else '❌'} json_out es dict con 'aggregate'")
        print(f"  {'✅' if ok3 else '❌'} json_out es JSON-serializable")
        print(f"  {'✅' if ok4 else '❌'} narrative contiene 'SENTIMIENTO DE NOTICIAS'")
        print(f"  {'✅' if ok5 else '❌'} json_out contiene 'signal_breakdown'")

        passed = sum([ok1, ok2, ok3, ok4, ok5])
        print(f"Test 8: {passed}/5 pasaron")
        return passed == 5

    except Exception as e:
        print(f"  {FAIL} Excepción: {e}")
        traceback.print_exc()
        print("Test 8: 0 pasaron")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = {
        "T1 Negaciones":   test_negation(),
        "T2 Tier weights": test_tier_weighting(),
        "T3 Decay temp.":  test_temporal_decay(),
        "T4 Narrativas":   test_narrative_detection(),
        "T5 ImpactWindow": test_impact_window(),
        "T6 RSS fetch":    test_rss_fetch(),
        "T7 Context dict": test_build_context(),
        "T8 Output doble": test_output_formats(),
    }

    print("\n" + "=" * 62)
    print("RESUMEN FINAL")
    print("=" * 62)
    for name, ok in results.items():
        print(f"  {'✅' if ok else '❌'} {name}")
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{len(results)}")
