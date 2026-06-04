"""
test_pipeline.py — Tests del pipeline completo de bias BTC

Uso:
  python agente_ia/news/test_pipeline.py
"""

import sys, importlib.util, os
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
_HERE = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / 'config' / '.env')

PASS = "✅"
FAIL = "❌"
SKIP = "⏭️ "


def _load(filename: str):
    path = _HERE / filename
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Test 1 — On-chain signals (sin DB, sin red) ───────────────────────────────

def test_onchain_signals():
    print("\n" + "=" * 62)
    print("TEST 1 — Señales on-chain determinísticas (sin DB, sin red)")
    print("=" * 62)

    m = _load("03_onchain_scraper.py")
    cases = [
        ("exchange_netflow", -5000,  [], "bullish"),
        ("exchange_netflow",  8000,  [], "bearish"),
        ("exchange_netflow",  1000,  [], "neutral"),
        ("mvrv_ratio",         0.8,  [], "bullish"),
        ("mvrv_ratio",         4.0,  [], "bearish"),
        ("mvrv_ratio",         2.0,  [], "neutral"),
        ("sopr",              0.95,  [], "bullish"),
        ("sopr",              1.08,  [], "bearish"),
    ]

    passed = 0
    for metric, value, history, expected in cases:
        signal, strength = m._apply_signal_rule(metric, value, history)
        ok = (signal == expected)
        if ok:
            passed += 1
        icon = PASS if ok else FAIL
        print(f"  {icon} {metric}={value} → {signal} (strength={strength:.2f})  esperado: {expected}")

    print(f"\nTest 1: {passed}/{len(cases)} pasaron")
    return passed, len(cases)


# ── Test 2 — Micro signals (sin DB, sin red) ──────────────────────────────────

def test_micro_signals():
    print("\n" + "=" * 62)
    print("TEST 2 — Señales microestructura (sin DB, sin red)")
    print("=" * 62)

    m = _load("04_micro_scraper.py")
    cases_funding = [
        (0.0003,  "bearish"),   # 33% anual
        (0.0001,  "bearish"),   # 11% anual (> 15% threshold?  No: 0.0001*3*365*100=10.9%)
        (-0.0002, "bullish"),   # negativo
        (0.00005, "neutral"),   # ~5% anual
    ]
    # Recalcular: 0.0001 * 3 * 365 * 100 = 10.95% < 15 → neutral
    cases_funding[1] = (0.0001, "neutral")

    passed = 0
    total  = 0
    for rate, expected in cases_funding:
        sig, strength = m.funding_signal(rate)
        ok = (sig == expected)
        if ok:
            passed += 1
        total += 1
        annual = rate * 3 * 365 * 100
        icon = PASS if ok else FAIL
        print(f"  {icon} funding_rate={rate} ({annual:.1f}%/año) → {sig}  esperado: {expected}")

    cases_ls = [
        (2.0, "bearish"),
        (1.5, "bearish"),
        (0.6, "bullish"),
        (0.8, "bullish"),
        (1.1, "neutral"),
    ]
    print()
    for ratio, expected in cases_ls:
        sig, strength = m.long_short_signal(ratio)
        ok = (sig == expected)
        if ok:
            passed += 1
        total += 1
        icon = PASS if ok else FAIL
        print(f"  {icon} long_short={ratio} → {sig}  esperado: {expected}")

    cases_fg = [
        (15,  "bullish"),
        (30,  "bullish"),
        (82,  "bearish"),
        (70,  "bearish"),
        (50,  "neutral"),
    ]
    print()
    for value, expected in cases_fg:
        sig, strength = m.fear_greed_signal(value)
        ok = (sig == expected)
        if ok:
            passed += 1
        total += 1
        icon = PASS if ok else FAIL
        print(f"  {icon} fear_greed={value} → {sig}  esperado: {expected}")

    print(f"\nTest 2: {passed}/{total} pasaron")
    return passed, total


# ── Test 3 — Score de noticias con keywords ───────────────────────────────────

def test_news_keywords():
    print("\n" + "=" * 62)
    print("TEST 3 — Score de noticias con keywords (sin DB)")
    print("=" * 62)

    from datetime import datetime, timezone
    m_news = _load("01_news_scraper.py")
    now    = datetime.now(timezone.utc)

    def score_text(text: str) -> float:
        return m_news.score_article(text, "reuters", 1, now)["sentiment_raw"]

    cases = [
        ("Bitcoin surge rally adoption ETF approval institutional inflow record",   "positivo", lambda s: s > 0),
        ("Bitcoin crash ban lawsuit fraud collapse capitulation warning correction", "negativo", lambda s: s < 0),
        ("Bitcoin traded today with mixed signals in global markets",                "neutro",   lambda s: abs(s) < 0.5),
        ("ETF inflow record high adoption surge",                                    "positivo", lambda s: s > 0.5),
    ]

    passed = 0
    for text, expected_label, check in cases:
        s = score_text(text)
        ok = check(s)
        if ok:
            passed += 1
        icon = PASS if ok else FAIL
        print(f"  {icon} [{expected_label:8}] score={s:+.2f}  \"{text[:55]}...\"")

    print(f"\nTest 3: {passed}/{len(cases)} pasaron")
    return passed, len(cases)


# ── Test 4 — Binance API pública (requiere red) ───────────────────────────────

def test_binance_api():
    print("\n" + "=" * 62)
    print("TEST 4 — Binance Futures API pública (requiere red)")
    print("=" * 62)

    m = _load("04_micro_scraper.py")
    passed = 0
    total  = 0

    # Funding rate
    data = m.fetch_funding_rate()
    total += 1
    if data and "funding_rate" in data:
        passed += 1
        rate = data["funding_rate"]
        annual = rate * 3 * 365 * 100
        print(f"  {PASS} Funding rate: {rate:.6f}  ({annual:.2f}%/año)  signal={data['signal']}")
    else:
        print(f"  {FAIL} Funding rate: sin datos ({data})")

    # Open interest
    oi = m.fetch_open_interest()
    total += 1
    if oi and "oi_usd" in oi:
        passed += 1
        print(f"  {PASS} Open Interest: ${oi['oi_usd']:,.0f}  change24h={oi.get('oi_change_24h', 'N/A')}")
    else:
        print(f"  {FAIL} Open Interest: sin datos ({oi})")

    # BTC price
    price = m.fetch_btc_price()
    total += 1
    if price and price > 0:
        passed += 1
        print(f"  {PASS} BTC price: ${price:,.2f}")
    else:
        print(f"  {FAIL} BTC price: sin datos ({price})")

    # Fear & Greed
    fg = m.fetch_fear_greed()
    total += 1
    if fg and "value" in fg:
        passed += 1
        print(f"  {PASS} Fear & Greed: {fg['value']}/100 — {fg['label']}  signal={fg['signal']}")
    else:
        print(f"  {FAIL} Fear & Greed: sin datos ({fg})")

    print(f"\nTest 4: {passed}/{total} pasaron")
    return passed, total


# ── Test 5 — CryptoQuant API (requiere key) ───────────────────────────────────

def test_cryptoquant_api():
    print("\n" + "=" * 62)
    print("TEST 5 — CryptoQuant API on-chain (requiere CRYPTOQUANT_API_KEY)")
    print("=" * 62)

    api_key = os.getenv("CRYPTOQUANT_API_KEY", "").strip()
    if not api_key:
        print(f"  {SKIP} CRYPTOQUANT_API_KEY no configurada — test omitido")
        print("  Para habilitarlo: añade CRYPTOQUANT_API_KEY=tu_key en config/.env")
        return None, None

    m = _load("03_onchain_scraper.py")
    passed = 0
    total  = 2

    # Exchange netflow
    data = m._cq_get("/btc/exchange-flows/netflow", {"exchange": "all", "window": "day", "limit": 7})
    if data and len(data) > 0:
        latest = data[-1]
        value  = latest.get("value", "N/A")
        print(f"  {PASS} Exchange netflow: {value} BTC/día")
        passed += 1
    else:
        print(f"  {FAIL} Exchange netflow: sin datos (key inválida o rate limit)")

    # MVRV
    data = m._cq_get("/btc/market-indicator/mvrv-ratio", {"limit": 7})
    if data and len(data) > 0:
        latest = data[-1]
        value  = latest.get("value", "N/A")
        print(f"  {PASS} MVRV ratio: {value}")
        passed += 1
    else:
        print(f"  {FAIL} MVRV ratio: sin datos")

    print(f"\nTest 5: {passed}/{total} pasaron")
    return passed, total


# ── Test 6 — Farside scraping (requiere red) ──────────────────────────────────

def test_farside_scraping():
    print("\n" + "=" * 62)
    print("TEST 6 — Farside ETF scraping (requiere red)")
    print("=" * 62)

    m = _load("05_etf_scraper.py")
    records = m.scrape_farside(days_back=10)

    total  = 2
    passed = 0

    if records:
        passed += 1
        print(f"  {PASS} Scraping OK: {len(records)} registros parseados")

        # Agrupar por fecha
        by_date: dict = {}
        for r in records:
            d = str(r["flow_date"])
            if d not in by_date:
                by_date[d] = {}
            by_date[d][r["vehicle"]] = r["flow_usd"]

        sorted_dates = sorted(by_date.keys(), reverse=True)[:3]
        print(f"\n  Últimos {len(sorted_dates)} días con datos:")
        for d in sorted_dates:
            flows = by_date[d]
            total_flow = sum(flows.values())
            vehicles = ", ".join(f"{k}:{v:+.0f}M" for k, v in list(flows.items())[:4])
            print(f"    {d}: total={total_flow:+.0f}M  |  {vehicles}")
    else:
        print(f"  {FAIL} Scraping sin datos — Farside puede estar bloqueando")
        print("  Sugerencia: verificar conectividad o usar un proxy")

    # Test función parse
    test_parses = [
        ("340.1",   340.1),
        ("$340.1",  340.1),
        ("(50.1)",  -50.1),
        ("-",       None),
        ("",        None),
        ("1,234.5", 1234.5),
    ]
    parse_ok = all(m._parse_flow(raw) == expected for raw, expected in test_parses)
    if parse_ok:
        passed += 1
        print(f"  {PASS} _parse_flow: todos los casos correctos")
    else:
        for raw, expected in test_parses:
            got = m._parse_flow(raw)
            if got != expected:
                print(f"  {FAIL} _parse_flow({raw!r}) = {got}, esperado {expected}")

    print(f"\nTest 6: {passed}/{total} pasaron")
    return passed, total


# ── Test 7 — Pipeline completo ────────────────────────────────────────────────

def test_full_pipeline():
    print("\n" + "=" * 62)
    print("TEST 7 — Pipeline completo")
    print("=" * 62)

    m = _load("run_pipeline.py")
    total = 2
    passed = 0

    print("  Ejecutando run_full_pipeline()...")
    try:
        context = m.run_full_pipeline(days_back=1)
        if context and "BIAS SNAPSHOT" in context:
            passed += 1
            print(f"  {PASS} Pipeline ejecutado — contexto LLM generado ({len(context)} chars)")
        else:
            print(f"  {FAIL} Pipeline ejecutado pero output incorrecto: {context[:100]!r}")
    except Exception as e:
        print(f"  {FAIL} Pipeline error: {e}")

    print("\n" + "-" * 62)
    print("OUTPUT COMPLETO DEL CONTEXTO LLM:")
    print("-" * 62)
    try:
        context2 = m.get_llm_context(max_age_hours=1)
        print(context2)
        if context2 and "BIAS SNAPSHOT" in context2:
            passed += 1
    except Exception as e:
        print(f"Error get_llm_context: {e}")

    print(f"\nTest 7: {passed}/{total} pasaron")
    return passed, total


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = []

    r1 = test_onchain_signals()
    results.append(("T1 Onchain signals",   r1))

    r2 = test_micro_signals()
    results.append(("T2 Micro signals",     r2))

    r3 = test_news_keywords()
    results.append(("T3 News keywords",     r3))

    r4 = test_binance_api()
    results.append(("T4 Binance API",       r4))

    r5 = test_cryptoquant_api()
    if r5[0] is None:
        results.append(("T5 CryptoQuant",   (0, 0)))
    else:
        results.append(("T5 CryptoQuant",   r5))

    r6 = test_farside_scraping()
    results.append(("T6 Farside scraping",  r6))

    r7 = test_full_pipeline()
    results.append(("T7 Full pipeline",     r7))

    print("\n" + "=" * 62)
    print("RESUMEN FINAL")
    print("=" * 62)
    total_passed = total_total = 0
    for name, (p, t) in results:
        if t == 0:
            icon = SKIP
            label = "omitido"
        else:
            icon = PASS if p == t else (FAIL if p == 0 else "⚠️ ")
            label = f"{p}/{t}"
        total_passed += p or 0
        total_total  += t or 0
        print(f"  {icon} {name:<25} {label}")

    print(f"\nTotal: {total_passed}/{total_total}")
