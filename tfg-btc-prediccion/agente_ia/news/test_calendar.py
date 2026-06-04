"""
test_calendar.py — Tests de la lógica de bias determinístico y scraping

Uso:
  python agente_ia/news/test_calendar.py
"""

import sys, importlib.util
from pathlib import Path

# Path a la raíz del proyecto
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / 'config' / '.env')

# Importar 02_calendar_scraper.py por path (nombre con dígito = no importable directamente)
_spec = importlib.util.spec_from_file_location(
    "calendar_scraper",
    Path(__file__).parent / "02_calendar_scraper.py",
)
cal = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cal)

_calculate_surprise  = cal._calculate_surprise
_calculate_btc_bias  = cal._calculate_btc_bias
scrape_forexfactory  = cal.scrape_forexfactory
get_upcoming_bias    = cal.get_upcoming_bias

PASS = "✅"
FAIL = "❌"


# ── Test 1 ────────────────────────────────────────────────────────────────────

def run_bias_tests():
    """Verifica la lógica de bias determinístico sin base de datos."""
    print("\n" + "=" * 62)
    print("TEST 1 — Bias determinístico (sin base de datos)")
    print("=" * 62)

    # (event_name, impact, actual, forecast, expected_bias, expected_dir)
    test_cases = [
        ("CPI m/m",            "high",   0.1,   0.3,  +1, "miss"),
        ("Core CPI m/m",       "high",   0.4,   0.2,  -1, "beat"),
        ("Non-Farm Payrolls",  "high", 150.0, 200.0,  +1, "miss"),
        ("Non-Farm Payrolls",  "high", 300.0, 200.0,  -1, "beat"),
        ("Unemployment Rate",  "high",   4.2,   4.0,  +1, "beat"),
        ("Federal Funds Rate", "high",   5.5,  5.75,  +1, "miss"),
        ("GDP q/q",            "high",   2.5,   2.0, None, "beat"),  # señal mixta
        ("Random Event XYZ",   "low",    1.0,   1.0,   0, "inline"),
    ]

    passed = 0
    for event_name, impact, actual, forecast, exp_bias, exp_dir in test_cases:
        surprise, surprise_pct, surprise_dir = _calculate_surprise(actual, forecast)
        bias, reason = _calculate_btc_bias(event_name, impact, surprise_dir, actual, forecast)

        dir_ok  = (surprise_dir == exp_dir)
        bias_ok = (exp_bias is None) or (bias == exp_bias)
        ok = dir_ok and bias_ok
        if ok:
            passed += 1

        icon = PASS if ok else FAIL
        exp_bias_str = str(exp_bias) if exp_bias is not None else "cualquiera"
        print(f"\n{icon} {event_name}")
        print(f"   actual={actual}, forecast={forecast}")
        print(f"   surprise_dir: {str(surprise_dir):8s}  esperado: {exp_dir}")
        print(f"   btc_bias:     {str(bias):6s}  esperado: {exp_bias_str}")
        if not ok:
            print(f"   ⚠  {reason}")

    print(f"\n{'=' * 62}")
    print(f"Test 1: {passed}/{len(test_cases)} pasaron")
    return passed, len(test_cases)


# ── Test 2 ────────────────────────────────────────────────────────────────────

def run_forexfactory_test():
    """Scraping real de ForexFactory sin API key."""
    print("\n" + "=" * 62)
    print("TEST 2 — Scraping real ForexFactory (days_ahead=3)")
    print("=" * 62)

    try:
        events = scrape_forexfactory(days_ahead=3)
        if events:
            print(f"{PASS} OK: {len(events)} eventos obtenidos")
            print("\nPrimeros 5 eventos:")
            for ev in events[:5]:
                bias_str = f"bias={ev['btc_bias']}" if ev.get('btc_bias') is not None else "bias=?"
                print(
                    f"  [{ev['event_datetime'].strftime('%Y-%m-%d %H:%M')} UTC] "
                    f"{ev['currency']} | {ev['impact']:6s} | "
                    f"{ev['event_name'][:38]:38s} | {bias_str}"
                )
            return True
        else:
            print(f"{FAIL} Sin eventos — posible bloqueo o sin datos en el rango")
            print("   Sugerencias:")
            print("   - ForexFactory bloquea requests directos con frecuencia")
            print("   - Alternativas: Selenium/Playwright, API de TradingEconomics (TE_API_KEY)")
            return False
    except Exception as e:
        print(f"{FAIL} Error: {e}")
        print("   Sugerencias:")
        print("   - Verifica conectividad de red")
        print("   - Usa TE_API_KEY en config/.env para TradingEconomics")
        return False


# ── Test 3 ────────────────────────────────────────────────────────────────────

def run_db_test():
    """Consulta get_upcoming_bias() contra la BD."""
    print("\n" + "=" * 62)
    print("TEST 3 — get_upcoming_bias() con base de datos")
    print("=" * 62)

    try:
        from db.db_utils import get_engine
        engine = get_engine()
        upcoming = get_upcoming_bias(hours_ahead=48, engine=engine)
        print(f"{PASS} BD conectada. Próximos eventos high/medium (48h): {len(upcoming)}")
        for ev in upcoming[:5]:
            print(
                f"  [{ev['datetime'][:16]}] {ev['event'][:35]:35s} "
                f"bias={ev['btc_bias']}  ({ev['hours_away']}h)"
            )
        if not upcoming:
            print("   (tabla economic_calendar vacía — ejecuta 02_calendar_scraper.py primero)")
        return True
    except Exception as e:
        print(f"{FAIL} {e}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    passed, total = run_bias_tests()
    ff_ok = run_forexfactory_test()
    db_ok = run_db_test()

    print("\n" + "=" * 62)
    print("RESUMEN FINAL")
    print("=" * 62)
    print(f"  Test 1 (bias lógica):   {passed}/{total}  {PASS if passed == total else FAIL}")
    print(f"  Test 2 (ForexFactory):  {PASS if ff_ok else FAIL + ' (scraping bloqueado o sin datos)'}")
    print(f"  Test 3 (DB upcoming):   {PASS if db_ok else FAIL}")
