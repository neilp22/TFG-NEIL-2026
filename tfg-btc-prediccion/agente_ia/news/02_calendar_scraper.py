"""
02_calendar_scraper.py — Scraper de calendario económico con bias BTC determinístico

Fuentes (en orden de preferencia):
  1. investing.com  — scraping HTML (sin API pública)
  2. forexfactory   — scraping HTML
  3. tradingeconomics — API REST (requiere TE_API_KEY en .env)

Uso:
  python agente_ia/02_calendar_scraper.py              # Próximos 7 días
  python agente_ia/02_calendar_scraper.py --days 14    # Próximas 2 semanas
  python agente_ia/02_calendar_scraper.py --update-actuals  # Rellenar datos reales publicados
"""

import os, sys, logging, time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / 'config' / '.env')

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import requests
from bs4 import BeautifulSoup
from dateutil import parser as dateparser
from sqlalchemy import text
from db.db_utils import get_engine

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# ── Configuración ────────────────────────────────────────────────────────────

# Eventos de alto impacto en BTC con sus reglas de bias
# Formato: event_keyword → {bullish_if, bearish_if, neutral_threshold, description}
BTC_BIAS_RULES: dict[str, dict] = {
    # FED / Política monetaria
    "federal funds rate": {
        "bullish_if":  "miss",    # Fed sube menos de lo esperado (dovish)
        "bearish_if":  "beat",    # Fed sube más de lo esperado (hawkish)
        "description": "Fed rate: hike > expected = hawkish = bearish BTC",
    },
    "fomc": {
        "bullish_if":  "miss",
        "bearish_if":  "beat",
        "description": "FOMC decision",
    },
    "interest rate": {
        "bullish_if":  "miss",
        "bearish_if":  "beat",
        "description": "Interest rate: cut > expected = bullish BTC",
    },
    # Inflación
    "cpi": {
        "bullish_if":  "miss",    # Inflación menor de lo esperado → Fed menos agresivo
        "bearish_if":  "beat",
        "description": "CPI: below forecast = pivot narrative = bullish BTC",
    },
    "core cpi": {
        "bullish_if":  "miss",
        "bearish_if":  "beat",
        "description": "Core CPI: same logic as headline CPI",
    },
    "pce": {
        "bullish_if":  "miss",
        "bearish_if":  "beat",
        "description": "PCE: Fed's preferred inflation gauge",
    },
    "core pce": {
        "bullish_if":  "miss",
        "bearish_if":  "beat",
        "description": "Core PCE: most watched by Fed",
    },
    "ppi": {
        "bullish_if":  "miss",
        "bearish_if":  "beat",
        "description": "PPI: producer prices, leading indicator for CPI",
    },
    # Empleo
    "non-farm payrolls": {
        "bullish_if":  "miss",    # Empleo débil → Fed dovish
        "bearish_if":  "beat",
        "description": "NFP: weak jobs = dovish Fed = bullish BTC",
    },
    "nfp": {
        "bullish_if":  "miss",
        "bearish_if":  "beat",
        "description": "NFP shorthand",
    },
    "unemployment rate": {
        "bullish_if":  "beat",    # Tasa de desempleo SUBE = bearish economía = bullish BTC
        "bearish_if":  "miss",    # Nota: aquí beat/miss están invertidos (mayor = mejor para BTC)
        "inverted":    True,
        "description": "Unemployment: higher rate = weaker economy = dovish Fed = bullish BTC",
    },
    "initial jobless claims": {
        "bullish_if":  "beat",    # Más solicitudes = empleo débil = dovish
        "bearish_if":  "miss",
        "inverted":    True,
        "description": "Jobless claims: higher = weaker labor = dovish",
    },
    # Crecimiento
    "gdp": {
        "bullish_if":  "beat",    # GDP fuerte → risk-on, pero también hawkish. Efecto mixto.
        "bearish_if":  "miss",
        "confidence":  0.5,       # Reducir peso — señal ambigua
        "description": "GDP: mixed signal for BTC (risk-on vs hawkish)",
    },
    # Actividad manufacturera / servicios
    "ism manufacturing": {
        "bullish_if":  "beat",
        "bearish_if":  "miss",
        "confidence":  0.4,
        "description": "ISM Mfg: >50 = expansion, bullish risk assets",
    },
    "ism services": {
        "bullish_if":  "beat",
        "bearish_if":  "miss",
        "confidence":  0.4,
        "description": "ISM Services",
    },
    # Confianza
    "consumer confidence": {
        "bullish_if":  "beat",
        "bearish_if":  "miss",
        "confidence":  0.3,
        "description": "Consumer confidence: higher = risk-on",
    },
    "michigan": {
        "bullish_if":  "beat",
        "bearish_if":  "miss",
        "confidence":  0.3,
        "description": "UMich sentiment",
    },
    # Vivienda (baja correlación con BTC)
    "housing starts": {
        "bullish_if":  "beat",
        "bearish_if":  "miss",
        "confidence":  0.2,
        "description": "Housing: weak BTC correlation",
    },
}

# Multiplicadores de impacto por nivel
IMPACT_MULTIPLIERS = {
    "high":   1.0,
    "medium": 0.5,
    "low":    0.15,
}

# Headers anti-bot
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

NEUTRAL_THRESHOLD = 0.03  # surprise < 3% del forecast = inline (25bp en tipos de interés ya es relevante)


# ── Cálculo de bias ──────────────────────────────────────────────────────────

def _find_bias_rule(event_name: str) -> Optional[dict]:
    """Busca la regla de bias para un nombre de evento (case-insensitive, partial match)."""
    name_lower = event_name.lower()
    # Match exacto primero
    for keyword, rule in BTC_BIAS_RULES.items():
        if keyword in name_lower:
            return {"keyword": keyword, **rule}
    return None


def _calculate_surprise(actual: Optional[float], forecast: Optional[float]) -> tuple:
    """
    Calcula surprise y surprise_dir.
    Devuelve (surprise, surprise_pct, surprise_dir).
    """
    if actual is None or forecast is None:
        return None, None, None

    surprise = actual - forecast

    if forecast != 0:
        surprise_pct = (surprise / abs(forecast)) * 100
    else:
        surprise_pct = 0.0 if surprise == 0 else (100.0 if surprise > 0 else -100.0)

    # Determinar dirección
    if abs(surprise_pct) < NEUTRAL_THRESHOLD * 100:
        surprise_dir = "inline"
    elif surprise > 0:
        surprise_dir = "beat"
    else:
        surprise_dir = "miss"

    return surprise, surprise_pct, surprise_dir


def _calculate_btc_bias(
    event_name: str,
    impact: str,
    surprise_dir: Optional[str],
    actual: Optional[float] = None,
    forecast: Optional[float] = None,
) -> tuple[Optional[int], Optional[str]]:
    """
    Calcula el bias BTC determinístico.
    Devuelve (btc_bias: -1|0|+1|None, bias_reason: str).
    """
    if surprise_dir is None or surprise_dir == "inline":
        rule = _find_bias_rule(event_name)
        if surprise_dir == "inline":
            return 0, f"Inline with forecast — neutral bias. Rule: {rule['description'] if rule else 'no rule'}"
        # Evento futuro sin datos aún: señal de anticipación pre-evento
        if rule and impact == "high":
            # Evento macro de alto impacto próximo → mercado suele reducir riesgo (risk-off BTC)
            return -1, f"Pre-event risk-off — {rule['description']}"
        if rule and impact == "medium":
            return 0, f"Pending — {rule['description']}"
        return 0, "Pending actual data — no BTC bias rule"

    rule = _find_bias_rule(event_name)
    if not rule:
        return 0, f"No BTC bias rule for: {event_name}"

    confidence = rule.get("confidence", 0.8)
    impact_mult = IMPACT_MULTIPLIERS.get(impact, 0.5)

    # El bias_score final (para bias_snapshots) usará: bias × confidence × impact_mult
    # Aquí solo retornamos el signo
    if surprise_dir == rule["bullish_if"]:
        bias = +1
        direction_label = "bullish"
    elif surprise_dir == rule["bearish_if"]:
        bias = -1
        direction_label = "bearish"
    else:
        bias = 0
        direction_label = "neutral"

    reason = (
        f"{rule['description']} | "
        f"Surprise: {surprise_dir} | Impact: {impact} | "
        f"Confidence: {confidence:.0%} | Impact multiplier: {impact_mult}"
    )

    return bias, reason


# ── Scraper investing.com ────────────────────────────────────────────────────

def _parse_float(s: str) -> Optional[float]:
    """Parsea strings como '3.2%', '245K', '1.2T' a float."""
    if not s or s.strip() in ('', '-', '—', 'N/A'):
        return None
    s = s.strip().replace(',', '').replace('%', '').replace(' ', '')
    multiplier = 1.0
    if s.endswith('K'):
        multiplier = 1_000
        s = s[:-1]
    elif s.endswith('M'):
        multiplier = 1_000_000
        s = s[:-1]
    elif s.endswith('B'):
        multiplier = 1_000_000_000
        s = s[:-1]
    elif s.endswith('T'):
        multiplier = 1_000_000_000_000
        s = s[:-1]
    try:
        return float(s) * multiplier
    except ValueError:
        return None


def scrape_investing_com(days_ahead: int = 7) -> list[dict]:
    """
    Scrape investing.com economic calendar.
    Nota: investing.com usa JS pesado. Si falla, cae a ForexFactory.
    """
    session = requests.Session()
    session.headers.update(HEADERS)

    # investing.com requiere cookies y tokens CSRF — usamos su endpoint JSON interno
    url = "https://www.investing.com/economic-calendar/Service/getCalendarFilteredData"

    now_utc = datetime.now(timezone.utc)
    date_from = now_utc.strftime("%Y-%m-%d")
    date_to   = (now_utc + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    payload = {
        "country[]":    ["5"],  # 5 = USA (más relevante para BTC)
        "importance[]": ["3", "2", "1"],  # High, Medium, Low
        "timeZone":     "55",  # UTC
        "timeFilter":   "timeRemain",
        "currentTab":   "custom",
        "submitFilters": "1",
        "limit_from":   "0",
        "dateFrom":     date_from,
        "dateTo":       date_to,
    }

    try:
        resp = session.post(url, data=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        html = data.get("data", "")
        if not html:
            raise ValueError("Empty response from investing.com")

        return _parse_investing_html(html, source="investing_com")

    except Exception as e:
        log.warning(f"investing.com falló: {e} — intentando ForexFactory")
        return scrape_forexfactory(days_ahead)


def _parse_investing_html(html: str, source: str) -> list[dict]:
    """Parsea el HTML de la tabla de investing.com."""
    soup = BeautifulSoup(html, "html.parser")
    rows = soup.find_all("tr", {"class": lambda c: c and "js-event-item" in c})
    events = []
    current_date = None

    for row in rows:
        try:
            # Fecha
            date_attr = row.get("data-event-datetime", "")
            if date_attr:
                dt = dateparser.parse(date_attr)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                event_dt = dt.astimezone(timezone.utc)
            elif current_date:
                event_dt = current_date
            else:
                continue

            # Nombre
            name_el = row.find("td", {"class": "event"})
            if not name_el:
                continue
            event_name = name_el.get_text(strip=True)

            # País / divisa
            flag = row.find("td", {"class": "flagCur"})
            country  = flag.get("title", "").split()[0][:2].upper() if flag else "US"
            currency = flag.get_text(strip=True)[:3] if flag else "USD"

            # Impacto (número de toros/estrellas)
            impact_el = row.find("td", {"class": "sentiment"})
            bull_count = len(impact_el.find_all("i", {"class": "grayFullBullishIcon"})) if impact_el else 0
            if bull_count >= 3:
                impact = "high"
            elif bull_count == 2:
                impact = "medium"
            else:
                impact = "low"

            # Valores
            actual   = _parse_float(row.find("td", {"id": lambda i: i and "actual" in i}).get_text(strip=True) if row.find("td", {"id": lambda i: i and "actual" in i}) else "")
            forecast = _parse_float(row.find("td", {"id": lambda i: i and "forecast" in i}).get_text(strip=True) if row.find("td", {"id": lambda i: i and "forecast" in i}) else "")
            previous = _parse_float(row.find("td", {"id": lambda i: i and "previous" in i}).get_text(strip=True) if row.find("td", {"id": lambda i: i and "previous" in i}) else "")

            surprise, surprise_pct, surprise_dir = _calculate_surprise(actual, forecast)
            btc_bias, bias_reason = _calculate_btc_bias(event_name, impact, surprise_dir, actual, forecast)

            events.append({
                "event_datetime": event_dt,
                "event_name":     event_name,
                "country":        country[:2],
                "currency":       currency[:3],
                "impact":         impact,
                "forecast":       forecast,
                "previous":       previous,
                "actual":         actual,
                "surprise":       surprise,
                "surprise_pct":   surprise_pct,
                "surprise_dir":   surprise_dir,
                "btc_bias":       btc_bias,
                "bias_reason":    bias_reason,
                "source":         source,
            })

        except Exception as e:
            log.debug(f"Error parseando fila investing.com: {e}")
            continue

    return events


# ── Scraper ForexFactory (fallback) ─────────────────────────────────────────

def scrape_forexfactory(days_ahead: int = 7) -> list[dict]:
    """
    Scrape ForexFactory como fuente alternativa.
    FF tiene semanas completas — tomamos la semana actual + siguiente.
    """
    session = requests.Session()
    session.headers.update(HEADERS)

    events = []
    now = datetime.now(timezone.utc)

    # FF usa URLs tipo /calendar?week=jan1.2024
    for week_offset in range(0, (days_ahead // 7) + 2):
        week_start = now + timedelta(weeks=week_offset)
        url = f"https://www.forexfactory.com/calendar?week={week_start.strftime('%b%d.%Y').lower()}"

        try:
            resp = session.get(url, timeout=20)
            resp.raise_for_status()
            week_events = _parse_forexfactory_html(resp.text, week_start, source="forexfactory")
            events.extend(week_events)
            time.sleep(1.5)  # cortés con el servidor
        except Exception as e:
            log.warning(f"ForexFactory semana {week_start.date()}: {e}")

    # Filtrar solo el rango deseado
    cutoff = now + timedelta(days=days_ahead)
    return [e for e in events if now <= e["event_datetime"] <= cutoff]


def _parse_forexfactory_html(html: str, reference_date: datetime, source: str) -> list[dict]:
    """Parsea el HTML de ForexFactory."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", {"class": "calendar__table"})
    if not table:
        return []

    events = []
    current_date = reference_date.date()
    current_time = None

    for row in table.find_all("tr", {"class": lambda c: c and "calendar__row" in c}):
        try:
            # Fecha (solo aparece en la primera fila del día)
            date_el = row.find("td", {"class": "calendar__date"})
            if date_el and date_el.get_text(strip=True):
                date_text = date_el.get_text(strip=True)
                try:
                    # FF formato: "Mon Jan 1"
                    parsed = dateparser.parse(f"{date_text} {reference_date.year}")
                    if parsed:
                        current_date = parsed.date()
                except Exception:
                    pass

            # Hora — FF muestra en ET (UTC-4 verano / UTC-5 invierno)
            time_el = row.find("td", {"class": "calendar__time"})
            if time_el and time_el.get_text(strip=True):
                time_text = time_el.get_text(strip=True)
                try:
                    from dateutil.tz import gettz
                    tz_et = gettz("America/New_York")
                    parsed_time = dateparser.parse(
                        f"{current_date} {time_text}",
                        tzinfos={"ET": tz_et, "EST": gettz("America/New_York"), "EDT": tz_et},
                    )
                    if parsed_time:
                        if parsed_time.tzinfo is None:
                            parsed_time = parsed_time.replace(tzinfo=tz_et)
                        current_time = parsed_time.astimezone(timezone.utc)
                except Exception:
                    current_time = None

            if current_time is None:
                continue

            # Divisa / País
            currency_el = row.find("td", {"class": "calendar__currency"})
            currency = currency_el.get_text(strip=True)[:3] if currency_el else ""
            if not currency or currency not in ("USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD"):
                continue  # Solo divisas G7 son relevantes para BTC

            country_map = {
                "USD": "US", "EUR": "EU", "GBP": "GB",
                "JPY": "JP", "CHF": "CH", "CAD": "CA", "AUD": "AU"
            }
            country = country_map.get(currency, "XX")

            # Impacto — ForexFactory usa: icon--ff-impact-red (high), -ora (medium), -yel/-gra (low)
            impact_el = row.find("td", {"class": "calendar__impact"})
            impact = "low"
            if impact_el:
                for tag in impact_el.find_all(["span", "i"]):
                    cls = " ".join(tag.get("class", [])).lower()
                    if "red" in cls or "impact-3" in cls:
                        impact = "high"
                        break
                    elif "ora" in cls or "orange" in cls or "impact-2" in cls:
                        impact = "medium"
                        break

            # Nombre evento
            event_el = row.find("td", {"class": "calendar__event"})
            event_name = event_el.get_text(strip=True) if event_el else ""
            if not event_name:
                continue

            # Valores
            actual_el   = row.find("td", {"class": "calendar__actual"})
            forecast_el = row.find("td", {"class": "calendar__forecast"})
            previous_el = row.find("td", {"class": "calendar__previous"})

            actual   = _parse_float(actual_el.get_text(strip=True)   if actual_el   else "")
            forecast = _parse_float(forecast_el.get_text(strip=True) if forecast_el else "")
            previous = _parse_float(previous_el.get_text(strip=True) if previous_el else "")

            surprise, surprise_pct, surprise_dir = _calculate_surprise(actual, forecast)
            btc_bias, bias_reason = _calculate_btc_bias(event_name, impact, surprise_dir, actual, forecast)

            events.append({
                "event_datetime": current_time,
                "event_name":     event_name,
                "country":        country,
                "currency":       currency,
                "impact":         impact,
                "forecast":       forecast,
                "previous":       previous,
                "actual":         actual,
                "surprise":       surprise,
                "surprise_pct":   surprise_pct,
                "surprise_dir":   surprise_dir,
                "btc_bias":       btc_bias,
                "bias_reason":    bias_reason,
                "source":         source,
            })

        except Exception as e:
            log.debug(f"Error parseando fila FF: {e}")
            continue

    return events


# ── TradingEconomics API (premium, opcional) ─────────────────────────────────

def scrape_tradingeconomics(days_ahead: int = 7) -> list[dict]:
    """
    TradingEconomics API — requiere TE_API_KEY en .env
    Mejor calidad de datos y más fiable que scraping.
    """
    api_key = os.getenv("TE_API_KEY", "").strip()
    if not api_key:
        log.info("TE_API_KEY no configurado — omitiendo TradingEconomics")
        return []

    now = datetime.now(timezone.utc)
    date_from = now.strftime("%Y-%m-%d")
    date_to   = (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    url = (
        f"https://api.tradingeconomics.com/calendar/country/united states"
        f"?c={api_key}&d1={date_from}&d2={date_to}&f=json"
    )

    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.warning(f"TradingEconomics API error: {e}")
        return []

    events = []
    for item in data:
        try:
            dt_str = item.get("Date") or item.get("date", "")
            dt = dateparser.parse(dt_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            event_dt = dt.astimezone(timezone.utc)

            importance = str(item.get("Importance", "1"))
            if importance == "3":
                impact = "high"
            elif importance == "2":
                impact = "medium"
            else:
                impact = "low"

            actual   = _parse_float(str(item.get("Actual",   "") or ""))
            forecast = _parse_float(str(item.get("Forecast", "") or ""))
            previous = _parse_float(str(item.get("Previous", "") or ""))
            event_name = item.get("Event") or item.get("event", "Unknown")

            surprise, surprise_pct, surprise_dir = _calculate_surprise(actual, forecast)
            btc_bias, bias_reason = _calculate_btc_bias(event_name, impact, surprise_dir, actual, forecast)

            events.append({
                "event_datetime": event_dt,
                "event_name":     event_name,
                "country":        item.get("Country", "US")[:2].upper(),
                "currency":       item.get("Currency", "USD")[:3],
                "impact":         impact,
                "forecast":       forecast,
                "previous":       previous,
                "actual":         actual,
                "surprise":       surprise,
                "surprise_pct":   surprise_pct,
                "surprise_dir":   surprise_dir,
                "btc_bias":       btc_bias,
                "bias_reason":    bias_reason,
                "source":         "tradingeconomics",
            })
        except Exception as e:
            log.debug(f"TE parse error: {e}")

    return events


# ── DB Insert ────────────────────────────────────────────────────────────────

def _insert_events(events: list[dict], engine) -> int:
    if not events:
        return 0
    inserted = 0
    with engine.begin() as conn:
        for ev in events:
            try:
                result = conn.execute(text("""
                    INSERT INTO economic_calendar (
                        event_datetime, event_name, country, currency, impact,
                        forecast, previous, actual, unit,
                        surprise, surprise_pct, surprise_dir,
                        btc_bias, bias_reason, source
                    )
                    VALUES (
                        :event_datetime, :event_name, :country, :currency, :impact,
                        :forecast, :previous, :actual, NULL,
                        :surprise, :surprise_pct, :surprise_dir,
                        :btc_bias, :bias_reason, :source
                    )
                    ON CONFLICT (event_datetime, event_name, country)
                    DO UPDATE SET
                        impact       = EXCLUDED.impact,
                        actual       = COALESCE(EXCLUDED.actual,       economic_calendar.actual),
                        surprise     = COALESCE(EXCLUDED.surprise,     economic_calendar.surprise),
                        surprise_pct = COALESCE(EXCLUDED.surprise_pct, economic_calendar.surprise_pct),
                        surprise_dir = COALESCE(EXCLUDED.surprise_dir, economic_calendar.surprise_dir),
                        btc_bias     = EXCLUDED.btc_bias,
                        bias_reason  = EXCLUDED.bias_reason,
                        fetched_at   = NOW()
                    WHERE economic_calendar.actual IS NULL
                """), {**ev, "surprise_dir": ev.get("surprise_dir"), "btc_bias": ev.get("btc_bias")})
                inserted += result.rowcount
            except Exception as e:
                log.debug(f"Calendar insert error ({ev.get('event_name')}): {e}")
    return inserted


def update_actuals(engine) -> int:
    """
    Re-calcula bias para eventos que ya tienen datos reales pero
    aún no tienen surprise_dir calculado.
    """
    with engine.begin() as conn:
        pending = conn.execute(text("""
            SELECT id, event_name, impact, actual, forecast
            FROM economic_calendar
            WHERE actual IS NOT NULL
              AND surprise_dir IS NULL
        """)).fetchall()

        updated = 0
        for row in pending:
            surprise, surprise_pct, surprise_dir = _calculate_surprise(row.actual, row.forecast)
            btc_bias, bias_reason = _calculate_btc_bias(
                row.event_name, row.impact, surprise_dir, row.actual, row.forecast
            )
            conn.execute(text("""
                UPDATE economic_calendar
                SET surprise     = :surprise,
                    surprise_pct = :surprise_pct,
                    surprise_dir = :surprise_dir,
                    btc_bias     = :btc_bias,
                    bias_reason  = :bias_reason
                WHERE id = :id
            """), {
                "surprise": surprise, "surprise_pct": surprise_pct,
                "surprise_dir": surprise_dir, "btc_bias": btc_bias,
                "bias_reason": bias_reason, "id": row.id,
            })
            updated += 1

    log.info(f"update_actuals: {updated} eventos actualizados")
    return updated


# ── Función principal ────────────────────────────────────────────────────────

def run_all(days_ahead: int = 7, update_actuals_flag: bool = False) -> dict:
    engine = get_engine()

    if update_actuals_flag:
        n = update_actuals(engine)
        return {"updated_actuals": n}

    log.info(f"Calendar scraper: próximos {days_ahead} días")

    # Intentar fuentes en orden
    events = scrape_tradingeconomics(days_ahead)  # mejor calidad si hay API key
    if not events:
        events = scrape_investing_com(days_ahead)
    if not events:
        log.error("Todas las fuentes fallaron")
        return {"error": "all_sources_failed"}

    # Filtrar solo eventos relevantes para BTC (USD primero, luego EUR/GBP)
    priority_currencies = {"USD", "EUR", "GBP"}
    events_prioritized = [e for e in events if e["currency"] in priority_currencies]
    events_other       = [e for e in events if e["currency"] not in priority_currencies]

    all_events = events_prioritized + events_other
    inserted = _insert_events(all_events, engine)

    # Resumen de próximos eventos high-impact
    high_impact = [
        {
            "datetime": e["event_datetime"].isoformat(),
            "event":    e["event_name"],
            "impact":   e["impact"],
            "btc_bias": e["btc_bias"],
            "reason":   e["bias_reason"],
        }
        for e in events_prioritized
        if e["impact"] == "high"
    ]
    high_impact.sort(key=lambda x: x["datetime"])

    summary = {
        "total_events":   len(all_events),
        "inserted":       inserted,
        "high_impact":    high_impact[:10],  # top 10 más próximos
        "timestamp":      datetime.now(timezone.utc).isoformat(),
    }

    log.info(f"Calendar: {len(all_events)} eventos, {inserted} insertados, {len(high_impact)} high-impact")
    return summary


def get_upcoming_bias(hours_ahead: int = 48, engine=None) -> list[dict]:
    """
    Consulta rápida: próximos eventos high-impact con bias calculado.
    Útil para el pipeline de bias_snapshots.
    """
    if engine is None:
        engine = get_engine()

    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=hours_ahead)

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT
                event_datetime, event_name, country, currency, impact,
                forecast, actual, surprise_dir, btc_bias, bias_reason
            FROM economic_calendar
            WHERE event_datetime BETWEEN :now AND :cutoff
              AND impact IN ('high', 'medium')
            ORDER BY event_datetime ASC
        """), {"now": now, "cutoff": cutoff}).fetchall()

    return [
        {
            "datetime":     r.event_datetime.isoformat(),
            "event":        r.event_name,
            "country":      r.country,
            "impact":       r.impact,
            "actual":       r.actual,
            "forecast":     r.forecast,
            "surprise_dir": r.surprise_dir,
            "btc_bias":     r.btc_bias,
            "bias_reason":  r.bias_reason,
            "hours_away":   round((r.event_datetime - now).total_seconds() / 3600, 1),
        }
        for r in rows
    ]


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--days",           type=int, default=7)
    ap.add_argument("--update-actuals", action="store_true")
    args = ap.parse_args()

    result = run_all(days_ahead=args.days, update_actuals_flag=args.update_actuals)
    print(json.dumps(result, indent=2, default=str))
