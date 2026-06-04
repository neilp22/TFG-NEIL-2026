"""
05_etf_scraper.py — Scraping de flujos ETF Bitcoin desde Farside Investors

Uso:
  python agente_ia/news/05_etf_scraper.py
"""

import sys, re, logging
from datetime import datetime, timezone, timedelta, date
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / 'config' / '.env')

import requests
from bs4 import BeautifulSoup
from sqlalchemy import text
from db.db_utils import get_engine

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

FARSIDE_URL = "https://farside.co.uk/bitcoin-etf-flow/"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Columnas de ETF en el orden de la tabla Farside
ETF_VEHICLES = ["IBIT", "FBTC", "BITB", "ARKB", "BTCO", "EZBC", "BRRR", "HODL", "BTCW", "GBTC", "BTC"]

VEHICLE_TYPES = {
    "IBIT": "spot_etf", "FBTC": "spot_etf", "BITB": "spot_etf",
    "ARKB": "spot_etf", "BTCO": "spot_etf", "EZBC": "spot_etf",
    "BRRR": "spot_etf", "HODL": "spot_etf", "BTCW": "spot_etf",
    "GBTC": "spot_etf", "BTC": "spot_etf",
}


def _parse_flow(raw: str) -> Optional[float]:
    """
    Convierte strings de flujo a float (millones USD).
    Ejemplos: "$340.1" → 340.1, "(50.1)" → -50.1, "-" → None
    """
    s = raw.strip() if raw else ""
    if not s or s in ("-", "—", ""):
        return None
    # Negativos entre paréntesis: (50.1) → -50.1
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    s = s.replace("$", "").replace(",", "").replace(" ", "")
    try:
        return float(s)
    except ValueError:
        return None


def _find_etf_table(soup: BeautifulSoup):
    """Busca la tabla de flujos ETF entre todas las tablas de la página."""
    tables = soup.find_all("table")
    best_table, best_col_map = None, {}

    for table in tables:
        rows = table.find_all("tr")
        if len(rows) < 5:
            continue
        header_row = rows[0]
        headers = [th.get_text(strip=True) for th in header_row.find_all(["th", "td"])]
        col_map = {}
        for i, h in enumerate(headers):
            hu = h.upper().strip()
            if "DATE" in hu:
                col_map["date"] = i
            elif "TOTAL" in hu:
                col_map["total"] = i
            else:
                for etf in ETF_VEHICLES:
                    if etf == hu and etf not in col_map:
                        col_map[etf] = i

        etf_cols = sum(1 for e in ETF_VEHICLES if e in col_map)
        # Prefer tables with date + at least 2 ETF columns
        if "date" in col_map and etf_cols >= 2:
            if etf_cols > sum(1 for e in ETF_VEHICLES if e in best_col_map):
                best_table, best_col_map = table, col_map

    # Fallback: largest table
    if best_table is None and tables:
        best_table = max(tables, key=lambda t: len(t.find_all("tr")))

    return best_table, best_col_map


def scrape_farside(days_back: int = 30) -> list[dict]:
    """Descarga y parsea la tabla de Farside. Devuelve lista de dicts por (date, vehicle)."""
    urls = [FARSIDE_URL, FARSIDE_URL.rstrip("/")]
    resp = None
    for url in urls:
        try:
            r = requests.get(url, headers=HEADERS, timeout=25)
            r.raise_for_status()
            resp = r
            break
        except Exception as e:
            log.warning(f"Farside fetch error ({url}): {e}")

    if resp is None:
        return []

    soup  = BeautifulSoup(resp.text, "html.parser")
    table, col_map = _find_etf_table(soup)

    if not table:
        log.warning("Farside: no se encontró ninguna tabla HTML")
        return []

    if "date" not in col_map:
        # Last-resort: try to detect columns by position in largest table
        rows_all = table.find_all("tr")
        if rows_all:
            header_row = rows_all[0]
            headers = [th.get_text(strip=True) for th in header_row.find_all(["th", "td"])]
            log.warning(f"Farside: columna Date no detectada. Headers muestra: {headers[:10]}")
        return []

    cutoff  = date.today() - timedelta(days=days_back)
    records = []

    for row in table.find_all("tr")[1:]:
        cells = row.find_all(["td", "th"])
        if not cells:
            continue

        date_raw = cells[col_map["date"]].get_text(strip=True) if col_map["date"] < len(cells) else ""
        if not date_raw or "Total" in date_raw or not date_raw[0].isdigit() and not date_raw[:3].isalpha():
            continue
        try:
            parsed_date = _parse_date(date_raw)
            if parsed_date is None or parsed_date < cutoff:
                continue
        except Exception:
            continue

        for etf in ETF_VEHICLES:
            if etf not in col_map or col_map[etf] >= len(cells):
                continue
            flow_raw = cells[col_map[etf]].get_text(strip=True)
            flow_usd = _parse_flow(flow_raw)
            if flow_usd is None:
                continue
            records.append({
                "flow_date":    parsed_date,
                "vehicle":      etf,
                "vehicle_type": VEHICLE_TYPES.get(etf, "spot_etf"),
                "flow_usd":     flow_usd,
            })

    log.info(f"Farside: {len(records)} registros parseados ({days_back}d)")
    return records


def _parse_date(s: str) -> Optional[date]:
    from dateutil import parser as dp
    try:
        return dp.parse(s).date()
    except Exception:
        return None


def _calculate_streaks(records: list[dict]) -> dict[str, dict]:
    """Calcula rachas de inflow/outflow consecutivos por vehículo."""
    from collections import defaultdict
    by_vehicle: dict[str, list] = defaultdict(list)
    for r in records:
        by_vehicle[r["vehicle"]].append((r["flow_date"], r["flow_usd"]))

    streaks = {}
    for vehicle, flows in by_vehicle.items():
        flows_sorted = sorted(flows, key=lambda x: x[0])
        inflow_streak = outflow_streak = 0
        for _, flow in reversed(flows_sorted):
            if flow > 0:
                if outflow_streak == 0:
                    inflow_streak += 1
                else:
                    break
            elif flow < 0:
                if inflow_streak == 0:
                    outflow_streak += 1
                else:
                    break
            else:
                break
        streaks[vehicle] = {
            "consecutive_days_inflow":  inflow_streak,
            "consecutive_days_outflow": outflow_streak,
        }
    return streaks


def _insert_flows(records: list[dict], engine) -> int:
    if not records:
        return 0

    streaks = _calculate_streaks(records)
    inserted = 0

    with engine.begin() as conn:
        for r in records:
            streak = streaks.get(r["vehicle"], {})
            try:
                result = conn.execute(text("""
                    INSERT INTO institutional_flows
                        (flow_date, asset, vehicle, vehicle_type, flow_usd,
                         consecutive_days_inflow, consecutive_days_outflow, source)
                    VALUES
                        (:flow_date, 'BTC', :vehicle, :vehicle_type, :flow_usd,
                         :inflow_streak, :outflow_streak, 'farside')
                    ON CONFLICT (flow_date, vehicle) DO UPDATE SET
                        flow_usd = EXCLUDED.flow_usd,
                        consecutive_days_inflow  = EXCLUDED.consecutive_days_inflow,
                        consecutive_days_outflow = EXCLUDED.consecutive_days_outflow,
                        fetched_at = NOW()
                """), {
                    "flow_date":      r["flow_date"],
                    "vehicle":        r["vehicle"],
                    "vehicle_type":   r["vehicle_type"],
                    "flow_usd":       r["flow_usd"],
                    "inflow_streak":  streak.get("consecutive_days_inflow", 0),
                    "outflow_streak": streak.get("consecutive_days_outflow", 0),
                })
                inserted += result.rowcount
            except Exception as e:
                log.debug(f"ETF insert {r['vehicle']} {r['flow_date']}: {e}")

    return inserted


def run_all(engine=None) -> dict:
    if engine is None:
        engine = get_engine()

    records = scrape_farside(days_back=30)
    inserted = _insert_flows(records, engine)
    log.info(f"ETF flows: {len(records)} registros, {inserted} insertados/actualizados")
    return {"records": len(records), "inserted": inserted}


def calculate_institutional_score(engine=None) -> Optional[float]:
    """
    Score institucional basado en flujos ETF de los últimos 5 días hábiles.
    Retorna float en [-1.0, +1.0].
    """
    if engine is None:
        engine = get_engine()

    cutoff = date.today() - timedelta(days=10)  # margen amplio para 5 días hábiles

    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT
                flow_date,
                vehicle,
                flow_usd,
                consecutive_days_inflow,
                consecutive_days_outflow
            FROM institutional_flows
            WHERE flow_date >= :cutoff
              AND vehicle_type = 'spot_etf'
              AND asset = 'BTC'
            ORDER BY flow_date DESC
        """), {"cutoff": cutoff}).fetchall()

    if not rows:
        return None

    # Agrupar por fecha, tomar últimos 5 días hábiles
    from collections import defaultdict
    by_date: dict[date, float] = defaultdict(float)
    for r in rows:
        by_date[r.flow_date] += r.flow_usd

    sorted_dates = sorted(by_date.keys(), reverse=True)[:5]
    if not sorted_dates:
        return None

    # Score base: total flow normalizado
    total_5d = sum(float(by_date[d]) for d in sorted_dates)
    daily_avg = total_5d / len(sorted_dates)
    # ±500M/día = score ±1.0
    score = max(-1.0, min(1.0, daily_avg / 500.0))

    # GBTC penaliza doble si tiene outflows
    gbtc_rows = [r for r in rows if r.vehicle == "GBTC" and r.flow_date in sorted_dates]
    gbtc_total = sum(float(r.flow_usd) for r in gbtc_rows)
    if gbtc_total < 0:
        gbtc_penalty = max(-0.3, gbtc_total / 1000.0)  # -1B GBTC = -0.3 adicional
        score = max(-1.0, score + gbtc_penalty)

    # Bonus/penalización por rachas
    max_inflow_streak  = max((r.consecutive_days_inflow  or 0) for r in rows) if rows else 0
    max_outflow_streak = max((r.consecutive_days_outflow or 0) for r in rows) if rows else 0

    if max_inflow_streak >= 5:
        score = min(1.0, score + 0.15)
    if max_outflow_streak >= 5:
        score = max(-1.0, score - 0.15)

    return round(score, 3)


if __name__ == "__main__":
    engine = get_engine()
    result = run_all(engine)
    print("ETF result:", result)
    score = calculate_institutional_score(engine)
    print(f"Institutional score: {score}")
