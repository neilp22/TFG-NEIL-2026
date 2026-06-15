"""
analysis/ict_signals.py
─────────────────────────────────────────────────────────────────────────────
Detectores ICT (Inner Circle Trader) avanzados para BTC.

Implementa cuatro detectores institucionales clásicos:

    1. Liquidity Sweeps  ─ caza de stops minoristas (reversal de alta prob.)
    2. BOS / CHoCH       ─ Break of Structure / Change of Character
    3. Premium / Discount ─ posicionamiento dentro del rango (50% rule)
    4. Daily / Weekly Levels ─ liquidity pools institucionales

Más una función de scoring (`score_ict_advanced`) que combina todas las
señales activas en una vela en un score [-1, +1] para integración con el
sistema de confluencia (`analysis/backtest_confluence.py`,
`agente_ia/05_tools.py`).

Constraints:
    • Sin look-ahead: detección causal (solo info <= vela actual)
    • Solo pandas + numpy (sin nuevas dependencias)
    • Performance: <2 s sobre 500 velas 4h
    • DataFrame de entrada con índice timestamp UTC y columnas:
          open, high, low, close, volume + indicadores opcionales
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ═════════════════════════════════════════════════════════════════════════════
# 1) LIQUIDITY SWEEPS  (stop hunts)
# ═════════════════════════════════════════════════════════════════════════════
def detect_liquidity_sweeps(df: pd.DataFrame, lookback_swings: int = 10) -> pd.DataFrame:
    """
    Detecta "liquidity sweeps" o stop hunts: maniobras institucionales en las
    que el precio perfora momentáneamente un swing low/high previo (activando
    stops minoristas) pero cierra al otro lado, dejando una mecha falsa.

    - Bullish sweep (señal long): low de la vela perfora un swing low previo
      por ≥0.1% pero close queda POR ENCIMA de dicho swing low. Captura sells
      minoristas y suele preceder un rally.
    - Bearish sweep (señal short): high perfora swing high previo por ≥0.1%
      pero close queda POR DEBAJO. Captura buys y suele preceder caída.

    Args:
        df: DataFrame con columnas open/high/low/close indexado por timestamp.
        lookback_swings: cuántos swings recientes (highs+lows) considerar
            como niveles candidatos a ser cazados.

    Returns:
        DataFrame con columnas añadidas:
            liq_sweep_bull (bool), liq_sweep_bear (bool),
            swept_level (float, nivel exacto cazado o NaN).
    """
    out = df.copy()
    n = len(out)
    out["liq_sweep_bull"] = False
    out["liq_sweep_bear"] = False
    out["swept_level"] = np.nan

    if n < 7:
        return out

    highs = out["high"].values
    lows = out["low"].values
    closes = out["close"].values

    # Pivots de 3 velas (un swing requiere 3 velas a cada lado para confirmarse,
    # así que el swing en índice i solo es "conocido" en el bar i+3).
    swing_high_idx: list[int] = []
    swing_low_idx: list[int] = []
    for i in range(3, n - 3):
        window_h = highs[i - 3:i + 4]
        window_l = lows[i - 3:i + 4]
        if highs[i] == window_h.max() and (window_h == highs[i]).sum() == 1:
            swing_high_idx.append(i)
        if lows[i] == window_l.min() and (window_l == lows[i]).sum() == 1:
            swing_low_idx.append(i)

    threshold = 0.001  # 0.1% de perforación mínima

    for i in range(6, n):
        # Solo se conocen swings confirmados con al menos 3 velas posteriores:
        # swing en idx j está confirmado a partir de bar j+3.
        confirmed_highs = [j for j in swing_high_idx if j + 3 <= i - 1]
        confirmed_lows = [j for j in swing_low_idx if j + 3 <= i - 1]

        recent_highs = confirmed_highs[-lookback_swings:]
        recent_lows = confirmed_lows[-lookback_swings:]

        # Bullish sweep: perfora un swing low por ≥0.1% pero cierra encima
        for j in recent_lows:
            lvl = lows[j]
            if lvl <= 0:
                continue
            if lows[i] < lvl * (1 - threshold) and closes[i] > lvl:
                out.iat[i, out.columns.get_loc("liq_sweep_bull")] = True
                out.iat[i, out.columns.get_loc("swept_level")] = lvl
                break

        # Bearish sweep: perfora un swing high por ≥0.1% pero cierra debajo
        for j in recent_highs:
            lvl = highs[j]
            if lvl <= 0:
                continue
            if highs[i] > lvl * (1 + threshold) and closes[i] < lvl:
                out.iat[i, out.columns.get_loc("liq_sweep_bear")] = True
                # No sobrescribir swept_level si ya hay bullish (raro pero posible)
                if pd.isna(out.iat[i, out.columns.get_loc("swept_level")]):
                    out.iat[i, out.columns.get_loc("swept_level")] = lvl
                break

    return out


# ═════════════════════════════════════════════════════════════════════════════
# 2) BOS / CHoCH  (Break of Structure / Change of Character)
# ═════════════════════════════════════════════════════════════════════════════
def detect_bos_choch(df: pd.DataFrame, lookback: int = 3) -> pd.DataFrame:
    """
    Detecta rupturas de estructura de mercado al estilo ICT/SMC.

    - BOS (Break of Structure): continuación de tendencia. En uptrend, cierre
      por encima del último swing high confirmado; en downtrend, cierre por
      debajo del último swing low.
    - CHoCH (Change of Character): REVERSAL de tendencia. En uptrend, cierre
      por debajo del último swing low (cambio a bajista); en downtrend, cierre
      por encima del último swing high (cambio a alcista).

    Detección de tendencia con los últimos 4 swings:
        HH + HL → 'up'      (higher-high, higher-low)
        LH + LL → 'down'
        mixto   → 'side'

    Args:
        df: DataFrame OHLC con índice timestamp.
        lookback: nº de velas a cada lado para confirmar un pivot. Default 3.

    Returns:
        DataFrame con columnas:
            swing_high, swing_low (bool),
            bos_bull, bos_bear, choch_bull, choch_bear (bool),
            trend (str: 'up' / 'down' / 'side').
    """
    out = df.copy()
    n = len(out)

    out["swing_high"] = False
    out["swing_low"] = False
    out["bos_bull"] = False
    out["bos_bear"] = False
    out["choch_bull"] = False
    out["choch_bear"] = False
    out["trend"] = "side"

    if n < 2 * lookback + 2:
        return out

    highs = out["high"].values
    lows = out["low"].values
    closes = out["close"].values

    # 1) Pivot detection (causal: el pivot en i se confirma en i+lookback)
    for i in range(lookback, n - lookback):
        wh = highs[i - lookback:i + lookback + 1]
        wl = lows[i - lookback:i + lookback + 1]
        if highs[i] == wh.max() and (wh == highs[i]).sum() == 1:
            out.iat[i, out.columns.get_loc("swing_high")] = True
        if lows[i] == wl.min() and (wl == lows[i]).sum() == 1:
            out.iat[i, out.columns.get_loc("swing_low")] = True

    swing_high_mask = out["swing_high"].values
    swing_low_mask = out["swing_low"].values

    # 2) Recorrido bar-a-bar: en cada cierre, usamos solo swings ya confirmados
    #    (es decir, swings cuyo bar de confirmación j+lookback <= i-1).
    trend_arr = np.array(["side"] * n, dtype=object)
    current_trend = "side"

    for i in range(2 * lookback + 1, n):
        cutoff = i - lookback  # último índice donde un swing pudo confirmarse
        sh_idx = [j for j in range(cutoff) if swing_high_mask[j]]
        sl_idx = [j for j in range(cutoff) if swing_low_mask[j]]

        # Determinar tendencia con últimos 2 swing highs y 2 swing lows
        if len(sh_idx) >= 2 and len(sl_idx) >= 2:
            last_h, prev_h = highs[sh_idx[-1]], highs[sh_idx[-2]]
            last_l, prev_l = lows[sl_idx[-1]], lows[sl_idx[-2]]
            hh = last_h > prev_h
            hl = last_l > prev_l
            lh = last_h < prev_h
            ll = last_l < prev_l
            if hh and hl:
                current_trend = "up"
            elif lh and ll:
                current_trend = "down"
            else:
                current_trend = "side"

        trend_arr[i] = current_trend

        # Comprobar rupturas usando último swing high / low confirmado
        last_sh_lvl = highs[sh_idx[-1]] if sh_idx else np.nan
        last_sl_lvl = lows[sl_idx[-1]] if sl_idx else np.nan
        c = closes[i]

        if current_trend == "up":
            if not np.isnan(last_sh_lvl) and c > last_sh_lvl:
                out.iat[i, out.columns.get_loc("bos_bull")] = True
            if not np.isnan(last_sl_lvl) and c < last_sl_lvl:
                out.iat[i, out.columns.get_loc("choch_bear")] = True
                current_trend = "down"   # cambio de carácter
        elif current_trend == "down":
            if not np.isnan(last_sl_lvl) and c < last_sl_lvl:
                out.iat[i, out.columns.get_loc("bos_bear")] = True
            if not np.isnan(last_sh_lvl) and c > last_sh_lvl:
                out.iat[i, out.columns.get_loc("choch_bull")] = True
                current_trend = "up"
        else:  # side
            # En lateral, una ruptura clara también dispara CHoCH hacia el nuevo régimen
            if not np.isnan(last_sh_lvl) and c > last_sh_lvl:
                out.iat[i, out.columns.get_loc("choch_bull")] = True
                current_trend = "up"
            elif not np.isnan(last_sl_lvl) and c < last_sl_lvl:
                out.iat[i, out.columns.get_loc("choch_bear")] = True
                current_trend = "down"

    out["trend"] = trend_arr
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 3) PREMIUM / DISCOUNT  (regla del 50%)
# ═════════════════════════════════════════════════════════════════════════════
def compute_premium_discount(df: pd.DataFrame, range_lookback: int = 50) -> pd.DataFrame:
    """
    Divide el rango reciente al 50%: mitad superior = "premium" (caro, bias
    short), mitad inferior = "discount" (barato, bias long), centro = equilibrio.

    Concepto ICT: las instituciones venden en premium y compran en discount.
    Operar en el lado correcto del 50% mejora el R:R.

    Args:
        df: DataFrame OHLC.
        range_lookback: ventana rolling para definir el rango.

    Returns:
        DataFrame con columnas:
            range_high, range_low, range_mid,
            pd_position_pct (0=fondo, 100=top),
            is_premium (>65%), is_discount (<35%),
            is_equilibrium (45-55%).
    """
    out = df.copy()
    out["range_high"] = out["high"].rolling(range_lookback, min_periods=range_lookback).max()
    out["range_low"] = out["low"].rolling(range_lookback, min_periods=range_lookback).min()
    out["range_mid"] = (out["range_high"] + out["range_low"]) / 2.0

    rng = (out["range_high"] - out["range_low"]).replace(0, np.nan)
    out["pd_position_pct"] = (out["close"] - out["range_low"]) / rng * 100.0

    out["is_premium"] = out["pd_position_pct"] > 65
    out["is_discount"] = out["pd_position_pct"] < 35
    out["is_equilibrium"] = (out["pd_position_pct"] >= 45) & (out["pd_position_pct"] <= 55)

    return out


# ═════════════════════════════════════════════════════════════════════════════
# 4) DAILY / WEEKLY LEVELS  (liquidity pools)
# ═════════════════════════════════════════════════════════════════════════════
def detect_daily_weekly_levels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula los highs/lows del día y semana PREVIOS y la distancia del close
    actual a esos niveles.

    Concepto ICT: los PDH/PDL (Previous Day High/Low) y PWH/PWL son "liquidity
    pools" donde se acumulan stops. El precio tiende a buscarlos y reaccionar
    cerca de ellos.

    Args:
        df: DataFrame OHLC con índice DatetimeIndex (UTC recomendado).

    Returns:
        DataFrame con columnas:
            prior_day_high, prior_day_low,
            prior_week_high, prior_week_low,
            dist_to_daily_high_pct, dist_to_daily_low_pct,
            dist_to_weekly_high_pct, dist_to_weekly_low_pct,
            near_liquidity_pool (bool, |dist| < 0.5% a cualquier pool).
    """
    out = df.copy()

    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, utc=True)

    # Daily resample
    daily = out[["high", "low"]].resample("1D").agg({"high": "max", "low": "min"})
    daily["prior_day_high"] = daily["high"].shift(1)
    daily["prior_day_low"] = daily["low"].shift(1)

    # Weekly resample (semana ISO terminando en domingo: 'W')
    weekly = out[["high", "low"]].resample("1W").agg({"high": "max", "low": "min"})
    weekly["prior_week_high"] = weekly["high"].shift(1)
    weekly["prior_week_low"] = weekly["low"].shift(1)

    # Reindex hacia el granularidad original con forward-fill
    out["prior_day_high"] = daily["prior_day_high"].reindex(out.index, method="ffill")
    out["prior_day_low"] = daily["prior_day_low"].reindex(out.index, method="ffill")
    out["prior_week_high"] = weekly["prior_week_high"].reindex(out.index, method="ffill")
    out["prior_week_low"] = weekly["prior_week_low"].reindex(out.index, method="ffill")

    close = out["close"].replace(0, np.nan)
    out["dist_to_daily_high_pct"] = (out["prior_day_high"] - close) / close * 100.0
    out["dist_to_daily_low_pct"] = (out["prior_day_low"] - close) / close * 100.0
    out["dist_to_weekly_high_pct"] = (out["prior_week_high"] - close) / close * 100.0
    out["dist_to_weekly_low_pct"] = (out["prior_week_low"] - close) / close * 100.0

    near = (
        out[
            [
                "dist_to_daily_high_pct",
                "dist_to_daily_low_pct",
                "dist_to_weekly_high_pct",
                "dist_to_weekly_low_pct",
            ]
        ]
        .abs()
        .min(axis=1)
    )
    out["near_liquidity_pool"] = near < 0.5

    return out


# ═════════════════════════════════════════════════════════════════════════════
# SCORE GLOBAL  (combina las 4 detecciones en un único score [-1, +1])
# ═════════════════════════════════════════════════════════════════════════════
def score_ict_advanced(
    row: pd.Series,
    df_slice: pd.DataFrame,
    lookback: int = 50,
) -> tuple[float, list[str]]:
    """
    Convierte las señales ICT activas en una vela en un score [-1, +1].

    Pesos:
        liq_sweep_bull/bear   ±0.5
        bos_bull/bear         ±0.4
        choch_bull/bear       ±0.6   (más fuerte: cambio de carácter)
        is_discount + close>ema_50   +0.2
        is_premium  + close<ema_50   -0.2
        near_liquidity_pool   ±0.3   (signo según discount/premium)

    Aporta el promedio (suma / nº de señales activas), clipeado a [-1, +1],
    para no penalizar momentos sin actividad ICT.

    Args:
        row: pd.Series con los flags ICT (resultado de los detectores).
        df_slice: slice del DataFrame hasta esa vela inclusive (df.iloc[:i+1]).
            No se usa para look-ahead; reservado para extensiones futuras.
        lookback: parámetro reservado para usos rolling adicionales.

    Returns:
        (score, signals): score float en [-1, +1] y lista de strings con las
        señales activas para logging/auditoría.
    """
    contributions: list[float] = []
    signals: list[str] = []

    # Liquidity sweeps
    if bool(row.get("liq_sweep_bull", False)):
        contributions.append(+0.5)
        signals.append("liq_sweep_bull(+0.5)")
    if bool(row.get("liq_sweep_bear", False)):
        contributions.append(-0.5)
        signals.append("liq_sweep_bear(-0.5)")

    # BOS
    if bool(row.get("bos_bull", False)):
        contributions.append(+0.4)
        signals.append("bos_bull(+0.4)")
    if bool(row.get("bos_bear", False)):
        contributions.append(-0.4)
        signals.append("bos_bear(-0.4)")

    # CHoCH (más fuerte)
    if bool(row.get("choch_bull", False)):
        contributions.append(+0.6)
        signals.append("choch_bull(+0.6)")
    if bool(row.get("choch_bear", False)):
        contributions.append(-0.6)
        signals.append("choch_bear(-0.6)")

    # Premium / Discount + alineación con EMA50
    close = row.get("close", np.nan)
    ema_50 = row.get("ema_50", np.nan)
    is_discount = bool(row.get("is_discount", False))
    is_premium = bool(row.get("is_premium", False))

    if is_discount and not pd.isna(close) and not pd.isna(ema_50) and close > ema_50:
        contributions.append(+0.2)
        signals.append("discount+above_ema50(+0.2)")
    if is_premium and not pd.isna(close) and not pd.isna(ema_50) and close < ema_50:
        contributions.append(-0.2)
        signals.append("premium+below_ema50(-0.2)")

    # Liquidity pool proximity (dirección según D/P)
    if bool(row.get("near_liquidity_pool", False)):
        if is_discount:
            contributions.append(+0.3)
            signals.append("near_pool+discount(+0.3)")
        elif is_premium:
            contributions.append(-0.3)
            signals.append("near_pool+premium(-0.3)")

    if not contributions:
        return 0.0, signals

    raw = sum(contributions) / len(contributions)
    score = float(np.clip(raw, -1.0, 1.0))
    return score, signals


# ═════════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    import time
    import traceback
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))

    try:
        from sqlalchemy import text  # noqa: E402

        from db.db_utils import get_engine  # noqa: E402

        # Reusar compute_indicators del backtest si está disponible
        try:
            from analysis.backtest_confluence import compute_indicators  # type: ignore
        except Exception:
            def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
                out = df.copy()
                out["ema_9"] = out["close"].ewm(span=9, adjust=False).mean()
                out["ema_21"] = out["close"].ewm(span=21, adjust=False).mean()
                out["ema_50"] = out["close"].ewm(span=50, adjust=False).mean()
                out["ema_200"] = out["close"].ewm(span=200, adjust=False).mean()
                delta = out["close"].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss.replace(0, np.nan)
                out["rsi_14"] = 100 - (100 / (1 + rs))
                tr = pd.concat(
                    [
                        out["high"] - out["low"],
                        (out["high"] - out["close"].shift()).abs(),
                        (out["low"] - out["close"].shift()).abs(),
                    ],
                    axis=1,
                ).max(axis=1)
                out["atr_14"] = tr.rolling(14).mean()
                return out

        engine = get_engine()
        sql = text(
            """
            SELECT timestamp, open, high, low, close, volume
            FROM btc_ohlcv
            WHERE timeframe = '4h'
            ORDER BY timestamp DESC
            LIMIT 200
            """
        )
        with engine.connect() as conn:
            rows = conn.execute(sql).fetchall()

        if not rows:
            raise RuntimeError("No se encontraron velas 4h en btc_ohlcv")

        df = pd.DataFrame(
            rows, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index().astype(float)

        print(f"[self-test] Cargadas {len(df)} velas 4h "
              f"({df.index[0]} → {df.index[-1]})")

        t0 = time.time()
        df = compute_indicators(df)
        df = detect_liquidity_sweeps(df, lookback_swings=10)
        df = detect_bos_choch(df, lookback=3)
        df = compute_premium_discount(df, range_lookback=50)
        df = detect_daily_weekly_levels(df)
        elapsed = time.time() - t0
        print(f"[self-test] Detectores completados en {elapsed:.2f}s")

        print("\n[self-test] Conteo de señales detectadas:")
        flags = [
            "liq_sweep_bull",
            "liq_sweep_bear",
            "swing_high",
            "swing_low",
            "bos_bull",
            "bos_bear",
            "choch_bull",
            "choch_bear",
            "is_premium",
            "is_discount",
            "is_equilibrium",
            "near_liquidity_pool",
        ]
        for f in flags:
            if f in df.columns:
                print(f"   {f:<25} {int(df[f].fillna(False).astype(bool).sum())}")

        print("\n[self-test] Distribución de trend:")
        if "trend" in df.columns:
            print(df["trend"].value_counts().to_string())

        # Scoring sobre últimas 10 filas con alguna señal activa
        signal_cols = [
            "liq_sweep_bull",
            "liq_sweep_bear",
            "bos_bull",
            "bos_bear",
            "choch_bull",
            "choch_bear",
            "near_liquidity_pool",
        ]
        active_mask = df[signal_cols].fillna(False).astype(bool).any(axis=1)
        active = df[active_mask].tail(10)

        print(f"\n[self-test] Últimas {len(active)} velas con señales ICT activas "
              f"(de {int(active_mask.sum())} totales):")
        for ts, row in active.iterrows():
            i = df.index.get_loc(ts)
            score, signals = score_ict_advanced(row, df.iloc[: i + 1])
            sig_str = ", ".join(signals) if signals else "—"
            print(f"   {ts}  score={score:+.3f}  trend={row.get('trend','?'):<4}  "
                  f"close={row['close']:.2f}  [{sig_str}]")

        print("\n[self-test] OK")

    except Exception as e:  # noqa: BLE001
        print(f"\n[self-test] FALLO: {e}\n")
        traceback.print_exc()
        sys.exit(1)
