# pipeline/indicators.py
# Modulo centralizado de calculo de indicadores tecnicos
import pandas as pd
import numpy as np

def calc_rsi(series: pd.Series, period=14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(com=period-1, min_periods=period).mean()
    avg_l = loss.ewm(com=period-1, min_periods=period).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_macd(series, fast=12, slow=26, signal=9):
    ema_f  = series.ewm(span=fast, adjust=False).mean()
    ema_s  = series.ewm(span=slow, adjust=False).mean()
    macd   = ema_f - ema_s
    sig    = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig

def calc_bollinger(series, period=20, std=2.0):
    sma    = series.rolling(period).mean()
    stddev = series.rolling(period).std()
    return sma + std*stddev, sma - std*stddev

def calc_atr(high, low, close, period=14) -> pd.Series:
    """Average True Range."""
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(com=period-1, min_periods=period).mean()

def calc_obv(close, volume) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()

def calc_stochastic(high, low, close, period=14) -> pd.Series:
    """Stochastic %K."""
    lowest  = low.rolling(period).min()
    highest = high.rolling(period).max()
    denom   = (highest - lowest).replace(0, np.nan)
    return 100 * (close - lowest) / denom

def calc_williams_r(high, low, close, period=14) -> pd.Series:
    """Williams %R (rango -100 a 0)."""
    highest = high.rolling(period).max()
    lowest  = low.rolling(period).min()
    denom   = (highest - lowest).replace(0, np.nan)
    return -100 * (highest - close) / denom

def calc_order_blocks(open_: pd.Series, close: pd.Series,
                       returns: pd.Series, threshold=0.005):
    """
    Detecta Order Blocks.
    threshold: retorno minimo del movimiento siguiente para confirmar el OB (0.5% por defecto).
    Devuelve dos Series booleanas: ob_bullish, ob_bearish.
    """
    n = len(close)
    ob_bull = pd.Series(False, index=close.index)
    ob_bear = pd.Series(False, index=close.index)

    for i in range(1, n):
        prev_bearish = close.iloc[i-1] < open_.iloc[i-1]
        prev_bullish = close.iloc[i-1] > open_.iloc[i-1]
        next_ret     = returns.iloc[i] if i < n else 0

        if prev_bearish and next_ret > threshold:
            ob_bull.iloc[i-1] = True   # OB Bullish en la vela anterior
        if prev_bullish and next_ret < -threshold:
            ob_bear.iloc[i-1] = True   # OB Bearish en la vela anterior

    return ob_bull, ob_bear

def calc_fair_value_gaps(high: pd.Series, low: pd.Series):
    """
    Detecta Fair Value Gaps entre tres velas consecutivas.
    Devuelve dos Series booleanas: fvg_up (alcista), fvg_down (bajista).
    """
    fvg_up   = pd.Series(False, index=high.index)
    fvg_down = pd.Series(False, index=high.index)

    for i in range(1, len(high) - 1):
        # FVG alcista: hueco entre high[i-1] y low[i+1]
        if low.iloc[i+1] > high.iloc[i-1]:
            fvg_up.iloc[i] = True
        # FVG bajista: hueco entre low[i-1] y high[i+1]
        if high.iloc[i+1] < low.iloc[i-1]:
            fvg_down.iloc[i] = True

    return fvg_up, fvg_down

# Añadir al final de pipeline/indicators.py

def calc_volume_profile(high: pd.Series, low: pd.Series,
                         close: pd.Series, volume: pd.Series,
                         window: int = 24, n_bins: int = 50):
    """
    Calcula POC, VA_High y VA_Low con ventana rodante.
    window: numero de velas hacia atras (24 = 1 dia en datos horarios).
    n_bins: numero de niveles de precio en los que dividir el rango.
    Devuelve tres Series: poc, va_high, va_low.
    """
    poc_s    = pd.Series(np.nan, index=close.index)
    va_high_s = pd.Series(np.nan, index=close.index)
    va_low_s  = pd.Series(np.nan, index=close.index)

    for i in range(window, len(close)):
        w_high   = high.iloc[i-window:i]
        w_low    = low.iloc[i-window:i]
        w_close  = close.iloc[i-window:i]
        w_volume = volume.iloc[i-window:i]

        price_min = w_low.min()
        price_max = w_high.max()
        if price_max <= price_min: continue

        bins   = np.linspace(price_min, price_max, n_bins + 1)
        # Asignar cada vela al bin de precio mas cercano
        vol_by_bin = np.zeros(n_bins)
        for j in range(len(w_close)):
            idx = np.searchsorted(bins, w_close.iloc[j], side='right') - 1
            idx = min(max(idx, 0), n_bins - 1)
            vol_by_bin[idx] += w_volume.iloc[j]

        poc_idx    = np.argmax(vol_by_bin)
        poc_price  = (bins[poc_idx] + bins[poc_idx+1]) / 2
        poc_s.iloc[i] = poc_price

        # Value Area: bins que acumulan el 70% del volumen total
        total_vol  = vol_by_bin.sum()
        target_vol = total_vol * 0.70
        sorted_idx = np.argsort(vol_by_bin)[::-1]
        accum = 0; va_bins = []
        for idx in sorted_idx:
            accum += vol_by_bin[idx]
            va_bins.append(idx)
            if accum >= target_vol: break

        va_high_s.iloc[i] = bins[max(va_bins) + 1]
        va_low_s.iloc[i]  = bins[min(va_bins)]

    return poc_s, va_high_s, va_low_s
