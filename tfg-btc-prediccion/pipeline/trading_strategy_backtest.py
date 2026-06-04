"""
trading_strategy_backtest.py
==============================
Estrategia de trading basada en indicadores técnicos (RSI, MACD, Bollinger Bands)
con posiciones largas y cortas. Comparativa contra buy-and-hold.

Métricas reportadas:
  - Accuracy de señales
  - Total return, Sharpe Ratio, Max Drawdown
  - Nº operaciones, % operaciones ganadoras
  - Comparativa directa vs Buy & Hold

Uso:
  python trading_strategy_backtest.py
  python trading_strategy_backtest.py --start 2022-01-01 --end 2024-01-01
  python trading_strategy_backtest.py --no-short   # solo largos

Neul Padras · TFG 2025/26
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from sqlalchemy import text

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from db.db_utils import get_engine

# ── Estilo visual ──────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor':   '#161b22',
    'axes.edgecolor':   '#30363d',
    'axes.labelcolor':  '#c9d1d9',
    'xtick.color':      '#8b949e',
    'ytick.color':      '#8b949e',
    'text.color':       '#c9d1d9',
    'grid.color':       '#21262d',
    'grid.linestyle':   '--',
    'grid.alpha':       0.5,
    'font.family':      'monospace',
})

ORANGE  = '#f7931a'   # Bitcoin
GREEN   = '#3fb950'   # largo / ganancia
RED     = '#f85149'   # corto / pérdida
BLUE    = '#58a6ff'   # buy & hold
PURPLE  = '#a371f7'   # señales


# ══════════════════════════════════════════════════════════════════════
#  1. CARGA DE DATOS
# ══════════════════════════════════════════════════════════════════════

def load_data(asset: str = 'BTC',
              start: str = '2021-01-01',
              end:   str = None) -> pd.DataFrame:
    engine = get_engine()
    end_filter = "AND date <= :end" if end else ""
    query = f"""
        SELECT
            date, close, returns, label,
            rsi_14, macd, macd_signal,
            bb_upper, bb_lower,
            sma_7, sma_30,
            sentiment_avg,
            fear_greed
        FROM daily_features
        WHERE asset = :asset
          AND date  >= :start
          {end_filter}
        ORDER BY date ASC
    """
    params = {'asset': asset, 'start': start}
    if end:
        params['end'] = end

    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params=params)

    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df = df.dropna(subset=['close', 'rsi_14', 'macd'])

    # Calcular features de sentimiento derivadas (no existen en el schema)
    df['sentiment_avg']      = df['sentiment_avg'].fillna(0.0)
    df['fear_greed']         = df['fear_greed'].fillna(df['fear_greed'].median())
    df['sentiment_7d']       = df['sentiment_avg'].rolling(7).mean()
    df['sentiment_momentum'] = df['sentiment_avg'].diff()

    print(f"✅ Datos cargados: {len(df)} días  "
          f"({df.index.min().date()} → {df.index.max().date()})")
    return df


# ══════════════════════════════════════════════════════════════════════
#  2. GENERACIÓN DE SEÑALES
# ══════════════════════════════════════════════════════════════════════
#
#  Lógica de la estrategia (reglas manuales sobre indicadores técnicos):
#
#  SEÑAL LARGA (+1):  entrar comprado cuando hay consenso alcista
#    - RSI < 70  (no sobrecomprado)  AND
#    - MACD > MACD_signal  (momentum alcista)  AND
#    - close > SMA_30  (tendencia alcista de fondo)
#
#  SEÑAL CORTA (-1):  vender en corto cuando hay consenso bajista
#    - RSI > 30  (no sobrevendido, para evitar rebotes)  AND
#    - MACD < MACD_signal  (momentum bajista)  AND
#    - close < SMA_30  (tendencia bajista de fondo)
#
#  SEÑAL NEUTRA (0):  fuera del mercado en situaciones ambiguas
#    - RSI entre 45 y 55 (mercado lateral sin dirección clara)
#    - O cuando Bollinger se estrecha (baja volatilidad, sin señal)
#
#  CONFIANZA ALTA: se añade una capa extra de filtro usando
#    - Bandas de Bollinger: largo solo si close > bb_lower (soporte)
#    - Corto solo si close < bb_upper (resistencia)
#    - Sentimiento como confirmación (si está disponible)

def generate_signals(df: pd.DataFrame,
                     allow_short: bool = True,
                     use_sentiment: bool = True) -> pd.DataFrame:
    """
    Genera la columna 'signal' con valores:  1 (largo), -1 (corto), 0 (fuera)
    y 'confidence' con valores: 'high', 'medium', 'low'
    """
    df = df.copy()

    # ── Condiciones básicas ───────────────────────────────────────────
    rsi        = df['rsi_14']
    macd_cross = df['macd'] > df['macd_signal']   # True = alcista
    trend_up   = df['close'] > df['sma_30']        # True = sobre SMA30
    bb_width   = df['bb_upper'] - df['bb_lower']
    bb_width_pct = bb_width / df['close']           # ancho relativo

    # Bollinger: precio en zona baja o alta de la banda
    near_bb_lower = df['close'] <= df['bb_lower'] * 1.02   # cerca del soporte
    near_bb_upper = df['close'] >= df['bb_upper'] * 0.98   # cerca de resistencia

    # Condición de mercado lateral (evitar operar en rangos estrechos)
    lateral = (rsi > 45) & (rsi < 55) & (bb_width_pct < 0.04)

    # ── Señal LARGA ───────────────────────────────────────────────────
    long_signal = (
        (rsi < 70) &          # no sobrecomprado
        (rsi > 40) &          # con algo de momentum
        macd_cross &           # MACD cruzando al alza
        trend_up &             # sobre la media de 30 días
        ~lateral               # no mercado lateral
    )

    # ── Señal CORTA ───────────────────────────────────────────────────
    short_signal = (
        (rsi > 30) &           # no sobrevendido (evitar rebotes)
        (rsi < 60) &           # sin momentum alcista fuerte
        ~macd_cross &          # MACD cruzando a la baja
        ~trend_up &            # bajo la media de 30 días
        ~lateral               # no mercado lateral
    )

    # ── Señal base ────────────────────────────────────────────────────
    df['signal'] = 0
    df.loc[long_signal,  'signal'] =  1
    if allow_short:
        df.loc[short_signal, 'signal'] = -1

    # ── Nivel de confianza ────────────────────────────────────────────
    # HIGH: señal técnica + Bollinger confirma + (sentimiento si disponible)
    # MEDIUM: señal técnica + uno de los dos filtros extra
    # LOW: solo señal técnica básica

    def calc_confidence(row):
        sig = row['signal']
        if sig == 0:
            return 'none'

        score = 0

        # Confirmación Bollinger
        if sig == 1 and row['close'] > row['bb_lower']:
            score += 1
        if sig == -1 and row['close'] < row['bb_upper']:
            score += 1

        # Confirmación sentimiento (si está disponible)
        if use_sentiment and pd.notna(row.get('sentiment_7d')):
            if sig == 1 and row['sentiment_7d'] > 0:
                score += 1
            if sig == -1 and row['sentiment_7d'] < 0:
                score += 1

        # Confirmación Fear & Greed
        if pd.notna(row.get('fear_greed')):
            if sig == 1 and row['fear_greed'] > 50:
                score += 1
            if sig == -1 and row['fear_greed'] < 50:
                score += 1

        if score >= 2:
            return 'high'
        elif score == 1:
            return 'medium'
        else:
            return 'low'

    df['confidence'] = df.apply(calc_confidence, axis=1)

    # ── Resumen de señales ────────────────────────────────────────────
    total = len(df)
    n_long  = (df['signal'] ==  1).sum()
    n_short = (df['signal'] == -1).sum()
    n_out   = (df['signal'] ==  0).sum()
    n_high  = (df['confidence'] == 'high').sum()

    print(f"\n📊 Señales generadas ({total} días):")
    print(f"   Largo  (+1): {n_long:4d} días  ({n_long/total*100:.1f}%)")
    print(f"   Corto  (-1): {n_short:4d} días  ({n_short/total*100:.1f}%)")
    print(f"   Fuera  ( 0): {n_out:4d} días  ({n_out/total*100:.1f}%)")
    print(f"   Alta confianza: {n_high} señales")

    return df


# ══════════════════════════════════════════════════════════════════════
#  3. BACKTEST
# ══════════════════════════════════════════════════════════════════════

def run_backtest(df: pd.DataFrame,
                 commission: float = 0.001,   # 0.1% por operación (Binance)
                 high_conf_only: bool = False
                 ) -> pd.DataFrame:
    """
    Simula el backtest día a día.

    - La señal del día t se ejecuta al cierre del día t
      (compra/vende al precio de cierre).
    - El retorno se obtiene al día t+1.
    - Comisión aplicada en cada cambio de posición.
    - Si high_conf_only=True, solo opera cuando confidence='high'.

    Returns el DataFrame con columnas adicionales de backtest.
    """
    df = df.copy()

    # Filtro de confianza alta
    if high_conf_only:
        df['signal_active'] = np.where(
            df['confidence'] == 'high', df['signal'], 0
        )
    else:
        df['signal_active'] = df['signal']

    # Retorno del día siguiente (lo que gana la posición)
    df['next_return'] = df['returns'].shift(-1)

    # Retorno de la estrategia antes de comisiones
    df['strategy_return_gross'] = df['signal_active'] * df['next_return']

    # Detectar cambios de posición para aplicar comisiones
    df['position_change'] = df['signal_active'].diff().abs()
    df['commission_cost'] = df['position_change'] * commission

    # Retorno neto de la estrategia
    df['strategy_return'] = df['strategy_return_gross'] - df['commission_cost']

    # Buy & Hold (siempre largo, sin comisiones)
    df['bh_return'] = df['next_return']

    # Retornos acumulados
    df['strategy_cumret'] = df['strategy_return'].cumsum().apply(np.exp)
    df['bh_cumret']       = df['bh_return'].cumsum().apply(np.exp)

    # Eliminar última fila (sin retorno futuro)
    df = df.dropna(subset=['next_return'])

    return df


# ══════════════════════════════════════════════════════════════════════
#  4. MÉTRICAS
# ══════════════════════════════════════════════════════════════════════

def compute_metrics(df: pd.DataFrame) -> dict:
    """Calcula todas las métricas de evaluación del TFG."""

    ret = df['strategy_return'].dropna()
    bh  = df['bh_return'].dropna()

    # ── Retorno total ─────────────────────────────────────────────────
    total_ret_strat = np.exp(ret.sum()) - 1
    total_ret_bh    = np.exp(bh.sum()) - 1

    # ── Sharpe Ratio (anualizado, base 365 para cripto) ───────────────
    sharpe_strat = (ret.mean() / ret.std()) * np.sqrt(365) if ret.std() > 0 else 0
    sharpe_bh    = (bh.mean() / bh.std())  * np.sqrt(365) if bh.std() > 0 else 0

    # ── Max Drawdown ──────────────────────────────────────────────────
    def max_drawdown(returns):
        cum = returns.cumsum().apply(np.exp)
        peak = cum.cummax()
        dd = (cum - peak) / peak
        return dd.min()

    mdd_strat = max_drawdown(ret)
    mdd_bh    = max_drawdown(bh)

    # ── Accuracy de señales (días que predijo bien la dirección) ──────
    active = df[df['signal_active'] != 0].copy()
    if len(active) > 0:
        correct = ((active['signal_active'] == 1) & (active['next_return'] > 0)) | \
                  ((active['signal_active'] == -1) & (active['next_return'] < 0))
        accuracy = correct.sum() / len(active)
    else:
        accuracy = 0

    # ── Operaciones ───────────────────────────────────────────────────
    trades = (df['position_change'] > 0).sum()

    # ── Días en el mercado ────────────────────────────────────────────
    days_in  = (df['signal_active'] != 0).sum()
    days_out = (df['signal_active'] == 0).sum()

    # ── Calmar Ratio (retorno / max drawdown) ─────────────────────────
    calmar = total_ret_strat / abs(mdd_strat) if mdd_strat != 0 else 0

    metrics = {
        # Estrategia
        'total_return_strategy':  round(total_ret_strat * 100, 2),
        'total_return_bh':        round(total_ret_bh    * 100, 2),
        'sharpe_strategy':        round(sharpe_strat, 3),
        'sharpe_bh':              round(sharpe_bh,    3),
        'max_drawdown_strategy':  round(mdd_strat * 100, 2),
        'max_drawdown_bh':        round(mdd_bh    * 100, 2),
        'accuracy_signals':       round(accuracy * 100, 2),
        'n_trades':               int(trades),
        'days_in_market':         int(days_in),
        'days_out_market':        int(days_out),
        'calmar_ratio':           round(calmar, 3),
        'total_days':             len(df),
    }
    return metrics


def print_metrics(metrics: dict, high_conf: bool = False):
    label = "ALTA CONFIANZA" if high_conf else "TODAS LAS SEÑALES"
    print(f"\n{'═'*55}")
    print(f"  RESULTADOS — {label}")
    print(f"{'═'*55}")
    print(f"  {'Métrica':<30} {'Estrategia':>10}  {'B&H':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Retorno total (%)':<30} {metrics['total_return_strategy']:>10.2f}  "
          f"{metrics['total_return_bh']:>10.2f}")
    print(f"  {'Sharpe Ratio (anual)':<30} {metrics['sharpe_strategy']:>10.3f}  "
          f"{metrics['sharpe_bh']:>10.3f}")
    print(f"  {'Max Drawdown (%)':<30} {metrics['max_drawdown_strategy']:>10.2f}  "
          f"{metrics['max_drawdown_bh']:>10.2f}")
    print(f"  {'Calmar Ratio':<30} {metrics['calmar_ratio']:>10.3f}  {'—':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Accuracy señales (%)':<30} {metrics['accuracy_signals']:>10.2f}  {'—':>10}")
    print(f"  {'Nº operaciones':<30} {metrics['n_trades']:>10}  {'—':>10}")
    print(f"  {'Días en mercado':<30} {metrics['days_in_market']:>10}  "
          f"{metrics['total_days']:>10}")
    print(f"{'═'*55}")

    # Veredicto
    if metrics['sharpe_strategy'] > metrics['sharpe_bh']:
        print(f"\n  ✅ Sharpe de la estrategia SUPERA al B&H")
    else:
        delta = metrics['sharpe_bh'] - metrics['sharpe_strategy']
        print(f"\n  ⚠️  Sharpe de la estrategia por debajo del B&H en {delta:.3f}")

    if metrics['accuracy_signals'] > 53:
        print(f"  ✅ Accuracy > 53% (umbral de viabilidad del TFG)")
    else:
        print(f"  ⚠️  Accuracy < 53% (umbral de viabilidad del TFG)")


# ══════════════════════════════════════════════════════════════════════
#  5. VISUALIZACIÓN
# ══════════════════════════════════════════════════════════════════════

def plot_backtest(df_all: pd.DataFrame,
                 df_high: pd.DataFrame,
                 metrics_all: dict,
                 metrics_high: dict,
                 save_path: str = 'backtest_results.png'):

    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(4, 2, figure=fig,
                             hspace=0.45, wspace=0.3,
                             height_ratios=[3, 1.5, 1.5, 1.5])

    # ── Panel 1 (ancho completo): Curvas de capital ───────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df_all.index,  df_all['bh_cumret'],
             color=BLUE,   lw=2.0, label='Buy & Hold', alpha=0.9)
    ax1.plot(df_all.index,  df_all['strategy_cumret'],
             color=ORANGE, lw=2.0, label='Estrategia (todas señales)')
    ax1.plot(df_high.index, df_high['strategy_cumret'],
             color=GREEN,  lw=1.8, label='Estrategia (alta confianza)',
             linestyle='--')
    ax1.axhline(1, color='#30363d', lw=0.8, linestyle=':')
    ax1.set_yscale('log')
    ax1.set_ylabel('Capital acumulado (escala log)', fontsize=10)
    ax1.set_title('Backtest — Estrategia técnica RSI + MACD + BB  vs  Buy & Hold',
                  fontsize=13, pad=10)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(axis='y')

    # Zonas de posición (largo=verde, corto=rojo)
    for i in range(len(df_all) - 1):
        sig = df_all['signal_active'].iloc[i]
        if sig == 1:
            ax1.axvspan(df_all.index[i], df_all.index[i+1],
                        alpha=0.04, color=GREEN, lw=0)
        elif sig == -1:
            ax1.axvspan(df_all.index[i], df_all.index[i+1],
                        alpha=0.04, color=RED, lw=0)

    # ── Panel 2: Drawdown comparado ───────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :], sharex=ax1)

    def rolling_dd(returns):
        cum  = returns.cumsum().apply(np.exp)
        peak = cum.cummax()
        return ((cum - peak) / peak) * 100

    dd_strat = rolling_dd(df_all['strategy_return'])
    dd_bh    = rolling_dd(df_all['bh_return'])
    ax2.fill_between(df_all.index, dd_strat, 0,
                     color=ORANGE, alpha=0.4, label='Estrategia')
    ax2.fill_between(df_all.index, dd_bh, 0,
                     color=BLUE,   alpha=0.25, label='Buy & Hold')
    ax2.set_ylabel('Drawdown (%)', fontsize=10)
    ax2.set_title('Drawdown', fontsize=10)
    ax2.legend(fontsize=8)

    # ── Panel 3 izq: RSI con señales ─────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax3.plot(df_all.index, df_all['rsi_14'],
             color=PURPLE, lw=1.2, label='RSI 14')
    ax3.axhline(70, color=RED,   lw=0.8, linestyle='--', alpha=0.7)
    ax3.axhline(30, color=GREEN, lw=0.8, linestyle='--', alpha=0.7)
    ax3.axhline(50, color='#30363d', lw=0.6, linestyle=':')
    ax3.set_ylabel('RSI', fontsize=10)
    ax3.set_title('RSI 14', fontsize=10)
    ax3.set_ylim(0, 100)
    ax3.legend(fontsize=8)

    # ── Panel 3 der: MACD ─────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1], sharex=ax1)
    ax4.plot(df_all.index, df_all['macd'],
             color=ORANGE, lw=1.2, label='MACD')
    ax4.plot(df_all.index, df_all['macd_signal'],
             color=BLUE,   lw=1.2, label='Signal', linestyle='--')
    ax4.axhline(0, color='#30363d', lw=0.6)
    macd_hist = df_all['macd'] - df_all['macd_signal']
    ax4.bar(df_all.index, macd_hist,
            color=np.where(macd_hist >= 0, GREEN, RED),
            alpha=0.4, width=1)
    ax4.set_ylabel('MACD', fontsize=10)
    ax4.set_title('MACD (12, 26, 9)', fontsize=10)
    ax4.legend(fontsize=8)

    # ── Panel 4 izq: Tabla de métricas ───────────────────────────────
    ax5 = fig.add_subplot(gs[3, 0])
    ax5.axis('off')
    tabla_data = [
        ['Métrica',            'Todas señales',                   'Alta conf.',                    'B&H'],
        ['Retorno total (%)',   f"{metrics_all['total_return_strategy']:.1f}",
                                f"{metrics_high['total_return_strategy']:.1f}",
                                f"{metrics_all['total_return_bh']:.1f}"],
        ['Sharpe Ratio',        f"{metrics_all['sharpe_strategy']:.3f}",
                                f"{metrics_high['sharpe_strategy']:.3f}",
                                f"{metrics_all['sharpe_bh']:.3f}"],
        ['Max Drawdown (%)',     f"{metrics_all['max_drawdown_strategy']:.1f}",
                                f"{metrics_high['max_drawdown_strategy']:.1f}",
                                f"{metrics_all['max_drawdown_bh']:.1f}"],
        ['Accuracy (%)',         f"{metrics_all['accuracy_signals']:.1f}",
                                f"{metrics_high['accuracy_signals']:.1f}", '—'],
        ['Nº operaciones',       f"{metrics_all['n_trades']}",
                                f"{metrics_high['n_trades']}", '—'],
        ['Días en mercado',      f"{metrics_all['days_in_market']}",
                                f"{metrics_high['days_in_market']}",
                                f"{metrics_all['total_days']}"],
    ]
    t = ax5.table(cellText=tabla_data[1:], colLabels=tabla_data[0],
                  loc='center', cellLoc='center')
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.6)
    for (r, c), cell in t.get_celld().items():
        cell.set_facecolor('#161b22' if r > 0 else '#21262d')
        cell.set_edgecolor('#30363d')
        cell.set_text_props(color='#c9d1d9')
    ax5.set_title('Resumen de métricas', fontsize=10, pad=8)

    # ── Panel 4 der: Distribución de retornos diarios ─────────────────
    ax6 = fig.add_subplot(gs[3, 1])
    ax6.hist(df_all['strategy_return'].dropna() * 100,
             bins=60, color=ORANGE, alpha=0.7, label='Estrategia', density=True)
    ax6.hist(df_all['bh_return'].dropna() * 100,
             bins=60, color=BLUE,   alpha=0.4, label='Buy & Hold', density=True)
    ax6.axvline(0, color='white', lw=0.8, linestyle='--')
    ax6.set_xlabel('Retorno diario (%)', fontsize=9)
    ax6.set_ylabel('Densidad', fontsize=9)
    ax6.set_title('Distribución retornos diarios', fontsize=10)
    ax6.legend(fontsize=8)

    # Formatear eje X de todos los paneles compartidos
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)

    fig.suptitle(
        'Backtest — Estrategia Técnica RSI + MACD + Bollinger Bands  ·  BTC/USDT  ·  TFG 2025/26',
        fontsize=12, y=1.01, color='#c9d1d9'
    )

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'\n✅ Guardado: {save_path}')


# ══════════════════════════════════════════════════════════════════════
#  6. ANÁLISIS DE CONFIANZA
# ══════════════════════════════════════════════════════════════════════

def plot_confidence_analysis(df: pd.DataFrame,
                              save_path: str = 'backtest_confidence.png'):
    """
    Muestra cómo varía la accuracy y el retorno según el nivel de confianza.
    Este gráfico es directamente citable en la memoria del TFG.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    levels = ['high', 'medium', 'low']
    colors = [GREEN, ORANGE, RED]
    results = []

    for level in levels:
        mask = df['confidence'] == level
        sub  = df[mask].copy()
        if len(sub) == 0:
            results.append({'level': level, 'n': 0, 'accuracy': 0, 'return': 0})
            continue
        active = sub[sub['signal_active'] != 0]
        if len(active) > 0:
            correct = (
                ((active['signal_active'] == 1)  & (active['next_return'] > 0)) |
                ((active['signal_active'] == -1) & (active['next_return'] < 0))
            )
            acc = correct.sum() / len(active) * 100
        else:
            acc = 0
        ret = np.exp(sub['strategy_return'].sum()) - 1
        results.append({'level': level, 'n': len(sub),
                        'accuracy': acc, 'return': ret * 100})

    # Accuracy por nivel
    ax = axes[0]
    accs = [r['accuracy'] for r in results]
    bars = ax.bar(levels, accs, color=colors, width=0.5)
    ax.axhline(53, color='white', lw=1.2, linestyle='--',
               label='Umbral TFG (53%)')
    ax.axhline(50, color='#30363d', lw=0.8, linestyle=':')
    for bar, v in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{v:.1f}%', ha='center', fontsize=11, color='white')
    ax.set_title('Accuracy por nivel de confianza', fontsize=11)
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 75)
    ax.legend(fontsize=8)

    # Retorno acumulado por nivel
    ax2 = axes[1]
    rets = [r['return'] for r in results]
    bar_colors2 = [GREEN if v > 0 else RED for v in rets]
    bars2 = ax2.bar(levels, rets, color=bar_colors2, width=0.5, alpha=0.8)
    ax2.axhline(0, color='#30363d', lw=0.8)
    for bar, v in zip(bars2, rets):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + (1 if v >= 0 else -3),
                 f'{v:.1f}%', ha='center', fontsize=11, color='white')
    ax2.set_title('Retorno acumulado por nivel de confianza', fontsize=11)
    ax2.set_ylabel('Retorno total (%)')

    # Nº de señales por nivel
    ax3 = axes[2]
    ns = [r['n'] for r in results]
    ax3.bar(levels, ns, color=colors, width=0.5, alpha=0.8)
    for i, v in enumerate(ns):
        ax3.text(i, v + 2, str(v), ha='center', fontsize=11, color='white')
    ax3.set_title('Nº de señales por nivel de confianza', fontsize=11)
    ax3.set_ylabel('Nº días')

    fig.suptitle('Análisis de confianza de señales  ·  ¿Operar solo con alta confianza?',
                 fontsize=12, y=1.02)
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'✅ Guardado: {save_path}')


# ══════════════════════════════════════════════════════════════════════
#  7. PREDICCIONES ML Y BACKTEST CON UMBRAL DE CONFIANZA
# ══════════════════════════════════════════════════════════════════════

def generate_ml_predictions(df: pd.DataFrame,
                             min_train_days: int = 365,
                             test_days: int = 60) -> pd.DataFrame:
    """
    Walk-forward expandido: genera prob_up para TODOS los días después de
    los primeros min_train_days, sin data leakage.
    Cada split entrena con todos los datos anteriores al período de test.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        os.system("pip install xgboost")
        from xgboost import XGBClassifier
    from sklearn.preprocessing import StandardScaler

    feature_cols = [
        'rsi_14', 'macd', 'macd_signal',
        'bb_upper', 'bb_lower', 'sma_7', 'sma_30',
        'fear_greed', 'returns', 'sentiment_avg',
    ]

    df_clean = df.dropna(subset=['rsi_14', 'macd', 'sma_30', 'returns']).copy()
    df_clean['sentiment_avg'] = df_clean['sentiment_avg'].fillna(0.0)
    df_clean['fear_greed']    = df_clean['fear_greed'].fillna(df_clean['fear_greed'].median())

    n = len(df_clean)
    if n < min_train_days + test_days:
        print(f"⚠  Datos insuficientes para walk-forward "
              f"(necesita ≥ {min_train_days + test_days} días, tiene {n})")
        return pd.DataFrame()

    # Calcular número de splits necesarios para cubrir todos los días post-min_train
    n_splits = int(np.ceil((n - min_train_days) / test_days))
    print(f"\n🔄 Walk-forward expandido: {n_splits} splits × {test_days} días")
    print(f"   Datos totales: {n} días  |  Mín. entrenamiento: {min_train_days} días")
    print(f"   Cobertura: {df_clean.index[min_train_days].date()} → {df_clean.index[-1].date()}")

    all_preds = []
    for i in range(n_splits):
        train_end = min_train_days + i * test_days
        test_start = train_end
        test_end   = min(train_end + test_days, n)
        if test_start >= n:
            break

        train_df = df_clean.iloc[:train_end]
        test_df  = df_clean.iloc[test_start:test_end]

        X_train = train_df[feature_cols].values
        y_train = train_df['label'].values.astype(int)
        X_test  = test_df[feature_cols].values

        scaler  = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_train)
        X_te_sc = scaler.transform(X_test)

        model = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, eval_metric='logloss', verbosity=0,
        )
        model.fit(X_tr_sc, y_train)
        prob_up = model.predict_proba(X_te_sc)[:, 1]

        preds = pd.DataFrame({
            'prob_up': prob_up,
            'label':   test_df['label'].values,
        }, index=test_df.index)
        all_preds.append(preds)
        print(f"    Split {i+1:>2}/{n_splits}: "
              f"[{test_df.index[0].date()} → {test_df.index[-1].date()}]  "
              f"train={len(train_df)}d  prob_up media: {prob_up.mean():.3f}")

    return pd.concat(all_preds) if all_preds else pd.DataFrame()


def run_ml_threshold_analysis(df: pd.DataFrame,
                               predictions_df: pd.DataFrame,
                               thresholds: list,
                               commission: float = 0.001) -> tuple:
    """
    Para cada umbral de confianza ML:
      - Solo opera cuando prob_up > umbral  Y  RSI ∈ [35, 65]
      - Calcula: días operados, cobertura, accuracy, retorno total, Sharpe
    Devuelve (DataFrame de resultados, bh_ret, bh_sharpe).
    """
    merged = df.join(predictions_df[['prob_up']], how='inner')
    merged = merged.dropna(subset=['prob_up', 'rsi_14', 'returns'])

    merged['next_return'] = merged['returns'].shift(-1)
    merged = merged.dropna(subset=['next_return'])

    bh_ret    = np.exp(merged['next_return'].sum()) - 1
    bh_std    = merged['next_return'].std()
    bh_sharpe = (merged['next_return'].mean() / bh_std) * np.sqrt(365) if bh_std > 0 else 0

    print(f"\n{'═'*70}")
    print("BACKTEST ML — UMBRAL DE CONFIANZA + FILTRO RSI [35-65]")
    print(f"Período: {merged.index.min().date()} → {merged.index.max().date()}  "
          f"({len(merged)} días)")
    print(f"{'═'*70}")
    print(f"{'Umbral':>8}  {'Días op.':>9}  {'Cobertura':>10}  "
          f"{'Accuracy':>10}  {'Retorno%':>10}  {'Sharpe':>8}")
    print(f"{'─'*70}")

    results = []
    for thr in thresholds:
        rsi_ok = (merged['rsi_14'] >= 35) & (merged['rsi_14'] <= 65)
        signal = ((merged['prob_up'] > thr) & rsi_ok).astype(int)

        active   = merged[signal == 1]
        n_days   = len(active)
        coverage = n_days / len(merged) * 100

        if n_days == 0:
            print(f"  {thr:.2f}   {'0':>9}  {'0.0%':>10}  {'—':>10}  {'—':>10}  {'—':>8}")
            results.append({
                'threshold': thr, 'days_traded': 0, 'coverage_pct': 0.0,
                'accuracy_pct': None, 'total_return_pct': 0.0, 'sharpe': None,
            })
            continue

        correct  = (active['next_return'] > 0).sum()
        accuracy = correct / n_days

        strat_ret       = signal * merged['next_return']
        pos_change      = signal.diff().abs().fillna(0)
        net_ret         = strat_ret - pos_change * commission

        total_ret = np.exp(net_ret.sum()) - 1
        net_std   = net_ret.std()
        sharpe    = (net_ret.mean() / net_std) * np.sqrt(365) if net_std > 0 else 0

        print(f"  {thr:.2f}   {n_days:>9}  {coverage:>9.1f}%  "
              f"{accuracy*100:>9.1f}%  {total_ret*100:>9.1f}%  {sharpe:>7.3f}")

        results.append({
            'threshold':        thr,
            'days_traded':      n_days,
            'coverage_pct':     round(coverage, 1),
            'accuracy_pct':     round(accuracy * 100, 2),
            'total_return_pct': round(total_ret * 100, 2),
            'sharpe':           round(sharpe, 3),
        })

    print(f"{'─'*70}")
    print(f"  {'B&H':>6}   {len(merged):>9}  {'100.0%':>10}  "
          f"{'—':>10}  {bh_ret*100:>9.1f}%  {bh_sharpe:>7.3f}")
    print(f"{'═'*70}")

    df_results = pd.DataFrame(results)
    df_results.to_csv('results/backtest_threshold_analysis.csv', index=False)
    print("✅ Resultados → results/backtest_threshold_analysis.csv")

    return df_results, bh_ret, bh_sharpe


def plot_threshold_analysis(threshold_results: pd.DataFrame,
                             bh_ret: float,
                             bh_sharpe: float,
                             save_path: str = 'backtest_threshold_ml.png'):
    """Gráfica comparativa de umbrales de confianza ML vs Buy & Hold."""
    import matplotlib
    matplotlib.use('Agg')

    df = threshold_results.dropna(subset=['accuracy_pct'])
    if df.empty:
        print("⚠  Sin datos para graficar (todos los umbrales sin señales)")
        return

    thresholds = [str(t) for t in df['threshold']]
    x = np.arange(len(thresholds))
    w = 0.5

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        'Backtest ML — Umbral de confianza + filtro RSI [35–65]  ·  BTC/USDT  ·  TFG 2025/26',
        fontsize=12, color='#c9d1d9',
    )

    # ── Panel 1: Accuracy ──────────────────────────────────────────────
    ax = axes[0]
    colors = [GREEN if v > 53 else RED for v in df['accuracy_pct']]
    bars   = ax.bar(x, df['accuracy_pct'], width=w, color=colors, alpha=0.85)
    ax.axhline(53, color='white', lw=1.2, linestyle='--',
               label='Umbral TFG (53%)', alpha=0.8)
    ax.axhline(50, color='#30363d', lw=0.8, linestyle=':')
    for bar, val in zip(bars, df['accuracy_pct']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', fontsize=10, color='#e6edf3')
    ax.set_xticks(x)
    ax.set_xticklabels([f'≥{t}' for t in thresholds])
    ax.set_ylim(0, 80)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy por umbral', fontsize=11)
    ax.legend(fontsize=8, framealpha=0.3)
    ax.grid(True, alpha=0.3, axis='y')

    # ── Panel 2: Retorno total ─────────────────────────────────────────
    ax2 = axes[1]
    rets   = df['total_return_pct'].values
    colors2 = [GREEN if v > 0 else RED for v in rets]
    bars2   = ax2.bar(x, rets, width=w, color=colors2, alpha=0.85)
    ax2.axhline(bh_ret * 100, color=BLUE, lw=1.5, linestyle='--',
                label=f'Buy & Hold ({bh_ret*100:.1f}%)', alpha=0.9)
    ax2.axhline(0, color='#30363d', lw=0.8)
    for bar, val in zip(bars2, rets):
        offset = 1 if val >= 0 else -4
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                 f'{val:.1f}%', ha='center', fontsize=10, color='#e6edf3')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'≥{t}' for t in thresholds])
    ax2.set_ylabel('Retorno total (%)')
    ax2.set_title('Retorno acumulado vs B&H', fontsize=11)
    ax2.legend(fontsize=8, framealpha=0.3)
    ax2.grid(True, alpha=0.3, axis='y')

    # ── Panel 3: Sharpe Ratio ──────────────────────────────────────────
    ax3 = axes[2]
    sharpes   = df['sharpe'].values
    colors3   = [GREEN if v > bh_sharpe else ORANGE for v in sharpes]
    bars3     = ax3.bar(x, sharpes, width=w, color=colors3, alpha=0.85)
    ax3.axhline(bh_sharpe, color=BLUE, lw=1.5, linestyle='--',
                label=f'Buy & Hold ({bh_sharpe:.3f})', alpha=0.9)
    ax3.axhline(0, color='#30363d', lw=0.8)
    for bar, val in zip(bars3, sharpes):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', fontsize=10, color='#e6edf3')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'≥{t}' for t in thresholds])
    ax3.set_ylabel('Sharpe Ratio (anual)')
    ax3.set_title('Sharpe Ratio vs B&H', fontsize=11)
    ax3.legend(fontsize=8, framealpha=0.3)
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Gráfica → {save_path}")


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset',        default='BTC')
    parser.add_argument('--start',        default='2021-01-01')
    parser.add_argument('--end',          default=None)
    parser.add_argument('--no-short',     action='store_true')
    parser.add_argument('--commission',   type=float, default=0.001)
    parser.add_argument('--ml-threshold', action='store_true',
                        help='Ejecutar análisis de umbrales de confianza ML')
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"  BACKTEST — {args.asset}  |  {args.start} → {args.end or 'hoy'}")
    print(f"  Cortos: {'NO' if args.no_short else 'SÍ'}  |  "
          f"Comisión: {args.commission*100:.2f}%")
    print(f"{'='*60}\n")

    # 1. Cargar datos
    df = load_data(args.asset, args.start, args.end)

    # 2. Generar señales técnicas
    df = generate_signals(df, allow_short=not args.no_short)

    # 3. Backtest — todas las señales técnicas
    df_all  = run_backtest(df, commission=args.commission, high_conf_only=False)

    # 4. Backtest — solo alta confianza técnica
    df_high = run_backtest(df, commission=args.commission, high_conf_only=True)

    # 5. Métricas estrategia técnica
    metrics_all  = compute_metrics(df_all)
    metrics_high = compute_metrics(df_high)

    print_metrics(metrics_all,  high_conf=False)
    print_metrics(metrics_high, high_conf=True)

    # 6. Gráficas estrategia técnica
    plot_backtest(df_all, df_high, metrics_all, metrics_high,
                  save_path=f'backtest_{args.asset}_{args.start[:4]}.png')
    plot_confidence_analysis(df_all,
                  save_path=f'backtest_confidence_{args.asset}.png')

    # ── Análisis de umbrales ML ────────────────────────────────────────
    if args.ml_threshold:
        print(f"\n{'='*60}")
        print("  ANÁLISIS DE UMBRALES DE CONFIANZA ML")
        print(f"{'='*60}")

        # Cargar predicciones precomputadas solo si cubren el período completo
        pred_path = os.path.join(os.path.dirname(__file__),
                                 '..', 'results', 'ml_predictions.csv')
        predictions_df = pd.DataFrame()
        if os.path.exists(pred_path):
            cached = pd.read_csv(pred_path, index_col=0, parse_dates=True)
            # Reutilizar cache solo si cubre ≥ 80% del período disponible
            coverage = len(cached) / max(len(df) - 365, 1)
            if coverage >= 0.80:
                print(f"📂 Cargando predicciones ML desde {pred_path} "
                      f"({len(cached)} días, cobertura {coverage*100:.0f}%)")
                predictions_df = cached
            else:
                print(f"⚠  Cache insuficiente ({len(cached)} días, "
                      f"cobertura {coverage*100:.0f}%) — regenerando...")

        if predictions_df.empty:
            print("🔄 Generando predicciones ML con walk-forward expandido "
                  "(desde 2022)...")
            predictions_df = generate_ml_predictions(df, min_train_days=365)
            if not predictions_df.empty:
                predictions_df.to_csv(pred_path)
                print(f"✅ Predicciones guardadas → {pred_path} "
                      f"({len(predictions_df)} días)")

        if predictions_df.empty:
            print("❌ No se pudieron generar predicciones ML.")
        else:
            thresholds = [0.55, 0.60, 0.65, 0.70]
            thr_results, bh_ret, bh_sharpe = run_ml_threshold_analysis(
                df, predictions_df, thresholds, commission=args.commission
            )
            plot_threshold_analysis(
                thr_results, bh_ret, bh_sharpe,
                save_path=f'backtest_threshold_{args.asset}.png',
            )


if __name__ == '__main__':
    main()