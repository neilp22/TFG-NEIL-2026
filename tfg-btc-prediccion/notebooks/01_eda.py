# notebooks/01_eda.py
# Análisis Exploratorio de Datos (EDA) para el TFG
# Genera 4 gráficas para el Informe Seguimiento 1
# Uso: python notebooks/01_eda.py

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sqlalchemy import text
from db.db_utils import get_engine

# ── Configuración visual ──────────────────────────────────────────────────────
plt.rcParams['figure.facecolor'] = '#0d1117'
plt.rcParams['axes.facecolor']   = '#161b22'
plt.rcParams['axes.edgecolor']   = '#30363d'
plt.rcParams['text.color']       = '#e6edf3'
plt.rcParams['axes.labelcolor']  = '#e6edf3'
plt.rcParams['xtick.color']      = '#8b949e'
plt.rcParams['ytick.color']      = '#8b949e'
plt.rcParams['grid.color']       = '#21262d'
plt.rcParams['grid.linewidth']   = 0.8
plt.rcParams['font.family']      = 'monospace'

ACCENT   = '#f7931a'   # naranja Bitcoin
BLUE     = '#58a6ff'
GREEN    = '#3fb950'
RED      = '#f85149'
PURPLE   = '#bc8cff'

os.makedirs('notebooks/figures', exist_ok=True)

# ── Cargar datos ──────────────────────────────────────────────────────────────
print("Conectando a la base de datos...")
engine = get_engine()

with engine.connect() as conn:
    df = pd.read_sql(text("""
        SELECT date, close, returns, label,
               rsi_14, macd, bb_upper, bb_lower,
               sma_7, sma_30,
               sentiment_avg, sentiment_std, sentiment_count,
               fear_greed
        FROM daily_features
        WHERE asset = 'BTC'
        ORDER BY date ASC
    """), conn)

df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

print(f"Dataset cargado: {len(df)} días | {df.index.min().date()} → {df.index.max().date()}")
print(f"Columnas con nulos:")
nulls = df.isnull().sum()
for col, n in nulls[nulls > 0].items():
    print(f"  {col}: {n} nulos ({n/len(df)*100:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICA 1: Precio histórico BTC con medias móviles
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerando Gráfica 1: Precio histórico...")

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                          gridspec_kw={'height_ratios': [3, 1, 1]})
fig.suptitle('Bitcoin (BTC/USDT) — Análisis Técnico Histórico',
             fontsize=14, color='#e6edf3', y=0.98)

# Panel 1: Precio + SMA
ax1 = axes[0]
ax1.plot(df.index, df['close'], color=ACCENT, linewidth=1.0, label='Precio cierre', zorder=3)
ax1.plot(df.index, df['sma_7'],  color=BLUE,   linewidth=0.8, alpha=0.8, label='SMA 7d')
ax1.plot(df.index, df['sma_30'], color=PURPLE, linewidth=0.8, alpha=0.8, label='SMA 30d')

# Bandas de Bollinger
bb_valid = df[['bb_upper', 'bb_lower', 'close']].dropna()
ax1.fill_between(bb_valid.index, bb_valid['bb_upper'], bb_valid['bb_lower'],
                 alpha=0.08, color=BLUE, label='Bollinger Bands')

ax1.set_ylabel('Precio (USD)', fontsize=10)
ax1.legend(loc='upper left', fontsize=8, framealpha=0.3)
ax1.grid(True, alpha=0.3)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Panel 2: RSI
ax2 = axes[1]
rsi_valid = df['rsi_14'].dropna()
ax2.plot(rsi_valid.index, rsi_valid, color=BLUE, linewidth=0.9)
ax2.axhline(70, color=RED,   linestyle='--', alpha=0.6, linewidth=0.8)
ax2.axhline(30, color=GREEN, linestyle='--', alpha=0.6, linewidth=0.8)
ax2.axhline(50, color='#8b949e', linestyle=':', alpha=0.4, linewidth=0.6)
ax2.fill_between(rsi_valid.index, rsi_valid, 70,
                 where=(rsi_valid >= 70), alpha=0.2, color=RED)
ax2.fill_between(rsi_valid.index, rsi_valid, 30,
                 where=(rsi_valid <= 30), alpha=0.2, color=GREEN)
ax2.set_ylabel('RSI (14)', fontsize=10)
ax2.set_ylim(0, 100)
ax2.grid(True, alpha=0.3)

# Panel 3: Sentiment
ax3 = axes[2]
sent_valid = df['sentiment_avg'].dropna()
colors_sent = [GREEN if v > 0 else RED for v in sent_valid]
ax3.bar(sent_valid.index, sent_valid, color=colors_sent, alpha=0.7, width=1)
ax3.axhline(0, color='#8b949e', linewidth=0.8, linestyle='--')
ax3.set_ylabel('Sentiment\n(FinBERT)', fontsize=10)
ax3.set_ylim(-1, 1)
ax3.grid(True, alpha=0.3)

ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax3.xaxis.set_major_locator(mdates.YearLocator())
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=0)

plt.tight_layout()
plt.savefig('notebooks/figures/01_precio_tecnico_sentiment.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Guardada: notebooks/figures/01_precio_tecnico_sentiment.png")


# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICA 2: Distribución de la variable objetivo (label)
# ─────────────────────────────────────────────────────────────────────────────
print("Generando Gráfica 2: Distribución del label...")

label_valid = df['label'].dropna()
counts = label_valid.value_counts().sort_index()
pct_up   = label_valid.mean() * 100
pct_down = 100 - pct_up

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Distribución de la Variable Objetivo (label)',
             fontsize=13, color='#e6edf3')

# Barras
ax = axes[0]
bars = ax.bar(['Bajada (0)', 'Subida (1)'],
              [counts.get(0, 0), counts.get(1, 0)],
              color=[RED, GREEN], alpha=0.85, width=0.5)
ax.set_ylabel('Número de días', fontsize=10)
ax.set_title('Conteo de días alcistas vs bajistas', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, [counts.get(0, 0), counts.get(1, 0)]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            f'{val}\n({val/len(label_valid)*100:.1f}%)',
            ha='center', va='bottom', fontsize=10, color='#e6edf3')

# Serie temporal del label acumulado
ax2 = axes[1]
rolling_pct = label_valid.rolling(90).mean() * 100
ax2.plot(rolling_pct.index, rolling_pct, color=ACCENT, linewidth=1.2)
ax2.axhline(50, color='#8b949e', linestyle='--', linewidth=0.8, alpha=0.6)
ax2.fill_between(rolling_pct.index, rolling_pct, 50,
                 where=(rolling_pct >= 50), alpha=0.15, color=GREEN)
ax2.fill_between(rolling_pct.index, rolling_pct, 50,
                 where=(rolling_pct < 50),  alpha=0.15, color=RED)
ax2.set_ylabel('% días alcistas (media móvil 90d)', fontsize=10)
ax2.set_title('Tendencia alcista a lo largo del tiempo', fontsize=10)
ax2.set_ylim(20, 80)
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig('notebooks/figures/02_distribucion_label.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Guardada: notebooks/figures/02_distribucion_label.png")


# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICA 3: Correlación de features con el label
# ─────────────────────────────────────────────────────────────────────────────
print("Generando Gráfica 3: Correlación de features...")

features = ['rsi_14', 'macd', 'sma_7', 'sma_30',
            'bb_upper', 'bb_lower', 'sentiment_avg',
            'fear_greed', 'returns']

corr = df[features + ['label']].corr()['label'].drop('label').sort_values()

fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle('Correlación de Features con la Variable Objetivo (label)',
             fontsize=13, color='#e6edf3')

colors = [GREEN if v > 0 else RED for v in corr.values]
bars = ax.barh(corr.index, corr.values, color=colors, alpha=0.85, height=0.6)

ax.axvline(0, color='#8b949e', linewidth=1.0)
ax.axvline(0.05,  color='#8b949e', linewidth=0.6, linestyle=':', alpha=0.5)
ax.axvline(-0.05, color='#8b949e', linewidth=0.6, linestyle=':', alpha=0.5)

for bar, val in zip(bars, corr.values):
    ax.text(val + (0.002 if val >= 0 else -0.002), bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center',
            ha='left' if val >= 0 else 'right',
            fontsize=9, color='#e6edf3')

ax.set_xlabel('Correlación de Pearson con label', fontsize=10)
ax.grid(True, alpha=0.3, axis='x')
ax.set_xlim(corr.min() - 0.05, corr.max() + 0.08)

# Anotación para el sentiment
if 'sentiment_avg' in corr.index:
    sent_corr = corr['sentiment_avg']
    color_note = GREEN if sent_corr > 0 else RED
    ax.annotate('← Sentiment\n   (FinBERT)',
                xy=(sent_corr, list(corr.index).index('sentiment_avg')),
                xytext=(sent_corr + 0.04, list(corr.index).index('sentiment_avg') + 0.5),
                fontsize=8, color=color_note,
                arrowprops=dict(arrowstyle='->', color=color_note, lw=0.8))

plt.tight_layout()
plt.savefig('notebooks/figures/03_correlacion_features.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Guardada: notebooks/figures/03_correlacion_features.png")


# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICA 4: Sentiment vs Precio — relación visual
# ─────────────────────────────────────────────────────────────────────────────
print("Generando Gráfica 4: Sentiment vs Precio...")

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                          gridspec_kw={'height_ratios': [2, 1]})
fig.suptitle('Sentimiento de Noticias (FinBERT) vs Precio BTC',
             fontsize=13, color='#e6edf3')

# Panel superior: Precio
ax1 = axes[0]
ax1.plot(df.index, df['close'], color=ACCENT, linewidth=1.0, label='Precio BTC')
ax1.set_ylabel('Precio (USD)', fontsize=10)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9, framealpha=0.3)

# Panel inferior: Sentiment media móvil 7 días
ax2 = axes[1]
sent = df['sentiment_avg'].fillna(0)
sent_ma7 = sent.rolling(7, min_periods=1).mean()

ax2.bar(df.index, sent, color=[GREEN if v > 0 else RED for v in sent],
        alpha=0.3, width=1, label='Sentiment diario')
ax2.plot(df.index, sent_ma7, color=BLUE, linewidth=1.2,
         label='Media móvil 7d')
ax2.axhline(0, color='#8b949e', linewidth=0.8, linestyle='--')
ax2.set_ylabel('Compound Score\n(FinBERT)', fontsize=10)
ax2.set_ylim(-1, 1)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9, framealpha=0.3)

ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('notebooks/figures/04_sentiment_vs_precio.png', dpi=150, bbox_inches='tight')
plt.close()
print("  → Guardada: notebooks/figures/04_sentiment_vs_precio.png")


# ─────────────────────────────────────────────────────────────────────────────
# RESUMEN ESTADÍSTICO
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("RESUMEN ESTADÍSTICO DEL DATASET")
print("="*55)
print(f"  Período:          {df.index.min().date()} → {df.index.max().date()}")
print(f"  Total días:       {len(df)}")
print(f"  Días alcistas:    {int(label_valid.sum())} ({pct_up:.1f}%)")
print(f"  Días bajistas:    {int((label_valid == 0).sum())} ({pct_down:.1f}%)")
print(f"  Precio mín:       ${df['close'].min():,.0f}")
print(f"  Precio máx:       ${df['close'].max():,.0f}")
print(f"  Días con sentiment: {df['sentiment_avg'].notna().sum()} ({df['sentiment_avg'].notna().mean()*100:.1f}%)")
print(f"  Días con fear_greed:{df['fear_greed'].notna().sum()} ({df['fear_greed'].notna().mean()*100:.1f}%)")

if 'sentiment_avg' in corr.index:
    print(f"\n  Correlación sentiment_avg → label: {corr['sentiment_avg']:.4f}")
if 'fear_greed' in corr.index:
    print(f"  Correlación fear_greed    → label: {corr['fear_greed']:.4f}")
if 'rsi_14' in corr.index:
    print(f"  Correlación rsi_14        → label: {corr['rsi_14']:.4f}")

print("\n✅ Las 4 gráficas están en notebooks/figures/")
print("   Úsalas directamente en el Informe Seguimiento 1")