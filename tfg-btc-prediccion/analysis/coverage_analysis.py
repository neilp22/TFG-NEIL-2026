# analysis/coverage_analysis.py
# Análisis detallado de cobertura de noticias en el corpus
# Muestra qué días, meses y años tienen pocas o ninguna noticia
# Uso: python analysis/coverage_analysis.py

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
plt.rcParams['font.family']      = 'monospace'

ACCENT = '#f7931a'
BLUE   = '#58a6ff'
GREEN  = '#3fb950'
RED    = '#f85149'
PURPLE = '#bc8cff'

os.makedirs('notebooks/figures', exist_ok=True)
os.makedirs('analysis', exist_ok=True)

engine = get_engine()

print("="*60)
print("ANÁLISIS DE COBERTURA DE NOTICIAS")
print("="*60)

# ─────────────────────────────────────────────────────────────────────────────
# 1. COBERTURA POR AÑO
# ─────────────────────────────────────────────────────────────────────────────

print("\n📅 COBERTURA POR AÑO:")
print("-"*60)

with engine.connect() as conn:
    df_year = pd.read_sql(text("""
        SELECT
            EXTRACT(YEAR FROM date)::int AS year,
            COUNT(*) AS total_dias,
            SUM(CASE WHEN sentiment_avg IS NOT NULL THEN 1 ELSE 0 END)
                AS dias_con_sentiment,
            SUM(CASE WHEN sentiment_avg IS NULL THEN 1 ELSE 0 END)
                AS dias_sin_sentiment,
            ROUND(AVG(sentiment_count), 1) AS media_noticias_dia,
            MAX(sentiment_count) AS max_noticias_dia,
            ROUND(100.0 *
                SUM(CASE WHEN sentiment_avg IS NOT NULL THEN 1 ELSE 0 END)
                / COUNT(*), 1) AS pct_cobertura
        FROM daily_features
        WHERE asset = 'BTC'
        GROUP BY year
        ORDER BY year
    """), conn)

print(f"{'Año':<6} {'Días totales':<14} {'Con noticia':<13} "
      f"{'Sin noticia':<13} {'Media/día':<11} {'Max/día':<9} {'Cobertura'}")
print("-"*80)
for _, r in df_year.iterrows():
    bar_len = int(r['pct_cobertura'] / 5)
    bar = '█' * bar_len + '░' * (20 - bar_len)
    status = '✅' if r['pct_cobertura'] >= 70 else (
             '⚠️ ' if r['pct_cobertura'] >= 30 else '❌')
    print(f"{int(r['year']):<6} {int(r['total_dias']):<14} "
          f"{int(r['dias_con_sentiment']):<13} "
          f"{int(r['dias_sin_sentiment']):<13} "
          f"{float(r['media_noticias_dia']) if pd.notna(r['media_noticias_dia']) else 0.0:<11.1f} "
          f"{int(r['max_noticias_dia']) if pd.notna(r['max_noticias_dia']) else 0:<9} "
          f"{r['pct_cobertura']:.1f}% {status} {bar}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. COBERTURA POR MES
# ─────────────────────────────────────────────────────────────────────────────

print("\n\n📅 COBERTURA POR MES (solo meses con <50% cobertura):")
print("-"*60)

with engine.connect() as conn:
    df_month = pd.read_sql(text("""
        SELECT
            EXTRACT(YEAR FROM date)::int AS year,
            EXTRACT(MONTH FROM date)::int AS month,
            TO_CHAR(date, 'YYYY-MM') AS periodo,
            COUNT(*) AS total_dias,
            SUM(CASE WHEN sentiment_avg IS NOT NULL THEN 1 ELSE 0 END)
                AS dias_con_sentiment,
            ROUND(100.0 *
                SUM(CASE WHEN sentiment_avg IS NOT NULL THEN 1 ELSE 0 END)
                / COUNT(*), 1) AS pct_cobertura,
            COALESCE(ROUND(AVG(sentiment_count), 0), 0)
                AS media_noticias_dia
        FROM daily_features
        WHERE asset = 'BTC'
        GROUP BY year, month, periodo
        ORDER BY year, month
    """), conn)

df_bad = df_month[df_month['pct_cobertura'] < 50].copy()
if df_bad.empty:
    print("  ✅ Todos los meses tienen >50% de cobertura")
else:
    print(f"{'Período':<12} {'Días':<7} {'Con noticia':<13} "
          f"{'Cobertura':<12} {'Media/día'}")
    print("-"*55)
    for _, r in df_bad.iterrows():
        status = '❌' if r['pct_cobertura'] < 10 else '⚠️ '
        print(f"  {r['periodo']:<10} {int(r['total_dias']):<7} "
              f"{int(r['dias_con_sentiment']):<13} "
              f"{r['pct_cobertura']:.1f}% {status}    "
              f"{int(r['media_noticias_dia'])} not/día")


# ─────────────────────────────────────────────────────────────────────────────
# 3. RESUMEN GLOBAL
# ─────────────────────────────────────────────────────────────────────────────

print("\n\n📊 RESUMEN GLOBAL:")
print("-"*60)

with engine.connect() as conn:
    total = pd.read_sql(text("""
        SELECT
            COUNT(*) AS total_dias,
            SUM(CASE WHEN sentiment_avg IS NOT NULL THEN 1 ELSE 0 END)
                AS dias_con_sentiment,
            SUM(CASE WHEN sentiment_avg IS NULL THEN 1 ELSE 0 END)
                AS dias_sin_sentiment,
            MIN(date) AS desde,
            MAX(date) AS hasta,
            ROUND(100.0 *
                SUM(CASE WHEN sentiment_avg IS NOT NULL THEN 1 ELSE 0 END)
                / COUNT(*), 1) AS pct_global
        FROM daily_features
        WHERE asset = 'BTC'
    """), conn).iloc[0]

    fuentes = pd.read_sql(text("""
        SELECT source,
               COUNT(*) AS total_textos,
               MIN(timestamp)::date AS desde,
               MAX(timestamp)::date AS hasta
        FROM raw_texts
        WHERE asset = 'BTC'
        GROUP BY source
        ORDER BY total_textos DESC
    """), conn)

print(f"  Período total:       {total['desde']} → {total['hasta']}")
print(f"  Días totales:        {int(total['total_dias'])}")
print(f"  Con sentimiento:     {int(total['dias_con_sentiment'])} "
      f"({total['pct_global']}%)")
print(f"  Sin sentimiento:     {int(total['dias_sin_sentiment'])} "
      f"({100 - float(total['pct_global']):.1f}%)")

print(f"\n  Textos por fuente:")
for _, r in fuentes.iterrows():
    print(f"    {r['source']:<30} {int(r['total_textos']):>7} textos  "
          f"| {r['desde']} → {r['hasta']}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. HUECOS CONSECUTIVOS SIN NOTICIAS
# ─────────────────────────────────────────────────────────────────────────────

print("\n\n🔍 PERÍODOS SIN NOTICIAS (huecos de >30 días consecutivos):")
print("-"*60)

with engine.connect() as conn:
    df_all = pd.read_sql(text("""
        SELECT date,
               CASE WHEN sentiment_avg IS NOT NULL THEN 1 ELSE 0 END
                   AS tiene_noticia
        FROM daily_features
        WHERE asset = 'BTC'
        ORDER BY date
    """), conn)

df_all['date'] = pd.to_datetime(df_all['date'])
df_all = df_all.sort_values('date').reset_index(drop=True)

# Detectar huecos consecutivos
gaps = []
gap_start = None
gap_len = 0

for _, row in df_all.iterrows():
    if row['tiene_noticia'] == 0:
        if gap_start is None:
            gap_start = row['date']
        gap_len += 1
    else:
        if gap_start is not None and gap_len >= 30:
            gaps.append({
                'inicio': gap_start.date(),
                'fin': (row['date'] - pd.Timedelta(days=1)).date(),
                'dias': gap_len
            })
        gap_start = None
        gap_len = 0

# Cerrar último hueco si existe
if gap_start is not None and gap_len >= 30:
    gaps.append({
        'inicio': gap_start.date(),
        'fin': df_all['date'].iloc[-1].date(),
        'dias': gap_len
    })

if not gaps:
    print("  ✅ No hay huecos de más de 30 días consecutivos")
else:
    gaps_df = pd.DataFrame(gaps).sort_values('dias', ascending=False)
    print(f"  {'Inicio':<15} {'Fin':<15} {'Días sin noticia'}")
    print("  " + "-"*45)
    for _, g in gaps_df.iterrows():
        priority = "🔴 CRÍTICO" if g['dias'] > 180 else (
                   "🟡 IMPORTANTE" if g['dias'] > 90 else "🟢 Menor")
        print(f"  {str(g['inicio']):<15} {str(g['fin']):<15} "
              f"{int(g['dias']):<18} {priority}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. GRÁFICAS
# ─────────────────────────────────────────────────────────────────────────────

print("\n\nGenerando gráficas...")

fig, axes = plt.subplots(3, 1, figsize=(16, 12))
fig.suptitle('Análisis de Cobertura del Corpus de Noticias — BTC',
             fontsize=14, color='#e6edf3', y=0.99)

# ── Panel 1: Cobertura diaria (heatmap por año) ───────────────────────────────
ax1 = axes[0]

with engine.connect() as conn:
    df_daily = pd.read_sql(text("""
        SELECT date,
               COALESCE(sentiment_count, 0) AS n_noticias,
               CASE WHEN sentiment_avg IS NOT NULL THEN 1 ELSE 0 END
                   AS tiene_noticia
        FROM daily_features
        WHERE asset = 'BTC'
        ORDER BY date
    """), conn)

df_daily['date'] = pd.to_datetime(df_daily['date'])

# Colorear por número de noticias
colors = []
for _, r in df_daily.iterrows():
    n = r['n_noticias']
    if n == 0:
        colors.append(RED)
    elif n < 3:
        colors.append('#ffa500')
    elif n < 10:
        colors.append(BLUE)
    else:
        colors.append(GREEN)

ax1.bar(df_daily['date'], df_daily['n_noticias'],
        color=colors, alpha=0.8, width=1)
ax1.set_ylabel('Noticias por día', fontsize=10)
ax1.set_title('Noticias por día (rojo = sin noticia, verde = >10 noticias)',
              fontsize=10)
ax1.grid(True, alpha=0.2, axis='y')

from matplotlib.patches import Patch
legend = [
    Patch(color=RED,      label='0 noticias'),
    Patch(color='#ffa500',label='1-2 noticias'),
    Patch(color=BLUE,     label='3-9 noticias'),
    Patch(color=GREEN,    label='≥10 noticias'),
]
ax1.legend(handles=legend, fontsize=8, framealpha=0.3, loc='upper left')

# ── Panel 2: Cobertura mensual (%) ────────────────────────────────────────────
ax2 = axes[1]

df_month['fecha'] = pd.to_datetime(df_month['periodo'] + '-01')
bar_colors = [GREEN if p >= 70 else (BLUE if p >= 30 else RED)
              for p in df_month['pct_cobertura']]

ax2.bar(df_month['fecha'], df_month['pct_cobertura'],
        color=bar_colors, alpha=0.85, width=25)
ax2.axhline(70, color=GREEN, linestyle='--', linewidth=0.8,
            alpha=0.6, label='Objetivo 70%')
ax2.axhline(30, color='#ffa500', linestyle='--', linewidth=0.8,
            alpha=0.6, label='Umbral mínimo 30%')
ax2.set_ylabel('% días con noticia', fontsize=10)
ax2.set_ylim(0, 110)
ax2.set_title('Cobertura mensual (% de días con al menos 1 noticia)',
              fontsize=10)
ax2.legend(fontsize=8, framealpha=0.3)
ax2.grid(True, alpha=0.2, axis='y')
ax2.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, p: f'{x:.0f}%'))

# ── Panel 3: Cobertura anual (barras apiladas) ────────────────────────────────
ax3 = axes[2]

x = np.arange(len(df_year))
bars_con = ax3.bar(x, df_year['dias_con_sentiment'],
                   color=GREEN, alpha=0.8, label='Días con noticia')
bars_sin = ax3.bar(x, df_year['dias_sin_sentiment'],
                   bottom=df_year['dias_con_sentiment'],
                   color=RED, alpha=0.8, label='Días sin noticia')

for i, (_, r) in enumerate(df_year.iterrows()):
    ax3.text(i, r['total_dias'] + 5, f"{r['pct_cobertura']:.0f}%",
             ha='center', va='bottom', fontsize=9, color='#e6edf3')

ax3.set_xticks(x)
ax3.set_xticklabels([str(int(y)) for y in df_year['year']], fontsize=9)
ax3.set_ylabel('Días', fontsize=10)
ax3.set_title('Cobertura anual: días con y sin noticia', fontsize=10)
ax3.legend(fontsize=9, framealpha=0.3)
ax3.grid(True, alpha=0.2, axis='y')

for ax in [ax1, ax2]:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())

plt.tight_layout()
plt.savefig('notebooks/figures/08_coverage_analysis.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  → Guardada: notebooks/figures/08_coverage_analysis.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. RECOMENDACIONES
# ─────────────────────────────────────────────────────────────────────────────

print("\n\n💡 RECOMENDACIONES PARA COMPLETAR EL CORPUS:")
print("="*60)

for _, r in df_year.iterrows():
    pct = float(r['pct_cobertura'])
    year = int(r['year'])
    if pct < 10:
        print(f"  ❌ {year}: {pct:.1f}% cobertura — CRÍTICO. "
              f"Buscar dataset específico para este año.")
    elif pct < 50:
        print(f"  ⚠️  {year}: {pct:.1f}% cobertura — "
              f"Cargar Pushshift o dataset Kaggle adicional.")
    elif pct < 80:
        print(f"  🔵 {year}: {pct:.1f}% cobertura — "
              f"Aceptable. Se puede mejorar con CryptoPanic.")
    else:
        print(f"  ✅ {year}: {pct:.1f}% cobertura — Buena cobertura.")

print(f"\n  Cobertura global actual: {total['pct_global']}%")
print(f"  Objetivo recomendado:    >70%")
print(f"  Días que faltan cubrir:  {int(total['dias_sin_sentiment'])}")