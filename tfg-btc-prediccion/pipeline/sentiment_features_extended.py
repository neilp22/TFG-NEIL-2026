"""
sentiment_features_extended.py
================================
Añade las features de sentimiento ampliadas a daily_features.

Nuevas columnas:
  - sentiment_7d        : media móvil 7 días del compound_score
  - sentiment_14d       : media móvil 14 días del compound_score
  - sentiment_momentum  : diferencia 7d - 14d (¿está mejorando el sentimiento?)
  - sentiment_7d_std    : desviación estándar 7 días (incertidumbre/dispersión)
  - news_volume_7d      : media móvil 7 días del nº de textos procesados por día

Uso:
  python sentiment_features_extended.py            # procesa BTC
  python sentiment_features_extended.py --asset ETH  # otro activo
  python sentiment_features_extended.py --dry-run    # solo muestra, no escribe en BD

Neul Padras · TFG 2025/26
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from sqlalchemy import text

# ── Permite ejecutar desde cualquier carpeta del proyecto ──────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from db.db_utils import get_engine


# ══════════════════════════════════════════════════════════════════════
#  PASO 1: Migración — añadir columnas nuevas si no existen
# ══════════════════════════════════════════════════════════════════════

NEW_COLUMNS = {
    'sentiment_7d':       'NUMERIC(6,4)',
    'sentiment_14d':      'NUMERIC(6,4)',
    'sentiment_momentum': 'NUMERIC(6,4)',
    'sentiment_7d_std':   'NUMERIC(6,4)',
    'news_volume_7d':     'NUMERIC(10,2)',
}

def migrate_columns(engine):
    """
    Añade las columnas nuevas a daily_features si no existen.
    Es seguro ejecutarlo varias veces: usa IF NOT EXISTS.
    """
    with engine.begin() as conn:
        for col, dtype in NEW_COLUMNS.items():
            conn.execute(text(f"""
                ALTER TABLE daily_features
                ADD COLUMN IF NOT EXISTS {col} {dtype}
            """))
    print(f"✅ Migración OK — columnas verificadas: {list(NEW_COLUMNS.keys())}")


# ══════════════════════════════════════════════════════════════════════
#  PASO 2: Carga de datos desde daily_features
# ══════════════════════════════════════════════════════════════════════

def load_daily_features(engine, asset: str) -> pd.DataFrame:
    """
    Carga date, sentiment_avg y sentiment_count ordenados por fecha.
    Solo necesitamos estas columnas para calcular las nuevas features.
    """
    with engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT
                date,
                sentiment_avg,
                sentiment_count
            FROM daily_features
            WHERE asset = :asset
            ORDER BY date ASC
        """), conn, params={'asset': asset})

    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    total = len(df)
    con_sent = df['sentiment_avg'].notna().sum()
    print(f"📊 Cargadas {total} filas para {asset}  "
          f"({con_sent} días con sentiment_avg, "
          f"{total - con_sent} sin datos textuales)")
    return df


# ══════════════════════════════════════════════════════════════════════
#  PASO 3: Cálculo de las nuevas features
# ══════════════════════════════════════════════════════════════════════

def compute_extended_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula las 5 features de sentimiento ampliadas.

    Notas de diseño:
    - min_periods=1 en los rolling para no perder las primeras filas
      (aunque con pocos datos el valor será poco fiable, es mejor que NULL).
      Si prefieres NaN en los primeros días, cambia a min_periods=7/14.
    - sentiment_momentum puede ser NaN si sentiment_14d es NaN,
      pero eso solo afecta a los primeros 14 días del dataset.
    """

    # ── Media móvil 7 días ─────────────────────────────────────────────
    # Justificación: el lag analysis mostró que t-4 y t-8 tienen más
    # correlación que t o t-1. La media de 7 días captura ese efecto
    # y suaviza el ruido de días con pocos textos.
    df['sentiment_7d'] = (
        df['sentiment_avg']
        .rolling(window=7, min_periods=1)
        .mean()
        .round(4)
    )

    # ── Media móvil 14 días ────────────────────────────────────────────
    # Ventana más larga para capturar el régimen de sentimiento
    # a medio plazo (ciclos de 2 semanas en narrativa mediática).
    df['sentiment_14d'] = (
        df['sentiment_avg']
        .rolling(window=14, min_periods=1)
        .mean()
        .round(4)
    )

    # ── Momentum del sentimiento ───────────────────────────────────────
    # ¿Está mejorando o empeorando el sentimiento?
    # Positivo → la narrativa reciente es mejor que la de 2 semanas.
    # Negativo → el sentimiento se está deteriorando.
    # Este feature captura el cambio de régimen (ej: paso de mercado
    # bajista a alcista en la narrativa mediática).
    df['sentiment_momentum'] = (
        (df['sentiment_7d'] - df['sentiment_14d'])
        .round(4)
    )

    # ── Desviación estándar 7 días ─────────────────────────────────────
    # Mide la dispersión/incertidumbre del sentimiento.
    # Alta std → días con opiniones muy divididas (más volátil).
    # Baja std  → consenso en el mercado mediático.
    df['sentiment_7d_std'] = (
        df['sentiment_avg']
        .rolling(window=7, min_periods=2)
        .std()
        .round(4)
    )

    # ── Volumen de noticias (media móvil 7d) ───────────────────────────
    # Un aumento repentino de cobertura mediática, independientemente
    # del tono, suele preceder movimientos fuertes de precio.
    # Se usa sentiment_count como proxy del volumen diario de noticias.
    df['news_volume_7d'] = (
        df['sentiment_count']
        .rolling(window=7, min_periods=1)
        .mean()
        .round(2)
    )

    # ── Estadísticas de las nuevas features ───────────────────────────
    new_cols = ['sentiment_7d', 'sentiment_14d', 'sentiment_momentum',
                'sentiment_7d_std', 'news_volume_7d']
    print("\n📈 Estadísticas de las nuevas features:")
    print(df[new_cols].describe().round(4).to_string())

    return df


# ══════════════════════════════════════════════════════════════════════
#  PASO 4: Escritura en la BD
# ══════════════════════════════════════════════════════════════════════

def update_db(engine, df: pd.DataFrame, asset: str) -> int:
    """
    Hace UPDATE de las nuevas columnas en daily_features.
    Solo actualiza filas que ya existen (no inserta nuevas).
    Convierte NaN → None para que PostgreSQL los acepte como NULL.
    """
    updated = 0
    rows_to_update = df.reset_index()  # date vuelve a ser columna

    with engine.begin() as conn:
        for _, row in rows_to_update.iterrows():

            # Convertir NaN/NaT a None para PostgreSQL
            def safe(val):
                if pd.isna(val):
                    return None
                return float(val) if isinstance(val, (np.floating, float)) else val

            conn.execute(text("""
                UPDATE daily_features
                SET
                    sentiment_7d        = :s7d,
                    sentiment_14d       = :s14d,
                    sentiment_momentum  = :smom,
                    sentiment_7d_std    = :sstd,
                    news_volume_7d      = :nvol,
                    updated_at          = NOW()
                WHERE date  = :date
                  AND asset = :asset
            """), {
                'date':  row['date'].date(),
                'asset': asset,
                's7d':   safe(row['sentiment_7d']),
                's14d':  safe(row['sentiment_14d']),
                'smom':  safe(row['sentiment_momentum']),
                'sstd':  safe(row['sentiment_7d_std']),
                'nvol':  safe(row['news_volume_7d']),
            })
            updated += 1

    return updated


# ══════════════════════════════════════════════════════════════════════
#  PASO 5: Verificación post-escritura
# ══════════════════════════════════════════════════════════════════════

def verify(engine, asset: str, n: int = 5):
    """Muestra las últimas n filas con las nuevas columnas para verificar."""
    with engine.connect() as conn:
        df_check = pd.read_sql(text("""
            SELECT
                date,
                sentiment_avg,
                sentiment_7d,
                sentiment_14d,
                sentiment_momentum,
                sentiment_7d_std,
                news_volume_7d
            FROM daily_features
            WHERE asset = :asset
              AND sentiment_7d IS NOT NULL
            ORDER BY date DESC
            LIMIT :n
        """), conn, params={'asset': asset, 'n': n})

    print(f"\n✅ Verificación — últimas {n} filas con nuevas features ({asset}):")
    print(df_check.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Añade features de sentimiento ampliadas a daily_features'
    )
    parser.add_argument('--asset',   default='BTC',
                        help='Activo a procesar (default: BTC)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Calcula pero no escribe en la BD')
    args = parser.parse_args()

    asset = args.asset.upper()
    print(f"{'='*55}")
    print(f"  sentiment_features_extended.py  |  asset={asset}")
    print(f"{'='*55}\n")

    engine = get_engine()

    # 1. Migración de columnas
    if not args.dry_run:
        migrate_columns(engine)
    else:
        print("⚠️  DRY-RUN: se omite la migración de columnas")

    # 2. Cargar datos
    df = load_daily_features(engine, asset)

    # 3. Calcular features
    df = compute_extended_features(df)

    # 4. Escribir o solo mostrar
    if args.dry_run:
        print("\n⚠️  DRY-RUN: no se escribe en la BD.")
        print("Primeras 10 filas calculadas:")
        print(df[['sentiment_avg', 'sentiment_7d', 'sentiment_14d',
                   'sentiment_momentum', 'sentiment_7d_std',
                   'news_volume_7d']].head(10).to_string())
    else:
        print(f"\n💾 Escribiendo en daily_features...")
        n = update_db(engine, df, asset)
        print(f"✅ Actualizadas {n} filas para asset={asset}")

        # 5. Verificar
        verify(engine, asset)

    print(f"\n{'='*55}")
    print("  Features disponibles para el pipeline de ML:")
    for col, desc in {
        'sentiment_avg':      'Score diario bruto (ya existía)',
        'sentiment_7d':       'Media móvil 7d  ← nueva',
        'sentiment_14d':      'Media móvil 14d ← nueva',
        'sentiment_momentum': '7d - 14d        ← nueva',
        'sentiment_7d_std':   'Std 7d          ← nueva',
        'news_volume_7d':     'Volumen noticias 7d ← nueva',
    }.items():
        print(f"  • {col:<22} {desc}")
    print(f"{'='*55}")


if __name__ == '__main__':
    main()