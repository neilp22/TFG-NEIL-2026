"""
ver_noticias.py — Leer noticias recientes y analizar sentimiento con FinBERT

Uso:
  python agente_ia/ver_noticias.py
  python agente_ia/ver_noticias.py --fuente coindesk
  python agente_ia/ver_noticias.py --horas 6
"""

import sys, os, argparse
sys.path.insert(0, '.')

from dotenv import load_dotenv
load_dotenv('config/.env')

from sqlalchemy import text
from db.db_utils import get_engine

# ── Cargar noticias recientes ─────────────────────────────────────────────────

ap = argparse.ArgumentParser()
ap.add_argument('--fuente', default='', help='Filtrar por fuente (ej: coindesk)')
ap.add_argument('--horas',  type=int, default=24, help='Últimas N horas (default: 24)')
ap.add_argument('--n',      type=int, default=20, help='Cuántas mostrar (default: 20)')
args = ap.parse_args()

engine = get_engine()
fuente_filter = "AND source = :fuente" if args.fuente else ""

with engine.connect() as conn:
    rows = conn.execute(text(f"""
        SELECT id, source, timestamp, text, url
        FROM raw_texts
        WHERE timestamp > NOW() - INTERVAL '{args.horas} hours'
          AND asset = 'BTC'
          {fuente_filter}
        ORDER BY timestamp DESC
        LIMIT :n
    """), {'n': args.n, 'fuente': args.fuente} if args.fuente else {'n': args.n}).fetchall()

if not rows:
    print(f"No hay noticias en las últimas {args.horas}h.")
    sys.exit(0)

articles = [{'id': r[0], 'source': r[1], 'timestamp': r[2], 'text': r[3], 'url': r[4]} for r in rows]

# ── Mostrar lista numerada ────────────────────────────────────────────────────

print(f"\n{'='*70}")
print(f"  {len(articles)} noticias (últimas {args.horas}h){' de '+args.fuente if args.fuente else ''}")
print(f"{'='*70}\n")

for i, a in enumerate(articles, 1):
    ts   = a['timestamp'].strftime('%m-%d %H:%M UTC')
    text = a['text'][:90].replace('\n', ' ')
    print(f"  [{i:2}] [{a['source']:<18}] {ts}  {text}...")

print()

# ── Selección interactiva ─────────────────────────────────────────────────────

while True:
    try:
        raw = input("Escribe el número para ver completa + analizar (o 'q' para salir): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nSaliendo.")
        break

    if raw.lower() in ('q', 'salir', 'exit'):
        break

    try:
        idx = int(raw) - 1
        if not (0 <= idx < len(articles)):
            print(f"  Número fuera de rango (1-{len(articles)})")
            continue
    except ValueError:
        print("  Escribe un número válido.")
        continue

    a = articles[idx]
    ts = a['timestamp'].strftime('%Y-%m-%d %H:%M:%S UTC')

    print(f"\n{'─'*70}")
    print(f"  Fuente:  {a['source']}")
    print(f"  Hora:    {ts}")
    print(f"  URL:     {a['url'] or '(sin url)'}")
    print(f"{'─'*70}")
    print(f"\n{a['text']}\n")
    print(f"{'─'*70}")

    # ── Sentimiento FinBERT ───────────────────────────────────────────────────
    print("  Analizando sentimiento con FinBERT...")

    try:
        from transformers import pipeline as hf_pipeline
        finbert = hf_pipeline(
            'text-classification',
            model='ProsusAI/finbert',
            top_k=None,
            device=-1,
        )
        text_input = a['text'][:512]
        results = finbert([text_input])[0]

        scores = {r['label']: r['score'] for r in results}
        pos  = scores.get('positive', 0)
        neg  = scores.get('negative', 0)
        neu  = scores.get('neutral',  0)
        compound = pos - neg

        # Etiqueta dominante
        dominant = max(scores, key=scores.get)
        label_emoji = {'positive': '📈 POSITIVO', 'negative': '📉 NEGATIVO', 'neutral': '➡️  NEUTRO'}[dominant]

        # Bias bull/bear
        if compound > 0.2:
            bias = '🐂 BULLISH'
        elif compound < -0.2:
            bias = '🐻 BEARISH'
        else:
            bias = '↔️  NEUTRAL'

        print(f"\n  SENTIMIENTO FINBERT")
        print(f"  ───────────────────────────────")
        print(f"  Resultado:   {label_emoji}  (conf: {scores[dominant]*100:.1f}%)")
        print(f"  Bias:        {bias}")
        print(f"  ───────────────────────────────")
        print(f"  Positivo:  {pos*100:5.1f}%  {'█' * int(pos*20)}")
        print(f"  Negativo:  {neg*100:5.1f}%  {'█' * int(neg*20)}")
        print(f"  Neutro:    {neu*100:5.1f}%  {'█' * int(neu*20)}")
        print(f"  Compound:  {compound:+.4f}  (−1=muy bearish … +1=muy bullish)")
        print()

    except ImportError:
        print("  ⚠️  transformers no instalado.")
        print("  Instala con: pip install transformers torch")
    except Exception as e:
        print(f"  ❌ Error FinBERT: {e}")

    print()
