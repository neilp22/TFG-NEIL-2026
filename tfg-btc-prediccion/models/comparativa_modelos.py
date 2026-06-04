# models/comparativa_modelos.py
# Genera la tabla comparativa final de todos los modelos
# Requiere haber ejecutado antes: arima_model.py, diagnostico.py, lstm_model.py
# Uso: python models/comparativa_modelos.py

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams['figure.facecolor'] = '#0d1117'
plt.rcParams['axes.facecolor']   = '#161b22'
plt.rcParams['axes.edgecolor']   = '#30363d'
plt.rcParams['text.color']       = '#e6edf3'
plt.rcParams['axes.labelcolor']  = '#e6edf3'
plt.rcParams['xtick.color']      = '#8b949e'
plt.rcParams['ytick.color']      = '#8b949e'
plt.rcParams['grid.color']       = '#21262d'
plt.rcParams['font.family']      = 'monospace'

os.makedirs('notebooks/figures', exist_ok=True)

ACCENT = '#f7931a'
GREEN  = '#3fb950'
RED    = '#f85149'
BLUE   = '#58a6ff'
PURPLE = '#bc8cff'
GRAY   = '#8b949e'


def load_results():
    """Carga los CSVs de resultados de cada modelo."""
    results = {}

    files = {
        'ARIMA':              'results/arima_metrics.csv',
        'XGBoost (precio)':   'results/xgboost_mejorado.csv',
        'LSTM':               'results/lstm_metrics.csv',
    }

    for name, path in files.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Para XGBoost separar los dos modelos
            if 'model' in df.columns:
                df_price = df[df['model'] == 'price_only']
                df_sent  = df[df['model'] == 'price_sentiment']
                if not df_price.empty:
                    results['XGBoost (precio)'] = df_price
                if not df_sent.empty:
                    results['XGBoost (precio + sentiment)'] = df_sent
            else:
                results[name] = df
        else:
            print(f"  ⚠ No encontrado: {path}")

    return results


def summarize(df):
    return {
        'acc_mean': df['accuracy'].mean(),
        'acc_std':  df['accuracy'].std(),
        'f1_mean':  df['f1'].mean(),
        'f1_std':   df['f1'].std(),
        'auc_mean': df['auc_roc'].mean(),
        'auc_std':  df['auc_roc'].std(),
    }


def plot_comparison(results):
    models = list(results.keys())
    summaries = {m: summarize(df) for m, df in results.items()}

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle('Comparativa de Modelos — Walk-Forward Validation (5 splits × 60 días)',
                 fontsize=13, color='#e6edf3')

    metrics   = ['acc', 'f1', 'auc']
    titles    = ['Accuracy', 'F1-Score', 'AUC-ROC']
    baselines = [0.50, 0.50, 0.50]

    colors = [GRAY, BLUE, BLUE, ACCENT, PURPLE]
    colors = colors[:len(models)]

    for ax, metric, title, baseline in zip(axes, metrics, titles, baselines):
        means = [summaries[m][f'{metric}_mean'] for m in models]
        stds  = [summaries[m][f'{metric}_std']  for m in models]

        x = np.arange(len(models))
        bars = ax.bar(x, means, color=colors[:len(models)],
                      alpha=0.85, width=0.6,
                      yerr=stds, capsize=4,
                      error_kw={'color': '#e6edf3', 'linewidth': 1})

        ax.axhline(baseline, color=RED, linestyle='--',
                   linewidth=0.8, alpha=0.7, label='Baseline (0.50)')
        ax.axhline(0.53, color=GREEN, linestyle=':',
                   linewidth=0.8, alpha=0.7, label='Objetivo (0.53)')

        ax.set_title(title, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(' ', '\n') for m in models],
                           fontsize=8)
        ax.set_ylim(0.30, 0.75)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=7, framealpha=0.3)

        # Valores encima de barras
        for bar, val, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + std + 0.008,
                    f'{val:.3f}', ha='center', va='bottom',
                    fontsize=8, color='#e6edf3')

    plt.tight_layout()
    plt.savefig('notebooks/figures/09_comparativa_modelos.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  → Guardada: notebooks/figures/09_comparativa_modelos.png")


def print_table(results):
    summaries = {m: summarize(df) for m, df in results.items()}

    print("\n" + "="*75)
    print("TABLA COMPARATIVA FINAL — TODOS LOS MODELOS")
    print("="*75)
    print(f"{'Modelo':<35} {'Accuracy':>12} {'F1-Score':>12} {'AUC-ROC':>12}")
    print("─"*75)

    # Ordenar por AUC
    sorted_models = sorted(summaries.keys(),
                           key=lambda m: summaries[m]['auc_mean'],
                           reverse=True)

    for model in sorted_models:
        s = summaries[model]
        best = '← MEJOR' if model == sorted_models[0] else ''
        print(f"  {model:<33} "
              f"{s['acc_mean']:.3f}±{s['acc_std']:.3f}  "
              f"{s['f1_mean']:.3f}±{s['f1_std']:.3f}  "
              f"{s['auc_mean']:.3f}±{s['auc_std']:.3f}  {best}")

    print("─"*75)
    print(f"  {'Baseline aleatorio':<33} {'0.500':>12} {'0.500':>12} {'0.500':>12}")
    print("="*75)

    # Guardar como CSV
    rows = []
    for model in sorted_models:
        s = summaries[model]
        rows.append({
            'model':    model,
            'acc_mean': round(s['acc_mean'], 4),
            'acc_std':  round(s['acc_std'],  4),
            'f1_mean':  round(s['f1_mean'],  4),
            'f1_std':   round(s['f1_std'],   4),
            'auc_mean': round(s['auc_mean'], 4),
            'auc_std':  round(s['auc_std'],  4),
        })
    pd.DataFrame(rows).to_csv('results/comparativa_final.csv', index=False)
    print("\n✅ Guardado en results/comparativa_final.csv")


if __name__ == '__main__':
    print("Cargando resultados de todos los modelos...")
    results = load_results()

    if not results:
        print("❌ No hay resultados. Ejecuta primero:")
        print("   python models/arima_model.py")
        print("   python models/diagnostico.py")
        print("   python models/lstm_model.py")
    else:
        print(f"  Modelos encontrados: {list(results.keys())}")
        print_table(results)
        print("\nGenerando gráfica comparativa...")
        plot_comparison(results)
        print("\n✅ Todo listo para el Informe Seguimiento 2")