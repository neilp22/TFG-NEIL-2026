# models/trading_simulation.py
# Simulación de trading corregida
# Lógica: si modelo predice subida → estar en mercado ese día
# Uso: python models/trading_simulation.py

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sqlalchemy import text
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
except ImportError:
    os.system("pip install xgboost")
    from xgboost import XGBClassifier

from db.db_utils import get_engine

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
GREEN  = '#3fb950'
RED    = '#f85149'

os.makedirs('notebooks/figures', exist_ok=True)
os.makedirs('results', exist_ok=True)

FEATURES = ['returns', 'rsi_14', 'macd', 'bb_upper', 'bb_lower',
            'sma_7', 'sma_30', 'sentiment_avg', 'fear_greed']
TRANSACTION_COST = 0.001
INITIAL_CAPITAL  = 10000


def load_data():
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT date, close, returns, label,
                   rsi_14, macd, bb_upper, bb_lower,
                   sma_7, sma_30, sentiment_avg, fear_greed
            FROM daily_features
            WHERE asset = 'BTC' AND label IS NOT NULL
            ORDER BY date ASC
        """), conn)
    df['date'] = pd.to_datetime(df['date'])
    return df.set_index('date')


def prepare(df):
    df = df.copy()
    df['sentiment_avg'] = df['sentiment_avg'].fillna(0)
    df['fear_greed']    = df['fear_greed'].fillna(df['fear_greed'].median())
    return df.dropna(subset=['returns', 'rsi_14', 'macd'])


def train_and_predict(df):
    df  = prepare(df)
    idx = int(len(df) * 0.80)
    train, test = df.iloc[:idx], df.iloc[idx:]

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(train[FEATURES].values)
    X_test  = scaler.transform(test[FEATURES].values)
    y_train = train['label'].values.astype(int)
    y_test  = test['label'].values.astype(int)

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=n_neg / n_pos,
        random_state=42, eval_metric='logloss', verbosity=0,
        reg_alpha=0.1, reg_lambda=1.0,
    )
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"  Train: {len(train)}d | Test: {len(test)}d")
    print(f"  Período: {test.index[0].date()} → {test.index[-1].date()}")
    print(f"  Accuracy: {accuracy_score(y_test,y_pred):.3f} | AUC: {roc_auc_score(y_test,y_proba):.3f}")
    print(f"  Pred ↑: {y_pred.sum()} | ↓: {(y_pred==0).sum()}")

    return test.copy(), y_pred


def simulate(test_df, preds):
    """
    Estrategia correcta:
    - Día i: el modelo predice si el día i+1 sube o baja
    - Si pred[i]=1 → compramos al cierre del día i, vendemos al cierre del día i+1
    - El retorno que capturamos es close[i+1]/close[i] - 1
    - Buy & Hold: compra día 0, mantiene hasta el final
    """
    np.random.seed(42)
    closes = test_df['close'].values
    n = len(closes)

    # Retorno del día i = close[i]/close[i-1] - 1
    daily_ret = np.zeros(n)
    for i in range(1, n):
        daily_ret[i] = closes[i] / closes[i-1] - 1

    rand_preds = np.random.randint(0, 2, n)

    eq_hold  = np.zeros(n)
    eq_model = np.zeros(n)
    eq_rand  = np.zeros(n)

    cap_hold  = INITIAL_CAPITAL * (1 - TRANSACTION_COST)
    cap_model = INITIAL_CAPITAL
    cap_rand  = INITIAL_CAPITAL

    prev_m = -1
    prev_r = -1

    for i in range(n):
        r = daily_ret[i]

        # Buy & Hold
        cap_hold *= (1 + r)
        eq_hold[i] = cap_hold

        # Modelo
        p = int(preds[i])
        if p == 1:
            if prev_m != 1:
                cap_model *= (1 - TRANSACTION_COST)
            cap_model *= (1 + r)
        else:
            if prev_m == 1:
                cap_model *= (1 - TRANSACTION_COST)
        prev_m = p
        eq_model[i] = cap_model

        # Aleatorio
        rp = int(rand_preds[i])
        if rp == 1:
            if prev_r != 1:
                cap_rand *= (1 - TRANSACTION_COST)
            cap_rand *= (1 + r)
        else:
            if prev_r == 1:
                cap_rand *= (1 - TRANSACTION_COST)
        prev_r = rp
        eq_rand[i] = cap_rand

    sim = test_df.copy()
    sim['buy_hold']  = eq_hold
    sim['model']     = eq_model
    sim['random']    = eq_rand
    sim['pred']      = preds
    sim['in_market'] = (preds == 1).astype(int)
    return sim


def sharpe(eq):
    r = pd.Series(eq).pct_change().dropna()
    return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0


def max_dd(eq):
    eq = pd.Series(eq)
    return float((eq / eq.cummax() - 1).min() * 100)


def plot_all(sim, test_df):
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(
        'Simulación de Trading — XGBoost + Sentiment vs Buy & Hold\n'
        f'Capital inicial: ${INITIAL_CAPITAL:,} | Coste/operación: {TRANSACTION_COST*100:.1f}%',
        fontsize=13, color='#e6edf3', y=0.99
    )
    gs = fig.add_gridspec(4, 2, hspace=0.5, wspace=0.3)

    fh = sim['buy_hold'].iloc[-1]
    fm = sim['model'].iloc[-1]
    fr = sim['random'].iloc[-1]
    rh = (fh / INITIAL_CAPITAL - 1) * 100
    rm = (fm / INITIAL_CAPITAL - 1) * 100
    rr = (fr / INITIAL_CAPITAL - 1) * 100

    # Panel 1: Capital
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(sim.index, sim['buy_hold'], color=ACCENT, lw=1.8,
             label=f'Buy & Hold ({rh:+.1f}%)')
    ax1.plot(sim.index, sim['model'],   color=GREEN,  lw=1.8,
             label=f'XGBoost + Sentiment ({rm:+.1f}%)')
    ax1.plot(sim.index, sim['random'],  color='#8b949e', lw=0.8,
             alpha=0.5, linestyle='--', label=f'Aleatorio ({rr:+.1f}%)')
    ax1.axhline(INITIAL_CAPITAL, color='#30363d', lw=0.8, linestyle=':')
    in_m = sim['in_market'].values
    for i in range(1, len(sim)):
        if in_m[i]:
            ax1.axvspan(sim.index[i-1], sim.index[i], alpha=0.04, color=GREEN)
    ax1.set_ylabel('Capital (USD)')
    ax1.set_title('Evolución del Capital')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax1.legend(fontsize=9, framealpha=0.3)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Precio + señales
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(test_df.index, test_df['close'], color=ACCENT, lw=1.0)
    p = sim['pred'].values
    for i in range(1, len(sim)):
        if p[i] == 1 and p[i-1] == 0:
            ax2.axvline(sim.index[i], color=GREEN, alpha=0.5, lw=0.8)
        elif p[i] == 0 and p[i-1] == 1:
            ax2.axvline(sim.index[i], color=RED, alpha=0.5, lw=0.8)
    from matplotlib.lines import Line2D
    ax2.legend(handles=[
        Line2D([0],[0], color=ACCENT, label='Precio BTC'),
        Line2D([0],[0], color=GREEN,  label='Entrada al mercado'),
        Line2D([0],[0], color=RED,    label='Salida del mercado'),
    ], fontsize=8, framealpha=0.3)
    ax2.set_ylabel('Precio (USD)')
    ax2.set_title('Señales de Compra/Venta del Modelo')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax2.grid(True, alpha=0.3)

    # Panel 3: Drawdown
    ax3 = fig.add_subplot(gs[2, :])
    dd_h = (sim['buy_hold'] / sim['buy_hold'].cummax() - 1) * 100
    dd_m = (sim['model']    / sim['model'].cummax()    - 1) * 100
    ax3.fill_between(sim.index, dd_h, 0, alpha=0.3, color=ACCENT, label='Buy & Hold')
    ax3.fill_between(sim.index, dd_m, 0, alpha=0.3, color=GREEN,  label='Modelo')
    ax3.plot(sim.index, dd_h, color=ACCENT, lw=0.8)
    ax3.plot(sim.index, dd_m, color=GREEN,  lw=0.8)
    ax3.set_ylabel('Drawdown (%)')
    ax3.set_title('Drawdown desde el Máximo Histórico')
    ax3.legend(fontsize=9, framealpha=0.3)
    ax3.grid(True, alpha=0.3)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}%'))

    # Panel 4: Retornos mensuales
    ax4 = fig.add_subplot(gs[3, 0])
    mh = sim['buy_hold'].resample('ME').last().pct_change().dropna() * 100
    mm = sim['model'].resample('ME').last().pct_change().dropna() * 100
    x  = np.arange(len(mm))
    ax4.bar(x - 0.2, mh.values, 0.35, color=ACCENT, alpha=0.7, label='Buy & Hold')
    ax4.bar(x + 0.2, mm.values, 0.35, color=GREEN,  alpha=0.7, label='Modelo')
    ax4.axhline(0, color='#8b949e', lw=0.8)
    ax4.set_xticks(x)
    ax4.set_xticklabels([d.strftime('%b %y') for d in mm.index],
                         rotation=45, ha='right', fontsize=7)
    ax4.set_ylabel('Retorno mensual (%)')
    ax4.set_title('Retornos Mensuales')
    ax4.legend(fontsize=8, framealpha=0.3)
    ax4.grid(True, alpha=0.3, axis='y')

    # Panel 5: Tabla
    ax5 = fig.add_subplot(gs[3, 1])
    ax5.axis('off')
    n_ops    = int(abs(np.diff(sim['in_market'].values)).sum())
    dias_m   = int(sim['in_market'].sum())
    sh       = sharpe(sim['buy_hold'].values)
    sm       = sharpe(sim['model'].values)
    sr       = sharpe(sim['random'].values)
    ddh      = max_dd(sim['buy_hold'].values)
    ddm      = max_dd(sim['model'].values)

    rows = [
        ['Métrica',        'Buy & Hold',     'XGBoost',        'Aleatorio'],
        ['Capital final',  f'${fh:,.0f}',    f'${fm:,.0f}',    f'${fr:,.0f}'],
        ['Retorno total',  f'{rh:+.1f}%',    f'{rm:+.1f}%',    f'{rr:+.1f}%'],
        ['Sharpe Ratio',   f'{sh:.2f}',       f'{sm:.2f}',       f'{sr:.2f}'],
        ['Max Drawdown',   f'{ddh:.1f}%',     f'{ddm:.1f}%',     '—'],
        ['Días mercado',   f'{len(sim)}',     f'{dias_m}',       f'~{len(sim)//2}'],
        ['Nº operaciones', '1',               f'{n_ops}',        '—'],
    ]
    tbl = ax5.table(cellText=rows[1:], colLabels=rows[0],
                    cellLoc='center', loc='center', bbox=[0,0,1,1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor('#1a2a1a' if c == 2 and r > 0 else '#161b22')
        cell.set_edgecolor('#30363d')
        cell.set_text_props(color='#e6edf3',
                            fontweight='bold' if r == 0 else 'normal')
        if r == 0:
            cell.set_facecolor('#21262d')
    ax5.set_title('Métricas Comparativas', fontsize=10, pad=12)

    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.savefig('notebooks/figures/07_trading_simulation.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  → Guardada: notebooks/figures/07_trading_simulation.png")
    return {'rh':rh,'rm':rm,'sh':sh,'sm':sm,'ddh':ddh,'ddm':ddm,
            'dias_m':dias_m,'n_ops':n_ops}


if __name__ == '__main__':
    print("="*55)
    print("SIMULACIÓN DE TRADING")
    print("="*55)
    df = load_data()
    print(f"Dataset: {len(df)} días")
    print("\nEntrenando...")
    test_df, preds = train_and_predict(df)
    print("\nSimulando...")
    sim = simulate(test_df, preds)
    print("\nGenerando gráficas...")
    m = plot_all(sim, test_df)
    sim.to_csv('results/trading_simulation.csv')
    print("  → Guardada: results/trading_simulation.csv")
    print("\n" + "="*55)
    print("RESUMEN FINAL")
    print("="*55)
    print(f"  Buy & Hold:     ${sim['buy_hold'].iloc[-1]:,.0f}  ({m['rh']:+.1f}%)")
    print(f"  Modelo XGBoost: ${sim['model'].iloc[-1]:,.0f}  ({m['rm']:+.1f}%)")
    print(f"  Sharpe B&H:     {m['sh']:.2f}")
    print(f"  Sharpe Modelo:  {m['sm']:.2f}")
    print(f"  Max DD B&H:     {m['ddh']:.1f}%")
    print(f"  Max DD Modelo:  {m['ddm']:.1f}%")
    print(f"  Días mercado:   {m['dias_m']} de {len(sim)}")
    print(f"  Operaciones:    {m['n_ops']}")