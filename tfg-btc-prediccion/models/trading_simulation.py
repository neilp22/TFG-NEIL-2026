# models/trading_simulation.py
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.data_loader import load_dataset, walk_forward_splits, FEATURES, TARGET

TRANSACTION_COST = 0.001  # 0.1% por operacion

def sharpe_ratio(returns, periods_per_year=365):
    """Calcula el Sharpe Ratio anualizado (sin tasa libre de riesgo)."""
    if returns.std() == 0: return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(periods_per_year)

def simulate_strategy(predictions, actual_returns, dates):
    """
    Simula la estrategia de trading y devuelve metricas.
    predictions:    array de 0/1
    actual_returns: retornos logaritmicos reales del periodo de test
    """
    strategy_returns = []
    for pred, ret in zip(predictions, actual_returns):
        if pred == 1:  # compramos
            net_ret = ret - TRANSACTION_COST
        else:          # cash
            net_ret = 0.0
        strategy_returns.append(net_ret)

    strategy_returns = np.array(strategy_returns)
    bnh_returns      = actual_returns  # buy-and-hold: siempre comprado

    return {
        'sharpe_strategy':  sharpe_ratio(strategy_returns),
        'sharpe_bnh':       sharpe_ratio(bnh_returns),
        'cumret_strategy':  np.exp(strategy_returns.sum()) - 1,
        'cumret_bnh':       np.exp(bnh_returns.sum()) - 1,
        'n_trades':         int(predictions.sum()),
    }

def evaluate_trading(model_name='xgboost', n_splits=5, test_size=60):
    from xgboost import XGBClassifier
    from sklearn.ensemble import RandomForestClassifier

    df     = load_dataset()
    splits = walk_forward_splits(df, n_splits=n_splits, test_size=test_size)

    all_metrics = []
    for i, (X_train, y_train, X_test, y_test, dates) in enumerate(splits):
        if model_name == 'xgboost':
            model = XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                eval_metric='logloss', random_state=42, verbosity=0
            )
        else:
            model = RandomForestClassifier(
                n_estimators=200, max_depth=6,
                min_samples_leaf=10, random_state=42, n_jobs=-1
            )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Obtener los retornos reales del periodo de test
        test_start = len(X_train)
        test_end   = test_start + len(X_test)
        actual_rets = df['returns'].values[test_start:test_end]

        metrics = simulate_strategy(preds, actual_rets, dates)
        metrics['split'] = i + 1
        all_metrics.append(metrics)
        print(f'  Split {i+1}: Sharpe={metrics["sharpe_strategy"]:.3f} | '
              f'BnH Sharpe={metrics["sharpe_bnh"]:.3f} | Trades={metrics["n_trades"]}')

    res_df = pd.DataFrame(all_metrics)
    print(f'\n--- {model_name.upper()} Trading ---')
    print(f'Sharpe Ratio (estrategia): {res_df.sharpe_strategy.mean():.3f}')
    print(f'Sharpe Ratio (buy&hold):   {res_df.sharpe_bnh.mean():.3f}')
    return res_df

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    print('=== Simulacion XGBoost ===')
    xgb_trading = evaluate_trading('xgboost')
    xgb_trading.to_csv('results/xgboost_trading.csv', index=False)

    print('\n=== Simulacion Random Forest ===')
    rf_trading = evaluate_trading('random_forest')
    rf_trading.to_csv('results/rf_trading.csv', index=False)