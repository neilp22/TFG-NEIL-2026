# models/arima_model.py
import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.data_loader import load_dataset, walk_forward_splits

warnings.filterwarnings('ignore')

def predict_arima_split(train_returns, test_returns, order=(2,0,2)):
    """
    Walk-forward ARIMA: re-entrena en cada paso anadiendo el dato real.
    Devuelve predicciones binarias (1=sube, 0=baja).
    """
    history = list(train_returns)
    preds   = []

    for actual in test_returns:
        try:
            model  = ARIMA(history, order=order)
            result = model.fit()
            forecast = result.forecast(steps=1)[0]
            preds.append(1 if forecast > 0 else 0)
        except Exception:
            preds.append(1)  # fallback: predecir subida (clase mayoritaria)
        history.append(actual)  # aniadir el valor real al historico

    return np.array(preds)

def evaluate_arima(n_splits=5, test_size=60, order=(2,0,2)):
    """Evalua ARIMA con walk-forward validation."""
    df     = load_dataset()
    splits = walk_forward_splits(df, n_splits=n_splits, test_size=test_size)

    results = []
    for i, (X_train, y_train, X_test, y_test, dates) in enumerate(splits):
        # ARIMA solo usa la columna 'returns' (columna 8 en FEATURES)
        # Recargamos directamente del df para mayor claridad
        train_df = df.iloc[:len(X_train)]
        test_df  = df.iloc[len(X_train):len(X_train)+len(X_test)]

        preds = predict_arima_split(
            train_df['returns'].values,
            test_df['returns'].values,
            order=order
        )

        acc = accuracy_score(y_test, preds)
        f1  = f1_score(y_test, preds, zero_division=0)
        auc = roc_auc_score(y_test, preds)
        results.append({'split': i+1, 'accuracy': acc, 'f1': f1, 'auc': auc})
        print(f'  Split {i+1}: Acc={acc:.3f} | F1={f1:.3f} | AUC={auc:.3f}')

    res_df = pd.DataFrame(results)
    print('\n--- ARIMA Resultados ---')
    print(f'Accuracy: {res_df.accuracy.mean():.3f} +/- {res_df.accuracy.std():.3f}')
    print(f'F1-Score: {res_df.f1.mean():.3f}      +/- {res_df.f1.std():.3f}')
    print(f'AUC-ROC:  {res_df.auc.mean():.3f}      +/- {res_df.auc.std():.3f}')
    return res_df

if __name__ == '__main__':
    print('Evaluando ARIMA (puede tardar ~5-10 min por walk-forward)...')
    results = evaluate_arima(n_splits=5, test_size=60, order=(2,0,2))
    results.to_csv('results/arima_results.csv', index=False)
    print('Resultados guardados en results/arima_results.csv')

