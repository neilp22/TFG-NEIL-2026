# models/tree_model.py
import pandas as pd
import numpy as np
import json
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.data_loader import load_dataset, walk_forward_splits, FEATURES

def evaluate_model(model_name='xgboost', n_splits=5,
                   test_size=60, random_state=42):
    """Evalua XGBoost o RandomForest con walk-forward validation."""
    df     = load_dataset()
    splits = walk_forward_splits(df, n_splits=n_splits, test_size=test_size)

    results          = []
    feature_importances = np.zeros(len(FEATURES))

    for i, (X_train, y_train, X_test, y_test, dates) in enumerate(splits):

        if model_name == 'xgboost':
            model = XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=random_state,
                verbosity=0
            )
        else:  # random_forest
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=6,
                min_samples_leaf=10,
                random_state=random_state,
                n_jobs=-1
            )

        model.fit(X_train, y_train)
        preds  = model.predict(X_test)
        probas = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        f1  = f1_score(y_test, preds, zero_division=0)
        auc = roc_auc_score(y_test, probas)
        results.append({'split': i+1, 'accuracy': acc, 'f1': f1, 'auc': auc})
        feature_importances += model.feature_importances_
        print(f'  Split {i+1}: Acc={acc:.3f} | F1={f1:.3f} | AUC={auc:.3f}')

    res_df = pd.DataFrame(results)
    print(f'\n--- {model_name.upper()} Resultados ---')
    print(f'Accuracy: {res_df.accuracy.mean():.3f} +/- {res_df.accuracy.std():.3f}')
    print(f'F1-Score: {res_df.f1.mean():.3f}      +/- {res_df.f1.std():.3f}')
    print(f'AUC-ROC:  {res_df.auc.mean():.3f}      +/- {res_df.auc.std():.3f}')

    # Feature importance media sobre todos los splits
    fi = pd.Series(
        feature_importances / len(splits),
        index=FEATURES
    ).sort_values(ascending=False)
    print(f'\nTop 5 features mas importantes:')
    print(fi.head())

    return res_df, fi

if __name__ == '__main__':
    import os; os.makedirs('results', exist_ok=True)

    print('=== XGBoost ===')
    xgb_res, xgb_fi = evaluate_model('xgboost')
    xgb_res.to_csv('results/xgboost_results.csv', index=False)
    xgb_fi.to_csv('results/xgboost_feature_importance.csv', header=['importance'])

    print('\n=== Random Forest ===')
    rf_res, rf_fi = evaluate_model('random_forest')
    rf_res.to_csv('results/rf_results.csv', index=False)
    rf_fi.to_csv('results/rf_feature_importance.csv', header=['importance'])

# Anadir al final de models/tree_model.py o ejecutar en un notebook:

from models.data_loader import load_dataset, walk_forward_splits, FEATURES
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import numpy as np

FEATURES_NO_SENTIMENT = [f for f in FEATURES if f != 'sentiment_avg']

def compare_with_without_sentiment(n_splits=5, test_size=60):
    df     = load_dataset()
    splits = walk_forward_splits(df, n_splits=n_splits, test_size=test_size)

    auc_with    = []
    auc_without = []

    for X_train, y_train, X_test, y_test, dates in splits:
        # Con sentiment (columna completa)
        m1 = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                           eval_metric='logloss', random_state=42, verbosity=0)
        m1.fit(X_train, y_train)
        auc_with.append(roc_auc_score(y_test, m1.predict_proba(X_test)[:,1]))

        # Sin sentiment: reconstruir X sin esa columna
        idx_sent = FEATURES.index('sentiment_avg')
        X_tr_ns  = np.delete(X_train, idx_sent, axis=1)
        X_te_ns  = np.delete(X_test,  idx_sent, axis=1)
        m2 = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                           eval_metric='logloss', random_state=42, verbosity=0)
        m2.fit(X_tr_ns, y_train)
        auc_without.append(roc_auc_score(y_test, m2.predict_proba(X_te_ns)[:,1]))

    print(f'AUC con sentiment:    {np.mean(auc_with):.4f} +/- {np.std(auc_with):.4f}')
    print(f'AUC sin sentiment:    {np.mean(auc_without):.4f} +/- {np.std(auc_without):.4f}')
    print(f'Delta AUC (impacto):  {np.mean(auc_with)-np.mean(auc_without):+.4f}')

compare_with_without_sentiment()

