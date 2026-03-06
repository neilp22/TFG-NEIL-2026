# models/lstm_model.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.data_loader import load_dataset, FEATURES, TARGET

WINDOW = 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out).view(-1)


def make_sequences(X, y, window=WINDOW):
    Xs, ys = [], []
    for i in range(window, len(X)):
        Xs.append(X[i-window:i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def train_lstm(X_train, y_train, input_size, epochs=30, batch_size=32, lr=1e-3):
    # Reemplazar cualquier NaN/Inf que pueda quedar tras el scaler
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

    X_seq, y_seq = make_sequences(X_train, y_train)
    X_t = torch.tensor(X_seq, dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y_seq, dtype=torch.float32).to(DEVICE)

    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model     = LSTMClassifier(input_size=input_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_b, y_b in loader:
            optimizer.zero_grad()
            preds = model(X_b)
            loss  = criterion(preds, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f'    Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f}')
    return model


def predict_lstm(model, X_train, X_test):
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    X_combined = np.concatenate([X_train[-WINDOW:], X_test], axis=0)
    X_seq, _   = make_sequences(X_combined, np.zeros(len(X_combined)))
    X_t = torch.tensor(X_seq, dtype=torch.float32).to(DEVICE)

    model.eval()
    with torch.no_grad():
        logits = model(X_t).cpu().numpy()

    # Clip logits antes del sigmoid para evitar NaN por overflow
    logits = np.clip(logits, -50, 50)
    probas = 1 / (1 + np.exp(-logits))

    # Seguridad extra: reemplazar cualquier NaN residual con 0.5 (neutro)
    probas = np.nan_to_num(probas, nan=0.5)

    preds = (probas >= 0.5).astype(int)
    return preds, probas


def evaluate_lstm(n_splits=5, test_size=60):
    from models.data_loader import walk_forward_splits
    df     = load_dataset()
    splits = walk_forward_splits(df, n_splits=n_splits, test_size=test_size)

    results = []
    for i, (X_train, y_train, X_test, y_test, dates) in enumerate(splits):
        print(f'\n  Split {i+1}/{n_splits}...')

        scaler  = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_train)
        X_te_sc = scaler.transform(X_test)

        model = train_lstm(X_tr_sc, y_train, input_size=len(FEATURES))
        preds, probas = predict_lstm(model, X_tr_sc, X_te_sc)

        acc = accuracy_score(y_test, preds)
        f1  = f1_score(y_test, preds, zero_division=0)
        auc = roc_auc_score(y_test, probas)
        results.append({'split': i+1, 'accuracy': acc, 'f1': f1, 'auc': auc})
        print(f'  Acc={acc:.3f} | F1={f1:.3f} | AUC={auc:.3f}')

    res_df = pd.DataFrame(results)
    print('\n--- LSTM Resultados ---')
    print(f'Accuracy: {res_df.accuracy.mean():.3f} +/- {res_df.accuracy.std():.3f}')
    print(f'F1-Score: {res_df.f1.mean():.3f}      +/- {res_df.f1.std():.3f}')
    print(f'AUC-ROC:  {res_df.auc.mean():.3f}      +/- {res_df.auc.std():.3f}')
    return res_df


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    print(f'Dispositivo: {DEVICE}')
    results = evaluate_lstm(n_splits=5, test_size=60)
    results.to_csv('results/lstm_results.csv', index=False)
    print('Resultados guardados en results/lstm_results.csv')