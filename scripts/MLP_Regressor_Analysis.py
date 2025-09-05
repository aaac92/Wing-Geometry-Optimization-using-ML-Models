import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_size, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        return self.net(x)

# Custom weighted loss function
def make_weighted_mse(power: float = 1.0, mode: str = "linear"):
    """
    Returns a weighted MSE loss function.
    - power: controls aggressiveness (higher = stronger penalty on high values)
    - mode: "linear", "exponential", or "clipped"
    """
    def weighted_mse(y_pred, y_true):
        # normalize to [0,1] so scaling is stable
        normed = y_true / y_true.max()
        if mode == "linear":
            weights = 1.0 + normed ** power
        elif mode == "exponential":
            weights = torch.exp(power * normed)
        elif mode == "clipped":
            weights = 1.0 + torch.clamp(normed, min=0.5) ** power
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        return torch.mean(((y_pred - y_true) ** 2) * weights)
    
    return weighted_mse

w_mse = make_weighted_mse(power=1.2)
huber = nn.SmoothL1Loss()

def combo_loss(y_pred, y_true, lam=0.8):
    return lam * huber(y_pred, y_true) + (1-lam) * w_mse(y_pred, y_true)

# Training loop
def train_one_model(seed, X_train, y_train, X_val, y_val,
                    input_dim, lr=1e-3, weight_decay=1e-5,
                    batch_size=128, max_epochs=200, patience=20):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = MLPRegressor(input_dim, hidden_size=batch_size, dropout_rate=0.05).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)

    best_val_loss = np.inf
    best_state = None
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        idx = np.random.permutation(len(X_train_t))
        for i in range(0, len(idx), batch_size):
            batch_idx = idx[i:i+batch_size]
            xb = X_train_t[batch_idx]
            yb = y_train_t[batch_idx]
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = (pred, yb)
            opt.zero_grad()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = combo_loss(val_pred, y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    model.load_state_dict(best_state)
    return model

# Ensemble training
def train_ensemble(n_models, X_train, y_train, X_val, y_val, X_test, y_test):
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    input_dim = X_train.shape[1]
    models = []
    preds_test = []

    for i in range(n_models):
        print(f"Training model {i+1}/{n_models} ...")
        m = train_one_model(seed=42+i, X_train=X_train, y_train=y_train,
                            X_val=X_val, y_val=y_val, input_dim=input_dim)
        models.append(m)

        m.eval()
        with torch.no_grad():
            pt = m(torch.tensor(X_test, dtype=torch.float32)).numpy().ravel()
        preds_test.append(pt)

    preds_test = np.mean(preds_test, axis=0)
    return preds_test, models

# Metrics
def top_decile_metrics(y_true, y_pred, q=90):
    thr = np.percentile(y_true, q)
    mask = y_true >= thr
    return {
        "mae_top": mean_absolute_error(y_true[mask], y_pred[mask]),
        "r2_top": r2_score(y_true[mask], y_pred[mask])
    }

# Example run
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
# Add this path to the list of places Python looks for modules.
sys.path.insert(0, project_root)
# Path to regressor dataset
Training_File = "Airfoils_for_Regressor_E_Dataset_16.csv"
data_path = project_root / Training_File

df = pd.read_csv(data_path)

# All columns except the last one are features
X = df.iloc[:, :-1].values.astype(np.float32)
# The very last column is the target
y = df.iloc[:, -1:].values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

preds, models = train_ensemble(5, X_train, y_train, X_val, y_val, X_test, y_test)

# global metrics
# print("MAE:", mean_absolute_error(y_test, preds))
# print("R²:", r2_score(y_test, preds))
# print("Top decile metrics:", top_decile_metrics(y_test, preds))
