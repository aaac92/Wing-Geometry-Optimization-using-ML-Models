# training_manager.py
import os
import json
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
import torch.optim as optim

import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation


# === Utility functions === #
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def tail_r2_score(y_true, y_pred, top_frac=0.05):
    """Compute R² only on the top fraction of y values (default top 5%)."""
    threshold = np.quantile(y_true, 1 - top_frac)
    mask = y_true >= threshold
    return r2_score(y_true[mask], y_pred[mask]) if mask.sum() > 0 else np.nan


# === PyTorch MLP Model === #
class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_size=64, dropout_rate=0.2, activation="relu"):
        super().__init__()
        act_layer = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "leakyrelu": nn.LeakyReLU()
        }.get(activation.lower(), nn.ReLU())

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            act_layer,
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            act_layer,
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze()


# === Training Manager Class === #
class TrainingManager:
    def __init__(
        self,
        model_type="mlp",
        hidden_size=64,
        dropout_rate=0.2,
        activation="relu",
        loss_fn="mse",
        learning_rate=1e-3,
        patience=15,
        weighting="none",  # "none", "proportional", "quantile"
        alpha=1.0,
        quantile=0.95,
        device=None,
        random_state=42,
    ):
        self.model_type = model_type.lower()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.loss_fn = loss_fn.lower()
        self.learning_rate = learning_rate
        self.patience = patience
        self.weighting = weighting
        self.alpha = alpha
        self.quantile = quantile
        self.random_state = random_state

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.scaler = None
        self.model = None

    # === Data Preparation === #
    def _load_and_prepare_data(self, data_path):
        df = pd.read_csv(data_path)
        X = df.iloc[:, :-1].values.astype(np.float32)
        y = df.iloc[:, -1].values.astype(np.float32)
        return X, y

    # === Loss Function === #
    def _get_loss(self):
        if self.loss_fn == "mse":
            return nn.MSELoss(reduction="none")
        elif self.loss_fn == "mae":
            return nn.L1Loss(reduction="none")
        elif self.loss_fn == "poisson":
            return nn.PoissonNLLLoss(log_input=False, full=False, reduction="none")
        else:
            raise ValueError(f"Unsupported loss: {self.loss_fn}")

    # === Training Loop for MLP === #
    def _train_mlp_cv(self, X, y, n_splits=5, max_epochs=500, batch_size=1024):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        metrics_all = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"\n=== Fold {fold+1}/{n_splits} ===")
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            # Standardize features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            self.scaler = scaler

            # Convert to tensors
            X_train_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
            y_train_t = torch.tensor(y_train, dtype=torch.float32).to(self.device)
            X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            y_val_t = torch.tensor(y_val, dtype=torch.float32).to(self.device)

            # Model, optimizer, loss
            self.model = MLPRegressor(
                input_dim=X.shape[1],
                hidden_size=self.hidden_size,
                dropout_rate=self.dropout_rate,
                activation=self.activation
            ).to(self.device)

            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            criterion = self._get_loss()

            best_r2 = -np.inf
            patience_counter = 0

            for epoch in range(max_epochs):
                self.model.train()
                permutation = torch.randperm(X_train_t.size(0))
                epoch_losses = []

                for i in range(0, X_train_t.size(0), batch_size):
                    optimizer.zero_grad()
                    indices = permutation[i:i+batch_size]
                    batch_x, batch_y = X_train_t[indices], y_train_t[indices]

                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)

                    # Weighting
                    if self.weighting == "proportional":
                        weights = 1 + self.alpha * (batch_y / batch_y.mean())
                        loss = loss * weights
                    elif self.weighting == "quantile":
                        threshold = torch.quantile(y_train_t, self.quantile)
                        weights = torch.where(batch_y >= threshold, 1+self.alpha, 1.0)
                        loss = loss * weights

                    loss = loss.mean()
                    loss.backward()
                    optimizer.step()
                    epoch_losses.append(loss.item())

                # Validation
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(X_val_t).cpu().numpy()
                r2 = r2_score(y_val, val_pred)

                print(f"Epoch {epoch+1}: Train Loss={np.mean(epoch_losses):.4f} Val R²={r2:.4f}", end="\r")

                # Early stopping
                if r2 > best_r2:
                    best_r2 = r2
                    patience_counter = 0
                    best_state = self.model.state_dict()
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"\nEarly stopping at epoch {epoch+1}")
                        break

            # Load best weights
            self.model.load_state_dict(best_state)

            # Metrics
            y_val_pred = self.model(X_val_t).detach().cpu().numpy()
            metrics = {
                "r2": r2_score(y_val, y_val_pred),
                "rmse": rmse(y_val, y_val_pred),
                "mae": mean_absolute_error(y_val, y_val_pred),
                "tail_r2": tail_r2_score(y_val, y_val_pred, top_frac=0.05)
            }
            metrics_all.append(metrics)
            print("Fold metrics:", metrics)

        return pd.DataFrame(metrics_all)

    # === Training entrypoint === #
    def train(self, data_path, n_splits=5, max_epochs=500, batch_size=1024):
        X, y = self._load_and_prepare_data(data_path)

        if self.model_type == "mlp":
            return self._train_mlp_cv(X, y, n_splits=n_splits, max_epochs=max_epochs, batch_size=batch_size)

        elif self.model_type == "lgbm":
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            metrics_all = []

            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                print(f"\n=== LGBM Fold {fold+1}/{n_splits} ===")
                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]

                model = lgb.LGBMRegressor(
                    objective="regression",
                    n_estimators=1000,
                    learning_rate=self.learning_rate,
                    n_jobs=max(1, os.cpu_count()-2),
                    random_state=self.random_state
                )
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[
                        early_stopping(stopping_rounds=self.patience),
                        log_evaluation(period=0)  # replaces verbose=False
                    ]
                )
                self.model = model

                y_val_pred = model.predict(X_val)
                metrics = {
                    "r2": r2_score(y_val, y_val_pred),
                    "rmse": rmse(y_val, y_val_pred),
                    "mae": mean_absolute_error(y_val, y_val_pred),
                    "tail_r2": tail_r2_score(y_val, y_val_pred, top_frac=0.05)
                }
                metrics_all.append(metrics)
                print("Fold metrics:", metrics)

            return pd.DataFrame(metrics_all)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    # === Evaluate / Predict === #
    def evaluate(self, X_test, y_test):
        if isinstance(self.model, nn.Module):
            self.model.eval()
            X_test_t = torch.tensor(self.scaler.transform(X_test), dtype=torch.float32).to(self.device)
            with torch.no_grad():
                y_pred = self.model(X_test_t).cpu().numpy()
        else:
            y_pred = self.model.predict(X_test)

        metrics = {
            "r2": r2_score(y_test, y_pred),
            "rmse": rmse(y_test, y_pred),
            "mae": mean_absolute_error(y_test, y_pred),
            "tail_r2": tail_r2_score(y_test, y_pred, top_frac=0.05)
        }
        return metrics

    def predict(self, X_new):
        if isinstance(self.model, nn.Module):
            X_new_t = torch.tensor(self.scaler.transform(X_new), dtype=torch.float32).to(self.device)
            with torch.no_grad():
                return self.model(X_new_t).cpu().numpy()
        else:
            return self.model.predict(X_new)

    # === Save / Load === #
    def save_model(self, folder_path, file_name_prefix):
        os.makedirs(folder_path, exist_ok=True)

        meta = {
            "model_type": self.model_type,
            "hidden_size": self.hidden_size,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            "loss_fn": self.loss_fn,
            "learning_rate": self.learning_rate,
            "patience": self.patience,
            "weighting": self.weighting,
            "alpha": self.alpha,
            "quantile": self.quantile,
            "random_state": self.random_state,
        }

        with open(Path(folder_path)/f"{file_name_prefix}.json", "w") as f:
            json.dump(meta, f, indent=4)

        if self.model_type == "mlp":
            torch.save(self.model.state_dict(), Path(folder_path)/f"{file_name_prefix}.pth")
            joblib.dump(self.scaler, Path(folder_path)/f"{file_name_prefix}_scaler.joblib")
        elif self.model_type == "lgbm":
            joblib.dump(self.model, Path(folder_path)/f"{file_name_prefix}.joblib")

    @classmethod
    def load_model(cls, folder_path, file_name_prefix, **kwargs):
        with open(Path(folder_path)/f"{file_name_prefix}.json", "r") as f:
            meta = json.load(f)

        meta.update(kwargs)
        manager = cls(**meta)

        if meta["model_type"] == "mlp":
            state_dict = torch.load(Path(folder_path)/f"{file_name_prefix}.pth", map_location=manager.device)
            manager.model = MLPRegressor(
                input_dim=kwargs.get("input_dim", None),  # Must be passed manually
                hidden_size=meta["hidden_size"],
                dropout_rate=meta["dropout_rate"],
                activation=meta["activation"]
            ).to(manager.device)
            manager.model.load_state_dict(state_dict)
            manager.scaler = joblib.load(Path(folder_path)/f"{file_name_prefix}_scaler.joblib")
        elif meta["model_type"] == "lgbm":
            manager.model = joblib.load(Path(folder_path)/f"{file_name_prefix}.joblib")

        return manager
