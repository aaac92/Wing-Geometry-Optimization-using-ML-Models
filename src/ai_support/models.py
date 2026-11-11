import copy
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from torch.utils.data import DataLoader, Subset, TensorDataset
import joblib
import json
from math import comb

from .data_split import shape_groups, shapes_text, split_by_shape, kfold_by_shape
from .scoring import classification_metrics

# Training protocol shared by every model class (see each train()):
#   Test split: test_split (default 20%) of the SHAPES, with every wing configuration of each, never used in
#       training. Every split below is also by shape (ai_support.data_split explains why): a shape is never on
#       both sides. Before 2026-11-10 all of them were by row.
#   k >= 2 (default 5): k-fold cross-validation on the training split measures the configuration; the mean
#       fold scores are returned as CV_* metrics (the tuning scripts rank on them).
#   Final fit: one model trained on the whole training split except FINAL_VALIDATION_SHARE of it, which only
#       early-stops it. That model is the one saved and the one scored on the test split.
#   k = 0 or 1: no cross-validation, only the final fit (about k + 1 times faster; ml_gym's default).
# Before 2026-11-04 the saved model was simply the last fold's model (64% of the data) and the other folds
# were thrown away.
FINAL_VALIDATION_SHARE = 0.1


def _final_split(groups, random_state, stratify=None):
    """(train indices, early-stopping indices) of the final fit, by shape (`groups`: shape number of every row)."""
    return split_by_shape(groups, FINAL_VALIDATION_SHARE, random_state, stratify=stratify)


def _split_note(groups, train_i, test_i, random_state):
    """One console line that says what the data is and how it was split."""
    print(f"Data: {shapes_text(groups)} | split by shape (seed {random_state}): "
          f"{len(train_i):,} training rows, {len(test_i):,} test rows")


def _lgbm_eval_kwargs(X_val, y_val):
    """Early-stopping data for LGBMRegressor.fit: eval_X/eval_y (eval_set is deprecated); eval_set only on
    LightGBM versions older than the eval_X argument (e.g. 4.6.0)."""
    import inspect
    if "eval_X" in inspect.signature(lgb.LGBMRegressor.fit).parameters:
        return {"eval_X": X_val, "eval_y": y_val}
    return {"eval_set": [(X_val, y_val)]}


def _print_cv(k, scores):
    """Cross-validation summary: mean of every fold score, then the fold values."""
    print(f"\n==== Cross-validation ({k} folds) ====")
    for name, values in scores.items():
        print(f"{name:<10} mean {np.mean(values):.4f} | per fold: " + ", ".join(f"{v:.4f}" for v in values))


# Fine-tuning (fine_tune of MLP_Regressor and LGBM_Regressor): a short update of a trained model with new
# simulations, used by the iterative inference. The new rows are weighted up (new_weight) and trained together
# with "anchor" rows of the original dataset, so the model learns the new region without forgetting the rest.
# Scaler, feature expansion and target transform stay frozen (the model keeps its input and output scales).
# A held-out part of the new shapes and of the anchor rows decides whether the update is kept: it must lower the
# error on the held-out new shapes and may raise the anchor error by at most anchor_tolerance (relative) or
# anchor_abs_tolerance (E units), whichever is larger: a purely relative limit on a base error of ~0.1 E rejected
# updates that moved the anchors by only 0.01 E (2026-11-04 synthetic test). A rejected update
# leaves the model exactly as it was. from_base=True restarts every update from the originally loaded model, so
# repeated updates (one per iteration, each with all simulations so far) never drift; the gate then compares the
# update with that original model, which has never seen any new shape (an earlier kept update may have trained on
# the shapes now held out, which would make its error look better than it is and reject good updates).

def _fine_tune_split(n_new, groups_new, n_anchor, holdout, random_state):
    """(new train idx, new held-out idx, anchor train idx, anchor held-out idx). New rows are split by group (shape),
    so the three wing configurations of one shape never sit on both sides."""
    rng = np.random.default_rng(random_state)
    groups = np.asarray(groups_new if groups_new is not None else np.arange(n_new))
    unique = np.unique(groups)
    n_hold = int(round(holdout * len(unique))) if len(unique) >= 2 else 0
    n_hold = min(max(n_hold, 1 if len(unique) >= 2 else 0), len(unique) - 1) if len(unique) else 0
    held = set(rng.choice(unique, size=n_hold, replace=False).tolist()) if n_hold else set()
    new_val = np.array([i for i in range(n_new) if groups[i] in held], dtype=int)
    new_train = np.array([i for i in range(n_new) if groups[i] not in held], dtype=int)
    perm = rng.permutation(n_anchor)
    n_anchor_hold = int(round(holdout * n_anchor))
    return new_train, new_val, perm[n_anchor_hold:], perm[:n_anchor_hold]


def _gate(before_new, after_new, before_anchor, after_anchor, anchor_tolerance, anchor_abs_tolerance):
    """True when the update lowers the new-shape error and keeps the anchor error within tolerance."""
    new_ok = after_new < before_new if (before_new is not None and after_new is not None) else True
    allowed = max(before_anchor * anchor_tolerance, anchor_abs_tolerance) if before_anchor is not None else 0.0
    anchor_ok = (after_anchor <= before_anchor + allowed) if (before_anchor is not None and after_anchor is not None) else True
    return bool(new_ok and anchor_ok)


def _lgbm_combo_objective(y_true, y_pred):
    """
    Analytical Gradient and Hessian for the combo_loss
    (80% Huber Loss + 20% Exponentially Weighted MSE).
    """
    lam = 0.8
    power = 3.5

    diff = y_pred - y_true
    abs_diff = np.abs(diff)

    # Huber Loss Gradients and Hessians (delta = 1.0)
    grad_huber = np.where(abs_diff < 1.0, diff, np.sign(diff))
    hess_huber = np.where(abs_diff < 1.0, 1.0, 0.0)

    # Weighted MSE Gradients and Hessians
    y_max = np.max(y_true)
    y_max = y_max if y_max > 0 else 1e-8
    normed = y_true / y_max
    weights = np.exp(power * normed)

    grad_wmse = 2.0 * diff * weights
    hess_wmse = 2.0 * weights

    # Combine
    grad = lam * grad_huber + (1.0 - lam) * grad_wmse
    hess = lam * hess_huber + (1.0 - lam) * hess_wmse

    return grad, hess


def _lgbm_combo_eval(y_true, y_pred):
    """
    Evaluation metric corresponding to the custom_combo_objective.
    """
    lam = 0.8
    power = 3.5

    diff = y_pred - y_true
    abs_diff = np.abs(diff)

    # Huber Loss
    huber_loss = np.where(abs_diff < 1.0, 0.5 * (diff**2), abs_diff - 0.5)

    # Weighted MSE
    y_max = np.max(y_true)
    y_max = y_max if y_max > 0 else 1e-8
    normed = y_true / y_max
    weights = np.exp(power * normed)
    wmse_loss = (diff**2) * weights

    # Combined loss
    loss = lam * huber_loss + (1.0 - lam) * wmse_loss

    return 'combo_loss', np.mean(loss), False


def _lgbm_objective_and_metric(objective):
    """(objective passed to LightGBM, early-stopping eval metric) for an objective name ('combo' is custom)."""
    if objective == 'combo':
        return _lgbm_combo_objective, _lgbm_combo_eval
    if objective in ['huber', 'quantile', 'logcosh']:
        return objective, objective
    return objective, 'mae'

class CSTFeatureEngineer:
    """
    Model-agnostic, stateless (row-wise) input expansion for CST-coefficient datasets.
    Used by LGBM_Regressor and MLP_Regressor. Any other model class needs the same calls:
    fit() + transform() in _load_and_prepare_data, transform() in predict() (skipped for
    'raw'), and to_dict()/from_dict() in save_model/load_model.

    The expansion is computed in memory (a few vectorised matrix products; about 65 MB of
    float32 for 400k rows in the widest mode). Nothing is written to disk. It uses no
    statistics of the data, so applying it before the train/test split leaks nothing.

    MODES
    - 'raw'          identity (original columns only).
    - 'station'      raw + thickness and camber at n_stations chordwise stations
                     (+2*n_stations columns).
    - 'sum_station'  raw + (upper - lower) and (upper + lower) per CST index
                     (+2*n_coef columns) + the station features.

    STATION FEATURES
    With upper/lower coefficient vectors U, L (n_coef = order + 1) and the CST basis
    S_i(x) = C(x) * Bernstein_{i,order}(x), C(x) = sqrt(x) * (1 - x):
        thickness(x) = sum_i (U_i - L_i) * S_i(x)
        camber(x)    = sum_i ((U_i + L_i) / 2) * S_i(x)
    Both are LINEAR in the coefficients. If the input columns are deltas applied to a
    fixed base airfoil, these equal the true thickness/camber up to a constant offset,
    which tree models ignore. Assumes standard CST (class function exponents 0.5 and 1.0).

    Input columns are located by name prefix (default 'cst_l_<k>' / 'cst_u_<k>'), so the
    raw column order of the CSV does not matter at fit time; transform() then expects
    inputs in that same raw column order.
    """
    MODES = ('raw', 'station', 'sum_station')

    def __init__(self, mode='raw', lower_prefix='cst_l_', upper_prefix='cst_u_', n_stations=9):
        if mode not in self.MODES:
            raise ValueError(f"Invalid feature mode '{mode}'. Choose one of: {self.MODES}")
        self.mode = mode
        self.lower_prefix = lower_prefix
        self.upper_prefix = upper_prefix
        self.n_stations = n_stations
        self.lower_idx = None
        self.upper_idx = None
        self.n_input = None
        self._basis = None  # (n_stations, n_coef) CST shape-function matrix

    @staticmethod
    def _find(columns, prefix):
        """Column positions whose names are prefix + integer, ordered by that integer."""
        found = [(int(str(c)[len(prefix):]), i) for i, c in enumerate(columns)
                 if str(c).startswith(prefix) and str(c)[len(prefix):].isdigit()]
        if not found:
            raise ValueError(f"No columns named '{prefix}<k>' found in the input columns.")
        return [i for _, i in sorted(found)]

    def _build_basis(self):
        order = len(self.lower_idx) - 1
        x = np.linspace(0.02, 0.98, self.n_stations)
        class_fn = np.sqrt(x) * (1.0 - x)
        bernstein = np.column_stack([comb(order, i) * x**i * (1.0 - x)**(order - i) for i in range(order + 1)])
        self._basis = bernstein * class_fn[:, None]

    def fit(self, columns):
        """Resolves the CST column positions from the raw feature-column names."""
        columns = list(columns)
        self.n_input = len(columns)
        if self.mode != 'raw':
            self.lower_idx = self._find(columns, self.lower_prefix)
            self.upper_idx = self._find(columns, self.upper_prefix)
            if len(self.lower_idx) != len(self.upper_idx):
                raise ValueError("Different number of lower and upper CST columns.")
            self._build_basis()
        return self

    def transform(self, X):
        """Returns a float32 array: the raw columns followed by the mode's engineered columns."""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self.mode == 'raw':
            return X.astype(np.float32)
        if self._basis is None:
            raise RuntimeError("CSTFeatureEngineer is not fitted. Call fit(columns) first.")
        if X.shape[1] != self.n_input:
            raise ValueError(f"Expected {self.n_input} raw feature columns, got {X.shape[1]}.")
        X = X.astype(np.float64)
        lower, upper = X[:, self.lower_idx], X[:, self.upper_idx]
        blocks = [X]
        if self.mode == 'sum_station':
            blocks += [upper - lower, upper + lower]
        blocks += [(upper - lower) @ self._basis.T, ((upper + lower) / 2.0) @ self._basis.T]
        return np.column_stack(blocks).astype(np.float32)

    def to_dict(self):
        """JSON-serialisable configuration, so a saved model can rebuild the same inputs."""
        return {'mode': self.mode, 'lower_prefix': self.lower_prefix, 'upper_prefix': self.upper_prefix,
                'n_stations': self.n_stations, 'n_input': self.n_input,
                'lower_idx': self.lower_idx, 'upper_idx': self.upper_idx}

    @classmethod
    def from_dict(cls, d):
        obj = cls(d['mode'], d['lower_prefix'], d['upper_prefix'], d['n_stations'])
        obj.n_input, obj.lower_idx, obj.upper_idx = d['n_input'], d['lower_idx'], d['upper_idx']
        if obj.mode != 'raw':
            obj._build_basis()
        return obj

class _PyTorchMLPRegressor(nn.Module):
    """
    Internal PyTorch model for MLP Regressor.
    """
    def __init__(self, input_dim, hidden_layer_sizes, dropout_rate):
        super().__init__()
        layers = []
        current_dim = input_dim

        for h_size in hidden_layer_sizes:
            layers.append(nn.Linear(current_dim, h_size))
            layers.append(nn.BatchNorm1d(h_size))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = h_size

        # Output layer
        layers.append(nn.Linear(current_dim, 1))
        
        # Unpack list into Sequential
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
class MLP_Regressor:
    """
    A class to encapsulate the MLP Regressor for predicting the output value.
    Handles data loading, normalization and scaling, as well as an advanced training
    loop with early stopping.
    """
    def __init__(self,
                 hidden_layer_sizes=[128, 128, 128, 128, 64],
                 dropout_rate=0.05,
                 learning_rate=1e-3,
                 batch_size=1024,
                 epochs=500,
                 patience=16,
                 k=5,
                 test_split=0.2,
                 random_state=42,
                 weight='None',
                 log_trans=False,
                 feature_mode='raw'):
        
        # Hyperparameters
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # Other training parameters
        self.epochs = epochs
        self.patience = patience
        self.weight = weight

        # Data manipulation parameters
        self.k = k
        self.random_state = random_state
        self.test_split = test_split

        # Class components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler = None
        self.input_dim = None # Will be determined after data is loaded
        self.log_transform = log_trans
        self.y_min = None
        # Input feature expansion: 'raw' | 'station' | 'sum_station' (see CSTFeatureEngineer).
        # Fitted to the CSV's column names in _load_and_prepare_data; input_dim follows from it.
        self.feature_mode = feature_mode
        self.feature_engineer = CSTFeatureEngineer(mode=feature_mode)
        
        if self.device.type == "cuda":
            print(f"MLP Regressor initialized. Training will use device: {torch.cuda.get_device_name()}")
        else:
            print(f"MLP Regressor initialized. Training will use device: {self.device}")

    def _load_and_prepare_data(self, data_path):
        """
        Loads data from a single CSV file with a header, assuming the last column is the target. 
        For regressors this method should also normalize the data.

        PARAMETERS:
        - data_path: Full path to csv file that contains training data
        """
        df = pd.read_csv(data_path)

        feature_df = df.iloc[:, :-1]
        self.feature_engineer.fit(feature_df.columns)
        features = self.feature_engineer.transform(feature_df.values)
        output = df.iloc[:, -1].values.astype(np.float32)

        # Optional logarithmic output transformation
        if self.log_transform:
            self.y_min = float(output.min())
            output = np.log1p(output - self.y_min)

        # Data split for testing, by shape (ai_support.data_split)
        groups = shape_groups(feature_df)
        train_i, test_i = split_by_shape(groups, self.test_split, self.random_state)
        self.train_groups = groups[train_i]   # shape of every training row: folds and the final split use it
        _split_note(groups, train_i, test_i, self.random_state)
        train_data, test_data, train_targets, test_targets = features[train_i], features[test_i], output[train_i], output[test_i]

        # Normalization
        self.scaler = StandardScaler()
        train_data = self.scaler.fit_transform(train_data)
        test_data = self.scaler.transform(test_data)

        # Convert to tensor
        train_data = torch.tensor(train_data, dtype=torch.float32).to(self.device)
        test_data = torch.tensor(test_data, dtype=torch.float32).to(self.device)

        # Convert targets to 2D tensor for regression (N, 1) instead of (N,)
        train_targets = torch.tensor(train_targets, dtype=torch.float32).unsqueeze(1).to(self.device)
        test_targets = torch.tensor(test_targets, dtype=torch.float32).unsqueeze(1).to(self.device)

        self.input_dim = train_data.shape[1]

        return train_data, train_targets, test_data, test_targets
    
    def _find_max_batch_size(self, X_t, y_t, weights_t):
        """
        Dynamically finds the maximum batch size that fits in GPU VRAM by
        catching OutOfMemory exceptions and halving the batch size. 
        (Use only if you want to train with Large Batches) (Currently out of service)
        """
        if self.device.type == 'cpu':
            return len(X_t)
            
        print("Dynamically calculating maximum batch size...")
        batch_size = len(X_t) 
        
        while batch_size > 0:
            try:
                torch.cuda.empty_cache()
                
                # Slice dummy batch
                X_dummy = X_t[:batch_size]
                y_dummy = y_t[:batch_size].view(-1, 1)
                w_dummy = weights_t[:batch_size].view(-1, 1)
                
                # Initialize dummy model and optimizer to simulate exact memory footprint
                dummy_model = _PyTorchMLPRegressor(
                    input_dim=self.input_dim,
                    hidden_layer_sizes=self.hidden_layer_sizes,
                    dropout_rate=self.dropout_rate
                ).to(self.device)
                
                dummy_optimizer = optim.Adam(dummy_model.parameters(), lr=self.learning_rate)
                
                # Dummy forward and backward pass
                dummy_model.train()
                dummy_optimizer.zero_grad()
                
                preds = dummy_model(X_dummy)

                # Simple MSE for gradient calculation footprint
                loss = nn.MSELoss()(preds, y_dummy) 
                loss.backward()
                dummy_optimizer.step()

                # If execution reaches this point, the batch size fits in VRAM.
                # Clean up dummy variables to free VRAM for actual training.                
                del dummy_model, dummy_optimizer, X_dummy, y_dummy, w_dummy, preds, loss
                torch.cuda.empty_cache()

                # Apply a 0.9 safety margin to prevent borderline OOM during actual training loops                
                optimal_batch = int(batch_size * 0.9) 
                print(f"Optimal batch size found: {optimal_batch}")
                
                return optimal_batch
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Free whatever was allocated before the crash
                    torch.cuda.empty_cache()
                    # Halve the batch size and try again
                    batch_size //= 2
                else:
                    # Raise any other unexpected PyTorch errors
                    raise e
                    
        raise RuntimeError("Could not find a batch size that fits in GPU VRAM.")

    def train(self, data_path, refit=True):
        """
        Trains the regressor and returns metrics (protocol at the top of this module).

        Returns the test-split 'MAE' and 'R2_Score' of the final model and, with k >= 2, the cross-validation
        means 'CV_MAE' and 'CV_R2'. refit=False skips the final fit and returns only the CV scores (used by
        Optuna trials, which never keep the model).

        PARAMETERS:
        - data_path: Same data path used for the CSV in _load_and_prepare_data()
        - refit: Train the final model after cross-validation (default True)
        """

        # Load data
        X, y, X_test, y_test = self._load_and_prepare_data(data_path)
        weights_train = self._sample_weights(y)

        # Create dataset from tensors
        train_dataset = TensorDataset(X, y, weights_train)

        def run(train_i, val_i):
            """Trains a fresh model on train_i, early-stopped on val_i. Returns (val MAE, val R2, best epoch)."""
            train_loader = DataLoader(Subset(train_dataset, train_i), batch_size=self.batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(Subset(train_dataset, val_i), batch_size=self.batch_size, shuffle=False)
            self.model = _PyTorchMLPRegressor(
                input_dim=self.input_dim,
                hidden_layer_sizes=self.hidden_layer_sizes,
                dropout_rate=self.dropout_rate
            ).to(self.device)
            best_epoch = self._fit(train_loader, val_loader)
            mae, r2 = self._loader_metrics(val_loader)
            return mae, r2, best_epoch

        metrics = {}
        if self.k >= 2:
            folds = kfold_by_shape(self.train_groups, self.k, self.random_state)
            scores = {"MAE": [], "R2": []}
            print(f"Starting regressor cross-validation: {self.k} folds (by shape), batch size {self.batch_size}...")
            for fold, (train_i, val_i) in enumerate(folds, start=1):
                mae, r2, best_epoch = run(train_i, val_i)
                print(f"Fold {fold}/{self.k}: MAE {mae:.4f}, R^2 {r2:.4f}, best epoch {best_epoch}")
                scores["MAE"].append(mae)
                scores["R2"].append(r2)
            _print_cv(self.k, scores)
            metrics = {"CV_MAE": float(np.mean(scores["MAE"])), "CV_R2": float(np.mean(scores["R2"]))}
            if not refit:
                return metrics

        train_i, val_i = _final_split(self.train_groups, self.random_state)
        print(f"\nFinal fit on {len(train_i):,} rows (early stopping on {len(val_i):,})...")
        _, _, best_epoch = run(train_i, val_i)
        print(f"Final model: best epoch {best_epoch}")
        return {**self._evaluate(X_test, y_test), **metrics}

    def _sample_weights(self, y):
        """Per-row weights of the 'weight' mode for a (N, 1) target tensor, normalised to a mean of 1 (ones when the
        mode weights inside the loss instead)."""
        if self.weight == 'instance':
            # This uses the target value as the weight
            weights = y
            weights_train = weights*(len(y)/weights.sum())
        elif self.weight == 'frequency':
            # This uses bin-based inverse frequency weighting.
            # Creates bins and counts data inside of them to assign a weight value to that bin of data.
            # Bins that are less dense will have a higher weight value.
            num_bins = 20
            y_flat = y.flatten()
            y_min, y_max = y_flat.min().item(), y_flat.max().item()
            edges = torch.linspace(y_min, y_max, steps=num_bins + 1, device=self.device)
            bin_idx = torch.bucketize(y_flat, edges, right=False) - 1
            bin_idx = torch.clamp(bin_idx, min=0, max=num_bins-1).to(torch.long)
            bin_counts = torch.bincount(bin_idx, minlength=num_bins).float()
            inv_freq = torch.where(bin_counts == 0,
                       torch.tensor(1.0, device=bin_counts.device),
                       1.0 / bin_counts)
            weights = inv_freq[bin_idx]
            weights_train = weights*(len(y_flat)/weights.sum())
        else:
            weights_train = torch.ones(len(y), dtype=torch.float32)
        return weights_train

    def _prepare(self, X, y=None):
        """Raw feature rows (and E values) -> model tensors on the device: feature expansion, frozen scaler and the
        model's own target transform (log1p(E - y_min) for log models; E below y_min is clipped to y_min)."""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self.feature_engineer.mode != 'raw':
            X = self.feature_engineer.transform(X)
        X_t = torch.tensor(self.scaler.transform(X), dtype=torch.float32).to(self.device)
        if y is None:
            return X_t
        y = np.asarray(y, dtype=np.float32)
        if self.log_transform:
            y = np.log1p(np.maximum(y - self.y_min, 0.0))
        return X_t, torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(self.device)

    def _tensor_mae(self, X_t, y_t):
        """MAE in E units of the current model on prepared tensors (None for no rows)."""
        if len(y_t) == 0:
            return None
        loader = DataLoader(TensorDataset(X_t, y_t, torch.ones(len(y_t))), batch_size=8192, shuffle=False)
        return float(self._loader_metrics(loader)[0])

    def fine_tune(self, X_new, y_new, X_anchor, y_anchor, groups_new=None, new_weight=10, epochs=50, patience=5,
                  lr_factor=0.1, holdout=0.2, anchor_tolerance=0.05, anchor_abs_tolerance=0.1, random_state=None, from_base=True,
                  force_accept=False, verbose=False):
        """
        Short update with new simulations (see the fine-tuning note at the top of this module): starts from the
        original weights (from_base) or the current ones and trains at most `epochs` epochs at lr_factor x the
        learning rate on the anchor rows plus the new rows repeated new_weight times, early-stopped (patience) on
        the held-out rows; the best epoch's weights are kept. X_* are raw feature rows (CSV column order), y_* the E
        values; groups_new names the shape of each new row. Returns a report: accepted, MAE of the held-out new
        shapes and anchor rows before/after, best epoch. force_accept keeps the update without the gate (used to
        rebuild an update that was accepted before).
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model is not trained. Please call train() or load_model() first.")
        if self.device.type == "cuda":
            # Returns the GPU memory PyTorch keeps cached from earlier work (large prediction batches) to the
            # driver: the backward pass creates its own cuBLAS handle outside PyTorch's cache, and on a small GPU
            # that failed with CUBLAS_STATUS_ALLOC_FAILED while the cache held the memory.
            torch.cuda.empty_cache()
        if getattr(self, "_base_state", None) is None:
            self._base_state = copy.deepcopy(self.model.state_dict())
        previous = copy.deepcopy(self.model.state_dict())
        new_train, new_val, anchor_train, anchor_val = _fine_tune_split(len(y_new), groups_new, len(y_anchor), holdout, random_state)
        Xn, yn = self._prepare(X_new, y_new)
        Xa, ya = self._prepare(X_anchor, y_anchor)
        to_t = lambda idx: torch.as_tensor(idx, dtype=torch.long, device=self.device)
        if from_base:  # the reference of the gate: the original model (see the fine-tuning note)
            self.model.load_state_dict(self._base_state)
        before_new = self._tensor_mae(Xn[to_t(new_val)], yn[to_t(new_val)])
        before_anchor = self._tensor_mae(Xa[to_t(anchor_val)], ya[to_t(anchor_val)])

        repeat = max(1, int(round(new_weight)))
        train_idx_new = np.tile(new_train, repeat)
        X_train = torch.cat([Xa[to_t(anchor_train)], Xn[to_t(train_idx_new)]])
        y_train = torch.cat([ya[to_t(anchor_train)], yn[to_t(train_idx_new)]])
        # Early stopping watches the held-out NEW shapes (the region being learned); the anchors are protected by
        # the gate. Without held-out new shapes the held-out anchors early-stop the update.
        val_rows = (Xn[to_t(new_val)], yn[to_t(new_val)]) if len(new_val) else (Xa[to_t(anchor_val)], ya[to_t(anchor_val)])
        if len(val_rows[1]) == 0:
            val_rows = (X_train, y_train)
        w_train = self._sample_weights(y_train).to(self.device)
        w_val = self._sample_weights(val_rows[1]).to(self.device)
        batch = max(2, min(self.batch_size, len(y_train)))
        train_loader = DataLoader(TensorDataset(X_train, y_train, w_train), batch_size=batch, shuffle=True, drop_last=True)
        val_loader = DataLoader(TensorDataset(val_rows[0], val_rows[1], w_val), batch_size=batch, shuffle=False)

        if from_base:
            self.model.load_state_dict(self._base_state)
        saved = (self.learning_rate, self.epochs, self.patience)
        self.learning_rate, self.epochs, self.patience = self.learning_rate * lr_factor, int(epochs), int(patience)
        if random_state is not None:
            torch.manual_seed(int(random_state))
        try:
            best_epoch = self._fit(train_loader, val_loader, verbose=False, freeze_batchnorm=True)
        finally:
            self.learning_rate, self.epochs, self.patience = saved
        after_new = self._tensor_mae(Xn[to_t(new_val)], yn[to_t(new_val)])
        after_anchor = self._tensor_mae(Xa[to_t(anchor_val)], ya[to_t(anchor_val)])
        accepted = force_accept or _gate(before_new, after_new, before_anchor, after_anchor, anchor_tolerance,
                                         anchor_abs_tolerance)
        if not accepted:
            self.model.load_state_dict(previous)
        self.model.eval()
        report = {"accepted": bool(accepted), "new_rows": int(len(y_new)), "new_held_out": int(len(new_val)),
                  "anchor_rows": int(len(y_anchor)), "mae_new_before": before_new, "mae_new_after": after_new,
                  "mae_anchor_before": before_anchor, "mae_anchor_after": after_anchor, "best_epoch": int(best_epoch)}
        if verbose:
            print(f"MLP fine-tune: {'kept' if accepted else 'rejected'} | held-out new MAE {before_new} -> {after_new} | "
                  f"anchor MAE {before_anchor} -> {after_anchor} | best epoch {best_epoch}")
        return report

    def _loader_metrics(self, loader):
        """(MAE, R2) of the current model on a DataLoader of (features, targets, weights), in E units."""
        self.model.eval()
        pred, target = [], []
        with torch.no_grad():
            for features, targets, _ in loader:
                preds = self.model(features.to(self.device)).cpu().numpy()
                targets = targets.cpu().numpy()
                if self.log_transform:
                    targets = (np.expm1(targets) + self.y_min).astype(np.float32)
                    preds = (np.expm1(preds) + self.y_min).astype(np.float32)
                pred.extend(preds)
                target.extend(targets)
        return mean_absolute_error(target, pred), r2_score(target, pred)

    def _fit(self, train_loader, val_loader, verbose=True, freeze_batchnorm=False):
        '''
        Trains model on a given dataset, this method is for internal use.
        This method also contains and chooses custom or standard loss functions and handles early stopping.

        PARAMETERS:
        - model: model object initialized each fold
        - train_loader: DataLoader with the shuffled folds
        - loss_fn: Loss function to be used in this training cycle

        Returns the epoch (1-based) with the lowest validation loss; the model ends with those weights.
        '''
        # Custom weighted loss function
        def make_weighted_mse(power: float = 1.0, mode: str = "linear"):
            """
            Returns a weighted MSE loss function.
            - power: controls aggressiveness (higher = stronger penalty)
            - mode: "linear", "exponential", or "clipped"
            """
            def weighted_mse(y_pred, y_true):
                # normalize to [0,1] so scaling is stable
                normed = y_true / y_true.max()
                if mode == "linear":
                    weights = 1.0 + normed ** power
                elif mode == "exponential":
                    weights = torch.exp(power * normed)
                else:
                    raise ValueError(f"Unknown mode: {mode}")
                return torch.mean(((y_pred - y_true) ** 2) * weights)
            return weighted_mse

        # Custom combined loss function
        def combo_loss(y_pred, y_true, lam=0.8):
            w_mse = make_weighted_mse(power=3.5, mode='exponential')
            huber = nn.SmoothL1Loss()
            return lam * huber(y_pred, y_true) + (1-lam) * w_mse(y_pred, y_true)

        # Define loss_fn for this loop        
        if self.weight == 'frequency' or self.weight == 'instance':
            loss_fn = nn.MSELoss(reduction='none')
        elif self.weight == 'combo':
            loss_fn = lambda yp, yt: combo_loss(yp, yt, lam=0.8)
        elif self.weight == 'exponential':
            loss_fn = make_weighted_mse(power=2, mode='exponential')
        elif self.weight == 'linear':
            loss_fn = make_weighted_mse(power=5, mode='linear')
        elif self.weight == 'None':
            loss_fn = nn.SmoothL1Loss()

        # Data fit start 
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=self.patience, factor=0.5)

        best_val_loss = float('inf')
        early_stop_counter = 0
        best_model_state = None
        best_epoch = 0

        for epoch in range(self.epochs):
            # Validation below switches to eval mode; without this, dropout would only act in the first epoch
            self.model.train()
            if freeze_batchnorm:
                # Fine-tuning: BatchNorm keeps normalising with the saved running statistics and does not update
                # them (dropout still trains). Updating them from a few small batches wrecks a trained model.
                for module in self.model.modules():
                    if isinstance(module, nn.BatchNorm1d):
                        module.eval()

            for features, targets, weights in train_loader:
                features, targets, weights = features.to(self.device), targets.to(self.device), weights.to(self.device)

                # Resets optimizer
                optimizer.zero_grad()

                # Forward Pass
                pred = self.model(features)

                if self.weight == 'instance' or self.weight == 'frequency':
                    weights = weights.view(-1, 1)

                    # Calculate the raw, per-sample loss, used for some weight modes
                    unweighted_loss = loss_fn(pred, targets)

                # Apply weights                
                if self.weight == 'instance' or self.weight == 'frequency':
                    weighted_loss = unweighted_loss*weights
                if self.weight == 'combo':
                    weighted_loss = combo_loss(pred, targets)
                if self.weight == 'exponential' or self.weight == 'linear' or self.weight == 'None':
                    loss = loss_fn(pred, targets)

                # Take the mean to get the final loss for backpropagation
                if self.weight == 'instance' or self.weight == 'frequency' or self.weight == 'combo':
                    loss = weighted_loss.mean()

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

            # Validation for early stopping
            self.model.eval()
            val_loss_cum = 0.0 # Stores a cumulative value of the loss to then use it to obtain the average loss
            with torch.no_grad():
                for val_features, val_targets, val_weights in val_loader:
                    val_pred = self.model(val_features)
                    val_weights = val_weights.to(self.device)

                    if self.weight == 'instance' or self.weight == 'frequency':
                        val_weights = val_weights.view(-1, 1)
                        # Calculate the raw, per-sample loss, used for some weight modes
                        val_unweighted_loss = loss_fn(val_pred, val_targets)

                    # Apply weights
                    if self.weight == 'instance' or self.weight == 'frequency':
                        val_weighted_loss = val_unweighted_loss*val_weights
                    if self.weight == 'combo':
                        val_weighted_loss = combo_loss(val_pred, val_targets)
                    if self.weight == 'exponential' or self.weight == 'linear' or self.weight == 'None':
                        val_loss = loss_fn(val_pred, val_targets)

                    # Take the mean to get the final val_loss
                    if self.weight == 'instance' or self.weight == 'frequency' or self.weight == 'combo':
                        val_loss = val_weighted_loss.mean()

                    val_loss_cum += val_loss.item()
            
            average_val_loss = val_loss_cum/len(val_loader)
            scheduler.step(average_val_loss)

            # Early stopping check
            if average_val_loss < best_val_loss:
                best_val_loss = average_val_loss
                early_stop_counter = 0
                best_epoch = epoch + 1
                # A copy: state_dict() returns the live tensors, which later epochs keep changing
                best_model_state = copy.deepcopy(self.model.state_dict())
            else:
                early_stop_counter += 1
                if early_stop_counter >= self.patience:
                    if verbose:
                        print(f"Early stopping triggered at epoch {epoch + 1}. Best validation loss: {best_val_loss:.4f}")
                    break

            if verbose and epoch % 100 == 0:
                print(f"Loss at epoch {epoch}: {val_loss.item():.4f}")

        # Restore the weights from the best validation-loss epoch
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        return best_epoch

    def _evaluate(self, X_test, y_test):
        '''
        Evaluates the model on the test dataset, returns metrics.

        PARAMETERS:
        - X_test: Test features tensor
        - y_test: Test targets tensor
        '''
        if not all([self.model, self.scaler]):
            raise RuntimeError("Model is not trained. Please call train() first.")
        self.model.eval()

        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=len(X_test), shuffle=False)

        pred_cumulative = []
        targets_cumulative = []
        with torch.no_grad():
            for features, targets in test_loader:
                features, targets = features.to(self.device), targets.to(self.device)

                test_preds = self.model(features)

                # Convert to numpy for manipulation
                features = features.cpu().numpy()
                test_preds = test_preds.cpu().numpy()
                targets = targets.cpu().numpy()

                # Remove log transform in case it's used
                if self.log_transform:
                    # Inverse transformation
                    targets = (np.expm1(targets) + self.y_min).astype(np.float32)
                    test_preds = (np.expm1(test_preds) + self.y_min).astype(np.float32)

                # Accumulates data for calculation
                pred_cumulative.extend(test_preds)
                targets_cumulative.extend(targets)
        
        avg_r2 = r2_score(targets_cumulative, pred_cumulative)
        avg_mae = mean_absolute_error(targets_cumulative, pred_cumulative)

        metrics = {'MAE': avg_mae, 'R2_Score': avg_r2}

        return metrics
    
    def predict(self, X):
        '''
        Method to predict targets from input data.

        PARAMETERS:
        - X: numpy array or list of datapoints for the regressor, unscaled.
        '''
        if not all([self.model, self.scaler]):
            raise RuntimeError("Model is not trained. Please call train() first.")
        
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Expand with the feature mode used in training ('raw' skips this step)
        if self.feature_engineer.mode != 'raw':
            X = self.feature_engineer.transform(X)

        # Apply same transformations as in training
        X = self.scaler.transform(X)
        X = torch.tensor(X, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            preds = self.model(X)
            preds = preds.cpu().numpy()
            if self.log_transform:
                preds = np.expm1(preds) + self.y_min
        
        predictions = preds

        return predictions
    
    def save_model(self, folder_path, file_name_prefix):
        """
        Saves the model, scaler and metadata. Creates target directory if it doesn't exist.

        PARAMETERS:
        - folder_path: Directory where the data will be stored
        - file_name_prefix: Prefix name to identify data; prefix_model, prefix_scaler and prefix_meta
        """
        if not all([self.model, self.scaler]):
            raise RuntimeError("Model not trained yet. No model to save, please call train() first.")
        
        os.makedirs(folder_path, exist_ok=True)
        
        torch.save(self.model.state_dict(), os.path.join(folder_path, f"{file_name_prefix}_model.pth"))
        joblib.dump(self.scaler, os.path.join(folder_path, f"{file_name_prefix}_scaler.joblib"))

        metadata = {'input_dim': self.input_dim, 'y_min': self.y_min,
                    'feature_engineer': self.feature_engineer.to_dict()}
        with open(os.path.join(folder_path, f"{file_name_prefix}_meta.json"), 'w') as f:
            json.dump(metadata, f)

        print(f"Regressor model and transformers saved in folder: {folder_path}")

    @classmethod
    def load_model(cls, folder_path, file_name_prefix, **kwargs):
        """
        Loads a model and its associated transformers.

        PARAMETERS:
        - folder_path: Directory where the data is stored
        - file_name_prefix: Prefix name to identify data; prefix_model, prefix_scaler and prefix_meta
        """
        with open(os.path.join(folder_path, f"{file_name_prefix}_meta.json"), 'r') as f:
            metadata = json.load(f)

        # Re-create the class instance, passing any saved hyperparameters
        regressor = cls(**kwargs)
        regressor.input_dim = metadata['input_dim']
        regressor.y_min = metadata.get('y_min', None)
        # Models saved without 'feature_engineer' keep the default 'raw' mode
        if 'feature_engineer' in metadata:
            regressor.feature_engineer = CSTFeatureEngineer.from_dict(metadata['feature_engineer'])
            regressor.feature_mode = regressor.feature_engineer.mode

        # Load the scaler
        regressor.scaler = joblib.load(os.path.join(folder_path, f"{file_name_prefix}_scaler.joblib"))

        # Instantiate model with correct dimensions and load weights
        regressor.model = _PyTorchMLPRegressor(
            input_dim=regressor.input_dim,
            hidden_layer_sizes=regressor.hidden_layer_sizes,
            dropout_rate=regressor.dropout_rate
        )
        model_weights_path = os.path.join(folder_path, f"{file_name_prefix}_model.pth")
        state_dict = torch.load(model_weights_path, map_location=regressor.device, weights_only=True)
        regressor.model.load_state_dict(state_dict)
        regressor.model.to(regressor.device)
        regressor.model.eval()

        print(f"Regressor model loaded successfully from folder: {folder_path}")
        return regressor
    
class LGBM_Regressor:
    """
    A class to encapsulate the LightGBM Regressor.
    This class handles scaling and training with native early stopping.
    """
    def __init__(self,
                 n_estimators=1000,
                 learning_rate=0.1,
                 lr_decay=None,
                 num_leaves=20,
                 max_depth=10,
                 stopping_rounds=100,
                 k=5,
                 test_size=0.2,
                 random_state=42,
                 objective='regression_l2',
                 weight='None',
                 feature_mode='raw'):
        
        # Hyperparameters for Grid Search
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        # lr_decay is OPTIONAL and OFF by default (None = constant learning rate).
        # If set to a float in (0, 1), learning_rate is multiplied by lr_decay**iter
        # every boosting round via lgb.reset_parameter (iter restarts at 0 on every
        # fit() call). Values well below 1 collapse the effective LR quickly
        # (0.90 -> <1% of its initial value by ~44 rounds) and starve large
        # n_estimators budgets; on dataset_17 a fixed-parameter test went from
        # 1.79 to 2.28 test MAE with 0.90 vs. no decay.
        self.lr_decay = lr_decay
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.weight = weight
        
        # Other training parameters
        self.stopping_rounds = stopping_rounds
        self.k = k
        self.test_size = test_size
        self.test_split = test_size # Mapping for _load_and_prepare_data consistency
        self.random_state = random_state

        # Input feature expansion: 'raw' | 'station' | 'sum_station' (see CSTFeatureEngineer).
        # Constructing the engineer here validates the mode immediately; it is fitted to
        # the CSV's column names in _load_and_prepare_data.
        self.feature_mode = feature_mode
        self.feature_engineer = CSTFeatureEngineer(mode=feature_mode)

        # Class components
        self.model = None
        self.scaler = None
        self.objective = objective
        
        print("LGBMRegressor initialized.")

    def _load_and_prepare_data(self, data_path):
        """
        Loads data from a single CSV file with a header, assuming the
        last column is the target.

        The raw feature columns are expanded by self.feature_engineer according to
        self.feature_mode before the train/test split. The expansion is computed in
        memory on every call and is row-wise/stateless, so it leaks nothing across
        the split. In 'raw' mode this is identical to the original behaviour.

        PARAMETERS:
        - data_path: Full path to csv file that contains training data
        """
        df = pd.read_csv(data_path)

        feature_df = df.iloc[:, :-1]
        output = df.iloc[:, -1].values.astype(np.float32) # LightGBM prefers a 1D array for y

        self.feature_engineer.fit(feature_df.columns)
        features = self.feature_engineer.transform(feature_df.values)

        # Data split for testing, by shape (ai_support.data_split)
        groups = shape_groups(feature_df)
        train_i, test_i = split_by_shape(groups, self.test_split, self.random_state)
        self.train_groups = groups[train_i]   # shape of every training row: folds and the final split use it
        _split_note(groups, train_i, test_i, self.random_state)

        return features[train_i], output[train_i], features[test_i], output[test_i]

    def train(self, data_path, refit=True):
        """
        Trains the LGBM regressor and returns metrics (protocol at the top of this module).

        Returns the test-split 'MAE' and 'R2_Score' of the final model and, with k >= 2, the cross-validation
        means 'CV_MAE' and 'CV_R2'. refit=False skips the final fit and returns only the CV scores (used by
        Optuna trials, which never keep the model).

        LightGBM's own messages are silenced (verbosity -1): its frequent "No further splits with positive
        gain, best gain: -inf" only means a tree stopped below num_leaves because no leaf could be split any
        more (min_child_samples); the tree is still used. Each fit reports its best tree count instead, and
        says when early stopping never triggered (all n_estimators used: the cap, not convergence, ended it).

        PARAMETERS:
        - data_path: Same path to csv file as _load_and_prepare_data()
        - refit: Train the final model after cross-validation (default True)
        """

        # Load data
        X, y, X_test, y_test = self._load_and_prepare_data(data_path)
        weights_train = self._sample_weights(y)

        # Scale Features
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        X_test = self.scaler.transform(X_test)

        # Objective function mapping ('combo': custom 80% Huber + 20% exponentially weighted MSE)
        lgbm_objective, eval_metric = _lgbm_objective_and_metric(self.objective)

        def fit_one(train_i, val_i):
            """A fresh model trained on train_i, early-stopped on val_i. Returns (model, best tree count, ran all trees)."""
            model = lgb.LGBMRegressor(
                objective=lgbm_objective,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=-1
            )
            # Callbacks are built per fit (early_stopping holds per-run state).
            # The learning-rate decay callback is only attached when lr_decay is set.
            callbacks = [lgb.early_stopping(self.stopping_rounds, verbose=False)]
            if self.lr_decay is not None:
                callbacks.append(lgb.reset_parameter(learning_rate=lambda iter: self.learning_rate * (self.lr_decay ** iter)))
            model.fit(
                X[train_i], y[train_i],
                sample_weight=weights_train[train_i] if weights_train is not None else None,
                **_lgbm_eval_kwargs(X[val_i], y[val_i]),
                eval_metric=eval_metric,
                callbacks=callbacks
            )
            best = int(model.best_iteration_ or self.n_estimators)
            ran_all = model.booster_.current_iteration() >= self.n_estimators
            return model, best, ran_all

        def tree_note(best, ran_all):
            return f"best tree count {best}" + (" (all trees used: early stopping did not trigger)" if ran_all else "")

        metrics = {}
        if self.k >= 2:
            folds = kfold_by_shape(self.train_groups, self.k, self.random_state)
            scores = {"MAE": [], "R2": []}
            capped = 0
            print(f"Starting LGBM cross-validation: {self.k} folds (by shape), up to {self.n_estimators} trees...")
            for fold, (train_i, val_i) in enumerate(folds, start=1):
                model, best, ran_all = fit_one(train_i, val_i)
                y_preds_fold = model.predict(X[val_i])
                mae = mean_absolute_error(y[val_i], y_preds_fold)
                r2 = r2_score(y[val_i], y_preds_fold)
                print(f"Fold {fold}/{self.k}: MAE {mae:.4f}, R^2 {r2:.4f}, {tree_note(best, ran_all)}")
                scores["MAE"].append(mae)
                scores["R2"].append(r2)
                capped += ran_all
            _print_cv(self.k, scores)
            if capped:
                print(f"[NOTE] {capped} of {self.k} folds used all {self.n_estimators} trees: a larger n_estimators "
                      f"would keep improving the model (and make it slower to predict).")
            metrics = {"CV_MAE": float(np.mean(scores["MAE"])), "CV_R2": float(np.mean(scores["R2"]))}
            if not refit:
                return metrics

        train_i, val_i = _final_split(self.train_groups, self.random_state)
        print(f"\nFinal fit on {len(train_i):,} rows (early stopping on {len(val_i):,})...")
        self.model, best, ran_all = fit_one(train_i, val_i)
        print(f"Final model: {tree_note(best, ran_all)}")
        return {**self._evaluate(X_test, y_test), **metrics}

    def _sample_weights(self, y):
        """Per-row training weights of the 'weight' mode (None: unweighted), normalised to a mean of 1."""
        y = np.asarray(y)
        if self.weight == 'instance':
            weights = y
            weights_train = weights*(len(y)/weights.sum())
            weights = y
            weights_train = weights*(len(y)/weights.sum())
        elif self.weight == 'frequency':
            num_bins = 20
            y_min, y_max = y.min().item(), y.max().item()
            bin_indices = torch.bucketize(torch.tensor(y), torch.linspace(y_min, y_max, num_bins+1), right=False).numpy() - 1
            bin_indices = np.clip(bin_indices, 0, num_bins - 1)
            bin_counts = np.bincount(bin_indices, minlength=num_bins).astype(float)
            inv_freq = np.where(bin_counts == 0, 1.0, 1.0/bin_counts)
            weights = inv_freq[bin_indices]
            weights_train = weights*(len(y)/weights.sum())
        elif self.weight == 'linear':
            ymin, ymax = y.min(), y.max()
            alpha = 4
            weights = 1 + alpha * (y - ymin) / (ymax - ymin)
            weights_train = weights * len(y) / weights.sum()
        elif self.weight == 'exponential':
            beta = 2.0
            weights = np.exp(beta * y / y.max())
            weights_train = weights * len(y) / weights.sum()
        elif self.weight == 'quantile':
            q = 10
            quantiles = np.linspace(0, 1, q + 1)
            bin_edges = np.quantile(y, quantiles)
            bin_edges = np.unique(bin_edges)
            if len(bin_edges) < 2:
                weights_train = np.ones_like(y, dtype=float)
            else:
                bin_indices = np.digitize(y, bin_edges, right=False)
                bin_indices[y == bin_edges[-1]] = len(bin_edges) - 1
                final_indices = bin_indices - 1
                value_counts = np.bincount(final_indices)
                value_counts = np.where(value_counts == 0, 1, value_counts)
                weights_map = 1.0 / value_counts
                weights = weights_map[final_indices]
                weights_train = weights * len(y) / weights.sum()
        elif self.weight == 'polynomial':
            gamma = 3
            weights = (y / y.max()) ** gamma
            weights_train = weights * len(y) / weights.sum()
        else:
            weights_train = None
        return weights_train

    def _transform(self, X):
        """Raw feature rows -> the scaled model inputs (feature expansion + frozen scaler)."""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self.feature_engineer.mode != 'raw':
            X = self.feature_engineer.transform(X)
        return self.scaler.transform(X)

    def fine_tune(self, X_new, y_new, X_anchor, y_anchor, groups_new=None, new_weight=10.0, n_trees=300, lr_factor=0.5,
                  stopping_rounds=30, holdout=0.2, anchor_tolerance=0.05, anchor_abs_tolerance=0.1, random_state=None, from_base=True,
                  force_accept=False, verbose=False):
        """
        Short update with new simulations (see the fine-tuning note at the top of this module): continues boosting
        from the original model (from_base) or the current one with up to n_trees extra trees at lr_factor x the
        learning rate, on the new rows (sample weight new_weight) plus the anchor rows, early-stopped on the held-out
        rows. X_* are raw feature rows (CSV column order), y_* the E values; groups_new names the shape of each new
        row. Returns a report: accepted, MAE of the held-out new shapes and anchor rows before/after, trees added.
        force_accept keeps the update without the gate (used to rebuild an update that was accepted before).
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model is not trained. Please call train() or load_model() first.")
        if getattr(self, "_base_model", None) is None:
            self._base_model = self.model
        previous = self.model
        X_new, y_new = self._transform(X_new), np.asarray(y_new, dtype=np.float32)
        X_anchor, y_anchor = self._transform(X_anchor), np.asarray(y_anchor, dtype=np.float32)
        new_train, new_val, anchor_train, anchor_val = _fine_tune_split(len(y_new), groups_new, len(y_anchor), holdout, random_state)

        def mae(model, X, y):
            return float(mean_absolute_error(y, model.predict(X))) if len(y) else None

        reference = self._base_model if from_base else previous   # the gate's reference (see the fine-tuning note)
        before_new = mae(reference, X_new[new_val], y_new[new_val])
        before_anchor = mae(reference, X_anchor[anchor_val], y_anchor[anchor_val])
        X_train = np.vstack([X_anchor[anchor_train], X_new[new_train]])
        y_train = np.concatenate([y_anchor[anchor_train], y_new[new_train]])
        weights = np.concatenate([np.ones(len(anchor_train)), np.full(len(new_train), float(new_weight))])
        base_weights = self._sample_weights(y_train)
        if base_weights is not None:
            weights = weights * base_weights
        # Early stopping watches the held-out NEW shapes (the region being learned); the anchors are protected by
        # the gate. Without held-out new shapes the held-out anchors early-stop the update.
        val_idx_X, val_idx_y = (X_new[new_val], y_new[new_val]) if len(new_val) else (X_anchor[anchor_val], y_anchor[anchor_val])
        objective, eval_metric = _lgbm_objective_and_metric(self.objective)
        start = self._base_model if from_base else previous
        candidate = lgb.LGBMRegressor(objective=objective, n_estimators=int(n_trees),
                                      learning_rate=self.learning_rate * lr_factor, num_leaves=self.num_leaves,
                                      max_depth=self.max_depth, random_state=self.random_state if random_state is None else random_state,
                                      n_jobs=-1, verbosity=-1)
        fit_kwargs = {}
        if len(val_idx_y):
            fit_kwargs = {**_lgbm_eval_kwargs(val_idx_X, val_idx_y), "eval_metric": eval_metric,
                          "callbacks": [lgb.early_stopping(stopping_rounds, verbose=False)]}
        candidate.fit(X_train, y_train, sample_weight=weights, init_model=start.booster_, **fit_kwargs)
        after_new = mae(candidate, X_new[new_val], y_new[new_val])
        after_anchor = mae(candidate, X_anchor[anchor_val], y_anchor[anchor_val])
        accepted = force_accept or _gate(before_new, after_new, before_anchor, after_anchor, anchor_tolerance,
                                         anchor_abs_tolerance)
        self.model = candidate if accepted else previous
        added = candidate.booster_.current_iteration() - start.booster_.current_iteration()
        report = {"accepted": bool(accepted), "new_rows": int(len(y_new)), "new_held_out": int(len(new_val)),
                  "anchor_rows": int(len(y_anchor)), "mae_new_before": before_new, "mae_new_after": after_new,
                  "mae_anchor_before": before_anchor, "mae_anchor_after": after_anchor, "trees_added": int(added)}
        if verbose:
            print(f"LGBM fine-tune: {'kept' if accepted else 'rejected'} | held-out new MAE {before_new} -> {after_new} | "
                  f"anchor MAE {before_anchor} -> {after_anchor} | {added} trees")
        return report

    def _evaluate(self, X_test, y_test):
        '''
        Evaluates the model on the test dataset, returns metrics.

        PARAMETERS:
        - X_test: Test features
        - y_test: Test targets
        '''
        y_preds = self.model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_preds)
        r2 = r2_score(y_test, y_preds)
        
        # The trainers (tuning scripts, ml_gym) print and save the full test report (ai_support.scoring.score_and_save)
        return {'MAE': mae, 'R2_Score': r2}

    def predict(self, X):
        '''
        Method to predict targets from input data.

        PARAMETERS:
        - X: numpy array or list of datapoints for the regressor, unscaled.
        '''
        if not all([self.model, self.scaler]):
            raise RuntimeError("Model is not trained. Please call train() first.")
        
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # X is expected in the raw CSV feature-column order; expand it with the same
        # feature mode used at fit time. 'raw' skips the step so inputs reach the scaler
        # untouched, exactly as before feature_mode existed.
        if self.feature_engineer.mode != 'raw':
            X = self.feature_engineer.transform(X)

        # Apply transformations to data
        X = self.scaler.transform(X)

        predictions = self.model.predict(X)

        return predictions
        
    def save_model(self, folder_path, file_name_prefix):
        """
        Saves the trained model and the scaler. Creates target directory if it doesn't exist.
        
        PARAMETERS:
        - folder_path: Directory where the data will be stored
        - file_name_prefix: Prefix name to identify data; prefix_model and prefix_scaler
        """
        if not all([self.model, self.scaler]):
            raise RuntimeError("Model not trained yet. Cannot save.")
        
        os.makedirs(folder_path, exist_ok=True)
        
        # Use joblib to save the model and scaler
        joblib.dump(self.model, os.path.join(folder_path, f"{file_name_prefix}_model.joblib"))
        joblib.dump(self.scaler, os.path.join(folder_path, f"{file_name_prefix}_scaler.joblib"))

        # Persist the feature-engineering configuration; predict() needs it to
        # rebuild the same input columns the model was trained on.
        meta_path = os.path.join(folder_path, f"{file_name_prefix}_meta.json")
        with open(meta_path, 'w') as f:
            json.dump({'feature_engineer': self.feature_engineer.to_dict()}, f)

        print(f"LGBM model and scaler saved in folder: {folder_path}")

    @classmethod
    def load_model(cls, folder_path, file_name_prefix):
        """
        Loads a trained model and its scaler.
        
        PARAMETERS:
        - folder_path: Directory where the data is stored
        - file_name_prefix: Prefix name to identify data; prefix_model and prefix_scaler
        """
        # Create a dummy instance to load into
        lgbm_regressor = cls()
        
        model_path = os.path.join(folder_path, f"{file_name_prefix}_model.joblib")
        scaler_path = os.path.join(folder_path, f"{file_name_prefix}_scaler.joblib")
        meta_path = os.path.join(folder_path, f"{file_name_prefix}_meta.json")
        
        lgbm_regressor.model = joblib.load(model_path)
        lgbm_regressor.scaler = joblib.load(scaler_path)

        # Restore the feature-engineering configuration. Models saved without it
        # are treated as 'raw' (no feature expansion).
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            if 'feature_engineer' in meta:
                lgbm_regressor.feature_engineer = CSTFeatureEngineer.from_dict(meta['feature_engineer'])
                lgbm_regressor.feature_mode = lgbm_regressor.feature_engineer.mode
        
        print(f"LGBM model loaded successfully from folder: {folder_path}")
        return lgbm_regressor
    
class SKNN_Regressor:
    """
    A class to encapsulate a Scikit-learn Polynomial Linear Regression model.
    This class uses a pipeline to handle training.
    """
    def __init__(self,
                 degree=2,
                 k=5,
                 test_size=0.2,
                 random_state=42,
                 weight='None'):
        
        # Hyperparameter for Grid Search
        self.degree = degree
        
        # Other training parameters
        self.k = k
        self.test_size = test_size
        self.test_split = test_size # Mapping for _load_and_prepare_data consistency
        self.random_state = random_state
        self.weight = weight

        # The entire pipeline (scaler, poly features, model) will be stored here
        self.model = None
        
        print(f"SKNN_Regressor initialized with polynomial degree={self.degree}.")

    def _load_and_prepare_data(self, data_path):
        """
        Loads data from a single CSV file with a header, assuming the
        last column is the target.

        PARAMETERS:
        - data_path: Full path to csv file that contains training data
        """
        df = pd.read_csv(data_path)

        features = df.iloc[:, :-1].values.astype(np.float32)
        output = df.iloc[:, -1].values.astype(np.float32)

        # Data split for testing, by shape (ai_support.data_split)
        groups = shape_groups(df.iloc[:, :-1])
        train_i, test_i = split_by_shape(groups, self.test_split, self.random_state)
        self.train_groups = groups[train_i]   # shape of every training row: the folds use it
        _split_note(groups, train_i, test_i, self.random_state)

        return features[train_i], output[train_i], features[test_i], output[test_i]
    
    def train(self, data_path, refit=True):
        """
        Trains the SKNN regressor and returns metrics (protocol at the top of this module; with no early
        stopping, the final fit uses the whole training split).

        Returns the test-split 'MAE' and 'R2_Score' of the final model and, with k >= 2, 'CV_MAE' and 'CV_R2'.

        PARAMETERS:
        - data_path: Same path to csv file as _load_and_prepare_data()
        - refit: Train the final model after cross-validation (default True)
        """
        # Load data
        X, y, X_test, y_test = self._load_and_prepare_data(data_path)
        
        # Choose the weight type
        if self.weight == 'instance':
            weights = y
            weights_train = weights*(len(y)/weights.sum())
        elif self.weight == 'frequency':
            num_bins = 20
            y_min, y_max = y.min().item(), y.max().item()
            bin_indices = torch.bucketize(torch.tensor(y), torch.linspace(y_min, y_max, num_bins+1), right=False).numpy() - 1
            bin_counts = np.bincount(bin_indices, minlength=num_bins).astype(float)
            inv_freq = np.where(bin_counts == 0, 1.0, 1.0/bin_counts)
            weights = inv_freq[bin_indices]
            weights_train = weights*(len(y)/weights.sum())
        elif self.weight == 'linear':
            ymin, ymax = y.min(), y.max()
            alpha = 4
            weights = 1 + alpha * (y - ymin) / (ymax - ymin)
            weights_train = weights * len(y) / weights.sum()
        elif self.weight == 'exponential':
            beta = 2.0
            weights = np.exp(beta * y / y.max())
            weights_train = weights * len(y) / weights.sum()
        elif self.weight == 'quantile':
            q = 10
            quantiles = np.linspace(0, 1, q + 1)
            bin_edges = np.quantile(y, quantiles)
            bin_edges = np.unique(bin_edges)
            if len(bin_edges) < 2:
                weights_train = np.ones_like(y, dtype=float)
            else:
                bin_indices = np.digitize(y, bin_edges, right=False)
                bin_indices[y == bin_edges[-1]] = len(bin_edges) - 1
                final_indices = bin_indices - 1
                value_counts = np.bincount(final_indices)
                value_counts = np.where(value_counts == 0, 1, value_counts)
                weights_map = 1.0 / value_counts
                weights = weights_map[final_indices]
                weights_train = weights * len(y) / weights.sum()
        elif self.weight == 'polynomial':
            gamma = 3
            weights = (y / y.max()) ** gamma
            weights_train = weights * len(y) / weights.sum()
        else:
            weights_train = None

        def fit_one(train_i):
            model = make_pipeline(StandardScaler(), PolynomialFeatures(degree=self.degree), LinearRegression())
            if weights_train is not None:
                model.fit(X[train_i], y[train_i], linearregression__sample_weight=weights_train[train_i])
            else:
                model.fit(X[train_i], y[train_i])
            return model

        metrics = {}
        if self.k >= 2:
            folds = kfold_by_shape(self.train_groups, self.k, self.random_state)
            scores = {"MAE": [], "R2": []}
            print(f"Starting SKNN cross-validation: {self.k} folds (by shape), degree {self.degree} polynomial...")
            for fold, (train_i, val_i) in enumerate(folds, start=1):
                model = fit_one(train_i)
                y_preds_fold = model.predict(X[val_i])
                mae = mean_absolute_error(y[val_i], y_preds_fold)
                r2 = r2_score(y[val_i], y_preds_fold)
                print(f"Fold {fold}/{self.k}: MAE {mae:.4f}, R^2 {r2:.4f}")
                scores["MAE"].append(mae)
                scores["R2"].append(r2)
            _print_cv(self.k, scores)
            metrics = {"CV_MAE": float(np.mean(scores["MAE"])), "CV_R2": float(np.mean(scores["R2"]))}
            if not refit:
                return metrics

        self.model = fit_one(np.arange(len(y)))
        return {**self._evaluate(X_test, y_test), **metrics}

    def _evaluate(self, X_test, y_test):
        """
        Calculates and returns performance metrics on the test set.

        PARAMETERS:
        - X_test: Test features 
        - y_test: Test targets 
        """

        # The pipeline handles the scaling and transformations automatically
        y_preds = self.model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_preds)
        r2 = r2_score(y_test, y_preds)
        
        # The trainers (tuning scripts) print and save the full test report (ai_support.scoring.score_and_save)
        return {'MAE': mae, 'R2_Score': r2}
    
    def predict(self, X):
        """
        Makes predictions on new, un-transformed data. In this case this method is mostly created for consistency.

        PARAMETERS:
        - X: numpy array or list of datapoints for the regressor, unscaled.
        """
        if self.model is None:
            raise RuntimeError("Model is not trained. Please call train() first.")
        
        return self.model.predict(X)
    
    def save_model(self, folder_path, file_name_prefix):
        """
        Saves the entire trained pipeline. Creates target directory if it doesn't exist.
        
        PARAMETERS:
        - folder_path: Directory where the data will be stored
        - file_name_prefix: Prefix name to identify data; prefix_pipeline
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet. Cannot save.")
        
        os.makedirs(folder_path, exist_ok=True)
        
        # Save the entire pipeline object
        model_path = os.path.join(folder_path, f"{file_name_prefix}_pipeline.joblib")
        joblib.dump(self.model, model_path)

        print(f"SKNN pipeline saved to: {model_path}")

    @classmethod
    def load_model(cls, folder_path, file_name_prefix):
        """
        Loads a trained pipeline.
        
        PARAMETERS:
        - folder_path: Directory where the data is stored
        - file_name_prefix: Prefix name to identify data; prefix_pipeline
        """
        model_path = os.path.join(folder_path, f"{file_name_prefix}_pipeline.joblib")
        
        loaded_model = joblib.load(model_path)
        
        # Create a class instance and assign the loaded model to it
        # We can find the degree from the loaded pipeline's parameters
        degree = loaded_model.get_params()['polynomialfeatures__degree']
        sknn_regressor = cls(degree=degree)
        sknn_regressor.model = loaded_model
        
        print(f"SKNN pipeline loaded successfully from: {model_path}")

        return sknn_regressor
    
class _PyTorchMLPClassifier(nn.Module):
    """
    Internal PyTorch model for the classifier.
    """
    def __init__(self, input_dim, hidden_layer_sizes):
        super().__init__()
        layers = []
        current_dim = input_dim

        for h_size in hidden_layer_sizes:
            layers.append(nn.Linear(current_dim, h_size))
            layers.append(nn.ReLU())
            current_dim = h_size

        # Output layer
        layers.append(nn.Linear(current_dim, 1))
        
        # Unpack list into Sequential
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Classifier:
    """
    A class to encapsulate the MLP classifier. 
    Handles data scaling, Stratified K-Fold cross-validation, learning rate scheduling via ReduceLROnPlateau, and early stopping based on validation loss.
    """
    def __init__(self, 
                 hidden_layer_sizes=[128], 
                 epochs=400, 
                 lr=0.075,
                 patience=40,
                 factor=0.5,
                 batch_size=2048,
                 k=5,
                 test_size=0.2, 
                 random_state=42):
        
        # Hyperparameters
        self.hidden_layer_sizes = hidden_layer_sizes
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self.factor = factor
        # Mini-batch size, a hyperparameter like the regressors'. Until 2026-11-11 the batch was the largest one
        # that fitted in GPU memory (in practice the whole training set), so an epoch was a single gradient step.
        self.batch_size = batch_size

        # Other training parameters
        self.k = k
        self.test_size = test_size
        self.random_state = random_state

        # Class components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler = None
        self.input_dim = None
        self.feature_columns = None
        print(f"Classifier initialized. Training will use device: {self.device}")

    def _load_and_prepare_data(self, data_path):
        """
        Loads data from the CSV file, assuming it contains features
        and a target column named 'valid'.

        PARAMETERS:
        - data_path: Full path to csv file that contains training data
        """
        df = pd.read_csv(data_path)

        if 'valid' not in df.columns:
            raise ValueError("Input CSV must contain a 'valid' column for classification.")

        # One row per shape: classifier CSVs written before 2026-11-10 repeated every valid shape once per wing
        # configuration, which put the same shape in the training, validation and test rows.
        rows_read = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        if len(df) < rows_read:
            print(f"[NOTE] {rows_read - len(df):,} repeated rows of '{os.path.basename(str(data_path))}' removed "
                  f"({rows_read:,} -> {len(df):,}): a classifier dataset has one row per shape.")

        # Identify target and features
        feature_df = df.drop(columns=['valid'])
        y = df['valid'].values.astype(np.float32)
        X = feature_df.values.astype(np.float32)

        self.input_dim = X.shape[1]
        self.feature_columns = feature_df.columns.tolist()

        # Split data by shape, keeping the share of valid and invalid shapes on both sides (ai_support.data_split)
        groups = shape_groups(feature_df)
        train_i, test_i = split_by_shape(groups, self.test_size, self.random_state, stratify=y)
        self.train_groups = groups[train_i]   # shape of every training row: folds and the final split use it
        _split_note(groups, train_i, test_i, self.random_state)
        print(f"      valid shapes {int((y == 1).sum()):,} | invalid shapes {int((y == 0).sum()):,}")
        X, X_test, y, y_test = X[train_i], X[test_i], y[train_i], y[test_i]

        # Scale features
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        X_test = self.scaler.transform(X_test)

        # Convert to Tensors
        X = torch.from_numpy(X).to(self.device)
        y = torch.from_numpy(y.reshape(-1, 1)).to(self.device)
        X_test = torch.from_numpy(X_test).to(self.device)
        y_test = torch.from_numpy(y_test.reshape(-1, 1)).to(self.device)

        return X, y, X_test, y_test

    def train(self, data_path, refit=True):
        """
        Trains the classifier and returns metrics (protocol at the top of this module; folds and the final
        early-stopping slice are stratified by class).

        Returns the test-split metrics of the final model and, with k >= 2, the cross-validation means
        'CV_Accuracy', 'CV_Precision', 'CV_Recall' and 'CV_F1'. refit=False skips the final fit and returns
        only the CV scores (used by Optuna trials, which never keep the model).

        PARAMETERS:
        - data_path: Same data path used for the CSV in _load_and_prepare_data()
        - refit: Train the final model after cross-validation (default True)
        """
        # Load data
        X, y, X_test, y_test = self._load_and_prepare_data(data_path)
        print(f"Data loaded successfully from {data_path}. Input features: {self.input_dim}")

        # Detach tensors for Stratified K-Fold logic
        y_np = y.cpu().numpy().flatten()

        def run(train_i, val_i, label):
            """Trains a fresh model on train_i, early-stopped on val_i. Returns (validation metrics, best epoch)."""
            X_train_fold, X_val_fold = X[train_i], X[val_i]
            y_train_fold, y_val_fold = y[train_i], y[val_i]

            # Create dataloaders using the dynamically calculated batch size
            train_loader = DataLoader(TensorDataset(X_train_fold, y_train_fold), batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(TensorDataset(X_val_fold, y_val_fold), batch_size=self.batch_size, shuffle=False)

            # Initialize model, optimizer, and scheduler
            self.model = _PyTorchMLPClassifier(input_dim=self.input_dim, hidden_layer_sizes=self.hidden_layer_sizes).to(self.device)
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=self.patience // 2, factor=self.factor)

            neg_count = (y_train_fold == 0).sum().item()
            pos_count = (y_train_fold == 1).sum().item()
            if pos_count > 0 and neg_count > 0:
                weight_val = neg_count / pos_count
            else:
                weight_val = 1.0
            weight = torch.tensor([weight_val], device=self.device, dtype=torch.float32)

            # Loss for training (weighted) and validation (unweighted)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)
            loss_fn_eval = nn.BCEWithLogitsLoss()

            best_val_loss = float('inf')
            early_stop_counter = 0
            best_model_state = None
            best_epoch = 0

            # Training loop
            for epoch in range(self.epochs):
                self.model.train()
                epoch_train_loss = 0.0

                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    pred = self.model(X_batch)
                    loss = loss_fn(pred, y_batch)
                    loss.backward()
                    optimizer.step()
                    epoch_train_loss += loss.item()

                # Validation phase
                self.model.eval()
                val_loss_cum = 0.0
                with torch.no_grad():
                    for X_val_batch, y_val_batch in val_loader:
                        val_pred = self.model(X_val_batch)
                        val_loss = loss_fn_eval(val_pred, y_val_batch)
                        val_loss_cum += val_loss.item()

                avg_val_loss = val_loss_cum / len(val_loader)
                scheduler.step(avg_val_loss)

                # Early stopping check
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    early_stop_counter = 0
                    best_epoch = epoch + 1
                    # A copy: state_dict() returns the live tensors, which later epochs keep changing
                    best_model_state = copy.deepcopy(self.model.state_dict())
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= self.patience:
                        print(f"Early stopping triggered at epoch {epoch + 1}. Best validation loss: {best_val_loss:.4f}")
                        break

                if (epoch + 1) % 100 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    avg_train_loss = epoch_train_loss / len(train_loader)
                    print(f'{label}, Epoch [{epoch+1}/{self.epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}')

            # Restore best model weights
            if best_model_state:
                self.model.load_state_dict(best_model_state)

            # Validation metrics
            self.model.eval()
            val_pred_labels_list = []
            y_val_true_list = []
            with torch.no_grad():
                for X_val_batch, y_val_batch in val_loader:
                    val_pred_probs = torch.sigmoid(self.model(X_val_batch))
                    val_pred_labels_list.extend((val_pred_probs > 0.5).cpu().numpy())
                    y_val_true_list.extend(y_val_batch.cpu().numpy())
            y_true = np.array(y_val_true_list)
            val_pred_labels = np.array(val_pred_labels_list)
            full = classification_metrics(y_true, val_pred_labels)
            # Valid-class scores (the tuning script ranks on F1) plus the two that show the invalid class
            scores = {"Accuracy": full["Accuracy"], "Precision": full["Precision"], "Recall": full["Recall"],
                      "F1": full["F1-Score"], "Invalid_Recall": full["Invalid_Recall"],
                      "Balanced_Accuracy": full["Balanced_Accuracy"]}
            return scores, best_epoch

        metrics = {}
        if self.k >= 2:
            folds = kfold_by_shape(self.train_groups, self.k, self.random_state, stratify=y_np)
            scores = {"Accuracy": [], "Precision": [], "Recall": [], "F1": [], "Invalid_Recall": [], "Balanced_Accuracy": []}
            print(f"Starting classifier cross-validation: {self.k} folds (by shape, stratified), batch size {self.batch_size}...")
            for fold, (train_i, val_i) in enumerate(folds, start=1):
                fold_scores, best_epoch = run(train_i, val_i, f"Fold {fold}")
                print(f"Fold {fold}/{self.k}: " + ", ".join(f"{k} {v:.4f}" for k, v in fold_scores.items()) + f", best epoch {best_epoch}")
                for name, value in fold_scores.items():
                    scores[name].append(value)
            _print_cv(self.k, scores)
            metrics = {f"CV_{name}": float(np.mean(values)) for name, values in scores.items()}
            if not refit:
                return metrics

        train_i, val_i = _final_split(self.train_groups, self.random_state, stratify=y_np)
        print(f"\nFinal fit on {len(train_i):,} rows (early stopping on {len(val_i):,})...")
        _, best_epoch = run(train_i, val_i, "Final fit")
        print(f"Final model: best epoch {best_epoch}")
        return {**self._evaluate(X_test, y_test), **metrics}

    def _evaluate(self, X_test, y_test):
        """
        Evaluates the model on the test dataset, returns metrics.

        PARAMETERS:
        - X_test: Test features tensor
        - y_test: Test targets tensor
        """
        if not all([self.model, self.scaler]):
            raise RuntimeError("Model is not trained. Please call train() first.")

        self.model.eval()
        with torch.no_grad():
            y_pred_logits = self.model(X_test)
            y_pred_probs = torch.sigmoid(y_pred_logits)
            y_pred_labels = (y_pred_probs > 0.5).cpu().numpy()
        
        y_true = y_test.cpu().numpy()

        # Valid-class scores, invalid-class scores and the confusion matrix (ai_support.scoring). The trainers
        # (tuning scripts, ml_gym) print and save the full test report with ai_support.scoring.score_and_save.
        return classification_metrics(y_true, y_pred_labels)

    def predict(self, X):
        """
        Predicts the probability of validity for new data.
        
        PARAMETERS:
        - X: numpy array or list of datapoints for the classifier, unscaled.
        """
        if not all([self.model, self.scaler]):
            raise RuntimeError("Model is not trained. Please call train() first.")
            
        if isinstance(X, pd.DataFrame):
            X = X.values

        if X.ndim == 1:
            X = X.reshape(1, -1)

        self.model.eval()
        X = self.scaler.transform(X)
        X = torch.from_numpy(X).to(torch.float32).to(self.device)
        
        with torch.no_grad():
            logits = self.model(X)
            probabilities = torch.sigmoid(logits)
        
        return probabilities.cpu().numpy()

    def save_model(self, folder_path, file_name_prefix):
        """
        Saves the model state, scaler, and metadata.
        """
        if not all([self.model, self.scaler]):
            print("No model to save. Train a model first!")
            return
        
        os.makedirs(folder_path, exist_ok=True)
        model_path = os.path.join(folder_path, f"{file_name_prefix}_model.pth")
        scaler_path = os.path.join(folder_path, f"{file_name_prefix}_scaler.joblib")
        metadata_path = os.path.join(folder_path, f"{file_name_prefix}_meta.json")
        
        torch.save(self.model.state_dict(), model_path)
        joblib.dump(self.scaler, scaler_path)
        
        with open(metadata_path, 'w') as f:
            json.dump({'input_dim': self.input_dim}, f)

        print(f"Model saved in folder: {folder_path} with prefix: {file_name_prefix}")

    @classmethod
    def load_model(cls, folder_path, file_name_prefix, **kwargs):
        """
        Loads a model, scaler, and metadata from files.

        PARAMETERS:
        - folder_path: Directory where the data is stored
        - file_name_prefix: Prefix name to identify data; prefix_model, prefix_scaler and prefix_meta
        """
        model_path = os.path.join(folder_path, f"{file_name_prefix}_model.pth")
        scaler_path = os.path.join(folder_path, f"{file_name_prefix}_scaler.joblib")
        metadata_path = os.path.join(folder_path, f"{file_name_prefix}_meta.json")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        loaded_classifier = cls(**kwargs)
        loaded_classifier.input_dim = metadata['input_dim']
        
        loaded_classifier.model = _PyTorchMLPClassifier(
            input_dim=loaded_classifier.input_dim, 
            hidden_layer_sizes=loaded_classifier.hidden_layer_sizes
        )
        state_dict = torch.load(model_path, map_location=loaded_classifier.device, weights_only=True)
        loaded_classifier.model.load_state_dict(state_dict)
        loaded_classifier.model.to(loaded_classifier.device)
        loaded_classifier.model.eval()

        loaded_classifier.scaler = joblib.load(scaler_path)

        print(f"Classifier loaded successfully from folder: {folder_path}")
        return loaded_classifier