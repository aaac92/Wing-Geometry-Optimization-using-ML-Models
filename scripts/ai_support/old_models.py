import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from torch.utils.data import DataLoader, TensorDataset
import joblib
import json

class _PyTorchMLPClassifier(nn.Module):
    """Internal PyTorch model for the classifier."""
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.net(x)

class Classifier:
    """
    An class for the MLP classifier
    """
    def __init__(self, hidden_size=128, epochs=400, lr=0.075, test_size=0.2, random_state=42):
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr
        self.test_size = test_size
        self.random_state = random_state
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
        """
        df = pd.read_csv(data_path)

        if 'valid' not in df.columns:
            raise ValueError("Input CSV must contain a 'valid' column for classification.")

        # Identify target and features
        y = df['valid'].values.astype(np.float32)
        X = df.drop(columns=['valid']).values.astype(np.float32)

        self.input_dim = X.shape[1]
        self.feature_columns = df.drop(columns=['valid']).columns.tolist()

        return X, y

    def train(self, data_path):
        """
        Executes the full training pipeline using a single data file.
        """
        # Load data
        X, y = self._load_and_prepare_data(data_path)
        print(f"Data loaded successfully from {data_path}. Input features: {self.input_dim}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Convert to Tensors
        X_train_t = torch.from_numpy(X_train_scaled).to(self.device)
        y_train_t = torch.from_numpy(y_train.reshape(-1, 1)).to(self.device)
        self.X_test_t = torch.from_numpy(X_test_scaled).to(self.device)
        self.y_test_t = torch.from_numpy(y_test.reshape(-1, 1)).to(self.device)

        # Initialize model, loss, and optimizer
        self.model = _PyTorchMLPClassifier(input_dim=self.input_dim, hidden_size=self.hidden_size).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        neg_count = (y_train == 0).sum().item()
        pos_count = (y_train == 1).sum().item()
        pos_weight = torch.tensor([pos_count / neg_count], device=self.device) if pos_count > 0 else torch.tensor([1.0])
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Training loop
        print("Starting classifier training...")
        for epoch in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            pred = self.model(X_train_t)
            loss = loss_fn(pred, y_train_t)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}')
        
        print("Classifier training complete.")
        return self.evaluate()

    def evaluate(self):
        """Evaluates the model and prints metrics."""
        if self.model is None or self.X_test_t is None or self.y_test_t is None:
            print("Model has not been trained yet. Call train() first.")
            return

        self.model.eval()
        with torch.no_grad():
            y_pred_logits = self.model(self.X_test_t)
            y_pred_probs = torch.sigmoid(y_pred_logits)
            y_pred_labels = (y_pred_probs > 0.5).cpu().numpy()
        
        y_true = self.y_test_t.cpu().numpy()
        
        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred_labels),
            "Precision": precision_score(y_true, y_pred_labels),
            "Recall": recall_score(y_true, y_pred_labels),
            "F1-Score": f1_score(y_true, y_pred_labels),
            "F2-Score": fbeta_score(y_true, y_pred_labels, beta=2.0),
        }
        
        print("\n--- Model Evaluation on Test Set ---")
        for name, value in metrics.items():
            print(f"{name:<10}: {value:.4f}")
        print("------------------------------------\n")
        
        return metrics

    def predict_proba(self, X_new):
        """Predicts the probability of validity for new data."""
        if not all([self.model, self.scaler]):
            raise RuntimeError("Model is not trained. Please call train() first.")
            
        if isinstance(X_new, pd.DataFrame):
            X_new = X_new.values

        if X_new.ndim == 1:
            X_new = X_new.reshape(1, -1)

        self.model.eval()
        X_scaled = self.scaler.transform(X_new)
        X_t = torch.from_numpy(X_scaled).to(torch.float32).to(self.device)
        
        with torch.no_grad():
            logits = self.model(X_t)
            probabilities = torch.sigmoid(logits)
        
        return probabilities.cpu().numpy()

    def save_model(self, folder_path, file_name_prefix):
        """Saves the model state, scaler, and metadata."""
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
        """Loads a model, scaler, and metadata from files."""
        model_path = os.path.join(folder_path, f"{file_name_prefix}_model.pth")
        scaler_path = os.path.join(folder_path, f"{file_name_prefix}_scaler.joblib")
        metadata_path = os.path.join(folder_path, f"{file_name_prefix}_meta.json")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        loaded_classifier = cls(**kwargs)
        loaded_classifier.input_dim = metadata['input_dim']
        
        loaded_classifier.model = _PyTorchMLPClassifier(input_dim=loaded_classifier.input_dim, hidden_size=loaded_classifier.hidden_size)
        loaded_classifier.model.load_state_dict(torch.load(model_path))
        loaded_classifier.model.to(loaded_classifier.device)
        loaded_classifier.model.eval()

        loaded_classifier.scaler = joblib.load(scaler_path)

        print(f"Classifier loaded successfully from folder: {folder_path}")
        return loaded_classifier

class _PyTorchMLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_size, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        return self.net(x)

class MLP_Regressor:
    """
    A class to encapsulate the MLP Regressor for predicting the output value.
    Handles polynomial feature creation, scaling, and an advanced training
    loop with early stopping.
    """
    def __init__(self,
                 hidden_size=384,
                 dropout_rate=0.8,
                 learning_rate=0.05,
                 noise_level=0.05,
                 epochs=600,
                 patience=40,
                 poly_degree=2,
                 test_size=0.2,
                 random_state=42,
                 weight=None,
                 os_mode = 'None',
                 rep_factor=10,
                 log_trans=False):
        
        # Hyperparameters
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.noise_level = noise_level
        
        # Other training parameters
        self.epochs = epochs
        self.patience = patience
        self.weight = weight

        # Data manipulation parameters
        self.poly_degree = poly_degree
        self.test_size = test_size
        self.random_state = random_state
        self.os_mode = os_mode
        self.rep_factor = rep_factor

        # Class components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.poly_transformer = None
        self.scaler = None
        self.input_dim = None # Will be determined after polynomial transformation
        self.log_transform = log_trans
        
        print(f"MLP Regressor initialized. Training will use device: {self.device}")

    def _load_and_prepare_data(self, data_path):
        """
        Loads data from a single CSV file with a header, assuming the
        last column is the target.
        """
        df = pd.read_csv(data_path)

        # Oversampling
        if self.os_mode == 'None':
            X = df.iloc[:, :-1].values.astype(np.float32)
            y = df.iloc[:, -1].values.astype(np.float32) # LightGBM prefers a 1D array for y

        if self.os_mode == 'top':
        # Identify top % performers based on target column (last column)
            y = df.iloc[:, -1]
            top = 90
            threshold = np.percentile(y, top)  # top % cutoff
            df_top = df[y >= threshold]

            # Oversample top performers
            df_augmented = pd.concat([df, pd.concat([df_top]*self.rep_factor, ignore_index=True)], ignore_index=True)
            print(f"Original dataset size: {len(df)}")
            print(f"Augmented dataset size: {len(df_augmented)} (with top {100-top}% duplicated {self.rep_factor}x)")

            # All columns except the last one are features
            X = df_augmented.iloc[:, :-1].values.astype(np.float32)
            # The very last column is the target
            y = df_augmented.iloc[:, -1].values.astype(np.float32) 

        elif self.os_mode == 'hybrid':
            y = df.iloc[:, -1]
            top = 90
            mid = 75
            low = 50
            threshold_upper = np.percentile(y, top)  # top % cutoff
            threshold_mid = np.percentile(y, mid)
            threshold_bot = np.percentile(y, low)
            df_top = df[y >= threshold_upper]
            df_mid = df[(y >= threshold_mid) & (y < threshold_upper)]
            df_bot = df[(y >= threshold_bot) & (y < threshold_mid)]

            # Oversample top performers
            df_augmented_top = pd.concat([df, pd.concat([df_top]*self.rep_factor, ignore_index=True)], ignore_index=True)
            df_augmented_mid = pd.concat([df_augmented_top, pd.concat([df_mid]*(int(self.rep_factor/3)), ignore_index=True)], ignore_index=True)
            df_augmented_hybrid = pd.concat([df_augmented_mid, pd.concat([df_bot]*(int(self.rep_factor/5)), ignore_index=True)], ignore_index=True)
            print(f"Original dataset size: {len(df)}")
            print(f"Augmented Hybrid dataset size: {len(df_augmented_hybrid)} \n(with top {100-top}% duplicated {self.rep_factor}x, \nmid {mid}% to {top}% duplicated {int(self.rep_factor/3)}x and \nbot {low}% to mid {mid}% duplicated {int(self.rep_factor/5)}x)")

            # All columns except the last one are features
            X = df_augmented_hybrid.iloc[:, :-1].values.astype(np.float32)
            # The very last column is the target
            y = df_augmented_hybrid.iloc[:, -1].values.astype(np.float32) # LightGBM prefers a 1D array for y
    
        return X, y

    def train(self, data_path):
        """
        Executes the full training pipeline for the regressor.
        Returns a dictionary of performance metrics on the test set.
        """
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

                else:
                    raise ValueError(f"Unknown mode: {mode}")

                return torch.mean(((y_pred - y_true) ** 2) * weights)

            return weighted_mse

        def combo_loss(y_pred, y_true, lam=0.8):

            w_mse = make_weighted_mse(power=3.5, mode='exponential')
            huber = nn.SmoothL1Loss()

            return lam * huber(y_pred, y_true) + (1-lam) * w_mse(y_pred, y_true)
        
        # Load data
        X, y = self._load_and_prepare_data(data_path)

        # Logarithmic output transformation
        if self.log_transform == True:
            y_log = np.log1p(y)
            # Split log data
            X_train, X_test, y_log_train, y_log_test = train_test_split(
                X, y_log, test_size=self.test_size, random_state=self.random_state
            )
        else:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )

        # Polynomial Features
        self.poly_transformer = PolynomialFeatures(degree=self.poly_degree, include_bias=False)
        X_train_poly = self.poly_transformer.fit_transform(X_train)
        X_test_poly = self.poly_transformer.transform(X_test)
        self.input_dim = X_train_poly.shape[1]
        
        # Scale Features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_poly)
        X_test_scaled = self.scaler.transform(X_test_poly)

        # Convert to Tensors
        X_train_t = torch.from_numpy(X_train_scaled).to(self.device)
        X_test_t = torch.from_numpy(X_test_scaled).to(self.device)
        if self.log_transform == True:
            y_train_t = torch.from_numpy(y_log_train).to(self.device)
            y_test_t = torch.from_numpy(y_log_test).to(self.device)
        else:
            y_train_t = torch.from_numpy(y_train).to(self.device)
            y_test_t = torch.from_numpy(y_test).to(self.device)

        # Obtain weights for some features
        if self.weight == 'instance':
            # This uses the target value as the weight
            weights_train = y_train_t
        if self.weight == 'frequency':
            # This uses the target value frequency as the weight
            y_train_series = pd.Series(y_train.flatten())
            value_counts = y_train_series.value_counts()
            weights_map = 1 / value_counts
            weights = y_train_series.map(weights_map)
            weight = weights*len(y_train) / weights.sum()
            weights_train = torch.tensor(weight.values, dtype=torch.float32).to(self.device)
        else:
            weights_train = torch.ones(len(y_train), dtype=torch.float32)

        # Initialize Model, Loss, Optimizer, and Scheduler
        self.model = _PyTorchMLPRegressor(
            input_dim=self.input_dim,
            hidden_size=self.hidden_size,
            dropout_rate=self.dropout_rate
        ).to(self.device)
        
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
        #loss_fn = nn.PoissonNLLLoss(reduction='none')
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate) # , weight_decay=1e-4
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=self.patience // 2, factor=0.5)

        # Training Loop with Early Stopping
        best_val_loss = float('inf')
        early_stop_counter = 0
        best_model_state = None

        # Create dataset from tensors
        train_dataset = TensorDataset(X_train_t, y_train_t, weights_train)
        # DataLoader batching + shuffling
        train_loader = DataLoader(train_dataset, batch_size=524288, shuffle=True)

        print(f"Starting regressor training for up to {self.epochs} epochs...")
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            
            for X_batch, y_batch, weights_batch_t in train_loader:
                X_batch, y_batch, weights_batch_t = X_batch.to(self.device), y_batch.to(self.device), weights_batch_t.to(self.device)

                # Ensure target tensors have the right shape [N, 1] to match model output
                y_batch = y_batch.view(-1, 1)

                # Data augmentation 
                noise = torch.randn_like(X_batch) * self.noise_level
                X_batch_aug = X_batch + noise

                # Forward pass
                pred = self.model(X_batch_aug)

                if self.weight == 'instance' or self.weight == 'frequency':
                    weights_batch_t_reshaped = weights_batch_t.view(-1, 1)

                    # Calculate the raw, per-sample loss, used for some weight modes
                    unweighted_loss = loss_fn(pred, y_batch)
                
                # Apply weights
                if self.weight == 'instance' or self.weight == 'frequency':
                    weighted_loss = unweighted_loss*weights_batch_t_reshaped
                if self.weight == 'combo':
                    weighted_loss = combo_loss(pred, y_batch)
                if self.weight == 'exponential' or self.weight == 'linear' or self.weight == 'None':
                    loss = loss_fn(pred, y_batch)

                # Take the mean to get the final loss for backpropagation
                if self.weight == 'instance' or self.weight == 'frequency' or self.weight == 'combo':
                    loss = weighted_loss.mean()

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_test_t)
                y_test_t_reshaped = y_test_t.view(-1, 1)
                
                loss_fn_eval = nn.MSELoss(reduction='mean')
                val_loss = loss_fn_eval(val_pred, y_test_t_reshaped)
            
            scheduler.step(val_loss)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                best_model_state = self.model.state_dict() # Save the best model
            else:
                early_stop_counter += 1
                if early_stop_counter >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}. Best validation loss: {best_val_loss:.4f}")
                    break
        
        print("Training complete.")
        # Load the best model state found during training
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        return self.evaluate(X_test_t, y_test_t)

    def evaluate(self, X_test_t, y_test_t):
        """Calculates and returns performance metrics on the test set."""
        self.model.eval()
        with torch.no_grad():
            if self.log_transform:
                log_preds_tensor = self.model(X_test_t)
            else:
                preds_tensor = self.model(X_test_t)
        
        # Move tensors to CPU and convert to NumPy for scikit-learn metrics
        # Note: y_test_t contains the log-transformed true values if self.log_transform == True
        if self.log_transform == True:
            y_log_true_np = y_test_t.cpu().numpy()
            y_log_preds_np = log_preds_tensor.cpu().numpy()
        else:
            y_true_np = y_test_t.cpu().numpy()
            y_preds_np = preds_tensor.cpu().numpy()

        # Inverse transform BOTH predictions and true values to get them back to the original scale for self.log_transform == True
        if self.log_transform == True:
            y_true_original = np.expm1(y_log_true_np)
            y_preds_original = np.expm1(y_log_preds_np)

            mae = mean_absolute_error(y_true_original, y_preds_original)
            r2 = r2_score(y_true_original, y_preds_original)
        else:
            mae = mean_absolute_error(y_true_np, y_preds_np)
            r2 = r2_score(y_true_np, y_preds_np)
        
        metrics = {'MAE': mae, 'R2_Score': r2}
        
        print("\n--- Regressor Evaluation on Test Set ---")
        for name, value in metrics.items():
            print(f"{name:<10}: {value:.4f}")
        print("----------------------------------------\n")
        
        return metrics

    def predict(self, X_new):
        """Makes predictions on new, un-transformed data."""
        if not all([self.model, self.poly_transformer, self.scaler]):
            raise RuntimeError("Model is not trained. Please call train() first.")
        
        if X_new.ndim == 1:
            X_new = X_new.reshape(1, -1)

        # Apply the same transformations as in training
        X_poly = self.poly_transformer.transform(X_new)
        X_scaled = self.scaler.transform(X_poly)
        X_t = torch.from_numpy(X_scaled).to(torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            predictions_t = self.model(X_t)
        if self.log_transform == True:
            log_predictions_np = predictions_t.cpu().numpy()
            predictions = np.expm1(log_predictions_np)
        else:
            predictions = predictions_t.cpu().numpy()
            
        return predictions
        
    def save_model(self, folder_path, file_name_prefix):
        """Saves the model, polynomial transformer, scaler, and metadata."""
        if not all([self.model, self.poly_transformer, self.scaler]):
            raise RuntimeError("Model not trained yet. Cannot save.")
        
        os.makedirs(folder_path, exist_ok=True)
        
        torch.save(self.model.state_dict(), os.path.join(folder_path, f"{file_name_prefix}_model.pth"))
        joblib.dump(self.poly_transformer, os.path.join(folder_path, f"{file_name_prefix}_poly.joblib"))
        joblib.dump(self.scaler, os.path.join(folder_path, f"{file_name_prefix}_scaler.joblib"))

        metadata = {'input_dim': self.input_dim}
        with open(os.path.join(folder_path, f"{file_name_prefix}_meta.json"), 'w') as f:
            json.dump(metadata, f)

        print(f"Regressor model and transformers saved in folder: {folder_path}")

    @classmethod
    def load_model(cls, folder_path, file_name_prefix, **kwargs):
        """Loads a model and its associated transformers."""
        with open(os.path.join(folder_path, f"{file_name_prefix}_meta.json"), 'r') as f:
            metadata = json.load(f)

        # Re-create the class instance, passing any saved hyperparameters
        regressor = cls(**kwargs)
        regressor.input_dim = metadata['input_dim']

        # Load transformers first
        regressor.poly_transformer = joblib.load(os.path.join(folder_path, f"{file_name_prefix}_poly.joblib"))
        regressor.scaler = joblib.load(os.path.join(folder_path, f"{file_name_prefix}_scaler.joblib"))

        # Instantiate model with correct dimensions and load weights
        regressor.model = _PyTorchMLPRegressor(
            input_dim=regressor.input_dim,
            hidden_size=regressor.hidden_size,
            dropout_rate=regressor.dropout_rate
        )
        regressor.model.load_state_dict(torch.load(os.path.join(folder_path, f"{file_name_prefix}_model.pth")))
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
                 num_leaves=20,
                 max_depth=10,
                 stopping_rounds=100,
                 test_size=0.2,
                 random_state=42,
                 objective = 'regression_l2',
                 weight='None',
                 os_mode='None',
                 rep_factor=0):
        
        # Hyperparameters for Grid Search
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.weight = weight
        
        # Other training parameters
        self.stopping_rounds = stopping_rounds
        self.test_size = test_size
        self.random_state = random_state
        self.rep_factor = rep_factor

        # Class components
        self.model = None
        self.scaler = None
        self.objective = objective
        self.os_mode = os_mode
        
        print("LGBMRegressor initialized.")

    def _load_and_prepare_data(self, data_path):
        """
        Loads data from a single CSV file with a header, assuming the
        last column is the target.
        """
        df = pd.read_csv(data_path)

        # Oversampling
        if self.os_mode == 'None':
            X = df.iloc[:, :-1].values.astype(np.float32)
            y = df.iloc[:, -1].values.astype(np.float32) # LightGBM prefers a 1D array for y

        if self.os_mode == 'top':
        # Identify top % performers based on target column (last column)
            y = df.iloc[:, -1]
            top = 90
            threshold = np.percentile(y, top)  # top % cutoff
            df_top = df[y >= threshold]

            # Oversample top performers
            df_augmented = pd.concat([df, pd.concat([df_top]*self.rep_factor, ignore_index=True)], ignore_index=True)
            print(f"Original dataset size: {len(df)}")
            print(f"Augmented dataset size: {len(df_augmented)} (with top {100-top}% duplicated {self.rep_factor}x)")

            # All columns except the last one are features
            X = df_augmented.iloc[:, :-1].values.astype(np.float32)
            # The very last column is the target
            y = df_augmented.iloc[:, -1].values.astype(np.float32) 

        elif self.os_mode == 'hybrid':
            y = df.iloc[:, -1]
            top = 90
            mid = 75
            low = 50
            threshold_upper = np.percentile(y, top)  # top % cutoff
            threshold_mid = np.percentile(y, mid)
            threshold_bot = np.percentile(y, low)
            df_top = df[y >= threshold_upper]
            df_mid = df[(y >= threshold_mid) & (y < threshold_upper)]
            df_bot = df[(y >= threshold_bot) & (y < threshold_mid)]

            # Oversample top performers
            df_augmented_top = pd.concat([df, pd.concat([df_top]*self.rep_factor, ignore_index=True)], ignore_index=True)
            df_augmented_mid = pd.concat([df_augmented_top, pd.concat([df_mid]*(int(self.rep_factor/3)), ignore_index=True)], ignore_index=True)
            df_augmented_hybrid = pd.concat([df_augmented_mid, pd.concat([df_bot]*(int(self.rep_factor/5)), ignore_index=True)], ignore_index=True)
            print(f"Original dataset size: {len(df)}")
            print(f"Augmented Hybrid dataset size: {len(df_augmented_hybrid)} \n(with top {100-top}% duplicated {self.rep_factor}x, \nmid {mid}% to {top}% duplicated {int(self.rep_factor/3)} and \nbot {low}% to mid {mid}% duplicated {int(self.rep_factor/5)})")

            # All columns except the last one are features
            X = df_augmented_hybrid.iloc[:, :-1].values.astype(np.float32)
            # The very last column is the target
            y = df_augmented_hybrid.iloc[:, -1].values.astype(np.float32) # LightGBM prefers a 1D array for y
        
        return X, y

    def train(self, data_path):
        """
        Executes the full training pipeline for the LGBM regressor.
        Returns a dictionary of performance metrics on the test set.
        """
        # Load data
        X, y = self._load_and_prepare_data(data_path)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Choose the weight type
        if self.weight == 'instance':
            weight = y_train
        elif self.weight == 'frequency':
            y_train_series = pd.Series(y_train.flatten())
            value_counts = y_train_series.value_counts()
            weights_map = 1 / value_counts
            weights = y_train_series.map(weights_map)
            weight = weights * len(y_train) / weights.sum()
        elif self.weight == 'linear':
            ymin, ymax = y_train.min(), y_train.max()
            alpha = 4
            weight = 1 + alpha * (y_train - ymin) / (ymax - ymin)
            weight = weight * len(y_train) / weight.sum()
        elif self.weight == 'exponential':
            beta = 2.0
            weight = np.exp(beta * y_train / y_train.max())
            weight = weight * len(y_train) / weight.sum()
        elif self.weight == 'quantile':
            q_bins = pd.qcut(y_train, q=10, duplicates='drop')
            value_counts = q_bins.value_counts()
            weights_map = 1 / value_counts
            weight = pd.Series(q_bins).map(weights_map).to_numpy()
            weight = weight * len(y_train) / weight.sum()
        elif self.weight == 'polynomial':
            gamma = 3
            weight = (y_train / y_train.max()) ** gamma
            weight = weight * len(y_train) / weight.sum()
        else:
            weight = None

        # Scale Features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Initialize Model
        self.model = lgb.LGBMRegressor(
            objective=self.objective, # MSE loss is base
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Train with early stopping
        print(f"Starting LGBM training for up to {self.n_estimators} estimators...")
        self.model.fit(
            X_train_scaled, y_train,
            sample_weight=weight,
            eval_set=[(X_test_scaled, y_test)],
            eval_metric='mae',
            callbacks=[lgb.early_stopping(self.stopping_rounds, verbose=False)]
        )
        
        print("Training complete.")
        return self.evaluate(X_test_scaled, y_test)

    def evaluate(self, X_test_scaled, y_test):
        """Calculates and returns performance metrics on the scaled test set."""
        y_preds = self.model.predict(X_test_scaled)
        
        mae = mean_absolute_error(y_test, y_preds)
        r2 = r2_score(y_test, y_preds)
        
        metrics = {'MAE': mae, 'R2_Score': r2}
        
        print("\n--- LGBM Regressor Evaluation on Test Set ---")
        for name, value in metrics.items():
            print(f"{name:<10}: {value:.4f}")
        print("-------------------------------------------\n")
        
        return metrics

    def predict(self, X_new):
        """Makes predictions on new, un-scaled data."""
        if not all([self.model, self.scaler]):
            raise RuntimeError("Model is not trained. Please call train() first.")
        
        if X_new.ndim == 1:
            X_new = X_new.reshape(1, -1)
            
        X_scaled = self.scaler.transform(X_new)
        predictions = self.model.predict(X_scaled)
        return predictions
        
    def save_model(self, folder_path, file_name_prefix):
        """Saves the trained model and the scaler."""
        if not all([self.model, self.scaler]):
            raise RuntimeError("Model not trained yet. Cannot save.")
        
        os.makedirs(folder_path, exist_ok=True)
        
        # Use joblib to save the model and scaler
        joblib.dump(self.model, os.path.join(folder_path, f"{file_name_prefix}_model.joblib"))
        joblib.dump(self.scaler, os.path.join(folder_path, f"{file_name_prefix}_scaler.joblib"))

        print(f"LGBM model and scaler saved in folder: {folder_path}")

    @classmethod
    def load_model(cls, folder_path, file_name_prefix):
        """Loads a trained model and its scaler."""
        # Create a dummy instance to load into
        lgbm_regressor = cls()
        
        model_path = os.path.join(folder_path, f"{file_name_prefix}_model.joblib")
        scaler_path = os.path.join(folder_path, f"{file_name_prefix}_scaler.joblib")
        
        lgbm_regressor.model = joblib.load(model_path)
        lgbm_regressor.scaler = joblib.load(scaler_path)
        
        print(f"LGBM model loaded successfully from folder: {folder_path}")
        return lgbm_regressor
    
class SKNN_Regressor:
    """
    A class to encapsulate a Scikit-learn Polynomial Linear Regression model.
    It uses a pipeline to chain scaling, polynomial feature creation,
    and linear regression.
    """
    def __init__(self,
                 degree=2,
                 test_size=0.2,
                 random_state=42,
                 os_mode='None',
                 rep_factor=0):
        
        # Hyperparameter for Grid Search
        self.degree = degree
        
        # Other training parameters
        self.test_size = test_size
        self.random_state = random_state
        self.os_mode = os_mode
        self.rep_factor = rep_factor

        # The entire pipeline (scaler, poly features, model) will be stored here
        self.model = None
        
        print(f"SKNN_Regressor initialized with polynomial degree={self.degree}.")

    def _load_and_prepare_data(self, data_path):
        """
        Loads data from a single CSV file with a header, assuming the
        last column is the target.
        """
        df = pd.read_csv(data_path)

        # Oversampling
        if self.os_mode == 'None':
            X = df.iloc[:, :-1].values.astype(np.float32)
            y = df.iloc[:, -1].values.astype(np.float32) # LightGBM prefers a 1D array for y

        if self.os_mode == 'top':
        # Identify top % performers based on target column (last column)
            y = df.iloc[:, -1]
            top = 90
            threshold = np.percentile(y, top)  # top % cutoff
            df_top = df[y >= threshold]

            # Oversample top performers
            df_augmented = pd.concat([df, pd.concat([df_top]*self.rep_factor, ignore_index=True)], ignore_index=True)
            print(f"Original dataset size: {len(df)}")
            print(f"Augmented dataset size: {len(df_augmented)} (with top {100-top}% duplicated {self.rep_factor}x)")

            # All columns except the last one are features
            X = df_augmented.iloc[:, :-1].values.astype(np.float32)
            # The very last column is the target
            y = df_augmented.iloc[:, -1].values.astype(np.float32) 

        elif self.os_mode == 'hybrid':
            y = df.iloc[:, -1]
            top = 90
            mid = 75
            low = 50
            threshold_upper = np.percentile(y, top)  # top % cutoff
            threshold_mid = np.percentile(y, mid)
            threshold_bot = np.percentile(y, low)
            df_top = df[y >= threshold_upper]
            df_mid = df[(y >= threshold_mid) & (y < threshold_upper)]
            df_bot = df[(y >= threshold_bot) & (y < threshold_mid)]

            # Oversample top performers
            df_augmented_top = pd.concat([df, pd.concat([df_top]*self.rep_factor, ignore_index=True)], ignore_index=True)
            df_augmented_mid = pd.concat([df_augmented_top, pd.concat([df_mid]*(int(self.rep_factor/3)), ignore_index=True)], ignore_index=True)
            df_augmented_hybrid = pd.concat([df_augmented_mid, pd.concat([df_bot]*(int(self.rep_factor/5)), ignore_index=True)], ignore_index=True)
            print(f"Original dataset size: {len(df)}")
            print(f"Augmented Hybrid dataset size: {len(df_augmented_hybrid)} \n(with top {100-top}% duplicated {self.rep_factor}x, \nmid {mid}% to {top}% duplicated {int(self.rep_factor/3)}x and \nbot {low}% to mid {mid}% duplicated {int(self.rep_factor/5)}x)")

            # All columns except the last one are features
            X = df_augmented_hybrid.iloc[:, :-1].values.astype(np.float32)
            # The very last column is the target
            y = df_augmented_hybrid.iloc[:, -1].values.astype(np.float32) # LightGBM prefers a 1D array for y
        
        return X, y

    def train(self, data_path):
        """
        Executes the full training pipeline for the SKNN regressor.
        Returns a dictionary of performance metrics on the test set.
        """
        # Load data
        X, y = self._load_and_prepare_data(data_path)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Initialize the Scikit-learn pipeline
        # This one object handles all steps automatically.
        self.model = make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=self.degree),
            LinearRegression()
        )
        
        # Train the entire pipeline
        print(f"Starting SKNN training for degree={self.degree} polynomial...")
        self.model.fit(X_train, y_train)
        
        print("Training complete.")
        return self.evaluate(X_test, y_test)

    def evaluate(self, X_test, y_test):
        """Calculates and returns performance metrics on the test set."""
        # The pipeline handles the scaling and transformations automatically
        y_preds = self.model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_preds)
        r2 = r2_score(y_test, y_preds)
        
        metrics = {'MAE': mae, 'R2_Score': r2}
        
        print("\n--- SKNN Regressor Evaluation on Test Set ---")
        for name, value in metrics.items():
            print(f"{name:<10}: {value:.4f}")
        print("-------------------------------------------\n")
        
        return metrics

    def predict(self, X_new):
        """Makes predictions on new, un-transformed data."""
        if self.model is None:
            raise RuntimeError("Model is not trained. Please call train() first.")
        
        return self.model.predict(X_new)
        
    def save_model(self, folder_path, file_name_prefix):
        """Saves the entire trained pipeline."""
        if self.model is None:
            raise RuntimeError("Model not trained yet. Cannot save.")
        
        os.makedirs(folder_path, exist_ok=True)
        
        # Save the entire pipeline object
        model_path = os.path.join(folder_path, f"{file_name_prefix}_pipeline.joblib")
        joblib.dump(self.model, model_path)

        print(f"SKNN pipeline saved to: {model_path}")

    @classmethod
    def load_model(cls, folder_path, file_name_prefix):
        """Loads a trained pipeline."""
        model_path = os.path.join(folder_path, f"{file_name_prefix}_pipeline.joblib")
        
        loaded_model = joblib.load(model_path)
        
        # Create a class instance and assign the loaded model to it
        # We can find the degree from the loaded pipeline's parameters
        degree = loaded_model.get_params()['polynomialfeatures__degree']
        sknn_regressor = cls(degree=degree)
        sknn_regressor.model = loaded_model
        
        print(f"SKNN pipeline loaded successfully from: {model_path}")
        return sknn_regressor
    
class Quantile_LGBM_Regressor:
    """
    A class to encapsulate a LightGBM Regressor for quantile regression.
    This class trains a separate model for each specified quantile and handles
    data loading and oversampling.
    """
    def __init__(self,
                 quantiles=[0.1, 0.5, 0.9],
                 test_size=0.2,
                 random_state=42,
                 n_estimators=2000,
                 learning_rate=0.1,
                 num_leaves=30,
                 max_depth=-1,
                 **kwargs):
        """
        Initializes the Quantile_LGBM_Regressor.

        Args:
            quantiles (list): A list of floats representing the quantiles to predict.
            test_size (float): The proportion of the dataset to include in the test split for evaluation.
            random_state (int): Controls the randomness of the dataset splitting.
            n_estimators (int): Number of boosting rounds.
            learning_rate (float): Boosting learning rate.
            num_leaves (int): Maximum tree leaves for base learners.
            max_depth (int): Maximum tree depth for base learners.
            os_mode (str): Oversampling mode ('None', 'top', 'hybrid').
            rep_factor (int): Repetition factor for oversampling.
            **kwargs: Catches any other standard LGBMRegressor hyperparameters
                      (e.g., reg_alpha=0.1, colsample_bytree=0.8) and passes
                      them to the model.
        """
        self.quantiles = quantiles
        self.test_size = test_size
        self.random_state = random_state

        # Store all LGBM-specific hyperparameters in a dictionary for easy passing
        self.lgbm_kwargs = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'num_leaves': num_leaves,
            'max_depth': max_depth,
            **kwargs
        }
        
        self.models = {}  # A dictionary to hold a model for each quantile
        self.X_test = None
        self.y_test = None
        print("Quantile_LGBM_Regressor initialized.")

    def _load_and_prepare_data(self, data_path):
        """
        Loads and prepares data from a CSV file, with optional oversampling.
        Assumes the last column is the target variable.
        """
        df = pd.read_csv(data_path)

        X = df.iloc[:, :-1].values.astype(np.float32)
        y = df.iloc[:, -1].values.astype(np.float32)
        
        return X, y

    def train(self, data_path):
        """
        Executes the full training pipeline using data from a file path.

        Args:
            data_path (str): The path to the training CSV file.
        """
        X, y = self._load_and_prepare_data(data_path)
        
        # Split data for evaluation if test_size is greater than 0
        if self.test_size > 0:
            X_train, self.X_test, y_train, self.y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
        else:
            X_train, y_train = X, y

        for q in self.quantiles:
            print(f"Training model for quantile: {q:.2f}")
            # The core of quantile regression is setting the objective and alpha
            model = lgb.LGBMRegressor(objective='quantile', alpha=q, **self.lgbm_kwargs)
            model.fit(X_train, y_train)
            self.models[q] = model
        
        print("All quantile models trained successfully.")

    def evaluate(self):
        """
        Evaluates the performance of each trained model on the hold-out test set.
        
        Returns:
            dict: A dictionary of evaluation metrics for each quantile.
        """
        if not self.models or self.X_test is None or self.y_test is None:
            raise RuntimeError("Models are not trained or no test set is available for evaluation. "
                               "Please call train() with a test_size > 0 first.")
        
        all_metrics = {}
        print("\n--- Quantile Regressor Evaluation on Test Set ---")
        for q, model in self.models.items():
            y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            # Mean Absolute Percentage Error (MAPE)
            mape = np.mean(np.abs((self.y_test - y_pred) / np.where(self.y_test == 0, 1e-8, self.y_test)))*100
            
            # Mean Bias Error (MBE) - to check for systematic bias
            mbe = np.mean(y_pred - self.y_test)

            quantile_metrics = {
                'MAE': mae,
                'R2_Score': r2,
                'MAPE (%)': mape,
                'Systematic_Bias (MBE)': mbe
            }
            all_metrics[f'quantile_{q:.2f}'] = quantile_metrics
            
            print(f"\nMetrics for Quantile = {q:.2f}:")
            for name, value in quantile_metrics.items():
                print(f"  {name:<24}: {value:.4f}")
        print("\n-------------------------------------------------\n")
        
        return all_metrics

    def predict(self, X_new):
        """
        Makes predictions on new data using each trained quantile model.
        
        Args:
            X_new (np.array or pd.DataFrame): The new data to make predictions on.
        
        Returns:
            pd.DataFrame: A DataFrame with predictions for each quantile.
        """
        if not self.models:
            raise RuntimeError("Models are not trained. Please call train() first.")

        predictions_df = pd.DataFrame()
        for q, model in self.models.items():
            col_name = f'quantile_{q:.2f}_pred'
            predictions_df[col_name] = model.predict(X_new)
        
        return predictions_df

    def save_model(self, folder_path, file_name_prefix):
        """
        Saves all trained quantile models to the specified folder.
        
        Args:
            folder_path (str): The path to the directory where models should be saved.
            file_name_prefix (str): A prefix for the saved file names.
        """
        if not self.models:
            raise RuntimeError("Models not trained yet. Cannot save.")
        
        os.makedirs(folder_path, exist_ok=True)
        
        for q, model in self.models.items():
            # Sanitize quantile for filename (e.g., 0.5 -> 0_5)
            q_str = str(q).replace('.', '_')
            model_path = os.path.join(folder_path, f"{file_name_prefix}_quantile_{q_str}_model.joblib")
            joblib.dump(model, model_path)
        
        print(f"All quantile LGBM models saved in folder: {folder_path}")

    @classmethod
    def load_model(cls, folder_path, file_name_prefix, quantiles):
        """
        Loads a set of trained quantile models from a directory.
        
        Args:
            folder_path (str): The path to the directory where the models are saved.
            file_name_prefix (str): The prefix of the saved file names.
            quantiles (list): The list of quantiles corresponding to the models to be loaded.
        
        Returns:
            An instance of Quantile_LGBM_Regressor with the loaded models.
        """
        quantile_regressor = cls(quantiles=quantiles)
        
        for q in quantiles:
            q_str = str(q).replace('.', '_')
            model_path = os.path.join(folder_path, f"{file_name_prefix}_quantile_{q_str}_model.joblib")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at: {model_path}")
                
            loaded_model = joblib.load(model_path)
            quantile_regressor.models[q] = loaded_model
            
        print(f"Quantile models for {quantiles} loaded successfully from folder: {folder_path}")

        return quantile_regressor