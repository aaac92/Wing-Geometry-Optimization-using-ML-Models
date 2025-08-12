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
        pos_weight = torch.tensor([neg_count / pos_count], device=self.device) if pos_count > 0 else torch.tensor([1.0])
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
    def load_model(cls, folder_path, file_name_prefix):
        """Loads a model, scaler, and metadata from files."""
        model_path = os.path.join(folder_path, f"{file_name_prefix}_model.pth")
        scaler_path = os.path.join(folder_path, f"{file_name_prefix}_scaler.joblib")
        metadata_path = os.path.join(folder_path, f"{file_name_prefix}_meta.json")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        loaded_classifier = cls()
        loaded_classifier.input_dim = metadata['input_dim']
        
        loaded_classifier.model = _PyTorchMLPClassifier(input_dim=loaded_classifier.input_dim)
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
                 random_state=42):
        
        # Hyperparameters
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.noise_level = noise_level
        
        # Other training parameters
        self.epochs = epochs
        self.patience = patience
        self.poly_degree = poly_degree
        self.test_size = test_size
        self.random_state = random_state

        # Class components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.poly_transformer = None
        self.scaler = None
        self.input_dim = None # Will be determined after polynomial transformation
        
        print(f"MLP Regressor initialized. Training will use device: {self.device}")

    def _load_and_prepare_data(self, data_path):
        """
        Loads data from a single CSV file with a header, assuming the
        last column is the target.
        """
        df = pd.read_csv(data_path)

        # All columns except the last one are features
        X = df.iloc[:, :-1].values.astype(np.float32)
        # The very last column is the target
        y = df.iloc[:, -1:].values.astype(np.float32)
    
        return X, y

    def train(self, data_path):
        """
        Executes the full training pipeline for the regressor.
        Returns a dictionary of performance metrics on the test set.
        """
        # Load data
        X, y = self._load_and_prepare_data(data_path)
        
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
        y_train_t = torch.from_numpy(y_train).to(self.device)
        X_test_t = torch.from_numpy(X_test_scaled).to(self.device)
        y_test_t = torch.from_numpy(y_test).to(self.device)

        # Initialize Model, Loss, Optimizer, and Scheduler
        self.model = _PyTorchMLPRegressor(
            input_dim=self.input_dim,
            hidden_size=self.hidden_size,
            dropout_rate=self.dropout_rate
        ).to(self.device)
        
        loss_fn = nn.SmoothL1Loss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=self.patience // 2, factor=0.5)

        # Training Loop with Early Stopping
        best_val_loss = float('inf')
        early_stop_counter = 0
        best_model_state = None

        print(f"Starting regressor training for up to {self.epochs} epochs...")
        for epoch in range(self.epochs):
            self.model.train()
            
            # Data Augmentation: Add noise
            noise = torch.randn_like(X_train_t) * self.noise_level
            X_train_augmented = X_train_t + noise

            # Forward pass
            pred = self.model(X_train_augmented)
            loss = loss_fn(pred, y_train_t)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_test_t)
                val_loss = loss_fn(val_pred, y_test_t)
            
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
            preds_tensor = self.model(X_test_t)
        
        # Move tensors to CPU and convert to NumPy for scikit-learn metrics
        y_true_np = y_test_t.cpu().numpy()
        y_preds_np = preds_tensor.cpu().numpy()

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
            predictions = self.model(X_t)
            
        return predictions.cpu().numpy()
        
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
    
class LGBMRegressor:
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
                 random_state=42):
        
        # Hyperparameters for Grid Search
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        
        # Other training parameters
        self.stopping_rounds = stopping_rounds
        self.test_size = test_size
        self.random_state = random_state

        # Class components
        self.model = None
        self.scaler = None
        
        print("LGBMRegressor initialized.")

    def _load_and_prepare_data(self, data_path):
        """
        Loads data from a single CSV file with a header, assuming the
        last column is the target.
        """
        df = pd.read_csv(data_path)
        # All columns except the last one are features
        X = df.iloc[:, :-1].values.astype(np.float32)
        # The very last column is the target
        y = df.iloc[:, -1].values.astype(np.float32) # LightGBM prefers a 1D array for y
        
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
        
        # Scale Features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Initialize Model
        self.model = lgb.LGBMRegressor(
            objective='regression_l1', # MAE loss
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