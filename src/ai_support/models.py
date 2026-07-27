import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from torch.utils.data import DataLoader, Subset, TensorDataset
import joblib
import json

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
                 epochs=500,
                 patience=50,
                 k=5,
                 test_split=0.2,
                 random_state=92,
                 weight=None,
                 log_trans=False):
        
        # Hyperparameters
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
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

        features = df.iloc[:, :-1].values.astype(np.float32)
        output = df.iloc[:, -1].values.astype(np.float32)

        # Optional logarithmic output transformation and data split for testing
        if self.log_transform == True:
            # Split log data
            output = np.log1p(output)
            train_data, test_data, train_targets, test_targets = train_test_split(features, output, test_size=self.test_split, random_state=self.random_state)
        else:
            # Split data
            train_data, test_data, train_targets, test_targets = train_test_split(features, output, test_size=self.test_split, random_state=self.random_state)

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

    def train(self, data_path):
        """
        Executes the full training pipeline for the regressor.
        Returns a dictionary of performance metrics on the test set.

        PARAMETERS:
        - data_path: Same data path used for the CSV in _load_and_prepare_data()
        """

        # Load data
        X, y, X_test, y_test = self._load_and_prepare_data(data_path) 

        # Obtain weights for some features
        if self.weight == 'instance':
            # This uses the target value as the weight
            weights = y
            weights_train = weights*(len(y)/weights.sum())
        if self.weight == 'frequency':
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

        # --- DYNAMIC BATCH SIZE INTEGRATION ---
        calculated_batch_size = self._find_max_batch_size(X, y, weights_train)

        best_model_state = None

        # Create dataset from tensors
        train_dataset = TensorDataset(X, y, weights_train)
        kf = KFold(n_splits=self.k, shuffle=True, random_state=self.random_state) 
        all_scores_mae = []
        all_scores_r2 = []

        print(f"Starting regressor training using {self.k} folds...")
        for fold, (train_i, val_i) in enumerate(kf.split(train_dataset)):
            print(f"Processing fold: {fold + 1}/{self.k}")

            # Creating subsets
            train_subset = Subset(train_dataset, train_i)
            val_subset = Subset(train_dataset, val_i)

            # Create dataloaders using the dynamically calculated batch size
            train_loader = DataLoader(train_subset, batch_size=calculated_batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=calculated_batch_size, shuffle=True)

            # Initialize model
            self.model = _PyTorchMLPRegressor(
                input_dim=self.input_dim,
                hidden_layer_sizes=self.hidden_layer_sizes,
                dropout_rate=self.dropout_rate
            ).to(self.device)

            # Fit model for this fold
            self._fit(train_loader, val_loader)

            self.model.eval()
            pred, target = [], []
            with torch.no_grad():
                for features, targets, _ in val_loader:
                    features, targets = features.to(self.device), targets.to(self.device)

                    preds = self.model(features)

                    preds = preds.cpu().numpy()
                    targets = targets.cpu().numpy()

                    if self.log_transform == True:
                        targets = np.expm1(targets).astype(np.float32)
                        preds = np.expm1(preds).astype(np.float32)

                    pred.extend(preds)
                    target.extend(targets)

            r2 = r2_score(target, pred)
            mae = mean_absolute_error(target, pred)
            print(f"Fold {fold + 1} MAE: {mae:.4f}, R^2: {r2:.4f}")
            all_scores_mae.append(mae)
            all_scores_r2.append(r2)

        print("\n==== Cross-validation results ====")
        print("MAE per fold:", all_scores_mae)
        print("R^2 per fold:", all_scores_r2)
        print("Average MAE:", np.mean(all_scores_mae))
        print("Average R^2:", np.mean(all_scores_r2))
        
        print("\nTraining complete.")
        # Load the best model state found during training
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        return self._evaluate(X_test, y_test)
    
    def _fit(self, train_loader, val_loader):
        '''
        Trains model on a given dataset, this method is for internal use.
        This method also contains and chooses custom or standard loss functions and handles early stopping.

        PARAMETERS:
        - model: model object initialized each fold
        - train_loader: DataLoader with the shuffled folds
        - loss_fn: Loss function to be used in this training cycle
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

        for epoch in range(self.epochs):

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
                best_model_state = self.model.state_dict() # Save the best model
            else:
                early_stop_counter += 1
                if early_stop_counter >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}. Best validation loss: {best_val_loss:.4f}")
                    break

            if epoch % 100 == 0:
                print(f"Loss at epoch {epoch}: {val_loss.item():.4f}")

    def _evaluate(self, X_test, y_test):
        '''
        Evaluates the model on the test dataset, returns metrics.

        PARAMETERS:
        - X_test: Test features tensor
        - y_test: Test targets tensor
        '''
        if not all([self.model, self.scaler]):
            raise RuntimeError("Model is not trained. Please call train() first.")

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
                if self.log_transform == True:
                    # Inverse transformation
                    targets = np.expm1(targets).astype(np.float32)
                    test_preds = np.expm1(test_preds).astype(np.float32)

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

        # Apply same transformations as in training
        X = self.scaler.transform(X)
        X = torch.tensor(X, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            preds = self.model(X)
            preds = preds.cpu().numpy()
            if self.log_transform == True:
                preds = np.expm1(preds)
        
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

        metadata = {'input_dim': self.input_dim}
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

        # Load the scaler
        regressor.scaler = joblib.load(os.path.join(folder_path, f"{file_name_prefix}_scaler.joblib"))

        # Instantiate model with correct dimensions and load weights
        regressor.model = _PyTorchMLPRegressor(
            input_dim=regressor.input_dim,
            hidden_layer_sizes=regressor.hidden_layer_sizes,
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
                 lr_decay=0.99,
                 num_leaves=20,
                 max_depth=10,
                 stopping_rounds=100,
                 k=5,
                 test_size=0.2,
                 random_state=42,
                 objective='regression_l2',
                 weight='None'):
        
        # Hyperparameters for Grid Search
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
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

        # Class components
        self.model = None
        self.scaler = None
        self.objective = objective
        
        print("LGBMRegressor initialized.")

    def _load_and_prepare_data(self, data_path):
        """
        Loads data from a single CSV file with a header, assuming the
        last column is the target.

        PARAMETERS:
        - data_path: Full path to csv file that contains training data
        """
        df = pd.read_csv(data_path)

        features = df.iloc[:, :-1].values.astype(np.float32)
        output = df.iloc[:, -1].values.astype(np.float32) # LightGBM prefers a 1D array for y

        # Data split for testing
        train_data, test_data, train_targets, test_targets = train_test_split(features, output, test_size=self.test_split, random_state=self.random_state)

        return train_data, train_targets, test_data, test_targets

    def train(self, data_path):
        """
        Executes the full training pipeline for the LGBM regressor.
        Returns a dictionary of performance metrics on the test set.

        PARAMETERS:
        - data_path: Same path to csv file as _load_and_prepare_data()
        """
        
        def custom_combo_objective(y_true, y_pred):
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

        def custom_combo_eval(y_true, y_pred):
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

        # Scale Features
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        X_test = self.scaler.transform(X_test)

        # Objective function mapping
        if self.objective == 'combo':
            lgbm_objective = custom_combo_objective
            eval_metric = custom_combo_eval
        else:
            lgbm_objective = self.objective
            if self.objective in ['regression_l2', 'regression_l1']:
                eval_metric = 'mae'
            elif self.objective in ['huber', 'quantile', 'logcosh']:
                eval_metric = self.objective
            else:
                eval_metric = 'mae'

        kf = KFold(n_splits=self.k, shuffle=True, random_state=self.random_state)
        all_scores_mae = []
        all_scores_r2 = []

        print(f"Starting LGBM training using {self.k} folds for up to {self.n_estimators} estimators...")
        
        for fold, (train_i, val_i) in enumerate(kf.split(X)):
            print(f"Processing fold: {fold + 1}/{self.k}")
            X_train_fold, X_val_fold = X[train_i], X[val_i]
            y_train_fold, y_val_fold = y[train_i], y[val_i]
            
            weights_train_fold = weights_train[train_i] if weights_train is not None else None

            # Initialize Model
            self.model = lgb.LGBMRegressor(
                objective=lgbm_objective,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1
            )
            
            self.model.fit(
                X_train_fold, y_train_fold,
                sample_weight=weights_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                eval_metric=eval_metric,
                callbacks=[
                    lgb.early_stopping(self.stopping_rounds, verbose=False),
                    lgb.reset_parameter(learning_rate=lambda iter: self.learning_rate * (self.lr_decay ** iter))
                ]
            )

            y_preds_fold = self.model.predict(X_val_fold)
            mae = mean_absolute_error(y_val_fold, y_preds_fold)
            r2 = r2_score(y_val_fold, y_preds_fold)
            
            print(f"Fold {fold + 1} MAE: {mae:.4f}, R^2: {r2:.4f}")
            all_scores_mae.append(mae)
            all_scores_r2.append(r2)

        print("\n==== Cross-validation results ====")
        print(f"MAE per fold: {all_scores_mae}")
        print(f"R^2 per fold: {all_scores_r2}")
        print(f"Average MAE: {np.mean(all_scores_mae):.4f}")
        print(f"Average R^2: {np.mean(all_scores_r2):.4f}")
        print("Training complete.")

        return self._evaluate(X_test, y_test)

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
        
        metrics = {'MAE': mae, 'R2_Score': r2}
        
        print("\n--- LGBM Regressor Evaluation on Test Set ---")
        for name, value in metrics.items():
            print(f"{name:<10}: {value:.4f}")
        print("-------------------------------------------\n")
        
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
        
        lgbm_regressor.model = joblib.load(model_path)
        lgbm_regressor.scaler = joblib.load(scaler_path)
        
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

        # Data split for testing
        train_data, test_data, train_targets, test_targets = train_test_split(features, output, test_size=self.test_split, random_state=self.random_state)

        return train_data, train_targets, test_data, test_targets
    
    def train(self, data_path):
        """
        Executes the full training pipeline for the SKNN regressor.
        Returns a dictionary of performance metrics on the test set.

        PARAMETERS:
        - data_path: Same path to csv file as _load_and_prepare_data()
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

        kf = KFold(n_splits=self.k, shuffle=True, random_state=self.random_state)
        all_scores_mae = []
        all_scores_r2 = []

        print(f"Starting SKNN training using {self.k} folds for degree={self.degree} polynomial...")
        
        for fold, (train_i, val_i) in enumerate(kf.split(X)):
            print(f"Processing fold: {fold + 1}/{self.k}")
            X_train_fold, X_val_fold = X[train_i], X[val_i]
            y_train_fold, y_val_fold = y[train_i], y[val_i]
            
            weights_train_fold = weights_train[train_i] if weights_train is not None else None

            # Initialize the Scikit-learn pipeline
            self.model = make_pipeline(
                StandardScaler(),
                PolynomialFeatures(degree=self.degree),
                LinearRegression()
            )
            
            if weights_train_fold is not None:
                self.model.fit(X_train_fold, y_train_fold, linearregression__sample_weight=weights_train_fold)
            else:
                self.model.fit(X_train_fold, y_train_fold)

            y_preds_fold = self.model.predict(X_val_fold)
            mae = mean_absolute_error(y_val_fold, y_preds_fold)
            r2 = r2_score(y_val_fold, y_preds_fold)
            
            print(f"Fold {fold + 1} MAE: {mae:.4f}, R^2: {r2:.4f}")
            all_scores_mae.append(mae)
            all_scores_r2.append(r2)
        
        print("\n==== Cross-validation results ====")
        print(f"MAE per fold: {all_scores_mae}")
        print(f"R^2 per fold: {all_scores_r2}")
        print(f"Average MAE: {np.mean(all_scores_mae):.4f}")
        print(f"Average R^2: {np.mean(all_scores_r2):.4f}")
        print("Training complete.")

        return self._evaluate(X_test, y_test)
    
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
        
        metrics = {'MAE': mae, 'R2_Score': r2}
        
        print("\n--- SKNN Regressor Evaluation on Test Set ---")
        for name, value in metrics.items():
            print(f"{name:<10}: {value:.4f}")
        print("-------------------------------------------\n")
        
        return metrics
    
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
                 k=5,
                 test_size=0.2, 
                 random_state=42):
        
        # Hyperparameters
        self.hidden_layer_sizes = hidden_layer_sizes
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self.factor = factor

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

        # Identify target and features
        y = df['valid'].values.astype(np.float32)
        X = df.drop(columns=['valid']).values.astype(np.float32)

        self.input_dim = X.shape[1]
        self.feature_columns = df.drop(columns=['valid']).columns.tolist()

        # Split data
        X, X_test, y, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state, stratify=y)

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

    def _find_max_batch_size(self, X_t, y_t):
        """
        Dynamically finds the maximum batch size that fits in GPU VRAM by
        catching OutOfMemory exceptions and halving the batch size.
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
                y_dummy = y_t[:batch_size]
                
                # Initialize dummy model and optimizer
                dummy_model = _PyTorchMLPClassifier(
                    input_dim=self.input_dim,
                    hidden_layer_sizes=self.hidden_layer_sizes
                ).to(self.device)
                
                dummy_optimizer = optim.Adam(dummy_model.parameters(), lr=self.lr)
                
                # Dummy forward and backward pass
                dummy_model.train()
                dummy_optimizer.zero_grad()
                
                preds = dummy_model(X_dummy)
                loss = nn.BCEWithLogitsLoss()(preds, y_dummy) 
                loss.backward()
                dummy_optimizer.step()
                
                del dummy_model, dummy_optimizer, X_dummy, y_dummy, preds, loss
                torch.cuda.empty_cache()
                
                optimal_batch = int(batch_size * 0.9) 
                print(f"Optimal batch size found: {optimal_batch}")
                
                return optimal_batch
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    batch_size //= 2
                else:
                    raise e
                    
        raise RuntimeError("Could not find a batch size that fits in GPU VRAM.")

    def train(self, data_path):
        """
        Executes the full training pipeline using a single data file.

        PARAMETERS:
        - data_path: Same data path used for the CSV in _load_and_prepare_data()
        """
        # Load data
        X, y, X_test, y_test = self._load_and_prepare_data(data_path)
        print(f"Data loaded successfully from {data_path}. Input features: {self.input_dim}")

        # --- DYNAMIC BATCH SIZE INTEGRATION ---
        calculated_batch_size = self._find_max_batch_size(X, y)

        skf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=self.random_state)
        
        # Detach tensors for Stratified K-Fold logic
        X_np, y_np = X.cpu().numpy(), y.cpu().numpy().flatten()
        
        all_acc, all_prec, all_rec, all_f1 = [], [], [], []

        print(f"Starting classifier training using {self.k} folds...")
        for fold, (train_i, val_i) in enumerate(skf.split(X_np, y_np)):
            print(f"Processing fold: {fold + 1}/{self.k}")
            
            X_train_fold, X_val_fold = X[train_i], X[val_i]
            y_train_fold, y_val_fold = y[train_i], y[val_i]

            # Create dataloaders using the dynamically calculated batch size
            train_dataset = TensorDataset(X_train_fold, y_train_fold)
            val_dataset = TensorDataset(X_val_fold, y_val_fold)
            
            train_loader = DataLoader(train_dataset, batch_size=calculated_batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=calculated_batch_size, shuffle=False)

            # Initialize model, optimizer, and scheduler
            self.model = _PyTorchMLPClassifier(input_dim=self.input_dim, hidden_layer_sizes=self.hidden_layer_sizes).to(self.device)
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=self.patience // 2, factor=self.factor)

            neg_count = (y_train_fold == 0).sum().item()
            pos_count = (y_train_fold == 1).sum().item()
            weight = torch.tensor([pos_count / neg_count], device=self.device) if pos_count > 0 else torch.tensor([1.0], device=self.device)
            
            # Loss for training (weighted) and validation (unweighted)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)
            loss_fn_eval = nn.BCEWithLogitsLoss()

            best_val_loss = float('inf')
            early_stop_counter = 0
            best_model_state = None

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
                    best_model_state = self.model.state_dict()
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= self.patience:
                        print(f"Early stopping triggered at epoch {epoch + 1}. Best validation loss: {best_val_loss:.4f}")
                        break

                if (epoch + 1) % 100 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    avg_train_loss = epoch_train_loss / len(train_loader)
                    print(f'Fold {fold + 1}, Epoch [{epoch+1}/{self.epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}')

            # Restore best model weights for this fold's evaluation
            if best_model_state:
                self.model.load_state_dict(best_model_state)

            # Fold validation metrics
            self.model.eval()
            val_pred_labels_list = []
            y_val_true_list = []
            
            with torch.no_grad():
                for X_val_batch, y_val_batch in val_loader:
                    val_pred_logits = self.model(X_val_batch)
                    val_pred_probs = torch.sigmoid(val_pred_logits)
                    val_pred_labels_list.extend((val_pred_probs > 0.5).cpu().numpy())
                    y_val_true_list.extend(y_val_batch.cpu().numpy())
            
            y_true = np.array(y_val_true_list)
            val_pred_labels = np.array(val_pred_labels_list)
            
            acc = accuracy_score(y_true, val_pred_labels)
            prec = precision_score(y_true, val_pred_labels, zero_division=0)
            rec = recall_score(y_true, val_pred_labels, zero_division=0)
            f1 = f1_score(y_true, val_pred_labels, zero_division=0)
            
            all_acc.append(acc)
            all_prec.append(prec)
            all_rec.append(rec)
            all_f1.append(f1)
        
        print("\n==== Cross-validation results ====")
        print(f"Average Accuracy: {np.mean(all_acc):.4f}")
        print(f"Average Precision: {np.mean(all_prec):.4f}")
        print(f"Average Recall: {np.mean(all_rec):.4f}")
        print(f"Average F1-Score: {np.mean(all_f1):.4f}")
        print("Classifier training complete.")

        return self._evaluate(X_test, y_test)

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
        loaded_classifier.model.load_state_dict(torch.load(model_path))
        loaded_classifier.model.to(loaded_classifier.device)
        loaded_classifier.model.eval()

        loaded_classifier.scaler = joblib.load(scaler_path)

        print(f"Classifier loaded successfully from folder: {folder_path}")
        return loaded_classifier