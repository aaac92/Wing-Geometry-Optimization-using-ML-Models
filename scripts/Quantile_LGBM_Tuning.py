import sys
import os
import re
import json
import random
import optuna
from pathlib import Path
from ai_support import Quantile_LGBM_Regressor

# === Setup and Configuration === #
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
# Add the project root to the list of places Python looks for modules.
sys.path.insert(0, str(project_root))
# Path to regressor dataset
Training_File = "Airfoils_for_Regressor_E_Dataset_16_autocorrected.csv"
DATA_PATH = project_root / Training_File
# Automatically determine dataset number and set up save folder
match = re.search(r"(?<=Dataset_)\d+", Training_File)
if match:
    dataset_num = int(match.group())
# Folder where the best trained models will be saved
MODEL_SAVE_FOLDER = project_root / "AI_Models" / f"Dataset_{dataset_num}" / "optimized_LGBM_Q_regressor"
os.makedirs(MODEL_SAVE_FOLDER, exist_ok=True)
if not DATA_PATH.exists():
    print(f"[ERROR] Dataset not found at '{DATA_PATH}'. Please check the file path.")
    sys.exit()

# Define the quantiles to be used for all training and optimization
QUANTILES = [0.25, 0.5, 0.75, 0.9, 0.95]

def objective(trial: optuna.trial.Trial) -> float:
    """
    Optuna objective function to find the best hyperparameters for the Quantile_LGBM_Regressor.
    """
    # Define the search space for the hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 2500, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
    }
    
    model = Quantile_LGBM_Regressor(
        quantiles=QUANTILES,
        random_state=42,
        **params
    )
    
    try:
        # Train the model (the class handles the train/test split internally)
        model.train(data_path=str(DATA_PATH))
        
        # Evaluate to get metrics
        metrics = model.evaluate()
        
        # Calculate the average MAE across all quantiles as the objective value
        average_mae = sum(m['MAE'] for m in metrics.values()) / len(metrics)
        
        return average_mae

    except Exception as e:
        print(f"Trial failed with error: {e}")
        # Return a large value to prune this trial
        return float('inf')

def run_optimization(n_trials: int = 100):
    """
    Sets up and runs the Optuna optimization study.
    """
    print(f"--- Starting Optuna optimization for {n_trials} trials ---")
    
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials)
    
    print("\n--- Optimization Finished ---")
    print(f"Best trial number: {study.best_trial.number}")
    print(f"Best average MAE: {study.best_value:.4f}")
    print("Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
        
    # Retrain the final best model with the optimal parameters
    print("\n--- Retraining best model ---")
    best_model = Quantile_LGBM_Regressor(quantiles=QUANTILES, random_state=42, **study.best_params)
    best_model.train(data_path=str(DATA_PATH))
    
    return best_model, study.best_params

def retrain_with_best_params(best_params: dict):
    """
    Retrains a model using a predefined set of optimal parameters.
    """
    print("--- Retraining model with predefined best parameters ---")
    model = Quantile_LGBM_Regressor(quantiles=QUANTILES, random_state=42, **best_params)
    model.train(data_path=str(DATA_PATH))
    print("\n--- Final Evaluation ---")
    model.evaluate()
    
    return model

if __name__ == "__main__":
    # Boolean selector to switch between modes
    RUN_OPTIMIZATION = True

    best_model = None
    best_params = {}

    if RUN_OPTIMIZATION:
        best_model, best_params = run_optimization(n_trials=200)
    else:
        # Manually define parameters
        predefined_params = {
            'n_estimators': 1800,
            'learning_rate': 0.025,
            'num_leaves': 75,
            'max_depth': 15,
            'reg_alpha': 0.1,
            'reg_lambda': 0.2
        }
        best_model = retrain_with_best_params(predefined_params)
        best_params = predefined_params

    # === Final Saving and Reporting === #
    # Create name lists for saving the model
    names = ["Murata", "Barbatos", "Beelzebul", "Buer", "Focalors", "Istaroth", "Phanes", "Asmoday", "Paimon", "Alice", "Ronnova"]
    best_model_name = random.choice(names)
    
    print("\n--- Saving Best Performing Model ---")
    if best_model:
        # Create a dictionary of parameters for saving to JSON
        best_result_to_save = {
            'quantiles': QUANTILES,
            'random_state': 42,
            **best_params
        }

        json_name = f"{best_model_name}_Q_regressor_model_parameters.json"
        param_file = MODEL_SAVE_FOLDER / json_name
        
        with open(param_file, 'w') as json_file:
            json.dump(best_result_to_save, json_file, indent=4)
        
        print(f"Best model parameters saved to: {param_file}")

        file_name_prefix = f"{best_model_name}_Q_regressor"
        best_model.save_model(str(MODEL_SAVE_FOLDER), file_name_prefix)
    else:
        print("No best model was captured to save.")