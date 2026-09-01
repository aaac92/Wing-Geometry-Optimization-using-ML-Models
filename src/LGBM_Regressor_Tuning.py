import sys
import os
import optuna
import json
import random
from pathlib import Path
from ai_support import LGBM_Regressor

# === Directory configuration === #
# Get the path to the src directory.
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
datasets_dir = project_root / "data" / "datasets"
# Path to dataset
dataset_num = 17
dataset_name = f"regressor_dataset_{dataset_num}"
dataset_dir = datasets_dir / f"dataset_{dataset_num}"
training_file_dir = dataset_dir / f"{dataset_name}.csv"
model_save_dir = project_root / "models" / f"dataset_{dataset_num}" / "lgbm_regressor"
os.makedirs(model_save_dir, exist_ok=True)
# Verify that dataset exists
if not os.path.exists(training_file_dir):
    print(f"[ERROR] Dataset not found at '{training_file_dir}'. Please check the file path.")
    sys.exit()

# === Optuna Objective Function === #
N_TRIALS = 60
def objective(trial):
    """
    Takes a trial object from Optuna, suggests hyperparameters for LightGBM, 
    trains the regressor, and evaluates the performance.
    """
    print("\n" + "="*50)
    print(f"Starting Trial {trial.number + 1}/{N_TRIALS}")

    # Dynamically build hyperparameter space
    n_estimators = trial.suggest_int('n_estimators', 1000, 4000, step=500)
    lr = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    lr_decay = trial.suggest_float('lr_decay', 0.90, 0.99)
    num_leaves = trial.suggest_int('num_leaves', 20, 100, step=20)
    max_depth = trial.suggest_int('max_depth', 5, 20, step=5)
    objective_type = trial.suggest_categorical('objective', ['regression_l2', 'combo'])
    weight = trial.suggest_categorical('weight', ['None', 'instance', 'frequency', 'linear', 'exponential', 'quantile', 'polynomial'])

    print(f"PARAMETERS:")
    print(f"Estimators = {n_estimators}, LR = {lr:.4f}, LR Decay = {lr_decay:.4f}")
    print(f"Leaves = {num_leaves}, Max Depth = {max_depth}")
    print(f"Objective = {objective_type}, Weight = {weight}")
    print("="*50)

    # Instantiate and train the regressor
    regressor = LGBM_Regressor(
        n_estimators=n_estimators,
        learning_rate=lr,
        lr_decay=lr_decay,
        num_leaves=num_leaves,
        max_depth=max_depth,
        objective=objective_type,
        weight=weight
    )

    final_metrics = regressor.train(data_path=training_file_dir)

    # Optimize for MAE
    mae = final_metrics.get('MAE', float('inf'))
    
    trial.set_user_attr('R2-Score', final_metrics.get('R2_Score', float('-inf')))

    return mae

# === Optuna Study Execution === #
print(f"\nStarting Optuna Optimization for {N_TRIALS} Trials...")

study = optuna.create_study(direction='minimize') 
study.optimize(objective, n_trials=N_TRIALS) 


# === Final Report and Model Saving === #
print("\n" + "#"*60)
print("Optimization Complete. Analyzing Results...")
print("#"*60 + "\n")

print(f"Best MAE: {study.best_value:.4f}")
print("Best parameters found:")
best_params = study.best_params
for key, value in best_params.items():
    print(f"  - {key}: {value}")

# Retrain the final best model
print("\n--- Retraining and Saving the Best Model ---")
final_model = LGBM_Regressor(
    n_estimators=best_params['n_estimators'],
    learning_rate=best_params['learning_rate'],
    lr_decay=best_params['lr_decay'],
    num_leaves=best_params['num_leaves'],
    max_depth=best_params['max_depth'],
    objective=best_params['objective'],
    weight=best_params['weight']
)
final_model.train(data_path=training_file_dir)

names = [ "Marbas", "Orobas", "Agares", "Foras", "Phobos", "Nemesis", "Kuutar"]
best_mae_name = random.choice(names)

# Save parameters for best model
json_name = f"{best_mae_name}_regressor_model_parameters.json"
param_file = os.path.join(model_save_dir, json_name)
with open(param_file, 'w') as json_file:
    json.dump(best_params, json_file, indent=4)

# Save model
final_model.save_model(model_save_dir, best_mae_name)

# Save all results
results_df = study.trials_dataframe()
results_csv_path = model_save_dir / f"{best_mae_name}_optuna_search_results.csv"
results_df.to_csv(results_csv_path, index=False)
print(f"\nFull optimization results saved to '{results_csv_path}'")