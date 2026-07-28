import torch
import sys
import os
import optuna
import json
import random
from pathlib import Path
from ai_support import MLP_Regressor

# === Mandatory GPU Check === #
if not torch.cuda.is_available():
    print("ERROR: Execution stopped. This script requires a CUDA-enabled GPU, unless you want to wait 1 million years between each epoch.")
    sys.exit()
else:
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")

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
model_save_dir = project_root / "models" / f"dataset_{dataset_num}" / "mlp_regressor"
os.makedirs(model_save_dir, exist_ok=True)
# Verify that dataset exists
if not os.path.exists(dataset_dir):
    print(f"[ERROR] Dataset not found at '{training_file_dir}'. Please check the file path.")
    sys.exit()

# === Optuna Objective Function === #
N_TRIALS = 12
def objective(trial):
    """
    Takes a trial object from Optuna, suggests hyperparameters, 
    dynamically constructs model depth/width, and evaluates the regressor.
    """
    print("\n" + "="*50)
    print(f"Starting Trial {trial.number + 1}/{N_TRIALS}")

    # Dynamically build network architecture
    n_layers = trial.suggest_int('n_layers', 2, 6)
    hidden_layer_sizes = [
        trial.suggest_int(f'n_units_l{i}', 64, 512) for i in range(n_layers)
    ]
    
    dr = trial.suggest_float('dropout_rate', 0.01, 0.05)
    lr = trial.suggest_float('learning_rate', 0.001, 0.05, log=True)
    log_trans = trial.suggest_categorical('log_trans', [True])
    patience = trial.suggest_int('patience', 30, 80)

    print(f"PARAMETERS:")
    print(f"Layers = {hidden_layer_sizes}")
    print(f"Dropout Rate = {dr}")
    print(f"Learning Rate = {lr}")
    print(f"Log Transform = {log_trans}")
    print(f"Patience = {patience}")
    print("="*50)

    # Instantiate and train the regressor
    regressor = MLP_Regressor(
        hidden_layer_sizes=hidden_layer_sizes,
        dropout_rate=dr,
        learning_rate=lr,
        patience=patience,
        weight='None',
        log_trans=log_trans
    )

    final_metrics = regressor.train(dataset_dir=training_file_dir)

    # Optimize for MAE
    mae = final_metrics.get('MAE', 0)
    
    trial.set_user_attr('R2-Score', final_metrics.get('R2_Score', 0))

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

# Reconstruct best architecture list
best_hidden_layer_sizes = [
    best_params[f'n_units_l{i}'] for i in range(best_params['n_layers'])
]

# Retrain the final best model
print("\n--- Retraining and Saving the Best Model ---")
final_model = MLP_Regressor(
    hidden_layer_sizes=best_hidden_layer_sizes,
    dropout_rate=best_params['dropout_rate'],
    learning_rate=best_params['learning_rate'],
    patience=best_params['patience'],
    weight='None',
    log_trans=best_params['log_trans']
)
final_model.train(dataset_dir=dataset_dir)

names = ["Murata", "Beelzebul", "Buer", "Odysseus", "Paimon", "Ronnova", "Columbina"]
best_mae_name = random.choice(names)

# Save parameters for best model (Dumping `best_params` inherently maps the entire dynamic architecture)
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