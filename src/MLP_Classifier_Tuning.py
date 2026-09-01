import torch
import sys
import os
import optuna
import json
import random
from pathlib import Path
from ai_support import Classifier

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
dataset_name = f"classifier_dataset_{dataset_num}"
dataset_dir = datasets_dir / f"dataset_{dataset_num}"
training_file_dir = dataset_dir / f"{dataset_name}.csv"
model_save_dir = project_root / "models" / f"dataset_{dataset_num}" / "classifier"
os.makedirs(model_save_dir, exist_ok=True)
# Verify that dataset exists
if not os.path.exists(training_file_dir):
    print(f"[ERROR] Dataset not found at '{training_file_dir}'. Please check the file path.")
    sys.exit()

# === Optuna Objective Function === #
N_TRIALS = 50
def objective(trial):
    """
    Takes a trial object from Optuna, suggests hyperparameters including
    dynamic network architecture, trains the model, and returns its performance score.
    """
    print("\n" + "="*50)
    print(f"Starting Trial {trial.number + 1}/{N_TRIALS}")

    # Dynamically build network architecture (number of layers and nodes per layer)
    n_layers = trial.suggest_int('n_layers', 1, 4)
    hidden_layer_sizes = [
        trial.suggest_int(f'n_units_l{i}', 128, 512) for i in range(n_layers)
    ]
    
    lr = trial.suggest_float('learning_rate', 0.01, 0.1, log=True)
    epochs = trial.suggest_int('epochs', 400, 600)
    patience = trial.suggest_int('patience', 20, 60) # Exploring early stopping efficiency

    print(f"PARAMETERS: Layers={hidden_layer_sizes}, Learning Rate={lr}, Epochs={epochs}, Patience={patience}")
    print("="*50)

    # Instantiate and train the classifier with dynamic architectures
    classifier = Classifier(
        hidden_layer_sizes=hidden_layer_sizes,
        epochs=epochs,
        lr=lr,
        patience=patience
    )

    final_metrics = classifier.train(data_path=training_file_dir)

    # Optimize for F1-Score
    f1_score = final_metrics.get('F1-Score', 0)
    
    trial.set_user_attr('F2-Score', final_metrics.get('F2-Score', 0))
    trial.set_user_attr('Accuracy', final_metrics.get('Accuracy', 0))

    return f1_score


# === Optuna Study Execution === #
print(f"\nStarting Optuna Optimization for {N_TRIALS} Trials...")

study = optuna.create_study(direction='maximize') 
study.optimize(objective, n_trials=N_TRIALS) 


# === Final Report and Model Saving === #
print("\n" + "#"*60)
print("Optimization Complete. Analyzing Results...")
print("#"*60 + "\n")

print(f"Best F1-Score: {study.best_value:.4f}")
print("Best parameters found:")
best_params = study.best_params
for key, value in best_params.items():
    print(f"  - {key}: {value}")

# Reconstruct best architecture list
best_hidden_layer_sizes = [
    best_params[f'n_units_l{i}'] for i in range(best_params['n_layers'])
]

# Retrain the final best model with the optimal parameters
print("\n--- Retraining and Saving the Best Model ---")
final_model = Classifier(
    hidden_layer_sizes=best_hidden_layer_sizes,
    epochs=best_params['epochs'],
    lr=best_params['learning_rate'],
    patience=best_params['patience']
)

final_model.train(data_path=training_file_dir)

# Setup names for saving
names = ["Barbatos", "Focalors", "Istaroth", "Calipso", "Asmoday", "Lauma"]
best_f1_name = random.choice(names)

# Save parameters for best model
json_name = f"{best_f1_name}_parameters.json"
param_file = os.path.join(model_save_dir, json_name)
with open(param_file, 'w') as json_file:
    json.dump(best_params, json_file, indent=4)

# Save the model
final_model.save_model(model_save_dir, best_f1_name)

# Save all results to a CSV file for analysis
results_df = study.trials_dataframe()
results_csv_path = model_save_dir / f"{best_f1_name}_optuna_search_results.csv"
results_df.to_csv(results_csv_path, index=False)
print(f"\nFull optimization results saved to '{results_csv_path}'")