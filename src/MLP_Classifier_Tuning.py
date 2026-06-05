import torch
import sys
import os
import re
import optuna
from pathlib import Path
from ai_support import Classifier

# === Mandatory GPU Check === #
# This script will not run without a CUDA-enabled GPU.
if not torch.cuda.is_available():
    print("ERROR: Execution stopped. This script requires a CUDA-enabled GPU.")
    print("PyTorch did not find an available GPU on this system.")
    sys.exit()
else:
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")


# === Setup and Configuration === #
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
# Add this path to the list of places Python looks for modules.
sys.path.insert(0, project_root)
# Path to dataset
Training_File = "Airfoils_for_Classifier_Dataset_16.csv"
DATA_PATH = project_root / Training_File
match = re.search(r"(?<=Dataset_)\d+", Training_File)
if match:
    dataset_num = int(match.group())
# Folder where the best trained models will be saved
MODEL_SAVE_FOLDER = project_root / "AI_Models" / f"Dataset_{dataset_num}" / "optimized_classifier_model"
os.makedirs(MODEL_SAVE_FOLDER, exist_ok=True)
# Check if the dataset exists before starting
if not os.path.exists(DATA_PATH):
    print(f"[ERROR] Dataset not found at '{DATA_PATH}'. Please check the file path.")
    sys.exit()

# === Optuna Objective Function === #
# This function defines a single training run. Optuna calls it for each trial.
N_TRIALS = 30
def objective(trial):
    """
    Takes a trial object from Optuna, which suggests
    hyperparameters, trains a model, and returns its performance score.
    """
    print("\n" + "="*50)
    print(f"Starting Trial {trial.number + 1}/{N_TRIALS}")

    # Define the hyperparameter search space 
    hidden_size = trial.suggest_int('hidden_size', 256, 512)
    lr = trial.suggest_float('learning_rate', 0.05, 0.1, log=True)
    epochs = trial.suggest_int('epochs', 400, 600)

    print(f"PARAMETERS: Hidden Size={hidden_size}, Learning Rate={lr}, Epochs={epochs}")
    print("="*50)

    # Instantiate and train the classifier with the suggested hyperparameters.
    classifier = Classifier(
        hidden_size=hidden_size,
        epochs=epochs,
        lr=lr
    )

    final_metrics = classifier.train(data_path=DATA_PATH)

    # Return the metric to optimize, in the classifier case the F1-Score.
    f1_score = final_metrics.get('F1-Score', 0)
    
    # Tracking of other attributes
    trial.set_user_attr('F2-Score', final_metrics.get('F2-Score', 0))
    trial.set_user_attr('Accuracy', final_metrics.get('Accuracy', 0))

    return f1_score


# === Optuna Study Execution === #
print(f"\nStarting Optuna Optimization for {N_TRIALS} Trials...")

study = optuna.create_study(direction='maximize') # Create study
study.optimize(objective, n_trials=N_TRIALS) # Optimize study


# === Final Report and Model Saving === #
print("\n" + "#"*60)
print("Optimization Complete. Analyzing Results...")
print("#"*60 + "\n")

# Get the best trial results from the study.
print(f"Best F1-Score: {study.best_value:.4f}")
print("Best parameters found:")
best_params = study.best_params
for key, value in best_params.items():
    print(f"  - {key}: {value}")

# Retrain the final best model with the optimal parameters.
print("\n--- Retraining and Saving the Best Model ---")
final_model = Classifier(
    hidden_size=best_params['hidden_size'],
    epochs=best_params['epochs'],
    lr=best_params['learning_rate']
)
# Final training
final_model.train(data_path=DATA_PATH)

# Create a dynamic filename from the best parameters and save the model
file_name = (f"classifier_hs_{best_params['hidden_size']}_"
             f"lr_{best_params['learning_rate']}_e_{best_params['epochs']}")
final_model.save_model(MODEL_SAVE_FOLDER, file_name)

# Save all results to a CSV file for analysis.
results_df = study.trials_dataframe()
results_csv_path = MODEL_SAVE_FOLDER / "optuna_search_results.csv"
results_df.to_csv(results_csv_path, index=False)
print(f"\nFull optimization results saved to '{results_csv_path}'")