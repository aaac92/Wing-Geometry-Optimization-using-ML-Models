import torch
import sys
import os
import re
import json
import optuna
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from pathlib import Path
from ai_support import MLP_Regressor # Import the regressor class

# === Mandatory GPU Check === #
# Remove conditional if CPU is desired but this is absolutely not recomended
if not torch.cuda.is_available():
    print("ERROR: Execution stopped. This script requires a CUDA-enabled GPU, unless you want to wait 1 million years between each epoch.")
    print("PyTorch did not find an available GPU on this system.")
    sys.exit()
else:
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")

# == Setup and Configuration === #
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
# Add this path to the list of places Python looks for modules.
sys.path.insert(0, project_root)
# Path to regressor dataset
Training_File = "Airfoils_for_Regressor_E_Dataset_16_autocorrected.csv"
DATA_PATH = project_root / Training_File
match = re.search(r"(?<=Dataset_)\d+", Training_File)
if match:
    dataset_num = int(match.group())
# Folder where the best trained models will be saved
MODEL_SAVE_FOLDER = project_root / "AI_Models" / f"Dataset_{dataset_num}" / "optimized_MLP_regressor_model"
os.makedirs(MODEL_SAVE_FOLDER, exist_ok=True)
if not os.path.exists(DATA_PATH):
    print(f"[ERROR] Dataset not found at '{DATA_PATH}'. Please check the file path.")
    sys.exit()

# === Optuna Objective Function === #
# This function defines a single training run. Optuna calls it for each trial.
N_TRIALS = 12
def objective(trial):
    """
    Takes a trial object from Optuna, which suggests
    hyperparameters, trains a model, and returns its performance score.
    """
    print("\n" + "="*50)
    print(f"Starting Trial {trial.number + 1}/{N_TRIALS}")

    # Define the hyperparameter search space 
    hs = trial.suggest_int('hidden_size', 384, 512)
    dr = trial.suggest_float('dropout_rate', 0.01, 0.05)
    lr = trial.suggest_float('learning_rate', 0.01, 0.05, log=True)
    log_trans = trial.suggest_categorical('log_trans', [True])

    print(f"PARAMETERS:")
    print(f"Hidden Size = {hs}")
    print(f"Dropout Rate = {dr}")
    print(f"Learning Rate = {lr}")
    print(f"Log Transform = {log_trans}")
    print("="*50)

    # Instantiate and train the classifier with the suggested hyperparameters.
    regressor = MLP_Regressor(
        hidden_size=hs,
        dropout_rate=dr,
        learning_rate=lr,
        weight='None',
        log_trans=log_trans
    )

    final_metrics = regressor.train(data_path=DATA_PATH)

    # Return the metric to optimize, in the classifier case the F1-Score.
    mae = final_metrics.get('MAE', 0)
    
    # Tracking of other attributes
    trial.set_user_attr('R2-Score', final_metrics.get('R2_Score', 0))

    return mae

# === Optuna Study Execution === #
print(f"\nStarting Optuna Optimization for {N_TRIALS} Trials...")

study = optuna.create_study(direction='minimize') # Create study
study.optimize(objective, n_trials=N_TRIALS) # Optimize study


# === Final Report and Model Saving === #
print("\n" + "#"*60)
print("Optimization Complete. Analyzing Results...")
print("#"*60 + "\n")

# Get the best trial results from the study.
print(f"Best MAE: {study.best_value:.4f}")
print("Best parameters found:")
best_params = study.best_params
for key, value in best_params.items():
    print(f"  - {key}: {value}")

# Retrain the final best model with the optimal parameters.
print("\n--- Retraining and Saving the Best Model ---")
final_model = MLP_Regressor(
    hidden_size=best_params['hidden_size'],
    dropout_rate=best_params['dropout_rate'],
    learning_rate=best_params['learning_rate'],
    weight='None',
    log_trans=best_params['log_trans']
)
# Final training
final_model.train(data_path=DATA_PATH)

# Create a dynamic filename from the best parameters and save the model
names = ["Murata", "Barbatos", "Beelzebul", "Buer", "Focalors", "Istaroth", "Phanes", "Asmoday", "Paimon", "Alice", "Ronnova", "Kuutar", "Columbina"]
best_mae_name = random.choice(names)
# Save parameters for best model
json_name = f"{best_mae_name}_regressor_model_parameters.json"
param_file = os.path.join(MODEL_SAVE_FOLDER, json_name)
with open(param_file, 'w') as json_file:
    json.dump(best_params, json_file, indent=4)
# Save model
file_name_prefix = f"{best_mae_name}_regressor_model"
final_model.save_model(MODEL_SAVE_FOLDER, file_name_prefix)

# Save all results to a CSV file for analysis.
results_df = study.trials_dataframe()
results_csv_path = MODEL_SAVE_FOLDER / "optuna_search_results.csv"
results_df.to_csv(results_csv_path, index=False)
print(f"\nFull optimization results saved to '{results_csv_path}'")

# === Error Analysis on the Best Model === #
print("\n" + "#"*60)
print("Performing Error Analysis on the Best Model...")
print("#"*60 + "\n")

if final_model:
    # Create an output folder for the analysis files
    ANALYSIS_OUTPUT_FOLDER = MODEL_SAVE_FOLDER / "error_analysis"
    os.makedirs(ANALYSIS_OUTPUT_FOLDER, exist_ok=True)
    print(f"Saving analysis files to: {ANALYSIS_OUTPUT_FOLDER}")

    # Load and split the data as it was during training
    print("Re-creating test set for analysis...")
    df_analysis = pd.read_csv(DATA_PATH)
    X_analysis = df_analysis.iloc[:, :-1].values
    y_analysis = df_analysis.iloc[:, -1].values

    _, X_test, _, y_test = train_test_split(
        X_analysis, y_analysis, test_size=0.2, random_state=42
    )

    # Make predictions on the recreated test set
    print("Generating predictions with the best model...")
    y_pred = final_model.predict(X_test)

    # Ensure y_test and y_pred are flattened to 1D arrays
    y_test_flat = y_test.ravel()
    y_pred_flat = y_pred.ravel()
    residuals = y_test_flat - y_pred_flat

    # --- Plot 1: Actual vs. Predicted (Saved as PNG) ---
    print("Generating and saving 'Actual vs. Predicted' plot...")
    plt.figure(figsize=(8, 7)) # Create a new, single figure
    sns.scatterplot(x=y_test_flat, y=y_pred_flat, alpha=0.6)
    perfect_line = np.linspace(min(y_test_flat), max(y_test_flat), 100)
    plt.plot(perfect_line, perfect_line, color='red', linestyle='--')
    plt.title('Best Model: Actual vs. Predicted Values', fontsize=14)
    plt.xlabel('Actual Values (Test Set)', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    # Save the figure to a file 
    plt.savefig(ANALYSIS_OUTPUT_FOLDER / "actual_vs_predicted.png")
    plt.close()

    # --- Plot 2: Residuals vs. Predicted (Saved as PNG) ---
    print("Generating and saving 'Residuals vs. Predicted' plot...")
    plt.figure(figsize=(8, 7)) # Create another new figure
    sns.scatterplot(x=y_pred_flat, y=residuals, alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title('Best Model: Residuals vs. Predicted Values', fontsize=14)
    plt.xlabel('Predicted Values', fontsize=12)
    plt.ylabel('Residuals (Actual - Predicted)', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(ANALYSIS_OUTPUT_FOLDER / "residuals_vs_predicted.png")
    plt.close() 

    # --- Save Top 10 Worst Predictions to a .txt file ---
    print("Finding worst outliers and saving report to .txt file...")
    errors = np.abs(residuals)
    top_10_worst_indices = np.argsort(errors)[-10:][::-1]
    
    # Define the path for the report file
    report_file_path = ANALYSIS_OUTPUT_FOLDER / "worst_predictions_report.txt"

    # Use 'with open' to create and write to the file
    with open(report_file_path, 'w') as f:
        f.write("--- Top 10 Worst Predictions on Test Set (by Absolute Error) ---\n\n")
        
        for i, idx in enumerate(top_10_worst_indices):
            features = X_test[idx]
            actual = y_test_flat[idx]
            predicted = y_pred_flat[idx]
            
            # Format the feature string
            if len(features) == 6:
                c1, c2, c3, c4, c5, c6 = features
                feature_str = f"[{c1:.4f}, {c2:.4f}, {c3:.4f}] [{c4:.4f}, {c5:.4f}, {c6:.4f}]"
            else:
                feature_str = "Features: [" + ", ".join([f"{f:.4f}" for f in features]) + "]"
            
            # Write each entry to the file
            f.write(f"{i+1}.- {feature_str}\n")
            f.write(f"     Actual: {actual:.4f}, Predicted: {predicted:.4f}, Error: {errors[idx]:.4f}\n\n")
    
    print("Error analysis complete.")

else:
    print("No best model was found, skipping error analysis.")