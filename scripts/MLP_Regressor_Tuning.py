import torch
import sys
import os
import itertools
import pandas as pd
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
# Path to your regressor dataset
DATA_PATH = project_root / 'Airfoils_for_Regressor_E_Dataset_15.csv'
# Folder where the best trained models will be saved
MODEL_SAVE_FOLDER = project_root / "optimized_MLP_regressor_model"

if not os.path.exists(DATA_PATH):
    print(f"[ERROR] Dataset not found at '{DATA_PATH}'. Please check the file path.")
    sys.exit()

# Define the grid of hyperparameters for the regressor.
param_grid = {
    'hidden_size': [256, 384, 512],
    'dropout_rate': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    'learning_rate': [0.01, 0.05, 0.075, 0.1, 0.25],
    'noise_level': [0.01, 0.02]
}

# === Grid Search Execution === #
all_results = []
best_mae_info = {'score': float('inf'), 'model': None, 'params': None}
best_r2_info = {'score': float('-inf'), 'model': None, 'params': None}

parameter_combinations = list(itertools.product(
    param_grid['hidden_size'],
    param_grid['dropout_rate'],
    param_grid['learning_rate'],
    param_grid['noise_level']
))

print(f"\nStarting Regressor Grid Search for {len(parameter_combinations)} Combinations...")

for i, params in enumerate(parameter_combinations):
    hs, dr, lr, nl = params
    
    print("\n" + "="*60)
    print(f"RUN {i+1}/{len(parameter_combinations)}")
    print(f"PARAMETERS: HiddenSize={hs}, Dropout={dr}, LR={lr}, Noise={nl}")
    print("="*60)

    # Instantiate the regressor with the current set of parameters
    regressor = MLP_Regressor(
        hidden_size=hs,
        dropout_rate=dr,
        learning_rate=lr,
        noise_level=nl
    )

    # Train the model and get the final evaluation metrics
    final_metrics = regressor.train(data_path=DATA_PATH)
    
    # Check if the current model is a new "best" and capture it
    current_mae = final_metrics.get('MAE', float('inf'))
    current_r2 = final_metrics.get('R2_Score', float('-inf'))

    if current_mae < best_mae_info['score']:
        best_mae_info['score'] = current_mae
        best_mae_info['model'] = regressor
        best_mae_info['params'] = params
        print(f"New Best MAE: {best_mae_info['score']:.4f}. Model captured.")

    if current_r2 > best_r2_info['score']:
        best_r2_info['score'] = current_r2
        best_r2_info['model'] = regressor
        best_r2_info['params'] = params
        print(f"New Best R2 Score: {best_r2_info['score']:.4f}. Model captured.")

    # Store the results of the run
    run_result = {
        'hidden_size': hs,
        'dropout_rate': dr,
        'learning_rate': lr,
        'noise_level': nl,
        'MAE': current_mae,
        'R2_Score': current_r2
    }
    all_results.append(run_result)


# === Final Saving and Reporting === #
print("\n" + "#"*60)
print("Grid Search Complete. Analyzing Results...")
print("#"*60 + "\n")

# Save the Best Models
print("--- Saving Best Performing Models ---")
if best_mae_info['model']:
    hs, dr, lr, nl = best_mae_info['params']
    file_name = f"regressor_hs_{hs}_dr_{dr}_lr_{lr}_nl_{nl}"
    best_mae_info['model'].save_model(MODEL_SAVE_FOLDER, file_name)
else:
    print("No best MAE model was captured to save.")

if best_r2_info['model']:
    if best_r2_info['model'] is not best_mae_info['model']:
        hs, dr, lr, nl = best_r2_info['params']
        file_name = f"regressor_hs_{hs}_dr_{dr}_lr_{lr}_nl_{nl}"
        best_r2_info['model'].save_model(MODEL_SAVE_FOLDER, file_name)
    else:
        print("Best R2 model is the same as the best MAE model. Not saving a duplicate.")
else:
    print("No best R2 model was captured to save.")

# Final Report
results_df = pd.DataFrame(all_results)
best_mae_run = results_df.sort_values(by="MAE", ascending=True).iloc[0]
best_r2_run = results_df.sort_values(by="R2_Score", ascending=False).iloc[0]

print("\n--- Best Parameters Based on MAE (Relative to other models, lower is better) ---")
print(best_mae_run)

print("\n--- Best Parameters Based on R2 Score From 0 to 1, 1 is best) ---")
print(best_r2_run)

results_csv_path = MODEL_SAVE_FOLDER / "regressor_grid_search_results.csv"
results_df.to_csv(results_csv_path, index=False)
print(f"\nFull grid search results have been saved to '{results_csv_path}'")