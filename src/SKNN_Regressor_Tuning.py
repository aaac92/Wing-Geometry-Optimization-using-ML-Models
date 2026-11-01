import sys
import os
import json
import random
import itertools
import pandas as pd
from pathlib import Path
from ai_support import SKNN_Regressor 
from file_handling_support.model_names import pick_model_name

# === Directory configuration === #
# Get the path to the src directory.
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
datasets_dir = project_root / "data" / "datasets"
# Path to dataset
dataset_num = "17.5" # text: decimal datasets are rebuilds (17.5 rebuilds 17)
dataset_name = f"regressor_dataset_{dataset_num}"
dataset_dir = datasets_dir / f"dataset_{dataset_num}"
training_file_dir = dataset_dir / f"{dataset_name}.csv"
model_save_dir = project_root / "models" / f"dataset_{dataset_num}" / "sknn_regressor"
# Verify that dataset exists
os.makedirs(model_save_dir, exist_ok=True)
if not os.path.exists(training_file_dir):
    print(f"[ERROR] Dataset not found at '{training_file_dir}'. Please check the file path.")
    sys.exit()

# Parameters that this model takes
param_grid = {
    'degree': [2, 3, 4, 5, 6],
    'weight': ['None', 'instance', 'frequency', 'linear', 'exponential', 'quantile', 'polynomial'] 
}

# === Initialize Best Model Trackers === #
# For MAE, a lower score is better.
best_mae_info = {'score': float('inf'), 'model': None, 'params': None}

# === Grid Search Execution === #
all_results = []

parameter_combinations = list(itertools.product(
    param_grid['degree'],
    param_grid['weight']
))

print(f"\nStarting SKNN Regressor Grid Search for {len(parameter_combinations)} Combinations...")

for i, params in enumerate(parameter_combinations):
    d, w = params
    
    print("\n" + "="*80)
    print(f"RUN {i+1}/{len(parameter_combinations)}")
    print(f"PARAMETERS: Degree={d}, Weight Mode={w}")
    print("="*80)

    # Instantiate the regressor with the current parameters
    regressor = SKNN_Regressor(
        degree=d,
        weight=w 
    )

    # Train the model using the correct data_path string
    final_metrics = regressor.train(data_path=str(training_file_dir))
    
    # Check if the current model is a new "best" and capture it
    current_mae = final_metrics.get('MAE', float('inf'))
    current_r2 = final_metrics.get('R2_Score', float('-inf'))

    if current_mae < best_mae_info['score']:
        best_mae_info['score'] = current_mae
        best_mae_info['model'] = regressor
        best_mae_info['params'] = params
        print(f"New Best MAE: {best_mae_info['score']:.4f}. Model captured.")

    # Store the results of the run
    run_result = {
        'Degree': d,
        'Weight Mode': w,
        'MAE': current_mae,
        'R2_Score': current_r2
    }
    all_results.append(run_result)

print("\n" + "#"*60)
print("Grid Search Complete. Analyzing Results...")
print("#"*60 + "\n")

# === Final Saving and Reporting === #
names = ["Andras", "Stolas", "Haagenti", "Amy", "Decarabia", "Hypnos", "Luonnotar"]
# Avoids the names of models already saved for this dataset (any type, any version, old models included)
best_mae_name = pick_model_name(names, project_root / "models" / f"dataset_{dataset_num}", kind="SKNN regressor")

print("--- Saving Best Performing Model ---")
if best_mae_info['model']:
    d, w = best_mae_info['params']
    best_result = {
        'Degree': d,
        'Weight Mode': w
    }
    json_name = f"{best_mae_name}_parameters.json"
    param_file = os.path.join(model_save_dir, json_name)
    with open(param_file, 'w') as json_file:
        json.dump(best_result, json_file, indent=4)
        
    best_mae_info['model'].save_model(model_save_dir, best_mae_name)
else:
    print("No best MAE model was captured to save.")

# --- Final Report ---
results_df = pd.DataFrame(all_results)
best_mae_run = results_df.sort_values(by="MAE", ascending=True).iloc[0]

print("\n--- Best Parameters Based on MAE (Lower is Better) ---")
print(f"\nName: {best_mae_name}")
print(best_mae_run)

# Dynamically save the results to match the reference point syntax
results_csv_path = model_save_dir / f"{best_mae_name}_grid_search_results.csv"
results_df.to_csv(results_csv_path, index=False)
print(f"\nFull grid search results have been saved to '{results_csv_path}'")