import torch
import sys
import os
import itertools
import pandas as pd
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
DATA_PATH = project_root / 'Airfoils_for_Classifier_Dataset_15.csv'
MODEL_SAVE_FOLDER = project_root / "optimized_classifier_model"

# Check if the dataset exists before starting
if not os.path.exists(DATA_PATH):
    print(f"[ERROR] Dataset not found at '{DATA_PATH}'. Please check the file path.")
    sys.exit()

# Define the grid of hyperparameters to search through.
param_grid = {
    'hidden_size': [256, 512],
    'learning_rate': [0.075, 0.1, 0.5],
    'epochs': [600, 700]
}


# === Grid Search Execution === #
all_results = []
best_f1_info = {'score': -1.0, 'model': None, 'params': None}
best_f2_info = {'score': -1.0, 'model': None, 'params': None}

# Create a list of all hyperparameter combinations
parameter_combinations = list(itertools.product(
    param_grid['hidden_size'],
    param_grid['learning_rate'],
    param_grid['epochs']
))

print(f"\nStarting Grid Search for {len(parameter_combinations)} Combinations...")

for i, params in enumerate(parameter_combinations):
    hidden_size, lr, epochs = params
    
    print("\n" + "="*50)
    print(f"RUN {i+1}/{len(parameter_combinations)}")
    print(f"PARAMETERS: Hidden Size={hidden_size}, Learning Rate={lr}, Epochs={epochs}")
    print("="*50)

    # Instantiate the classifier with the current set of parameters
    classifier = Classifier(
        hidden_size=hidden_size,
        epochs=epochs,
        lr=lr
    )

    # The train method automatically calls evaluate and returns the final metrics
    final_metrics = classifier.train(data_path=DATA_PATH)

    # Check if the current model is a new "best" and capture it if so
    current_f1 = final_metrics.get('F1-Score', 0)
    current_f2 = final_metrics.get('F2-Score', 0)

    if current_f1 > best_f1_info['score']:
        best_f1_info['score'] = current_f1
        best_f1_info['model'] = classifier
        best_f1_info['params'] = params # `params` holds the current (hs, lr, e) tuple
        print(f"New Best F1-Score: {best_f1_info['score']:.4f}. Model captured.")

    if current_f2 > best_f2_info['score']:
        best_f2_info['score'] = current_f2
        best_f2_info['model'] = classifier
        best_f2_info['params'] = params
        print(f"New Best F2-Score: {best_f2_info['score']:.4f}. Model captured.")

    # Store the results of the run
    run_result = {
        'hidden_size': hidden_size,
        'learning_rate': lr,
        'epochs': epochs,
        'Accuracy': final_metrics.get('Accuracy', 0),
        'F1-Score': final_metrics.get('F1-Score', 0),
        'F2-Score': final_metrics.get('F2-Score', 0)
    }
    all_results.append(run_result)


# === Final Report === #
print("\n" + "#"*60)
print("Grid Search Complete. Analyzing Results...")
print("#"*60 + "\n")

# Save the best models found during the entire search
print("--- Saving Best Performing Models ---")
if best_f1_info['model']:
    # Unpack the parameters from the stored tuple
    hs, lr, e = best_f1_info['params']
    # Create the dynamic filename
    file_name = f"classifier_hs_{hs}_lr_{lr}_e_{e}"
    best_f1_info['model'].save_model(MODEL_SAVE_FOLDER, file_name)
else:
    print("No best F1 model was captured to save.")

# Save the best F2 model with its dynamic name
if best_f2_info['model']:
    # Check if the F2 model is the same object as the F1 model to avoid saving a duplicate
    if best_f2_info['model'] is not best_f1_info['model']:
        hs, lr, e = best_f2_info['params']
        file_name = f"classifier_hs_{hs}_lr_{lr}_e_{e}"
        best_f2_info['model'].save_model(MODEL_SAVE_FOLDER, file_name)
    else:
        print("Best F2 model is the same as the best F1 model. Not saving a duplicate.")
else:
    print("No best F2 model was captured to save.")

# Convert results to a pandas DataFrame for easy analysis
results_df = pd.DataFrame(all_results)

# Find and display the best parameters based on F1-Score
best_f1_run = results_df.sort_values(by="F1-Score", ascending=False).iloc[0]

# Find and display the best parameters based on F2-Score
best_f2_run = results_df.sort_values(by="F2-Score", ascending=False).iloc[0]

print("--- Best Parameters Based on F1-Score ---")
print(best_f1_run)

print("\n--- Best Parameters Based on F2-Score (Emphasizes Recall) ---")
print(best_f2_run)

# Save the full results log to a CSV file for your records
results_csv_path = MODEL_SAVE_FOLDER / "grid_search_results.csv"
results_df.to_csv(results_csv_path, index=False)
print(f"\nFull grid search results have been saved to '{results_csv_path}'")