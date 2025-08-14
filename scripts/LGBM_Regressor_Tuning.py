import sys
import os
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from pathlib import Path
from ai_support import LGBM_Regressor

# === Setup and Configuration === #
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
# Add this path to the list of places Python looks for modules.
sys.path.insert(0, project_root)
# Path to regressor dataset
DATA_PATH = project_root / 'Airfoils_for_Regressor_E_Dataset_15.csv'
# Folder where the best trained models will be saved
MODEL_SAVE_FOLDER = project_root / "optimized_LGBM_regressor_model"

if not os.path.exists(DATA_PATH):
    print(f"[ERROR] Dataset not found at '{DATA_PATH}'. Please check the file path.")
    sys.exit()

# Define the grid of hyperparameters for the LGBM regressor.
param_grid = {
    'n_estimators': [2000, 2500],
    'learning_rate': [0.05, 0.1, 0.5],
    'num_leaves': [30, 40, 50],
    'max_depth': [15, 20, 25]
}


# === Initialize Best Model Trackers === #
# For MAE, a lower score is better.
best_mae_info = {'score': float('inf'), 'model': None, 'params': None}
# For R2 Score, a higher score is better.
best_r2_info = {'score': float('-inf'), 'model': None, 'params': None}


# === Grid Search Execution === #
all_results = []

parameter_combinations = list(itertools.product(
    param_grid['n_estimators'],
    param_grid['learning_rate'],
    param_grid['num_leaves'],
    param_grid['max_depth']
))

print(f"\nStarting LGBM Regressor Grid Search for {len(parameter_combinations)} Combinations...")

for i, params in enumerate(parameter_combinations):
    ne, lr, nl, md = params
    
    print("\n" + "="*60)
    print(f"RUN {i+1}/{len(parameter_combinations)}")
    print(f"PARAMETERS: N_Estimators={ne}, LR={lr}, NumLeaves={nl}, MaxDepth={md}")
    print("="*60)

    # Instantiate the regressor with the current set of parameters
    regressor = LGBM_Regressor(
        n_estimators=ne,
        learning_rate=lr,
        num_leaves=nl,
        max_depth=md
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
        'n_estimators': ne,
        'learning_rate': lr,
        'num_leaves': nl,
        'max_depth': md,
        'MAE': current_mae,
        'R2_Score': current_r2
    }
    all_results.append(run_result)
print("\n" + "#"*60)
print("Grid Search Complete. Analyzing Results...")
print("#"*60 + "\n")

# === Error Analysis on the Best Model (by MAE) === #
print("\n" + "#"*60)
print("Performing Error Analysis on the Best MAE Model...")
print("#"*60 + "\n")

# We need to recreate the exact test set that the best model was evaluated on.
# It is crucial that these parameters match the ones used in the model's class.
# All provided regressor classes use test_size=0.2 and random_state=42.

# Select the best model found during the grid search
best_model = best_mae_info['model']

if best_model:
    # Load and split the data exactly as it was during training
    print("Re-creating test set for analysis...")
    df_analysis = pd.read_csv(DATA_PATH)
    X_analysis = df_analysis.iloc[:, :-1].values
    y_analysis = df_analysis.iloc[:, -1].values

    _, X_test, _, y_test = train_test_split(
        X_analysis, y_analysis, test_size=0.2, random_state=42
    )

    # Make predictions on the recreated test set
    print("Generating predictions with the best model...")
    y_pred = best_model.predict(X_test)

    # Perform the error analysis and visualization
    # Ensure y_test and y_pred are flattened to 1D arrays for plotting
    y_test_flat = y_test.ravel()
    y_pred_flat = y_pred.ravel()
    
    residuals = y_test_flat - y_pred_flat

    # Plot: Actual vs. Predicted
    plt.figure(figsize=(16, 7))
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=y_test_flat, y=y_pred_flat, alpha=0.6)
    perfect_line = np.linspace(min(y_test_flat), max(y_test_flat), 100)
    plt.plot(perfect_line, perfect_line, color='red', linestyle='--')
    plt.title('Best Model: Actual vs. Predicted Values', fontsize=14)
    plt.xlabel('Actual Values (Test Set)', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.grid(True)

    # Plot: Residuals vs. Predicted
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=y_pred_flat, y=residuals, alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title('Best Model: Residuals vs. Predicted Values', fontsize=14)
    plt.xlabel('Predicted Values', fontsize=12)
    plt.ylabel('Residuals (Actual - Predicted)', fontsize=12)
    plt.grid(True)
    
    plt.suptitle('Error Analysis for Best Performing Model', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # Find and display the 10 worst outliers based on absolute error
    print("\n--- Top 10 Worst Predictions on Test Set (by Absolute Error) ---")
    errors = np.abs(residuals)
    top_10_worst_indices = np.argsort(errors)[-10:][::-1]
    
    for i, idx in enumerate(top_10_worst_indices):
        features = X_test[idx]
        actual = y_test_flat[idx]
        predicted = y_pred_flat[idx]
        
        feature_str = ""
        # Use your specific format if there are exactly 6 features
        if len(features) == 6:
            c1, c2, c3, c4, c5, c6 = features
            feature_str = f"[{c1:.4f}, {c2:.4f}, {c3:.4f}] [{c4:.4f}, {c5:.4f}, {c6:.4f}]"
        else:
            # Otherwise, use a generic format
            feature_str = "Features: [" + ", ".join([f"{f:.4f}" for f in features]) + "]"
            
        print(f"\n{i+1}.- {feature_str}")
        print(f"     Actual: {actual:.4f}, Predicted: {predicted:.4f}, Error: {errors[idx]:.4f}")
else:
    print("No best MAE model was found, skipping error analysis.")

# === Final Saving and Reporting === #
# --- Save the Best Models ---
print("--- Saving Best Performing Models ---")
if best_mae_info['model']:
    ne, lr, nl, md = best_mae_info['params']
    file_name = f"lgbm_ne_{ne}_lr_{lr}_nl_{nl}_md_{md}"
    best_mae_info['model'].save_model(MODEL_SAVE_FOLDER, file_name)
else:
    print("No best MAE model was captured to save.")

if best_r2_info['model']:
    if best_r2_info['model'] is not best_mae_info['model']:
        ne, lr, nl, md = best_r2_info['params']
        file_name = f"lgbm_ne_{ne}_lr_{lr}_nl_{nl}_md_{md}"
        best_r2_info['model'].save_model(MODEL_SAVE_FOLDER, file_name)
    else:
        print("Best R2 model is the same as the best MAE model. Not saving a duplicate.")
else:
    print("No best R2 model was captured to save.")

# Final Report
results_df = pd.DataFrame(all_results)
best_mae_run = results_df.sort_values(by="MAE", ascending=True).iloc[0]
best_r2_run = results_df.sort_values(by="R2_Score", ascending=False).iloc[0]

print("\n--- Best Parameters Based on MAE (Lower is Better) ---")
print(best_mae_run)

print("\n--- Best Parameters Based on R2 Score (Higher is Better) ---")
print(best_r2_run)

results_csv_path = MODEL_SAVE_FOLDER / "lgbm_grid_search_results.csv"
results_df.to_csv(results_csv_path, index=False)
print(f"\nFull grid search results have been saved to '{results_csv_path}'")