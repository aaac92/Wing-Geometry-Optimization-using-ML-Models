import sys
import os
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from pathlib import Path

# Add the project root to the Python path to allow importing 'ai_support'
try:
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    sys.path.insert(0, str(project_root))
    from ai_support import MLP_Regressor, LGBM_Regressor, SKNN_Regressor
except (ImportError, IndexError):
    print("Could not import the 'ai_support' module.")
    print("Please ensure this script is in the same directory as the other scripts,")
    print("and that 'ai_support.py' is in the parent directory.")
    sys.exit(1)


# === Helper Functions (from AI_Inference.py) === #

def find_model_prefix(folder_path: Path, unique_suffix: str) -> str:
    """
    Scans a folder to find a single model file and returns its prefix.
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"The specified model folder does not exist: {folder_path}")

    model_files = [f for f in os.listdir(folder_path) if f.endswith(unique_suffix)]

    if len(model_files) == 0:
        raise FileNotFoundError(f"AUTOMATION ERROR: No model file with suffix '{unique_suffix}' found in '{folder_path}'.")

    if len(model_files) > 1:
        raise ValueError(f"AUTOMATION ERROR: Ambiguous files. Found {len(model_files)} models with suffix '{unique_suffix}' in '{folder_path}'. Please ensure only one optimized model is in this folder.")

    full_filename = model_files[0]
    prefix = full_filename.removesuffix(unique_suffix)
    print(f"Found model prefix in '{folder_path.name}': {prefix}")
    return prefix

def parse_mlp_prefix_for_hyperparams(prefix: str, path: str) -> dict:
    """
    Loads hyperparameters from the JSON _parameters file.
    """
    filename = f"{prefix}_parameters.json"
    filepath = os.path.join(path, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Parameters file not found: {filepath}")

    with open(filepath, "r") as f:
        params = json.load(f)

    try:
        hidden_size = int(params["hidden_size"])
        dropout_rate = float(params["dropout_rate"])
    except KeyError as e:
        raise ValueError(f"Missing required key in JSON: {e}")

    hyperparams = {"hidden_size": hidden_size, "dropout_rate": dropout_rate}
    print(f"   - Parsed MLP hyperparameters from JSON: {hyperparams}")
    return hyperparams


# === Configuration and Path Setup === #

# Define the root directory of the project
# Note: This assumes the script is in the same folder as your tuning scripts
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent

# --- IMPORTANT: Configure the training file name used for the regressors --- #
# This must match the file used in your tuning scripts.
Training_File = "Airfoils_for_Regressor_E_Dataset_16.csv"
DATA_PATH = project_root / Training_File

# Check if the data file exists
if not os.path.exists(DATA_PATH):
    print(f"[ERROR] Dataset not found at '{DATA_PATH}'. Please check the file path.")
    sys.exit(1)

# Automatically determine the Dataset number from the filename
match = re.search(r"(?<=Dataset_)\d+", Training_File)
if match:
    dataset_num = int(match.group())
else:
    print(f"[ERROR] Could not extract dataset number from filename: '{Training_File}'")
    sys.exit(1)

# Define paths to the saved model folders
MODEL_PATHS = project_root / 'AI_Models' / f'Dataset_{dataset_num}'
MLP_REG_FOLDER = MODEL_PATHS / "optimized_MLP_regressor_model"
LGBM_REG_FOLDER = MODEL_PATHS / "optimized_LGBM_regressor_model"
SKNN_REG_FOLDER = MODEL_PATHS / "optimized_SKNN_regressor_model"


# === Model Discovery and Loading === #

print("# === Starting Error Analysis Plot Generation === #")
print("\n# === Finding and loading all trained regressor models... === #")

try:
    # Find model prefixes automatically
    mlp_reg_prefix = find_model_prefix(MLP_REG_FOLDER, '_model.pth')
    lgbm_reg_prefix = find_model_prefix(LGBM_REG_FOLDER, '_model.joblib')
    sknn_reg_prefix = find_model_prefix(SKNN_REG_FOLDER, '_pipeline.joblib')

    # Load models using the logic from the inference script
    mlp_hyperparams = parse_mlp_prefix_for_hyperparams(mlp_reg_prefix, MLP_REG_FOLDER)
    mlp_regressor = MLP_Regressor.load_model(MLP_REG_FOLDER, mlp_reg_prefix, **mlp_hyperparams)

    lgbm_regressor = LGBM_Regressor.load_model(LGBM_REG_FOLDER, lgbm_reg_prefix)
    sknn_regressor = SKNN_Regressor.load_model(SKNN_REG_FOLDER, sknn_reg_prefix)

except (FileNotFoundError, ValueError, NameError) as e:
    print(f"\nSETUP FAILED: {e}")
    print("Please ensure model folders exist and contain the correct saved files.")
    sys.exit(1)


# === Data Preparation for Analysis === #

print("\n# === Re-creating the test set for analysis... === #")
# Load the dataset and split it exactly as it was done during training
# The tuning scripts specify test_size=0.2 and random_state=42
df_analysis = pd.read_csv(DATA_PATH)
X_analysis = df_analysis.iloc[:, :-1].values
y_analysis = df_analysis.iloc[:, -1].values

_, X_test, _, y_test = train_test_split(
    X_analysis, y_analysis, test_size=0.2, random_state=42
)
print(f"Test set created with {len(X_test)} samples.")


# === Error Analysis and Plotting === #

print("\n# === Generating predictions and plotting results... === #")

# A dictionary to hold the loaded models and their names for easy iteration
models_to_analyze = {
    "MLP Regressor": mlp_regressor,
    "LGBM Regressor": lgbm_regressor,
    "SKNN Regressor": sknn_regressor
}

# Loop through each model to generate predictions and plots
for model_name, model in models_to_analyze.items():
    print(f"\n--- Analyzing {model_name} ---")

    # Generate predictions on the test set
    y_pred = model.predict(X_test)

    # Ensure y_test and y_pred are flattened 1D arrays for plotting
    y_test_flat = y_test.ravel()
    y_pred_flat = y_pred.ravel()

    # Calculate residuals
    residuals = y_test_flat - y_pred_flat

    # --- Plot 1: Actual vs. Predicted Values ---
    plt.figure(figsize=(9, 8)) # Creates a new window for this plot
    sns.scatterplot(x=y_test_flat, y=y_pred_flat, alpha=0.6)
    # Add a red dashed line representing a perfect prediction
    perfect_line = np.linspace(min(y_test_flat), max(y_test_flat), 100)
    plt.plot(perfect_line, perfect_line, color='red', linestyle='--')
    plt.title(f'{model_name}: Actual vs. Predicted Values', fontsize=16, fontweight='bold')
    plt.xlabel('Actual Values (Test Set)', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    # --- Plot 2: Residuals vs. Predicted Values ---
    plt.figure(figsize=(9, 8)) # Creates another new window for this plot
    sns.scatterplot(x=y_pred_flat, y=residuals, alpha=0.6)
    # Add a red dashed line at y=0, where residuals are zero
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title(f'{model_name}: Residuals vs. Predicted Values', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Values', fontsize=12)
    plt.ylabel('Residuals (Actual - Predicted)', fontsize=12)
    plt.grid(True)
    plt.tight_layout()

print("\nAll plots generated. Displaying now...")
plt.show()

print("\nAnalysis complete.")