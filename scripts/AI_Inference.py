import numpy as np
import pandas as pd
import re
import os
import sys
import json
from pathlib import Path
from ai_support import Classifier, MLP_Regressor, LGBM_Regressor, SKNN_Regressor

def find_model_prefix(folder_path: Path, unique_suffix: str) -> str:
    """
    Scans a folder to find a single model file and returns its prefix.
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"The specified model folder does not exist: {folder_path}")
        
    # Find all files in the directory that end with the unique suffix
    model_files = [f for f in os.listdir(folder_path) if f.endswith(unique_suffix)]
    
    if len(model_files) == 0:
        raise FileNotFoundError(f"AUTOMATION ERROR: No model file with suffix '{unique_suffix}' found in '{folder_path}'.")
    
    if len(model_files) > 1:
        raise ValueError(f"AUTOMATION ERROR: Ambiguous files. Found {len(model_files)} models with suffix '{unique_suffix}' in '{folder_path}'. Please ensure only one optimized model is in this folder.")
    
    # Get the full filename (e.g., 'classifier_hs_256_model.pth')
    full_filename = model_files[0]
    
    # Remove the unique suffix to get the prefix (e.g., 'classifier_hs_256')
    prefix = full_filename.removesuffix(unique_suffix)
    
    print(f"Found model prefix in '{folder_path.name}': {prefix}")
    return prefix

def parse_mlp_prefix_for_hyperparams(prefix: str, path: str) -> dict:
    """
    Loads hyperparameters from the JSON _parameters file, 
    located in the given path. Extracts only hidden_size and dropout_rate.
    
    Args:
        prefix (str): Filename prefix used to identify the parameters file.
        path (str): Directory path where the JSON file is located.
    
    Returns:
        dict: Dictionary containing 'hidden_size' and 'dropout_rate'.
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
    
def parse_classifier_prefix_for_hyperparams(prefix: str) -> dict:
    """
    Parses a filename prefix to extract hidden_size for the Classifier.
    """
    try:
        hidden_size = int(re.search(r'hs_(\d+)', prefix).group(1))
        hyperparams = {'hidden_size': hidden_size}
        print(f"   - Parsed Classifier hyperparameters: {hyperparams}")
        return hyperparams
    except AttributeError:
        raise ValueError(f"Could not parse hidden_size from Classifier prefix: '{prefix}'.")

def generate_candidates(num_samples, num_features, ranges_upper, ranges_lower):
    """
    Generate candidate inputs constrained to feature ranges like the database generation phase.

    Parameters
    ----------
    num_samples : int
        Number of candidates to generate
    num_features : int
        Number of input features (classifier.input_dim)
    ranges_upper : dict
        Ranges for upper surface {'first', 'middle', 'last'}
    ranges_lower : dict
        Ranges for lower surface {'first', 'middle', 'last'}

    Returns
    -------
    np.ndarray
        Candidate array of shape (num_samples, num_features)
    """

    # Split into upper + lower (assuming symmetric feature count)
    half = num_features // 2
    num_middle = half - 2 

    feature_ranges = []

    # Build ranges for lower
    feature_ranges.append(ranges_lower["first"])  # first
    feature_ranges.extend([ranges_lower["middle"]] * num_middle)  # middle
    feature_ranges.append(ranges_lower["last"])   # last

    # Build ranges for upper
    feature_ranges.append(ranges_upper["first"])
    feature_ranges.extend([ranges_upper["middle"]] * num_middle)
    feature_ranges.append(ranges_upper["last"])

    assert len(feature_ranges) == num_features, "Mismatch in feature ranges"

    # Generate samples per feature
    candidates = np.zeros((num_samples, num_features), dtype=np.float32)
    for i, (low, high, step) in enumerate(feature_ranges):
        candidates[:, i] = np.random.uniform(low, high, size=num_samples)

    return candidates

def report_best_result(model_name: str, best_value: float, best_input: np.ndarray, feature_names: list):
    """
    Helper function to print a formatted result for a model.
    """
    print("\n" + "="*50)
    print(f"Best Result from: {model_name}")
    print("="*50)
    print(f"Highest Predicted Output (E): {best_value:.4f}")
    print("\nOptimal Input Parameters Found:")
    
    best_input_series = pd.Series(best_input, index=feature_names, name="Values")
    print(best_input_series.to_string())
    print("="*50)

def save_ranked_results_to_json(model_name: str, results_list: list, output_folder: Path):
    """
    Takes a list of collected results, ranks them, and saves them to a JSON file.
    Each item in results_list is a dict: {'prediction': value, 'inputs': array}
    """
    # Sort the list of dictionaries by the 'prediction' value in descending order
    sorted_results = sorted(results_list, key=lambda x: x['prediction'], reverse=True)

    final_json_list = []
    for i, result in enumerate(sorted_results):
        result_dict = {
            "rank": i + 1,
            "E_predicted": float(result['prediction']),
            "features": result['inputs'].tolist()  # Convert numpy array to a plain list
        }
        final_json_list.append(result_dict)

    # Define the output filename and path
    filename = f"{model_name}_best_10_results.json"
    filepath = output_folder / filename

    # Write the data to a JSON file
    with open(filepath, 'w') as f:
        json.dump(final_json_list, f, indent=4)
    
    print(f"   - Saved ranked results for '{model_name}' to '{filepath.name}'")

def save_ground_truth_to_json(best_value: float, best_input: np.ndarray, output_folder: Path):
    """
    Saves the single best result from the training data to a JSON file.
    """
    dataset_data = [{
        "rank": 1,
        "E_actual": float(best_value),
        "features": best_input.tolist() # Convert numpy array to a plain list
    }]
    
    # Define the output filename and path
    filename = "dataset_best_result.json"
    filepath = output_folder / filename

    # Write the data to a JSON file
    with open(filepath, 'w') as f:
        json.dump(dataset_data, f, indent=4)

    print(f"   - Saved ground truth best result to '{filepath.name}'")

# === CONFIGURE MODEL FOLDERS === #
# Define the root directory of project
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
# Add this path to the list of places Python looks for modules.
sys.path.insert(0, project_root)
# Path of data for ground truth
Training_File = "Airfoils_for_Regressor_E_Dataset_16_autocorrected.csv"
REGRESSOR_DATA_PATH = project_root / Training_File
match = re.search(r"(?<=Dataset_)\d+", Training_File)
if match:
    dataset_num = int(match.group())
MODEL_PATHS = project_root / 'AI_Models' / f'Dataset_{dataset_num}'
# Set the folder path for each of the 4 best models.
# The script will automatically find the files inside these folders.
CLASSIFIER_FOLDER = MODEL_PATHS / "optimized_classifier_model"
MLP_REG_FOLDER = MODEL_PATHS / "optimized_MLP_regressor_model"
LGBM_REG_FOLDER = MODEL_PATHS / "optimized_LGBM_regressor_model"
SKNN_REG_FOLDER = MODEL_PATHS / "optimized_SKNN_regressor_model"
# Output folder
RESULTS_FOLDER = project_root / 'AI_Inference_Results'
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# === INFERENCE EXECUTION === #
print("# === Starting Final Automated Inference Pipeline === #")

# Automatically find model prefixes
try:
    classifier_prefix = find_model_prefix(CLASSIFIER_FOLDER, '_model.pth')
    mlp_reg_prefix = find_model_prefix(MLP_REG_FOLDER, '_model.pth')
    lgbm_reg_prefix = find_model_prefix(LGBM_REG_FOLDER, '_model.joblib')
    sknn_reg_prefix = find_model_prefix(SKNN_REG_FOLDER, '_pipeline.joblib')
except (FileNotFoundError, ValueError) as e:
    print(f"SETUP FAILED: {e}")
    sys.exit()


# Load All Models
print("\n# === Loading all trained models... === #")
classifier_hyperparams = parse_classifier_prefix_for_hyperparams(classifier_prefix)
classifier = Classifier.load_model(CLASSIFIER_FOLDER, classifier_prefix, **classifier_hyperparams)
mlp_hyperparams = parse_mlp_prefix_for_hyperparams(mlp_reg_prefix, MLP_REG_FOLDER)
mlp_regressor = MLP_Regressor.load_model(MLP_REG_FOLDER, mlp_reg_prefix, **mlp_hyperparams)
lgbm_regressor = LGBM_Regressor.load_model(LGBM_REG_FOLDER, lgbm_reg_prefix)
sknn_regressor = SKNN_Regressor.load_model(SKNN_REG_FOLDER, sknn_reg_prefix)

# Loop for 10 inferences
num_inference_runs = 10
mlp_best_of_runs = []
lgbm_best_of_runs = []
sknn_best_of_runs = []

for i in range(num_inference_runs):
    print("\n" + "="*50)
    print(f"# === Starting Inference Run {i+1}/{num_inference_runs} === #")
    print("="*50)

    # Generate Candidate Inputs for this run
    print("\n# === Generating random candidate inputs... === #")
    num_samples = 50000
    num_features = classifier.input_dim
    extra = 1 + 1/24 # Multiplier for search space
    ranges_upper = { # Use same ranges as Database_Generation
        'first': (-1/24, 1/24, 0.0625),
        'middle': (-1/24, 1/12, 1/24),
        'last': (-1/24, 0, 0.125)
    }
    ranges_lower = {
        'first': (1/24, 0.0625, 0.0625),
        'middle': (1/24, 0.125*extra, 1/24),
        'last': (-1/24, 0, 0.125)
    }
    candidate_inputs = generate_candidates(num_samples, num_features, ranges_upper, ranges_lower)
    print(f"Generated {num_samples} candidates for run {i+1}/{num_inference_runs}")

    # Classify and Filter Candidates
    print("\n# === Using classifier to find valid candidates... === #")
    validity_probs = classifier.predict_proba(candidate_inputs)
    validity_threshold = 0.99
    valid_indices = np.where(validity_probs.flatten() > validity_threshold)[0]

    if len(valid_indices) == 0:
        print(f"No valid candidates found in this run with a threshold > {validity_threshold}. Skipping run.")
        continue

    valid_inputs = candidate_inputs[valid_indices]
    print(f"Found {len(valid_inputs)} likely valid candidates.")

    # Get Predictions and Find the BEST ONE for this run
    print(f"\n# === Getting predictions and finding best result of run {i+1}/{num_inference_runs} === #")
    preds_mlp = mlp_regressor.predict(valid_inputs).flatten()
    preds_lgbm = lgbm_regressor.predict(valid_inputs).flatten()
    preds_sknn = sknn_regressor.predict(valid_inputs).flatten()

    # Find and store the single best result from this run for each model
    best_idx_mlp = np.argmax(preds_mlp)
    mlp_best_of_runs.append({
        'prediction': preds_mlp[best_idx_mlp],
        'inputs': valid_inputs[best_idx_mlp]
    })

    best_idx_lgbm = np.argmax(preds_lgbm)
    lgbm_best_of_runs.append({
        'prediction': preds_lgbm[best_idx_lgbm],
        'inputs': valid_inputs[best_idx_lgbm]
    })

    best_idx_sknn = np.argmax(preds_sknn)
    sknn_best_of_runs.append({
        'prediction': preds_sknn[best_idx_sknn],
        'inputs': valid_inputs[best_idx_sknn]
    })
    print(f"Best results from run {i+1}/{num_inference_runs} have been collected.")

# === FINAL ANALYSIS AND SAVING === #
print("\n" + "="*50)
print("# === All Inference Runs Complete. Ranking and Saving Final Results... === #")
print("="*50)

# Save Ground Truth Best
try:
    training_df = pd.read_csv(REGRESSOR_DATA_PATH)
    best_actual_idx = training_df.iloc[:, -1].idxmax()
    best_actual_row = training_df.loc[best_actual_idx]
    best_actual_value = best_actual_row.iloc[-1]
    best_actual_inputs = best_actual_row.iloc[:-1].values
    
    save_ground_truth_to_json(best_actual_value, best_actual_inputs, RESULTS_FOLDER)

except FileNotFoundError:
    print(f"\nCould not find regressor training data at '{REGRESSOR_DATA_PATH}'. Skipping the ground truth comparison.")

# Rank and Save the collected top results from all runs
save_ranked_results_to_json("MLP_Regressor", mlp_best_of_runs, RESULTS_FOLDER)
save_ranked_results_to_json("LGBM_Regressor", lgbm_best_of_runs, RESULTS_FOLDER)
save_ranked_results_to_json("SKNN_Regressor", sknn_best_of_runs, RESULTS_FOLDER)

print("\nPipeline finished successfully. Results saved in 'AI_Inference_Results' folder.")