import pandas as pd
import numpy as np
import sys
import os
import json
from pathlib import Path
from ai_support import Quantile_LGBM_Regressor, Classifier

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

REGRESSOR_FOLDER_PATH = project_root / "AI_Models" / "Dataset_16" / "optimized_LGBM_Q_regressor"
REGRESSOR_NAME = "Focalors"  

NUM_SAMPLES = 50000  
OUTPUT_FILENAME = f"generated_data_predictions_{REGRESSOR_NAME}.csv"

def generate_candidates(num_samples, num_features, ranges_upper, ranges_lower):
    """
    Generate candidate inputs constrained to feature ranges.
    """
    # Split into upper + lower (assuming symmetric feature count)
    half = num_features // 2
    num_middle = half - 2

    feature_ranges = []

    # Build ranges for lower surface
    feature_ranges.append(ranges_lower["first"])
    feature_ranges.extend([ranges_lower["middle"]] * num_middle)
    feature_ranges.append(ranges_lower["last"])

    # Build ranges for upper surface
    feature_ranges.append(ranges_upper["first"])
    feature_ranges.extend([ranges_upper["middle"]] * num_middle)
    feature_ranges.append(ranges_upper["last"])

    assert len(feature_ranges) == num_features, "Mismatch in feature ranges"

    # Generate samples per feature
    candidates = np.zeros((num_samples, num_features), dtype=np.float32)
    for i, (low, high, _) in enumerate(feature_ranges): # Step is not used in np.random.uniform
        candidates[:, i] = np.random.uniform(low, high, size=num_samples)

    return candidates


def run_inference_with_generated_data():
    """
    Main function to load models, generate data, run inference, and save results.
    """
    print("🚀 Starting inference process with in-memory data generation...")

    # --- 1. Load the Classifier to get input dimensions ---
    print(f"🧠 Loading classifier '{CLASSIFIER_NAME}' to determine feature count...")
    try:
        classifier_model = Classifier.load_model(
            folder_path=str(CLASSIFIER_FOLDER_PATH),
            file_name_prefix=CLASSIFIER_NAME
        )
        num_features = classifier_model.input_dim
        print(f"✅ Classifier loaded. Determined input features: {num_features}")
    except FileNotFoundError as e:
        print(f"❌ ERROR: Could not load classifier model. Please check paths.")
        print(f"Details: {e}")
        return

    # --- 2. Load the Quantile Regressor Model ---
    param_file_path = REGRESSOR_FOLDER_PATH / f"{REGRESSOR_NAME}_Q_regressor_model_parameters.json"
    if not param_file_path.exists():
        print(f"❌ ERROR: Regressor parameter file not found at '{param_file_path}'")
        return

    print(f"📝 Loading regressor parameters from: {param_file_path}")
    with open(param_file_path, 'r') as f:
        regressor_params = json.load(f)
    quantiles_to_load = regressor_params.get('quantiles')

    print(f"🧠 Loading trained quantile regressor '{REGRESSOR_NAME}'...")
    try:
        regressor_model = Quantile_LGBM_Regressor.load_model(
            folder_path=str(REGRESSOR_FOLDER_PATH),
            file_name_prefix=f"{REGRESSOR_NAME}_Q_regressor",
            quantiles=quantiles_to_load
        )
        print("✅ Regressor loaded successfully!")
    except FileNotFoundError as e:
        print(f"❌ ERROR: Could not load regressor model files.")
        print(f"Details: {e}")
        return

    # --- 3. Generate Candidate Data In-Memory ---
    print(f"🔧 Generating {NUM_SAMPLES} candidate data points...")
    extra = 1 + 1/24  # Multiplier for search space
    ranges_upper = {
        'first': (-1/24, 1/24, 0.0625),
        'middle': (-1/24, 1/12, 1/24),
        'last': (-1/24, 0, 0.125)
    }
    ranges_lower = {
        'first': (1/24, 0.0625, 0.0625),
        'middle': (1/24, 0.125 * extra, 1/24),
        'last': (-1/24, 0, 0.125)
    }
    
    candidate_inputs = generate_candidates(NUM_SAMPLES, num_features, ranges_upper, ranges_lower)
    print("✅ Data generation complete.")

    # --- 4. Perform Inference ---
    print("⏳ Making predictions on the generated data...")
    predictions_df = regressor_model.predict(candidate_inputs)
    print("✅ Predictions generated!")

    # --- 5. Combine and Save the Results ---
    print("📎 Combining generated data with predictions...")
    # Convert numpy array to DataFrame for easy concatenation
    input_features_df = pd.DataFrame(candidate_inputs, columns=[f'feature_{i}' for i in range(num_features)])
    results_df = pd.concat([input_features_df, predictions_df], axis=1)

    output_path = REGRESSOR_FOLDER_PATH / OUTPUT_FILENAME
    print(f"💾 Saving results to: {output_path}")
    results_df.to_csv(output_path, index=False)
    
    print("\n🎉 Inference complete! The output file contains the generated features plus their quantile predictions.")
    print(f"Output file location: {output_path}")


if __name__ == "__main__":
    run_inference_with_generated_data()