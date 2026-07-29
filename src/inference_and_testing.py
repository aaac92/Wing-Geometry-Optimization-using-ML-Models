import os
import sys
import json
import shutil
import tempfile
import re
import math
from collections import defaultdict
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import torch

# Import custom support modules (Ensure these are accessible in your project root)
from simulation_support import Xfoil_Polar_Creator as XPR
from simulation_support import Key_Point_Extraction, HossainWingRec_Creation, HossainWingFlat_Creation, HossainWing60DEG_Creation, AVL_Run_Case
from geometric_support import Airfoil_Generator
from file_handling_support import compression_Alma, get_all_files_from
from ai_support import Classifier, MLP_Regressor, LGBM_Regressor, SKNN_Regressor

# === Mandatory GPU Check === #
if not torch.cuda.is_available():
    print("ERROR: Execution stopped. This script requires a CUDA-enabled GPU.")
    sys.exit()
else:
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")

# === Directory configuration === #
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

datasets_dir = project_root / "data" / "datasets"
support_files_dir = project_root / "data" / "support"
binaries_dir = project_root / "bin"
results_dir = project_root / "results"

if sys.platform != "win32":
    xfoil_bin = binaries_dir / "xfoil"
    avl_bin = binaries_dir / "avl"
else:
    xfoil_bin = binaries_dir / "xfoil.exe"
    avl_bin = binaries_dir / "avl.exe"

case_file = support_files_dir / "Hossain_sqwing_case.run"
base_airfoil = support_files_dir / "naca0012_unit.csv"

dataset_num = 17
dataset_dir = datasets_dir / f"dataset_{dataset_num}"
training_file_dir = dataset_dir / f"regressor_dataset_{dataset_num}.csv"

models_save_dir = project_root / "models" / f"dataset_{dataset_num}"
classifier_dir = models_save_dir / "classifier"
mlp_regressor_dir = models_save_dir / "mlp_regressor"
lgbm_regressor_dir = models_save_dir / "lgbm_regressor"
sknn_regressor_dir = models_save_dir / "sknn_regressor"

final_results_dir = results_dir / f"dataset_{dataset_num}"
os.makedirs(final_results_dir, exist_ok=True)
dataset_name_prefix = f"AI_Test_Results_dataset_{dataset_num}"

# Registry mapping each wing-configuration identifier to its .geo creator class.
# Extend this tuple list if new wing configurations are added to the dataset.
WING_CONFIG_REGISTRY = [
    (HossainWingRec_Creation, "rec"),
    (HossainWingFlat_Creation, "flat"),
    (HossainWing60DEG_Creation, "60DEG"),
]

def find_model_prefix(folder_path: Path, unique_suffix: str) -> str:
    """Scans a folder to find a single model file and returns its prefix."""
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Model folder missing: {folder_path}")
    model_files = [f for f in os.listdir(folder_path) if f.endswith(unique_suffix)]
    if len(model_files) == 0:
        raise FileNotFoundError(f"No model with suffix '{unique_suffix}' in '{folder_path}'.")
    if len(model_files) > 1:
        raise ValueError(f"Ambiguous files. Found {len(model_files)} models in '{folder_path}'.")
    return model_files[0].removesuffix(unique_suffix)

def parse_mlp_prefix_for_hyperparams(prefix: str, path: Path) -> dict:
    """Loads MLP hyperparameters from the JSON parameters file."""
    filepath = path / f"{prefix}_parameters.json"
    if not filepath.exists():
        raise FileNotFoundError(f"Parameters file not found: {filepath}")
    with open(filepath, "r") as f:
        params = json.load(f)
    return {"hidden_size": int(params["hidden_size"]), "dropout_rate": float(params["dropout_rate"])}

def parse_classifier_prefix_for_hyperparams(prefix: str) -> dict:
    """Parses filename prefix to extract hidden_size for the Classifier."""
    try:
        return {'hidden_size': int(re.search(r'hs_(\d+)', prefix).group(1))}
    except AttributeError:
        raise ValueError(f"Could not parse hidden_size from prefix: '{prefix}'.")

def parse_dataset_columns(csv_path: Path):
    """
    Reads only the dataset header and classifies every feature column by type.
    The last CSV column is always the regression target and is excluded.

    Returns
    -------
    feature_columns : list[str]  -- all feature columns, in CSV order
    cst_l_cols      : list[str]  -- 'cst_l_X' columns, sorted by X
    cst_u_cols      : list[str]  -- 'cst_u_X' columns, sorted by X
    config_cols     : list[str]  -- remaining columns (the one-hot wing configuration flags)
    """
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    feature_columns = header[:-1]
    cst_l_cols = sorted(
        (c for c in feature_columns if re.fullmatch(r'cst_l_\d+', c)),
        key=lambda c: int(c.rsplit('_', 1)[1])
    )
    cst_u_cols = sorted(
        (c for c in feature_columns if re.fullmatch(r'cst_u_\d+', c)),
        key=lambda c: int(c.rsplit('_', 1)[1])
    )
    config_cols = [c for c in feature_columns if c not in cst_l_cols and c not in cst_u_cols]
    if not config_cols:
        raise ValueError(f"No hot-encoded wing configuration columns found in '{csv_path.name}'.")
    return feature_columns, cst_l_cols, cst_u_cols, config_cols

def match_config_columns(config_cols: list) -> dict:
    """Maps each one-hot config column name to its (creator_class, identifier) via substring match."""
    mapping = {}
    for col in config_cols:
        matches = [entry for entry in WING_CONFIG_REGISTRY if entry[1].lower() in col.lower()]
        if len(matches) != 1:
            raise ValueError(
                f"Could not uniquely match dataset column '{col}' to a wing configuration in "
                f"WING_CONFIG_REGISTRY. Matches found: {matches}. Update the registry to resolve this."
            )
        mapping[col] = matches[0]
    return mapping

def generate_cst_candidates(num_samples, n_cst, ranges_upper, ranges_lower):
    """Generates random CST-only candidate inputs (no wing configuration attached)."""
    feature_ranges = (
        [ranges_lower["first"]] + [ranges_lower["middle"]] * (n_cst - 2) + [ranges_lower["last"]] +
        [ranges_upper["first"]] + [ranges_upper["middle"]] * (n_cst - 2) + [ranges_upper["last"]]
    )
    candidates = np.zeros((num_samples, n_cst + n_cst), dtype=np.float32)
    for i, (low, high, step) in enumerate(feature_ranges):
        candidates[:, i] = np.random.uniform(low, high, size=num_samples)
    return candidates

def expand_with_configs(cst_candidates, feature_columns, cst_l_cols, cst_u_cols, config_cols):
    """
    Matches every CST combination to every hot-encoded wing configuration, producing one full
    feature row (in dataset column order) per (cst combination, config) pair -- e.g. 10 CST
    combinations x 3 configurations => 30 rows.

    Returns
    -------
    full_candidates : np.ndarray, shape (num_samples * num_configs, num_features)
    config_labels   : np.ndarray of str, the config column active in each row
    """
    num_samples = cst_candidates.shape[0]
    num_configs = len(config_cols)
    cst_col_order = cst_l_cols + cst_u_cols

    # Row r corresponds to (sample r // num_configs, config r % num_configs) in both blocks below.
    cst_repeated = np.repeat(cst_candidates, num_configs, axis=0)
    one_hot_block = np.tile(np.eye(num_configs, dtype=np.float32), (num_samples, 1))

    cst_pos = {col: i for i, col in enumerate(cst_col_order)}
    full_candidates = np.zeros((num_samples * num_configs, len(feature_columns)), dtype=np.float32)
    for f_idx, col in enumerate(feature_columns):
        if col in cst_pos:
            full_candidates[:, f_idx] = cst_repeated[:, cst_pos[col]]
        elif col in config_cols:
            full_candidates[:, f_idx] = one_hot_block[:, config_cols.index(col)]

    config_labels = np.tile(np.array(config_cols), num_samples)
    return full_candidates, config_labels

def get_active_config(feature_row: np.ndarray, feature_columns: list, config_cols: list) -> str:
    """Returns the name of the one-hot config column active (== 1) in a single feature row."""
    active = [c for c in config_cols if feature_row[feature_columns.index(c)] == 1]
    if len(active) != 1:
        raise ValueError(f"Expected exactly one active wing configuration column, found: {active}")
    return active[0]

def extract_force_data(file_path: Path):
    """Extracts airfoil name, CL, and CD from a force file. Returns NaN on failure."""
    airfoil, cl_tot, cd_tot = None, np.nan, np.nan
    if not file_path.exists():
        return airfoil, np.nan, np.nan
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if "Configuration:" in line and " for " in line:
                    airfoil = line.split(" for ", 1)[1].strip()
                elif 'CLtot =' in line:
                    cl_tot = float(line.split('=')[1].strip())
                elif 'CDtot =' in line:
                    cd_tot = float(line.split('=')[1].strip())
        if airfoil is not None:
            return airfoil, cl_tot, cd_tot
    except Exception as e:
        print(f"Warning: Failed to parse force data in {file_path.name}: {e}")
    return None, np.nan, np.nan

def save_results_to_txt(results_data: list, output_folder: Path):
    """Saves formatted text report handling NaN values properly."""
    filepath = output_folder / f"{dataset_name_prefix}_final_report.txt"
    sorted_results = sorted(
        results_data,
        key=lambda x: -np.inf if math.isnan(x['simulated_E']) else x['simulated_E'],
        reverse=True
    )
    with open(filepath, 'w') as f:
        f.write("="*110 + "\n")
        f.write(f"Final Aerodynamic Efficiency (E) Report for Dataset {dataset_num}\n")
        f.write("="*110 + "\n")
        header = (f"{'Rank':<5} {'Airfoil Name':<30} {'Wing Config':<12} {'Simulated E':<15} "
                  f"{'Predicted E':<15} {'Relative Error (%)':<20}\n")
        f.write(header)
        f.write("-"*110 + "\n")

        for i, item in enumerate(sorted_results):
            sim_E_str = f"{item['simulated_E']:.4f}" if not math.isnan(item['simulated_E']) else "NaN"
            pred_E_str = f"{item['predicted_E']:.4f}" if item['predicted_E'] is not None else "--"
            err = item['relative_error']
            error_str = f"{err:.2f}" if err is not None and not math.isnan(err) else "NaN"

            f.write(f"{i+1:<5} {item['airfoil_name']:<30} {item['wing_config']:<12} "
                    f"{sim_E_str:<15} {pred_E_str:<15} {error_str:<20}\n")
        f.write("="*110 + "\n")
    print(f"Successfully saved text report to '{filepath.name}'")

if __name__ == "__main__":
    start_time = datetime.now()
    print(f"Pipeline started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # === Dataset Schema Parsing === #
    print("# === Parsing Dataset Schema === #")
    feature_columns, cst_l_cols, cst_u_cols, config_cols = parse_dataset_columns(training_file_dir)
    if len(cst_l_cols) != len(cst_u_cols):
        raise ValueError(f"Asymmetric CST columns: {len(cst_l_cols)} lower vs {len(cst_u_cols)} upper.")
    n_cst = len(cst_l_cols)
    config_registry = match_config_columns(config_cols)
    print(f"Found {n_cst} CST coefficients per surface and {len(config_cols)} wing configuration(s): "
          f"{[config_registry[c][1] for c in config_cols]}")

    # === Model Initialization === #
    print("\n# === Discovering and Loading AI Models === #")
    classifier_prefix = find_model_prefix(classifier_dir, '_model.pth')
    mlp_reg_prefix = find_model_prefix(mlp_regressor_dir, '_model.pth')
    lgbm_reg_prefix = find_model_prefix(lgbm_regressor_dir, '_model.joblib')
    sknn_reg_prefix = find_model_prefix(sknn_regressor_dir, '_pipeline.joblib')

    classifier_hyperparams = parse_classifier_prefix_for_hyperparams(classifier_prefix)
    classifier = Classifier.load_model(classifier_dir, classifier_prefix, **classifier_hyperparams)

    mlp_hyperparams = parse_mlp_prefix_for_hyperparams(mlp_reg_prefix, mlp_regressor_dir)
    mlp_regressor = MLP_Regressor.load_model(mlp_regressor_dir, mlp_reg_prefix, **mlp_hyperparams)

    lgbm_regressor = LGBM_Regressor.load_model(lgbm_regressor_dir, lgbm_reg_prefix)
    sknn_regressor = SKNN_Regressor.load_model(sknn_regressor_dir, sknn_reg_prefix)

    # Verify classifier input
    num_cst_features = len(cst_l_cols) + len(cst_u_cols)
    if classifier.input_dim != num_cst_features:
        raise ValueError(
            f"Classifier input_dim ({classifier.input_dim}) does not match the "
            f"{num_cst_features} CST feature columns found in '{training_file_dir.name}'."
        )

    # === Inference & Ranking === #
    num_inference_runs = 10
    num_samples = 50000
    master_candidates = {}

    extra = 1 + 1/24
    ranges_upper = {'first': (-1/24, 1/24, 0.0625), 'middle': (-1/24, 1/12, 1/24), 'last': (-1/24, 0, 0.125)}
    ranges_lower = {'first': (1/24, 0.0625, 0.0625), 'middle': (1/24, 0.125*extra, 1/24), 'last': (-1/24, 0, 0.125)}

    # {model display name (proper noun prefix): (model object, list of best-of-run results)}
    model_registry = {
        mlp_reg_prefix: (mlp_regressor, []),
        lgbm_reg_prefix: (lgbm_regressor, []),
        sknn_reg_prefix: (sknn_regressor, []),
    }

    for i in range(num_inference_runs):
        print(f"\n--- Inference Run {i+1}/{num_inference_runs} ---")
        # Generate airfoil candidate cloud.
        cst_candidates = generate_cst_candidates(num_samples, n_cst, ranges_upper, ranges_lower)

        # Classifier works on CST combinations; invalid ones are discarded.
        validity_probs = classifier.predict_proba(cst_candidates)
        valid_indices = np.where(validity_probs.flatten() > 0.99)[0]
        if len(valid_indices) == 0:
            print("No valid candidates found in this run. Skipping.")
            continue
        valid_cst = cst_candidates[valid_indices]

        # Final wing configuration from valid CST combinations
        full_candidates, config_labels = expand_with_configs(valid_cst, feature_columns, cst_l_cols, cst_u_cols, config_cols)

        for model_name, (model, raw_best) in model_registry.items():
            preds = model.predict(full_candidates).flatten()
            best_idx = np.argmax(preds)
            raw_best.append({'E': preds[best_idx], 'features': full_candidates[best_idx], 'config_col': config_labels[best_idx]})

    # Rank each model's best results and populate the master dictionary
    for model_name, (_, raw_best) in model_registry.items():
        raw_best.sort(key=lambda x: x['E'], reverse=True)
        for rank, item in enumerate(raw_best, start=1):
            master_candidates[f"{model_name}_rank_{rank}"] = {
                "features": item['features'],
                "predicted_E": float(item['E']),
                "config_col": item['config_col'],
            }

    try:
        training_df = pd.read_csv(training_file_dir)
        best_idx = training_df.iloc[:, -1].idxmax()
        best_row = training_df.loc[best_idx]
        gt_features = best_row[feature_columns].to_numpy(dtype=np.float32)
        master_candidates["Ground_Truth_Best"] = {
            "features": gt_features,
            "predicted_E": float(best_row.iloc[-1]),  # Actual E from dataset
            "config_col": get_active_config(gt_features, feature_columns, config_cols),
        }
    except FileNotFoundError:
        print(f"\nCould not find ground truth dataset at '{training_file_dir}'.")

    # Export Top Predictions to JSON before simulation
    cst_col_idx = [feature_columns.index(c) for c in cst_l_cols + cst_u_cols]
    predictions_export_path = final_results_dir / f"{dataset_name_prefix}_predictions_only.json"
    with open(predictions_export_path, 'w') as f:
        export_data = {
            name: {
                "predicted_E": data["predicted_E"],
                "cst_coeffs": data["features"][cst_col_idx].tolist(),
                "wing_config": config_registry[data["config_col"]][1],
            }
            for name, data in master_candidates.items()
        }
        json.dump(export_data, f, indent=4)
    print(f"\n=> Initial predictions saved to {predictions_export_path.name}")

    # === Geometry Instantiation & Aerodynamic Simulation === #
    print("\n# === Beginning Data Validation === #")
    ASEQ = [-20, 20, 0.5]
    ANGLES = [ASEQ[0] + j * ASEQ[2] for j in range(int((ASEQ[1] - ASEQ[0]) / ASEQ[2]) + 1)]
    rar_Alma = compression_Alma()

    # Utilize tempfile for automatic cleanup and cross-platform compatibility
    with tempfile.TemporaryDirectory() as temp_dir_name:
        temp_root = Path(temp_dir_name)

        # Declare al temp sub-directories
        airfoils_temp = temp_root / "airfoils"
        polars_temp = temp_root / "polars"
        geo_temp = temp_root / "geometry"
        forces_temp = temp_root / "forces"
        support_temp = temp_root / "support"
        for d in [airfoils_temp, polars_temp, geo_temp, forces_temp, support_temp]:
            d.mkdir(parents=True, exist_ok=True)

        # Generate Airfoils
        print("\n--- Generating Airfoil Geometry ---")
        airfoil_creator = Airfoil_Generator(base_airfoil_csv=base_airfoil, n_cst=n_cst, verbose=False)
        for name, data in tqdm(master_candidates.items(), desc="Creating .dat files"):
            cst_coeffs = data['features'][cst_col_idx]
            x_mod, yu_mod, yl_mod = airfoil_creator.generate(cst_coeffs=cst_coeffs)
            airfoil_creator._create_dat_file(output_dir=airfoils_temp, n_cst=n_cst, x=x_mod, yl=yl_mod, yu=yu_mod, file_name=name)

        # Run XFOIL
        print("\n--- Generating Polars (XFOIL) ---")
        airfoil_paths = get_all_files_from(airfoils_temp, extension='dat')
        xfoil_case = XPR(xfoil_exe_path=str(xfoil_bin), 
                         main_dir=temp_root, 
                         airfoil_dir=airfoils_temp, 
                         polar_dir=polars_temp, 
                         verbose=False)
        for path in tqdm(airfoil_paths, desc="Running XFOIL"):
            airfoil_name = Path(path).stem
            polar_path = polars_temp / f"polar_{airfoil_name}.txt"
            # Does not require try/except due to internal handler
            xfoil_case.run_simulation(airfoil_file_path=path, 
                                      polar_file_path=str(polar_path), 
                                      alfa_sequence=ANGLES)

        # Create Wing Geometries, each candidate uses the creator matching its own hot-encoded config
        print("\n--- Generating Wing Geometry (.geo files) ---")
        polar_paths = get_all_files_from(polars_temp, extension='txt')
        key_points_data = Key_Point_Extraction(polar_file_paths=polar_paths).process_files()

        airfoil_paths_by_config = defaultdict(list)
        for name, data in master_candidates.items():
            airfoil_paths_by_config[data['config_col']].append(str(airfoils_temp / f"{name}.dat"))

        for config_col, subset_paths in airfoil_paths_by_config.items():
            creator_class, identifier = config_registry[config_col]
            print(f"  -> {len(subset_paths)} airfoil(s) under '{identifier}' configuration")
            geo_creator = creator_class(
                key_points_data=key_points_data,
                airfoil_abs_paths=subset_paths,
                output_dir=str(geo_temp),
                verbose=False,
                pb=True
            )
            geo_creator.create_geo_files()

        # Calculate Final Forces (AVL)
        print("\n--- Calculating Final Coefficients (AVL) ---")
        geo_paths = get_all_files_from(geo_temp, extension='avl')
        shutil.copy(case_file, support_temp)
        case_file_temp = support_temp / case_file.name

        for path in tqdm(geo_paths, desc="Running AVL sequentially"):
            avl_case = AVL_Run_Case(geo_files=[path], case_file=str(case_file_temp), avl_exe_path=str(avl_bin), output_dir=forces_temp)
            # Does not require try/except due to internal handler
            avl_case.run()

        print("\n--- Archiving Raw Data Files ---")
        rar_Alma.compress_folder(airfoils_temp, final_results_dir, f"{dataset_name_prefix}_airfoils")
        rar_Alma.compress_folder(polars_temp, final_results_dir, f"{dataset_name_prefix}_polars")
        rar_Alma.compress_folder(geo_temp, final_results_dir, f"{dataset_name_prefix}_geo")
        rar_Alma.compress_folder(forces_temp, final_results_dir, f"{dataset_name_prefix}_forces")

        # === Final Validation & Reporting === #
        print("\n--- Processing Final Aerodynamic Efficiency Validation ---")
        force_file_paths = get_all_files_from(forces_temp, extension='txt')
        final_results = []

        for fpath in tqdm(force_file_paths, desc="Validating simulated E vs predicted E"):
            airfoil_name, cl, cd = extract_force_data(Path(fpath))
            if airfoil_name is None:
                continue  # Skip unparseable files entirely

            candidate = master_candidates.get(airfoil_name, {})
            predicted_E = candidate.get("predicted_E")
            wing_config = config_registry[candidate["config_col"]][1] if "config_col" in candidate else "unknown"

            # Handle NaN for divergence cases (CD = 0 or missing data)
            if cd == 0 or math.isnan(cd) or math.isnan(cl):
                simulated_E, relative_error = np.nan, np.nan
            else:
                simulated_E = cl / cd
                relative_error = abs((simulated_E - predicted_E) / simulated_E) * 100 if predicted_E is not None else np.nan

            final_results.append({
                "airfoil_name": airfoil_name,
                "wing_config": wing_config,
                "simulated_CL": cl,
                "simulated_CD": cd,
                "simulated_E": simulated_E,
                "predicted_E": predicted_E,
                "relative_error": relative_error
            })

        # Save Final Reports
        if final_results:
            json_report_path = final_results_dir / f"{dataset_name_prefix}_final_report.json"
            with open(json_report_path, 'w') as f:
                # Convert NaN to None for strict JSON compliance
                clean_results = [
                    {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in res.items()}
                    for res in final_results
                ]
                json.dump(clean_results, f, indent=4)
            print(f"\n=> JSON final report saved to {json_report_path.name}")

            save_results_to_txt(final_results, final_results_dir)
        else:
            print("\nWARNING: No valid result geometries evaluated successfully.")

    # tempfile block ends here - cleanup is fully automatic.

    end_time = datetime.now()
    print(f"\nProcess completed successfully at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total execution time: {end_time - start_time}")