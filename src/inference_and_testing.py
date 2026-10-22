"""
Inference + validation pipeline for the trained airfoil/wing AI models.

Usage:
    python inference_and_testing.py --classifier <name> --models <name1> [<name2> ...]
                                    [--dataset 17] [--wing-mode Hossain_config|single_configuration]
                                    [--seed HEX [--decimal]]

Flow:
    1. For each of the 10 inference rounds (one fixed seed per round, shared by all models):
       uniform random CST candidates + local-refinement candidates around the best dataset designs
       -> classifier validity filter -> pairing with every wing configuration
       -> every selected regressor predicts sequentially -> best candidate per model is stored.
    2. Results of all models are pooled, repeated feature vectors are merged (the models that
       selected them are kept as metadata), and the unique candidates are exported.
    3. Every unique candidate is simulated sequentially (XFOIL -> key points -> .avl geometry -> AVL).
    4. Simulated E is compared against each model's predicted E and reports are written.

No multiprocessing is used anywhere in this script.
"""
import argparse
import os
import sys
import json
import shutil
import re
import math
import secrets
import time
import platform
from collections import defaultdict
from types import SimpleNamespace
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
from ai_support import Classifier, MLP_Regressor, LGBM_Regressor

# =============================================================================================== #
# === Constants === #
# =============================================================================================== #

# --- Simulation conditions: MUST match Hossain_DB_Gen_linux.py (dataset generation) --- #
ASEQ = [-20, 20, 0.5]
RE = 2.5e5
MACH = 0.09281
NACA_KP_FILE = "naca_0012_M0-9281_RE250k_M0-093_RE250000.txt"  # AVL 3.32 section-local CDCL reference

# --- Inference settings --- #
NUM_INFERENCE_RUNS = 10
NUM_SAMPLES = 1_000_000        # uniform (global) CST samples per inference round
NUM_LOCAL_SAMPLES = 500_000    # local-refinement samples per round, drawn around the best dataset designs (0 = off)
LOCAL_TOP_K = 20               # number of distinct top-E dataset designs used as local search centers
LOCAL_SIGMA_FRACTION = 0.10    # local noise std dev, as a fraction of each coefficient's range width
PREDICT_CHUNK = 250_000        # rows per model call; keeps GPU/CPU memory bounded for large candidate sets
VALIDITY_THRESHOLD = 0.99
SEED_POOL_SIZE = 10000  # per-round seeds are drawn (without repetition) from [0, SEED_POOL_SIZE)
MAX_MASTER_SEED = 0xFFFFFFFF  # upper limit of the master seed (8 hex digits); raise it here if ever needed

# --- Candidate search window (CST deltas) --- #
# Defaults replicate the dataset 17 generation ranges (Hossain_DB_Gen_linux.py). This is an
# exploration window: narrow it manually once a promising direction is found.
# Format: (start_value, end_value, step); only start/end are used for continuous sampling.
RANGES_UPPER = {
    'LE': (0, 0.125, 0.0625),
    'inter': (0, 0.125, 1/24),
    'TE': (0, 0.125, 0.125),
}
RANGES_LOWER = {
    'LE': (-0.125, 0, 0.0625),
    'inter': (0, 0.125, 1/24),
    'TE': (0, 0.125, 0.125),
}

# --- Wing configuration modes --- #
HOSSAIN_MODE = "Hossain_config"
SINGLE_MODE = "single_configuration"
SINGLE_CONFIG_KEY = "__single__"  # internal key used when the dataset has no one-hot columns

# Registry mapping each Hossain wing-configuration identifier to its .geo creator class.
# Extend this list if new wing configurations are added to the dataset.
HOSSAIN_CONFIG_REGISTRY = [
    (HossainWingRec_Creation, "rec"),
    (HossainWingFlat_Creation, "flat"),
    (HossainWing60DEG_Creation, "60DEG"),
]

# Regressor type per model folder prefix (folders: mlp_regressor, mlp_regressor_station, ...)
REGRESSOR_FOLDER_TYPES = {'mlp_regressor': 'mlp', 'lgbm_regressor': 'lgbm', 'sknn_regressor': 'sknn'}

# --- User-facing messages for unavailable / incompatible functionality --- #
SKNN_UNAVAILABLE_MESSAGE = (
    "SKNN functionality is currently unavailable.\n"
    "   The SKNN module is undergoing 'improvements'. I use that word generously.\n"
    "   This model has been set aside for the rest of the run. Don't worry, it won't notice."
)
SINGLE_UNAVAILABLE_MESSAGE = (
    "single_configuration functionality is currently unavailable.\n"
    "   Its wing geometry creator was never updated, and I would hate to let it near a wing.\n"
    "   Please use the default Hossain_config mode. It is, after all, the only one that works."
)
MLP_INCOMPATIBLE_MESSAGE = (
    "The selected model is not compatible with this calculation scheme.\n"
    "   Please verify that the model has its corresponding JSON parameters file stored in the same directory as itself.\n"
    "   If your model doesn't have a JSON file like this one, find its parameters and retrain with the updated scheme.\n"
    "   I'm sure retraining will be a delightful use of your time. This model has been excluded from the run."
)


# =============================================================================================== #
# === CLI and paths === #
# =============================================================================================== #

def parse_master_seed(raw: str, decimal: bool) -> int:
    """
    Parses the master seed. Hexadecimal is the default format (optional '0x' prefix); decimal needs --decimal.
    NOTE: a digits-only string (e.g. '1234') is also valid hexadecimal, so without --decimal it is read as hex.
    """
    text = raw.strip().lower()
    if decimal:
        if not re.fullmatch(r'\d+', text):
            raise ValueError(f"'{raw}' is not a valid decimal number. --decimal expects digits only.")
        value = int(text)
    else:
        digits = text[2:] if text.startswith('0x') else text
        if not re.fullmatch(r'[0-9a-f]+', digits):
            raise ValueError(f"'{raw}' is not a valid hexadecimal seed (0-9, a-f). "
                             f"If this is a decimal value, add the --decimal argument.")
        value = int(digits, 16)
    if not 0 <= value <= MAX_MASTER_SEED:
        raise ValueError(f"Seed {raw} is out of range. Accepted values: 0 to {MAX_MASTER_SEED:x} (hex) / "
                         f"0 to {MAX_MASTER_SEED} (decimal).")
    return value


def parse_args():
    parser = argparse.ArgumentParser(description="Inference and simulation-based validation of trained AI models.")
    parser.add_argument('--dataset', type=int, default=17, help="Dataset number (default: 17).")
    parser.add_argument('--wing-mode', choices=[HOSSAIN_MODE, SINGLE_MODE], default=HOSSAIN_MODE,
                        help=f"'{HOSSAIN_MODE}': one-hot rec/flat/60DEG configurations (default). "
                             f"'{SINGLE_MODE}': dataset without one-hot columns.")
    parser.add_argument('--classifier', required=True, help="Name of the classifier (folder 'classifier').")
    parser.add_argument('--models', nargs='+', required=True,
                        help="Names of the regressors to run, e.g. --models Murata Buer. All run sequentially.")
    parser.add_argument('--seed', type=str, default=None,
                        help=f"Optional master seed, hexadecimal by default (0 to {MAX_MASTER_SEED:x}). The 10 per-round "
                             f"seeds derive from it; reuse it to get identical candidate clouds across invocations.")
    parser.add_argument('--decimal', action='store_true',
                        help="Interpret --seed as a decimal number instead of hexadecimal.")
    args = parser.parse_args()
    if len(set(args.models)) != len(args.models):
        parser.error("Duplicate names in --models.")
    if args.decimal and args.seed is None:
        parser.error("--decimal only makes sense together with --seed.")
    if args.seed is not None:
        try:
            args.seed = parse_master_seed(args.seed, args.decimal)
        except ValueError as e:
            parser.error(str(e))
    return args


def build_paths(dataset_num: int) -> SimpleNamespace:
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    sys.path.insert(0, str(project_root))

    binaries_dir = project_root / "bin"
    exe_ext = ".exe" if sys.platform == "win32" else ""
    support_files_dir = project_root / "data" / "support"
    models_dir = project_root / "models" / f"dataset_{dataset_num}"

    return SimpleNamespace(
        project_root=project_root,
        training_file=project_root / "data" / "datasets" / f"dataset_{dataset_num}" / f"regressor_dataset_{dataset_num}.csv",
        support_files_dir=support_files_dir,
        case_file=support_files_dir / "Hossain_sqwing_case.run",
        base_airfoil=support_files_dir / "naca0012_unit.csv",
        naca_kp_file=support_files_dir / "naca_kp" / NACA_KP_FILE,
        xfoil_bin=binaries_dir / f"xfoil{exe_ext}",
        avl_bin=binaries_dir / f"avl{exe_ext}",
        models_dir=models_dir,
        classifier_dir=models_dir / "classifier",
        results_root=project_root / "results" / f"dataset_{dataset_num}",
    )


def get_workspace() -> Path:
    """Short, fixed temp path (XFOIL/AVL do not tolerate long paths). Wiped at start and end of the run."""
    root = Path("C:/inference_temp") if platform.system() == "Windows" else Path.home() / ".inference_temp"
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    return root


def make_run_dir(results_root: Path, master_seed: int) -> Path:
    """results/dataset_N/<YYYYMMDD>_m_seed<master seed in hex>; a numeric suffix avoids overwriting."""
    base = f"{datetime.now().strftime('%Y%m%d')}_m_seed{master_seed:x}"
    run_dir, n = results_root / base, 1
    while run_dir.exists():
        n += 1
        run_dir = results_root / f"{base}_{n}"
    run_dir.mkdir(parents=True)
    return run_dir


# =============================================================================================== #
# === Dataset schema === #
# =============================================================================================== #

def parse_dataset_columns(csv_path: Path, wing_mode: str):
    """
    Reads only the dataset header and classifies every feature column by type.
    The last CSV column is always the regression target and is excluded.

    Returns
    -------
    feature_columns : list[str]  -- all feature columns, in CSV order
    cst_l_cols      : list[str]  -- 'cst_l_X' columns, sorted by X
    cst_u_cols      : list[str]  -- 'cst_u_X' columns, sorted by X
    config_cols     : list[str]  -- remaining columns (one-hot wing configuration flags; empty in single mode)
    """
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    feature_columns = header[:-1]
    cst_l_cols = sorted((c for c in feature_columns if re.fullmatch(r'cst_l_\d+', c)),
                        key=lambda c: int(c.rsplit('_', 1)[1]))
    cst_u_cols = sorted((c for c in feature_columns if re.fullmatch(r'cst_u_\d+', c)),
                        key=lambda c: int(c.rsplit('_', 1)[1]))
    if not cst_l_cols or len(cst_l_cols) != len(cst_u_cols):
        raise ValueError(f"Invalid CST columns in '{csv_path.name}': {len(cst_l_cols)} lower vs {len(cst_u_cols)} upper.")
    config_cols = [c for c in feature_columns if c not in cst_l_cols and c not in cst_u_cols]

    if wing_mode == HOSSAIN_MODE and not config_cols:
        raise ValueError(f"Wing mode '{HOSSAIN_MODE}' needs hot-encoded configuration columns, none found in "
                         f"'{csv_path.name}'. Use --wing-mode {SINGLE_MODE} for datasets without them.")
    if wing_mode == SINGLE_MODE and config_cols:
        raise ValueError(f"Wing mode '{SINGLE_MODE}' expects CST columns only, but '{csv_path.name}' also has "
                         f"{config_cols}. Use --wing-mode {HOSSAIN_MODE}.")
    return feature_columns, cst_l_cols, cst_u_cols, config_cols


def check_wing_mode_available(wing_mode: str):
    """Stops the run early when the selected wing mode is not functional yet."""
    if wing_mode == SINGLE_MODE:
        # ======================= PLACEHOLDER: single_configuration =======================
        # RecWingGeo_Creation has not been updated to the current geo-creator interface
        # (it takes different arguments). Once it is, delete this block; the code that
        # builds the registry for this mode already exists in build_config_registry().
        print(SINGLE_UNAVAILABLE_MESSAGE)
        sys.exit(1)
        # =================================================================================


def build_config_registry(wing_mode: str, config_cols: list) -> dict:
    """Maps each config key to (creator_class, identifier). One-hot columns are matched by substring."""
    if wing_mode == SINGLE_MODE:
        # UNVERIFIED: written for the future single_configuration mode (see check_wing_mode_available).
        # RecWingGeo_Creation's constructor arguments must be checked/adapted when it is updated.
        from simulation_support import RecWingGeo_Creation  # imported lazily: only needed in this mode
        return {SINGLE_CONFIG_KEY: (RecWingGeo_Creation, "single")}

    mapping = {}
    for col in config_cols:
        matches = [entry for entry in HOSSAIN_CONFIG_REGISTRY if entry[1].lower() in col.lower()]
        if len(matches) != 1:
            raise ValueError(
                f"Could not uniquely match dataset column '{col}' to a wing configuration in "
                f"HOSSAIN_CONFIG_REGISTRY. Matches found: {matches}. Update the registry to resolve this.")
        mapping[col] = matches[0]
    return mapping


# =============================================================================================== #
# === Candidate generation === #
# =============================================================================================== #

def build_feature_ranges(n_cst, ranges_upper, ranges_lower):
    """Per-coefficient (low, high, step) ranges, lower block first, then upper block."""
    return (
        [ranges_lower["LE"]] + [ranges_lower["inter"]] * (n_cst - 2) + [ranges_lower["TE"]] +
        [ranges_upper["LE"]] + [ranges_upper["inter"]] * (n_cst - 2) + [ranges_upper["TE"]]
    )


def generate_cst_candidates(num_samples, n_cst, ranges_upper, ranges_lower, rng):
    """Uniform random CST-only candidates (lower block, then upper block) from the given RNG.
    Sampling is continuous: the 'step' of each range is not used (it only describes the dataset grid)."""
    feature_ranges = build_feature_ranges(n_cst, ranges_upper, ranges_lower)
    candidates = np.zeros((num_samples, n_cst + n_cst), dtype=np.float32)
    for i, (low, high, _step) in enumerate(feature_ranges):
        candidates[:, i] = rng.uniform(low, high, size=num_samples)
    return candidates


def generate_local_candidates(parents, num_samples, n_cst, ranges_upper, ranges_lower, sigma_fraction, rng):
    """
    Local refinement: Gaussian perturbations around randomly chosen parent designs (e.g. the best dataset
    designs). The noise std dev is sigma_fraction of each coefficient's range width and the result is clipped
    to the search window, so candidates near a window edge stay on that edge instead of leaving the window.
    """
    feature_ranges = build_feature_ranges(n_cst, ranges_upper, ranges_lower)
    lows = np.array([min(r[0], r[1]) for r in feature_ranges], dtype=np.float32)
    highs = np.array([max(r[0], r[1]) for r in feature_ranges], dtype=np.float32)
    centers = parents[rng.integers(0, len(parents), size=num_samples)]
    noise = rng.normal(0.0, 1.0, size=centers.shape).astype(np.float32) * (highs - lows) * sigma_fraction
    return np.clip(centers + noise, lows, highs).astype(np.float32)


def predict_chunked(predict_fn, X, chunk_size=PREDICT_CHUNK):
    """Runs predict_fn on X in row chunks and returns one flat array (bounds GPU/CPU memory use)."""
    parts = [np.asarray(predict_fn(X[i:i + chunk_size])).reshape(-1) for i in range(0, len(X), chunk_size)]
    return np.concatenate(parts)


def expand_with_configs(cst_candidates, feature_columns, cst_l_cols, cst_u_cols, config_cols):
    """
    Pairs every CST combination with every hot-encoded wing configuration, producing one full
    feature row (in dataset column order) per (cst combination, config) pair -- e.g. 10 CST
    combinations x 3 configurations => 30 rows. Without configuration columns (single mode) the
    CST rows are only arranged in dataset column order.

    Returns
    -------
    full_candidates : np.ndarray, shape (num_samples * num_configs, num_features)
    config_labels   : np.ndarray of str, the config key active in each row
    """
    num_samples = cst_candidates.shape[0]
    num_configs = max(len(config_cols), 1)
    cst_col_order = cst_l_cols + cst_u_cols

    # Row r corresponds to (sample r // num_configs, config r % num_configs) in both blocks below.
    cst_repeated = np.repeat(cst_candidates, num_configs, axis=0)
    if config_cols:
        one_hot_block = np.tile(np.eye(num_configs, dtype=np.float32), (num_samples, 1))
        config_labels = np.tile(np.array(config_cols), num_samples)
    else:
        one_hot_block = None
        config_labels = np.full(num_samples, SINGLE_CONFIG_KEY)

    cst_pos = {col: i for i, col in enumerate(cst_col_order)}
    full_candidates = np.zeros((num_samples * num_configs, len(feature_columns)), dtype=np.float32)
    for f_idx, col in enumerate(feature_columns):
        if col in cst_pos:
            full_candidates[:, f_idx] = cst_repeated[:, cst_pos[col]]
        elif config_cols and col in config_cols:
            full_candidates[:, f_idx] = one_hot_block[:, config_cols.index(col)]
    return full_candidates, config_labels


def load_dataset_summary(csv_path: Path, cst_l_cols, cst_u_cols, config_cols, top_k: int):
    """
    Reads the training dataset once. Returns (best, parents):
      best    -- dict with the highest-E row (E, wing_config, cst_lower, cst_upper), or None if unavailable
      parents -- array (<= top_k, 2 * n_cst) with the distinct CST vectors of the highest-E rows
                 (lower block, then upper block): centers of the local refinement. None if unavailable.
    The dataset best is information only: it is neither ranked nor simulated.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        return None, None
    ordered = df.sort_values(df.columns[-1], ascending=False)
    best_row = ordered.iloc[0]
    active = [c for c in config_cols if best_row[c] == 1]
    config = active[0] if active else ("single configuration" if not config_cols else "unknown")
    best = {
        "E": float(best_row.iloc[-1]),
        "wing_config": config,
        "cst_lower": [round(float(best_row[c]), 6) for c in cst_l_cols],
        "cst_upper": [round(float(best_row[c]), 6) for c in cst_u_cols],
    }
    parents = ordered[cst_l_cols + cst_u_cols].drop_duplicates().head(top_k).to_numpy(dtype=np.float32)
    return best, parents


def print_dataset_best(best):
    """Information line: best E found in the training dataset."""
    if best is None:
        print("\n[INFO] Dataset best: training dataset not found.")
        return
    print(f"\n[INFO] Dataset best -> E = {best['E']:.4f} | config = {best['wing_config']}")
    print(f"       CST lower: {best['cst_lower']}")
    print(f"       CST upper: {best['cst_upper']}")


# =============================================================================================== #
# === Model discovery and loading === #
# =============================================================================================== #

def infer_hidden_layers(pth_path: Path) -> list:
    """
    Reads the hidden layer sizes from the saved state dict (the model metadata does not store them).
    Every 2-D '.weight' tensor is a Linear layer; the last one is the single-output layer.
    """
    state = torch.load(pth_path, map_location='cpu', weights_only=True)
    linears = [v for k, v in state.items() if k.endswith('.weight') and v.ndim == 2]
    if len(linears) < 2 or linears[-1].shape[0] != 1:
        raise ValueError(f"Unexpected network layout in '{pth_path.name}'; cannot infer hidden layers.")
    return [int(w.shape[0]) for w in linears[:-1]]


class ModelIncompatibleError(Exception):
    """The model was saved without the files this calculation scheme needs (it is skipped, not fatal)."""


def read_mlp_parameters(folder: Path, name: str, hidden_layers: list) -> dict:
    """
    Reads the tuning script's '{name}_regressor_model_parameters.json'. 'log_trans' is not stored in
    the model metadata and MLP_Regressor.load_model does not restore it, so it must come from here.
    """
    path = folder / f"{name}_regressor_model_parameters.json"
    if not path.exists():
        raise ModelIncompatibleError(f"MLP parameters file not found: {path}")
    with open(path, 'r') as f:
        params = json.load(f)
    if 'log_trans' not in params:
        raise ModelIncompatibleError(f"'log_trans' missing in {path.name}")
    if 'n_layers' in params:  # consistency check between the weights and the tuning parameters
        expected = [int(params[f'n_units_l{i}']) for i in range(int(params['n_layers']))]
        if expected != hidden_layers:
            raise ValueError(f"Layer mismatch for '{name}': weights {hidden_layers} vs parameters file {expected}.")
    return params


def list_regressor_names(models_dir: Path) -> list:
    names = []
    for sub in sorted(p for p in models_dir.iterdir() if p.is_dir()):
        if any(sub.name.startswith(prefix) for prefix in REGRESSOR_FOLDER_TYPES):
            names += [f.name[:-len('_meta.json')] for f in sub.glob('*_meta.json')]
    return sorted(names)


def resolve_regressor(name: str, models_dir: Path):
    """Finds the folder holding regressor 'name' and returns (folder, type)."""
    if not models_dir.is_dir():
        raise FileNotFoundError(f"Models folder does not exist: {models_dir}")
    matches = []
    for sub in sorted(p for p in models_dir.iterdir() if p.is_dir()):
        for prefix, mtype in REGRESSOR_FOLDER_TYPES.items():
            if not sub.name.startswith(prefix):
                continue
            # SKNN is detected by folder, whatever the file naming scheme is; the others need their metadata file.
            owns_name = any(sub.glob(f"{name}_*")) if mtype == 'sknn' else (sub / f"{name}_meta.json").exists()
            if owns_name:
                matches.append((sub, mtype))
    if not matches:
        raise FileNotFoundError(f"No regressor named '{name}' under '{models_dir}'. "
                                f"Available: {list_regressor_names(models_dir)}")
    if len(matches) > 1:
        raise ValueError(f"Regressor name '{name}' is ambiguous, found in: {[str(m[0]) for m in matches]}")
    return matches[0]


def load_regressor(name: str, folder: Path, mtype: str):
    """Returns the loaded regressor, or None when the model type is not available yet (SKNN)."""
    if mtype == 'sknn':
        # ============================ PLACEHOLDER: SKNN ============================
        # Replace the next two lines with the real loader once the SKNN module is updated, e.g.:
        #     return SKNN_Regressor.load_model(folder, name)
        # and import SKNN_Regressor from ai_support. Then the model loading loop no longer needs to skip it.
        print(f"[{name}] {SKNN_UNAVAILABLE_MESSAGE}")
        return None
        # ===========================================================================
    if mtype == 'mlp':
        hidden = infer_hidden_layers(folder / f"{name}_model.pth")
        params = read_mlp_parameters(folder, name, hidden)
        return MLP_Regressor.load_model(folder, name, hidden_layer_sizes=hidden,
                                        dropout_rate=float(params.get('dropout_rate', 0.0)),
                                        log_trans=bool(params['log_trans']))
    if mtype == 'lgbm':
        return LGBM_Regressor.load_model(folder, name)
    raise ValueError(f"Unsupported model type '{mtype}'.")


def load_classifier(name: str, folder: Path):
    pth = folder / f"{name}_model.pth"
    if not pth.exists():
        raise FileNotFoundError(f"Classifier '{name}' not found: {pth}")
    return Classifier.load_model(folder, name, hidden_layer_sizes=infer_hidden_layers(pth))


# =============================================================================================== #
# === Candidate pooling === #
# =============================================================================================== #

def dedupe_candidates(ranked_by_model: dict) -> list:
    """
    Merges identical feature vectors (configuration included) across models and rounds.
    Each unique candidate keeps a 'sources' list with every (model, rank, round, seed, predicted E)
    that selected it. Order of first appearance is deterministic (CLI model order, then rank).
    """
    unique = {}
    for items in ranked_by_model.values():
        for item in items:
            key = item['features'].tobytes()
            if key not in unique:
                unique[key] = {
                    'candidate_id': f"candidate_{len(unique) + 1:03d}",
                    'features': item['features'],
                    'config_col': item['config_col'],
                    'sources': [],
                }
            unique[key]['sources'].append({
                'model': item['model'], 'rank': item['rank'], 'round': item['round'],
                'round_seed': item['seed'], 'origin': item['origin'], 'predicted_E': item['E'],
            })
    return list(unique.values())


# =============================================================================================== #
# === Simulation result parsing and reporting === #
# =============================================================================================== #

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


def compute_simulated_E(cl, cd):
    """CL/CD, or NaN for divergence (missing data, CD = 0, non-finite) and for E = 0 (no relative error possible)."""
    if cd == 0 or not (math.isfinite(cl) and math.isfinite(cd)):
        return np.nan
    e = cl / cd
    if e == 0 or not math.isfinite(e):
        return np.nan
    return e


def save_results_to_txt(rows: list, output_folder: Path, prefix: str, dataset_num: int, dataset_best=None):
    """Saves formatted text report handling NaN values properly."""
    filepath = output_folder / f"{prefix}_final_report.txt"
    sorted_rows = sorted(rows, key=lambda x: -np.inf if math.isnan(x['simulated_E']) else x['simulated_E'], reverse=True)
    width = 128
    with open(filepath, 'w') as f:
        f.write("=" * width + "\n")
        f.write(f"Final Aerodynamic Efficiency (E) Report for Dataset {dataset_num}\n")
        f.write("=" * width + "\n")
        if dataset_best is not None:
            f.write(f"Dataset best (reference only, not simulated): E = {dataset_best['E']:.4f} | "
                    f"config = {dataset_best['wing_config']}\n")
            f.write(f"  CST lower: {dataset_best['cst_lower']}\n")
            f.write(f"  CST upper: {dataset_best['cst_upper']}\n")
        else:
            f.write("Dataset best: not available (training dataset not found)\n")
        f.write("-" * width + "\n")
        f.write(f"{'Rank':<5} {'Model':<16} {'Model Rank':<11} {'Candidate':<15} {'Wing Config':<12} {'Origin':<7} "
                f"{'Simulated E':<13} {'Predicted E':<13} {'Relative Error (%)':<20}\n")
        f.write("-" * width + "\n")
        for i, item in enumerate(sorted_rows):
            sim = f"{item['simulated_E']:.4f}" if not math.isnan(item['simulated_E']) else "NaN"
            pred = f"{item['predicted_E']:.4f}"
            err = f"{item['relative_error']:.2f}" if not math.isnan(item['relative_error']) else "NaN"
            f.write(f"{i + 1:<5} {item['model']:<16} {item['model_rank']:<11} {item['candidate_id']:<15} "
                    f"{item['wing_config']:<12} {item['origin']:<7} {sim:<13} {pred:<13} {err:<20}\n")
        f.write("=" * width + "\n")
    print(f"Successfully saved text report to '{filepath.name}'")


def write_json(path: Path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


# =============================================================================================== #
# === Main pipeline === #
# =============================================================================================== #

if __name__ == "__main__":
    args = parse_args()
    check_wing_mode_available(args.wing_mode)

    # === Mandatory GPU Check (classifier and MLP regressors are torch models) === #
    if not torch.cuda.is_available():
        print("ERROR: Execution stopped. This script requires a CUDA-enabled GPU.")
        sys.exit(1)
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")

    dataset_num = args.dataset
    paths = build_paths(dataset_num)
    dataset_name_prefix = f"AI_Test_Results_dataset_{dataset_num}"
    start_time = datetime.now()
    print(f"Pipeline started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # === Seeds: one per inference round, identical for every model === #
    # NumPy's Generator (np.random.default_rng) accepts any non-negative integer, so the accepted range
    # is simply MAX_MASTER_SEED. The 10 round seeds are a deterministic function of the master seed.
    if args.seed is not None:
        master_seed = args.seed
        print(f"Master seed supplied as {'decimal' if args.decimal else 'hexadecimal'}: "
              f"0x{master_seed:x} = {master_seed} (decimal)")
    else:
        master_seed = secrets.randbelow(MAX_MASTER_SEED + 1)
        print(f"Master seed generated: 0x{master_seed:x} = {master_seed} (decimal)")
    round_seeds = np.random.default_rng(master_seed).choice(SEED_POOL_SIZE, size=NUM_INFERENCE_RUNS, replace=False).tolist()
    print(f"Round seeds (hex): {[f'{s:x}' for s in round_seeds]}")

    # === Dataset Schema Parsing === #
    print("\n# === Parsing Dataset Schema === #")
    feature_columns, cst_l_cols, cst_u_cols, config_cols = parse_dataset_columns(paths.training_file, args.wing_mode)
    n_cst = len(cst_l_cols)
    config_registry = build_config_registry(args.wing_mode, config_cols)
    print(f"Wing mode '{args.wing_mode}': {n_cst} CST coefficients per surface, configuration(s): "
          f"{[v[1] for v in config_registry.values()]}")
    if not paths.naca_kp_file.exists():
        raise FileNotFoundError(f"NACA 0012 key point support file not found: {paths.naca_kp_file}")

    # === Model resolution (cheap, fails early) === #
    resolved = {name: resolve_regressor(name, paths.models_dir) for name in args.models}

    # === Model Initialization === #
    print("\n# === Loading AI Models === #")
    classifier = load_classifier(args.classifier, paths.classifier_dir)
    cst_col_idx = [feature_columns.index(c) for c in cst_l_cols + cst_u_cols]
    if classifier.input_dim != len(cst_col_idx):
        raise ValueError(f"Classifier input_dim ({classifier.input_dim}) does not match the "
                         f"{len(cst_col_idx)} CST feature columns found in '{paths.training_file.name}'.")

    regressors, skipped_models = {}, []
    for name, (folder, mtype) in resolved.items():
        try:
            model = load_regressor(name, folder, mtype)
        except ModelIncompatibleError as e:
            print(f"[{name}] {MLP_INCOMPATIBLE_MESSAGE}")
            skipped_models.append({"name": name, "type": mtype, "reason": str(e)})
            continue
        if model is None:  # type not available yet (SKNN); the message was printed by load_regressor
            skipped_models.append({"name": name, "type": mtype, "reason": "functionality currently unavailable"})
            continue
        n_input = getattr(getattr(model, 'feature_engineer', None), 'n_input', None)
        if n_input is not None and n_input != len(feature_columns):
            raise ValueError(f"Model '{name}' expects {n_input} raw input columns but the dataset has "
                             f"{len(feature_columns)} (check --wing-mode).")
        regressors[name] = model

    if not regressors:
        print("\nNo usable regressors remain. There is nothing left to calculate. Congratulations.")
        sys.exit(1)

    # === Run folder === #
    paths.results_root.mkdir(parents=True, exist_ok=True)
    run_dir = make_run_dir(paths.results_root, master_seed)
    print(f"Results folder: {run_dir}")

    # === Dataset summary: best row (information) and local-refinement centers === #
    dataset_best, local_parents = load_dataset_summary(paths.training_file, cst_l_cols, cst_u_cols, config_cols, LOCAL_TOP_K)
    use_local = local_parents is not None and NUM_LOCAL_SAMPLES > 0

    # === Metadata (written now, completed at the end of the run) === #
    metadata = {
        "run_status": "running",
        "started": start_time.strftime('%Y-%m-%d %H:%M:%S'),
        "dataset": dataset_num,
        "wing_mode": args.wing_mode,
        "wing_configurations": [v[1] for v in config_registry.values()],
        "master_seed": {"hex": f"{master_seed:x}", "decimal": master_seed},
        "round_seeds_hex": [f"{s:x}" for s in round_seeds],
        "classifier": {"name": args.classifier, "folder": str(paths.classifier_dir)},
        "models": [{"name": n, "type": resolved[n][1], "folder": str(resolved[n][0]),
                    "feature_mode": getattr(regressors[n], 'feature_mode', None)} for n in regressors],
        "skipped_models": skipped_models,
        "inference": {"num_runs": NUM_INFERENCE_RUNS, "num_samples": NUM_SAMPLES, "validity_threshold": VALIDITY_THRESHOLD,
                      "num_local_samples": NUM_LOCAL_SAMPLES if use_local else 0,
                      "local_centers": len(local_parents) if use_local else 0,
                      "local_sigma_fraction": LOCAL_SIGMA_FRACTION, "predict_chunk": PREDICT_CHUNK},
        "ranges_upper": RANGES_UPPER,
        "ranges_lower": RANGES_LOWER,
        "simulation": {"ASEQ": ASEQ, "Re": RE, "Mach": MACH, "naca_key_points_file": NACA_KP_FILE},
    }
    metadata_path = run_dir / f"{dataset_name_prefix}_run_metadata.json"
    write_json(metadata_path, metadata)

    print_dataset_best(dataset_best)
    if use_local:
        print(f"Local refinement: {NUM_LOCAL_SAMPLES} samples per round around {len(local_parents)} top dataset designs "
              f"(sigma = {LOCAL_SIGMA_FRACTION} x range width).")
    else:
        print("Local refinement disabled (no dataset available or NUM_LOCAL_SAMPLES = 0).")

    # === Inference & Ranking (all models, sequentially) === #
    raw_best = {name: [] for name in regressors}
    for r, seed in enumerate(round_seeds, start=1):
        print(f"\n--- Inference Run {r}/{NUM_INFERENCE_RUNS} (seed {seed:x}) ---")
        rng = np.random.default_rng(seed)
        cst_candidates = generate_cst_candidates(NUM_SAMPLES, n_cst, RANGES_UPPER, RANGES_LOWER, rng)
        n_global = len(cst_candidates)  # rows at or beyond this index come from the local refinement
        if use_local:
            local_candidates = generate_local_candidates(local_parents, NUM_LOCAL_SAMPLES, n_cst, RANGES_UPPER,
                                                         RANGES_LOWER, LOCAL_SIGMA_FRACTION, rng)
            cst_candidates = np.concatenate([cst_candidates, local_candidates])

        # Classifier works on CST combinations; invalid ones are discarded.
        validity_probs = predict_chunked(classifier.predict, cst_candidates)
        valid_indices = np.where(validity_probs > VALIDITY_THRESHOLD)[0]
        if len(valid_indices) == 0:
            print("No valid candidates found in this run. Skipping.")
            continue
        print(f"{len(valid_indices)} valid CST combinations out of {len(cst_candidates)} "
              f"({int((valid_indices >= n_global).sum())} from local refinement).")
        full_candidates, config_labels = expand_with_configs(
            cst_candidates[valid_indices], feature_columns, cst_l_cols, cst_u_cols, config_cols)
        num_configs = max(len(config_cols), 1)  # row i of full_candidates belongs to valid sample i // num_configs

        for name, model in regressors.items():
            t0 = time.perf_counter()
            preds = predict_chunked(model.predict, full_candidates).astype(np.float64)
            preds[~np.isfinite(preds)] = np.nan
            if np.isnan(preds).all():
                print(f"[WARNING] '{name}' returned only NaN/inf predictions in run {r}. Skipping its result.")
                continue
            best_idx = int(np.nanargmax(preds))
            origin = 'local' if valid_indices[best_idx // num_configs] >= n_global else 'global'
            print(f"  {name}: best predicted E = {preds[best_idx]:.4f} [{origin}] ({time.perf_counter() - t0:.1f}s)")
            raw_best[name].append({
                'model': name, 'round': r, 'seed': f"{seed:x}", 'E': float(preds[best_idx]), 'origin': origin,
                'features': full_candidates[best_idx].copy(), 'config_col': str(config_labels[best_idx]),
            })

    # Rank each model's best-of-run results
    for name, items in raw_best.items():
        items.sort(key=lambda x: x['E'], reverse=True)
        for rank, item in enumerate(items, start=1):
            item['rank'] = rank

    # === Pool, compare and remove repeated feature vectors === #
    candidates = dedupe_candidates(raw_best)
    total_entries = sum(len(v) for v in raw_best.values())
    print(f"\nPooled {total_entries} model results -> {len(candidates)} unique candidates "
          f"({total_entries - len(candidates)} repeated feature vectors merged).")
    if not candidates:
        print("\nWARNING: No candidates to validate. Nothing to simulate.")
        metadata.update({"run_status": "completed_no_candidates", "finished": datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
        write_json(metadata_path, metadata)
        sys.exit(0)

    predictions_export_path = run_dir / f"{dataset_name_prefix}_predictions_only.json"
    write_json(predictions_export_path, {
        c['candidate_id']: {
            "cst_coeffs": c['features'][cst_col_idx].tolist(),
            "wing_config": config_registry[c['config_col']][1],
            "sources": c['sources'],
        } for c in candidates
    })
    print(f"=> Initial predictions saved to {predictions_export_path.name}")

    # === Geometry Instantiation & Aerodynamic Simulation (sequential) === #
    print("\n# === Beginning Data Validation === #")
    ANGLES = [ASEQ[0] + j * ASEQ[2] for j in range(int((ASEQ[1] - ASEQ[0]) / ASEQ[2]) + 1)]
    rar_Alma = compression_Alma()
    temp_root = get_workspace()
    sim_results = {}

    try:
        airfoils_temp = temp_root / "airfoils"
        polars_temp = temp_root / "polars"
        geo_temp = temp_root / "geometry"
        forces_temp = temp_root / "forces"
        support_temp = temp_root / "support"
        for d in [airfoils_temp, polars_temp, geo_temp, forces_temp, support_temp]:
            d.mkdir(parents=True, exist_ok=True)

        # Generate Airfoils
        print("\n--- Generating Airfoil Geometry ---")
        airfoil_creator = Airfoil_Generator(base_airfoil_csv=paths.base_airfoil, n_cst=n_cst, verbose=False)
        for c in tqdm(candidates, desc="Creating .dat files"):
            x_mod, yu_mod, yl_mod = airfoil_creator.generate(cst_coeffs=c['features'][cst_col_idx])
            airfoil_creator._create_dat_file(output_dir=airfoils_temp, n_cst=n_cst, x=x_mod, yl=yl_mod, yu=yu_mod,
                                             file_name=c['candidate_id'])

        # Run XFOIL (same conditions as the dataset generation)
        print("\n--- Generating Polars (XFOIL) ---")
        airfoil_paths = get_all_files_from(airfoils_temp, extension='dat')
        xfoil_case = XPR(xfoil_exe_path=str(paths.xfoil_bin), main_dir=temp_root, airfoil_dir=airfoils_temp,
                         polar_dir=polars_temp, verbose=False)
        for path in tqdm(airfoil_paths, desc="Running XFOIL"):
            polar_path = polars_temp / f"polar_{Path(path).stem}.txt"
            # Does not require try/except due to internal handler
            xfoil_case.run_simulation(airfoil_file_path=path, polar_file_path=str(polar_path),
                                      alfa_sequence=ANGLES, visc=RE, mach=MACH)

        # Create Wing Geometries, each candidate uses the creator matching its own configuration
        print("\n--- Generating Wing Geometry (.avl files) ---")
        polar_paths = get_all_files_from(polars_temp, extension='txt')
        key_points = Key_Point_Extraction(polar_file_paths=polar_paths, verbose=True)
        key_points_data = key_points.process_files()
        key_points.save_error_log(os.path.join(geo_temp, 'KP_error_log.txt'))

        airfoil_paths_by_config = defaultdict(list)
        for c in candidates:
            airfoil_paths_by_config[c['config_col']].append(str(airfoils_temp / f"{c['candidate_id']}.dat"))

        for config_col, subset_paths in airfoil_paths_by_config.items():
            creator_class, identifier = config_registry[config_col]
            print(f"  -> {len(subset_paths)} airfoil(s) under '{identifier}' configuration")
            geo_creator = creator_class(
                key_points_data=key_points_data,
                airfoil_abs_paths=subset_paths,
                naca_key_points_data=str(paths.naca_kp_file),
                output_dir=str(geo_temp),
                verbose=False,
                pb=True
            )
            geo_creator.create_geo_files()
            geo_creator.save_error_log(os.path.join(geo_temp, f'geo_{identifier}_error_log.txt'))

        # Calculate Final Forces (AVL), sequentially
        print("\n--- Calculating Final Coefficients (AVL) ---")
        geo_paths = get_all_files_from(geo_temp, extension='avl')
        shutil.copy(paths.case_file, support_temp)
        case_file_temp = support_temp / paths.case_file.name
        for path in tqdm(geo_paths, desc="Running AVL sequentially"):
            avl_case = AVL_Run_Case(geo_files=[path], case_file=str(case_file_temp), avl_exe_path=str(paths.avl_bin),
                                    output_dir=forces_temp, debug=False)
            # Does not require try/except due to internal handler
            avl_case.run()

        print("\n--- Archiving Raw Data Files ---")
        rar_Alma.compress_folder(airfoils_temp, run_dir, f"{dataset_name_prefix}_airfoils")
        rar_Alma.compress_folder(polars_temp, run_dir, f"{dataset_name_prefix}_polars")
        rar_Alma.compress_folder(geo_temp, run_dir, f"{dataset_name_prefix}_geo")
        rar_Alma.compress_folder(forces_temp, run_dir, f"{dataset_name_prefix}_forces")

        # Parse simulation results
        print("\n--- Processing Final Aerodynamic Efficiency Validation ---")
        for fpath in tqdm(get_all_files_from(forces_temp, extension='txt'), desc="Reading force files"):
            airfoil_name, cl, cd = extract_force_data(Path(fpath))
            if airfoil_name is None:
                continue  # Skip unparseable files entirely
            sim_results[airfoil_name] = (cl, cd)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

    # === Final Validation & Reporting: one row per (model, rank) entry, sharing the candidate's simulation === #
    rows = []
    for c in candidates:
        cl, cd = sim_results.get(c['candidate_id'], (np.nan, np.nan))
        simulated_E = compute_simulated_E(cl, cd)
        for src in c['sources']:
            predicted_E = src['predicted_E']
            relative_error = abs((simulated_E - predicted_E) / simulated_E) * 100 if not math.isnan(simulated_E) else np.nan
            rows.append({
                "candidate_id": c['candidate_id'],
                "model": src['model'],
                "model_rank": src['rank'],
                "round": src['round'],
                "wing_config": config_registry[c['config_col']][1],
                "origin": src['origin'],
                "simulated_CL": cl,
                "simulated_CD": cd,
                "simulated_E": simulated_E,
                "predicted_E": predicted_E,
                "relative_error": relative_error,
            })

    json_report_path = run_dir / f"{dataset_name_prefix}_final_report.json"
    # Convert NaN to None for strict JSON compliance
    write_json(json_report_path, {
        "dataset_best": dataset_best,
        "results": [{k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()} for row in rows],
    })
    print(f"\n=> JSON final report saved to {json_report_path.name}")
    save_results_to_txt(rows, run_dir, dataset_name_prefix, dataset_num, dataset_best)

    end_time = datetime.now()
    metadata.update({
        "run_status": "completed",
        "finished": end_time.strftime('%Y-%m-%d %H:%M:%S'),
        "total_execution_time": str(end_time - start_time),
        "results": {"model_results": total_entries, "unique_candidates": len(candidates),
                    "simulated_candidates": len(sim_results)},
    })
    write_json(metadata_path, metadata)

    print(f"\nProcess completed successfully at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total execution time: {end_time - start_time}")
