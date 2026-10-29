"""
Inference + validation pipeline for the trained airfoil/wing AI models.

Usage:
    python inference_and_testing.py --classifier <name> --models <name1> [<name2> ...]
                                    [--dataset 17] [--wing-mode Hossain_config|single_configuration]
                                    [--seed HEX [--decimal]] [--csv PATH]

Flow:
    1. Candidate pool. For each seeded inference round: uniform CST samples, local samples around the best
       dataset designs and edge-probe samples just past the window edges where the best designs sit
       -> classifier validity filter -> pairing with every wing configuration -> EVERY regressor scores
       EVERY row. Only an elite subset of each round (best rows per model and by consensus) is kept.
    2. Region tagging. Each elite row gets its distance (in dataset grid steps) to the nearest dataset design
       of the same configuration, that design's E, and a label: on-grid / between / outside (the window).
    3. Shortlist. Diverse top-K per model, diverse top-K by consensus (lowest prediction across models),
       a few edge-probe designs, the other wing configurations of the best consensus shapes, and the top
       dataset designs as a reproducibility reference.
    4. Main simulation of the shortlist (XFOIL -> key points -> .avl geometry -> AVL).
    5. Robustness: small perturbations of the best simulated designs are simulated as well.
    6. Reports. result.txt: did any model find a design that beats the best dataset E (SUCCESS!! / Try again...),
       with the bare minimum around it. detailed_results.txt / .json: scorecard per selection group and per
       model, configuration check, candidate table and robustness table. Plus a parity plot.

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
from datetime import datetime, timedelta
from tqdm import tqdm
import torch
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from cst_modeling.foil import check_validity

# Import custom support modules (Ensure these are accessible in your project root)
from simulation_support import Xfoil_Polar_Creator as XPR
from simulation_support import Key_Point_Extraction, HossainWingRec_Creation, HossainWingFlat_Creation, HossainWing60DEG_Creation, AVL_Run_Case
from geometric_support import Airfoil_Generator
from file_handling_support import compression_Alma, get_all_files_from
from ai_support import Classifier, MLP_Regressor, LGBM_Regressor

# =============================================================================================== #
# === Constants === #
# =============================================================================================== #

# --- Simulation conditions: MUST match Hossain_DB_Gen.py (dataset generation) --- #
ASEQ = [-20, 20, 0.5]
RE = 2.5e5
MACH = 0.09281
NACA_KP_FILE = "naca_0012_M0-9281_RE250k_M0-093_RE250000.txt"  # AVL 3.32 section-local CDCL reference
SWEEP_MODE = "zero_out"   # alpha sweep, see simulation_support.alpha_sweep
KP_SEED = 42              # random_state of the key-point GPR fit

# --- Inference settings --- #
NUM_INFERENCE_RUNS = 10
NUM_SAMPLES = 600_000          # uniform (global) CST samples per inference round
NUM_LOCAL_SAMPLES = 800_000    # local-refinement samples per round, drawn around the best dataset designs (0 = off)
NUM_EDGE_SAMPLES = 400_000     # edge-probe samples per round (0 = off); only rows outside the window are kept
LOCAL_TOP_K = 30               # number of distinct top-E dataset designs used as local search centers
LOCAL_SIGMA_FRACTION = 0.10    # local noise std dev, as a fraction of each coefficient's range width
EDGE_SIGMA_STEPS = 0.5         # edge-probe noise std dev on the extended coefficients, in grid steps (reaches ~1 step past the edge)
PREDICT_CHUNK = 250_000        # rows per model call; keeps GPU/CPU memory bounded for large candidate sets
VALIDITY_THRESHOLD = 0.99
SEED_POOL_SIZE = 10000  # per-round seeds are drawn (without repetition) from [0, SEED_POOL_SIZE)
MAX_MASTER_SEED = 0xFFFFFFFF  # upper limit of the master seed (8 hex digits); raise it here if ever needed
ELITE_PER_CRITERION = 2000     # rows kept per round for each model's ranking and for the consensus ranking

# --- Shortlist --- #
TOP_K_PER_MODEL = 20           # diverse picks per model
TOP_K_CONSENSUS = 20           # diverse picks by consensus (lowest prediction across models)
TOP_K_EDGE = 12                # diverse edge-probe picks (outside the window), ranked by consensus
CONFIG_CHECK_SHAPES = 3        # best consensus shapes, also simulated in the configurations they were not picked with
REFERENCE_TOP = 5              # top dataset designs re-simulated as they are (reproducibility check)
DIVERSITY_MIN_STEPS = 0.5      # kept designs differ by at least this many grid steps in some CST coefficient
EDGE_MIN_SHARE = 0.5           # a window edge is probed if at least this share of the top dataset designs sit on it
HIT_FRACTION = 0.9             # a "hit" simulates at >= HIT_FRACTION x dataset best E

# --- Robustness --- #
ROBUST_DESIGNS = 10            # best simulated designs (distinct shapes) that get perturbed
ROBUST_PERTURBATIONS = 8       # perturbations per design
ROBUST_SIGMA_STEPS = 0.10      # perturbation std dev, in grid steps (0.05 was sized to the sequential-sweep artifact's E flips)
ROBUST_HOLD_FRACTION = 0.9     # a perturbation "holds" if it simulates at >= this fraction of its parent's E

# --- Candidate search window (CST deltas) --- #
# Defaults replicate the dataset 17 generation ranges (Hossain_DB_Gen.py). This is an
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

# Selection groups (a candidate can belong to several)
GROUP_CONSENSUS = "consensus"
GROUP_EDGE = "edge"
GROUP_CONFIG = "config_check"
GROUP_REFERENCE = "reference"

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
                        help=f"Optional master seed, hexadecimal by default (0 to {MAX_MASTER_SEED:x}). The round "
                             f"seeds derive from it; reuse it to get identical candidate clouds across invocations.")
    parser.add_argument('--decimal', action='store_true',
                        help="Interpret --seed as a decimal number instead of hexadecimal.")
    parser.add_argument('--csv', default=None,
                        help="Regressor dataset CSV that defines the dataset best, the local search centers and the "
                             "reference designs (default: data/datasets/dataset_<N>/regressor_dataset_<N>.csv). Use the "
                             "rectified CSV when the models were trained on it.")
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
    """Short, fixed temp path (XFOIL/AVL do not tolerate long paths). Wiped at start and end of each simulation stage."""
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


def window_bounds(n_cst, ranges_upper, ranges_lower):
    """(lows, highs) of the search window per CST coefficient (lower block, then upper block)."""
    feature_ranges = build_feature_ranges(n_cst, ranges_upper, ranges_lower)
    lows = np.array([min(r[0], r[1]) for r in feature_ranges], dtype=np.float32)
    highs = np.array([max(r[0], r[1]) for r in feature_ranges], dtype=np.float32)
    return lows, highs


def generate_cst_candidates(num_samples, n_cst, ranges_upper, ranges_lower, rng):
    """Uniform random CST-only candidates (lower block, then upper block) from the given RNG.
    Sampling is continuous: the 'step' of each range is not used (it only describes the dataset grid)."""
    feature_ranges = build_feature_ranges(n_cst, ranges_upper, ranges_lower)
    candidates = np.zeros((num_samples, n_cst + n_cst), dtype=np.float32)
    for i, (low, high, _step) in enumerate(feature_ranges):
        candidates[:, i] = rng.uniform(low, high, size=num_samples)
    return candidates


def generate_local_candidates(parents, num_samples, lows, highs, sigma_fraction, rng, clip_lows=None, clip_highs=None,
                              sigma=None):
    """
    Local refinement: Gaussian perturbations around randomly chosen parent designs (e.g. the best dataset
    designs). The noise std dev is sigma_fraction of each coefficient's window width, or the per-coefficient
    `sigma` array when given. The result is clipped to the search window (or to clip_lows/clip_highs when
    given), so candidates near an edge stay on it.
    """
    clip_lows = lows if clip_lows is None else clip_lows
    clip_highs = highs if clip_highs is None else clip_highs
    sigma = (highs - lows) * sigma_fraction if sigma is None else sigma
    centers = parents[rng.integers(0, len(parents), size=num_samples)]
    noise = rng.normal(0.0, 1.0, size=centers.shape).astype(np.float32) * sigma
    return np.clip(centers + noise, clip_lows, clip_highs).astype(np.float32)


def detect_edge_directions(parents, lows, highs, steps):
    """
    Window edges worth probing: a coefficient's edge is probed when at least EDGE_MIN_SHARE of the top dataset
    designs sit on it. The window is extended one grid step past each such edge.
    Returns (directions, ext_lows, ext_highs); directions is a list of (coefficient index, 'low'/'high', new bound).
    """
    directions = []
    ext_lows, ext_highs = lows.copy(), highs.copy()
    for i in range(len(lows)):
        if np.isclose(parents[:, i], highs[i], atol=1e-6).mean() >= EDGE_MIN_SHARE:
            ext_highs[i] = highs[i] + steps[i]
            directions.append((i, 'high', float(ext_highs[i])))
        if np.isclose(parents[:, i], lows[i], atol=1e-6).mean() >= EDGE_MIN_SHARE:
            ext_lows[i] = lows[i] - steps[i]
            directions.append((i, 'low', float(ext_lows[i])))
    return directions, ext_lows, ext_highs


def outside_window(cst_block, lows, highs):
    """True for every row with at least one CST coefficient outside the search window."""
    return ((cst_block < lows - 1e-7) | (cst_block > highs + 1e-7)).any(axis=1)


def predict_chunked(predict_fn, X, chunk_size=PREDICT_CHUNK):
    """Runs predict_fn on X in row chunks and returns one flat array (bounds GPU/CPU memory use)."""
    parts = [np.asarray(predict_fn(X[i:i + chunk_size])).reshape(-1) for i in range(0, len(X), chunk_size)]
    return np.concatenate(parts)


def predict_all(regressors: dict, X) -> np.ndarray:
    """(rows, models) matrix of predictions, one column per regressor in CLI order. Non-finite values become NaN."""
    preds = np.column_stack([predict_chunked(m.predict, X).astype(np.float64) for m in regressors.values()])
    preds[~np.isfinite(preds)] = np.nan
    return preds


def consensus_score(preds: np.ndarray) -> np.ndarray:
    """Most pessimistic prediction across models (NaN if any model returned NaN): high only if all models agree."""
    return np.min(preds, axis=1)


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


def with_config(features, config_cols, feature_columns, config_col):
    """Copy of a full feature row with its one-hot block switched to config_col."""
    row = features.copy()
    for c in config_cols:
        row[feature_columns.index(c)] = 1.0 if c == config_col else 0.0
    return row


def load_dataset_info(csv_path: Path, cst_l_cols, cst_u_cols, config_cols, top_k: int):
    """
    Reads the training dataset once. Returns None if it is unavailable, otherwise a namespace with:
      best         -- dict with the highest-E row (E, wing_config, cst_lower, cst_upper). Information + reference.
      e_max        -- highest E in the dataset
      parents      -- array (<= top_k, 2 * n_cst): distinct CST vectors of the highest-E rows (local search centers)
      reference    -- list of (cst vector, config key, E) for the REFERENCE_TOP highest-E rows
      steps        -- dataset grid step of every CST coefficient (lower block, then upper block)
      trees        -- {config key: (cKDTree over CST / steps, E array)} for nearest-design lookups
      config_order -- config keys sorted by mean E over the shapes simulated in every configuration
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        return None
    target = df.columns[-1]
    cst_cols = cst_l_cols + cst_u_cols
    if config_cols:
        df['_config'] = np.array(config_cols)[df[config_cols].to_numpy().argmax(axis=1)]
    else:
        df['_config'] = SINGLE_CONFIG_KEY
    ordered = df.sort_values(target, ascending=False)
    best_row = ordered.iloc[0]
    best_config = best_row['_config']
    best = {
        "E": float(best_row[target]),
        "wing_config": best_config if config_cols else "single configuration",
        "cst_lower": [round(float(best_row[c]), 6) for c in cst_l_cols],
        "cst_upper": [round(float(best_row[c]), 6) for c in cst_u_cols],
    }
    parents = ordered[cst_cols].drop_duplicates().head(top_k).to_numpy(dtype=np.float32)
    reference = [(r[cst_cols].to_numpy(dtype=np.float32), r['_config'], float(r[target]))
                 for _, r in ordered.head(REFERENCE_TOP).iterrows()]

    steps = []
    for col in cst_cols:
        u = np.unique(np.round(df[col].to_numpy(), 6))
        steps.append(float(np.min(np.diff(u))) if len(u) > 1 else 1.0)
    steps = np.array(steps, dtype=np.float64)

    trees = {}
    for key, part in df.groupby('_config'):
        trees[key] = (cKDTree(part[cst_cols].to_numpy(dtype=np.float64) / steps), part[target].to_numpy(dtype=np.float64))

    if config_cols:
        pivot = df.pivot_table(index=cst_cols, columns='_config', values=target).dropna()
        config_order = list(pivot.mean().sort_values(ascending=False).index)
    else:
        config_order = [SINGLE_CONFIG_KEY]
    return SimpleNamespace(best=best, e_max=float(ordered[target].iloc[0]), parents=parents, reference=reference,
                           steps=steps, trees=trees, config_order=config_order)


def tag_regions(cst_block, config_labels, info, lows, highs):
    """
    Distance (Chebyshev, in grid steps) to the nearest dataset design of the same configuration, that design's E,
    and the region label: 'outside' (past the window), 'on-grid' (a dataset design) or 'between' (interpolation).
    """
    n = len(cst_block)
    dist, near_e = np.full(n, np.nan), np.full(n, np.nan)
    scaled = cst_block.astype(np.float64) / info.steps
    for key, (tree, e_values) in info.trees.items():
        mask = config_labels == key
        if mask.any():
            d, j = tree.query(scaled[mask], p=np.inf)
            dist[mask], near_e[mask] = d, e_values[j]
    region = np.where(outside_window(cst_block, lows, highs), 'outside', np.where(dist < 1e-3, 'on-grid', 'between'))
    return dist, near_e, region


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
# === Shortlist === #
# =============================================================================================== #

def top_indices(score, k, mask=None):
    """Indices of the k highest finite scores (optionally restricted to mask), unordered."""
    s = np.where(np.isfinite(score), score, -np.inf)
    if mask is not None:
        s = np.where(mask, s, -np.inf)
    k = min(k, int(np.isfinite(s).sum()))
    if k == 0:
        return np.array([], dtype=int)
    return np.argpartition(-s, k - 1)[:k]


def diverse_top(score, cst_scaled, k, mask=None, min_steps=DIVERSITY_MIN_STEPS):
    """
    Walks the rows from the highest score down and keeps a row only if its CST vector differs from every kept
    row by at least min_steps grid steps in some coefficient. Stops at k rows. Returns indices in rank order.
    """
    s = np.where(np.isfinite(score), score, -np.inf)
    if mask is not None:
        s = np.where(mask, s, -np.inf)
    kept = []
    for i in np.argsort(-s):
        if len(kept) >= k or not np.isfinite(s[i]):
            break
        if kept and np.abs(cst_scaled[kept] - cst_scaled[i]).max(axis=1).min() < min_steps:
            continue
        kept.append(int(i))
    return kept


def add_candidate(unique: dict, features, config_col, preds, origin, group, rank, score, extra=None):
    """
    Adds a shortlisted row, merging identical feature vectors (configuration included). Each unique candidate
    keeps a 'sources' list with every (group, rank, score) that selected it.
    """
    key = np.asarray(features, dtype=np.float32).tobytes()
    if key not in unique:
        unique[key] = {
            'candidate_id': f"candidate_{len(unique) + 1:03d}",
            'features': np.asarray(features, dtype=np.float32),
            'config_col': str(config_col),
            'preds': np.asarray(preds, dtype=np.float64),
            'origin': origin,
            'sources': [],
            **(extra or {}),
        }
    unique[key]['sources'].append({'group': group, 'rank': rank, 'score': None if score is None else float(score)})


# =============================================================================================== #
# === Simulation === #
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


def simulate_candidates(candidates: list, stage: str, ctx) -> dict:
    """
    Runs the full simulation chain (airfoil .dat -> XFOIL -> key points -> .avl geometry -> AVL) sequentially for
    every candidate, archives the raw files of the stage and returns {candidate_id: result dict}.
    Result statuses: 'ok', 'invalid_geometry' (airfoil surfaces cross; not simulated), 'no_result' (a step failed).
    """
    results = {c['candidate_id']: {'status': 'no_result', 'CL': np.nan, 'CD': np.nan, 'E': np.nan} for c in candidates}
    if not candidates:
        return results
    print(f"\n# === Simulation stage '{stage}': {len(candidates)} candidate(s) === #")
    angles = [ASEQ[0] + j * ASEQ[2] for j in range(int((ASEQ[1] - ASEQ[0]) / ASEQ[2]) + 1)]
    temp_root = get_workspace()
    prefix = f"{ctx.prefix}_{stage}"
    try:
        airfoils_temp = temp_root / "airfoils"
        polars_temp = temp_root / "polars"
        geo_temp = temp_root / "geometry"
        forces_temp = temp_root / "forces"
        support_temp = temp_root / "support"
        for d in [airfoils_temp, polars_temp, geo_temp, forces_temp, support_temp]:
            d.mkdir(parents=True, exist_ok=True)

        # Generate Airfoils (geometrically invalid shapes are not simulated)
        print("\n--- Generating Airfoil Geometry ---")
        valid = []
        for c in tqdm(candidates, desc="Creating .dat files"):
            x_mod, yu_mod, yl_mod = ctx.airfoil_creator.generate(cst_coeffs=c['features'][ctx.cst_col_idx])
            if not check_validity(x_mod, yu_mod, yl_mod, verbose=False):
                results[c['candidate_id']]['status'] = 'invalid_geometry'
                continue
            ctx.airfoil_creator._create_dat_file(output_dir=airfoils_temp, n_cst=ctx.n_cst, x=x_mod, yl=yl_mod, yu=yu_mod,
                                                 file_name=c['candidate_id'])
            valid.append(c)
        if not valid:
            print("No geometrically valid airfoils in this stage.")
            return results

        # Run XFOIL (same conditions as the dataset generation)
        print("\n--- Generating Polars (XFOIL) ---")
        airfoil_paths = get_all_files_from(airfoils_temp, extension='dat')
        xfoil_case = XPR(xfoil_exe_path=str(ctx.paths.xfoil_bin), main_dir=temp_root, airfoil_dir=airfoils_temp,
                         polar_dir=polars_temp, verbose=False)
        for path in tqdm(airfoil_paths, desc="Running XFOIL"):
            polar_path = polars_temp / f"polar_{Path(path).stem}.txt"
            # Does not require try/except due to internal handler
            xfoil_case.run_simulation(airfoil_file_path=path, polar_file_path=str(polar_path),
                                      alfa_sequence=angles, visc=RE, mach=MACH, sweep_mode=SWEEP_MODE)

        # Create Wing Geometries, each candidate uses the creator matching its own configuration
        print("\n--- Generating Wing Geometry (.avl files) ---")
        polar_paths = get_all_files_from(polars_temp, extension='txt')
        key_points = Key_Point_Extraction(polar_file_paths=polar_paths, verbose=True, random_state=KP_SEED)
        key_points_data = key_points.process_files()
        key_points.save_error_log(os.path.join(geo_temp, 'KP_error_log.txt'))

        airfoil_paths_by_config = defaultdict(list)
        for c in valid:
            airfoil_paths_by_config[c['config_col']].append(str(airfoils_temp / f"{c['candidate_id']}.dat"))

        for config_col, subset_paths in airfoil_paths_by_config.items():
            creator_class, identifier = ctx.config_registry[config_col]
            print(f"  -> {len(subset_paths)} airfoil(s) under '{identifier}' configuration")
            geo_creator = creator_class(
                key_points_data=key_points_data,
                airfoil_abs_paths=subset_paths,
                naca_key_points_data=str(ctx.paths.naca_kp_file),
                output_dir=str(geo_temp),
                verbose=False,
                pb=True
            )
            geo_creator.create_geo_files()
            geo_creator.save_error_log(os.path.join(geo_temp, f'geo_{identifier}_error_log.txt'))

        # Calculate Final Forces (AVL), sequentially
        print("\n--- Calculating Final Coefficients (AVL) ---")
        geo_paths = get_all_files_from(geo_temp, extension='avl')
        shutil.copy(ctx.paths.case_file, support_temp)
        case_file_temp = support_temp / ctx.paths.case_file.name
        for path in tqdm(geo_paths, desc="Running AVL sequentially"):
            avl_case = AVL_Run_Case(geo_files=[path], case_file=str(case_file_temp), avl_exe_path=str(ctx.paths.avl_bin),
                                    output_dir=forces_temp, debug=False)
            # Does not require try/except due to internal handler
            avl_case.run()

        print("\n--- Archiving Raw Data Files ---")
        ctx.archiver.compress_folder(airfoils_temp, ctx.run_dir, f"{prefix}_airfoils")
        ctx.archiver.compress_folder(polars_temp, ctx.run_dir, f"{prefix}_polars")
        ctx.archiver.compress_folder(geo_temp, ctx.run_dir, f"{prefix}_geo")
        ctx.archiver.compress_folder(forces_temp, ctx.run_dir, f"{prefix}_forces")

        # Parse simulation results
        for fpath in tqdm(get_all_files_from(forces_temp, extension='txt'), desc="Reading force files"):
            airfoil_name, cl, cd = extract_force_data(Path(fpath))
            if airfoil_name is None or airfoil_name not in results:
                continue  # Skip unparseable files entirely
            e = compute_simulated_E(cl, cd)
            results[airfoil_name] = {'status': 'ok' if not math.isnan(e) else 'no_result', 'CL': cl, 'CD': cd, 'E': e}
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
    return results


# =============================================================================================== #
# === Reporting === #
# =============================================================================================== #

def safe_spearman(a, b):
    """Rank agreement between two sequences, or None when it is undefined (fewer than 3 points or constant)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(spearmanr(a, b)[0])


def group_score(candidate, group, model_names):
    """The value a group ranked by: the model's own prediction for a model group, the consensus otherwise."""
    if group.startswith("model:"):
        return float(candidate['preds'][model_names.index(group[len("model:"):])])
    return float(candidate['consensus'])


def build_scorecard(candidates, groups, model_names, e_best, e_max):
    """One row per selection group: how its simulated picks performed against the dataset best."""
    rows = []
    for group in groups:
        picks = [c for c in candidates if any(s['group'] == group for s in c['sources'])]
        if not picks:
            continue
        sims = [c for c in picks if c['status'] == 'ok']
        scores = np.array([group_score(c, group, model_names) for c in sims])
        sim_e = np.array([c['simulated_E'] for c in sims])
        rows.append({
            'group': group,
            'picks': len(picks),
            'simulated': len(sims),
            'failed': sum(c['status'] == 'no_result' for c in picks),
            'invalid_geometry': sum(c['status'] == 'invalid_geometry' for c in picks),
            'best_simulated_E': float(sim_e.max()) if len(sims) else None,
            'regret': float(e_best - sim_e.max()) if len(sims) else None,
            'median_simulated_E': float(np.median(sim_e)) if len(sims) else None,
            'hits': int((sim_e >= HIT_FRACTION * e_best).sum()),
            'mean_signed_error': float(np.mean(scores - sim_e)) if len(sims) else None,
            'rank_agreement': safe_spearman(scores, sim_e),
            'scores_above_dataset_max': int(sum(group_score(c, group, model_names) > e_max for c in picks)),
        })
    return rows


def build_model_table(candidates, model_names, e_max):
    """Every model judged on every simulated main-stage candidate (not only its own picks): a fair comparison."""
    sims = [c for c in candidates if c['status'] == 'ok']
    sim_e = np.array([c['simulated_E'] for c in sims])
    rows = []
    for m, name in enumerate(model_names):
        pred = np.array([c['preds'][m] for c in sims])
        ok = np.isfinite(pred)
        rows.append({
            'model': name,
            'candidates': int(ok.sum()),
            'mean_signed_error': float(np.mean(pred[ok] - sim_e[ok])) if ok.any() else None,
            'mean_abs_error': float(np.mean(np.abs(pred[ok] - sim_e[ok]))) if ok.any() else None,
            'rank_agreement': safe_spearman(pred[ok], sim_e[ok]),
            'predictions_above_dataset_max': int((pred[ok] > e_max).sum()),
        })
    return rows


def fmt(v, digits=2, width=None):
    """Number formatting that tolerates None/NaN."""
    text = "-" if v is None or (isinstance(v, float) and math.isnan(v)) else (f"{v:.{digits}f}" if isinstance(v, float) else str(v))
    return text if width is None else f"{text:<{width}}"


def short_sources(sources):
    labels = []
    for s in sources:
        g = s['group'][len("model:"):] if s['group'].startswith("model:") else s['group']
        labels.append(f"{g}#{s['rank']}")
    return ", ".join(labels)


# Plain block letters, 5 rows per glyph
BANNER_GLYPHS = {
    "S": [" ____  ", "/ ___| ", "\\___ \\ ", " ___) |", "|____/ "],
    "U": [" _   _ ", "| | | |", "| | | |", "| |_| |", " \\___/ "],
    "C": ["  ____ ", " / ___|", "| |    ", "| |___ ", " \\____|"],
    "E": [" _____ ", "| ____|", "|  _|  ", "| |___ ", "|_____|"],
    "T": [" _____ ", "|_   _|", "  | |  ", "  | |  ", "  |_|  "],
    "R": [" ____  ", "|  _ \\ ", "| |_) |", "|  _ < ", "|_| \\_\\"],
    "Y": ["__   __", "\\ \\ / /", " \\ V / ", "  | |  ", "  |_|  "],
    "A": ["    _    ", "   / \\   ", "  / _ \\  ", " / ___ \\ ", "/_/   \\_\\"],
    "G": ["  ____ ", " / ___|", "| |  _ ", "| |_| |", " \\____|"],
    "I": [" ___ ", "|_ _|", " | | ", " | | ", "|___|"],
    "N": [" _   _ ", "| \\ | |", "|  \\| |", "| |\\  |", "|_| \\_|"],
    "!": [" _ ", "| |", "| |", "|_|", "(_)"],
    ".": ["   ", "   ", "   ", " _ ", "(_)"],
    " ": ["   ", "   ", "   ", "   ", "   "],
}


def banner(text: str) -> list:
    """Big ASCII letters for the outcome line of result.txt."""
    glyphs = [BANNER_GLYPHS[ch] for ch in text.upper()]
    return ["".join(g[row].ljust(max(len(r) for r in g)) for g in glyphs).rstrip() for row in range(5)]


def find_outcome(candidates, model_names, e_dataset_best, reference_best_E):
    """
    Did any model find a better design? Candidates picked by a model, by consensus, by the edge probe or by the
    configuration check count; dataset designs re-simulated as they are (reference picks or exact on-grid rows)
    do not. A design must beat both the dataset best and that design's own re-simulation, so simulation
    differences alone cannot produce a success.
    """
    target = max(e_dataset_best, reference_best_E) if reference_best_E is not None else e_dataset_best
    improvements = []
    for c in candidates:
        if c['status'] != 'ok' or not math.isfinite(c['simulated_E']):
            continue
        if all(s['group'] == GROUP_REFERENCE for s in c['sources']) or c['grid_distance'] < 1e-3:
            continue
        if c['simulated_E'] > target:
            improvements.append(c)
    improvements.sort(key=lambda c: c['simulated_E'], reverse=True)
    per_model = {}
    for name in model_names:
        own = [c['simulated_E'] for c in candidates if c['status'] == 'ok'
               and any(s['group'] == f"model:{name}" for s in c['sources'])]
        per_model[name] = max(own) if own else None
    return {'success': bool(improvements), 'target_E': target, 'dataset_best_E': e_dataset_best,
            'reference_best_simulated_E': reference_best_E, 'improvements': improvements, 'best_per_model': per_model}


def save_result_report(path: Path, ctx, outcome: dict, best_candidate, robust_by_parent: dict, run_info: dict):
    """result.txt: the outcome in big letters and the bare minimum around it."""
    message = "SUCCESS!!" if outcome['success'] else "Try again..."
    target = outcome['target_E']
    lines = [""] + ["   " + row for row in banner(message)] + ["", "=" * 90]
    lines.append(f"Goal: a wing design with simulated E above the best of dataset {ctx.dataset_num}.")
    ref = outcome['reference_best_simulated_E']
    lines.append(f"Target: E > {target:.4f}  (dataset best {outcome['dataset_best_E']:.4f}"
                 + (f", re-simulated {ref:.4f})" if ref is not None else ")"))
    lines.append("")
    if best_candidate is not None:
        gain = 100 * (best_candidate['simulated_E'] - target) / target
        lines.append(f"Best design found: E = {best_candidate['simulated_E']:.4f} ({gain:+.2f}% vs target) | "
                     f"config {best_candidate['wing_config']} | picked by {short_sources(best_candidate['sources'])}")
        n_cst = len(best_candidate['cst_coeffs']) // 2
        lines.append(f"  CST lower: {[round(float(v), 4) for v in best_candidate['cst_coeffs'][:n_cst]]}")
        lines.append(f"  CST upper: {[round(float(v), 4) for v in best_candidate['cst_coeffs'][n_cst:]]}")
        rob = robust_by_parent.get(best_candidate['candidate_id'])
        if rob and rob['simulated']:
            lines.append(f"  Robustness: {rob['simulated']} small perturbations, E from {rob['min_E']:.2f} to "
                         f"{rob['max_E']:.2f} (hold rate {rob['hold_rate']:.0%})")
        lines.append("")
    if outcome['success']:
        lines.append(f"Designs above the target: {len(outcome['improvements'])}")
        for c in outcome['improvements'][:10]:
            lines.append(f"  {c['candidate_id']:<14} E = {c['simulated_E']:.4f}  {c['wing_config']:<6} "
                         f"{short_sources(c['sources'])}")
        lines.append("")
    lines.append("Best simulated E of each model's own picks:")
    for name, e in outcome['best_per_model'].items():
        verdict = "-" if e is None else ("beats the target" if e > target else f"{e - target:+.2f} from the target")
        lines.append(f"  {name:<18} {fmt(e, 4):>8}   {verdict}")
    lines += ["", f"Run: {run_info['folder']} | master seed {run_info['seed']} | {run_info['total_time']}",
              "Full details: detailed_results.txt", "=" * 90]
    path.write_text("\n".join(lines) + "\n", encoding='utf-8')
    print("\n".join(lines[:7]))
    print(f"Saved result summary to '{path.name}'")


def save_text_report(path: Path, ctx, report: dict):
    """Human-readable report: header, scorecards, configuration check, candidates and robustness."""
    model_names = ctx.model_names
    best = report['dataset_best']
    width = 150
    lines = ["=" * width, f"Aerodynamic Efficiency (E) Validation Report for Dataset {ctx.dataset_num}", "=" * width]
    if best is not None:
        lines += [f"Dataset best: E = {best['E']:.4f} | config = {best['wing_config']}",
                  f"  CST lower: {best['cst_lower']}", f"  CST upper: {best['cst_upper']}",
                  f"Hit = simulated E >= {HIT_FRACTION:.0%} of the dataset best ({HIT_FRACTION * best['E']:.2f}). "
                  f"Signed error = prediction - simulation (positive = the model was too optimistic)."]
    outcome = report['outcome']
    lines += [f"Outcome: {'SUCCESS!!' if outcome['success'] else 'Try again...'} | target E > {outcome['target_E']:.4f} | "
              f"designs above it: {len(outcome['improvement_ids'])}",
              f"Inference rounds completed: {report['rounds_completed']} | Total time: {report['total_time']}", ""]

    lines += ["--- Scorecard per selection group (each group judged with the score it ranked by) " + "-" * 60]
    lines.append(f"{'Group':<18} {'Picks':>5} {'Sim':>4} {'Fail':>4} {'Inval':>5} {'Best E':>8} {'Regret':>7} "
                 f"{'Median E':>9} {'Hits':>4} {'Signed err':>10} {'Rank agr.':>9} {'Score>max':>9}")
    for r in report['scorecard']:
        g = r['group'][len("model:"):] if r['group'].startswith("model:") else r['group']
        lines.append(f"{g:<18} {r['picks']:>5} {r['simulated']:>4} {r['failed']:>4} {r['invalid_geometry']:>5} "
                     f"{fmt(r['best_simulated_E']):>8} {fmt(r['regret']):>7} {fmt(r['median_simulated_E']):>9} "
                     f"{r['hits']:>4} {fmt(r['mean_signed_error']):>10} {fmt(r['rank_agreement']):>9} "
                     f"{r['scores_above_dataset_max']:>9}")
    lines.append("")

    lines += ["--- Every model on every simulated main-stage candidate " + "-" * 94]
    lines.append(f"{'Model':<18} {'N':>4} {'Signed err':>10} {'Abs err':>8} {'Rank agr.':>9} {'Pred>max':>8}")
    for r in report['model_table']:
        lines.append(f"{r['model']:<18} {r['candidates']:>4} {fmt(r['mean_signed_error']):>10} "
                     f"{fmt(r['mean_abs_error']):>8} {fmt(r['rank_agreement']):>9} {r['predictions_above_dataset_max']:>8}")
    lines.append("")

    cfg_names = [ctx.config_registry[c][1] for c in ctx.config_order]
    lines += [f"--- Configuration check (dataset order: {' > '.join(cfg_names)}) " + "-" * 80]
    for r in report['config_check']['model_winners']:
        preds = "  ".join(f"{k}={fmt(v)}" for k, v in r['predictions'].items())
        lines.append(f"{r['model']:<18} winner {r['candidate_id']} ({r['picked_config']}): {preds}  "
                     f"-> {'matches' if r['order_matches_dataset'] else 'DIFFERS FROM'} dataset order")
    verdicts = {True: "matches dataset order", False: "DIFFERS FROM dataset order", None: "incomplete (a simulation failed)"}
    for r in report['config_check']['simulated_shapes']:
        sims = "  ".join(f"{k}={fmt(v)}" for k, v in r['simulated_E'].items())
        lines.append(f"Consensus shape #{r['shape']} simulated: {sims}  -> {verdicts[r['order_matches_dataset']]}")
    lines.append("")

    lines += ["--- Candidates (sorted by simulated E) " + "-" * 112]
    model_hdr = " ".join(f"{n[:8]:>8}" for n in model_names)
    lines.append(f"{'Candidate':<14} {'Cfg':<6} {'Region':<15} {model_hdr} {'Spread':>6} {'Sim E':>7} {'Err(cons)':>9} "
                 f"{'Near E':>7}  {'Flags':<10} Picked by")
    for c in sorted(report['candidates'], key=lambda c: -np.inf if c['simulated_E'] is None else c['simulated_E'], reverse=True):
        region = f"{c['region']} {fmt(c['grid_distance'])}"
        preds = " ".join(f"{fmt(p):>8}" for p in c['predictions'].values())
        flags = []
        if c['above_dataset_max']:
            flags.append("pred>max")
        if c['status'] != 'ok':
            flags.append(c['status'])
        lines.append(f"{c['candidate_id']:<14} {c['wing_config']:<6} {region:<15} {preds} {fmt(c['spread']):>6} "
                     f"{fmt(c['simulated_E']):>7} {fmt(c['consensus_error']):>9} {fmt(c['nearest_dataset_E']):>7}  "
                     f"{','.join(flags):<10} {short_sources(c['sources'])}")
    lines.append("")

    rob = report['robustness']
    lines += ["--- Robustness (small perturbations of the best simulated designs) " + "-" * 83]
    if rob['status'] != 'completed':
        lines.append(f"Not run: {rob['status']}")
    else:
        lines.append(f"Perturbation std dev: {ROBUST_SIGMA_STEPS} grid steps. A perturbation holds if it simulates at "
                     f">= {ROBUST_HOLD_FRACTION:.0%} of its parent's E.")
        lines.append(f"{'Parent':<14} {'Parent E':>8} {'Pert.':>5} {'Sim':>4} {'Hold rate':>9} {'Mean E':>7} {'Min E':>7} {'Max E':>7}")
        for r in rob['parents']:
            lines.append(f"{r['parent']:<14} {fmt(r['parent_E']):>8} {r['perturbations']:>5} {r['simulated']:>4} "
                         f"{fmt(r['hold_rate']):>9} {fmt(r['mean_E']):>7} {fmt(r['min_E']):>7} {fmt(r['max_E']):>7}")
    lines.append("=" * width)
    path.write_text("\n".join(lines) + "\n", encoding='utf-8')
    print(f"Successfully saved text report to '{path.name}'")


def save_parity_plot(path: Path, candidates, model_names, e_best):
    """Predicted vs simulated E for every model on every simulated main-stage candidate."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; parity plot skipped.")
        return
    sims = [c for c in candidates if c['status'] == 'ok']
    if not sims:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    sim_e = np.array([c['simulated_E'] for c in sims])
    for m, name in enumerate(model_names):
        ax.scatter([c['preds'][m] for c in sims], sim_e, s=22, alpha=0.75, label=name)
    lo = min(np.nanmin(sim_e), np.nanmin([c['preds'] for c in sims])) - 2
    hi = max(np.nanmax(sim_e), np.nanmax([c['preds'] for c in sims]), e_best) + 2
    ax.plot([lo, hi], [lo, hi], ls='--', c='gray', lw=1, label='perfect prediction')
    ax.axhline(e_best, c='green', lw=1, label=f'dataset best ({e_best:.2f})')
    ax.set_xlabel("Predicted E")
    ax.set_ylabel("Simulated E")
    ax.set_title("Predicted vs simulated E (main stage)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved parity plot to '{path.name}'")


def to_json(value):
    """Recursively converts numpy types and NaN to plain JSON values."""
    if isinstance(value, dict):
        return {str(k): to_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_json(value.tolist())
    if isinstance(value, (np.floating, float)):
        return None if math.isnan(float(value)) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path: Path, data):
    with open(path, 'w') as f:
        json.dump(to_json(data), f, indent=4)


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
    if args.csv:
        paths.training_file = Path(args.csv).resolve()
    print(f"Dataset CSV: {paths.training_file}")
    dataset_name_prefix = f"AI_Test_Results_dataset_{dataset_num}"
    start_time = datetime.now()
    print(f"Pipeline started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # === Seeds: one per inference round, identical for every model === #
    # NumPy's Generator (np.random.default_rng) accepts any non-negative integer, so the accepted range
    # is simply MAX_MASTER_SEED. The round seeds are a deterministic function of the master seed.
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
    model_names = list(regressors)

    # === Dataset information: best row, local-search centers, grid steps, nearest-design lookup === #
    info = load_dataset_info(paths.training_file, cst_l_cols, cst_u_cols, config_cols, LOCAL_TOP_K)
    if info is None:
        raise FileNotFoundError(f"Training dataset not found: {paths.training_file}. It is needed for the local search, "
                                f"region tagging and the reference designs.")
    lows, highs = window_bounds(n_cst, RANGES_UPPER, RANGES_LOWER)
    cst_names = cst_l_cols + cst_u_cols
    edge_directions, ext_lows, ext_highs = detect_edge_directions(info.parents, lows, highs, info.steps)
    edge_sigma = (highs - lows) * LOCAL_SIGMA_FRACTION
    for i, _side, _bound in edge_directions:
        edge_sigma[i] = info.steps[i] * EDGE_SIGMA_STEPS
    use_local = NUM_LOCAL_SAMPLES > 0
    use_edge = NUM_EDGE_SAMPLES > 0 and bool(edge_directions)

    # === Run folder === #
    paths.results_root.mkdir(parents=True, exist_ok=True)
    run_dir = make_run_dir(paths.results_root, master_seed)
    print(f"Results folder: {run_dir}")

    # === Metadata (written now, completed at the end of the run) === #
    metadata = {
        "run_status": "running",
        "started": start_time.strftime('%Y-%m-%d %H:%M:%S'),
        "dataset": dataset_num,
        "dataset_csv": str(paths.training_file),
        "wing_mode": args.wing_mode,
        "wing_configurations": [v[1] for v in config_registry.values()],
        "master_seed": {"hex": f"{master_seed:x}", "decimal": master_seed},
        "round_seeds_hex": [f"{s:x}" for s in round_seeds],
        "classifier": {"name": args.classifier, "folder": str(paths.classifier_dir)},
        "models": [{"name": n, "type": resolved[n][1], "folder": str(resolved[n][0]),
                    "feature_mode": getattr(regressors[n], 'feature_mode', None)} for n in regressors],
        "skipped_models": skipped_models,
        "inference": {"num_runs": NUM_INFERENCE_RUNS, "num_samples": NUM_SAMPLES, "validity_threshold": VALIDITY_THRESHOLD,
                      "num_local_samples": NUM_LOCAL_SAMPLES, "local_centers": len(info.parents),
                      "local_sigma_fraction": LOCAL_SIGMA_FRACTION, "num_edge_samples": NUM_EDGE_SAMPLES if use_edge else 0,
                      "edge_sigma_grid_steps": EDGE_SIGMA_STEPS,
                      "elite_per_criterion": ELITE_PER_CRITERION, "predict_chunk": PREDICT_CHUNK},
        "shortlist": {"top_k_per_model": TOP_K_PER_MODEL, "top_k_consensus": TOP_K_CONSENSUS, "top_k_edge": TOP_K_EDGE,
                      "config_check_shapes": CONFIG_CHECK_SHAPES, "reference_top": REFERENCE_TOP,
                      "diversity_min_steps": DIVERSITY_MIN_STEPS, "consensus": "minimum prediction across models"},
        "robustness": {"designs": ROBUST_DESIGNS, "perturbations": ROBUST_PERTURBATIONS,
                       "sigma_grid_steps": ROBUST_SIGMA_STEPS, "hold_fraction": ROBUST_HOLD_FRACTION},
        "grid_steps": dict(zip(cst_names, info.steps.tolist())),
        "edge_probe": [{"coefficient": cst_names[i], "side": side, "extended_to": bound} for i, side, bound in edge_directions],
        "ranges_upper": RANGES_UPPER,
        "ranges_lower": RANGES_LOWER,
        "simulation": {"ASEQ": ASEQ, "Re": RE, "Mach": MACH, "naca_key_points_file": NACA_KP_FILE},
        "stage_times": {},
    }
    metadata_path = run_dir / f"{dataset_name_prefix}_run_metadata.json"
    write_json(metadata_path, metadata)

    print_dataset_best(info.best)
    print(f"Local refinement: {NUM_LOCAL_SAMPLES if use_local else 0} samples per round around {len(info.parents)} "
          f"top dataset designs (sigma = {LOCAL_SIGMA_FRACTION} x range width).")
    if use_edge:
        print(f"Edge probe: {NUM_EDGE_SAMPLES} samples per round (sigma = {EDGE_SIGMA_STEPS} grid steps on the extended "
              f"coefficients) past {[f'{cst_names[i]} {side} -> {bound:.4f}' for i, side, bound in edge_directions]}")
    else:
        print("Edge probe disabled (NUM_EDGE_SAMPLES = 0 or no edge shared by the top dataset designs).")

    # === Stage 1: candidate pool. Every model scores every row; an elite subset of each round is kept === #
    pool = defaultdict(list)
    rounds_completed, t_inference = 0, time.perf_counter()
    for r, seed in enumerate(round_seeds, start=1):
        print(f"\n--- Inference Run {r}/{NUM_INFERENCE_RUNS} (seed {seed:x}) ---")
        rng = np.random.default_rng(seed)
        blocks = [(generate_cst_candidates(NUM_SAMPLES, n_cst, RANGES_UPPER, RANGES_LOWER, rng), 'global')]
        if use_local:
            blocks.append((generate_local_candidates(info.parents, NUM_LOCAL_SAMPLES, lows, highs, LOCAL_SIGMA_FRACTION, rng), 'local'))
        if use_edge:
            edge = generate_local_candidates(info.parents, NUM_EDGE_SAMPLES, lows, highs, LOCAL_SIGMA_FRACTION, rng,
                                             clip_lows=ext_lows, clip_highs=ext_highs, sigma=edge_sigma)
            blocks.append((edge[outside_window(edge, lows, highs)], 'edge'))
        cst_candidates = np.concatenate([b for b, _ in blocks])
        origins = np.concatenate([np.full(len(b), o) for b, o in blocks])

        # Classifier works on CST combinations; invalid ones are discarded.
        validity_probs = predict_chunked(classifier.predict, cst_candidates)
        valid_indices = np.where(validity_probs > VALIDITY_THRESHOLD)[0]
        if len(valid_indices) == 0:
            print("No valid candidates found in this run. Skipping.")
            rounds_completed += 1
            continue
        valid_origins = origins[valid_indices]
        print(f"{len(valid_indices)} valid CST combinations out of {len(cst_candidates)} "
              f"({', '.join(f'{o}: {int((valid_origins == o).sum())}' for _, o in blocks)}).")
        full_candidates, config_labels = expand_with_configs(
            cst_candidates[valid_indices], feature_columns, cst_l_cols, cst_u_cols, config_cols)
        num_configs = max(len(config_cols), 1)  # row i of full_candidates belongs to valid sample i // num_configs
        row_origins = np.repeat(valid_origins, num_configs)

        t0 = time.perf_counter()
        preds = predict_all(regressors, full_candidates)
        consensus = consensus_score(preds)
        in_window = row_origins != 'edge'
        for m, name in enumerate(model_names):
            column = np.where(in_window, preds[:, m], np.nan)
            if np.isnan(column).all():
                print(f"[WARNING] '{name}' returned only NaN/inf predictions in run {r}.")
                continue
            i = int(np.nanargmax(column))
            print(f"  {name}: best predicted E = {preds[i, m]:.4f} [{row_origins[i]}]")
        print(f"  Scored {len(full_candidates)} rows with {len(model_names)} model(s) in {time.perf_counter() - t0:.1f}s")

        elite = set()
        for m in range(len(model_names)):
            elite.update(top_indices(preds[:, m], ELITE_PER_CRITERION, in_window).tolist())
        elite.update(top_indices(consensus, ELITE_PER_CRITERION, in_window).tolist())
        elite.update(top_indices(consensus, ELITE_PER_CRITERION, ~in_window).tolist())
        elite = np.array(sorted(elite), dtype=int)
        pool['features'].append(full_candidates[elite])
        pool['config'].append(config_labels[elite])
        pool['preds'].append(preds[elite])
        pool['origin'].append(row_origins[elite])
        pool['round'].append(np.full(len(elite), r))
        rounds_completed += 1

    metadata['stage_times']['inference'] = str(timedelta(seconds=round(time.perf_counter() - t_inference)))
    metadata['rounds_completed'] = rounds_completed
    if not pool['features']:
        print("\nWARNING: No candidates to validate. Nothing to simulate.")
        metadata.update({"run_status": "completed_no_candidates", "finished": datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
        write_json(metadata_path, metadata)
        sys.exit(0)
    pool = {k: np.concatenate(v) for k, v in pool.items()}
    pool_cst = pool['features'][:, cst_col_idx]
    pool_scaled = pool_cst.astype(np.float64) / info.steps
    pool_consensus = consensus_score(pool['preds'])
    _, _, pool_region = tag_regions(pool_cst, pool['config'], info, lows, highs)
    pool_inside = pool_region != 'outside'
    print(f"\nElite pool: {len(pool_cst)} rows ({int(pool_inside.sum())} inside the window, "
          f"{int((~pool_inside).sum())} edge probe).")

    # === Stage 2: shortlist === #
    unique = {}

    def add_pool_row(i, group, rank, score):
        add_candidate(unique, pool['features'][i], pool['config'][i], pool['preds'][i], str(pool['origin'][i]),
                      group, rank, score, {'round': int(pool['round'][i])})

    for m, name in enumerate(model_names):
        for rank, i in enumerate(diverse_top(pool['preds'][:, m], pool_scaled, TOP_K_PER_MODEL, pool_inside), start=1):
            add_pool_row(i, f"model:{name}", rank, pool['preds'][i, m])
    consensus_picks = diverse_top(pool_consensus, pool_scaled, TOP_K_CONSENSUS, pool_inside)
    for rank, i in enumerate(consensus_picks, start=1):
        add_pool_row(i, GROUP_CONSENSUS, rank, pool_consensus[i])
    for rank, i in enumerate(diverse_top(pool_consensus, pool_scaled, TOP_K_EDGE, ~pool_inside), start=1):
        add_pool_row(i, GROUP_EDGE, rank, pool_consensus[i])

    # Configuration check: the best consensus shapes in every configuration
    # Rows that are not in the pool yet; each entry is (features, config key, group, rank, origin, extra fields)
    extra = []
    for shape_rank, i in enumerate(consensus_picks[:CONFIG_CHECK_SHAPES], start=1):
        for col in config_cols:
            extra.append((with_config(pool['features'][i], config_cols, feature_columns, col), col, GROUP_CONFIG,
                          shape_rank, str(pool['origin'][i]), {'round': int(pool['round'][i])}))
    # Reproducibility reference: the top dataset designs exactly as they are in the dataset
    for ref_rank, (cst, config_key, e_dataset) in enumerate(info.reference, start=1):
        row = np.zeros(len(feature_columns), dtype=np.float32)
        row[cst_col_idx] = cst
        if config_cols:
            row = with_config(row, config_cols, feature_columns, config_key)
        extra.append((row, config_key, GROUP_REFERENCE, ref_rank, 'dataset', {'dataset_E': e_dataset}))
    if extra:
        extra_preds = predict_all(regressors, np.array([e[0] for e in extra], dtype=np.float32))
        for (row, cfg, group, rank, origin, fields), p in zip(extra, extra_preds):
            score = None if group == GROUP_REFERENCE else consensus_score(p[None])[0]
            add_candidate(unique, row, cfg, p, origin, group, rank, score, fields)

    candidates = list(unique.values())
    cand_cst = np.array([c['features'][cst_col_idx] for c in candidates])
    cand_dist, cand_near_e, cand_region = tag_regions(cand_cst, np.array([c['config_col'] for c in candidates]), info, lows, highs)
    for c, d, ne, reg in zip(candidates, cand_dist, cand_near_e, cand_region):
        c.update({'consensus': float(consensus_score(c['preds'][None])[0]), 'grid_distance': float(d),
                  'nearest_dataset_E': float(ne), 'region': str(reg)})
    total_sources = sum(len(c['sources']) for c in candidates)
    print(f"Shortlist: {total_sources} picks -> {len(candidates)} unique candidates "
          f"({total_sources - len(candidates)} repeated feature vectors merged).")

    predictions_export_path = run_dir / f"{dataset_name_prefix}_predictions_only.json"
    write_json(predictions_export_path, {
        c['candidate_id']: {
            "cst_coeffs": c['features'][cst_col_idx].tolist(),
            "wing_config": config_registry[c['config_col']][1],
            "predictions": dict(zip(model_names, c['preds'].tolist())),
            "consensus": c['consensus'],
            "region": c['region'], "grid_distance": c['grid_distance'], "nearest_dataset_E": c['nearest_dataset_E'],
            "origin": c['origin'],
            "sources": c['sources'],
        } for c in candidates
    })
    print(f"=> Initial predictions saved to {predictions_export_path.name}")

    # === Stage 3: main simulation === #
    ctx = SimpleNamespace(paths=paths, n_cst=n_cst, cst_col_idx=cst_col_idx, config_registry=config_registry,
                          run_dir=run_dir, prefix=dataset_name_prefix, archiver=compression_Alma(),
                          airfoil_creator=Airfoil_Generator(base_airfoil_csv=paths.base_airfoil, n_cst=n_cst, verbose=False),
                          dataset_num=dataset_num, model_names=model_names, config_order=info.config_order)
    t_main = time.perf_counter()
    main_results = simulate_candidates(candidates, "main", ctx)
    main_seconds = time.perf_counter() - t_main
    metadata['stage_times']['main_simulation'] = str(timedelta(seconds=round(main_seconds)))
    for c in candidates:
        res = main_results[c['candidate_id']]
        c.update({'status': res['status'], 'simulated_CL': res['CL'], 'simulated_CD': res['CD'], 'simulated_E': res['E']})
    write_json(metadata_path, metadata)

    # === Stage 4: robustness === #
    robustness = {'status': 'not run', 'parents': []}
    seconds_per_sim = main_seconds / max(len(candidates), 1)
    ranked = sorted((c for c in candidates if c['status'] == 'ok'), key=lambda c: c['simulated_E'], reverse=True)
    reference_best = next((c for c in candidates if any(s['group'] == GROUP_REFERENCE and s['rank'] == 1 for s in c['sources'])
                           and c['status'] == 'ok'), None)
    parents, seen = [], set()
    for c in ([reference_best] if reference_best else []) + ranked:
        key = c['features'][cst_col_idx].tobytes()
        if key not in seen:
            seen.add(key)
            parents.append(c)
    n_designs = min(ROBUST_DESIGNS, len(parents))
    per_design = ROBUST_PERTURBATIONS
    print(f"\n[TIME] Main simulation: {len(candidates)} candidates in {timedelta(seconds=round(main_seconds))} "
          f"(~{seconds_per_sim:.0f}s each). Robustness: {n_designs * per_design} simulations "
          f"(about {timedelta(seconds=round(n_designs * per_design * seconds_per_sim))}).")
    if not parents:
        robustness['status'] = 'no successfully simulated designs to perturb'
    else:
        print(f"Robustness: {n_designs} design(s) x {per_design} perturbation(s), "
              f"std dev {ROBUST_SIGMA_STEPS} grid steps.")
        rng = np.random.default_rng([master_seed, 1])
        rob_candidates = []
        for p_idx, parent in enumerate(parents[:n_designs], start=1):
            noise = rng.normal(0.0, ROBUST_SIGMA_STEPS, size=(per_design, len(cst_col_idx))) * info.steps
            for j in range(per_design):
                row = parent['features'].copy()
                row[cst_col_idx] = (parent['features'][cst_col_idx] + noise[j]).astype(np.float32)
                rob_candidates.append({'candidate_id': f"robust_{p_idx:02d}_{j + 1:02d}", 'features': row,
                                       'config_col': parent['config_col'], 'parent': parent['candidate_id']})
        rob_preds = predict_all(regressors, np.array([c['features'] for c in rob_candidates]))
        t_rob = time.perf_counter()
        rob_results = simulate_candidates(rob_candidates, "robustness", ctx)
        metadata['stage_times']['robustness_simulation'] = str(timedelta(seconds=round(time.perf_counter() - t_rob)))
        by_parent = defaultdict(list)
        for c, p in zip(rob_candidates, rob_preds):
            res = rob_results[c['candidate_id']]
            by_parent[c['parent']].append({'candidate_id': c['candidate_id'], 'status': res['status'], 'simulated_E': res['E'],
                                           'cst_coeffs': c['features'][cst_col_idx], 'predictions': dict(zip(model_names, p))})
        for parent in parents[:n_designs]:
            kids = by_parent[parent['candidate_id']]
            e = np.array([k['simulated_E'] for k in kids if k['status'] == 'ok'])
            robustness['parents'].append({
                'parent': parent['candidate_id'], 'parent_E': parent['simulated_E'],
                'parent_groups': short_sources(parent['sources']),
                'perturbations': len(kids), 'simulated': len(e),
                'hold_rate': float((e >= ROBUST_HOLD_FRACTION * parent['simulated_E']).mean()) if len(e) else None,
                'mean_E': float(e.mean()) if len(e) else None, 'min_E': float(e.min()) if len(e) else None,
                'max_E': float(e.max()) if len(e) else None, 'members': kids,
            })
        robustness['status'] = 'completed'

    # === Stage 5: reports === #
    print("\n--- Writing Reports ---")
    e_best = info.best['E']
    groups = [f"model:{n}" for n in model_names] + [GROUP_CONSENSUS, GROUP_EDGE, GROUP_CONFIG, GROUP_REFERENCE]
    scorecard = build_scorecard(candidates, groups, model_names, e_best, info.e_max)
    model_table = build_model_table(candidates, model_names, info.e_max)

    order = [config_registry[c][1] for c in info.config_order]
    model_winners = []
    for m, name in enumerate(model_names):
        winner = next((c for c in candidates for s in c['sources'] if s['group'] == f"model:{name}" and s['rank'] == 1), None)
        if winner is None or not config_cols:
            continue
        rows = np.array([with_config(winner['features'], config_cols, feature_columns, col) for col in info.config_order])
        p = predict_all({name: regressors[name]}, rows)[:, 0]
        model_winners.append({'model': name, 'candidate_id': winner['candidate_id'],
                              'picked_config': config_registry[winner['config_col']][1],
                              'predictions': dict(zip(order, p)),
                              'order_matches_dataset': bool(np.all(np.diff(p) < 0))})
    simulated_shapes = []
    for shape_rank in range(1, CONFIG_CHECK_SHAPES + 1):
        members = [c for c in candidates if any(s['group'] == GROUP_CONFIG and s['rank'] == shape_rank for s in c['sources'])]
        if not members:
            continue
        by_cfg = {config_registry[c['config_col']][1]: c['simulated_E'] for c in members}
        e = [by_cfg.get(o, np.nan) for o in order]
        complete = all(math.isfinite(x) for x in e)
        simulated_shapes.append({'shape': shape_rank, 'candidates': [c['candidate_id'] for c in members],
                                 'simulated_E': {o: by_cfg.get(o, np.nan) for o in order},
                                 'order_matches_dataset': bool(np.all(np.diff(e) < 0)) if complete else None})

    report_candidates = [{
        'candidate_id': c['candidate_id'],
        'wing_config': config_registry[c['config_col']][1],
        'cst_coeffs': c['features'][cst_col_idx],
        'origin': c['origin'],
        'region': c['region'], 'grid_distance': c['grid_distance'], 'nearest_dataset_E': c['nearest_dataset_E'],
        'dataset_E': c.get('dataset_E'),
        'predictions': dict(zip(model_names, c['preds'])),
        'consensus': c['consensus'],
        'spread': float(np.ptp(c['preds'])) if np.isfinite(c['preds']).all() else np.nan,
        'above_dataset_max': bool(np.nanmax(c['preds']) > info.e_max),
        'status': c['status'],
        'simulated_CL': c['simulated_CL'], 'simulated_CD': c['simulated_CD'], 'simulated_E': c['simulated_E'],
        'consensus_error': c['consensus'] - c['simulated_E'] if c['status'] == 'ok' else np.nan,
        'sources': c['sources'],
    } for c in candidates]

    reference_best = next((c for c in candidates if any(s['group'] == GROUP_REFERENCE and s['rank'] == 1 for s in c['sources'])
                           and c['status'] == 'ok'), None)
    outcome = find_outcome(report_candidates, model_names, e_best,
                           reference_best['simulated_E'] if reference_best else None)
    best_found = max((c for c in report_candidates if c['status'] == 'ok' and math.isfinite(c['simulated_E'])
                      and not all(s['group'] == GROUP_REFERENCE for s in c['sources']) and c['grid_distance'] >= 1e-3),
                     key=lambda c: c['simulated_E'], default=None)

    end_time = datetime.now()
    report = to_json({
        'outcome': {'success': outcome['success'], 'target_E': outcome['target_E'],
                    'dataset_best_E': outcome['dataset_best_E'],
                    'reference_best_simulated_E': outcome['reference_best_simulated_E'],
                    'improvement_ids': [c['candidate_id'] for c in outcome['improvements']],
                    'best_per_model': outcome['best_per_model']},
        'dataset_best': info.best,
        'rounds_completed': rounds_completed,
        'total_time': str(end_time - start_time).split('.')[0],
        'scorecard': scorecard,
        'model_table': model_table,
        'config_check': {'dataset_order': order, 'model_winners': model_winners, 'simulated_shapes': simulated_shapes},
        'candidates': report_candidates,
        'robustness': robustness,
    })
    json_report_path = run_dir / "detailed_results.json"
    write_json(json_report_path, report)
    print(f"=> JSON detailed results saved to {json_report_path.name}")
    save_text_report(run_dir / "detailed_results.txt", ctx, report)
    robust_by_parent = {p['parent']: p for p in robustness['parents']}
    save_result_report(run_dir / "result.txt", ctx, outcome, best_found, robust_by_parent,
                       {'folder': run_dir.name, 'seed': f"0x{master_seed:x}", 'total_time': report['total_time']})
    save_parity_plot(run_dir / f"{dataset_name_prefix}_parity.png", candidates, model_names, e_best)

    metadata.update({
        "run_status": "completed",
        "finished": end_time.strftime('%Y-%m-%d %H:%M:%S'),
        "total_execution_time": str(end_time - start_time),
        "results": {"pool_rows": len(pool_cst), "unique_candidates": len(candidates),
                    "simulated_candidates": sum(c['status'] == 'ok' for c in candidates),
                    "robustness_simulations": sum(len(p['members']) for p in robustness['parents'])},
        "outcome": "SUCCESS!!" if outcome['success'] else "Try again...",
    })
    write_json(metadata_path, metadata)

    print(f"\nProcess completed successfully at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total execution time: {end_time - start_time}")
