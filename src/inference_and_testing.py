"""
Iterative inference and simulation-based validation of the trained AI models (trust-region search).

Usage:
    python inference_and_testing.py --classifier <name> --models <name1> [<name2> ...]
                                    [--dataset 17.5] [--lane NAME] [--no-update] [--hours 24] [--workers N]
                                    [--wing-mode Hossain_config] [--seed HEX [--decimal]] [--csv PATH]
    python inference_and_testing.py --resume <run folder> [--dataset 17.5] [--hours H] [--workers N]

Lanes (searches that run side by side, each with its own copies of the models, its own tracks and its own data):
    one lane per regressor given (e.g. Agares = LGBM, Odysseus = MLP raw, Murata = MLP sum_station), 'consensus'
    (all regressors, ranked by their most pessimistic prediction) and 'random' (the control: random picks inside the
    same trust regions, no models). --lane NAME runs a single lane. --no-update freezes the models for the whole run
    (the ablation that shows what the fine-tuning adds; such a run writes no new dataset/model versions).

Flow:
    Iteration 0 (shared by every lane): seeded rounds of uniform, local (around the best dataset designs) and edge-probe
        (just past the window edges where the best designs sit) samples; every regressor scores every shape in every
        wing configuration. Each lane shortlists its own best shapes; the top dataset shapes and the dataset peaks
        are re-simulated as references. Every shortlisted shape is simulated in all wing configurations.
    Tracks: every lane starts one track (trust region) at each dataset peak (ai_support.trust_region.Dataset_Peaks);
        iteration-0 finds move a track or open a new one. After iteration 0 each lane fine-tunes its models.
    Iterations 1, 2, ...: per lane and track, sample the trust region, score with the lane's current models, pick
        diverse candidates (check_validity gate), simulate them all in one parallel batch, update every track (move /
        resize, ai_support.trust_region.Trust_Region), drop or merge tracks, fine-tune the lane's models with all of
        its simulations so far. A lane stops when its best E gains less than 0.5% for 2 iterations in a row.
    Time ceiling (--hours, default 24): checked only between iterations, so an iteration is never cut (the run may
        overshoot); the report says whether another iteration was still needed, or how much earlier the run finished.
    Final validation per lane: cold-start sweep check of the best designs (history-free XFOIL, SUCCESS needs it to
        agree), robustness perturbations, geometric filter status (information only), window status, models' view.
    Versions (not with --no-update): after i loop iterations the dataset and the models get a third version part:
        data/datasets/dataset_17.5/dataset_17.5.i/regressor_dataset_17.5.i.csv  (dataset + every valid new simulation
        of every lane that passes the laminar CD limit; no geometric filters, no classifier CSV) and
        models/dataset_17.5/dataset_17.5.i/<type folders>/<Name>_1.5.i_* (one final fine-tune of each model on it).

Recovery (power cuts, accidental shutdowns): every finished simulation is appended to simulations.jsonl and forced
to disk at once; the run state (lanes, tracks, picks, stop counters) is written atomically at every phase of every
iteration (recovery_status.json). --resume continues where the run stopped: finished simulations are reused, the
interrupted batch simulates only what is missing, and fine-tuned models are rebuilt from the saved simulations.

Outputs (results/dataset_<N>/<date>_m_seed<seed>): result.txt (SUCCESS!! / Try again..., per lane), detailed_results.txt
and .json, iterations.csv, simulations.csv, simulations.jsonl (every simulation with its polar, key points, .avl and
force files), convergence and parity plots, run metadata.

Simulations run in a process pool (simulation_support.simulate_shapes, --workers, default CPU threads - 1).
"""
import argparse
import contextlib
import io
import hashlib
import json
import math
import multiprocessing
import os
import platform
import re
import secrets
import shutil
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from cst_modeling.foil import check_validity

from simulation_support import Simulation_Settings, simulate_shapes, compare_runs, laminar_cd_limit, jsonable
from geometric_support import Airfoil_Generator, check_airfoil
from file_handling_support.dataset_names import (dataset_number_arg, dataset_csv, models_folder, parent_dataset,
                                                 expansion_number, dataset_folder)
from file_handling_support.model_names import base_name, expanded_model_name, model_version
from ai_support import Classifier, MLP_Regressor, LGBM_Regressor, Dataset_Peaks, Trust_Region, Lane_Monitor, diverse_top
from ux_support.recovery import atomic_write_json, atomic_write_text, append_jsonl, read_jsonl

# LightGBM models saved with feature names warn on every prediction from a plain array; the columns are in order
warnings.filterwarnings("ignore", message="X does not have valid feature names")

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

# --- Iteration 0 (shared candidate search) --- #
NUM_INFERENCE_RUNS = 5         # seeded rounds
NUM_SAMPLES = 400_000          # uniform (global) CST samples per round
NUM_LOCAL_SAMPLES = 600_000    # samples per round around the best dataset designs (0 = off)
NUM_EDGE_SAMPLES = 300_000     # edge-probe samples per round (0 = off); only shapes outside the window are kept
LOCAL_TOP_K = 30               # distinct top-E dataset designs used as local search centres
LOCAL_SIGMA_FRACTION = 0.10    # local noise std dev, as a fraction of each coefficient's window width
EDGE_SIGMA_STEPS = 0.5         # edge-probe noise std dev on the extended coefficients, in grid steps
EDGE_MIN_SHARE = 0.5           # a window edge is probed if at least this share of the top dataset designs sit on it
# Rows per model call; keeps GPU/CPU memory bounded. A chunk close to the free GPU memory can print a harmless
# out-of-memory WARNING from CUDACachingAllocator.cpp; only an OutOfMemoryError calls for a smaller value.
PREDICT_CHUNK = 250_000
SEED_POOL_SIZE = 10000         # round seeds are drawn (without repetition) from [0, SEED_POOL_SIZE)
MAX_MASTER_SEED = 0xFFFFFFFF   # upper limit of the master seed (8 hex digits)
ELITE_PER_LANE = 2000          # shapes kept per round and lane (inside the window; a quarter of it from the edge probe)
RANDOM_POOL_PER_ROUND = 400    # shapes kept per round for the random lane's iteration-0 picks
TOP_K_IN_WINDOW = 16           # iteration-0 picks per lane inside the window
TOP_K_EDGE = 8                 # iteration-0 picks per lane from the edge probe
RANDOM_PICKS_0 = 24            # iteration-0 picks of the random lane
REFERENCE_TOP = 5              # top dataset shapes re-simulated as they are (reproducibility check)
DIVERSITY_MIN_STEPS = 0.5      # iteration-0 picks differ by at least this many grid steps in some coefficient
VALIDITY_THRESHOLD = 0.99      # classifier probability shown as "classifier: valid" in the reports (not a gate)

# --- Trust-region loop --- #
PEAK_SHARE = 0.90              # dataset peaks: local maxima with E >= this share of their configuration's best E
MAX_START_TRACKS = 6           # at most this many dataset peaks start tracks
MAX_TRACKS_FACTOR = 2          # a lane follows at most this many times the number of dataset peaks
NEW_TRACK_MIN_STEPS = 2.0      # an iteration-0 design opens a new track this far (grid steps) from every track
TR_UNIFORM_SAMPLES = 20_000    # uniform samples per track and iteration
TR_REFINE_SAMPLES = 10_000     # extra samples around the best-predicted ones
TR_REFINE_TOP = 50
PICKS_PER_TRACK = 16           # shapes simulated per track and iteration
PICK_MIN_STEPS_FACTOR = 0.25   # picks differ by at least this x radius grid steps
EXPLORE_TOP_SHARE = 0.10       # consensus lane: exploration picks come from this top share by consensus
MIN_GAIN = 0.005               # stop rule: best E gains less than this (relative) ...
STALL_PATIENCE = 2             # ... this many iterations in a row
DROP_SHARE = 0.95              # a track below this share of its lane's best E is dropped (from iteration 2)
MERGE_STEPS = 1.0              # two tracks closer than this (grid steps) merge
DEFAULT_HOURS = 24.0
MAX_ITERATIONS = None          # optional hard cap on loop iterations (None: time ceiling and stop rule only)

# --- Fine-tuning (ai_support.models fine_tune) --- #
FT_NEW_WEIGHT = 10             # weight of a new simulation row against one dataset (anchor) row
FT_MIN_SHAPES = 5              # fewer new shapes than this: no update
FT_LOCAL_ANCHOR_STEPS = 1.5    # dataset rows this close (grid steps) to a track centre are anchors ...
FT_LOCAL_ANCHOR_MAX = 30_000   # ... at most this many
FT_RANDOM_ANCHORS = 20_000     # plus this many random dataset rows (keeps the rest of the space)

# --- Final validation --- #
VALIDATION_TOP = 3             # best designs per lane checked with a cold-start (history-free) simulation
ROBUST_DESIGNS = 3             # best designs per lane that get perturbed
ROBUST_PERTURBATIONS = 8
ROBUST_SIGMA_STEPS = 0.10      # perturbation std dev, in grid steps
ROBUST_HOLD_FRACTION = 0.9     # a perturbation "holds" at >= this fraction of its parent's E
REACH_SHARES = (1.00, 1.01, 1.02, 1.05)   # "simulations to reach this share of the dataset best E"

# --- Search window of iteration 0 (CST deltas; dataset 17 / 17.5 generation ranges) --- #
RANGES_UPPER = {'LE': (0, 0.125, 0.0625), 'inter': (0, 0.125, 1/24), 'TE': (0, 0.125, 0.125)}
RANGES_LOWER = {'LE': (-0.125, 0, 0.0625), 'inter': (0, 0.125, 1/24), 'TE': (0, 0.125, 0.125)}

# --- Files --- #
SIMULATIONS_FILE = "simulations.jsonl"
STATE_FILE = "recovery_status.json"
SHORTLIST_FILE = "inference_checkpoint.json"

# --- Wing configuration modes --- #
HOSSAIN_MODE = "Hossain_config"
SINGLE_MODE = "single_configuration"
HOSSAIN_IDENTIFIERS = ["rec", "flat", "60DEG"]

# Regressor type per model folder prefix (folders: mlp_regressor, mlp_regressor_sum_station, ...)
REGRESSOR_FOLDER_TYPES = {'mlp_regressor': 'mlp', 'lgbm_regressor': 'lgbm', 'sknn_regressor': 'sknn'}
TYPE_LABELS = {'mlp': 'MLP', 'lgbm': 'LGBM', 'sknn': 'SKNN'}
LANE_CONSENSUS = "consensus"
LANE_RANDOM = "random"

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
    parser = argparse.ArgumentParser(description="Iterative inference (trust-region search) and simulation-based validation "
                                                 "of trained AI models.")
    parser.add_argument('--dataset', type=dataset_number_arg, default="17.5", help="Dataset number, e.g. 17.5 (default: 17.5).")
    parser.add_argument('--wing-mode', choices=[HOSSAIN_MODE, SINGLE_MODE], default=None,
                        help=f"'{HOSSAIN_MODE}': one-hot rec/flat/60DEG configurations (default).")
    parser.add_argument('--classifier', default=None,
                        help="Classifier name (folder 'classifier'); its validity probability is reported, it filters "
                             "nothing. Required unless --resume is used.")
    parser.add_argument('--models', nargs='+', default=None,
                        help="Regressors, e.g. --models Agares_1.5 Odysseus_1.5 Murata_1.5. Each gets its own lane, plus "
                             "the consensus and random lanes. Required unless --resume is used.")
    parser.add_argument('--lane', default=None,
                        help=f"Run only this lane: a model name from --models, '{LANE_CONSENSUS}' or '{LANE_RANDOM}'.")
    parser.add_argument('--no-update', action='store_true',
                        help="Freeze the models for the whole run (no fine-tuning; no new dataset/model versions).")
    parser.add_argument('--hours', type=float, default=None,
                        help=f"Time ceiling in hours (default {DEFAULT_HOURS:g}). Checked between iterations only: an "
                             f"iteration is never cut, so the run can overshoot.")
    parser.add_argument('--workers', type=int, default=None,
                        help="Parallel simulation processes (default: CPU threads - 1).")
    parser.add_argument('--seed', type=str, default=None,
                        help=f"Master seed, hexadecimal by default (0 to {MAX_MASTER_SEED:x}). Every random choice of the "
                             f"run derives from it.")
    parser.add_argument('--decimal', action='store_true', help="Interpret --seed as a decimal number.")
    parser.add_argument('--csv', default=None,
                        help="Regressor dataset CSV (default: the dataset's regressor CSV). Use the rectified CSV when "
                             "the models were trained on it.")
    parser.add_argument('--resume', default=None, metavar='RUN_FOLDER',
                        help="Continue a stopped run in its own folder (a folder name under results/dataset_<N> or a "
                             "path). Its settings come from its metadata; only --hours and --workers may be given.")
    args = parser.parse_args()
    if args.decimal and args.seed is None:
        parser.error("--decimal only makes sense together with --seed.")
    if args.resume:
        given = [flag for flag, value in (('--classifier', args.classifier), ('--models', args.models), ('--seed', args.seed),
                                          ('--csv', args.csv), ('--wing-mode', args.wing_mode), ('--lane', args.lane))
                 if value is not None] + (['--no-update'] if args.no_update else [])
        if given:
            parser.error(f"--resume takes its settings from the earlier run; remove {', '.join(given)}.")
        return args
    if args.classifier is None or args.models is None:
        parser.error("--classifier and --models are required (unless --resume is used).")
    if len(set(args.models)) != len(args.models):
        parser.error("Duplicate names in --models.")
    if args.lane is not None and args.lane not in args.models + [LANE_CONSENSUS, LANE_RANDOM]:
        parser.error(f"--lane must be one of {args.models + [LANE_CONSENSUS, LANE_RANDOM]}.")
    args.wing_mode = args.wing_mode or HOSSAIN_MODE
    if args.seed is not None:
        try:
            args.seed = parse_master_seed(args.seed, args.decimal)
        except ValueError as e:
            parser.error(str(e))
    return args


def build_paths(dataset_num: str) -> SimpleNamespace:
    project_root = Path(__file__).resolve().parent.parent
    binaries_dir = project_root / "bin"
    exe_ext = ".exe" if sys.platform == "win32" else ""
    support_files_dir = project_root / "data" / "support"
    models_dir = models_folder(dataset_num, project_root)
    classifier_dir = models_dir / "classifier"
    if not classifier_dir.is_dir() and parent_dataset(dataset_num):   # expansions reuse their parent's classifier
        classifier_dir = models_folder(parent_dataset(dataset_num), project_root) / "classifier"
    return SimpleNamespace(
        project_root=project_root,
        training_file=dataset_csv(dataset_num, project_root),
        case_file=support_files_dir / "Hossain_sqwing_case.run",
        base_airfoil=support_files_dir / "naca0012_unit.csv",
        naca_kp_file=support_files_dir / "naca_kp" / NACA_KP_FILE,
        xfoil_bin=binaries_dir / f"xfoil{exe_ext}",
        avl_bin=binaries_dir / f"avl{exe_ext}",
        models_dir=models_dir,
        classifier_dir=classifier_dir,
        results_root=project_root / "results" / f"dataset_{dataset_num}",
    )


def temp_root() -> Path:
    """Short, fixed temp path for the simulation workers (XFOIL and AVL do not tolerate long paths)."""
    return Path("C:/inference_temp") if platform.system() == "Windows" else Path.home() / ".inference_temp"


def make_run_dir(results_root: Path, master_seed: int) -> Path:
    """results/dataset_N/<YYYYMMDD>_m_seed<master seed in hex>; a numeric suffix avoids overwriting."""
    base = f"{datetime.now().strftime('%Y%m%d')}_m_seed{master_seed:x}"
    run_dir, n = results_root / base, 1
    while run_dir.exists():
        n += 1
        run_dir = results_root / f"{base}_{n}"
    run_dir.mkdir(parents=True)
    return run_dir


def metadata_path_of(run_dir: Path, dataset_num: str) -> Path:
    return run_dir / f"AI_Test_Results_dataset_{dataset_num}_run_metadata.json"


def load_previous_run(args):
    """--resume: the earlier run folder and its metadata; its settings are copied into args."""
    folder = Path(args.resume)
    if not folder.is_dir():
        folder = build_paths(args.dataset).results_root / args.resume
    metadata_files = sorted(folder.glob("*_run_metadata.json")) if folder.is_dir() else []
    if not metadata_files:
        sys.exit(f"[ERROR] --resume: no run metadata found in '{folder}'.")
    previous = json.loads(metadata_files[0].read_text())
    if str(previous.get("run_status", "")).startswith("completed"):
        sys.exit(f"[ERROR] --resume: run '{folder.name}' is already completed. Start a new run instead.")
    if "lanes" not in previous:
        sys.exit("[ERROR] --resume: this run was made by the one-shot inference (before the iterative version); "
                 "it cannot be continued by this script. Start a new run.")
    args.dataset = str(previous['dataset'])
    args.wing_mode = previous['wing_mode']
    args.classifier = previous['classifier']['name']
    args.models = [m['name'] for m in previous['models']] + [m['name'] for m in previous.get('skipped_models', [])]
    args.seed = int(previous['master_seed']['decimal'])
    args.csv = previous['dataset_csv']
    args.lane = previous.get('lane_option')
    args.no_update = bool(previous.get('no_update', False))
    args.hours = args.hours if args.hours is not None else previous.get('hours', DEFAULT_HOURS)
    args.workers = args.workers if args.workers is not None else previous.get('workers')
    print(f"Resuming run '{folder.name}' (last status: {previous.get('run_status', 'unknown')})")
    return folder.resolve(), previous


def report_unfinished_runs(results_root: Path, dataset_num: str):
    """Points at earlier runs of this dataset that did not finish (they can be continued with --resume)."""
    if not results_root.is_dir():
        return
    unfinished = []
    for meta in sorted(results_root.glob("*/AI_Test_Results_dataset_*_run_metadata.json")):
        try:
            data = json.loads(meta.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if "lanes" in data and not str(data.get('run_status', '')).startswith("completed"):
            unfinished.append(meta.parent.name)
    if unfinished:
        print(f"[NOTE] Unfinished run(s) of dataset {dataset_num}: {', '.join(unfinished)}. To continue one instead:")
        print(f"       python src/inference_and_testing.py --resume {unfinished[-1]} --dataset {dataset_num}")


def file_fingerprint(files) -> str:
    """Short SHA-256 of the given files (names and contents); tells whether a model or a dataset changed."""
    digest = hashlib.sha256()
    for path in sorted(Path(f) for f in files):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def model_fingerprint(folder: Path, name: str) -> str:
    return file_fingerprint(p for p in folder.glob(f"{name}_*") if p.is_file())


def check_resume_fingerprints(previous: dict, current: dict):
    """Stops a resume if the dataset CSV or a model changed since the run started."""
    old = previous.get('fingerprints', {})
    pairs = [("the dataset CSV", old.get('dataset_csv'), current['dataset_csv']),
             (f"classifier {previous['classifier']['name']}", old.get('classifier'), current['classifier'])]
    pairs += [(f"model {n}", old.get('models', {}).get(n), f) for n, f in current['models'].items()]
    changed = [label for label, before, now in pairs if before != now]
    if changed:
        sys.exit(f"[ERROR] Cannot resume: {', '.join(changed)} changed since this run started (retrained or "
                 f"regenerated?), so its picks and fine-tuned models no longer apply. Start a new run instead.")


def write_json(path: Path, data):
    atomic_write_json(path, jsonable(data), indent=4)


def rng_for(master_seed: int, *parts) -> np.random.Generator:
    """Independent, reproducible random stream for one purpose (e.g. lane 2, iteration 5, track 1)."""
    return np.random.default_rng([int(master_seed)] + [int(p) for p in parts])


def seed_for(master_seed: int, *parts) -> int:
    return int(np.random.SeedSequence([int(master_seed)] + [int(p) for p in parts]).generate_state(1)[0])


def hms(seconds: float) -> str:
    """Duration as H:MM:SS (hours keep counting past 24)."""
    total = int(round(max(seconds, 0)))
    return f"{total // 3600}:{total % 3600 // 60:02d}:{total % 60:02d}"


# =============================================================================================== #
# === Dataset === #
# =============================================================================================== #

def parse_dataset_columns(csv_path: Path, wing_mode: str):
    """(feature columns, cst_l columns, cst_u columns, config columns) from the dataset header (last column = E)."""
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    feature_columns = header[:-1]
    cst_l_cols = sorted((c for c in feature_columns if re.fullmatch(r'cst_l_\d+', c)), key=lambda c: int(c.rsplit('_', 1)[1]))
    cst_u_cols = sorted((c for c in feature_columns if re.fullmatch(r'cst_u_\d+', c)), key=lambda c: int(c.rsplit('_', 1)[1]))
    if not cst_l_cols or len(cst_l_cols) != len(cst_u_cols):
        raise ValueError(f"Invalid CST columns in '{csv_path.name}': {len(cst_l_cols)} lower vs {len(cst_u_cols)} upper.")
    config_cols = [c for c in feature_columns if c not in cst_l_cols and c not in cst_u_cols]
    if wing_mode == HOSSAIN_MODE and not config_cols:
        raise ValueError(f"Wing mode '{HOSSAIN_MODE}' needs hot-encoded configuration columns, none found in '{csv_path.name}'.")
    return feature_columns, cst_l_cols, cst_u_cols, config_cols


def config_identifiers(config_cols: list) -> dict:
    """{config column: wing configuration identifier ('rec', 'flat', '60DEG')}, matched by name."""
    mapping = {}
    for col in config_cols:
        matches = [ident for ident in HOSSAIN_IDENTIFIERS if ident.lower() in col.lower()]
        if len(matches) != 1:
            raise ValueError(f"Could not uniquely match dataset column '{col}' to a wing configuration {HOSSAIN_IDENTIFIERS}.")
        mapping[col] = matches[0]
    return mapping


def load_dataset(csv_path: Path, feature_columns, cst_cols, config_cols) -> SimpleNamespace:
    """
    The dataset in memory: feature rows X, E values y, CST block, configuration of every row, grid steps, best row,
    local-search centres (distinct top-E shapes), reference shapes, nearest-shape lookup trees, configuration order.
    """
    df = pd.read_csv(csv_path)
    target = df.columns[-1]
    X = df[feature_columns].to_numpy(np.float32)
    y = df[target].to_numpy(np.float64)
    cst = df[cst_cols].to_numpy(np.float64)
    configs = np.array(config_cols)[df[config_cols].to_numpy().argmax(axis=1)]
    steps = []
    for col in cst_cols:
        u = np.unique(np.round(df[col].to_numpy(), 6))
        steps.append(float(np.min(np.diff(u))) if len(u) > 1 else 1.0)
    steps = np.array(steps, dtype=np.float64)
    order = np.argsort(-y)
    best = order[0]
    seen, parents, references = set(), [], []
    for i in order:
        key = shape_key(cst[i])
        if key in seen:
            continue
        seen.add(key)
        if len(parents) < LOCAL_TOP_K:
            parents.append(cst[i])
        if len(references) < REFERENCE_TOP:
            references.append((cst[i], configs[i], float(y[i])))
        if len(parents) >= LOCAL_TOP_K and len(references) >= REFERENCE_TOP:
            break
    trees = {key: (cKDTree(cst[configs == key] / steps), y[configs == key]) for key in config_cols}
    pivot = df.assign(_config=configs).pivot_table(index=cst_cols, columns='_config', values=target).dropna()
    config_order = list(pivot.mean().sort_values(ascending=False).index)
    return SimpleNamespace(X=X, y=y, cst=cst, cst_scaled=cst / steps, configs=configs, steps=steps,
                           best={"E": float(y[best]), "config": str(configs[best]), "cst": cst[best].tolist()},
                           e_max=float(y[best]), parents=np.array(parents, dtype=np.float32), references=references,
                           trees=trees, all_tree=cKDTree(np.unique(cst / steps, axis=0)), config_order=config_order)


# =============================================================================================== #
# === Shapes, sampling and scoring === #
# =============================================================================================== #

def shape_key(cst) -> str:
    return ",".join(f"{float(v):.6f}" for v in cst)


def shape_id(key: str) -> str:
    """Short file-name id of a shape (XFOIL reads at most 64 characters of a path)."""
    return "s" + hashlib.sha1(key.encode()).hexdigest()[:10]


def build_feature_ranges(n_cst, ranges_upper, ranges_lower):
    """Per-coefficient (low, high, step) ranges, lower block first, then upper block."""
    return ([ranges_lower["LE"]] + [ranges_lower["inter"]] * (n_cst - 2) + [ranges_lower["TE"]] +
            [ranges_upper["LE"]] + [ranges_upper["inter"]] * (n_cst - 2) + [ranges_upper["TE"]])


def window_bounds(n_cst):
    ranges = build_feature_ranges(n_cst, RANGES_UPPER, RANGES_LOWER)
    return (np.array([min(r[0], r[1]) for r in ranges], dtype=np.float32),
            np.array([max(r[0], r[1]) for r in ranges], dtype=np.float32))


def generate_uniform(num_samples, lows, highs, rng):
    return rng.uniform(lows, highs, size=(num_samples, len(lows))).astype(np.float32)


def generate_local(parents, num_samples, lows, highs, rng, clip_lows=None, clip_highs=None, sigma=None):
    """Gaussian perturbations around random parents, clipped to the window (or to clip_lows/clip_highs)."""
    clip_lows = lows if clip_lows is None else clip_lows
    clip_highs = highs if clip_highs is None else clip_highs
    sigma = (highs - lows) * LOCAL_SIGMA_FRACTION if sigma is None else sigma
    centres = parents[rng.integers(0, len(parents), size=num_samples)]
    noise = rng.normal(0.0, 1.0, size=centres.shape).astype(np.float32) * sigma
    return np.clip(centres + noise, clip_lows, clip_highs).astype(np.float32)


def detect_edge_directions(parents, lows, highs, steps):
    """Window edges shared by at least EDGE_MIN_SHARE of the top dataset designs; extended one grid step past them."""
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
    return ((cst_block < lows - 1e-7) | (cst_block > highs + 1e-7)).any(axis=1)


def predict_chunked(predict_fn, X, chunk_size=None):
    chunk_size = chunk_size or PREDICT_CHUNK
    parts = [np.asarray(predict_fn(X[i:i + chunk_size])).reshape(-1) for i in range(0, len(X), chunk_size)]
    return np.concatenate(parts) if parts else np.zeros(0)


def predict_all(models: dict, X) -> np.ndarray:
    """(rows, models) predictions; non-finite values become NaN."""
    preds = np.column_stack([predict_chunked(m.predict, X).astype(np.float64) for m in models.values()])
    preds[~np.isfinite(preds)] = np.nan
    return preds


def feature_rows(cst_block, ctx) -> np.ndarray:
    """Every shape in every configuration, rows ordered (shape, configuration), in dataset column order."""
    cst_block = np.atleast_2d(np.asarray(cst_block, dtype=np.float32))
    n, c = len(cst_block), ctx.n_configs
    rows = np.zeros((n * c, len(ctx.feature_columns)), dtype=np.float32)
    rows[:, ctx.cst_col_idx] = np.repeat(cst_block, c, axis=0)
    for j, col in enumerate(ctx.config_cols):
        rows[j::c, ctx.feature_columns.index(col)] = 1.0
    return rows


def feature_row(cst, config_col, ctx) -> np.ndarray:
    row = np.zeros(len(ctx.feature_columns), dtype=np.float32)
    row[ctx.cst_col_idx] = np.asarray(cst, dtype=np.float32)
    row[ctx.feature_columns.index(config_col)] = 1.0
    return row


def score_shapes(models: dict, cst_block, ctx) -> np.ndarray:
    """(shapes, configurations, models) predicted E."""
    cst_block = np.atleast_2d(cst_block)
    if len(cst_block) == 0:
        return np.zeros((0, ctx.n_configs, len(models)))
    return predict_all(models, feature_rows(cst_block, ctx)).reshape(len(cst_block), ctx.n_configs, len(models))


def lane_score(kind: str, preds: np.ndarray):
    """(score, best configuration index) of every shape: the lane's model in its best configuration, or for the
    consensus lane the most pessimistic model prediction in the best configuration."""
    per_config = np.min(preds, axis=2) if kind == LANE_CONSENSUS else preds[:, :, 0]
    per_config = np.where(np.isfinite(per_config), per_config, -np.inf)
    best_c = per_config.argmax(axis=1)
    score = per_config.max(axis=1)
    return np.where(np.isfinite(score), score, np.nan), best_c


def top_indices(score, k, mask=None):
    """Indices of the k highest finite scores (optionally restricted to mask), unordered."""
    s = np.where(np.isfinite(score), score, -np.inf)
    if mask is not None:
        s = np.where(mask, s, -np.inf)
    k = min(k, int(np.isfinite(s).sum()))
    return np.argpartition(-s, k - 1)[:k] if k > 0 else np.array([], dtype=int)


# =============================================================================================== #
# === Models === #
# =============================================================================================== #

def infer_hidden_layers(pth_path: Path) -> list:
    """Hidden layer sizes from a saved state dict (every 2-D '.weight' tensor is a Linear layer; the last is the output)."""
    state = torch.load(pth_path, map_location='cpu', weights_only=True)
    linears = [v for k, v in state.items() if k.endswith('.weight') and v.ndim == 2]
    if len(linears) < 2 or linears[-1].shape[0] != 1:
        raise ValueError(f"Unexpected network layout in '{pth_path.name}'; cannot infer hidden layers.")
    return [int(w.shape[0]) for w in linears[:-1]]


class ModelIncompatibleError(Exception):
    """The model was saved without the files this scheme needs (it is skipped, not fatal)."""


def read_parameters(folder: Path, name: str) -> dict:
    path = folder / f"{name}_regressor_model_parameters.json"
    if not path.exists():
        raise ModelIncompatibleError(f"Parameters file not found: {path}")
    return json.loads(path.read_text())


def list_regressor_names(models_dir: Path) -> list:
    names = []
    for sub in sorted(p for p in models_dir.iterdir() if p.is_dir()):
        if any(sub.name.startswith(prefix) for prefix in REGRESSOR_FOLDER_TYPES):
            names += [f.name[:-len('_meta.json')] for f in sub.glob('*_meta.json')]
    return sorted(names)


def resolve_regressor(name: str, models_dir: Path):
    """(folder, type) of regressor 'name'."""
    if not models_dir.is_dir():
        raise FileNotFoundError(f"Models folder does not exist: {models_dir}")
    matches = []
    for sub in sorted(p for p in models_dir.iterdir() if p.is_dir()):
        for prefix, mtype in REGRESSOR_FOLDER_TYPES.items():
            if sub.name.startswith(prefix):
                owns = any(sub.glob(f"{name}_*")) if mtype == 'sknn' else (sub / f"{name}_meta.json").exists()
                if owns:
                    matches.append((sub, mtype))
    if not matches:
        raise FileNotFoundError(f"No regressor named '{name}' under '{models_dir}'. Available: {list_regressor_names(models_dir)}")
    if len(matches) > 1:
        raise ValueError(f"Regressor name '{name}' is ambiguous, found in: {[str(m[0]) for m in matches]}")
    return matches[0]


def load_regressor(name: str, folder: Path, mtype: str):
    """The loaded regressor with its training hyperparameters (fine-tuning needs them), or None for SKNN. The model
    classes' own loading messages are silenced (every lane loads its own copies)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return _load_regressor(name, folder, mtype)


def _load_regressor(name: str, folder: Path, mtype: str):
    if mtype == 'sknn':
        return None
    params = read_parameters(folder, name)
    if mtype == 'mlp':
        hidden = infer_hidden_layers(folder / f"{name}_model.pth")
        if 'log_trans' not in params:
            raise ModelIncompatibleError(f"'log_trans' missing in {name}_regressor_model_parameters.json")
        if 'n_layers' in params:
            expected = [int(params[f'n_units_l{i}']) for i in range(int(params['n_layers']))]
            if expected != hidden:
                raise ValueError(f"Layer mismatch for '{name}': weights {hidden} vs parameters file {expected}.")
        model = MLP_Regressor.load_model(folder, name, hidden_layer_sizes=hidden,
                                         dropout_rate=float(params.get('dropout_rate', 0.0)), log_trans=bool(params['log_trans']),
                                         learning_rate=float(params.get('learning_rate', 1e-3)),
                                         batch_size=int(params.get('batch_size', 1024)), weight=params.get('weight', 'None'),
                                         patience=int(params.get('patience', 16)))
    elif mtype == 'lgbm':
        model = LGBM_Regressor.load_model(folder, name)
        for key in ('n_estimators', 'learning_rate', 'num_leaves', 'max_depth', 'objective', 'weight', 'lr_decay', 'stopping_rounds'):
            if key in params:
                setattr(model, key, params[key])
    else:
        raise ValueError(f"Unsupported model type '{mtype}'.")
    model.parameters_json = params
    return model


def load_classifier(name: str, folder: Path):
    pth = folder / f"{name}_model.pth"
    if not pth.exists():
        raise FileNotFoundError(f"Classifier '{name}' not found: {pth}")
    return Classifier.load_model(folder, name, hidden_layer_sizes=infer_hidden_layers(pth))


# =============================================================================================== #
# === Simulations (stored as they finish) === #
# =============================================================================================== #

class Simulation_Store:
    """
    Every simulation of the run, keyed by (shape key, protocol). Each finished simulation is appended to
    simulations.jsonl at once (forced to disk), so an interruption loses nothing that finished; a resumed run reloads
    the file and simulates only what is missing.
    """

    def __init__(self, path: Path, ctx):
        self.path = Path(path)
        self.ctx = ctx
        self.entries = {(e["key"], e["protocol"]): e for e in read_jsonl(self.path)}

    def run(self, key: str, protocol: str = SWEEP_MODE):
        entry = self.entries.get((key, protocol))
        return entry["run"] if entry else None

    def simulate(self, shapes: dict, protocol: str, stage: str, iteration: int, title: str) -> int:
        """Simulates the shapes ({key: cst}) not simulated yet with this protocol. Returns how many ran now."""
        missing = {k: {"cst": [float(v) for v in cst], "id": shape_id(k)} for k, cst in shapes.items()
                   if (k, protocol) not in self.entries}
        reused = len(shapes) - len(missing)
        print(f"\n--- Simulations: {title} ({len(missing)} to run" + (f", {reused} reused" if reused else "") + ") ---")
        if not missing:
            return 0

        def on_result(key, prot, run):
            entry = {"key": key, "id": missing[key]["id"], "cst": missing[key]["cst"], "protocol": prot, "stage": stage,
                     "iteration": iteration, "run": jsonable(run)}
            append_jsonl(self.path, entry)
            self.entries[(key, prot)] = entry

        t0 = time.perf_counter()
        simulate_shapes(missing, [protocol], self.ctx.workers, configs=self.ctx.identifiers,
                        settings={protocol: self.ctx.settings[protocol]}, temp_root=self.ctx.temp_root, store_polar=True,
                        store_files=True, on_result=on_result, title=f"Inference: {title}", use_gui=False)
        runs = [self.run(k, protocol) for k in missing]
        no_polar = sum(1 for r in runs if r is None or not r.get("polar_points"))
        valid = sum(1 for k in missing if best_valid_E(self.run(k, protocol), self.ctx)[0] is not None)
        status = defaultdict(int)
        for r in runs:
            status[(r or {}).get("status", "missing")] += 1
        print(f"{len(missing)} simulated in {hms(time.perf_counter() - t0)}: {valid} with a valid E | "
              + ", ".join(f"{s} {n}" for s, n in sorted(status.items())))
        if no_polar == len(missing) and len(missing) >= 3:
            errors = sorted({str((r or {}).get('error')) for r in runs})[:3]
            print("\n" + "!" * 90)
            print(f"[ERROR] XFOIL wrote no polar for any of the {len(missing)} airfoils of this batch, so nothing can be simulated.")
            for e in errors:
                print(f"        {e[:160]}")
            print("        1. Check XFOIL:      python src/simulation_support/polar_creation.py")
            print(f"        2. Continue the run: python src/inference_and_testing.py --resume {self.ctx.run_dir.name} "
                  f"--dataset {self.ctx.dataset_num}")
            print("!" * 90)
            self.ctx.metadata.update(run_status=f"stopped: no polars in '{title}'", finished=now_text())
            write_json(self.ctx.metadata_path, self.ctx.metadata)
            sys.exit(1)
        return len(missing)


def valid_E(run, identifier: str, ctx):
    """Simulated E of one configuration, or None: failed simulation or a polar below the laminar CD limit."""
    if not run or run.get("status") not in ("ok", "partial"):
        return None
    if run.get("cd_min") is None or run["cd_min"] < ctx.cd_limit:
        return None
    e = run.get("E", {}).get(identifier)
    return float(e) if e is not None and math.isfinite(e) else None


def best_valid_E(run, ctx):
    """(best E, its configuration identifier) over the configurations, or (None, None)."""
    best = (None, None)
    for ident in ctx.identifiers:
        e = valid_E(run, ident, ctx)
        if e is not None and (best[0] is None or e > best[0]):
            best = (e, ident)
    return best


def now_text() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


# =============================================================================================== #
# === Lanes === #
# =============================================================================================== #

class Lane:
    """One search: its models (own copies), tracks, stop rule, the shapes it requested and its fine-tuning history."""

    def __init__(self, name: str, kind: str, model_names: list, index: int):
        self.name, self.kind, self.model_names, self.index = name, kind, list(model_names), index
        self.models = {}
        self.tracks, self.removed = [], []
        self.monitor = None
        self.keys = {}           # shape key -> iteration it was first requested (simulation count order)
        self.picks = []          # every pick with its predictions
        self.accepted = {}       # model -> {"iteration", "centres", "seed"} of its kept fine-tune
        self.ft_reports = []
        self.history = []
        self.done, self.done_reason = False, None

    @property
    def label(self) -> str:
        return self.name

    def request(self, key: str, iteration: int):
        self.keys.setdefault(key, iteration)

    def to_dict(self) -> dict:
        return {"name": self.name, "kind": self.kind, "model_names": self.model_names, "index": self.index,
                "tracks": [t.to_dict() for t in self.tracks],
                "removed": [{"track": t.to_dict(), "reason": r, "iteration": it} for t, r, it in self.removed],
                "monitor": self.monitor.to_dict() if self.monitor else None, "keys": self.keys, "picks": self.picks,
                "accepted": self.accepted, "ft_reports": self.ft_reports, "history": self.history,
                "done": self.done, "done_reason": self.done_reason}

    @classmethod
    def from_dict(cls, d: dict) -> "Lane":
        lane = cls(d["name"], d["kind"], d["model_names"], d["index"])
        lane.tracks = [Trust_Region.from_dict(t) for t in d["tracks"]]
        lane.removed = [(Trust_Region.from_dict(r["track"]), r["reason"], r["iteration"]) for r in d["removed"]]
        lane.monitor = Lane_Monitor.from_dict(d["monitor"]) if d["monitor"] else None
        lane.keys, lane.picks, lane.accepted = dict(d["keys"]), list(d["picks"]), dict(d["accepted"])
        lane.ft_reports, lane.history = list(d["ft_reports"]), list(d["history"])
        lane.done, lane.done_reason = bool(d["done"]), d["done_reason"]
        return lane


def build_lanes(model_names: list, lane_option, model_types: dict) -> list:
    specs = [(n, "model", [n]) for n in model_names] + [(LANE_CONSENSUS, LANE_CONSENSUS, list(model_names)),
                                                       (LANE_RANDOM, LANE_RANDOM, [])]
    if lane_option:
        specs = [s for s in specs if s[0] == lane_option]
    return [Lane(name, kind, names, i) for i, (name, kind, names) in enumerate(specs)]


def lane_description(lane: Lane, ctx) -> str:
    if lane.kind == "model":
        mtype, feature = ctx.model_info[lane.name]
        return f"{lane.name} ({TYPE_LABELS.get(mtype, mtype)}, {feature} features)"
    if lane.kind == LANE_CONSENSUS:
        return f"consensus of {', '.join(lane.model_names)}"
    return "random picks inside the trust regions (no models)"


# --- Fine-tuning --- #

def lane_training_rows(lane: Lane, ctx, upto_iteration: int):
    """Feature rows, E values and shape keys of the lane's valid simulations requested up to that iteration
    (dataset shapes excluded: they are in the dataset already)."""
    rows, values, groups = [], [], []
    for key, it in lane.keys.items():
        if it > upto_iteration or key in ctx.dataset_keys:
            continue
        run = ctx.store.run(key)
        cst = ctx.store.entries[(key, SWEEP_MODE)]["cst"] if run else None
        for col, ident in ctx.identifier_of.items():
            e = valid_E(run, ident, ctx)
            if e is not None:
                rows.append(feature_row(cst, col, ctx))
                values.append(e)
                groups.append(key)
    if not rows:
        return np.zeros((0, len(ctx.feature_columns)), np.float32), np.zeros(0), []
    return np.array(rows, dtype=np.float32), np.array(values), groups


def anchor_rows(ctx, centres, seed: int):
    """Dataset rows near the track centres (local anchors) plus random dataset rows."""
    rng = np.random.default_rng(seed)
    near = np.zeros(len(ctx.data.y), dtype=bool)
    for c in centres:
        near |= np.abs(ctx.data.cst_scaled - np.asarray(c) / ctx.data.steps).max(axis=1) <= FT_LOCAL_ANCHOR_STEPS
    near_idx = np.where(near)[0]
    if len(near_idx) > FT_LOCAL_ANCHOR_MAX:
        near_idx = rng.choice(near_idx, FT_LOCAL_ANCHOR_MAX, replace=False)
    rand_idx = rng.choice(len(ctx.data.y), min(FT_RANDOM_ANCHORS, len(ctx.data.y)), replace=False)
    idx = np.unique(np.concatenate([near_idx, rand_idx]))
    return ctx.data.X[idx], ctx.data.y[idx]


def fine_tune_lane(lane: Lane, ctx, iteration: int):
    """Fine-tunes every model of the lane (from its original weights) on all of the lane's simulations so far."""
    if lane.kind == LANE_RANDOM or ctx.no_update or not lane.models:
        return
    X_new, y_new, groups = lane_training_rows(lane, ctx, iteration)
    n_shapes = len(set(groups))
    if n_shapes < FT_MIN_SHAPES:
        print(f"  {lane.label}: {n_shapes} new valid shape(s), fewer than {FT_MIN_SHAPES}: models not updated.")
        return
    centres = [t.centre.tolist() for t in lane.tracks] or [t.centre.tolist() for t, _, _ in lane.removed[-1:]]
    seed = seed_for(ctx.master_seed, 7, lane.index, iteration)
    X_anchor, y_anchor = anchor_rows(ctx, centres, seed)
    for name, model in lane.models.items():
        t0 = time.perf_counter()
        report = model.fine_tune(X_new, y_new, X_anchor, y_anchor, groups_new=groups, new_weight=FT_NEW_WEIGHT,
                                 random_state=seed, from_base=True)
        if report["accepted"]:
            lane.accepted[name] = {"iteration": iteration, "centres": centres, "seed": seed}
        lane.ft_reports.append({"iteration": iteration, "model": name, **report})
        print(f"  {lane.label}: {name} update {'KEPT' if report['accepted'] else 'rejected'} | {n_shapes} shapes "
              f"({len(y_new)} rows) | held-out new-shape MAE {fmt(report['mae_new_before'], 3)} -> "
              f"{fmt(report['mae_new_after'], 3)} | anchor MAE {fmt(report['mae_anchor_before'], 3)} -> "
              f"{fmt(report['mae_anchor_after'], 3)} | {hms(time.perf_counter() - t0)}")


def rebuild_fine_tunes(lanes: list, ctx):
    """--resume: rebuilds every kept fine-tune from the saved simulations (the models themselves are not stored)."""
    jobs = [(lane, name, acc) for lane in lanes for name, acc in lane.accepted.items()]
    if not jobs:
        return
    print(f"\nRebuilding {len(jobs)} fine-tuned model(s) from the saved simulations...")
    for lane, name, acc in jobs:
        X_new, y_new, groups = lane_training_rows(lane, ctx, acc["iteration"])
        X_anchor, y_anchor = anchor_rows(ctx, acc["centres"], acc["seed"])
        lane.models[name].fine_tune(X_new, y_new, X_anchor, y_anchor, groups_new=groups, new_weight=FT_NEW_WEIGHT,
                                    random_state=acc["seed"], from_base=True, force_accept=True)
        print(f"  {lane.label}: {name} rebuilt (update of iteration {acc['iteration']})")


# =============================================================================================== #
# === Iteration 0 === #
# =============================================================================================== #

def run_iteration0_inference(ctx, lanes: list, models: dict, round_seeds: list) -> dict:
    """Candidate rounds scored by every model; returns each lane's shortlist and the reference shapes."""
    names = list(models)
    lows, highs = ctx.lows, ctx.highs
    edge_dirs, ext_lows, ext_highs = detect_edge_directions(ctx.data.parents, lows, highs, ctx.data.steps)
    edge_sigma = (highs - lows) * LOCAL_SIGMA_FRACTION
    for i, _side, _bound in edge_dirs:
        edge_sigma[i] = ctx.data.steps[i] * EDGE_SIGMA_STEPS
    use_edge = NUM_EDGE_SAMPLES > 0 and bool(edge_dirs)
    print(f"Iteration 0: {len(round_seeds)} round(s) of {NUM_SAMPLES:,} uniform + {NUM_LOCAL_SAMPLES:,} local + "
          f"{NUM_EDGE_SAMPLES if use_edge else 0:,} edge-probe samples, scored by {len(names)} model(s) in "
          f"{ctx.n_configs} configuration(s).")
    if use_edge:
        print(f"Edge probe past: {[f'{ctx.cst_names[i]} {side} -> {bound:.4f}' for i, side, bound in edge_dirs]}")
    scored_lanes = [l for l in lanes if l.kind != LANE_RANDOM]
    pool = defaultdict(list)
    random_pool = []
    t0 = time.perf_counter()
    for r, seed in enumerate(round_seeds, start=1):
        rng = np.random.default_rng(seed)
        blocks = [(generate_uniform(NUM_SAMPLES, lows, highs, rng), 'global')]
        if NUM_LOCAL_SAMPLES > 0:
            blocks.append((generate_local(ctx.data.parents, NUM_LOCAL_SAMPLES, lows, highs, rng), 'local'))
        if use_edge:
            edge = generate_local(ctx.data.parents, NUM_EDGE_SAMPLES, lows, highs, rng, ext_lows, ext_highs, edge_sigma)
            blocks.append((edge[outside_window(edge, lows, highs)], 'edge'))
        cst = np.concatenate([b for b, _ in blocks])
        origin = np.concatenate([np.full(len(b), o) for b, o in blocks])
        inside = origin != 'edge'
        t_round = time.perf_counter()
        preds = score_shapes(models, cst, ctx) if scored_lanes else np.zeros((len(cst), ctx.n_configs, 0))
        elite = set()
        best_line = []
        for lane in scored_lanes:
            cols = [names.index(m) for m in lane.model_names]
            score, _ = lane_score(lane.kind, preds[:, :, cols])
            elite.update(top_indices(score, ELITE_PER_LANE, inside).tolist())
            elite.update(top_indices(score, ELITE_PER_LANE // 4, ~inside).tolist())
            if np.isfinite(score).any():
                best_line.append(f"{lane.label} {np.nanmax(np.where(inside, score, np.nan)):.3f}")
        elite = np.array(sorted(elite), dtype=int)
        pool['cst'].append(cst[elite])
        pool['preds'].append(preds[elite])
        pool['origin'].append(origin[elite])
        pool['round'].append(np.full(len(elite), r))
        random_pool.append(cst[rng.choice(np.where(inside)[0], min(RANDOM_POOL_PER_ROUND, int(inside.sum())), replace=False)])
        print(f"  Round {r}/{len(round_seeds)} (seed {seed:x}): {len(cst):,} shapes scored in {hms(time.perf_counter() - t_round)}"
              + (f" | best predicted E: {', '.join(best_line)}" if best_line else ""))
    pool = {k: np.concatenate(v) for k, v in pool.items()}
    random_pool = np.concatenate(random_pool)
    scaled = pool['cst'].astype(np.float64) / ctx.data.steps
    inside = pool['origin'] != 'edge'
    print(f"Elite pool: {len(pool['cst']):,} shapes ({int(inside.sum()):,} inside the window). "
          f"Inference time {hms(time.perf_counter() - t0)}.")

    shortlist = {}
    for lane in lanes:
        picks = []
        if lane.kind == LANE_RANDOM:
            rng = rng_for(ctx.master_seed, 3, lane.index, 0)
            for i in rng.permutation(len(random_pool)):
                if len(picks) >= RANDOM_PICKS_0:
                    break
                if ctx.geometry_ok(random_pool[i]):
                    picks.append({"key": shape_key(random_pool[i]), "cst": random_pool[i].tolist(), "group": "random",
                                  "score": None, "config": None, "preds": {}})
        else:
            cols = [names.index(m) for m in lane.model_names]
            score, best_c = lane_score(lane.kind, pool['preds'][:, :, cols])
            accept = lambda i: ctx.geometry_ok(pool['cst'][i])
            chosen = [(i, "in window") for i in diverse_top(score, scaled, TOP_K_IN_WINDOW, inside, DIVERSITY_MIN_STEPS, accept)]
            chosen += [(i, "edge") for i in diverse_top(score, scaled, TOP_K_EDGE, ~inside, DIVERSITY_MIN_STEPS, accept,
                                                         taken=scaled[[i for i, _ in chosen]] if chosen else None)]
            for i, group in chosen:
                picks.append({"key": shape_key(pool['cst'][i]), "cst": pool['cst'][i].tolist(), "group": group,
                              "score": float(score[i]), "config": ctx.identifiers[int(best_c[i])],
                              "preds": {m: pool['preds'][i, :, names.index(m)].tolist() for m in names}})
        shortlist[lane.name] = picks
    references = []
    seen = set()
    for cst, _config, e in ctx.data.references:
        key = shape_key(cst)
        if key not in seen:
            seen.add(key)
            references.append({"key": key, "cst": [float(v) for v in cst], "dataset_E": e})
    for p in ctx.peaks:
        key = shape_key(p["cst"])
        if key not in seen:
            seen.add(key)
            references.append({"key": key, "cst": [float(v) for v in p["cst"]], "dataset_E": p["E"]})
    return {"lanes": shortlist, "references": references, "pool_shapes": int(len(pool['cst'])),
            "rounds": len(round_seeds), "inference_time": hms(time.perf_counter() - t0)}


def start_tracks(lane: Lane, ctx, shortlist: dict):
    """Tracks at the dataset peaks (simulated E of the peak when valid), moved or added by the iteration-0 finds."""
    lane.tracks = []
    for i, p in enumerate(ctx.peaks, start=1):
        e, ident = best_valid_E(ctx.store.run(shape_key(p["cst"])), ctx)
        lane.tracks.append(Trust_Region(f"T{i}", p["cst"], e if e is not None else p["E"], ident or p["config"],
                                        ctx.data.steps))
    finds = []
    for pick in shortlist["lanes"][lane.name]:
        e, ident = best_valid_E(ctx.store.run(pick["key"]), ctx)
        if e is not None:
            finds.append((e, np.asarray(pick["cst"], dtype=np.float64), ident))
    finds.sort(key=lambda f: -f[0])
    for track in lane.tracks:
        inside = [f for f in finds if track.inside(f[1])[0] and f[0] > track.centre_E]
        if inside:
            e, cst, ident = inside[0]
            track.history.append({"iteration": 0, "E_before": track.centre_E, "E_after": e, "gain": e - track.centre_E,
                                  "rho": None, "radius_before": track.radius, "radius_after": track.radius,
                                  "action": "moved to the best iteration-0 design"})
            track.centre, track.centre_E, track.centre_config = cst, e, ident
    max_tracks = MAX_TRACKS_FACTOR * max(len(ctx.peaks), 1)
    for e, cst, ident in finds:
        if len(lane.tracks) >= max_tracks:
            break
        weakest = min(t.centre_E for t in lane.tracks) if lane.tracks else -np.inf
        far = all(t.distance(cst)[0] > NEW_TRACK_MIN_STEPS for t in lane.tracks)
        if e > weakest and far:
            lane.tracks.append(Trust_Region(f"T{len(lane.tracks) + 1}", cst, e, ident, ctx.data.steps,
                                            origin="iteration-0 design"))
    lane.monitor = Lane_Monitor(max(t.centre_E for t in lane.tracks), MIN_GAIN, STALL_PATIENCE)


# =============================================================================================== #
# === Loop iteration === #
# =============================================================================================== #

def pick_for_track(lane: Lane, track: Trust_Region, ctx, rng, taken_scaled) -> list:
    """The track's picks for this iteration (diverse, geometrically valid, not simulated by the lane before)."""
    steps = ctx.data.steps
    min_steps = PICK_MIN_STEPS_FACTOR * track.radius
    if lane.kind == LANE_RANDOM:
        cand = track.sample_uniform(PICKS_PER_TRACK * 30, rng)
        idx = diverse_top(rng.random(len(cand)), cand.astype(np.float64) / steps, PICKS_PER_TRACK, min_steps=min_steps,
                          accept=lambda i: ctx.geometry_ok(cand[i]), taken=taken_scaled)
        return [{"key": shape_key(cand[i]), "cst": cand[i].tolist(), "track": track.track_id, "group": "random",
                 "score": None, "config": None, "preds": {}} for i in idx]
    cand = track.sample_uniform(TR_UNIFORM_SAMPLES, rng)
    preds = score_shapes(lane.models, cand, ctx)
    score, _ = lane_score(lane.kind, preds)
    top = top_indices(score, TR_REFINE_TOP)
    if len(top) and TR_REFINE_SAMPLES > 0:
        extra = track.sample_around(cand[top], TR_REFINE_SAMPLES, rng)
        cand = np.vstack([cand, extra])
        preds = np.concatenate([preds, score_shapes(lane.models, extra, ctx)])
        score, _ = lane_score(lane.kind, preds)
    _, best_c = lane_score(lane.kind, preds)
    scaled = cand.astype(np.float64) / steps
    accept = lambda i: ctx.geometry_ok(cand[i])
    names = list(lane.models)
    chosen = []

    def add(indices, group):
        for i in indices:
            chosen.append((i, group))

    def taken_now():
        parts = [taken_scaled] if taken_scaled is not None and len(taken_scaled) else []
        if chosen:
            parts.append(scaled[[i for i, _ in chosen]])
        return np.vstack(parts) if parts else None

    if lane.kind == "model":
        add(diverse_top(score, scaled, PICKS_PER_TRACK, min_steps=min_steps, accept=accept, taken=taken_scaled), "model")
    else:
        per_model = 2 if len(names) >= 2 else 0
        explore = 2 if len(names) >= 2 else 0
        n_consensus = max(PICKS_PER_TRACK - per_model * len(names) - explore, 1)
        add(diverse_top(score, scaled, n_consensus, min_steps=min_steps, accept=accept, taken=taken_scaled), "consensus")
        for m, name in enumerate(names[:]):
            if per_model:
                own = np.where(np.isfinite(preds[:, :, m]), preds[:, :, m], -np.inf).max(axis=1)
                add(diverse_top(own, scaled, per_model, min_steps=min_steps, accept=accept, taken=taken_now()), f"model:{name}")
        if explore:
            cut = np.nanquantile(score, 1 - EXPLORE_TOP_SHARE) if np.isfinite(score).any() else np.inf
            spread = np.ptp(preds[np.arange(len(cand)), best_c, :], axis=1)
            add(diverse_top(spread, scaled, explore, mask=score >= cut, min_steps=min_steps, accept=accept,
                            taken=taken_now()), "explore")
    return [{"key": shape_key(cand[i]), "cst": cand[i].tolist(), "track": track.track_id, "group": group,
             "score": float(score[i]) if np.isfinite(score[i]) else None, "config": ctx.identifiers[int(best_c[i])],
             "preds": {name: preds[i, :, m].tolist() for m, name in enumerate(names)}} for i, group in chosen]


def pick_iteration(lanes: list, ctx, iteration: int) -> dict:
    """{lane name: picks} of every active lane for this iteration."""
    picks = {}
    for lane in lanes:
        if lane.done:
            continue
        taken = [np.asarray(ctx.store.entries[(k, SWEEP_MODE)]["cst"]) / ctx.data.steps
                 for k in lane.keys if (k, SWEEP_MODE) in ctx.store.entries]
        taken = np.array(taken) if taken else None
        lane_picks = []
        t0 = time.perf_counter()
        for t_index, track in enumerate(lane.tracks):
            rng = rng_for(ctx.master_seed, 5, lane.index, iteration, t_index)
            new = pick_for_track(lane, track, ctx, rng, taken)
            lane_picks += new
            if new:
                scaled = np.array([p["cst"] for p in new]) / ctx.data.steps
                taken = scaled if taken is None else np.vstack([taken, scaled])
        picks[lane.name] = lane_picks
        print(f"  {lane.label}: {len(lane_picks)} pick(s) from {len(lane.tracks)} track(s) in {hms(time.perf_counter() - t0)}")
    return picks


def pick_errors(picks: list, ctx) -> dict:
    """Mean absolute error of every model on the picks it scored (prediction made before the simulation)."""
    errors = defaultdict(list)
    for p in picks:
        run = ctx.store.run(p["key"])
        for name, values in p.get("preds", {}).items():
            for ident, pred in zip(ctx.identifiers, values):
                e = valid_E(run, ident, ctx)
                if e is not None and pred is not None and math.isfinite(pred):
                    errors[name].append(abs(pred - e))
    return {name: float(np.mean(v)) for name, v in errors.items() if v}


def update_lane(lane: Lane, picks: list, ctx, iteration: int):
    """Track updates, pruning, stop rule and the iteration record of one lane (after its picks were simulated)."""
    lines = []
    for track in lane.tracks:
        own = [p for p in picks if p["track"] == track.track_id]
        best = (None, None, None)
        for p in own:
            e, ident = best_valid_E(ctx.store.run(p["key"]), ctx)
            if e is not None and (best[0] is None or e > best[0]):
                best = (e, p["cst"], ident)
        scores = [p["score"] for p in own if p.get("score") is not None]
        step = track.update(iteration, best[0], best[1], best[2], predicted_E=max(scores) if scores else None)
        if step["gain"] is None:
            result = "no valid result"
        else:
            result = (f"best pick {step['E_before'] + step['gain']:.3f} "
                      f"({100 * step['gain'] / max(abs(step['E_before']), 1e-9):+.2f}%)")
        moved = step["gain"] is not None and step["gain"] > 0
        rho = f", rho {step['rho']:.2f}" if (step["rho"] is not None and moved) else ""
        lines.append(f"    track {track.track_id}: centre E {step['E_before']:.3f} -> {step['E_after']:.3f} | {result} | radius "
                     f"{step['radius_before']:.3f} -> {step['radius_after']:.3f} steps ({step['action']}{rho})")
    lane_best = max([t.centre_E for t in lane.tracks] + [lane.monitor.best_E])
    gain = lane.monitor.update(lane_best)
    kept, removed = Trust_Region.prune(lane.tracks, lane.monitor.best_E, iteration, DROP_SHARE, MERGE_STEPS)
    lane.tracks = kept
    for track, reason in removed:
        lane.removed.append((track, reason, iteration))
        lines.append(f"    track {track.track_id} removed: {reason}")
    if lane.monitor.done:
        lane.done, lane.done_reason = True, (f"converged: best E gained less than {MIN_GAIN:.1%} for "
                                             f"{STALL_PATIENCE} iterations in a row")
    elif not lane.tracks:
        lane.done, lane.done_reason = True, "no tracks left"
    errors = pick_errors(picks, ctx)
    lane.history.append({"iteration": iteration, "lane": lane.name, "tracks": len(lane.tracks), "shapes": len(picks),
                         "simulations_total": len(lane.keys), "best_E": lane.monitor.best_E, "gain": gain,
                         "stall": lane.monitor.stall, "pick_MAE": errors, "done": lane.done})
    print(f"\n  {lane.label}: best E {lane.monitor.best_E:.3f} ({gain:+.2%}) | {len(lane.tracks)} track(s) | "
          f"{len(picks)} shapes this iteration, {len(lane.keys)} in total | stall {lane.monitor.stall}/{STALL_PATIENCE}")
    for line in lines:
        print(line)
    if errors:
        print("    error on its picks (MAE, predicted before simulating): "
              + ", ".join(f"{n} {v:.3f}" for n, v in errors.items()))
    if lane.done:
        print(f"    -> lane finished: {lane.done_reason}")


# =============================================================================================== #
# === State (recovery) === #
# =============================================================================================== #

def save_state(ctx, lanes: list, **fields):
    ctx.state.update(fields)
    ctx.state["lanes"] = [lane.to_dict() for lane in lanes]
    ctx.state["elapsed_seconds"] = ctx.elapsed()
    ctx.state["saved"] = now_text()
    write_json(ctx.run_dir / STATE_FILE, ctx.state)


# =============================================================================================== #
# === Final validation === #
# =============================================================================================== #

def window_status(cst, ctx) -> dict:
    cst = np.asarray(cst, dtype=np.float64)
    out = []
    for i, name in enumerate(ctx.cst_names):
        if cst[i] > ctx.highs[i] + 1e-7:
            out.append(f"{name} +{(cst[i] - ctx.highs[i]) / ctx.data.steps[i]:.2f} steps")
        elif cst[i] < ctx.lows[i] - 1e-7:
            out.append(f"{name} -{(ctx.lows[i] - cst[i]) / ctx.data.steps[i]:.2f} steps")
    d, _ = ctx.data.all_tree.query(cst / ctx.data.steps, p=np.inf)
    region = "outside" if out else ("on-grid" if d < 1e-3 else "between")
    return {"region": region, "grid_distance": float(d), "outside": out}


def final_validation(lanes: list, ctx, models: dict, classifier) -> dict:
    """Cold-start check, robustness, filter and window status of every lane's best designs."""
    winners = {}
    for lane in lanes:
        ranked = []
        for key in lane.keys:
            if key in ctx.dataset_keys:
                continue
            e, ident = best_valid_E(ctx.store.run(key), ctx)
            if e is not None:
                ranked.append((e, key, ident))
        ranked.sort(reverse=True)
        winners[lane.name] = ranked[:max(VALIDATION_TOP, ROBUST_DESIGNS)]
    cst_of = lambda key: ctx.store.entries[(key, SWEEP_MODE)]["cst"]
    check_keys = {key: cst_of(key) for ranked in winners.values() for _, key, _ in ranked[:VALIDATION_TOP]}
    ctx.store.simulate(check_keys, "cold", "validation", -1, "cold-start check of the best designs (slow: one XFOIL start per angle)")

    robust = {}
    robust_shapes = {}
    for lane in lanes:
        rng = rng_for(ctx.master_seed, 9, lane.index)
        for e, key, ident in winners[lane.name][:ROBUST_DESIGNS]:
            if key in robust:
                continue
            base = np.asarray(cst_of(key))
            kids = []
            for _ in range(ROBUST_PERTURBATIONS):
                cst = (base + rng.normal(0.0, ROBUST_SIGMA_STEPS, size=len(base)) * ctx.data.steps).astype(np.float32)
                k = shape_key(cst)
                kids.append(k)
                robust_shapes[k] = cst.tolist()
            robust[key] = {"parent_E": e, "config": ident, "kids": kids}
    ctx.store.simulate(robust_shapes, SWEEP_MODE, "robustness", -1, "robustness perturbations")
    for key, info in robust.items():
        values = [valid_E(ctx.store.run(k), info["config"], ctx) for k in info["kids"]]
        values = np.array([v for v in values if v is not None])
        info.update(simulated=len(values), hold_rate=float((values >= ROBUST_HOLD_FRACTION * info["parent_E"]).mean()) if len(values) else None,
                    mean_E=float(values.mean()) if len(values) else None, min_E=float(values.min()) if len(values) else None,
                    max_E=float(values.max()) if len(values) else None)

    designs = {}
    names = list(models)
    for lane in lanes:
        rows = []
        for e, key, ident in winners[lane.name][:VALIDATION_TOP]:
            cst = np.asarray(cst_of(key), dtype=np.float64)
            x, yu, yl = ctx.airfoil_creator.generate(cst_coeffs=cst)
            filters = check_airfoil(x, yu, yl, cst, ctx.n_cst, name=shape_id(key))
            col = next(c for c, i in ctx.identifier_of.items() if i == ident)
            original = predict_all(models, feature_row(cst, col, ctx)[None])[0] if names else []
            updated = predict_all(lane.models, feature_row(cst, col, ctx)[None])[0] if lane.models else []
            prob = float(predict_chunked(classifier.predict, cst[None].astype(np.float32))[0]) if classifier is not None else None
            rows.append({"key": key, "id": shape_id(key), "cst": cst.tolist(), "E": e, "config": ident,
                         "E_all_configs": {i: valid_E(ctx.store.run(key), i, ctx) for i in ctx.identifiers},
                         "cold_check": compare_runs(ctx.store.run(key), ctx.store.run(key, "cold"), ctx.identifiers),
                         "cold_E": (ctx.store.run(key, "cold") or {}).get("E", {}).get(ident),
                         "robustness": robust.get(key), "filters": filters, "classifier_probability": prob,
                         "window": window_status(cst, ctx),
                         "original_predictions": dict(zip(names, map(float, original))),
                         "lane_predictions": dict(zip(lane.models, map(float, updated)))})
        designs[lane.name] = rows
    return designs


def reference_target(ctx) -> tuple:
    """(target E, re-simulated E of the dataset best shape): a design must beat both the dataset best and its re-simulation."""
    run = ctx.store.run(shape_key(ctx.data.best["cst"]))
    resim = valid_E(run, ctx.identifier_of.get(ctx.data.best["config"], ctx.data.best["config"]), ctx)
    return (max(ctx.data.e_max, resim) if resim is not None else ctx.data.e_max), resim


def lane_outcome(designs: list, target: float) -> str:
    if not designs or designs[0]["E"] <= target:
        return "Try again..."
    verdict = designs[0]["cold_check"]["verdict"]
    return "SUCCESS!!" if verdict == "agrees" else "SUCCESS??"


def simulations_to_reach(lane: Lane, ctx) -> dict:
    """Simulations (shapes requested, in order) the lane needed to reach each share of the dataset best E."""
    order = sorted(lane.keys.items(), key=lambda kv: kv[1])
    best, count, reached = -np.inf, 0, {}
    curve = []
    for key, _ in order:
        count += 1
        if key not in ctx.dataset_keys:
            e, _ = best_valid_E(ctx.store.run(key), ctx)
            if e is not None:
                best = max(best, e)
        curve.append((count, best if np.isfinite(best) else None))
        for share in REACH_SHARES:
            if share not in reached and np.isfinite(best) and best >= share * ctx.data.e_max:
                reached[share] = count
    return {"reached": {f"{share:.0%}": reached.get(share) for share in REACH_SHARES}, "curve": curve}


# =============================================================================================== #
# === Expanded dataset and models (versions X.Y.i) === #
# =============================================================================================== #

def save_expansion(ctx, lanes: list, iterations: int) -> dict:
    """Writes the expanded dataset (dataset + every valid new simulation) and one final fine-tune of every model."""
    if ctx.no_update:
        print("\n[NOTE] --no-update: the models were frozen, so this run writes no new dataset or model versions.")
        return {"status": "skipped (--no-update)"}
    if parent_dataset(ctx.dataset_num):
        print(f"\n[NOTE] Dataset {ctx.dataset_num} is already an expansion; no further version is written.")
        return {"status": "skipped (dataset is an expansion)"}
    number = expansion_number(ctx.dataset_num, iterations)
    lanes_of = defaultdict(set)
    for lane in lanes:
        for key in lane.keys:
            lanes_of[key].add(lane.name)
    rows, values, groups, side = [], [], [], []
    below_limit = 0
    for (key, protocol), entry in sorted(ctx.store.entries.items()):
        if protocol != SWEEP_MODE or key in ctx.dataset_keys:
            continue
        run = entry["run"]
        if run and run.get("cd_min") is not None and run["cd_min"] < ctx.cd_limit:
            below_limit += 1
            continue
        for col, ident in ctx.identifier_of.items():
            e = valid_E(run, ident, ctx)
            if e is not None:
                rows.append(feature_row(entry["cst"], col, ctx))
                values.append(e)
                groups.append(key)
                side.append({"id": entry["id"], "config": ident, "E": e, "stage": entry["stage"],
                             "iteration": entry["iteration"], "lanes": "+".join(sorted(lanes_of.get(key, []))) or "robustness"})
    if not rows:
        print("\n[NOTE] No valid new simulations: no expanded dataset written.")
        return {"status": "skipped (no valid new simulations)"}
    X_new, y_new = np.array(rows, dtype=np.float32), np.array(values)
    print(f"\n# === Expanded dataset {number}: {len(y_new)} new rows from {len(set(groups))} shapes "
          f"({below_limit} shape(s) below the laminar CD limit left out) === #")

    data_dir = dataset_folder(number, ctx.paths.project_root)
    model_dir = models_folder(number, ctx.paths.project_root)
    diverted = data_dir.exists() or model_dir.exists()
    if diverted:
        data_dir = ctx.run_dir / f"dataset_{number}" / "data"
        model_dir = ctx.run_dir / f"dataset_{number}" / "models"
        print("!" * 90)
        print(f"[WARNING] Version {number} already exists in data/datasets or models (another run with {iterations} "
              f"iteration(s)?). Nothing there is overwritten: this run's version is saved in the run folder instead:")
        print(f"          {data_dir.parent}")
        print("!" * 90)
    data_partial = data_dir.with_name(data_dir.name + ".partial")
    model_partial = model_dir.with_name(model_dir.name + ".partial")
    for d in (data_partial, model_partial):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    target = pd.read_csv(ctx.paths.training_file, nrows=0).columns[-1]
    new_df = pd.DataFrame(X_new, columns=ctx.feature_columns)
    new_df[target] = y_new
    full = pd.concat([pd.read_csv(ctx.paths.training_file), new_df], ignore_index=True)
    csv_path = data_partial / f"regressor_dataset_{number}.csv"
    full.to_csv(csv_path, index=False)
    side_df = pd.concat([pd.DataFrame(side), new_df[ctx.cst_names].reset_index(drop=True)], axis=1)
    side_df.to_csv(data_partial / f"expansion_rows_{number}.csv", index=False)

    reports = {}
    for name, (folder, mtype) in ctx.resolved.items():
        if name not in ctx.model_names:
            continue
        model = load_regressor(name, folder, mtype)
        new_name = expanded_model_name(name, iterations)
        t0 = time.perf_counter()
        report = model.fine_tune(X_new, y_new, ctx.data.X, ctx.data.y, groups_new=groups, new_weight=FT_NEW_WEIGHT,
                                 random_state=seed_for(ctx.master_seed, 11), from_base=True)
        out = model_partial / folder.name
        model.save_model(str(out), new_name)
        (out / f"{new_name}_regressor_model_parameters.json").write_text(json.dumps(model.parameters_json, indent=4))
        info = {"name": new_name, "base_name": base_name(name), "version": model_version(new_name),
                "mode": "inference fine-tune", "previous_model": str(folder / name),
                "dataset_csv": str(data_dir / csv_path.name), "run_folder": str(ctx.run_dir), "iterations": iterations,
                "new_rows": int(len(y_new)), "new_shapes": int(len(set(groups))), "fine_tune": report,
                "created": now_text(), "platform": platform.platform(),
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
        (out / f"{new_name}_training_info.json").write_text(json.dumps(jsonable(info), indent=4))
        reports[new_name] = report
        state = "update kept" if report["accepted"] else "UPDATE REJECTED by the gate: saved unchanged"
        print(f"  {name} -> {new_name}: {state} | held-out new-shape MAE {fmt(report['mae_new_before'], 3)} -> "
              f"{fmt(report['mae_new_after'], 3)} | anchor MAE {fmt(report['mae_anchor_before'], 3)} -> "
              f"{fmt(report['mae_anchor_after'], 3)} | {hms(time.perf_counter() - t0)}")
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(data_partial, data_dir)
    os.replace(model_partial, model_dir)
    print(f"Saved: {data_dir}")
    print(f"       {model_dir}")
    return {"status": "saved", "version": number, "dataset_dir": str(data_dir), "models_dir": str(model_dir),
            "diverted_to_run_folder": diverted, "new_rows": int(len(y_new)), "new_shapes": int(len(set(groups))),
            "below_laminar_limit": below_limit, "models": reports}


# =============================================================================================== #
# === Reports === #
# =============================================================================================== #

def fmt(v, digits=2, width=None):
    """Number formatting that tolerates None/NaN."""
    text = "-" if v is None or (isinstance(v, float) and math.isnan(v)) else (f"{v:.{digits}f}" if isinstance(v, float) else str(v))
    return text if width is None else f"{text:<{width}}"


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
    "?": [" ___ ", "|__ \\", "  / /", " |_| ", " (_) "],
    ".": ["   ", "   ", "   ", " _ ", "(_)"],
    " ": ["   ", "   ", "   ", "   ", "   "],
}


def banner(text: str) -> list:
    glyphs = [BANNER_GLYPHS[ch] for ch in text.upper()]
    return ["".join(g[row].ljust(max(len(r) for r in g)) for g in glyphs).rstrip() for row in range(5)]


def save_plots(ctx, lanes: list, reach: dict, target: float):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; plots skipped.")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for lane in lanes:
        curve = [(n, e) for n, e in reach[lane.name]["curve"] if e is not None]
        if curve:
            ax.step([c[0] for c in curve], [c[1] for c in curve], where="post", label=lane.label)
    ax.axhline(ctx.data.e_max, c="green", lw=1, ls="--", label=f"dataset best ({ctx.data.e_max:.2f})")
    ax.axhline(target, c="gray", lw=1, ls=":", label=f"target ({target:.2f})")
    ax.set_xlabel("Shapes simulated by the lane (each in every wing configuration)")
    ax.set_ylabel("Best simulated E found")
    ax.set_title("Convergence per lane")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ctx.run_dir / "convergence.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    names = ctx.model_names
    for name in names:
        xs, ys = [], []
        for lane in lanes:
            for p in lane.picks:
                values = p.get("preds", {}).get(name)
                if not values:
                    continue
                run = ctx.store.run(p["key"])
                for ident, pred in zip(ctx.identifiers, values):
                    e = valid_E(run, ident, ctx)
                    if e is not None and pred is not None and math.isfinite(pred):
                        xs.append(pred)
                        ys.append(e)
        if xs:
            ax.scatter(xs, ys, s=10, alpha=0.5, label=name)
    lims = ax.get_xlim()
    ax.plot(lims, lims, ls="--", c="gray", lw=1, label="perfect prediction")
    ax.axhline(ctx.data.e_max, c="green", lw=1, label="dataset best")
    ax.set_xlabel("Predicted E (when picked)")
    ax.set_ylabel("Simulated E")
    ax.set_title("Predicted vs simulated E of every pick")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ctx.run_dir / "parity.png", dpi=150)
    plt.close(fig)


def write_reports(ctx, lanes: list, designs: dict, reach: dict, expansion: dict, timing: dict):
    target, resim = reference_target(ctx)
    outcomes = {lane.name: lane_outcome(designs.get(lane.name, []), target) for lane in lanes}
    overall = "SUCCESS!!" if "SUCCESS!!" in outcomes.values() else ("SUCCESS??" if "SUCCESS??" in outcomes.values() else "Try again...")

    # --- result.txt --- #
    lines = [""] + ["   " + row for row in banner(overall)] + ["", "=" * 100]
    lines.append(f"Goal: a wing design with simulated E above the best of dataset {ctx.dataset_num}.")
    lines.append(f"Target: E > {target:.4f}  (dataset best {ctx.data.e_max:.4f}" + (f", re-simulated {resim:.4f})" if resim is not None else ")"))
    lines.append("SUCCESS!! = beats the target and the cold-start (history-free) simulation agrees; SUCCESS?? = beats the target "
                 "but the cold-start check disagrees (possible sweep artifact).")
    lines.append("")
    lines.append(f"{'Lane':<22} {'Outcome':<13} {'Best E':>8} {'vs target':>9} {'Shapes':>7} {'Iter':>5}  "
                 + "  ".join(f"{s:>5}" for s in reach[lanes[0].name]["reached"]) + "   (shapes simulated to reach that share of the dataset best)")
    for lane in lanes:
        best_E = designs[lane.name][0]["E"] if designs.get(lane.name) else None
        versus = f"{100 * (best_E - target) / target:+.2f}%" if best_E is not None else "-"
        iters = lane.history[-1]["iteration"] if lane.history else 0
        reached = "  ".join(f"{fmt(v):>5}" for v in reach[lane.name]["reached"].values())
        lines.append(f"{lane.label:<22} {outcomes[lane.name]:<13} {fmt(best_E, 4):>8} {versus:>9} {len(lane.keys):>7} "
                     f"{iters:>5}  {reached}")
    lines.append("")
    for lane in lanes:
        rows = designs.get(lane.name)
        if not rows:
            continue
        d = rows[0]
        n = ctx.n_cst
        lines.append(f"{lane.label}: best design {d['id']} E = {d['E']:.4f} ({d['config']}) | cold-start: {d['cold_check']['verdict']}"
                     f" | window: {d['window']['region']}" + (f" ({', '.join(d['window']['outside'])})" if d['window']['outside'] else "")
                     + f" | filters: {'pass' if d['filters']['passed'] else 'fail (' + str(d['filters']['reason']) + ')'}")
        lines.append(f"    CST lower: {[round(v, 4) for v in d['cst'][:n]]}  CST upper: {[round(v, 4) for v in d['cst'][n:]]}")
        rob = d.get("robustness")
        if rob and rob.get("simulated"):
            lines.append(f"    robustness: {rob['simulated']} perturbations, E {rob['min_E']:.2f} to {rob['max_E']:.2f} "
                         f"(hold rate {rob['hold_rate']:.0%})")
    lines.append("")
    lines += timing["lines"]
    if expansion.get("status") == "saved":
        lines.append(f"New versions: dataset {expansion['version']} ({expansion['new_rows']} new rows) and its models in "
                     f"{expansion['models_dir']}")
    else:
        lines.append(f"New versions: {expansion.get('status')}")
    lines += [f"Run: {ctx.run_dir.name} | master seed 0x{ctx.master_seed:x}", "Full details: detailed_results.txt", "=" * 100]
    atomic_write_text(ctx.run_dir / "result.txt", "\n".join(lines) + "\n")
    print("\n".join(lines))

    # --- detailed_results.txt --- #
    width = 150
    d_lines = ["=" * width, f"Iterative inference report, dataset {ctx.dataset_num}", "=" * width,
               f"Dataset best: E = {ctx.data.e_max:.4f} ({ctx.data.best['config']}) | target {target:.4f}",
               *ctx.peak_lines, ""]
    for lane in lanes:
        d_lines += [f"--- Lane {lane.label}: {lane_description(lane, ctx)} " + "-" * 60,
                    f"Outcome: {outcomes[lane.name]} | finished: {lane.done_reason or 'stopped by the time ceiling'}"]
        d_lines.append(f"{'Iter':>4} {'Tracks':>6} {'Shapes':>6} {'Total':>6} {'Best E':>8} {'Gain':>8} {'Stall':>5}  Pick MAE per model")
        for h in lane.history:
            d_lines.append(f"{h['iteration']:>4} {h['tracks']:>6} {h['shapes']:>6} {h['simulations_total']:>6} {h['best_E']:>8.3f} "
                           f"{h['gain']:>+8.2%} {h['stall']:>5}  " + ", ".join(f"{n} {v:.3f}" for n, v in h['pick_MAE'].items()))
        if lane.ft_reports:
            kept = sum(r["accepted"] for r in lane.ft_reports)
            d_lines.append(f"Fine-tuning: {kept} of {len(lane.ft_reports)} updates kept")
        for t in lane.tracks + [r[0] for r in lane.removed]:
            reason = next((r for tr, r, _ in lane.removed if tr is t), "active at the end")
            d_lines.append(f"  track {t.track_id} ({t.origin}): E {t.centre_E:.3f}, radius {t.radius:.3f} steps, "
                           f"{len(t.history)} steps | {reason}")
        for d in designs.get(lane.name, []):
            cold = d["cold_check"]
            d_lines.append(f"  design {d['id']}: E {d['E']:.4f} ({d['config']}) all configs "
                           + ", ".join(f"{k}={fmt(v, 3)}" for k, v in d["E_all_configs"].items())
                           + f" | cold E {fmt(d['cold_E'], 3)} ({cold['verdict']}, pre-stall same branch {fmt(cold['pre_stall_same_branch'], 3)})")
            d_lines.append(f"     window {d['window']['region']} {fmt(d['window']['grid_distance'])} steps from the nearest dataset "
                           f"shape | filters {'pass' if d['filters']['passed'] else 'fail: ' + str(d['filters']['reason'])} | "
                           f"classifier {fmt(d['classifier_probability'], 3)}")
            d_lines.append("     predictions original: " + ", ".join(f"{k}={v:.3f}" for k, v in d["original_predictions"].items())
                           + (" | lane's updated models: " + ", ".join(f"{k}={v:.3f}" for k, v in d["lane_predictions"].items())
                              if d["lane_predictions"] else ""))
        d_lines.append("")
    d_lines += timing["lines"] + ["=" * width]
    atomic_write_text(ctx.run_dir / "detailed_results.txt", "\n".join(d_lines) + "\n")

    # --- tables, json, plots --- #
    pd.DataFrame([h for lane in lanes for h in lane.history]).to_csv(ctx.run_dir / "iterations.csv", index=False)
    lanes_of = defaultdict(set)
    for lane in lanes:
        for key in lane.keys:
            lanes_of[key].add(lane.name)
    sim_rows = []
    for (key, protocol), entry in ctx.store.entries.items():
        run = entry["run"] or {}
        for ident in ctx.identifiers:
            sim_rows.append({"id": entry["id"], "protocol": protocol, "stage": entry["stage"], "iteration": entry["iteration"],
                             "lanes": "+".join(sorted(lanes_of.get(key, []))), "config": ident,
                             **dict(zip(ctx.cst_names, entry["cst"])), "status": run.get("status"), "cd_min": run.get("cd_min"),
                             "CL": run.get("CL", {}).get(ident), "CD": run.get("CD", {}).get(ident), "E": run.get("E", {}).get(ident),
                             "valid_E": valid_E(run, ident, ctx)})
    pd.DataFrame(sim_rows).to_csv(ctx.run_dir / "simulations.csv", index=False)
    write_json(ctx.run_dir / "detailed_results.json", {
        "outcome": overall, "lane_outcomes": outcomes, "target_E": target, "dataset_best": ctx.data.best,
        "reference_resimulated_E": resim, "peaks": ctx.peaks, "designs": designs,
        "simulations_to_reach": {k: v["reached"] for k, v in reach.items()},
        "lanes": [lane.to_dict() for lane in lanes], "expansion": expansion, "timing": timing})
    save_plots(ctx, lanes, reach, target)
    return overall, outcomes


# =============================================================================================== #
# === Main === #
# =============================================================================================== #

def main():
    args = parse_args()
    resume_dir, previous = load_previous_run(args) if args.resume else (None, None)
    if args.wing_mode == SINGLE_MODE:
        print(SINGLE_UNAVAILABLE_MESSAGE)
        sys.exit(1)
    if not torch.cuda.is_available():
        print("ERROR: Execution stopped. This script requires a CUDA-enabled GPU.")
        sys.exit(1)
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    session_start = time.perf_counter()
    start_time = datetime.now()
    dataset_num = args.dataset
    paths = build_paths(dataset_num)
    if args.csv:
        paths.training_file = Path(args.csv).resolve()
    hours = float(args.hours if args.hours is not None else DEFAULT_HOURS)
    workers = int(args.workers or max(1, multiprocessing.cpu_count() - 1))
    print(f"Dataset CSV: {paths.training_file}")
    print(f"Started at {start_time.strftime('%Y-%m-%d %H:%M:%S')} | time ceiling {hours:g} h | {workers} simulation worker(s)")
    if previous is None:
        report_unfinished_runs(paths.results_root, dataset_num)

    master_seed = args.seed if args.seed is not None else secrets.randbelow(MAX_MASTER_SEED + 1)
    print(f"Master seed: 0x{master_seed:x} = {master_seed} (decimal)")
    round_seeds = np.random.default_rng(master_seed).choice(SEED_POOL_SIZE, size=NUM_INFERENCE_RUNS, replace=False).tolist()

    # --- Dataset and models --- #
    feature_columns, cst_l_cols, cst_u_cols, config_cols = parse_dataset_columns(paths.training_file, args.wing_mode)
    cst_names = cst_l_cols + cst_u_cols
    n_cst = len(cst_l_cols)
    identifier_of = config_identifiers(config_cols)
    for path in (paths.naca_kp_file, paths.case_file, paths.base_airfoil, paths.xfoil_bin, paths.avl_bin):
        if not Path(path).exists():
            raise FileNotFoundError(f"Required file not found: {path}")
    resolved = {name: resolve_regressor(name, paths.models_dir) for name in args.models}
    print("\n# === Loading AI Models === #")
    with contextlib.redirect_stdout(io.StringIO()):
        classifier = load_classifier(args.classifier, paths.classifier_dir)
    print(f"  {args.classifier}: classifier (validity probability, reported only)")
    models, skipped, model_info = {}, [], {}
    for name, (folder, mtype) in resolved.items():
        try:
            model = load_regressor(name, folder, mtype)
        except ModelIncompatibleError as e:
            print(f"[{name}] {MLP_INCOMPATIBLE_MESSAGE}")
            skipped.append({"name": name, "type": mtype, "reason": str(e)})
            continue
        if model is None:
            print(f"[{name}] {SKNN_UNAVAILABLE_MESSAGE}")
            skipped.append({"name": name, "type": mtype, "reason": "functionality currently unavailable"})
            continue
        models[name] = model
        model_info[name] = (mtype, getattr(model, 'feature_mode', 'raw'))
        print(f"  {name}: {TYPE_LABELS.get(mtype, mtype)}, {model_info[name][1]} features "
              f"({folder.relative_to(paths.models_dir.parent) if folder.is_relative_to(paths.models_dir.parent) else folder})")
    if not models:
        print("\nNo usable regressors remain. There is nothing left to calculate. Congratulations.")
        sys.exit(1)
    model_names = list(models)
    fingerprints = {"dataset_csv": file_fingerprint([paths.training_file]),
                    "classifier": model_fingerprint(paths.classifier_dir, args.classifier),
                    "models": {n: model_fingerprint(resolved[n][0], n) for n in models}}
    if previous is not None:
        check_resume_fingerprints(previous, fingerprints)

    print("\n# === Dataset === #")
    data = load_dataset(paths.training_file, feature_columns, cst_names, config_cols)
    lows, highs = window_bounds(n_cst)
    run_dir = resume_dir if previous is not None else make_run_dir(paths.results_root, master_seed)
    print(f"Results folder: {run_dir}")

    ctx = SimpleNamespace(paths=paths, dataset_num=dataset_num, run_dir=run_dir, master_seed=master_seed, data=data,
                          feature_columns=feature_columns, cst_names=cst_names, n_cst=n_cst,
                          cst_col_idx=[feature_columns.index(c) for c in cst_names], config_cols=config_cols,
                          n_configs=len(config_cols), identifier_of=identifier_of,
                          identifiers=[identifier_of[c] for c in config_cols], lows=lows, highs=highs,
                          cd_limit=laminar_cd_limit(RE), workers=workers, temp_root=temp_root(), no_update=args.no_update,
                          model_names=model_names, model_info=model_info, resolved=resolved,
                          airfoil_creator=Airfoil_Generator(base_airfoil_csv=paths.base_airfoil, n_cst=n_cst, verbose=False))
    base_settings = dict(n_cst=n_cst, aseq=[float(a) for a in ASEQ], reynolds=RE, mach=MACH, base_airfoil=str(paths.base_airfoil),
                         naca_kp_file=str(paths.naca_kp_file), case_file=str(paths.case_file), xfoil_exe=str(paths.xfoil_bin),
                         avl_exe=str(paths.avl_bin), kp_seed=KP_SEED)
    ctx.settings = {SWEEP_MODE: Simulation_Settings(**base_settings, sweep_mode=SWEEP_MODE),
                    "cold": Simulation_Settings(**base_settings, sweep_mode="cold")}
    geometry_cache = {}

    def geometry_ok(cst) -> bool:
        key = shape_key(cst)
        if key not in geometry_cache:
            x, yu, yl = ctx.airfoil_creator.generate(cst_coeffs=np.asarray(cst, dtype=np.float64))
            geometry_cache[key] = bool(check_validity(x, yu, yl, verbose=False))
        return geometry_cache[key]

    ctx.geometry_ok = geometry_ok
    ctx.dataset_keys = set(shape_key(c) for c in np.unique(data.cst, axis=0))
    finder = Dataset_Peaks(data.cst, data.configs, data.y, data.steps, PEAK_SHARE, MAX_START_TRACKS)
    ctx.peaks = finder.find()
    for p in ctx.peaks:
        p["config"] = identifier_of.get(p["config"], p["config"])
        p["cst"] = [float(v) for v in p["cst"]]
    ctx.peak_lines = finder.summary_lines(ctx.peaks, cst_names)
    print(f"Dataset best: E = {data.e_max:.4f} ({data.best['config']}) | grid steps {np.round(data.steps, 4).tolist()}")
    print("\n".join(ctx.peak_lines))

    lanes = build_lanes(model_names, args.lane, model_info)
    if args.lane and not lanes:
        sys.exit(f"[ERROR] Lane '{args.lane}' is not available (its model may have been skipped).")
    for lane in lanes:
        lane.models = {n: load_regressor(n, *resolved[n]) for n in lane.model_names}   # own copies (fine-tuned per lane)
    print(f"\nLanes ({len(lanes)}):" + "".join(f"\n  {lane.label:<14} {lane_description(lane, ctx)}" for lane in lanes))
    if args.no_update:
        print("[NOTE] --no-update: the models stay frozen for the whole run (ablation).")

    # --- Metadata and state --- #
    metadata_path = metadata_path_of(run_dir, dataset_num)
    metadata = previous or {}
    metadata.update({
        "run_status": "running", "dataset": dataset_num, "dataset_csv": str(paths.training_file), "wing_mode": args.wing_mode,
        "master_seed": {"hex": f"{master_seed:x}", "decimal": master_seed}, "round_seeds_hex": [f"{s:x}" for s in round_seeds],
        "classifier": {"name": args.classifier, "folder": str(paths.classifier_dir)},
        "models": [{"name": n, "type": resolved[n][1], "folder": str(resolved[n][0]), "feature_mode": model_info[n][1]} for n in models],
        "skipped_models": skipped, "lanes": [lane.name for lane in lanes], "lane_option": args.lane, "no_update": args.no_update,
        "hours": hours, "workers": workers, "fingerprints": fingerprints,
        "settings": {"iteration0": {"rounds": NUM_INFERENCE_RUNS, "uniform": NUM_SAMPLES, "local": NUM_LOCAL_SAMPLES,
                                    "edge": NUM_EDGE_SAMPLES, "top_in_window": TOP_K_IN_WINDOW, "top_edge": TOP_K_EDGE,
                                    "random_picks": RANDOM_PICKS_0, "references": REFERENCE_TOP},
                     "trust_region": {"peak_share": PEAK_SHARE, "max_start_tracks": MAX_START_TRACKS, "r0": Trust_Region.R0,
                                      "r_min": Trust_Region.R_MIN, "r_max": Trust_Region.R_MAX, "picks_per_track": PICKS_PER_TRACK,
                                      "uniform_samples": TR_UNIFORM_SAMPLES, "refine_samples": TR_REFINE_SAMPLES,
                                      "min_gain": MIN_GAIN, "stall_patience": STALL_PATIENCE, "drop_share": DROP_SHARE,
                                      "merge_steps": MERGE_STEPS},
                     "fine_tune": {"new_weight": FT_NEW_WEIGHT, "min_shapes": FT_MIN_SHAPES, "local_anchor_steps": FT_LOCAL_ANCHOR_STEPS,
                                   "local_anchor_max": FT_LOCAL_ANCHOR_MAX, "random_anchors": FT_RANDOM_ANCHORS},
                     "validation": {"cold_top": VALIDATION_TOP, "robust_designs": ROBUST_DESIGNS,
                                    "robust_perturbations": ROBUST_PERTURBATIONS, "robust_sigma_steps": ROBUST_SIGMA_STEPS},
                     "simulation": {"ASEQ": ASEQ, "Re": RE, "Mach": MACH, "sweep_mode": SWEEP_MODE, "kp_seed": KP_SEED,
                                    "laminar_cd_limit": ctx.cd_limit}},
        "peaks": ctx.peaks,
    })
    metadata.setdefault("started", start_time.strftime('%Y-%m-%d %H:%M:%S'))
    if previous is not None:
        metadata["resumed"] = previous.get("resumed", []) + [start_time.strftime('%Y-%m-%d %H:%M:%S')]
    ctx.metadata, ctx.metadata_path = metadata, metadata_path
    write_json(metadata_path, metadata)

    state_path = run_dir / STATE_FILE
    ctx.state = json.loads(state_path.read_text()) if state_path.exists() else {}
    previous_elapsed = float(ctx.state.get("elapsed_seconds", 0.0))
    ctx.elapsed = lambda: previous_elapsed + (time.perf_counter() - session_start)
    ctx.store = Simulation_Store(run_dir / SIMULATIONS_FILE, ctx)
    if ctx.store.entries:
        print(f"Saved simulations reloaded: {len(ctx.store.entries)} ({SIMULATIONS_FILE}).")
    if ctx.state.get("lanes"):
        saved = {d["name"]: d for d in ctx.state["lanes"]}
        for i, lane in enumerate(lanes):
            restored = Lane.from_dict(saved[lane.name])
            restored.models = lane.models
            lanes[i] = restored
        rebuild_fine_tunes(lanes, ctx)

    # --- Iteration 0 --- #
    shortlist_path = run_dir / SHORTLIST_FILE
    if shortlist_path.exists():
        shortlist = json.loads(shortlist_path.read_text())
        print(f"\nIteration-0 shortlist reloaded ({SHORTLIST_FILE}).")
    else:
        print("\n" + "=" * 90 + "\n# === Iteration 0: shared candidate search === #\n" + "=" * 90)
        shortlist = run_iteration0_inference(ctx, lanes, models, round_seeds)
        write_json(shortlist_path, shortlist)
    if ctx.state.get("stage") is None:
        for lane in lanes:
            for ref in shortlist["references"]:
                lane.request(ref["key"], 0)
            for p in shortlist["lanes"][lane.name]:
                lane.request(p["key"], 0)
                lane.picks.append({**p, "iteration": 0, "track": "iteration 0"})
        save_state(ctx, lanes, stage="iteration0_picked", iteration=0)
    if ctx.state.get("stage") == "iteration0_picked":
        shapes = {r["key"]: r["cst"] for r in shortlist["references"]}
        for picks in shortlist["lanes"].values():
            shapes.update({p["key"]: p["cst"] for p in picks})
        ctx.store.simulate(shapes, SWEEP_MODE, "iteration 0", 0, f"iteration 0, {len(shapes)} shapes x {ctx.n_configs} configurations")
        print("\n# === Iteration 0: tracks and model updates === #")
        target, resim = reference_target(ctx)
        print(f"Dataset best {data.e_max:.4f}" + (f", re-simulated {resim:.4f}" if resim is not None else " (re-simulation failed)"))
        for lane in lanes:
            start_tracks(lane, ctx, shortlist)
            n0 = len(shortlist["lanes"][lane.name])
            lane.history.append({"iteration": 0, "lane": lane.name, "tracks": len(lane.tracks), "shapes": n0,
                                 "simulations_total": len(lane.keys), "best_E": lane.monitor.best_E, "gain": 0.0,
                                 "stall": 0, "pick_MAE": pick_errors(lane.picks, ctx), "done": False})
            print(f"  {lane.label}: {len(lane.tracks)} track(s), best E {lane.monitor.best_E:.3f} | "
                  + ", ".join(f"{t.track_id} {t.centre_E:.3f} ({t.origin})" for t in lane.tracks))
        for lane in lanes:
            fine_tune_lane(lane, ctx, 0)
        save_state(ctx, lanes, stage="loop", iteration=0, phase="done", iteration_seconds=[])

    # --- Trust-region loop --- #
    stop_reason = ctx.state.get("stop_reason")
    while stop_reason is None and ctx.state.get("stage") == "loop":
        iteration = int(ctx.state["iteration"]) + (0 if ctx.state.get("phase") == "picked" else 1)
        active = [lane for lane in lanes if not lane.done]
        if not active:
            stop_reason = "every lane converged"
            break
        if ctx.state.get("phase") != "picked":
            if ctx.elapsed() >= hours * 3600:
                stop_reason = "time ceiling"
                break
            if MAX_ITERATIONS is not None and iteration > MAX_ITERATIONS:
                stop_reason = f"MAX_ITERATIONS ({MAX_ITERATIONS})"
                break
        t_iter = time.perf_counter()
        print("\n" + "=" * 90 + f"\n# === Iteration {iteration} | elapsed {hms(ctx.elapsed())} of {hms(hours * 3600)} | "
              f"{len(active)} active lane(s) === #\n" + "=" * 90)
        if ctx.state.get("phase") == "picked":
            picks = ctx.state["pending_picks"]
            print("Picks of this iteration reloaded (the run was interrupted while simulating them).")
        else:
            picks = pick_iteration(active, ctx, iteration)
            for lane in active:
                for p in picks.get(lane.name, []):
                    lane.request(p["key"], iteration)
                    lane.picks.append({**p, "iteration": iteration})
            save_state(ctx, lanes, phase="picked", iteration=iteration, pending_picks=picks)
        shapes = {p["key"]: p["cst"] for lane_picks in picks.values() for p in lane_picks}
        ctx.store.simulate(shapes, SWEEP_MODE, f"iteration {iteration}", iteration,
                           f"iteration {iteration}, {len(shapes)} shapes x {ctx.n_configs} configurations")
        print("\n# === Updates === #")
        for lane in active:
            update_lane(lane, picks.get(lane.name, []), ctx, iteration)
        print()
        for lane in active:
            if not lane.done:
                fine_tune_lane(lane, ctx, iteration)
        seconds = ctx.state.get("iteration_seconds", []) + [time.perf_counter() - t_iter]
        save_state(ctx, lanes, phase="done", iteration=iteration, pending_picks=None, iteration_seconds=seconds)
        print(f"\nIteration {iteration} took {hms(seconds[-1])} | elapsed {hms(ctx.elapsed())} of {hms(hours * 3600)}")
    if ctx.state.get("stage") == "loop":
        save_state(ctx, lanes, stage="final", stop_reason=stop_reason)
    stop_reason = ctx.state.get("stop_reason")
    iterations_done = int(ctx.state.get("iteration", 0))

    # --- Final validation, reports, versions --- #
    print("\n" + "=" * 90 + f"\n# === Final validation (loop ended: {stop_reason}) === #\n" + "=" * 90)
    designs = final_validation(lanes, ctx, models, classifier)
    reach = {lane.name: simulations_to_reach(lane, ctx) for lane in lanes}
    expansion = save_expansion(ctx, lanes, iterations_done)
    elapsed = ctx.elapsed()
    still_active = [lane.label for lane in lanes if not lane.done]
    timing_lines = [f"Loop iterations completed: {iterations_done} | stop: {stop_reason}"]
    if stop_reason == "time ceiling" and still_active:
        timing_lines.append(f"[NOTE] The {hours:g} h ceiling was reached while {', '.join(still_active)} "
                            f"{'was' if len(still_active) == 1 else 'were'} still improving: another iteration was needed.")
    if elapsed < hours * 3600:
        timing_lines.append(f"Finished {hms(hours * 3600 - elapsed)} earlier than the {hours:g} h ceiling "
                            f"(active run time {hms(elapsed)}).")
    else:
        timing_lines.append(f"Active run time {hms(elapsed)} (ceiling {hours:g} h; the last iteration and the validation "
                            f"may run past it, an iteration is never cut).")
    timing = {"lines": timing_lines, "elapsed_seconds": elapsed, "stop_reason": stop_reason,
              "iterations": iterations_done, "iteration_seconds": ctx.state.get("iteration_seconds", [])}
    overall, outcomes = write_reports(ctx, lanes, designs, reach, expansion, timing)
    save_state(ctx, lanes, stage="completed")
    metadata.update({"run_status": "completed", "finished": now_text(), "active_run_time": hms(ctx.elapsed()),
                     "outcome": overall, "lane_outcomes": outcomes, "iterations": iterations_done, "stop_reason": stop_reason,
                     "simulations": len(ctx.store.entries), "expansion": expansion})
    write_json(metadata_path, metadata)
    print(f"\nProcess completed at {now_text()} | active run time {hms(ctx.elapsed())}")


if __name__ == "__main__":
    main()
