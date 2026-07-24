import os
import json
import shutil
import hashlib
import warnings
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from adjustText import adjust_text

from cst_modeling.foil import check_validity
from simulation_support import Xfoil_Polar_Creator as XPR, Key_Point_Extraction
from geometric_support import Airfoil_Generator, Validate_Airfoil, create_modification_combinations

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
warnings.filterwarnings("ignore", category=UserWarning)

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent

support_files_dir = project_root / "data" / "support"
binaries_dir = project_root / "bin"

validation_root = project_root / "data" / "singular_test_data" / "gpr_validation_data"
validation_airfoils_dir = validation_root / "airfoils"
validation_polars_dir = validation_root / "polars"
validation_results_dir = validation_root / "results"
manifest_path = validation_root / "manifest.json"

validation_temp_root = Path(r"C:\gpr_validation_temp")
validation_temp_airfoils = validation_temp_root / "airfoils"
validation_temp_polars = validation_temp_root / "polars"

BASE_AIRFOIL_PATH = support_files_dir / 'naca0012_unit.csv'
N_CST = 5
N_TEST_AIRFOILS = 32
RANDOM_SEED = 42
GPR_RANDOM_STATE = 42  # fixes GaussianProcessRegressor's optimizer restarts; see gpr_polar_fit
MAX_POLAR_RETRIES = 2  # additional XFOIL attempts for a polar that fails to complete (3 total attempts)

# gpr_polar_fit noise floors: flat (not AoA-dependent). Starting priors, not validated against real data.
CL_NOISE_FLOOR = 1e-4
CD_LOG_NOISE_FLOOR = 1e-4

# Dense ground-truth polar conditions (matches production settings)
ASEQ = [-20, 20, 0.1]
RE = 2.5e5
MACH = 0.09281

# Sparse resolutions to sweep during benchmarking
RESOLUTION_STEPS_TO_TEST = sorted([0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0])

# Geometric validity filters (same thresholds as Hossain_DB_Gen.py)
GEO_FILTER_PARAMS = {
    'threshold_max_t': 0.25,
    'threshold_min_t': 0.03,
    'threshold_t_x_max': 0.45,
    'threshold_t_x_min': 0.18,
    'activate_curvature_cont': True,
    'threshold_rLE': 0.01,
    'monotonic_activation': True,
    'threshold_max_camber': 0.12,
    'threshold_min_camber': -0.02
}

KEY_POINT_ERROR_THRESHOLD_PCT = 5.0

XFOIL_COMPLETION_FLAG = "XFOIL process completed successfully"
XFOIL_FAILURE_FLAG = "XFOIL process is not viable for this geometry."

COLORS = {
    # plot_aggregate_summary
    'agg_classical_line': '#3B9574', 
    'agg_classical_fill': '#81C6AA',
    'agg_gpr_line': '#4A4063',
    'agg_gpr_fill': '#D1B8E1',
    'agg_threshold': '#2A4E68',      

    # diagnostic & excluded plots
    'diag_ground_truth': '#808080', 
    'diag_sparse_input': '#2A4E68', 
    'diag_classical': '#3B9574', 
    'diag_gpr': '#4A4063', 
    'diag_gpr_fill': '#D1B8E1', 
    'diag_excluded_points': '#000000',
    'diag_excluded_title': '#8B0000', 
    
    # geometry plots
    'geo_line': '#000000',
    'geo_fill': '#808080'
}

def polar_is_complete(polar_path):
    """
    Returns True only if the polar file exists AND contains XFOIL's own
    completion flag. A file that exists but lacks the flag means XFOIL was
    invoked but never finished -- crashed, hung, or errored out partway
    through the requested alpha sequence -- and must never be treated as
    valid ground truth, cached, or parsed.
    """
    if not os.path.exists(polar_path):
        return False
    with open(polar_path, 'r') as f:
        return XFOIL_COMPLETION_FLAG in f.read()

def polar_is_unviable(polar_path):
    """
    Returns True if the polar file exists and contains the failure flag,
    indicating XFOIL exhausted all retries for this geometry.
    """
    if not os.path.exists(polar_path):
        return False
    with open(polar_path, 'r') as f:
        return XFOIL_FAILURE_FLAG in f.read()

def polar_status(polar_path):
    """
    Diagnostic-only description of a polar file's state (logging, not
    control flow -- polar_is_complete is still the boolean used
    everywhere else). Distinguishes "never written at all" from "written
    but never finished", which polar_is_complete alone collapses into the
    same False.
    """
    if not os.path.exists(polar_path):
        return "file does not exist"
    size = os.path.getsize(polar_path)
    if size == 0:
        return "file exists but is empty (0 bytes)"
    with open(polar_path, 'r') as f:
        content = f.read()
    if XFOIL_COMPLETION_FLAG in content:
        return "complete"
    if XFOIL_FAILURE_FLAG in content:
        return "unviable (failed all retries)"
    return f"file exists ({size} bytes, {content.count(chr(10)) + 1} lines) but missing completion flag"

def compute_data_signature(base_airfoil_path, n_cst, cmb_datalist, aseq, re, mach):
    """
    Builds a stable signature from the declared generation parameters so an
    existing dataset can be validated for reuse without re-running XFOIL.
    """
    hasher = hashlib.sha256()
    with open(base_airfoil_path, 'rb') as f:
        hasher.update(f.read())
    hasher.update(str(n_cst).encode())
    hasher.update(np.round(np.asarray(cmb_datalist, dtype=float), 8).tobytes())
    hasher.update(str(list(aseq)).encode())
    hasher.update(f"{re}_{mach}".encode())
    return hasher.hexdigest()

def load_manifest(path):
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return None

def save_manifest(path, signature, n_airfoils, aseq, re, mach, n_cst):
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        'signature': signature,
        'n_airfoils': n_airfoils,
        'aseq': list(aseq),
        're': re,
        'mach': mach,
        'n_cst': n_cst,
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(path, 'w') as f:
        json.dump(manifest, f, indent=2)

def signature_matches_manifest(manifest_path, signature):
    """
    Whether the declared generation parameters match what's recorded in
    the manifest. Deliberately says nothing about per-file completeness --
    that's handled separately in prepare_validation_dataset, which repairs
    individual incomplete polars rather than treating any incompleteness
    as a reason to discard and regenerate the whole cached dataset.
    """
    manifest = load_manifest(manifest_path)
    return manifest is not None and manifest.get('signature') == signature

def build_validation_cst_combinations(n_cst, n_target, seed=RANDOM_SEED):
    """
    Builds a diverse, small CST combination set for validation. Oversamples
    the combinatorial grid (candidates fail geometric validity filters at
    generation time, so more candidates than n_target are drawn) and
    randomly downselects for reproducibility. The unmodified base airfoil
    (all-zero CST deltas) is always included as entry 0.

    Range step sizes are widened relative to production
    (Hossain_DB_Gen.py's main block) to keep the combinatorial grid a
    manageable size for a validation-scale sample rather than a full
    training dataset sweep.
    """
    ranges_upper = {'LE': (0, 0.125, 0.0625), 'inter': (0, 0.125, 0.0625), 'TE': (0, 0.125, 0.125)}
    ranges_lower = {'LE': (-0.125, 0, 0.0625), 'inter': (0, 0.125, 0.0625), 'TE': (0, 0.125, 0.125)}

    full_grid = create_modification_combinations(n_cst, ranges_upper, ranges_lower, debug=False)

    rng = np.random.default_rng(seed)
    baseline = np.zeros((1, full_grid.shape[1]))

    oversample_target = min(len(full_grid), n_target * 4)
    if len(full_grid) > oversample_target:
        idx = rng.choice(len(full_grid), size=oversample_target, replace=False)
        sampled = full_grid[idx]
    else:
        sampled = full_grid

    return np.vstack([baseline, sampled])

def generate_validation_airfoils(base_airfoil_path, n_cst, cmb_datalist, output_dir, n_target,
                                  geo_filter_params, verbose=True):
    """
    Generates up to n_target geometrically valid airfoils from a candidate
    CST combination list, applying the same validity filter chain as
    Airfoil_Database_Generation. Stops once n_target valid airfoils are
    collected, or exhausts the candidate list and warns if short.

    Not multiprocessed: pure CST generation + geometric filtering over a few
    dozen candidates completes in well under a second per candidate, so pool
    overhead overhead would outweigh any benefit at this scale.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "invalid_airfoils_log.txt"

    airfoil_creator = Airfoil_Generator(
        base_airfoil_csv=str(base_airfoil_path),
        n_cst=n_cst,
        verbose=False,
    )

    valid_paths = []
    for i, cmb in enumerate(tqdm(cmb_datalist, desc="Generating validation airfoils", disable=not verbose)):
        if len(valid_paths) >= n_target:
            break

        airfoil_name = "NACA0012_baseline" if i == 0 else f"GPRVal_{i}"
        x_mod, yu_mod, yl_mod = airfoil_creator.generate(cst_coeffs=cmb, show_plot=False, debug=False)

        validity = check_validity(x_mod, yu_mod, yl_mod, verbose=False)
        log_reason = None

        if validity:
            try:
                va = Validate_Airfoil(
                    x=x_mod, yu=yu_mod, yl=yl_mod,
                    cst_coeffs=cmb, n_cst=n_cst, name=airfoil_name, verbose=False,
                )
                valid = 'valid'
                valid, log_reason = va.thickness_filters(
                    th_t_min=geo_filter_params['threshold_min_t'],
                    th_t_max=geo_filter_params['threshold_max_t'],
                    th_t_x_max=geo_filter_params['threshold_t_x_max'],
                    th_t_x_min=geo_filter_params['threshold_t_x_min'],
                )
                if valid == 'valid':
                    valid, log_reason = va.sharpness_filter(
                        curv_cont=geo_filter_params['activate_curvature_cont'],
                        th_rLE=geo_filter_params['threshold_rLE'],
                    )
                if valid == 'valid' and geo_filter_params['monotonic_activation']:
                    valid, log_reason = va.monotonic_thickness_filter()
                if valid == 'valid':
                    valid, log_reason = va.camber_filters(
                        th_c_max=geo_filter_params['threshold_max_camber'],
                        th_c_min=geo_filter_params['threshold_min_camber'],
                    )
                validity = (valid == 'valid')
            except ValueError as e:
                validity = False
                log_reason = f"{airfoil_name}: Validate_Airfoil raised an unexpected error: {e}"

        if validity:
            airfoil_creator._create_dat_file(
                output_dir=output_dir, n_cst=n_cst, x=x_mod, yl=yl_mod, yu=yu_mod,
                file_name=airfoil_name, debug=False,
            )
            valid_paths.append(output_dir / f"{airfoil_name}.dat")
        elif log_reason:
            with open(log_path, "a") as log:
                log.write(f"{log_reason}\n")

    if len(valid_paths) < n_target:
        print(f"[WARNING] Only {len(valid_paths)}/{n_target} valid airfoils were generated from the "
              f"candidate pool. Widen the CST ranges or increase the oversampling factor if this "
              f"recurs.")

    return valid_paths

def gt_polar_worker(args):
    """
    Worker function to run a single dense ground-truth XFOIL polar.
    Module-level for multiprocessing picklability.

    Recovery logic ported from the project's production polar_worker
    (Hossain_DB_Gen.py): if a polar file exists but is missing the
    completion flag, resume from the last alpha value actually written --
    parsed from the last data row that starts with a number -- instead of
    restarting the whole alpha sequence from ASEQ[0]. Falls back to the
    full sequence if the file doesn't exist or has no parseable data row.
    This composes with generate_validation_polars' outer retry loop: each
    successive retry now only asks XFOIL to cover the angles still
    missing, rather than redoing the whole sweep and discarding whatever
    progress the previous attempt made.

    Validates completion on both ends: before running (to skip work
    already done, or resume partial work) and after running (to catch a
    silent failure where run_simulation returns without raising, but
    XFOIL itself crashed, hung, or exited before finishing the requested
    alpha sequence).
    """
    (airfoil_path, polar_path, aseq, re, mach, xfoil_exe,
     temp_root, airfoil_dir, polar_dir) = args

    if polar_is_complete(polar_path):
        return polar_path, True  # already complete, skip

    # --- Recovery logic: resume from the last alpha actually written --- #
    if os.path.exists(polar_path):
        with open(polar_path, 'r') as f:
            lines = f.readlines()

        last_alfa = None
        for line in reversed(lines):
            try:
                last_alfa = float(line.strip().split()[0])
                break
            except (ValueError, IndexError):
                continue

        if last_alfa is not None:
            start_angle = last_alfa + aseq[2]
            num_steps = int((aseq[1] - start_angle) / aseq[2]) + 1
            angles = [start_angle + j * aseq[2] for j in range(num_steps)]
        else:  # file exists but is empty or has no parseable data row
            num_steps = int((aseq[1] - aseq[0]) / aseq[2]) + 1
            angles = [aseq[0] + j * aseq[2] for j in range(num_steps)]
    else:  # file doesn't exist yet
        num_steps = int((aseq[1] - aseq[0]) / aseq[2]) + 1
        angles = [aseq[0] + j * aseq[2] for j in range(num_steps)]

    if not angles:
        # Data already covers the full requested range but the completion
        # flag is somehow still missing -- nothing left to (re)compute.
        return polar_path, polar_is_complete(polar_path)

    try:
        xfoil_case = XPR(
            xfoil_exe_path=xfoil_exe,
            main_dir=temp_root,
            airfoil_dir=airfoil_dir,
            polar_dir=polar_dir,
            verbose=False,
        )
        xfoil_case.run_simulation(
            airfoil_file_path=airfoil_path,
            polar_file_path=polar_path,
            alfa_sequence=angles,
            visc=re,
            mach=mach,
        )
    except Exception as e:
        print(f"[ERROR] Ground-truth polar generation raised an exception for "
              f"{os.path.basename(airfoil_path)}: {e}")
        return polar_path, False

    if not polar_is_complete(polar_path):
        print(f"[ERROR] XFOIL did not report successful completion for {os.path.basename(airfoil_path)} "
              f"-- run_simulation returned without an exception. Polar status: {polar_status(polar_path)}.")
        return polar_path, False

    return polar_path, True

def generate_validation_polars(airfoil_paths, aseq, re, mach, temp_root, output_dir, max_retries=MAX_POLAR_RETRIES):
    """
    Runs XFOIL for the given validation airfoils at the dense ground-truth
    resolution, pooled at cpu_count() - 1 workers. Any airfoil whose polar
    doesn't complete (per polar_is_complete) is retried, only for that
    airfoil, up to max_retries additional times before being given up on --
    an XFOIL failure is sometimes transient (temp-file contention under
    parallel execution, a timing hiccup) rather than a genuine geometry
    problem, so one failed attempt shouldn't immediately exclude the
    airfoil.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    xfoil_exe = str(binaries_dir / 'xfoil.exe')

    remaining = list(airfoil_paths)
    attempt = 0

    while remaining and attempt <= max_retries:
        attempt_label = "initial attempt" if attempt == 0 else f"retry {attempt}/{max_retries}"

        tasks = []
        polar_path_by_airfoil = {}
        for ap in remaining:
            polar_path = str(output_dir / f"polar_{Path(ap).stem}.txt")
            polar_path_by_airfoil[str(ap)] = polar_path
            tasks.append((
                str(ap), polar_path, aseq, re, mach, xfoil_exe,
                str(temp_root), str(Path(ap).parent), str(output_dir),
            ))

        num_workers = max(1, min(cpu_count() - 1, len(tasks)))
        outcomes = {}
        with Pool(processes=num_workers, maxtasksperchild=100) as pool:
            for polar_path, ok in tqdm(pool.imap_unordered(gt_polar_worker, tasks), total=len(tasks),
                                        desc=f"Generating ground-truth polars ({attempt_label})"):
                outcomes[polar_path] = ok

        still_failing = [ap for ap in remaining if not outcomes.get(polar_path_by_airfoil[str(ap)], False)]

        if still_failing and attempt < max_retries:
            print(f"[INFO] {len(still_failing)} polar(s) failed on the {attempt_label}, retrying: "
                  f"{[Path(ap).stem for ap in still_failing]}")

        remaining = still_failing
        attempt += 1

    if remaining:
        print(f"[WARNING] {len(remaining)} polar(s) still failed to complete after {max_retries} "
              f"retr{'y' if max_retries == 1 else 'ies'}: {[Path(ap).stem for ap in remaining]}")

    return [output_dir / f"polar_{Path(ap).stem}.txt" for ap in airfoil_paths]

def preserve_failed_polars(temp_polar_dir, airfoil_paths, dest_dir):
    """
    Copies any incomplete polar file (partial content, or genuinely empty)
    out of the temp directory before it gets deleted, so a mid-run XFOIL
    crash leaves inspectable evidence behind instead of being silently
    destroyed by cleanup along with everything else in the temp root.
    Complete polars are left alone -- they already get persisted normally.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    preserved = []
    for ap in airfoil_paths:
        src = temp_polar_dir / f"polar_{Path(ap).stem}.txt"
        if src.exists() and not polar_is_complete(src):
            dest = dest_dir / src.name
            shutil.copy2(src, dest)
            preserved.append(dest)
    if preserved:
        print(f"[INFO] Preserved {len(preserved)} incomplete polar file(s) for inspection in "
              f"{dest_dir}: {[p.name for p in preserved]}")
    return preserved

def _persist_temp_dataset(airfoil_paths_temp):
    """
    Copies generated airfoils + completed polars from the short-path temp
    location to the final project data directory. Only airfoils with a
    verified-complete polar are included in the returned dataset.
    """
    dataset = []
    for ap in airfoil_paths_temp:
        dest_airfoil = validation_airfoils_dir / ap.name
        shutil.copy2(ap, dest_airfoil)

        src_polar = validation_temp_polars / f"polar_{ap.stem}.txt"
        if polar_is_complete(src_polar):
            dest_polar = validation_polars_dir / src_polar.name
            shutil.copy2(src_polar, dest_polar)
            dataset.append((dest_airfoil, dest_polar))
        else:
            print(f"[WARNING] Ground-truth polar for {ap.stem} did not complete successfully after "
                  f"retries and will not be included in the validation set.")
            with open(src_polar, 'a') as f:
                f.write(f"\n{XFOIL_FAILURE_FLAG}\n")

    invalid_log_src = validation_temp_airfoils / "invalid_airfoils_log.txt"
    if invalid_log_src.exists():
        validation_results_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(invalid_log_src, validation_results_dir / "invalid_airfoils_log.txt")

    return dataset

def _generate_fresh_dataset(cmb_datalist, signature, max_polar_retries):
    """
    Full from-scratch path: clears any stale cached data, generates all
    validation airfoils and their ground-truth polars, persists the
    results, and writes a fresh manifest. Used when the declared
    generation parameters changed (or force_regenerate was requested) --
    in that case there's nothing valid left to selectively repair.
    """
    print("[INFO] No cached dataset matches the declared parameters (or regeneration was forced). "
          "Generating validation airfoils and ground-truth polars from scratch.")

    for f in validation_airfoils_dir.glob("*.dat"):
        f.unlink()
    for f in validation_polars_dir.glob("*.txt"):
        f.unlink()

    validation_temp_airfoils.mkdir(parents=True, exist_ok=True)
    validation_temp_polars.mkdir(parents=True, exist_ok=True)

    airfoil_paths_temp = generate_validation_airfoils(
        BASE_AIRFOIL_PATH, N_CST, cmb_datalist, validation_temp_airfoils, N_TEST_AIRFOILS, GEO_FILTER_PARAMS
    )

    generate_validation_polars(airfoil_paths_temp, ASEQ, RE, MACH, validation_temp_root, validation_temp_polars,
                                max_retries=max_polar_retries)

    dataset = _persist_temp_dataset(airfoil_paths_temp)

    preserve_failed_polars(validation_temp_polars, airfoil_paths_temp, validation_results_dir / "failed_polars")
    shutil.rmtree(validation_temp_root, ignore_errors=True)

    if len(dataset) < len(airfoil_paths_temp):
        print(f"[WARNING] {len(airfoil_paths_temp) - len(dataset)} airfoil(s) lack a valid ground-truth "
              f"polar and were excluded from the validation set.")

    save_manifest(manifest_path, signature, len(dataset), ASEQ, RE, MACH, N_CST)
    return dataset

def prepare_validation_dataset(force_regenerate=False, max_polar_retries=MAX_POLAR_RETRIES):
    """
    Detects whether a cached validation dataset matches the declared
    generation parameters (via manifest signature).

    - If the parameters changed, or force_regenerate is set: nothing
      cached can be trusted, so everything is regenerated from scratch
      (_generate_fresh_dataset).
    - If the parameters match: the cache is reused, but every cached
      polar is checked for completion first. Any that are missing or
      incomplete (XFOIL crashed, hung, or was interrupted on a prior run)
      are repaired by re-running XFOIL for ONLY those specific airfoils,
      up to max_polar_retries additional attempts each -- not by
      discarding and regenerating the entire cached set over a handful
      of bad files.

    Returns
    -------
    list[tuple[Path, Path]]
        (airfoil_path, polar_path) pairs for the validation set.
    """
    validation_airfoils_dir.mkdir(parents=True, exist_ok=True)
    validation_polars_dir.mkdir(parents=True, exist_ok=True)

    cmb_datalist = build_validation_cst_combinations(N_CST, N_TEST_AIRFOILS, seed=RANDOM_SEED)
    signature = compute_data_signature(BASE_AIRFOIL_PATH, N_CST, cmb_datalist, ASEQ, RE, MACH)

    if force_regenerate or not signature_matches_manifest(manifest_path, signature):
        return _generate_fresh_dataset(cmb_datalist, signature, max_polar_retries)

    manifest = load_manifest(manifest_path)
    expected_count = manifest.get('n_airfoils', 0)
    airfoil_paths = sorted(validation_airfoils_dir.glob("*.dat"))

    if len(airfoil_paths) < expected_count:
        print(f"[WARNING] Only {len(airfoil_paths)}/{expected_count} cached airfoil geometries are "
              f"present -- more than just polars are missing. Falling back to a full regeneration.")
        return _generate_fresh_dataset(cmb_datalist, signature, max_polar_retries)

    dataset = []
    incomplete = []
    failed_polars_dir = validation_results_dir / "failed_polars"
    for ap in airfoil_paths:
        pp = validation_polars_dir / f"polar_{ap.stem}.txt"
        if polar_is_complete(pp):
            dataset.append((ap, pp))
        else:
            failed_pp = failed_polars_dir / f"polar_{ap.stem}.txt"
            if failed_pp.exists() and polar_is_unviable(failed_pp):
                pass
            else:
                incomplete.append(ap)

    if not incomplete:
        print(f"[INFO] Existing validation dataset matches the declared parameters and all "
              f"{len(dataset)} polars are complete. Reusing cached data.")
        return dataset

    print(f"[INFO] {len(incomplete)}/{len(airfoil_paths)} cached polar(s) are missing or incomplete. "
          f"Repairing by re-running XFOIL for: {[ap.stem for ap in incomplete]}")

    validation_temp_root.mkdir(parents=True, exist_ok=True)
    validation_temp_airfoils.mkdir(parents=True, exist_ok=True)
    validation_temp_polars.mkdir(parents=True, exist_ok=True)

    temp_airfoil_paths = []
    for ap in incomplete:
        dest = validation_temp_airfoils / ap.name
        shutil.copy2(ap, dest)
        temp_airfoil_paths.append(dest)

    generate_validation_polars(temp_airfoil_paths, ASEQ, RE, MACH, validation_temp_root, validation_temp_polars,
                                max_retries=max_polar_retries)

    for ap in incomplete:
        src_polar = validation_temp_polars / f"polar_{ap.stem}.txt"
        if polar_is_complete(src_polar):
            dest_polar = validation_polars_dir / src_polar.name
            shutil.copy2(src_polar, dest_polar)
            dataset.append((ap, dest_polar))
        else:
            print(f"[WARNING] {ap.stem}'s polar still failed to complete after repair attempts "
                  f"({polar_status(src_polar)}). Excluding it from this run.")
            with open(src_polar, 'a') as f:
                f.write(f"\n{XFOIL_FAILURE_FLAG}\n")

    preserve_failed_polars(validation_temp_polars, temp_airfoil_paths, validation_results_dir / "failed_polars")
    shutil.rmtree(validation_temp_root, ignore_errors=True)

    if len(dataset) != expected_count:
        save_manifest(manifest_path, signature, len(dataset), ASEQ, RE, MACH, N_CST)

    return dataset

def create_sparse_dataset(alpha_true, cl_true, cd_true, resolution_step):
    """
    Subsamples an existing (dense) polar at the requested alpha resolution
    by nearest-point selection, rather than re-running XFOIL. Isolates
    interpolation error from solver run-to-run variance.
    """
    if resolution_step <= 0:
        raise ValueError("Resolution step must be positive.")
    min_alpha, max_alpha = np.min(alpha_true), np.max(alpha_true)
    target_alphas = np.arange(min_alpha, max_alpha + resolution_step, resolution_step)
    sparse_indices = sorted({int(np.argmin(np.abs(alpha_true - ta))) for ta in target_alphas})
    return alpha_true[sparse_indices], cl_true[sparse_indices], cd_true[sparse_indices]

def robust_polar_fit(alpha_array, cl_array, cd_array, x_len=400):
    """
    Classical model: Savitzky-Golay smoothing + cubic spline interpolation.
    No uncertainty quantification available, so the bound slots are None.
    """
    n_points = len(alpha_array)
    if n_points < 4:
        return np.array([]), np.array([]), np.array([]), None, None, None, None

    alpha_interp = np.linspace(np.min(alpha_array), np.max(alpha_array), x_len)
    window_length = min(5, n_points // 2 * 2 + 1)
    cl_smooth = savgol_filter(cl_array, window_length, 2) if n_points >= 5 else cl_array
    cd_smooth = savgol_filter(cd_array, window_length, 2) if n_points >= 5 else cd_array

    cl_func = interp1d(alpha_array, cl_smooth, kind='cubic', bounds_error=False, fill_value="extrapolate")
    cd_func = interp1d(alpha_array, cd_smooth, kind='cubic', bounds_error=False, fill_value="extrapolate")

    return alpha_interp, cl_func(alpha_interp), cd_func(alpha_interp), None, None, None, None

def gpr_polar_fit(alpha_array, cl_array, cd_array, x_len=400):
    """
    GPR model: Matern(nu=1.5) kernel, flat noise floors via
    GaussianProcessRegressor's `alpha` parameter (CL_NOISE_FLOOR,
    CD_LOG_NOISE_FLOOR).

    C_D is fit in log-space: C_D is strictly positive and right-skewed,
    which violates the Gaussian-residual assumption a GP implicitly makes
    when fit directly. Fitting log(C_D) and exponentiating back respects
    positivity. C_L is left untransformed (signed, roughly symmetric).

    Because of the log transform, C_D's uncertainty band is NOT simply
    cd_pred +/- 2*std -- the +/-2 sigma interval is built in log-space
    first, then the two bounds are exponentiated independently, which
    comes out asymmetric (wider above, narrower below) as it should for a
    log-normal quantity. C_L's bounds are the ordinary symmetric mean
    +/- 2*std.

    Returns
    -------
    alpha_interp, cl_pred, cd_pred, cl_lower, cl_upper, cd_lower, cd_upper
        Both bound pairs are None if there isn't enough data to fit.
    """
    n_points = len(alpha_array)
    if n_points < 2:
        return np.array([]), np.array([]), np.array([]), None, None, None, None

    alpha_interp = np.linspace(np.min(alpha_array), np.max(alpha_array), x_len).reshape(-1, 1)
    alpha_X = alpha_array.reshape(-1, 1)

    kernel = ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) * Matern(
        length_scale=2.0, length_scale_bounds=(0.3, 15.0), nu=1.5
    )

    gpr_cl = GaussianProcessRegressor(
        kernel=kernel, alpha=CL_NOISE_FLOOR, n_restarts_optimizer=10, normalize_y=True,
        random_state=GPR_RANDOM_STATE,
    )
    gpr_cd = GaussianProcessRegressor(
        kernel=kernel, alpha=CD_LOG_NOISE_FLOOR, n_restarts_optimizer=10, normalize_y=True,
        random_state=GPR_RANDOM_STATE,
    )

    gpr_cl.fit(alpha_X, cl_array)

    # Tiny floor before the log to guard against a literal 0.0 in cd_array
    # (shouldn't occur in real XFOIL output, but log(0) would otherwise crash).
    cd_log_array = np.log(np.maximum(cd_array, 1e-12))
    gpr_cd.fit(alpha_X, cd_log_array)

    cl_pred, cl_std = gpr_cl.predict(alpha_interp, return_std=True)
    cl_lower, cl_upper = cl_pred - 2 * cl_std, cl_pred + 2 * cl_std

    cd_log_pred, cd_log_std = gpr_cd.predict(alpha_interp, return_std=True)
    cd_pred = np.exp(cd_log_pred)
    cd_lower = np.exp(cd_log_pred - 2 * cd_log_std)
    cd_upper = np.exp(cd_log_pred + 2 * cd_log_std)

    return alpha_interp.flatten(), cl_pred, cd_pred, cl_lower, cl_upper, cd_lower, cd_upper

MODEL_FIT_FUNCTIONS = {
    'classical': robust_polar_fit,
    'gpr': gpr_polar_fit,
}

def calculate_interpolation_rmse(alpha_true, cl_true, cd_true, alpha_interp, cl_interp, cd_interp):
    """Whole-curve RMSE of a model's fit against the dense ground truth."""
    if len(cl_interp) == 0:
        return np.inf, np.inf
    interp_opts = {'kind': 'linear', 'bounds_error': False, 'fill_value': 'extrapolate'}
    cl_func = interp1d(alpha_interp, cl_interp, **interp_opts)
    cd_func = interp1d(alpha_interp, cd_interp, **interp_opts)
    cl_pred = cl_func(alpha_true)
    cd_pred = cd_func(alpha_true)
    rmse_cl = float(np.sqrt(mean_squared_error(cl_true, cl_pred)))
    rmse_cd = float(np.sqrt(mean_squared_error(cd_true, cd_pred)))
    return rmse_cl, rmse_cd

def extract_key_points(cl_array, cd_array):
    """Extracts (CL, CD) at C_L,max / C_L,min / C_D,min."""
    if len(cl_array) == 0:
        return {'CL_max': (np.nan, np.nan), 'CL_min': (np.nan, np.nan), 'CD_min': (np.nan, np.nan)}
    return {
        'CL_max': (cl_array[np.argmax(cl_array)], cd_array[np.argmax(cl_array)]),
        'CL_min': (cl_array[np.argmin(cl_array)], cd_array[np.argmin(cl_array)]),
        'CD_min': (cl_array[np.argmin(cd_array)], cd_array[np.argmin(cd_array)]),
    }

def calculate_key_point_errors(true_points, pred_points, cl_range, cd_range):
    """
    Normalized Euclidean-distance error (%) per key point, combining the
    CL and CD components (each scaled by the ground truth's own dynamic
    range) into a single robust distance metric.
    """
    errors = {}
    cl_range = max(cl_range, 1e-9)
    cd_range = max(cd_range, 1e-9)
    for key in true_points:
        cl_t, cd_t = true_points[key]
        cl_p, cd_p = pred_points[key]
        if np.isnan(cl_p) or np.isnan(cd_p):
            errors[key] = np.inf
            continue
        err_cl = abs(cl_p - cl_t) / cl_range
        err_cd = abs(cd_p - cd_t) / cd_range
        errors[key] = float(np.sqrt(err_cl ** 2 + err_cd ** 2) * 100)
    return errors

def benchmark_worker(args):
    """
    Worker for a single (airfoil, resolution, model) task. Reports both
    whole-curve RMSE and key-point error independently rather than gating
    one on the other, so a model isn't marked as failed purely because its
    global fit oscillates somewhere away from the key points of interest.
    """
    (airfoil_name, resolution_step, model_name,
     alpha_true, cl_true, cd_true, key_points_true, cl_range, cd_range) = args

    base_result = {
        'airfoil': airfoil_name, 'resolution': resolution_step, 'model': model_name,
        'n_points': 0, 'rmse_cl': np.inf, 'rmse_cd': np.inf,
        'kp_error_max': np.inf, 'kp_error_CL_max': np.inf,
        'kp_error_CL_min': np.inf, 'kp_error_CD_min': np.inf,
    }

    try:
        alpha_sparse, cl_sparse, cd_sparse = create_sparse_dataset(alpha_true, cl_true, cd_true, resolution_step)
        n_points = len(alpha_sparse)
        base_result['n_points'] = n_points

        if n_points < 5:
            return base_result

        fit_func = MODEL_FIT_FUNCTIONS[model_name]
        alpha_i, cl_i, cd_i, *_ = fit_func(alpha_sparse, cl_sparse, cd_sparse)

        rmse_cl, rmse_cd = calculate_interpolation_rmse(alpha_true, cl_true, cd_true, alpha_i, cl_i, cd_i)
        key_points_pred = extract_key_points(cl_i, cd_i)
        kp_errors = calculate_key_point_errors(key_points_true, key_points_pred, cl_range, cd_range)

        base_result.update({
            'rmse_cl': rmse_cl,
            'rmse_cd': rmse_cd,
            'kp_error_max': max(kp_errors.values()) if kp_errors else np.inf,
            'kp_error_CL_max': kp_errors.get('CL_max', np.inf),
            'kp_error_CL_min': kp_errors.get('CL_min', np.inf),
            'kp_error_CD_min': kp_errors.get('CD_min', np.inf),
        })
        return base_result

    except Exception as e:
        base_result['error'] = str(e)
        return base_result

def run_benchmark_sweep(dataset, resolution_steps, kp_extractor):
    """
    Builds the full (airfoil x resolution x model) task grid and runs it in
    a single flattened pool at cpu_count() - 1 workers.

    Before an airfoil is admitted to the benchmark, its dense ground-truth
    polar must pass Key_Point_Extraction's own minimum-data-sufficiency
    gate (>=10 converged points, plus minimum coverage across six alpha
    bands spanning -20 to 20 deg -- see _interpolate_and_extract). This
    reuses the production gate directly by calling _interpolate_and_extract
    and catching Exception broadly, then reading type(e).__name__ to tell
    InsufficientPointsError / PoorDistributionError apart from a genuinely
    unexpected failure -- those two classes aren't re-exported from
    simulation_support's package __init__, so they can't be imported and
    matched by type. This still reuses the production logic exactly rather
    than re-implementing the six conditions here, avoiding the
    parsing/validity-logic drift seen across the earlier exploratory scripts.

    Airfoils that fail this gate -- typically because XFOIL failed to
    converge across large stretches of the alpha range, more common on
    aggressive geometry -- are excluded from the benchmark and aggregate
    statistics entirely, and returned separately for diagnostic plotting
    rather than silently degrading the results with unreliable ground truth.

    Parameters
    ----------
    dataset : list[tuple[Path, Path]]
        (airfoil_path, polar_path) pairs.
    resolution_steps : list[float]
    kp_extractor : Key_Point_Extraction
        Instance used for its canonical `_polar_extract` / `_polar_cleanup`
        parsing methods and its `_interpolate_and_extract` validity gate.

    Returns
    -------
    results : list[dict]
    ground_truth_cache : dict[str, tuple]
        {airfoil_name: (alpha_true, cl_true, cd_true)} for admitted airfoils.
    excluded_airfoils : dict[str, dict]
        {airfoil_name: {'airfoil_path', 'alpha', 'cl', 'cd', 'reason'}} for
        airfoils that failed the data-sufficiency gate.
    """
    tasks = []
    ground_truth_cache = {}
    excluded_airfoils = {}

    for airfoil_path, polar_path in dataset:
        airfoil_name = Path(airfoil_path).stem

        # Explicit completion check before any data splicing/interpolation. Upstream
        # generation and caching should already guarantee this, but this is the last
        # line of defense so a header-only (or missing) polar can never reach the parser.
        if not polar_is_complete(polar_path):
            status = polar_status(polar_path)
            print(f"[WARNING] {airfoil_name}'s polar is not usable ({status}) -- excluding from the "
                  f"benchmark before any parsing is attempted.")
            if os.path.exists(polar_path):
                # _polar_extract does a bare open() with no existence check -- only call it
                # when the file is actually there, or a missing file would crash this loop
                # instead of just being logged as an exclusion.
                alpha_list, cl_list, cd_list = kp_extractor._polar_extract(str(polar_path))
                alpha_raw, cl_raw, cd_raw = kp_extractor._polar_cleanup(alpha_list, cl_list, cd_list)
            else:
                alpha_raw, cl_raw, cd_raw = np.array([]), np.array([]), np.array([])
            excluded_airfoils[airfoil_name] = {
                'airfoil_path': airfoil_path, 'alpha': alpha_raw, 'cl': cl_raw, 'cd': cd_raw,
                'reason': f"IncompletePolarFile: {status}",
            }
            continue

        alpha_list, cl_list, cd_list = kp_extractor._polar_extract(str(polar_path))
        alpha_true, cl_true, cd_true = kp_extractor._polar_cleanup(alpha_list, cl_list, cd_list)

        if len(alpha_true) == 0:
            print(f"[WARNING] No valid ground-truth data parsed for {airfoil_name}, skipping entirely.")
            continue

        try:
            kp_extractor._interpolate_and_extract(str(polar_path))
        except Exception as e:
            exc_name = type(e).__name__
            if exc_name in ("InsufficientPointsError", "PoorDistributionError"):
                excluded_airfoils[airfoil_name] = {
                    'airfoil_path': airfoil_path, 'alpha': alpha_true, 'cl': cl_true, 'cd': cd_true,
                    'reason': f"{exc_name}: {e}",
                }
            else:
                # Not one of the two known data-sufficiency exceptions -- something
                # else broke inside _interpolate_and_extract. Still excluded (its
                # ground truth can't be trusted), but flagged distinctly so an
                # unexpected bug doesn't get silently filed away as a data issue.
                print(f"[WARNING] {airfoil_name} raised an unexpected error type ({exc_name}) during the "
                      f"data-sufficiency check, not one of the two known gate exceptions: {e}")
                excluded_airfoils[airfoil_name] = {
                    'airfoil_path': airfoil_path, 'alpha': alpha_true, 'cl': cl_true, 'cd': cd_true,
                    'reason': f"UNEXPECTED ({exc_name}): {e}",
                }
            continue

        key_points_true = extract_key_points(cl_true, cd_true)
        cl_range = float(np.ptp(cl_true))
        cd_range = float(np.ptp(cd_true))
        ground_truth_cache[airfoil_name] = (alpha_true, cl_true, cd_true)

        for res in resolution_steps:
            for model_name in MODEL_FIT_FUNCTIONS:
                tasks.append((
                    airfoil_name, res, model_name,
                    alpha_true, cl_true, cd_true, key_points_true, cl_range, cd_range,
                ))

    if excluded_airfoils:
        print(f"[INFO] {len(excluded_airfoils)} airfoil(s) excluded from the benchmark for failing the "
              f"minimum ground-truth data-sufficiency gate: {list(excluded_airfoils.keys())}")

    num_workers = max(1, cpu_count() - 1)
    results = []
    with Pool(processes=num_workers) as pool:
        for res in tqdm(pool.imap_unordered(benchmark_worker, tasks), total=len(tasks),
                         desc="Running GPR vs. classical benchmark"):
            results.append(res)

    return results, ground_truth_cache, excluded_airfoils

def export_exclusion_log(excluded_airfoils, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "excluded_airfoils_log.txt"
    with open(log_path, 'w') as f:
        f.write(f"Excluded airfoils: {len(excluded_airfoils)}\n")
        f.write("=" * 60 + "\n")
        for name, info in excluded_airfoils.items():
            f.write(f"{name}: {info['reason']} ({len(info['alpha'])} converged points)\n")
    return log_path

def aggregate_results(results):
    df = pd.DataFrame(results)
    df = df.replace([np.inf, -np.inf], np.nan)

    summary = df.groupby(['resolution', 'model']).agg(
        mean_kp_error=('kp_error_max', 'mean'),
        std_kp_error=('kp_error_max', 'std'),
        mean_rmse_cl=('rmse_cl', 'mean'),
        mean_rmse_cd=('rmse_cd', 'mean'),
        mean_n_points=('n_points', 'mean'),
        n_valid=('kp_error_max', lambda x: x.notna().sum()),
        n_total=('kp_error_max', 'size'),
    ).reset_index()

    return df, summary

def find_recommended_resolution(summary_df, model_name, threshold=KEY_POINT_ERROR_THRESHOLD_PCT):
    """
    Finds the coarsest resolution such that every finer (or equal)
    resolution tested also passed the threshold with zero failed airfoils.
    Scanning from fine to coarse and requiring an unbroken passing run
    avoids reporting a resolution that only "passes" due to a single
    non-monotonic dip in an otherwise failing region.
    """
    sub = summary_df[summary_df['model'] == model_name].sort_values('resolution').reset_index(drop=True)
    passing = (sub['mean_kp_error'] < threshold) & (sub['n_valid'] == sub['n_total'])

    recommended = None
    for i in range(len(sub)):
        if passing.iloc[:i + 1].all():
            recommended = sub.loc[i, 'resolution']
        else:
            break
    return recommended

def load_airfoil_geometry(dat_path):
    """
    Reads x,y coordinate pairs from an airfoil .dat file, skipping any
    non-numeric header/name line. Points are plotted in file order rather
    than re-sorted: any .dat file usable by XFOIL already traces a single
    continuous surface loop, so file order reproduces the shape correctly
    regardless of the specific writing convention used by
    Airfoil_Generator._create_dat_file (not independently verified against
    source -- check the first rendered geometry plot against a known
    airfoil shape on first run).
    """
    xs, ys = [], []
    with open(dat_path, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) != 2:
                continue
            try:
                xs.append(float(parts[0]))
                ys.append(float(parts[1]))
            except ValueError:
                continue
    return np.array(xs), np.array(ys)

def _plot_airfoil_shape(ax, x, y, title="Airfoil Geometry"):
    ax.plot(x, y, color=COLORS['geo_line'], linewidth=1.5)
    ax.fill(x, y, color=COLORS['geo_fill'], alpha=0.15)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x/c'); ax.set_ylabel('y/c')
    ax.set_title(title)

def plot_aggregate_summary(summary_df, output_dir):
    """
    Two separate Pareto-style views: mean key-point error vs. resolution (with
    +/-1 std band) and mean sparse-point count vs. resolution. Kept
    as two separate plots so the error/data-volume tradeoff stays legible.
    """
    plt.style.use('seaborn-v0_8-whitegrid')

    style = {
        'classical': (COLORS['agg_classical_line'], COLORS['agg_classical_fill'], 'o--'),
        'gpr': (COLORS['agg_gpr_line'], COLORS['agg_gpr_fill'], 'o-')
    }

    # --- Plot 1: KP Error vs Resolution ---
    fig1, ax1 = plt.subplots(figsize=(8, 7))
    for model_name, (line_color, fill_color, marker) in style.items():
        sub = summary_df[summary_df['model'] == model_name].sort_values('resolution')
        label_name = 'Savitzky-Golay + Splines cúbicos' if model_name == 'classical' else 'GPR'
        ax1.plot(sub['resolution'], sub['mean_kp_error'], marker, color=line_color, label=label_name)
        std = sub['std_kp_error'].fillna(0)
        ax1.fill_between(sub['resolution'], sub['mean_kp_error'] - std, sub['mean_kp_error'] + std,
                          color=fill_color, alpha=0.15, label=f'{label_name} \u00b1 1\u03c3')

    ax1.axhline(y=KEY_POINT_ERROR_THRESHOLD_PCT, color=COLORS['agg_threshold'], linestyle='--', linewidth=1.2,
                label=f'Umbral de error del {KEY_POINT_ERROR_THRESHOLD_PCT:.0f}%')
    ax1.set_xlabel('Paso de resolución (grados)')
    ax1.set_ylabel('Error medio máx. de puntos clave (%) \u00b1 1\u03c3')
    ax1.set_title(f'Error de puntos clave vs. Resolución (n={summary_df["n_total"].max()} perfiles)')
    ax1.set_ylim(bottom=0)
    ax1.legend()

    fig1.tight_layout()
    fig1.savefig(output_dir / "aggregate_summary_kp_error.png", dpi=150)
    plt.close(fig1)

    # --- Plot 2: Data Volume vs Resolution ---
    fig2, ax2 = plt.subplots(figsize=(8, 7))
    texts = []
    for model_name, (line_color, fill_color, marker) in style.items():
        sub = summary_df[summary_df['model'] == model_name].sort_values('resolution')
        ax2.plot(sub['resolution'], sub['mean_n_points'], marker, color=line_color)
        
        # Only add labels once to prevent duplicates (data volume is identical for both models)
        if model_name == 'classical':
            for x, y in zip(sub['resolution'], sub['mean_n_points']):
                texts.append(ax2.text(x, y, f'{y:.1f}', fontsize=10, color='black'))
            
    # Automatically adjust text to prevent overlaps
    adjust_text(texts, ax=ax2, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5, alpha=0.5))

    ax2.set_xlabel('Paso de resolución (grados)')
    ax2.set_ylabel('Número medio de puntos dispersos')
    ax2.set_title('Volumen de datos vs. Resolución')

    fig2.tight_layout()
    fig2.savefig(output_dir / "aggregate_summary_data_volume.png", dpi=150)
    plt.close(fig2)

def plot_airfoil_diagnostic(airfoil_name, airfoil_path, alpha_true, cl_true, cd_true, resolution_step, output_dir):
    """
    Per-airfoil QA plot at a given resolution: dense ground truth, sparse
    input, both model fits, and the GPR +/-2 sigma predictive uncertainty
    band on top, with the airfoil geometry itself along the bottom row.
    Only called for airfoils that passed the minimum-data-sufficiency gate.
    """
    alpha_sparse, cl_sparse, cd_sparse = create_sparse_dataset(alpha_true, cl_true, cd_true, resolution_step)
    alpha_c, cl_c, cd_c, *_ = robust_polar_fit(alpha_sparse, cl_sparse, cd_sparse)
    alpha_g, cl_g, cd_g, cl_lower, cl_upper, cd_lower, cd_upper = gpr_polar_fit(alpha_sparse, cl_sparse, cd_sparse)

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1.3])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    fig.suptitle(f'{airfoil_name} \u2014 Model Comparison at {resolution_step}\u00b0 Resolution',
                 fontsize=14, weight='bold')

    ax1.plot(alpha_true, cl_true, color=COLORS['diag_ground_truth'], linewidth=3, alpha=0.6, label='Ground Truth')
    ax1.plot(alpha_sparse, cl_sparse, marker='o', linestyle='', color=COLORS['diag_sparse_input'], markersize=5, label='Sparse Input')
    if len(alpha_c) > 0:
        ax1.plot(alpha_c, cl_c, linestyle='--', color=COLORS['diag_classical'], linewidth=2, label='Classical')
    if len(alpha_g) > 0:
        ax1.plot(alpha_g, cl_g, linestyle='-', color=COLORS['diag_gpr'], linewidth=2, label='GPR')
        if cl_lower is not None:
            ax1.fill_between(alpha_g, cl_lower, cl_upper, color=COLORS['diag_gpr_fill'], alpha=0.15, label='GPR \u00b12\u03c3')
    ax1.set_xlabel('Alpha (deg)'); ax1.set_ylabel('C_L')
    ax1.set_title('Lift Curve'); ax1.legend(fontsize=9)

    ax2.plot(alpha_true, cd_true, color=COLORS['diag_ground_truth'], linewidth=3, alpha=0.6, label='Ground Truth')
    ax2.plot(alpha_sparse, cd_sparse, marker='o', linestyle='', color=COLORS['diag_sparse_input'], markersize=5, label='Sparse Input')
    if len(alpha_c) > 0:
        ax2.plot(alpha_c, cd_c, linestyle='--', color=COLORS['diag_classical'], linewidth=2, label='Classical')
    if len(alpha_g) > 0:
        ax2.plot(alpha_g, cd_g, linestyle='-', color=COLORS['diag_gpr'], linewidth=2, label='GPR')
        if cd_lower is not None:
            # Band is asymmetric by construction (exponentiated log-space bounds) --
            # this is correct for a log-transformed, strictly-positive target.
            ax2.fill_between(alpha_g, cd_lower, cd_upper, color=COLORS['diag_gpr_fill'], alpha=0.15, label='GPR \u00b12\u03c3')
    ax2.set_xlabel('Alpha (deg)'); ax2.set_ylabel('C_D')
    ax2.set_title('Drag Curve'); ax2.legend(fontsize=9)

    x_geo, y_geo = load_airfoil_geometry(airfoil_path)
    _plot_airfoil_shape(ax3, x_geo, y_geo, title=f'{airfoil_name} Geometry')

    plt.tight_layout()
    safe_res = str(resolution_step).replace('.', 'p')
    out_path = output_dir / f"{airfoil_name}_diagnostic_{safe_res}deg.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

def plot_excluded_airfoil(airfoil_name, airfoil_path, alpha_true, cl_true, cd_true, reason, output_dir):
    """
    Diagnostic plot for an airfoil excluded from the benchmark for failing
    Key_Point_Extraction's minimum ground-truth data-sufficiency gate.
    Shows only the raw converged points -- fitting either model on data
    this sparse or poorly distributed would be misleading -- alongside the
    airfoil geometry, so the shape responsible for XFOIL's convergence
    trouble can be inspected directly.
    """
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1.3])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    fig.suptitle(f'{airfoil_name} \u2014 EXCLUDED (raw data only)\n{reason}',
                 fontsize=13, weight='bold', color=COLORS['diag_excluded_title'])

    ax1.plot(alpha_true, cl_true, marker='o', linestyle='-', color=COLORS['diag_excluded_points'], markersize=5, linewidth=1, label='Raw Converged Points')
    ax1.set_xlabel('Alpha (deg)'); ax1.set_ylabel('C_L')
    ax1.set_title(f'Lift Curve (n={len(alpha_true)} converged points)'); ax1.legend(fontsize=9)

    ax2.plot(alpha_true, cd_true, marker='o', linestyle='-', color=COLORS['diag_excluded_points'], markersize=5, linewidth=1, label='Raw Converged Points')
    ax2.set_xlabel('Alpha (deg)'); ax2.set_ylabel('C_D')
    ax2.set_title(f'Drag Curve (n={len(alpha_true)} converged points)'); ax2.legend(fontsize=9)

    x_geo, y_geo = load_airfoil_geometry(airfoil_path)
    _plot_airfoil_shape(ax3, x_geo, y_geo, title=f'{airfoil_name} Geometry')

    plt.tight_layout()
    out_path = output_dir / f"{airfoil_name}_EXCLUDED_raw.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

def export_results_csv(df, summary_df, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "gpr_validation_raw_results.csv", index=False)
    summary_df.to_csv(output_dir / "gpr_validation_summary.csv", index=False)

if __name__ == "__main__":
    start = datetime.now()
    print(f"GPR validation study started at {start.strftime('%Y-%m-%d %H:%M:%S')}")

    dataset = prepare_validation_dataset(force_regenerate=False)
    if not dataset:
        raise RuntimeError("No valid airfoil/polar pairs available for the validation study. Aborting.")
    print(f"[INFO] Validation dataset ready: {len(dataset)} airfoils.")

    kp_extractor = Key_Point_Extraction(polar_file_paths=[], verbose=False)

    results, ground_truth_cache, excluded_airfoils = run_benchmark_sweep(
        dataset, RESOLUTION_STEPS_TO_TEST, kp_extractor
    )

    if excluded_airfoils:
        export_exclusion_log(excluded_airfoils, validation_results_dir)

    df, summary = aggregate_results(results)
    export_results_csv(df, summary, validation_results_dir)

    plot_aggregate_summary(summary, validation_results_dir)

    recommended_gpr = find_recommended_resolution(summary, 'gpr')
    recommended_classical = find_recommended_resolution(summary, 'classical')
    print(f"[INFO] Recommended resolution \u2014 GPR: {recommended_gpr}\u00b0, "
          f"Classical: {recommended_classical}\u00b0 "
          f"(coarsest resolution below which every tested resolution also stayed under "
          f"{KEY_POINT_ERROR_THRESHOLD_PCT:.0f}% error with zero failed airfoils, "
          f"n={len(ground_truth_cache)} admitted airfoils, {len(excluded_airfoils)} excluded).")

    diagnostic_resolution = recommended_gpr if recommended_gpr is not None else RESOLUTION_STEPS_TO_TEST[0]
    airfoil_path_by_name = {Path(ap).stem: ap for ap, pp in dataset}

    for airfoil_name, (alpha_true, cl_true, cd_true) in ground_truth_cache.items():
        plot_airfoil_diagnostic(airfoil_name, airfoil_path_by_name[airfoil_name], alpha_true, cl_true, cd_true,
                                 diagnostic_resolution, validation_results_dir)

    for airfoil_name, info in excluded_airfoils.items():
        plot_excluded_airfoil(airfoil_name, info['airfoil_path'], info['alpha'], info['cl'], info['cd'],
                               info['reason'], validation_results_dir)

    end = datetime.now()
    print(f"GPR validation study completed at {end.strftime('%Y-%m-%d %H:%M:%S')} "
          f"(elapsed: {end - start}).")