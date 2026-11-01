"""
Builds a Hossain wing dataset: CST airfoils -> XFOIL polars -> key points and AVL geometry files -> AVL
forces -> training CSVs -> database mending (neighbour-consistency check and re-simulation of outliers).

Every stage is archived when it finishes, so an interrupted run resumes at the stage (and, inside a stage,
at the file) where it stopped. Each stage prints a short summary that is also saved to
data/logs/dataset_<N>/<n>_<stage>_summary.txt, next to the run's other logs (failed airfoils, failed XFOIL
angles, failed AVL runs, polars below the laminar drag limit).

Usage
-----
    python src/Hossain_DB_Gen.py               # builds dataset DATASET_NUM (set below)
    python src/Hossain_DB_Gen.py --smoke 40    # every stage on 40 airfoils, as dataset 'smoke' (pipeline test)
    python src/Hossain_DB_Gen.py --no-gui      # console progress bars instead of windows

Smoke runs use their own temp folders and the dataset name 'smoke'; delete data/datasets/dataset_smoke,
data/logs/dataset_smoke and data/dataset_build_data/*/*_dataset_smoke.rar afterwards.
"""
import argparse
import csv
import json
import numpy as np
import os
import queue
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from cst_modeling.foil import check_validity
from multiprocessing import Pool, Manager, cpu_count
# Import custom package class and methods
from simulation_support import Xfoil_Polar_Creator as XPR
from simulation_support import Key_Point_Extraction, HossainWingRec_Creation, HossainWing60DEG_Creation, HossainWingFlat_Creation, AVL_Run_Case, DEFAULT_SWEEP_MODE
from simulation_support.alpha_sweep import COMPLETION_FLAG
from simulation_support.polar_creation import AOA_LOG_HEADER
from geometric_support import Airfoil_Generator, Validate_Airfoil, create_modification_combinations
from file_handling_support import compression_Alma, Training_Data_Generator, get_all_files_from
from file_handling_support.dataset_names import dataset_number
from ux_support import recovery_json, recovery_Fabius, run_with_progress

# Platform: binaries carry .exe on Windows
IS_WINDOWS = sys.platform == "win32"
EXE_SUFFIX = ".exe" if IS_WINDOWS else ""

# === Run options === #
# Read at import time so that worker processes started with 'spawn' (Windows) build the same paths.
_cli = argparse.ArgumentParser(description="Builds a Hossain wing dataset.", add_help=__name__ == "__main__")
_cli.add_argument("--smoke", type=int, metavar="N", help="Pipeline test: every stage on N airfoils, as dataset 'smoke'.")
_cli.add_argument("--no-gui", action="store_true", help="Console progress bars instead of progress windows.")
ARGS, _ = _cli.parse_known_args()
SMOKE = ARGS.smoke
USE_GUI = False if ARGS.no_gui else None # None: window when a display is available

# Dataset number: a decimal number is a rebuild of an existing dataset (17.5 rebuilds 17 with the zero_out sweep)
DATASET_NUM = "smoke" if SMOKE else dataset_number("17.5")

# Get the path to the src directory.
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
# Define final directories
dataset_build_data = project_root / "data" / "dataset_build_data"
dataset_build_data.mkdir(exist_ok=True)
airfoils_dir = project_root / "data" / "dataset_build_data" / "airfoils"
force_files_dir = project_root / "data" / "dataset_build_data" / "force files"
polars_dir = project_root / "data" / "dataset_build_data" / "polars"
geo_files_dir = project_root / "data" / "dataset_build_data" / "geometry files"
support_files_dir = project_root / "data" / "support"
naca_kp_dir = project_root / "data" / "support" / "naca_kp"
logs_dir = project_root / "data" / "logs"
datasets_dir = project_root / "data" / "datasets"
binaries_dir = project_root / "bin"
# Logs of this run: stage summaries, failed airfoils, failed XFOIL angles, failed AVL runs, physical check
run_logs_dir = logs_dir / f"dataset_{DATASET_NUM}"
run_logs_dir.mkdir(parents=True, exist_ok=True)
# Define temp directories
# Temp directories with a short path are needed for XFOIL and AVL
# The mending temp root is kept independent of training_temp_root (not nested under it) -- XFOIL/AVL
# don't tolerate the resulting path length well.
temp_suffix = "_smoke" if SMOKE else ""
if IS_WINDOWS:
    training_temp_root = Path(rf'C:\training_temp{temp_suffix}') # Drive root keeps paths short, any drive can be used
    correction_temp_root = Path(rf'C:\correction_temp{temp_suffix}')
else:
    training_temp_root = Path.home() / f'.training_temp{temp_suffix}' # The dot prefix naturally hides the folder in Linux; user-writable, avoids needing root on '/'
    correction_temp_root = Path.home() / f'.correction_temp{temp_suffix}'
os.makedirs(training_temp_root, exist_ok=True)
if IS_WINDOWS:
    subprocess.run(['attrib', '+h', str(training_temp_root)], check=True, creationflags=subprocess.CREATE_NO_WINDOW)
airfoils_temp_dir = training_temp_root / "airfoils"
force_files_temp_dir = training_temp_root / "force files"
polars_temp_dir = training_temp_root / "polars"
support_files_temp_dir = training_temp_root / "support_files"
geo_files_temp_dir = training_temp_root / "geometry files"
training_files_temp_dir = training_temp_root / "training files"
# Define .rar paths and names for recovery logic
dataset_dir = datasets_dir / f"dataset_{DATASET_NUM}"
airfoil_dataset_name = f"airfoils_dataset_{DATASET_NUM}"
airfoil_rar_path = airfoils_dir / f"{airfoil_dataset_name}.rar"
polar_dataset_name = f"polars_dataset_{DATASET_NUM}"
polar_rar_path = polars_dir / f"{polar_dataset_name}.rar"
geo_dataset_name = f"geo_files_dataset_{DATASET_NUM}"
geo_rar_path = geo_files_dir / f"{geo_dataset_name}.rar"
force_dataset_name = f"force_files_dataset_{DATASET_NUM}"
force_rar_path = force_files_dir / f"{force_dataset_name}.rar"
training_files_dataset_name = f"training_files_dataset_{DATASET_NUM}"
training_files_rar_path = dataset_dir / f"{training_files_dataset_name}.rar"
mended_files_dataset_name = f"final_training_files_dataset_{DATASET_NUM}"
mended_files_rar_path = dataset_dir / f"{mended_files_dataset_name}.rar"

# Persistent logs written while stages run (they survive the temp folder cleanup)
invalid_airfoils_log = airfoils_temp_dir / "Invalid_airfoils_log.txt" # read back by Training_Data_Generator
aoa_log_path = run_logs_dir / "xfoil_failed_angles.csv"
polar_errors_path = run_logs_dir / "polar_errors.csv"
key_point_errors_path = run_logs_dir / "key_point_errors.csv"
below_limit_path = run_logs_dir / "polars_below_laminar_limit.json"
geo_errors_path = run_logs_dir / "geometry_errors.csv"
avl_failures_path = run_logs_dir / "avl_failures.csv"
failed_airfoils_path = run_logs_dir / "failed_airfoils.csv"
key_points_file = geo_files_temp_dir / "key_points.json" # key points of every polar, archived with the geometry files
key_points_partial = training_temp_root / "key_points_partial.json" # checkpoint while key points are extracted

# === Stages === #
@dataclass(frozen=True)
class Stage:
    """One pipeline stage: its name (console, recovery, summary file), position and output archive."""
    number: int
    name: str
    archive: Path

    @property
    def slug(self) -> str:
        return self.name.lower().replace(" ", "_")

    @property
    def status_key(self) -> str:
        return f"{self.slug}_complete"

    @property
    def summary_path(self) -> Path:
        return run_logs_dir / f"{self.number}_{self.slug}_summary.txt"

AIRFOIL_GENESIS = Stage(1, "Airfoil Genesis", airfoil_rar_path)
POLAR_FORMULATION = Stage(2, "Polar Formulation", polar_rar_path)
AVL_GEOMETRIC_FILE_FORMATION = Stage(3, "AVL Geometric File Formation", geo_rar_path)
FORCE_COMPUTATION = Stage(4, "Force Computation", force_rar_path)
TRAINING_FILES_INTEGRATION = Stage(5, "Training Files Integration", training_files_rar_path)
DATABASE_MENDING = Stage(6, "Database Mending", mended_files_rar_path)
STAGES = [AIRFOIL_GENESIS, POLAR_FORMULATION, AVL_GEOMETRIC_FILE_FORMATION, FORCE_COMPUTATION, TRAINING_FILES_INTEGRATION, DATABASE_MENDING]

# === Recovery Handling === #
rec_json = recovery_json(directory=training_temp_root) if SMOKE else recovery_json()
rec_Fabius = recovery_Fabius()

# === Aux objects === #
rar_Alma = compression_Alma()

# === Console and log helpers === #
def short_path(path) -> str:
    '''Path relative to the project root when possible (readable console output).'''
    try:
        return os.path.relpath(path, project_root)
    except ValueError:
        return str(path)

def stage_summary(stage: Stage, lines, files=()):
    '''Prints the stage summary and saves it to data/logs/dataset_<N>/<n>_<stage>_summary.txt.'''
    bar = "=" * 80
    text = [bar, f"{stage.name} - summary ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})", bar, *lines]
    files = [f for f in files if f and Path(f).exists()]
    if files:
        text += ["", "Files:"] + [f"  {short_path(f)}" for f in files]
    text.append(bar)
    print("\n" + "\n".join(text))
    stage.summary_path.write_text("\n".join(text) + "\n", encoding="utf-8")

def append_csv_line(path: Path, header: str, values):
    '''Appends one CSV line (one write call, so parallel workers don't interleave); writes the header first if new.'''
    line = ",".join(str(v).replace(",", ";").replace("\n", " ") for v in values) + "\n"
    try:
        with open(path, "a", encoding="utf-8") as f:
            if f.tell() == 0:
                f.write(header)
            f.write(line)
    except OSError:
        pass

def read_csv_rows(path: Path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def pct(part, whole) -> str:
    return f"{100 * part / whole:.1f}%" if whole else "-"

def top_counts(counter: Counter, n=5, indent="    "):
    return [f"{indent}{count:>8,}  {reason}" for reason, count in counter.most_common(n)]

# === Database Logic === #
def _airfoil_number(name: str) -> int:
    '''Test_123 -> 123 (-1 if the name has no number).'''
    try:
        return int(name.split('_')[-1])
    except (ValueError, IndexError):
        return -1

def _invalid_reason(log_line: str) -> str:
    '''Reason text of an Invalid_airfoils_log line (the part after "Reason:").'''
    reason = log_line.split("Reason:", 1)[-1].strip()
    return re.sub(r"\(.*", "", reason).strip() or "unknown"

def Airfoil_Genesis(base_airfoil_path, n_cst, cmb_datalist, generate_invalid=False, **kwargs):
    '''
    Fills a folder with generated airfoils matching the combination list and geometrically filtered for real plausible airfoils

    Parameters
    ----------
    base_airfoil_path : str, rstr
        Takes the absolute path to the base airfoil for modification

    n_cst : int
        Order of CST coefficients

    cmb_datalist : np.ndarray
        A list containing rows of cst coeeffiecients, format in Airfoil_Generator
    '''
    # Unpack filter parameters
    threshold_max_t = kwargs.get('threshold_max_t')
    threshold_min_t = kwargs.get('threshold_min_t')
    threshold_t_x_max = kwargs.get('threshold_t_x_max')
    threshold_t_x_min = kwargs.get('threshold_t_x_min')
    activate_curvature_cont = kwargs.get('activate_curvature_cont')
    threshold_rLE = kwargs.get('threshold_rLE')
    monotonic_activation = kwargs.get('monotonic_activation')
    threshold_max_camber = kwargs.get('threshold_max_camber')
    threshold_min_camber = kwargs.get('threshold_min_camber')

    # Recovery check
    if rec_Fabius.check_stage_status(stage_name=AIRFOIL_GENESIS.name, current_rar=AIRFOIL_GENESIS.archive, required_deps=None):
        return

    os.makedirs(airfoils_temp_dir, exist_ok=True)
    invalid_output_dir = airfoils_temp_dir / "invalid airfoils"

    # --- Recovery Logic --- #
    # Resume after the highest airfoil number found in the valid files or in the invalid log. That last
    # airfoil is generated again (its file or log line may be incomplete), so it is removed first.
    valid_numbers = [_airfoil_number(f.stem) for f in airfoils_temp_dir.glob('*.dat')]
    invalid_lines = invalid_airfoils_log.read_text().splitlines(keepends=True) if invalid_airfoils_log.exists() else []
    invalid_numbers = [_airfoil_number(line.split(".dat", 1)[0]) for line in invalid_lines if ".dat" in line]
    start_index = 0
    last = max(valid_numbers + invalid_numbers, default=0)
    if last > 0:
        start_index = last - 1
        (airfoils_temp_dir / f"Test_{last}.dat").unlink(missing_ok=True)
        kept = [line for line, num in zip([l for l in invalid_lines if ".dat" in l], invalid_numbers) if num < last]
        invalid_airfoils_log.write_text("".join(kept))
        print(f"Resuming {AIRFOIL_GENESIS.name} from airfoil Test_{start_index + 1}")

    # --- Generation Logic --- #
    airfoil_creator = Airfoil_Generator(
            base_airfoil_csv=base_airfoil_path,
            n_cst=n_cst,
            verbose=False
        )
    progress_queue = queue.Queue()

    def generate_all():
        with open(invalid_airfoils_log, "a") as log:
            for i in range(start_index, len(cmb_datalist)):
                cmb = cmb_datalist[i]
                airfoil_name = f"Test_{i+1}"
                x_mod, yu_mod, yl_mod = airfoil_creator.generate(
                    cst_coeffs=cmb,
                    show_plot=False,
                    debug=False
                    )

                # Silent check_validity before creating mod airfoil
                log_reason = None
                validity = check_validity(x_mod, yu_mod, yl_mod, verbose=False)

                if validity == True:
                    valid = 'valid'
                    try: # Wrapped in try block in case FoilGeoFeatures raises an unexpected error
                        va = Validate_Airfoil(
                            x=x_mod,
                            yu=yu_mod,
                            yl=yl_mod,
                            cst_coeffs=cmb,
                            n_cst=n_cst,
                            name=airfoil_name,
                            verbose=False
                            )

                        if valid == 'valid':
                            valid, log_reason = va.thickness_filters(th_t_min=threshold_min_t, th_t_max=threshold_max_t, th_t_x_max=threshold_t_x_max, th_t_x_min=threshold_t_x_min)
                        if valid == 'valid':
                            valid, log_reason = va.sharpness_filter(curv_cont=activate_curvature_cont, th_rLE=threshold_rLE)
                        if valid == 'valid':
                            if monotonic_activation:
                                valid, log_reason = va.monotonic_thickness_filter()
                        if valid == 'valid':
                            valid, log_reason = va.camber_filters(th_c_max=threshold_max_camber, th_c_min=threshold_min_camber)

                        if valid == 'invalid':
                            validity = False
                    except ValueError as e: # Validate_Airfoil formats its feature-extraction failures as a log line
                        validity = False
                        log_reason = str(e)

                    if validity:
                        airfoil_creator._create_dat_file(
                            output_dir=airfoils_temp_dir,
                            n_cst = n_cst,
                            x=x_mod,
                            yl=yl_mod,
                            yu=yu_mod,
                            file_name=airfoil_name,
                            debug=False
                            )
                else:
                    cmb_l = np.array(cmb[:n_cst])
                    cmb_u = np.array(cmb[n_cst:])
                    log_reason = f"{airfoil_name}.dat: cmb_l = {cmb_l}, cmb_u = {cmb_u} | Reason: Check_validity failed\n"

                if not validity: # If validity == False then we write to log, optionally save airfoil
                    if generate_invalid:
                        airfoil_creator._create_dat_file(
                            output_dir=invalid_output_dir,
                            n_cst = n_cst,
                            x=x_mod,
                            yl=yl_mod,
                            yu=yu_mod,
                            file_name=airfoil_name,
                            debug=False
                        )
                    log.write(f"{log_reason.rstrip()}\n")
                    log.flush()
                progress_queue.put(1)

    remaining = len(cmb_datalist) - start_index
    run_with_progress(generate_all, remaining, progress_queue, title=AIRFOIL_GENESIS.name,
                      header=f"Generating {remaining:,} of {len(cmb_datalist):,} CST combinations", details=False,
                      use_gui=USE_GUI, console_desc="Airfoils", unit="airfoil")

    # Save airfoils to main path in compressed form
    rar_Alma.compress_folder(airfoils_temp_dir, airfoils_dir, airfoil_dataset_name)

    # --- Summary --- #
    valid_count = len(list(airfoils_temp_dir.glob('*.dat')))
    reasons = Counter(_invalid_reason(line) for line in invalid_airfoils_log.read_text().splitlines() if ".dat" in line)
    rejected = sum(reasons.values())
    stage_summary(AIRFOIL_GENESIS, [
        f"CST combinations:   {len(cmb_datalist):>9,}",
        f"Valid airfoils:     {valid_count:>9,}  ({pct(valid_count, len(cmb_datalist))})",
        f"Rejected airfoils:  {rejected:>9,}  ({pct(rejected, len(cmb_datalist))}), by reason:",
        *top_counts(reasons, n=12),
    ], [airfoil_rar_path])

    return

def _resolve_polar_status(polar_abs_path, ASEQ, completion_flag):
    '''
    Determines an existing polar file's status against ASEQ, and repairs the
    completion flag if the file already has every required angle but the
    flag write itself was interrupted (e.g. killed between the last ALFA and
    QUIT/close). Without this repair, such a file recomputes an empty angle
    list on every future resume attempt and is silently treated as "not
    complete" forever, since it never re-enters run_simulation() to reach
    the completion_flag() call.

    Returns:
        status (str): 'complete', 'partial', or 'corrupt'
        angles_remaining (list[float] or None): angles still needed;
            None for 'corrupt' (caller decides the full-range reprocessing).
    '''
    with open(polar_abs_path, 'r') as f:
        content = f.read()

    if completion_flag in content:
        return 'complete', []

    lines = content.splitlines()

    # XFOIL always emits "alpha CL CD CDp CM Top_Xtr Bot_Xtr" as the column
    # header, regardless of version or flow regime. Data rows only ever
    # appear below it (past its dashed underline), so scanning must be
    # scoped to that region -- otherwise numeric tokens in the metadata
    # above (e.g. "1 1 Reynolds number fixed ... Mach number fixed") can be
    # misread as a valid alpha/CL data row.
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().split()[:2] == ['alpha', 'CL']:
            header_idx = i
            break

    if header_idx is None:
        return 'corrupt', None  # Column header missing -- malformed file

    last_alfa = -999
    for line in reversed(lines[header_idx + 2:]):
        parts = line.strip().split()
        if len(parts) >= 4:
            try:
                last_alfa = float(parts[0])
                float(parts[1])
                break
            except (ValueError, IndexError):
                continue

    if last_alfa <= -999:
        return 'corrupt', None  # Header present but no data rows beneath it

    start_angle = last_alfa + ASEQ[2]
    num_steps = int((ASEQ[1] - start_angle) / ASEQ[2]) + 1
    angles_remaining = [start_angle + j * ASEQ[2] for j in range(num_steps)]

    if not angles_remaining:
        # Every required angle is already in the file -- repair the flag.
        try:
            with open(polar_abs_path, "a") as f:
                f.write(f"\n{completion_flag}")
        except IOError as e:
            print(f"[ERROR] Could not repair completion flag on {polar_abs_path}: {e}")
        return 'complete', []

    return 'partial', angles_remaining

def polar_worker(args):
    '''
    Worker function to run XFOIL simulation for a single airfoil.
    '''
    airfoil_abs_path, polar_abs_path, ASEQ, Re, Mach, sweep_mode, binaries_dir, training_temp_root, airfoils_temp_dir, polars_temp_dir, progress_queue, index, worker_status_list, verbose, corrupted_files_list, aoa_log, polar_errors = args

    # Obtain airfoil name for GUI
    airfoil_name = os.path.basename(airfoil_abs_path)
    worker_status_list[index] = airfoil_name

    # --- Recovery Logic for each polar file --- #
    if os.path.exists(polar_abs_path):
        status, angles_remaining = _resolve_polar_status(polar_abs_path, ASEQ, COMPLETION_FLAG)

        if status == 'complete':
            if progress_queue:
                progress_queue.put(1)
            return  # Skip if already completed (or just repaired)

        if status == 'partial' and sweep_mode == "sequential":
            ANGLES = angles_remaining
        else: # 'corrupt', or 'partial' under zero_out: resuming mid-sequence would rebuild the warm-start path zero_out avoids
            if status == 'corrupt':
                if verbose:
                    print(f"[INFO] Corrupted file detected and deleted: {polar_abs_path}. Reverting to fresh file.")
                corrupted_files_list.append(polar_abs_path)
            try:
                os.remove(polar_abs_path)
            except OSError:
                pass
            num_steps = int((ASEQ[1] - ASEQ[0]) / ASEQ[2]) + 1
            ANGLES = [ASEQ[0] + j * ASEQ[2] for j in range(num_steps)]

    else: # File doesn't exist yet
         num_steps = int((ASEQ[1] - ASEQ[0]) / ASEQ[2]) + 1
         ANGLES = [ASEQ[0] + j * ASEQ[2] for j in range(num_steps)]

    if not ANGLES: # Nothing to compute
        if progress_queue:
            progress_queue.put(1)
            worker_status_list[index] = "Idle"
        return

    # --- Calculate Polar --- #
    try:
        xfoil_exe = os.path.join(binaries_dir, f'xfoil{EXE_SUFFIX}')
        # Create an instance of the runner
        xfoil_case = XPR(
            xfoil_exe_path=xfoil_exe,
            main_dir=training_temp_root,
            airfoil_dir=airfoils_temp_dir,
            polar_dir=polars_temp_dir,
            verbose=verbose,
            aoa_log_path=aoa_log
            )

        # Run the simulation
        xfoil_case.run_simulation(
            airfoil_file_path=airfoil_abs_path,
            polar_file_path=polar_abs_path,
            alfa_sequence=ANGLES,
            visc = Re,
            mach=Mach,
            sweep_mode=sweep_mode
        )

    except Exception as e:
        if verbose:
            print(f"\n[ERROR] A critical error occurred while processing {airfoil_name}: {e}")
        append_csv_line(Path(polar_errors), "airfoil,error\n", [airfoil_name, f"{type(e).__name__}: {e}"])
    finally:
        if progress_queue:
            progress_queue.put(1)

def Polar_Formulation(ASEQ, Re, Mach, sweep_mode=DEFAULT_SWEEP_MODE, verbose=False):
    '''
    Runs XFOIL to create drag polars for each airfoil using multiprocessing and a GUI.

    Parameters:
    ==============
    ASEQ (list): List containing:
                    ASEQ[0]: Smallest angle of attack to test.
                    ASEQ[1]: Largest angle of attack to test.
                    ASEQ[2]: Resolution for intermediate angles of attack.
    sweep_mode (str): Alpha sweep strategy, see simulation_support.alpha_sweep ('sequential' is legacy only).
    verbose (bool): Toggles prints, including XFOIL's per-angle retry warnings (always written to the angle log).
    '''
    # Recovery check
    rec_dependencies = [(airfoil_rar_path, airfoils_temp_dir)]
    if rec_Fabius.check_stage_status(stage_name=POLAR_FORMULATION.name, current_rar=POLAR_FORMULATION.archive, required_deps=rec_dependencies):
        return

    airfoil_abs_paths = get_all_files_from(airfoils_temp_dir, extension='dat')
    if not airfoil_abs_paths:
        print("No airfoils found to process for polar generation.")
        return

    # --- Pre-allocation recovery filtering --- #
    # polar_worker() also checks this flag, but only after a task has already
    # been dispatched to a pool worker. On a resumed run that check has to be
    # re-run for every already-completed airfoil (potentially tens of
    # thousands), which is what was driving the rapid XFOIL spawn/kill cycling.
    # Filtering here removes completed airfoils from the queue before the Pool
    # is ever created, matching the intended recovery design.
    complete_paths = []
    partial_paths = []   # polar file exists but no flag yet -> polar_worker resumes mid-sequence (sequential) or restarts (zero_out)
    fresh_paths = []     # no polar file yet -> starts from ASEQ[0]

    corrupted_count = 0
    for file in airfoil_abs_paths:
        airfoil_name = os.path.splitext(os.path.basename(file))[0]
        polar_abs_path = os.path.join(polars_temp_dir, f"polar_{airfoil_name}.txt")
        if os.path.exists(polar_abs_path):
            status, _ = _resolve_polar_status(polar_abs_path, ASEQ, COMPLETION_FLAG)
            if status == 'complete':  # includes flag repaired just now
                complete_paths.append(file)
            elif status == 'partial':
                partial_paths.append(file)
            else: # 'corrupt'
                if verbose:
                    print(f"[INFO] Corrupted file detected and deleted: {polar_abs_path}. Reverting to fresh start.")
                corrupted_count += 1
                try:
                    os.remove(polar_abs_path)
                except OSError:
                    pass
                fresh_paths.append(file)
        else:
            fresh_paths.append(file)

    if corrupted_count and not verbose:
        print(f"[INFO] {corrupted_count} corrupted polar file(s) detected and deleted. Reverting to fresh start.")

    os.makedirs(polars_temp_dir, exist_ok=True)  # first-ever run: XPR only creates this once a task spawns
    polar_files_on_disk = get_all_files_from(polars_temp_dir, extension='txt')
    matched_polar_count = len(complete_paths) + len(partial_paths)
    print(
        f"[INFO] Recovery scan: {len(airfoil_abs_paths)} airfoil(s) found | "
        f"{len(polar_files_on_disk)} polar file(s) on disk | "
        f"{len(complete_paths)} confirmed complete (flagged) | "
        f"{len(partial_paths)} partial (will {'resume' if sweep_mode == 'sequential' else 'restart'}) | "
        f"{len(fresh_paths)} not started."
    )
    if len(polar_files_on_disk) != matched_polar_count:
        print(
            f"[WARNING] {len(polar_files_on_disk)} polar file(s) on disk vs "
            f"{matched_polar_count} matched to a current airfoil by name "
            f"({len(polar_files_on_disk) - matched_polar_count} unmatched) -- check for a naming mismatch."
        )

    pending_airfoil_paths = partial_paths + fresh_paths
    corrupted_total = corrupted_count

    if pending_airfoil_paths:
        num_workers = max(1, min(cpu_count() - 1, len(pending_airfoil_paths)))
        if not aoa_log_path.exists():
            aoa_log_path.write_text(AOA_LOG_HEADER)

        with Manager() as manager:
            progress_queue = manager.Queue()
            worker_status_list = manager.list(["Initializing..."] * num_workers)
            corrupted_files_list = manager.list()

            tasks = []
            for i, file in enumerate(pending_airfoil_paths):
                airfoil_name = os.path.splitext(os.path.basename(file))[0]
                polar_abs_path = os.path.join(polars_temp_dir, f"polar_{airfoil_name}.txt")
                tasks.append((file, polar_abs_path, ASEQ, Re, Mach, sweep_mode, binaries_dir, training_temp_root, airfoils_temp_dir, polars_temp_dir, progress_queue, i % num_workers, worker_status_list, verbose, corrupted_files_list, str(aoa_log_path), str(polar_errors_path)))

            # Define the processing job to run in a separate thread
            def processing_job():
                with Pool(processes=num_workers, maxtasksperchild=100) as pool:
                    pool.map(polar_worker, tasks)

            header = f"Creating polars from {ASEQ[0]}° to {ASEQ[1]}° (Resolution: {ASEQ[2]}°, sweep: {sweep_mode})"
            run_with_progress(processing_job, len(tasks), progress_queue, title="Polar Calculation Progress", header=header,
                              status_list=worker_status_list, n_workers=num_workers, use_gui=USE_GUI, console_desc="Polars", unit="airfoil")
            corrupted_in_run = len(corrupted_files_list)
            corrupted_total += corrupted_in_run

        if corrupted_in_run and not verbose:
            print(f"[INFO] {corrupted_in_run} corrupted polar file(s) detected and deleted. Reverting to fresh file.")
    else:
        print("[INFO] All airfoils already have completed polars. Nothing to dispatch.")

    rar_Alma.compress_folder(polars_temp_dir, polars_dir, polar_dataset_name)

    # --- Summary --- #
    complete = sum(1 for p in polars_temp_dir.glob('polar_*.txt') if COMPLETION_FLAG in p.read_text(errors="ignore"))
    angle_rows = read_csv_rows(aoa_log_path)
    retries = sum(1 for r in angle_rows if r.get("event") == "retry")
    skipped = [r for r in angle_rows if r.get("event") == "skipped"]
    init_failed = sum(1 for r in angle_rows if r.get("event") == "init_failed")
    errors = read_csv_rows(polar_errors_path)
    stage_summary(POLAR_FORMULATION, [
        f"Airfoils:                   {len(airfoil_abs_paths):>9,}",
        f"Complete polars:            {complete:>9,}  ({pct(complete, len(airfoil_abs_paths))})",
        f"Missing or incomplete:      {len(airfoil_abs_paths) - complete:>9,}  (listed in failed_airfoils.csv at the end)",
        f"XFOIL angle retries:        {retries:>9,}",
        f"Angles skipped after retry: {len(skipped):>9,}  in {len({r['airfoil'] for r in skipped}):,} airfoil(s)",
        f"XFOIL restarts that failed: {init_failed:>9,}  (the pass stopped at that angle)",
        f"XFOIL runner errors:        {len(errors):>9,}",
        f"Corrupt polar files reset:  {corrupted_total:>9,}",
        f"Sweep: {sweep_mode} | ASEQ: {ASEQ} | Re = {Re:g} | Mach = {Mach:g}",
    ], [polar_rar_path, aoa_log_path, polar_errors_path])

    return

# --- Key point extraction (process pool) --- #
def _key_point_worker_init():
    '''One BLAS thread per worker: the pool already uses every core, extra threads would only compete.'''
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
    except ImportError:
        pass

def key_point_worker(args):
    '''Key points and lowest-drag point of one polar file (JSON-ready dict).'''
    polar_path, kp_seed, cd_limit, progress_queue, index, worker_status_list = args
    name = os.path.basename(polar_path)
    worker_status_list[index] = name
    record = {"file_path": name, "complete": False, "result": None, "error": None, "cd_min": None}
    try:
        record["complete"] = COMPLETION_FLAG in Path(polar_path).read_text(errors="ignore")
        extractor = Key_Point_Extraction(polar_file_paths=[str(polar_path)], random_state=kp_seed)
        record["result"], error = extractor.process_file(str(polar_path))
        record["error"] = error["error"] if error else None
        lowest = extractor.polar_cd_min(str(polar_path))
        if lowest is not None:
            record["cd_min"] = {"CD": lowest[0], "alpha": lowest[1], "CL": lowest[2]}
    except Exception as e:
        record["error"] = f"{type(e).__name__}: {e}"
    finally:
        progress_queue.put(1)
    return record

def _load_key_point_records(path: Path):
    if path.exists():
        try:
            with open(path) as f:
                return {r["file_path"]: r for r in json.load(f)["records"]}
        except (OSError, ValueError, KeyError):
            pass
    return {}

def _save_key_point_records(path: Path, records: dict, kp_seed):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump({"kp_seed": kp_seed, "records": list(records.values())}, f)
    os.replace(tmp, path)

def extract_key_points(kp_seed, cd_limit):
    '''
    Key points of every polar in a process pool, with a checkpoint (resumes where it stopped). Returns the
    records: one per polar file with the key points (or the error), whether the polar was complete and its
    lowest-drag point. Writes key_points.json (archived with the geometry files).
    '''
    polar_paths = sorted(get_all_files_from(polars_temp_dir, extension='txt'))
    records = _load_key_point_records(key_points_file) or _load_key_point_records(key_points_partial)
    pending = [p for p in polar_paths if p.name not in records]
    if records and pending:
        print(f"[INFO] Key points: {len(records):,} polar(s) already done, {len(pending):,} to go.")
    if pending:
        num_workers = max(1, min(cpu_count() - 1, len(pending)))
        with Manager() as manager:
            progress_queue = manager.Queue()
            worker_status_list = manager.list(["Initializing..."] * num_workers)
            tasks = [(str(p), kp_seed, cd_limit, progress_queue, i % num_workers, worker_status_list) for i, p in enumerate(pending)]

            def processing_job():
                last_save = time.time()
                with Pool(processes=num_workers, initializer=_key_point_worker_init, maxtasksperchild=500) as pool:
                    for record in pool.imap_unordered(key_point_worker, tasks, chunksize=4):
                        records[record["file_path"]] = record
                        if time.time() - last_save > 120:
                            _save_key_point_records(key_points_partial, records, kp_seed)
                            last_save = time.time()

            run_with_progress(processing_job, len(tasks), progress_queue, title="Key Point Extraction",
                              header=f"Fitting key points of {len(tasks):,} polars (GPR, seed {kp_seed})",
                              status_list=worker_status_list, n_workers=num_workers, use_gui=USE_GUI, console_desc="Key points", unit="polar")
    os.makedirs(geo_files_temp_dir, exist_ok=True)
    _save_key_point_records(key_points_file, records, kp_seed)
    key_points_partial.unlink(missing_ok=True)
    return [records[p.name] for p in polar_paths if p.name in records]

def AVL_Geometric_File_Formation(configs, kp_seed=None, reynolds=2.5e5):
    '''
    Generates AVL's .geo files that build the wing geometry, this script focuses on replicating Hossain's experiment so there are 3 wing configurations.
    kp_seed fixes the random_state of the key-point GPR fit (None keeps the legacy unseeded fit).
    Also checks every polar's lowest drag against the laminar flat-plate limit (see laminar_cd_limit).
    '''
    from file_handling_support.database_correction import laminar_cd_limit

    # Recovery check
    rec_dependencies = [
        (airfoil_rar_path, airfoils_temp_dir),
        (polar_rar_path, polars_temp_dir)
    ]
    if rec_Fabius.check_stage_status(stage_name=AVL_GEOMETRIC_FILE_FORMATION.name, current_rar=AVL_GEOMETRIC_FILE_FORMATION.archive, required_deps=rec_dependencies):
        return

    # --- Key points --- #
    cd_limit = laminar_cd_limit(reynolds)
    records = extract_key_points(kp_seed, cd_limit)
    key_points_data = [r["result"] for r in records if r["result"] is not None]
    kp_errors = Counter()
    with open(key_point_errors_path, "w", encoding="utf-8") as f:
        f.write("polar,error\n")
        for r in records:
            if r["result"] is None:
                f.write(f"{r['file_path']},{str(r['error']).replace(',', ';')}\n")
                kp_errors[re.sub(r"\d+", "N", str(r["error"]))[:110]] += 1

    # --- Physical check: polars whose lowest CD is below the laminar flat-plate limit --- #
    below = [{"airfoil": r["file_path"].replace("polar_", "").replace(".txt", ""), "polar": r["file_path"], **r["cd_min"]}
             for r in records if r["cd_min"] and r["cd_min"]["CD"] < cd_limit]
    with open(below_limit_path, "w") as f:
        json.dump({"laminar_cd_limit": cd_limit, "reynolds": reynolds,
                   "rule": "polar CDmin < 2 x 1.328 / sqrt(Re) (laminar flat plate, both surfaces)",
                   "polars_checked": sum(1 for r in records if r["cd_min"]), "polars": below}, f, indent=2)

    # --- Geometry files, one progress window per wing configuration --- #
    airfoil_abs_paths = get_all_files_from(airfoils_temp_dir, extension='dat')
    # NACA 0012 CDCL support file (AVL 3.32 requires section-local CDCL)
    naca_0012_kp_path = os.path.join(naca_kp_dir, "naca_0012_M0-9281_RE250k_M0-093_RE250000.txt")
    geo_counts = {}
    with open(geo_errors_path, "w", encoding="utf-8") as geo_log:
        geo_log.write("config,airfoil,error\n")
        for creator_class, identifier in configs:
            geo_creator = creator_class(
                key_points_data=key_points_data,
                airfoil_abs_paths=airfoil_abs_paths,
                naca_key_points_data=naca_0012_kp_path,
                output_dir=str(geo_files_temp_dir),
                verbose=False,
                pb=False
            )
            progress_queue = queue.Queue()
            run_with_progress(lambda: geo_creator.create_geo_files(progress_callback=lambda: progress_queue.put(1)),
                              len(geo_creator.airfoil_path_map), progress_queue, title=f"{identifier} geometry files",
                              header=f"Creating {identifier} AVL geometry files", details=False,
                              use_gui=USE_GUI, console_desc=f"{identifier} geo files", unit="file")
            # Airfoils without key points are already listed in key_point_errors.csv
            other_errors = [e for e in geo_creator.errors if not e['error'].startswith("Key point data not found")]
            for e in other_errors:
                geo_log.write(f"{identifier},{e['airfoil']},{e['error'].replace(',', ';')}\n")
            geo_counts[identifier] = (len(list(geo_files_temp_dir.glob(f"*_{identifier}_geo.avl"))), len(other_errors))

    rar_Alma.compress_folder(geo_files_temp_dir, geo_files_dir, geo_dataset_name)

    # --- Summary --- #
    lines = [
        f"Polar files:            {len(records):>9,}",
        f"Key points extracted:   {len(key_points_data):>9,}  ({pct(len(key_points_data), len(records))})",
        f"Key point failures:     {sum(kp_errors.values()):>9,}, most common:",
        *top_counts(kp_errors),
        f"Polars below the laminar CD limit ({cd_limit:.5f} at Re = {reynolds:g}): {len(below):,} (reported, not excluded)",
        "Geometry files:",
        *[f"    {identifier:<6} {count:>9,}  ({errors:,} error(s) besides missing key points)" for identifier, (count, errors) in geo_counts.items()],
    ]
    stage_summary(AVL_GEOMETRIC_FILE_FORMATION, lines, [geo_rar_path, key_point_errors_path, below_limit_path, geo_errors_path])

    return

def _force_file_for(geo_path) -> Path:
    '''Force file AVL_Run_Case writes for a geometry file.'''
    return force_files_temp_dir / f"force_{Path(geo_path).stem.replace('_geo', '')}.txt"

def avl_worker(args):
    '''
    Worker function to run AVL simulation for a single geometry file.
    '''
    geo_path, case_file, avl_exe, output_dir, progress_queue, index, worker_status_list, failures_path = args

    # Obtain geometric file for GUI
    geo_name = os.path.basename(geo_path)
    worker_status_list[index] = geo_name
    force_path = Path(output_dir) / f"force_{Path(geo_path).stem.replace('_geo', '')}.txt"

    try:
        avl_case = AVL_Run_Case(
            geo_files=[geo_path],
            case_file=case_file,
            avl_exe_path=avl_exe,
            output_dir=output_dir,
            debug=False
        )
        avl_case.run()
        # AVL_Run_Case.run() handles its own errors, so a missing force file is the sign of a failed run
        if not force_path.exists():
            append_csv_line(Path(failures_path), "geometry_file,reason\n", [geo_name, "no force file after 3 AVL attempts"])
    except Exception as e:
        append_csv_line(Path(failures_path), "geometry_file,reason\n", [geo_name, f"{type(e).__name__}: {e}"])
    finally:
        worker_status_list[index] = "Idle"
        if progress_queue:
            progress_queue.put(1)

def Force_Computation(verbose=False):
    '''
    Uses AVL to calculate the force coefficients from each geometric file using the global case file.
    '''
    # Recovery check
    rec_dependencies = [(geo_rar_path, geo_files_temp_dir)]
    if rec_Fabius.check_stage_status(stage_name=FORCE_COMPUTATION.name, current_rar=FORCE_COMPUTATION.archive, required_deps=rec_dependencies):
        return

    avl_exe = os.path.join(binaries_dir, f'avl{EXE_SUFFIX}')
    case_file = os.path.join(support_files_dir, "Hossain_sqwing_case.run")
    support_files_temp_dir.mkdir(exist_ok=True)
    case_file_temp = os.path.join(support_files_temp_dir, "Hossain_sqwing_case.run")
    shutil.copyfile(case_file, case_file_temp)
    os.makedirs(force_files_temp_dir, exist_ok=True)

    geo_abs_paths = get_all_files_from(geo_files_temp_dir, extension='avl')
    geo_files_to_process = [g for g in geo_abs_paths if not _force_file_for(g).exists()]

    if geo_files_to_process:
        num_workers = max(1, min(cpu_count() - 2, len(geo_files_to_process)))
        # A resumed run retries the geometry files that failed before, so only this run's failures are kept
        avl_failures_path.unlink(missing_ok=True)

        # chunksize=1: AVL_Run_Case's retry loop (up to 3 attempts, ~30s timeout per
        # command) makes per-task duration highly variable. Any chunksize > 1 locks a
        # worker into finishing its whole assigned batch before it can pull more work,
        # so a worker stuck with several slow/retrying tasks stalls while workers with
        # "easy" chunks run out of work and idle. Per-task IPC overhead is negligible
        # next to multi-second AVL subprocess calls, so there's no real cost to this.
        chunksize = 1
        if len(geo_files_to_process) < len(geo_abs_paths):
            print(f"Resuming with {len(geo_files_to_process)} remaining simulations.")
        with Manager() as manager:
            progress_queue = manager.Queue()
            worker_status_list = manager.list(["Initializing..."] * num_workers)

            tasks = [(path, case_file_temp, avl_exe, force_files_temp_dir, progress_queue, i % num_workers, worker_status_list, str(avl_failures_path)) for i, path in enumerate(geo_files_to_process)]

            def processing_job():
                with Pool(processes=num_workers, maxtasksperchild=100) as pool:
                    for _ in pool.imap_unordered(avl_worker, tasks, chunksize=chunksize):
                        pass

            with open(case_file_temp, 'r') as f:
                header = f.readline().strip() # Read first line for header
            run_with_progress(processing_job, len(tasks), progress_queue, title="AVL Coefficient Calculation", header=header,
                              status_list=worker_status_list, n_workers=num_workers, use_gui=USE_GUI, console_desc="AVL", unit="file")
    else:
        print("All force files appear to be generated. Compressing now.")

    rar_Alma.compress_folder(force_files_temp_dir, force_files_dir, force_dataset_name)

    # --- Summary --- #
    failures = read_csv_rows(avl_failures_path)
    by_config = Counter()
    written = Counter()
    for g in geo_abs_paths:
        identifier = Path(g).stem.replace("_geo", "").rsplit("_", 1)[-1]
        by_config[identifier] += 1
        written[identifier] += _force_file_for(g).exists()
    total_written = sum(written.values())
    stage_summary(FORCE_COMPUTATION, [
        f"Geometry files:   {len(geo_abs_paths):>9,}",
        f"Force files:      {total_written:>9,}  ({pct(total_written, len(geo_abs_paths))})",
        f"Failed AVL runs:  {len(geo_abs_paths) - total_written:>9,}",
        *[f"    {identifier:<6} {written[identifier]:>9,} of {count:,}" for identifier, count in by_config.items()],
    ], [force_rar_path, avl_failures_path])

    return

def write_failed_airfoils_log(configs, skipped_force_files):
    '''
    failed_airfoils.csv: every valid airfoil (passed Airfoil Genesis) that is missing from the regressor
    dataset for at least one wing configuration, with the first stage that failed and the reason.
    '''
    identifiers = [identifier for _, identifier in configs]
    records = _load_key_point_records(key_points_file)
    geo_errors = {(r["config"], r["airfoil"]): r["error"] for r in read_csv_rows(geo_errors_path)}
    skipped = {}
    for force_name, reason in skipped_force_files:
        stem = force_name.replace("force_", "", 1).replace(".txt", "")
        name, identifier = stem.rsplit("_", 1)
        skipped[(identifier, name)] = reason

    rows = []
    for dat in sorted(airfoils_temp_dir.glob("*.dat"), key=lambda p: _airfoil_number(p.stem)):
        name = dat.stem
        record = records.get(f"polar_{name}.txt")
        if record is None:
            rows.append((name, "Polar Formulation", "all", "no polar file"))
            continue
        if record["result"] is None:
            stage = "Polar Formulation" if not record["complete"] else "AVL Geometric File Formation"
            reason = ("XFOIL did not finish the polar; key points: " if not record["complete"] else "key points: ") + str(record["error"])
            rows.append((name, stage, "all", reason))
            continue
        for identifier in identifiers:
            geo = geo_files_temp_dir / f"{name}_{identifier}_geo.avl"
            if (identifier, name) in geo_errors or not geo.exists():
                rows.append((name, "AVL Geometric File Formation", identifier, geo_errors.get((identifier, name), "no geometry file")))
            elif not _force_file_for(geo).exists():
                rows.append((name, "Force Computation", identifier, "no force file (AVL failed)"))
            elif (identifier, name) in skipped:
                rows.append((name, "Training Files Integration", identifier, skipped[(identifier, name)]))

    with open(failed_airfoils_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["airfoil", "stage", "config", "reason"])
        writer.writerows(rows)
    return rows

def Training_Files_Integration(dataset_dir, n_cst, configs):
    '''
    Creates the final training files for the classifier and regressor models, and the failed-airfoil log.
    '''
    # Recovery check
    rec_dependencies = [
        (airfoil_rar_path, airfoils_temp_dir),
        (geo_rar_path, geo_files_temp_dir),
        (force_rar_path, force_files_temp_dir)
    ]
    if rec_Fabius.check_stage_status(stage_name=TRAINING_FILES_INTEGRATION.name, current_rar=TRAINING_FILES_INTEGRATION.archive, required_deps=rec_dependencies):
        return

    regression_file = f"regressor_dataset_{DATASET_NUM}.csv"
    classification_file = f"classifier_dataset_{DATASET_NUM}.csv"

    generator_eff = Training_Data_Generator(
        force_files_dir = force_files_temp_dir,
        airfoils_dir = airfoils_temp_dir,
        invalid_airfoils_path = invalid_airfoils_log,
        geo_files_dir = geo_files_temp_dir,
        output_dir = training_files_temp_dir,
        n_cst = n_cst,
        verbose = False,
        output_mode = 'Efficiency',
        append_kp=False
    )
    progress_queue = queue.Queue()
    run_with_progress(lambda: generator_eff.generate_training_files(regression_filename=regression_file,
                                                                    classification_filename=classification_file,
                                                                    progress_callback=lambda: progress_queue.put(1)),
                      generator_eff.count_force_files(), progress_queue, title=TRAINING_FILES_INTEGRATION.name,
                      header="Reading force files into the training CSVs", details=False,
                      use_gui=USE_GUI, console_desc="Force files", unit="file")

    dataset_dir.mkdir(parents=True, exist_ok=True)
    rar_Alma.compress_folder(training_files_temp_dir, dataset_dir, training_files_dataset_name)

    # --- Failed airfoils and summary --- #
    failed = write_failed_airfoils_log(configs, generator_eff.skipped)
    by_stage = Counter(stage for _, stage, _, _ in failed)
    counts = generator_eff.row_counts
    valid_airfoils = len(list(airfoils_temp_dir.glob("*.dat")))
    stage_summary(TRAINING_FILES_INTEGRATION, [
        f"Regressor rows:            {counts.get('regression', 0):>9,}  (airfoil x wing configuration)",
        f"Classifier rows:           {counts.get('classification_valid', 0) + counts.get('classification_invalid', 0):>9,}  "
        f"({counts.get('classification_valid', 0):,} valid, {counts.get('classification_invalid', 0):,} invalid)",
        f"Force files left out:      {len(generator_eff.skipped):>9,}",
        f"Valid airfoils with a failure: {len({name for name, _, _, _ in failed}):,} of {valid_airfoils:,}, by first failed stage:",
        *[f"    {count:>8,}  {stage}" for stage, count in by_stage.most_common()],
    ], [training_files_rar_path, failed_airfoils_path])

    return

def Database_Mending(dataset_dir, params, configs):
    '''
    Validates the regressor dataset with the neighbour-consistency check of Dataset_Rectifier
    (file_handling_support.database_correction), using the same simulation pipeline that built it:
    rows whose E is far from their grid neighbours (robust rule: median absolute deviation) are simulated
    again; a different result (e.g. an XFOIL timeout under full load in the bulk run) corrects the row, the
    same result confirms it, and rows whose new simulation fails are dropped. Writes the final CSVs to the
    dataset folder and the mending reports to dataset_<N>/mending.

    History: the legacy Database_Corrector stage blamed multiprocessing for outliers; the 2026-10-28 study of
    dataset 17 showed the chain is deterministic and the outliers came from the legacy 'sequential' alpha
    sweep (see simulation_support.alpha_sweep). Database_Corrector is no longer used here.
    '''
    from file_handling_support.database_correction import Dataset_Rectifier, Simulation_Settings, grid_levels_from_ranges

    # Recovery check
    rec_dependencies = [(training_files_rar_path, training_files_temp_dir)]
    if rec_Fabius.check_stage_status(stage_name=DATABASE_MENDING.name, current_rar=DATABASE_MENDING.archive, required_deps=rec_dependencies):
        return

    regression_file = f"regressor_dataset_{DATASET_NUM}.csv"
    classification_file = f"classifier_dataset_{DATASET_NUM}.csv"
    csv_path = training_files_temp_dir / regression_file
    if not csv_path.exists():
        raise FileNotFoundError(f"{short_path(csv_path)} not found: {TRAINING_FILES_INTEGRATION.name} did not write the regressor CSV.")

    settings = Simulation_Settings(
        n_cst=params["n_cst"], aseq=list(params["ASEQ"]), reynolds=params["Re"], mach=params["Mach"],
        base_airfoil=os.path.join(support_files_dir, 'naca0012_unit.csv'),
        naca_kp_file=os.path.join(naca_kp_dir, "naca_0012_M0-9281_RE250k_M0-093_RE250000.txt"),
        case_file=os.path.join(support_files_dir, "Hossain_sqwing_case.run"),
        xfoil_exe=os.path.join(binaries_dir, f'xfoil{EXE_SUFFIX}'), avl_exe=os.path.join(binaries_dir, f'avl{EXE_SUFFIX}'),
        sweep_mode=params["sweep_mode"], kp_seed=params["kp_seed"],
    )
    rule = params["mending_rule"]
    rectifier = Dataset_Rectifier(
        csv_path=csv_path, settings=settings, work_dir=dataset_dir / "mending", temp_root=correction_temp_root,
        grid_levels=grid_levels_from_ranges(params["n_cst"], params["ranges_lower"], params["ranges_upper"]),
        n_runs=1, rule=rule["rule"], robust_k=rule["robust_k"], min_deviation=rule["min_deviation"],
        threshold=rule["threshold"], drop_failed=rule["drop_failed"], use_gui=USE_GUI,
    )
    # The mended CSV replaces the integration output in the temp folder; the original stays in the training files archive
    paths = rectifier.rectify(output_csv=csv_path, max_rounds=5)
    with open(paths["json"]) as f:
        result = json.load(f)

    rar_Alma.compress_folder(training_files_temp_dir, dataset_dir, mended_files_dataset_name)
    final_regressor = dataset_dir / regression_file
    final_classifier = dataset_dir / classification_file
    shutil.copyfile(csv_path, final_regressor)
    shutil.copyfile(training_files_temp_dir / classification_file, final_classifier)

    totals = Counter()
    for r in result["rounds"]:
        totals.update({k: r[k] for k in ("corrected", "confirmed", "unresolved", "failed")})
    flags = result["initial_flags"]
    stage_summary(DATABASE_MENDING, [
        f"Flag rule: {result['rule']}",
        f"Rows:                      {flags['rows']:>9,}",
        f"Flagged in the first scan: {flags['rows_flagged']:>9,}  rows in {flags['airfoils_flagged']:,} airfoils",
        f"Re-simulated rows:         {sum(totals.values()):>9,}  (every configuration of the flagged airfoils)",
        f"    corrected   {totals['corrected']:>9,}",
        f"    confirmed   {totals['confirmed']:>9,}",
        f"    failed      {totals['failed']:>9,}  ({'dropped from the CSV' if rule['drop_failed'] else 'kept'})",
        f"Rounds: {len(result['rounds'])} | rows changed: {result['rows_changed']:,} | rows dropped: {result['failed_rows_dropped']:,}",
        f"Dataset best E: {result['dataset_best_before']:.2f} -> {result['dataset_best_after']:.2f}",
        f"Re-simulated airfoils below the laminar CD limit: {result['airfoils_below_laminar_limit']:,}",
    ], [final_regressor, final_classifier, paths["report"], paths.get("below_laminar_limit")])

    return

if __name__ == "__main__":
    # === Timer for Fun  === #
    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    start = datetime.now()
    print(f"Database generation started at: {start_time}.")
    # === Geometric Generation Setup === #
    base_airfoil = os.path.join(support_files_dir,'naca0012_unit.csv')
    n_cst = 5

    # Ranges for values that modify CST Curves (starting_value, end_value, step)
    ranges_upper = {
        'LE': (0, 0.125, 0.0625),       # range for value closest to LE
        'inter': (0, 0.125, 1/24),      # range for intermediate points
        'TE': (0, 0.125, 0.125)         # range for value closest to TE
    }
    ranges_lower = {
        'LE': (-0.125, 0, 0.0625),      # range for value closest to LE
        'inter': (0, 0.125, 1/24),      # range for intermediate points
        'TE': (0, 0.125, 0.125)         # range for value closest to TE
    }
    cmb_datalist = create_modification_combinations(n_cst, ranges_upper, ranges_lower, debug=True)
    if SMOKE:
        # A contiguous block from the middle of the grid, so neighbouring airfoils exist for the mending check
        first = len(cmb_datalist) // 2
        cmb_datalist = cmb_datalist[first:first + SMOKE]
        print(f"[SMOKE] Pipeline test on {len(cmb_datalist)} combinations, dataset '{DATASET_NUM}'.")

    # Wing configurations in case they are used
    configs = [
        (HossainWingRec_Creation, "rec"),
        (HossainWingFlat_Creation, "flat"),
        (HossainWing60DEG_Creation, "60DEG")
    ]
    num_configs = len(configs)

    # Geometric Filters thresholds and activations
    geo_filt_parm = {
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

    threshold_max_t = geo_filt_parm['threshold_max_t']
    threshold_min_t = geo_filt_parm['threshold_min_t']
    threshold_t_x_max = geo_filt_parm['threshold_t_x_max']
    threshold_t_x_min = geo_filt_parm['threshold_t_x_min']
    activate_curvature_cont = geo_filt_parm['activate_curvature_cont']
    threshold_rLE = geo_filt_parm['threshold_rLE']
    monotonic_activation = geo_filt_parm['monotonic_activation']
    threshold_max_camber = geo_filt_parm['threshold_max_camber']
    threshold_min_camber = geo_filt_parm['threshold_min_camber']

    # === Polar Alpha Range and Conditions === #
    ASEQ = [-20, 20, 0.5]
    Re = 2.5e5
    Mach = 0.09281
    SWEEP_MODE = DEFAULT_SWEEP_MODE # 'zero_out'; 'sequential' only reproduces legacy datasets (see simulation_support.alpha_sweep)
    KP_SEED = 42 # random_state of the key-point GPR fit, None reproduces the legacy unseeded fit

    # === Database Mending rule === #
    # robust: a row is flagged when |E - neighbour median| > max(3.5 x 1.4826 x neighbour MAD, 2.0); see Dataset_Rectifier
    MENDING_RULE = {'rule': 'robust', 'robust_k': 3.5, 'min_deviation': 2.0, 'threshold': 15.0, 'drop_failed': True}

    # === Dataset Parameter Logging === #
    # Dataset Number is a Global Variable, select at the top of the file.
    num_combinations = cmb_datalist.shape[0]*num_configs
    log_filename = logs_dir / f"Dataset_{DATASET_NUM}_{num_combinations}_data_points.log"

    if not log_filename.exists():
        with open(log_filename, 'w') as log_file:
            log_file.write(f"Dataset {DATASET_NUM} Generation Parameters\n")
            log_file.write("="*30 + "\n")
            log_file.write(f"CST Order (n_cst): {n_cst}\n")
            log_file.write("\nGeometric Filters:\n")
            log_file.write(f"  thickness_filters: threshold_max_t = {threshold_max_t}, threshold_min_t = {threshold_min_t}, threshold_t_x_max = {threshold_t_x_max}, threshold_t_x_min = {threshold_t_x_min}\n")
            log_file.write(f"  sharpness_filters: activate_curvature_cont = {activate_curvature_cont}, threshold_rLE = {threshold_rLE}\n")
            log_file.write(f"  monotonic_thickness_filter: {monotonic_activation}\n")
            log_file.write(f"  camber_filters: threshold_max_camber = {threshold_max_camber}, threshold_min_camber = {threshold_min_camber}\n")
            log_file.write("\nCST Coefficient Ranges:\n")
            log_file.write("\nLower surface:\n")
            log_file.write(f"  {ranges_lower}\n")
            log_file.write("\nUpper surface:\n")
            log_file.write(f"  {ranges_upper}\n")
            if configs:
                log_file.write("\nUsing the following wing configurations:\n")
                # Loop through the list of tuples
                for config_obj, _ in configs:
                    # Use __name__ to get the string representation of the class
                    log_file.write(f" - {config_obj.__name__}\n")
            # Read back by src/utilities/dataset_rectification.py
            log_file.write("\nFlow regime:\n")
            log_file.write(f"  Re = {Re}\n")
            log_file.write(f"  Mach = {Mach}\n")
            log_file.write(f"  ASEQ = {ASEQ}\n")
            log_file.write(f"  Base airfoil = {os.path.basename(base_airfoil)}\n")
            log_file.write("  NACA key point file = naca_0012_M0-9281_RE250k_M0-093_RE250000.txt\n")
            log_file.write(f"  Sweep mode = {SWEEP_MODE}\n")
            log_file.write(f"  Key point seed = {KP_SEED}\n")
            log_file.write("\nDatabase mending:\n")
            log_file.write(f"  {MENDING_RULE}\n")

    mending_params = {"n_cst": n_cst, "ASEQ": ASEQ, "Re": Re, "Mach": Mach, "sweep_mode": SWEEP_MODE, "kp_seed": KP_SEED,
                      "ranges_lower": ranges_lower, "ranges_upper": ranges_upper, "mending_rule": MENDING_RULE}

    # === Execution with Recovery === #
    pipeline = [
        (AIRFOIL_GENESIS, lambda: Airfoil_Genesis(base_airfoil, n_cst, cmb_datalist, **geo_filt_parm)),
        (POLAR_FORMULATION, lambda: Polar_Formulation(ASEQ, Re, Mach, sweep_mode=SWEEP_MODE)),
        (AVL_GEOMETRIC_FILE_FORMATION, lambda: AVL_Geometric_File_Formation(configs, kp_seed=KP_SEED, reynolds=Re)),
        (FORCE_COMPUTATION, lambda: Force_Computation()),
        (TRAINING_FILES_INTEGRATION, lambda: Training_Files_Integration(dataset_dir, n_cst, configs)),
        (DATABASE_MENDING, lambda: Database_Mending(dataset_dir, mending_params, configs)),
    ]
    status = rec_json.get_recovery_status()
    for stage, run_stage in pipeline:
        if not status.get(stage.status_key):
            print(f"\n=== Stage {stage.number}/{len(pipeline)}: {stage.name} ===")
            run_stage()
            status[stage.status_key] = True
            rec_json.save_recovery_status(status)

    # Execution completed successfully
    rec_json.recover_cleanup()
    shutil.rmtree(training_temp_root, ignore_errors=True)

    end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    end = datetime.now()
    total = end - start
    with open(log_filename, 'a') as log_file:
        log_file.write(f"\nExecution start: {start_time} | Execution end: {end_time}\n")
        log_file.write(f"Total execution time: {total}\n")
    print(f"\nDataset {DATASET_NUM} completed at {end_time} (total time {total}).")
    print(f"Dataset: {short_path(dataset_dir)}")
    print(f"Stage summaries and logs: {short_path(run_logs_dir)}")
