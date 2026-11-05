"""
xfoil_polar_unit_test.py

Standalone validation harness for an xfoil binary/build, independent of the
full production pipeline (Hossain_DB_Gen.py). Reuses the same building
blocks:

  - Airfoil generation + geometric filter chain, mirroring
    Airfoil_Database_Generation() (same GEO_FILT_PARM thresholds, same
    Validate_Airfoil sequence: thickness -> sharpness -> monotonic -> camber).
  - Xfoil_Polar_Creator (XPR), same class polar_worker() uses in production.

XFOIL is run against a short, OS-appropriate scratch path (test_temp_root) --
same reasoning as training_temp_root in the driver script: XFOIL can't handle
long directory names in its LOAD/PACC prompts. Once a run finishes, results
are moved out of that scratch path into a permanent, descriptive location
(test_root) under data/singular_test_data/. Never touches ~/.training_temp,
data/dataset_build_data, or the recovery JSON, so it's safe to run alongside
(or instead of) the real pipeline.

Intended use: point --xfoil-path at a newly built binary and run a batch to
see its crash/convergence rate before trusting it at full scale.

Usage:
    python xfoil_polar_unit_test.py --count 50
    python xfoil_polar_unit_test.py --count 200 --xfoil-path /path/to/other/xfoil
    python xfoil_polar_unit_test.py --count 25 --alpha-start -10 --alpha-end 10 --alpha-step 1 --seed 42
"""
import argparse
import csv
import os
import random
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from cst_modeling.foil import check_validity
from simulation_support import Xfoil_Polar_Creator as XPR
from geometric_support import Airfoil_Generator, Validate_Airfoil, create_modification_combinations
from geometric_support.filter_settings import DATASET_FILTERS, apply_filters

utilities_dir = Path(__file__).resolve().parent
current_dir = utilities_dir.parent      # matches current_dir in Hossain_DB_Gen.py
project_root = current_dir.parent       # matches project_root in Hossain_DB_Gen.py

# --- Short scratch path for XFOIL itself -- long paths break its LOAD/PACC prompts --- #
if sys.platform == "win32":
    test_temp_root = Path(r'C:\polar_unit_test_temp')  # any drive works, adjust as needed
else:
    test_temp_root = Path.home() / '.polar_unit_test_temp'

# --- Permanent, descriptive location -- results are moved here once XFOIL is done with them --- #
test_root = project_root / "data" / "singular_test_data" / "xfoil_bin_unit_test"

# === Geometric filters -- the settings Hossain_DB_Gen.py uses (geometric_support/filter_settings.py) === #
GEO_FILT_PARM = DATASET_FILTERS

# === CST combination ranges -- identical to Hossain_DB_Gen.py __main__ === #
RANGES_UPPER = {
    'LE': (0, 0.125, 0.0625),
    'inter': (0, 0.125, 1 / 24),
    'TE': (0, 0.125, 0.125)
}
RANGES_LOWER = {
    'LE': (-0.125, 0, 0.0625),
    'inter': (0, 0.125, 1 / 24),
    'TE': (0, 0.125, 0.125)
}
N_CST = 5

COMPLETION_FLAG = "XFOIL process completed successfully"


def generate_valid_test_airfoils(count, airfoils_dir, invalid_dir, log_path, seed=None):
    '''
    Generates exactly `count` geometrically-valid airfoils, using the same
    CST generation + Validate_Airfoil filter chain (thickness -> sharpness ->
    monotonic -> camber) as Airfoil_Database_Generation(). Combinations are
    sampled in random order out of the full production combination pool, so
    repeated small runs aren't always the same handful of shapes.

    Returns the list of written .dat paths (may be < count if the
    combination pool runs out or the filters reject most of the sample).
    '''
    os.makedirs(airfoils_dir, exist_ok=True)
    os.makedirs(invalid_dir, exist_ok=True)

    base_airfoil_path = project_root / "data" / "support" / "naca0012_unit.csv"

    cmb_datalist = create_modification_combinations(N_CST, RANGES_UPPER, RANGES_LOWER, debug=False)

    order = list(range(cmb_datalist.shape[0]))
    random.Random(seed).shuffle(order)

    airfoil_creator = Airfoil_Generator(
        base_airfoil_csv=str(base_airfoil_path),
        n_cst=N_CST,
        verbose=False
    )

    valid_paths = []
    tried = 0
    for idx in order:
        if len(valid_paths) >= count:
            break
        tried += 1
        cmb = cmb_datalist[idx]
        airfoil_name = f"UnitTest_{len(valid_paths) + 1}"

        x_mod, yu_mod, yl_mod = airfoil_creator.generate(cst_coeffs=cmb, show_plot=False, debug=False)

        validity = check_validity(x_mod, yu_mod, yl_mod, verbose=False)
        log_reason = None

        if validity:
            try:
                va = Validate_Airfoil(
                    x=x_mod, yu=yu_mod, yl=yl_mod,
                    cst_coeffs=cmb, n_cst=N_CST,
                    name=airfoil_name, verbose=False
                )
                valid, log_reason, _ = apply_filters(va, GEO_FILT_PARM)
                validity = (valid == 'valid')
            except ValueError as e:
                validity = False
                log_reason = f"{airfoil_name}: Validate_Airfoil raised unexpected error: {e}"

        if validity:
            airfoil_creator._create_dat_file(
                output_dir=airfoils_dir, n_cst=N_CST,
                x=x_mod, yl=yl_mod, yu=yu_mod,
                file_name=airfoil_name, debug=False
            )
            valid_paths.append(airfoils_dir / f"{airfoil_name}.dat")
        else:
            with open(log_path, "a") as log:
                log.write(f"{log_reason}\n")

    if len(valid_paths) < count:
        print(f"[WARNING] Only found {len(valid_paths)}/{count} valid airfoils after trying "
              f"{tried} combinations (pool exhausted or filters rejected most of the sample).")

    return valid_paths


def run_polar_tests(airfoil_paths, alpha_start, alpha_end, alpha_step, re, mach, xfoil_path,
                     polars_dir, main_dir, airfoils_dir, iter_count=200):
    '''
    Runs XPR.run_simulation() for each airfoil over the given alpha range,
    exactly as polar_worker() does in production (fresh XPR instance per
    airfoil), into an isolated polars directory. Returns a list of per-
    airfoil result dicts.
    '''
    os.makedirs(polars_dir, exist_ok=True)
    alfa_sequence = list(np.arange(alpha_start, alpha_end + alpha_step / 2, alpha_step))
    n_expected = len(alfa_sequence)

    results = []
    for i, airfoil_path in enumerate(airfoil_paths, start=1):
        airfoil_name = airfoil_path.stem
        polar_path = polars_dir / f"polar_{airfoil_name}.txt"

        t0 = time.time()
        unhandled_exception = False
        try:
            xfoil_case = XPR(
                xfoil_exe_path=str(xfoil_path),
                main_dir=str(main_dir),
                airfoil_dir=str(airfoils_dir),
                polar_dir=str(polars_dir),
                verbose=False
            )
            # This script was intended to use with the original sequential alpha sweep, it has been updated to maintain intended behavior. To see how the type of sweep affect simulation refer to alpha_sweep.py
            xfoil_case.run_simulation(
                airfoil_file_path=airfoil_path,
                polar_file_path=polar_path,
                alfa_sequence=alfa_sequence,
                visc=re,
                mach=mach,
                iter=iter_count,
                sweep_mode="sequential"
            )
        except Exception as e:
            unhandled_exception = True
            print(f"[ERROR] {airfoil_name}: unhandled exception -- {e}")
            traceback.print_exc()

        elapsed = time.time() - t0

        # --- Assess outcome directly from the polar file, same signal production's recovery scan uses --- #
        n_points = 0
        flagged_complete = False
        if polar_path.exists():
            content = polar_path.read_text()
            flagged_complete = COMPLETION_FLAG in content
            for line in content.splitlines():
                try:
                    float(line.strip().split()[0])
                    n_points += 1
                except (ValueError, IndexError):
                    continue

        fully_converged = flagged_complete and n_points >= n_expected
        # Flagged complete but short of the full sweep -> the exhausted-attempts
        # path wrote the flag without ever finishing the requested angles.
        exhausted_or_partial_but_flagged = flagged_complete and n_points < n_expected
        crashed_unflagged = not flagged_complete

        result = {
            'airfoil': airfoil_name,
            'elapsed_sec': round(elapsed, 1),
            'points_written': n_points,
            'points_requested': n_expected,
            'fully_converged': fully_converged,
            'exhausted_or_partial_but_flagged': exhausted_or_partial_but_flagged,
            'crashed_unflagged': crashed_unflagged,
            'unhandled_exception': unhandled_exception,
        }
        results.append(result)

        status = "OK" if fully_converged else ("FLAGGED/PARTIAL" if exhausted_or_partial_but_flagged else "UNRESOLVED")
        print(f"[{i}/{len(airfoil_paths)}] {airfoil_name}: {status} "
              f"({n_points}/{n_expected} points, {elapsed:.1f}s)")

    return results


def print_summary(results):
    total = len(results)
    if total == 0:
        print("No results to summarize.")
        return

    fully_converged = sum(1 for r in results if r['fully_converged'])
    exhausted_or_partial = sum(1 for r in results if r['exhausted_or_partial_but_flagged'])
    crashed_unflagged = sum(1 for r in results if r['crashed_unflagged'])
    unhandled = sum(1 for r in results if r['unhandled_exception'])
    avg_points = sum(r['points_written'] for r in results) / total
    n_expected = results[0]['points_requested']

    print("\n" + "=" * 60)
    print(f"XFOIL unit test summary -- {total} airfoil(s)")
    print("=" * 60)
    print(f"  Fully converged (all points, flagged complete):     {fully_converged}")
    print(f"  Flagged complete but exhausted attempts / partial:  {exhausted_or_partial}")
    print(f"  Crashed / never flagged (unresolved):               {crashed_unflagged}")
    print(f"  Unhandled exceptions in the wrapper itself:         {unhandled}")
    print(f"  Average data points written: {avg_points:.1f} / {n_expected}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Generate N geometrically-valid test airfoils and run them through XFOIL "
                    "to validate a binary/build before trusting it at full scale."
    )
    parser.add_argument('--count', type=int, default=25,
                        help="Number of valid airfoils to generate and test (default: 25)")
    parser.add_argument('--alpha-start', type=float, default=-20,
                        help="Alpha sweep start, degrees (default: -20, matches production ASEQ)")
    parser.add_argument('--alpha-end', type=float, default=20,
                        help="Alpha sweep end, degrees (default: 20)")
    parser.add_argument('--alpha-step', type=float, default=0.5,
                        help="Alpha sweep step, degrees (default: 0.5)")
    parser.add_argument('--re', type=float, default=2.5e5,
                        help="Reynolds number (default: matches production, 2.5e5)")
    parser.add_argument('--mach', type=float, default=0.09281,
                        help="Mach number (default: matches production)")
    parser.add_argument('--iter', type=int, default=200,
                        help="XFOIL viscous iteration count per point (default: 200)")
    parser.add_argument('--xfoil-path', type=str, default=str(project_root / "bin" / "xfoil"),
                        help="Path to the xfoil executable to test (default: <project_root>/bin/xfoil)")
    parser.add_argument('--seed', type=int, default=None,
                        help="Random seed for airfoil sampling, for a reproducible test batch")
    args = parser.parse_args()

    if args.count <= 0:
        print("[FATAL] --count must be positive.")
        sys.exit(1)

    if not os.path.exists(args.xfoil_path):
        print(f"[FATAL] xfoil executable not found at: {args.xfoil_path}")
        sys.exit(1)

    # --- Fresh short-path scratch space for this run -- wipe any leftovers first --- #
    if test_temp_root.exists():
        shutil.rmtree(test_temp_root)
    airfoils_dir = test_temp_root / "airfoils"
    invalid_dir = airfoils_dir / "invalid airfoils"
    polars_dir = test_temp_root / "polars"
    log_path = test_temp_root / "invalid_airfoils_log.txt"
    os.makedirs(test_temp_root, exist_ok=True)

    print(f"Testing xfoil binary: {args.xfoil_path}")
    print(f"XFOIL scratch path (short, OS-appropriate): {test_temp_root}")

    print(f"\nGenerating {args.count} geometrically-valid test airfoil(s)...")
    airfoil_paths = generate_valid_test_airfoils(args.count, airfoils_dir, invalid_dir, log_path, seed=args.seed)
    print(f"  -> {len(airfoil_paths)} valid airfoil(s) written to {airfoils_dir}")

    if not airfoil_paths:
        print("[FATAL] No valid airfoils generated -- nothing to test.")
        shutil.rmtree(test_temp_root, ignore_errors=True)
        sys.exit(1)

    print(f"\nRunning XFOIL on {len(airfoil_paths)} airfoil(s), "
          f"alpha {args.alpha_start} to {args.alpha_end} step {args.alpha_step}, "
          f"Re={args.re}, Mach={args.mach}...\n")

    results = run_polar_tests(
        airfoil_paths, args.alpha_start, args.alpha_end, args.alpha_step,
        args.re, args.mach, args.xfoil_path,
        polars_dir=polars_dir, main_dir=test_temp_root, airfoils_dir=airfoils_dir,
        iter_count=args.iter
    )

    print_summary(results)

    # --- Move results out of the short-path scratch space into the permanent, descriptive location --- #
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = test_root / f"run_{run_id}"
    os.makedirs(run_dir, exist_ok=True)

    shutil.move(str(airfoils_dir), str(run_dir / "airfoils"))
    shutil.move(str(polars_dir), str(run_dir / "polars"))
    if log_path.exists():
        shutil.move(str(log_path), str(run_dir / "invalid_airfoils_log.txt"))

    results_path = run_dir / "results.csv"
    with open(results_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    shutil.rmtree(test_temp_root, ignore_errors=True)

    print(f"\nResults moved to {run_dir}")
    print(f"Per-airfoil results written to {results_path}")


if __name__ == "__main__":
    main()