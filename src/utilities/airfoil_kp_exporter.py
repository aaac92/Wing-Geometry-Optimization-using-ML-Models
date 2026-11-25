"""
Simulates one airfoil with a chosen alpha-sweep protocol, extracts its six key points with a chosen GPR seed and
writes the key-point file straight into a support folder. Nothing else: no AVL runs, no comparisons.

Purpose
-------
The winglet sections of the flat and 60DEG AVL geometries use a base airfoil's key points as their CDCL drag polar
(AVL 3.32 reads CDCL per SECTION; simulation_support.geo_file_creation pastes the third line of the file after CDCL).
These files must come from the same simulation chain as the dataset.

The NACA 0012 file of datasets 17 and 17.5, naca_0012_M0-9281_RE250k_M0-093_RE250000.txt, was made from an old
XFOIL 6.99 polar (data/support/naca_polars/polar_naca_0012_M0-9281_RE250k.txt). That polar was a sequential sweep
from -20 deg every 0.2 deg. Its CDmin is 0.00128, below the laminar limit (laminar_cd_limit(2.5e5) = 0.00531).

This script makes such files with any protocol and seed, for NACA 0012 or for any other base airfoil a future dataset
may use. The file in use is named in simulation_support.support_files (NACA_KP_FILE).

Chain (the order of simulation_support.airfoil_simulation.simulate_airfoil_once)
-----
    Airfoil_Generator, all CST increments 0 (the airfoil the pipeline sees) -> check_validity -> .dat in a short
    temp folder (XFOIL reads at most 64 characters of a file name) -> Xfoil_Polar_Creator.run_simulation with the
    protocol -> read_polar_arrays -> polar_key_points (Key_Point_Extraction with random_state = --kp-seed)
Flow settings default to protocol_settings() (dataset 17 / 17.5): Re 2.5e5, Mach 0.09281, alpha -20 to 20 deg
every 0.5 deg, 200 XFOIL iterations, bin/xfoil(.exe).

Nothing is written when the polar's CDmin is below the laminar limit, when the polar has too few points (or too few
in the alpha ranges the key-point fit needs) or when the key-point extraction fails.

Output, in data/support/<--folder> or in --output-dir (created if missing; no folder above it is created)
------
    <file-name>          the key-point file geo_file_creation reads:
                             <name> Mach=<mach> Re=<re> Ncrit=<ncrit>
                             CLmin CDCLmin CLCDmin CDmin CLmax CDCLmax
                             <the six values>
    <stem>_polar.txt     the polar the values came from
Default file name: kp_extraction.py's <name>_M<mach>_RE<re>.txt with '.' -> '-', e.g. naca_0012_M0-093_RE250000.txt.
An existing file is overwritten only after confirmation (old and new key points are shown side by side first).

Usage
-----
    python src/utilities/airfoil_kp_exporter.py --airfoil naca0012_unit.csv --protocol zero_out --kp-seed 42 --output-dir <scratch>
    python src/utilities/airfoil_kp_exporter.py --airfoil naca0012_unit.csv --protocol zero_out --kp-seed 42 --folder naca_kp --file-name naca_0012_M0-093_RE250000_zero_out_s42.txt
"""
import argparse
import math
import re
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

utilities_dir = Path(__file__).resolve().parent
src_dir = utilities_dir.parent
project_root = src_dir.parent
support_files_dir = project_root / "data" / "support"
sys.path.insert(0, str(src_dir))

from cst_modeling.foil import check_validity  # noqa: E402
from geometric_support import Airfoil_Generator  # noqa: E402
from simulation_support import (SWEEP_MODES, Xfoil_Polar_Creator as XPR, laminar_cd_limit, polar_key_points,  # noqa: E402
                                protocol_settings, read_polar_arrays)
from simulation_support.key_point_processing import (KEY_POINT_FILE_HEADER, InsufficientPointsError,  # noqa: E402
                                                     PoorDistributionError, extract_polar_metadata, write_key_points_file)
from simulation_support.sweep_protocol_check import default_temp_root  # noqa: E402

AIRFOIL_ID = "kpx"  # names the temp .dat and polar: short, XFOIL reads at most 64 characters of a file name


def resolve_csv(arg: str) -> Path:
    """An airfoil CSV given as written, or inside data/support (as in airfoil_plotter.py)."""
    path = Path(arg)
    if not path.exists() and (support_files_dir / path).exists():
        path = support_files_dir / path
    if not path.exists():
        sys.exit(f"[ERROR] Airfoil file not found: {arg}")
    return path


def short(path: Path) -> str:
    """Path relative to the repository when it is inside it."""
    try:
        return str(Path(path).resolve().relative_to(project_root))
    except ValueError:
        return str(path)


def fail(message: str):
    sys.exit(f"\n[ERROR] {message}\n[ERROR] Nothing was written.")


def default_file_name(airfoil_name: str, meta: dict) -> str:
    """kp_extraction.py convention, <name>_M<mach>_RE<re>.txt, with '.' -> '-' (e.g. naca_0012_M0-093_RE250000.txt)."""
    slug = re.sub(r"[^0-9a-z]+", "_", airfoil_name.lower()).strip("_")
    return f"{slug}_M{meta['mach'].replace('.', '-')}_RE{meta['re'].replace('.', '-')}.txt"


def read_key_points(path: Path):
    """The six values of an existing key-point file (its last non-empty line), or None if they cannot be read."""
    try:
        lines = [line for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
        values = [float(v) for v in lines[-1].split()]
    except (OSError, ValueError, IndexError):
        return None
    return values if len(values) == 6 else None


def confirm_overwrite(target: Path, old, new) -> bool:
    """Shows the current and the new key points side by side and asks before overwriting."""
    print()
    if old is None:
        print(f"The key points of the existing {target.name} could not be read.")
    else:
        print(f"  {'':<8} {'current':>12} {'new':>12} {'change':>8}")
        for name, o, n in zip(KEY_POINT_FILE_HEADER.split(), old, new):
            change = f"{(n - o) / abs(o) * 100:+7.1f}%" if o != 0 else "     n/a"
            print(f"  {name:<8} {o:>12.6g} {n:>12.6g} {change}")
    try:
        answer = input(f"File {short(target)} exists. Overwrite it? [y/N] ")
    except EOFError:  # no console to answer
        answer = ""
    return answer.strip().lower() == "y"


def parse_args():
    parser = argparse.ArgumentParser(description="Simulate an airfoil and write its key-point file (the CDCL polar of the AVL winglet sections).")
    parser.add_argument('--airfoil', required=True, help="Airfoil CSV (path, or a file name in data/support), simulated with all CST increments at 0.")
    parser.add_argument('--protocol', required=True, choices=SWEEP_MODES, help="Alpha-sweep protocol.")
    parser.add_argument('--kp-seed', type=int, required=True, help="random_state of the key-point GPR fit (the dataset uses 42).")
    parser.add_argument('--re', type=float, help="Reynolds number (default 2.5e5, from protocol_settings).")
    parser.add_argument('--mach', type=float, help="Mach number (default 0.09281, from protocol_settings).")
    parser.add_argument('--aseq', type=float, nargs=3, metavar=('START', 'STOP', 'STEP'),
                        help="Angles of attack in deg (default -20 20 0.5, from protocol_settings).")
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument('--folder', help="Write into data/support/FOLDER (created if missing).")
    destination.add_argument('--output-dir', help="Write into this folder instead (created if missing; its parent must exist).")
    parser.add_argument('--file-name', help="Key-point file name (default: <airfoil>_M<mach>_RE<re>.txt from the polar header).")
    return parser.parse_args()


def export(args, settings, out_dir: Path, run_dir: Path):
    airfoil_path = Path(settings.base_airfoil)
    generator = Airfoil_Generator(base_airfoil_csv=str(airfoil_path), n_cst=settings.n_cst, verbose=False)
    name = generator.airfoil_base_name or airfoil_path.stem
    angles = settings.angles()
    print(f"Airfoil   {name} ({short(airfoil_path)}), all CST increments 0")
    print(f"Protocol  {settings.sweep_mode} | key-point seed {settings.kp_seed}")
    print(f"Flow      Re {settings.reynolds:g} | Mach {settings.mach:g} | alpha {settings.aseq[0]:g} to {settings.aseq[1]:g} "
          f"every {settings.aseq[2]:g} deg ({len(angles)} angles) | {settings.xfoil_iter} XFOIL iterations")

    # Same order as simulate_airfoil_once: airfoil -> validity -> .dat -> XFOIL -> polar -> key points
    x, yu, yl = generator.generate(cst_coeffs=np.zeros(2 * settings.n_cst))
    if not check_validity(x, yu, yl, verbose=False):
        fail(f"{name} fails check_validity after the CST fit; it was not simulated.")
    generator._create_dat_file(output_dir=run_dir, n_cst=settings.n_cst, x=x, yl=yl, yu=yu, file_name=AIRFOIL_ID)
    polar_dir = run_dir / "polars"
    polar_path = polar_dir / f"polar_{AIRFOIL_ID}.txt"
    print(f"\nRunning XFOIL ({settings.sweep_mode})...")
    XPR(xfoil_exe_path=settings.xfoil_exe, main_dir=run_dir, airfoil_dir=run_dir, polar_dir=polar_dir,
        verbose=False).run_simulation(run_dir / f"{AIRFOIL_ID}.dat", str(polar_path), angles, visc=settings.reynolds,
                                      mach=settings.mach, iter=settings.xfoil_iter, sweep_mode=settings.sweep_mode)

    alpha, cl, cd = read_polar_arrays(polar_path)
    print(f"Polar     {len(alpha)} converged points of {len(angles)} angles")
    if not len(alpha):
        fail("XFOIL returned no converged points.")
    i = int(np.argmin(cd))
    cd_min, limit = float(cd[i]), laminar_cd_limit(settings.reynolds)
    print(f"CDmin     {cd_min:.5f} at alpha {alpha[i]:g} deg | laminar limit {limit:.5f} | "
          + ("above the limit" if cd_min >= limit else "BELOW THE LIMIT"))
    if cd_min < limit:
        fail(f"CDmin {cd_min:.5f} is below the laminar limit {limit:.5f} at Re {settings.reynolds:g}: not a physical polar.")

    try:
        kp = polar_key_points(polar_path, settings.kp_seed)
    except InsufficientPointsError as e:
        fail(f"Too few polar points for the key-point fit. {e}")
    except PoorDistributionError as e:
        fail(f"Too few polar points in the alpha ranges the key-point fit needs. {e}")
    except Exception as e:
        fail(f"Key-point extraction failed: {type(e).__name__}: {e}")
    if not all(math.isfinite(v) for v in kp):
        fail(f"Key-point extraction returned non-finite values: {kp}")
    print("Key points")
    for label, value in zip(KEY_POINT_FILE_HEADER.split(), kp):
        print(f"  {label:<8} {value:.6g}")

    meta = extract_polar_metadata(polar_path)
    if meta['re'] == '0.0':
        fail(f"Mach, Re and Ncrit could not be read from the polar header of {polar_path}.")
    meta['name'] = name  # the polar header names the temp .dat; line 1 names the airfoil
    file_name = args.file_name or default_file_name(name, meta)
    target = out_dir / file_name
    polar_target = out_dir / f"{target.stem}_polar.txt"

    if target.exists() and not confirm_overwrite(target, read_key_points(target), kp):
        print("Not overwritten. Nothing was written.")
        return
    out_dir.mkdir(exist_ok=True)
    result = {'min_cl_point': {'CL': kp[0], 'CD': kp[1]},
              'min_cd_point': {'CL': kp[2], 'CD': kp[3]},
              'max_cl_point': {'CL': kp[4], 'CD': kp[5]}}
    write_key_points_file(result, target, meta)
    shutil.copyfile(polar_path, polar_target)
    print(f"\nWrote {short(target)}")
    print(f"      {short(polar_target)}")


if __name__ == "__main__":
    args = parse_args()
    airfoil_path = resolve_csv(args.airfoil)

    # protocol_settings() already points at bin/xfoil.exe on Windows and bin/xfoil elsewhere
    defaults = protocol_settings(args.protocol)
    settings = replace(defaults, base_airfoil=str(airfoil_path), kp_seed=args.kp_seed,
                       reynolds=defaults.reynolds if args.re is None else args.re,
                       mach=defaults.mach if args.mach is None else args.mach,
                       aseq=defaults.aseq if args.aseq is None else list(args.aseq))

    out_dir = support_files_dir / args.folder if args.folder else Path(args.output_dir)
    if not out_dir.parent.is_dir():
        sys.exit(f"[ERROR] The destination's parent folder does not exist: {out_dir.parent}")
    if args.file_name and Path(args.file_name).name != args.file_name:
        sys.exit("[ERROR] --file-name takes a file name only; choose the folder with --folder or --output-dir.")
    if not Path(settings.xfoil_exe).exists():
        sys.exit(f"[ERROR] XFOIL not found: {settings.xfoil_exe}")

    run_dir = default_temp_root() / "kp_export"
    shutil.rmtree(run_dir, ignore_errors=True)
    (run_dir / "polars").mkdir(parents=True)
    try:
        export(args, settings, out_dir, run_dir)
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
