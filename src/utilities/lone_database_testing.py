import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

current_file = Path(__file__).resolve()
project_root = current_file.parents[2]

from cst_modeling.foil import check_validity
from simulation_support import (
    Xfoil_Polar_Creator as XPR,
    Key_Point_Extraction,
    HossainWingRec_Creation,
    HossainWing60DEG_Creation,
    HossainWingFlat_Creation,
    AVL_Run_Case,
)
from geometric_support import Airfoil_Generator, Validate_Airfoil

WING_CONFIG_MAP = {
    "rec": (HossainWingRec_Creation, "rec"),
    "flat": (HossainWingFlat_Creation, "flat"),
    "60deg": (HossainWing60DEG_Creation, "60DEG")
}

def parse_cst_coefficients(cst_str: str, n_cst: int) -> np.ndarray:
    """Parses a numeric string into a numpy array of CST coefficients.

    Inputs:
        cst_str (str): Space or comma separated string of coefficient values.
        n_cst (int): CST polynomial order per surface.

    Returns:
        np.ndarray: 1D array of length 2 * n_cst containing lower and upper coefficients.
    """
    cleaned = cst_str.replace(",", " ").strip()
    values = [float(v) for v in cleaned.split()] if cleaned else []
    expected_len = 2 * n_cst
    if len(values) != expected_len:
        print(f"[WARNING] Expected {expected_len} CST values, got {len(values)}. Padding/truncating with zeros.")
        values = (values + [0.0] * expected_len)[:expected_len]
    return np.array(values, dtype=float)

def get_user_inputs() -> dict:
    """Collects simulation parameters via CLI arguments or interactive prompts with validation.

    Returns:
        dict: Dictionary of flow conditions, geometry configuration, and CST coefficients.
    """
    parser = argparse.ArgumentParser(description="Single-case pipeline execution script for debugging.")
    parser.add_argument("--case-id", type=str, default=None, help="Custom identifier string for testing case.")
    parser.add_argument("--n-cst", type=int, default=None, help="Order of CST polynomial.")
    parser.add_argument("--cst-coeffs", type=str, default=None, help="CST coefficients (space/comma separated).")
    parser.add_argument("--reynolds", type=float, default=None, help="Reynolds number.")
    parser.add_argument("--mach", type=float, default=None, help="Mach number.")
    parser.add_argument("--aseq", type=str, default=None, help="Alpha sweep sequence as start end step.")
    parser.add_argument("--wing-config", type=str, default=None, help="Wing configuration (rec, flat, 60deg).")
    parser.add_argument("--verbose", action="store_true", help="Enable raw solver output.")
    parser.add_argument("--run-file", type=str, default=None, help="Path to custom AVL run case file.")
    parser.add_argument("--base-airfoil", type=str, default=None, help="Path to base airfoil CSV file.")
    args = parser.parse_args()

    # Case ID
    case_id = args.case_id or input("Enter Case Identifier, or press enter for default [debug_case]: ").strip() or "debug_case"

    # CST Order (Mandatory)
    if args.n_cst is not None:
        n_cst = args.n_cst
    else:
        while True:
            n_cst_in = input("Enter CST order (n_cst): ").strip()
            if n_cst_in.isdigit():
                n_cst = int(n_cst_in)
                break
            print("[ERROR] CST order must be specified as an integer.")

    # CST Coefficients (Mandatory)
    if args.cst_coeffs is not None:
        cst_array = parse_cst_coefficients(args.cst_coeffs, n_cst)
    else:
        while True:
            cst_in = input(f"Enter {2*n_cst} space-separated CST coefficients (lower then upper): ").strip()
            if cst_in:
                cst_array = parse_cst_coefficients(cst_in, n_cst)
                break
            print("[ERROR] CST coefficients must be specified.")

    # Flow Conditions (Defaultable)
    reynolds = args.reynolds if args.reynolds is not None else float(input("Enter Reynolds number, or press enter for default [250000]: ").strip() or 2.5e5)
    mach = args.mach if args.mach is not None else float(input("Enter Mach number, or press enter for default [0.09281]: ").strip() or 0.09281)

    # Alpha sequence (Defaultable)
    if args.aseq is not None:
        aseq_parts = [float(x) for x in args.aseq.replace(",", " ").split()]
        aseq = aseq_parts if len(aseq_parts) == 3 else [-20.0, 20.0, 0.5]
    else:
        aseq_in = input("Enter Alpha sequence (start end step), or press enter for default [-20 20 0.5]: ").strip()
        aseq = [float(x) for x in aseq_in.split()] if aseq_in else [-20.0, 20.0, 0.5]

    # Wing Configuration Validation (Mandatory)
    config_in = args.wing_config.lower() if args.wing_config else None
    while config_in not in WING_CONFIG_MAP:
        if config_in is not None:
            print(f"[ERROR] Invalid wing configuration '{config_in}'. Valid choices are: {list(WING_CONFIG_MAP.keys())}")
        config_in = input("Enter wing configuration (rec / flat / 60deg): ").strip().lower()

    return {
        "case_id": case_id,
        "n_cst": n_cst,
        "cst_array": cst_array,
        "reynolds": reynolds,
        "mach": mach,
        "aseq": aseq,
        "wing_config": config_in,
        "verbose": args.verbose,
        "run_file": args.run_file,
        "base_airfoil": args.base_airfoil
    }

def extract_force_metrics(file_path: Path) -> Optional[Tuple[str, float, float]]:
    """Extracts airfoil identifier, total CL, and total CD from an AVL force output file.

    Inputs:
        file_path (Path): Path object pointing to the text force file.

    Returns:
        Optional[Tuple[str, float, float]]: Tuple of (airfoil_name, cl_tot, cd_tot) if successful, else None.
    """
    airfoil, cl_tot, cd_tot = None, None, None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if "Configuration:" in line and " for " in line:
                    airfoil = line.split(" for ", 1)[1].strip()
                elif 'CLtot =' in line:
                    cl_tot = float(line.split('=')[1].strip())
                elif 'CDtot =' in line:
                    cd_tot = float(line.split('=')[1].strip())
        if all((airfoil, cl_tot is not None, cd_tot is not None)):
            return airfoil, cl_tot, cd_tot
        print(f"[WARNING] Incomplete force data in {file_path}.")
        return None
    except FileNotFoundError:
        print(f"[ERROR] Force file not found: {file_path}")
        return None
    except (ValueError, IndexError) as e:
        print(f"[ERROR] Error parsing force data in {file_path}: {e}")
        return None

def main():
    print("Now executing Environment Setup & Input Ingestion")
    # Input Handling and Environment Setup
    inputs = get_user_inputs()
    
    lone_temp_root = Path.home() / ".lone_training_temp"
    airfoils_temp_dir = lone_temp_root / "airfoils"
    polars_temp_dir = lone_temp_root / "polars"
    geo_files_temp_dir = lone_temp_root / "geometry files"
    force_files_temp_dir = lone_temp_root / "force files"
    support_files_temp_dir = lone_temp_root / "support_files"

    for d in [lone_temp_root, airfoils_temp_dir, polars_temp_dir, geo_files_temp_dir, force_files_temp_dir, support_files_temp_dir]:
        d.mkdir(parents=True, exist_ok=True)

    bin_name_xfoil = "xfoil.exe" if sys.platform.startswith("win") else "xfoil"
    bin_name_avl = "avl.exe" if sys.platform.startswith("win") else "avl"
    xfoil_exe = project_root / "bin" / bin_name_xfoil
    avl_exe = project_root / "bin" / bin_name_avl

    support_files_dir = project_root / "data" / "support"
    base_airfoil_path = Path(inputs["base_airfoil"]) if inputs["base_airfoil"] else (support_files_dir / "naca0012_unit.csv")
    naca_0012_kp_path = support_files_dir / "naca_kp" / "naca_0012_M0-9281_RE250k_M0-093_RE250000.txt"
    case_file_src = Path(inputs["run_file"]) if inputs["run_file"] else (support_files_dir / "Hossain_sqwing_case.run")

    airfoil_name = inputs["case_id"]

    print("Now executing Airfoil Genesis & Geometric Validation")
    # Airfoil Genesis and Geometric Validation
    airfoil_creator = Airfoil_Generator(
        base_airfoil_csv=str(base_airfoil_path),
        n_cst=inputs["n_cst"],
        verbose=inputs["verbose"]
    )
    
    x_mod, yu_mod, yl_mod = airfoil_creator.generate(
        cst_coeffs=inputs["cst_array"],
        show_plot=False,
        debug=inputs["verbose"]
    )

    if not check_validity(x_mod, yu_mod, yl_mod, verbose=inputs["verbose"]):
        print("[WARNING] Airfoil failed cst_modeling check_validity check.")

    try:
        va = Validate_Airfoil(
            x=x_mod,
            yu=yu_mod,
            yl=yl_mod,
            cst_coeffs=inputs["cst_array"],
            n_cst=inputs["n_cst"],
            name=airfoil_name,
            verbose=inputs["verbose"]
        )
        filters = [
            va.thickness_filters(th_t_min=0.03, th_t_max=0.25, th_t_x_max=0.45, th_t_x_min=0.18),
            va.sharpness_filter(curv_cont=True, th_rLE=0.01),
            va.monotonic_thickness_filter(),
            va.camber_filters(th_c_max=0.12, th_c_min=-0.02)
        ]
        for status, reason in filters:
            if status == 'invalid':
                print(f"[WARNING] Geometric filter flagged invalid: {reason}")
    except Exception as e:
        print(f"[WARNING] Could not complete geometric validation filters: {e}")

    airfoil_creator._create_dat_file(
        output_dir=airfoils_temp_dir,
        n_cst=inputs["n_cst"],
        x=x_mod,
        yl=yl_mod,
        yu=yu_mod,
        file_name=airfoil_name,
        debug=inputs["verbose"]
    )
    airfoil_dat_path = airfoils_temp_dir / f"{airfoil_name}.dat"

    print("Now executing Aerodynamic Polar Formulation")
    # Aerodynamic Polar Formulation
    polar_file_path = polars_temp_dir / f"polar_{airfoil_name}.txt"
    xfoil_case = XPR(
        xfoil_exe_path=str(xfoil_exe),
        main_dir=lone_temp_root,
        airfoil_dir=airfoils_temp_dir,
        polar_dir=polars_temp_dir,
        verbose=inputs["verbose"]
    )
    
    aseq = inputs["aseq"]
    num_steps = int((aseq[1] - aseq[0]) / aseq[2]) + 1
    angles = [aseq[0] + j * aseq[2] for j in range(num_steps)]

    # This script was intended to use with the original sequential alpha sweep, it has been updated to maintain intended behavior. To see how the type of sweep affect simulation refer to alpha_sweep.py
    xfoil_case.run_simulation(
        airfoil_file_path=str(airfoil_dat_path),
        polar_file_path=str(polar_file_path),
        alfa_sequence=angles,
        visc=inputs["reynolds"],
        mach=inputs["mach"],
        sweep_mode="sequential"
    )

    print("Now executing Aerodynamic Key Point Extraction")
    # Aerodynamic Key Point Extraction
    key_points = Key_Point_Extraction(
        polar_file_paths=[str(polar_file_path)],
        verbose=inputs["verbose"]
    )
    key_points_data = key_points.process_files()

    print("Now executing AVL Configuration Formulation")
    # AVL Configuration Formulation
    creator_class, identifier = WING_CONFIG_MAP[inputs["wing_config"]]
    geo_creator = creator_class(
        key_points_data=key_points_data,
        airfoil_abs_paths=[str(airfoil_dat_path)],
        naca_key_points_data=str(naca_0012_kp_path),
        output_dir=str(geo_files_temp_dir),
        verbose=inputs["verbose"],
        pb=False
    )
    geo_creator.create_geo_files()
    
    geo_file_path = geo_files_temp_dir / f"{airfoil_name}_{identifier}_geo.avl"
    if not geo_file_path.exists():
        found_geos = list(geo_files_temp_dir.glob(f"*{airfoil_name}*.avl"))
        if found_geos:
            geo_file_path = found_geos[0]

    print("Now executing AVL Execution & Force Extraction")
    # AVL Execution and Force Extraction
    case_file_temp = support_files_temp_dir / case_file_src.name
    with open(case_file_src, 'r') as src_f, open(case_file_temp, 'w') as dst_f:
        dst_f.write(src_f.read())

    avl_case = AVL_Run_Case(
        geo_files=[str(geo_file_path)],
        case_file=str(case_file_temp),
        avl_exe_path=str(avl_exe),
        output_dir=str(force_files_temp_dir),
        debug=inputs["verbose"]
    )
    avl_case.run()

    force_file_path = force_files_temp_dir / f"force_{airfoil_name}_{identifier}_geo.txt"
    if not force_file_path.exists():
        found_forces = list(force_files_temp_dir.glob(f"force_*{airfoil_name}*.txt"))
        if found_forces:
            force_file_path = found_forces[0]

    metrics = extract_force_metrics(force_file_path)
    if metrics:
        name, cl, cd = metrics
        efficiency = cl / cd if cd != 0 else 0.0
        print("\n" + "="*40)
        print("SIMULATION RESULTS")
        print("="*40)
        print(f"Airfoil/Case : {name}")
        print(f"CL           : {cl:.6f}")
        print(f"CD           : {cd:.6f}")
        print(f"Efficiency   : {efficiency:.6f}")
        print("="*40 + "\n")

if __name__ == "__main__":
    main()