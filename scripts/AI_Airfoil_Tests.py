import os
import sys
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from simulation_support import Xfoil_Polar_Creator as XPR
from simulation_support import Key_Point_Extraction, RecWingGeo_Creation, AVL_Run_Case
from geometric_support import Airfoil_Generator
from file_handling_support import compression_Alma, get_all_files_from

# Get the path to the parent directory (xfoil_project).
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
# Add this path to the list of places Python looks for modules.
sys.path.insert(0, str(project_root))

# Main Directory and Naming Setup ---
dataset = "dataset_15"
# This prefix will be used for the main output folder and all archive names
dataset_name_prefix = f"AI_Test_Results_{dataset}"
# The main folder where all results for this dataset will be saved
output_root_folder = project_root / dataset_name_prefix

# === Final Directory Definitions (now inside the main output folder) === #
airfoils_dir = output_root_folder / "airfoils"
force_files_dir = output_root_folder / "force files"
polars_dir = output_root_folder / "polars"
geo_files_dir = output_root_folder / "geometry files"
support_files_dir = project_root / "support_files" # Support files stay in the project root

# === Temporary Directory Definitions (no changes needed here) === #
testing_temp_root = Path(r'C:\testing_temp') 
airfoils_temp_dir = testing_temp_root / "airfoils"
force_files_temp_dir = testing_temp_root / "force files"
polars_temp_dir = testing_temp_root / "polars"
geo_files_temp_dir = testing_temp_root / "geometry files"
support_files_temp_dir = testing_temp_root / "support_files"

def extract_force_data(file_path: Path):
    """Extracts airfoil name, CL, and CD from a force file."""
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
        print(f"Warning: Incomplete data in {file_path}. Skipping.")
        return None
    except FileNotFoundError:
        print(f"Error: Force file not found: {file_path}")
        return None
    except (ValueError, IndexError) as e:
        print(f"Error: Error parsing force data in {file_path}: {e}")
        return None

def create_temp_dirs():
    """Creates all necessary temporary directories."""
    os.makedirs(airfoils_temp_dir, exist_ok=True)
    os.makedirs(force_files_temp_dir, exist_ok=True)
    os.makedirs(polars_temp_dir, exist_ok=True)
    os.makedirs(geo_files_temp_dir, exist_ok=True)
    os.makedirs(support_files_temp_dir, exist_ok=True)

if __name__ == "__main__":
    start_time = datetime.now()
    print(f"AI Airfoil test process started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # === Initialize Helper Objects === #
    rar_Alma = compression_Alma()

    # === Test Setup === #
    n_cst = 5
    # --- 2. NEW: Use a dictionary to define AI airfoils with names ---
    ai_airfoils_data = {
        "Dataset Best" : np.array([-0.375, 0.375,-0.125, 0.125,-0.125, 0.125,-0.125,-0.125,-0.125, 0.125]),
        "AI_Airfoil_MLP": np.array([-0.061702, 0.056080,-0.016760,-0.120047,-0.460390, 0.057240, 0.193765,-0.213402,-0.064503,-0.060050]), 
        "AI_Airfoil_LGBM": np.array([-0.060542,-0.071230, 0.234374,-0.274851,-0.470854, 0.146988, 0.159061, 0.344600,-0.365011,-0.149187]), 
        "AI_Airfoil_SKNN": np.array([ 0.071458,-0.023502,-0.142943,-0.099897,-0.481719,-0.098045, 0.242992,-0.052766,-0.319298,-0.154036])  
    }
    base_airfoil_path = support_files_dir / 'naca0012_UNIT_0ALFA.csv'
    ASEQ = [-20, 20, 1]

    # Create all necessary directories (both temp and the new final output folder)
    create_temp_dirs()
    os.makedirs(output_root_folder, exist_ok=True)
    
    # Generate Airfoil Geometry Files
    print("\n--- Generating Airfoil Geometry ---")
    airfoil_creator = Airfoil_Generator(
        base_airfoil_csv=base_airfoil_path,
        n_cst=n_cst,
        verbose=False
    )
    for airfoil_name, cst_coeffs in tqdm(ai_airfoils_data.items(), desc="Creating .dat files"):
        x_mod, yu_mod, yl_mod = airfoil_creator.generate(cst_coeffs=cst_coeffs)
        airfoil_creator._create_dat_file(
            output_dir=airfoils_temp_dir,
            n_cst=n_cst,
            x=x_mod,
            yl=yl_mod,
            yu=yu_mod,
            file_name=airfoil_name # Use the name from the dictionary
        )
    print("Airfoil geometry generation complete.")
    rar_Alma.compress_folder(airfoils_temp_dir, output_root_folder, f"{dataset_name_prefix}_airfoils")
    
    # Generate Polars using XFOIL
    print("\n--- Generating Polars (XFOIL) ---")
    airfoil_paths = get_all_files_from(airfoils_temp_dir, extension='dat')
    xfoil_exe = support_files_dir / 'xfoil.exe'
    xfoil_case = XPR(
        xfoil_exe_path=str(xfoil_exe),
        main_dir=testing_temp_root,
        airfoil_dir=airfoils_temp_dir,
        polar_dir=polars_temp_dir,
        verbose=False
    )
    num_steps = int((ASEQ[1] - ASEQ[0]) / ASEQ[2]) + 1
    ANGLES = [ASEQ[0] + j * ASEQ[2] for j in range(num_steps)]

    for airfoil_path in tqdm(airfoil_paths, desc="Running XFOIL"):
        airfoil_name = Path(airfoil_path).stem
        polar_path = polars_temp_dir / f"polar_{airfoil_name}.txt"
        xfoil_case.run_simulation(
            airfoil_file_path=airfoil_path,
            polar_file_path=str(polar_path),
            alfa_sequence=ANGLES
        )
    print("Polar generation complete.")
    rar_Alma.compress_folder(polars_temp_dir, output_root_folder, f"{dataset_name_prefix}_polars")

    # Create Wing Geometry Files for AVL
    print("\n--- Generating Wing Geometry (.geo files) ---")
    polar_paths = get_all_files_from(polars_temp_dir, extension='txt')
    key_points_extractor = Key_Point_Extraction(polar_file_paths=polar_paths)
    key_points_data = key_points_extractor.process_files()
    geo_creator = RecWingGeo_Creation(
        key_points_data=key_points_data,
        airfoil_abs_paths=airfoil_paths,
        output_dir=str(geo_files_temp_dir),
        verbose=False,
        pb=True
    )
    geo_creator.create_geo_files()
    print("AVL geometry file creation complete.")
    rar_Alma.compress_folder(geo_files_temp_dir, output_root_folder, f"{dataset_name_prefix}_geo_files")
    
    # Calculate Final Coefficients using AVL
    print("\n--- Calculating Final Coefficients (AVL) ---")
    geo_paths = get_all_files_from(geo_files_temp_dir, extension='avl')
    avl_exe = support_files_dir / 'avl.exe'
    case_file = support_files_dir / "sq_wing.run"
    shutil.copy(case_file, support_files_temp_dir)
    case_file_temp = support_files_temp_dir / "sq_wing.run"

    for geo_path in tqdm(geo_paths, desc="Running AVL"):
        avl_case = AVL_Run_Case(
            geo_files=[geo_path],
            case_file=str(case_file_temp),
            avl_exe_path=str(avl_exe),
            output_dir=force_files_temp_dir
        )
        avl_case.run()
    print("Final coefficient calculation complete.")
    rar_Alma.compress_folder(force_files_temp_dir, output_root_folder, f"{dataset_name_prefix}_force_files")

    print("\n--- Calculating Aerodynamic Efficiency (E = CL/CD) ---")
    force_file_paths = get_all_files_from(force_files_temp_dir, extension='txt') 
    
    efficiency_results = {}
    
    for file_path in tqdm(force_file_paths, desc="Processing Force Files"):
        result = extract_force_data(Path(file_path))
        if result:
            airfoil_name, cl, cd = result
            if cd == 0 :
                efficiency = 0.0
            else:
                efficiency = cl / cd
            efficiency_results[airfoil_name] = efficiency

    # Finalization
    if efficiency_results:
        print("\n" + "="*45)
        print("Final Aerodynamic Efficiency (E) Report")
        print("="*45)
        sorted_results = sorted(efficiency_results.items(), key=lambda item: item[1], reverse=True)
        for name, E in sorted_results:
            print(f"  - {name:<20}: E = {E:.4f}")
        print("="*45)

    end_time = datetime.now()
    print(f"\nProcess completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total execution time: {end_time - start_time}")

    # Clean up temporary files
    try:
        shutil.rmtree(testing_temp_root)
        print(f"Successfully removed temporary directory: {testing_temp_root}")
    except OSError as e:
        print(f"Error removing temporary directory {testing_temp_root}: {e}")