import matplotlib.pyplot as plt
import numpy as np
import os
import contextlib
import sys
import io
import random
import subprocess
import shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count, get_context
from contextlib import redirect_stdout, redirect_stderr, contextmanager
from itertools import product
from tqdm import tqdm
from scipy.interpolate import interp1d
from cst_modeling.section import cst_foil_fit, cst_foil
from cst_modeling.foil import FoilGeoFeatures
from cst_modeling.foil import FoilModification, FoilGeoFeatures, check_validity
# Import de Clases creadas para este proyecto
from simulation_support import Xfoil_Polar_Creator as XPR
from simulation_support import Key_Point_Extraction, RecWingGeo_Creation, AVL_Run_Case
from geometric_support import Airfoil_Generator, Validate_Airfoil, create_modification_combinations
from file_mainp import File_Manipulation,get_all_files_from, create_rar_from_folder, xfoil_rel_path

# For custom package:
# Get the path to the parent directory (xfoil_project).
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
# Add this path to the list of places Python looks for modules.
sys.path.insert(0, project_root)
# Define final folders
airfoils_dir = project_root / "airfoils"
force_files_dir = project_root / "force files"
polars_dir = project_root / "polars"
AI_results_dir = project_root / "AI results"
support_files_dir = project_root / "support_files"
geo_files_dir = project_root / "geometry files"
# Define temp folders
# Temp folders with a short path are needed for XFOIL and AVL
training_temp_root = Path(r'P:\training_temp') # Uses my personal 'P' drive, any drive can be used
os.makedirs(training_temp_root, exist_ok=True)
subprocess.run(['attrib', '+h', training_temp_root], check=True, creationflags=subprocess.CREATE_NO_WINDOW)
airfoils_temp_dir = training_temp_root / "airfoils"
force_files_temp_dir = training_temp_root / "force files"
polars_temp_dir = training_temp_root / "polars"
AI_results_temp_dir = training_temp_root / "AI results"
support_files_temp_dir = training_temp_root / "support_files"
geo_files_temp_dir = training_temp_root / "geometry files"

# === Geometric Generation Setup === #
base_airfoil = os.path.join(support_files_dir,'naca0012_UNIT_0ALFA.csv')
n_cst = 3

# Ranges for values that modify CST Curves (starting_value, end_value, step)
ranges = {
    'first': (-1, 0, 1),   # range for value closest to LE
    'middle': (-1, 0, 1),  # range for middle points
    'last': (-1, 0, 1)     # range for value closest to TE
}
cmb_datalist = create_modification_combinations(n_cst, ranges, ranges, debug=False)

def Gemoetric_Database_Generation(base_airfoil_path, n_cst, cmb_datalist, generate_invalid=False):
    '''
    Fills a folder wuth generated airfoils matching the combination list and filtered for real plausible airfoils
    
    Parameters
    ----------
    base_airfoil_path : str, rstr
        Takes the absolute path to the base airfoil for modification

    n_cst : int
        Order of CST coefficients

    cmb_datalist : np.ndarray
        A list containing rows of cst coeeffiecients, format in Airfoil_Generator
    
    [TEMP] index : list
        Temporary limited list of random cmb_datalist values, will be replaced by 'for cmb in cmb_datalist'
    '''
    invalid_output_dir = airfoils_temp_dir / "invalid airfoils"
    log_file_path = airfoils_temp_dir / "Invalid_airfoils_log.txt"

    airfoil_creator = Airfoil_Generator(
            base_airfoil_csv=base_airfoil_path,
            n_cst=n_cst,
            verbose=False    
        )
    for i, cmb in enumerate(cmb_datalist):
        # Run the generation process
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
                    cst_coeffs=cmb_datalist[i],
                    n_cst=n_cst,
                    name=airfoil_name,
                    verbose=False
                    )
                
                if valid == 'valid':
                    valid, log_reason = va.thickness_filters(th_t_min=0.03, th_t_x=0.6)
                if valid == 'valid':
                    valid, log_reason = va.sharpness_filter(curv_cont=True, th_rLE=0.005)
                if valid == 'valid':
                    valid, log_reason = va.monotonic_thickness_filter()
                if valid == 'valid':
                    valid, log_reason = va.camber_filters(th_c_max=0.4, th_c_min=-0.4)

                if valid == 'invalid':
                    validity = False

                if validity:
                    airfoil_creator._create_dat_file(
                        output_dir=airfoils_temp_dir, 
                        x=x_mod,
                        yl=yl_mod,
                        yu=yu_mod,
                        file_name=airfoil_name,
                        debug=False
                        )
            except ValueError as e:
                print(f"Could not create Geo Features Object due to unexpected error: {e}")
        else:
            cst_coeffs = cmb_datalist[i]
            cmb_l = np.array(cst_coeffs[:n_cst])
            cmb_u = np.array(cst_coeffs[n_cst:])
            reason = "Check_validity failed"
            log_line = f"{airfoil_name}.dat: cmb_l = {cmb_l}, cmb_u = {cmb_u} | Reason: {reason}\n"
            valid, log_reason = 'invalid', log_line
        
        if not validity: # If validity == False then we write to log, optionally save airfoil
            if generate_invalid:
                airfoil_creator._create_dat_file(
                    output_dir=invalid_output_dir,
                    x=x_mod,
                    yl=yl_mod,
                    yu=yu_mod,
                    file_name=airfoil_name,
                    debug=False
                )
            with open(log_file_path, "a") as log:
                log.write(f"{log_reason}\n")

    # Save airfoils to main path in compressed form
    dataset_name = "airfoils_dataset_15"
    create_rar_from_folder(airfoils_temp_dir, airfoils_dir, dataset_name, verbose=False)

    return

def Polar_Database_Generation(verbose=False):
    # --- File preparation for class ---
    xfoil_exe = os.path.join(support_files_dir, 'xfoil.exe')
    print(xfoil_exe)
    
    airfoil_abs_paths = get_all_files_from(airfoils_temp_dir, extension='dat')
    #for file in airfoil_abs_paths:
    #    print(file)
    
    polar_abs_paths = []
    for file in airfoil_abs_paths:
        airfoil_name = os.path.splitext(os.path.basename(file))[0]
        polar_abs_path = os.path.join(polars_temp_dir, f"polar_{airfoil_name}.txt")
        polar_abs_paths.append(polar_abs_path)
    #print(polar_abs_paths[1])
    
    # --- ANGLE OF ATTACK SEQUENCE ---
    ASEQ = [-20, 20, 1]
    num_steps = int((ASEQ[1] - ASEQ[0]) / ASEQ[2]) + 1
    ANGLES = [ASEQ[0] + i * ASEQ[2] for i in range(num_steps)]
    # ---------------------------------
    
    for i, file in enumerate(airfoil_abs_paths): 
        try:
            # Create an instance of the runner
            xfoil_case = XPR(
                xfoil_exe_path=xfoil_exe,
                main_dir=training_temp_root,
                airfoil_dir=airfoils_temp_dir,
                polar_dir=polars_temp_dir,
                verbose=False
                )
    
            # Run the simulation
            sim_success = xfoil_case.run_simulation(
                airfoil_file_path=airfoil_abs_paths[i],
                polar_file_path = polar_abs_paths[i],
                alfa_sequence = ANGLES
            )
    
            if sim_success and verbose:
                print(f"\nSimulation successful! Output saved to {polar_abs_paths[i]}")
            else:
                if verbose:
                    print(f"\nSimulation failed. Check logs.")
    
        except FileNotFoundError as e:
            print(f"\n[ERROR] A file was not found: {e}")
        except Exception as e:
            print(f"\n[ERROR] A critical error occurred: {e}")

    # Save polars to main path in compressed form
    polar_dataset_name = "polars_dataset_15"
    create_rar_from_folder(polars_temp_dir, polars_dir, polar_dataset_name, verbose=False)
    
    return

def Geometric_File_Creation():
    polars_abs_paths = get_all_files_from(polars_temp_dir)
    
    key_points = Key_Point_Extraction(
        polar_file_paths=polars_abs_paths
        )
    
    key_points_data = key_points.process_files()
    KP_error_log = os.path.join(polars_temp_dir, 'KP_error_log.txt')
    key_points.save_error_log(KP_error_log)
    
    airfoil_abs_paths = get_all_files_from(airfoils_temp_dir, extension='dat')
    geo_creator = RecWingGeo_Creation(
                key_points_data=key_points_data,
                airfoil_abs_paths=airfoil_abs_paths,
                output_dir=str(geo_files_temp_dir),
                verbose=False
            )
    geo_creator.create_geo_files()
    geo_error = os.path.join(geo_files_temp_dir, 'geo_error_log.txt')
    geo_creator.save_error_log(geo_error)

    geo_files_name = "geo_files_dataset_15"
    create_rar_from_folder(geo_files_temp_dir, geo_files_dir, geo_files_name, verbose=False)

    return

def Final_Coefficient_Generation():
    avl_exe = os.path.join(support_files_dir, 'avl.exe')
    case_file = os.path.join(support_files_dir, "sq_wing.run")
    support_files_temp_dir.mkdir(exist_ok=True)
    case_file_temp = os.path.join(support_files_temp_dir, "sq_wing.run")
    shutil.copyfile(case_file, case_file_temp)
    geo_abs_paths = get_all_files_from(geo_files_temp_dir, extension='avl')

    # Set up instance for simulation
    avl_case = AVL_Run_Case(
            geo_files=geo_abs_paths,
            case_file=case_file_temp,
            avl_exe_path=avl_exe,
            output_dir=force_files_temp_dir,
            debug=True
        )
    # Run the simulation.
    avl_case.run()

    force_files_name = "force_files_dataset_15"
    create_rar_from_folder(force_files_temp_dir, force_files_dir, force_files_name, verbose=False)

    return

if __name__ == "__main__":
    #Gemoetric_Database_Generation(base_airfoil, n_cst, cmb_datalist, generate_invalid=True)
    #Polar_Database_Generation()
    #Geometric_File_Creation()
    Final_Coefficient_Generation()