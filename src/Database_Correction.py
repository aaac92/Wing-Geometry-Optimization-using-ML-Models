import pandas as pd
import numpy as np
import logging
import shutil
import time
import os
import sys
import subprocess
from pathlib import Path
from simulation_support import Xfoil_Polar_Creator as XPR
from simulation_support import Key_Point_Extraction, SpyeyeGeo_Creation, AVL_Run_Case
from geometric_support import Airfoil_Generator

# Get the path to the parent directory.
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
support_files_dir = project_root / "support_files" # Support files stay in the project root
# Add this path to the list of places Python looks for modules.
sys.path.insert(0, str(project_root))

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

def correct_csv_database(csv_file_path: Path):
    """
    Reads a CSV, finds the row with the maximum output, and iteratively corrects
    it using a simulation until the error is within a 2% threshold.

    Args:
        csv_file_path (str): The full path to the input CSV file.
    """
    start_time = time.time()
    
    # Create a hidden temporary folder in C:\ drive
    temp_folder = Path(r"C:\temp_csv_correction")
    log_file = os.path.join(temp_folder, 'csv_correction.log')
    # Temporary Directory Definitions
    airfoils_temp_dir = temp_folder / "airfoils"
    force_files_temp_dir = temp_folder / "force files"
    polars_temp_dir = temp_folder / "polars"
    geo_files_temp_dir = temp_folder / "geometry files"
    support_files_temp_dir = temp_folder / "support_files"

    try:
        os.makedirs(temp_folder, exist_ok=True)
        os.makedirs(airfoils_temp_dir, exist_ok=True)
        os.makedirs(force_files_temp_dir, exist_ok=True)
        os.makedirs(polars_temp_dir, exist_ok=True)
        os.makedirs(geo_files_temp_dir, exist_ok=True)
        os.makedirs(support_files_temp_dir, exist_ok=True)
        # Set the folder to be hidden (Windows specific)
        subprocess.call(["attrib", "+H", temp_folder])
    except Exception as e:
        print(f"Error creating temporary directory: {e}")
        return

    # Set up logging to file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logging.info(f"--- Starting correction process for: {os.path.basename(csv_file_path)} ---")

    try:
        original_filename = os.path.basename(csv_file_path)
        base, ext = os.path.splitext(original_filename)
        corrected_filename = f"{base}_corrected{ext}"
        corrected_csv_path = project_root / corrected_filename
        
        shutil.copy(csv_file_path, corrected_csv_path)
        logging.info(f"Created a working copy for corrections at: {corrected_csv_path}")
        
        # Read Header only to avoid memory overload
        header_df = pd.read_csv(corrected_csv_path, nrows=0)
        output_column_name = header_df.columns[-1]
        feature_column_names = header_df.columns[:-1]
        
        logging.info(f"Successfully loaded CSV. Features: {list(feature_column_names)}, Output: '{output_column_name}'")

        # Iteration and Correction Loop
        max_corrections = 500 # Safety break to prevent infinite loops
        correction_count = 0
        while correction_count < max_corrections:
            iteration = correction_count + 1
            logging.info(f"\n--- Validation Cycle {iteration} ---")
            
            df = pd.read_csv(corrected_csv_path)
            # Find the index and value of the maximum output
            max_value_index = df[output_column_name].idxmax()
            current_max_output = df.loc[max_value_index, output_column_name]
            # Extract the features for that row as a NumPy array
            features_to_simulate = df.loc[max_value_index, feature_column_names].values

            logging.info(f"Checking current max output: {current_max_output:.4f} at index {max_value_index}.")
            logging.info(f"CST Coefficients extracted: {features_to_simulate}")
            
            n_cst = 5
            airfoil_name = f"Correction_{iteration}"
            base_airfoil_path = support_files_dir / 'naca0012_UNIT_0ALFA.csv'
            ASEQ = [-20, 20, 0.5]

            # Generate Airfoil Geometry File
            print(f"\n--- Generating Airfoil Geometry {iteration} ---")
            airfoil_creator = Airfoil_Generator(
                base_airfoil_csv=base_airfoil_path,
                n_cst=n_cst,
                verbose=False
            )
            x_mod, yu_mod, yl_mod = airfoil_creator.generate(cst_coeffs=features_to_simulate)
            airfoil_creator._create_dat_file(
                output_dir=airfoils_temp_dir,
                n_cst=n_cst,
                x=x_mod,
                yl=yl_mod,
                yu=yu_mod,
                file_name=airfoil_name
            )
            
            print(f"Airfoil geometry {iteration} generation complete.")

            # Generate Polars using XFOIL
            print(f"\n--- Generating Polar {iteration} (XFOIL) ---")
            airfoil_path = airfoils_temp_dir / f"{airfoil_name}.dat"
            xfoil_exe = support_files_dir / 'xfoil.exe'
            xfoil_case = XPR(
                xfoil_exe_path=str(xfoil_exe),
                main_dir=temp_folder,
                airfoil_dir=airfoils_temp_dir,
                polar_dir=polars_temp_dir,
                verbose=False
            )
            num_steps = int((ASEQ[1] - ASEQ[0]) / ASEQ[2]) + 1
            ANGLES = [ASEQ[0] + j * ASEQ[2] for j in range(num_steps)]

            polar_path = polars_temp_dir / f"polar_{airfoil_name}.txt"
            xfoil_case.run_simulation(
                airfoil_file_path=airfoil_path,
                polar_file_path=str(polar_path),
                alfa_sequence=ANGLES
            )
            print(f"Polar generation {iteration} complete.")

            # Create Wing Geometry Files for AVL
            print(f"\n--- Generating Wing Geometry {iteration} (.geo files) ---")
            kp_extract = Key_Point_Extraction(polar_file_paths=[str(polar_path)])
            low_cl_point, high_cl_point, min_cd_point = kp_extract._interpolate_and_extract(str(polar_path))
            
            placeholder_list = [] # Needed for class logic
            KP_Results = {
                'file_path': os.path.basename(str(polar_path)),
                'min_cl_point': {'CL': low_cl_point[0], 'CD': low_cl_point[1]},
                'max_cl_point': {'CL': high_cl_point[0], 'CD': high_cl_point[1]},
                'min_cd_point': {'CL': min_cd_point[0], 'CD': min_cd_point[1]}
            }
            placeholder_list.append(KP_Results)
            
            geo_creator = SpyeyeGeo_Creation(
                key_points_data=placeholder_list,
                airfoil_abs_paths=[airfoil_path],
                output_dir=str(geo_files_temp_dir),
                verbose=False,
                pb=True
            )
            
            geo_creator.create_geo_files()

            print(f"AVL geometry file {iteration} creation complete.")

            # Calculate Final Coefficients using AVL
            print(f"\n--- Calculating Final Coefficients {iteration} (AVL) ---")
            geo_path = geo_files_temp_dir / f"{airfoil_name}_geo.avl"
            avl_exe = support_files_dir / 'avl.exe'
            case_file = support_files_dir / "sq_wing.run"
            shutil.copy(case_file, support_files_temp_dir)
            case_file_temp = support_files_temp_dir / "sq_wing.run"

            avl_case = AVL_Run_Case(
                geo_files=[geo_path],
                case_file=str(case_file_temp),
                avl_exe_path=str(avl_exe),
                output_dir=force_files_temp_dir
            )
            avl_case.run()

            print(f"Final coefficient calculation {iteration} complete.")

            print(f"\n--- Calculating Aerodynamic Efficiency {iteration} (E = CL/CD) ---")
            force_file_path = force_files_temp_dir / f"force_{airfoil_name}.txt"

            result = extract_force_data(Path(force_file_path))
            if result:
                _, cl, cd = result
                if cd == 0 :
                    efficiency = 0.0
                else:
                    efficiency = cl / cd
                simulated_output = efficiency
            
            logging.info(f"Simulation generated new output: {simulated_output:.4f}")

            # Correction Logic
            # Calculate the percentage error between the original value and the new one
            error = abs((simulated_output - current_max_output) / current_max_output)
            logging.info(f"Comparing with original value. Error: {error:.2%}")

            if error <= 0.02:
                logging.info("Error is within 2% threshold. The dataset's highest value is validated.")
                logging.info("No more corrections needed. Process complete.")

                break # Exit the loop
            # If the error is large, the value is incorrect. We replace it and continue.
            else:
                logging.warning(f"Error is too high. Replacing value at index {max_value_index}.")
                # Replace the Corrected Value 
                df.loc[max_value_index, output_column_name] = simulated_output
                df.to_csv(corrected_csv_path, index=False)
                logging.info(f"Updated {corrected_csv_path}. \n Modified index: {max_value_index}. \nOld: {current_max_output:.4f}, New: {simulated_output:.4f}")
                
                correction_count += 1
                
                # The loop will continue to the next iteration automatically.
        
        else:
            logging.error(f"Process stopped after reaching the maximum of {max_corrections} corrections. The dataset may still contain errors.")
            logging.info(f"The partially corrected data is available at: {corrected_csv_path}, it may be necessary to re-run the script.")

    except FileNotFoundError:
        logging.error(f"Error: The file '{csv_file_path}' was not found.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    finally:
        # Final "fun" metrics
        end_time = time.time()
        total_time = end_time - start_time
        logging.info(f"--- Process finished in {total_time:.2f} seconds. ---")

if __name__ == "__main__":
    database_path = project_root / "Airfoils_for_Regressor_E_Dataset_16_autocorrected.csv"
    
    if os.path.exists(database_path):
        correct_csv_database(database_path)
    else:
        print(f"File not found: '{database_path}'. Please update the path in the script.")