import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import subprocess
import shutil
import json
import threading
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from cst_modeling.foil import check_validity
from multiprocessing import Pool, Manager, cpu_count
# Import custom package class and methods
from simulation_support import Xfoil_Polar_Creator as XPR
from simulation_support import Key_Point_Extraction, RecWingGeo_Creation, AVL_Run_Case
from geometric_support import Airfoil_Generator, Validate_Airfoil, create_modification_combinations
from file_handling_support import compression_Alma, Training_Data_Generator, get_all_files_from
from ux_support import Progress_GUI, recovery_json, recovery_Fabius

# For custom package:
# Get the path to the parent directory (xfoil_project).
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
# Add this path to the list of places Python looks for modules.
sys.path.insert(0, project_root)
# Define final directories
airfoils_dir = project_root / "manual_airfoils"
support_files_dir = project_root / "support_files"

if __name__ == "__main__":
    base_airfoil_path = os.path.join(support_files_dir,'naca0012_UNIT_0ALFA.csv')
    n_cst = 5
    cmb = [-0.125, 0.375, 0.375, 0.375, 0.125, # Lower surface
           0.375, 0.375, 0.375, 0.375, 0.125] # Upper surface
    
    airfoil = Airfoil_Generator(
            base_airfoil_csv=base_airfoil_path,
            n_cst=n_cst,
            verbose=False    
        )
    
    airfoil.generate(
            cst_coeffs=cmb,
            show_fancy_plot=True,
            debug=False
            )