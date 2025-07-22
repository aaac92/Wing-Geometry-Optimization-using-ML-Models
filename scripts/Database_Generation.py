import matplotlib.pyplot as plt
import numpy as np
import os
import contextlib
import sys
import io
import random
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
from simulation_support import XFoilRunner as XR
from simulation_support import Key_Point_Extraction, RecWingGeo_Creation, AVL_Run_Case
from geometric_support import Airfoil_Tools_Exctract, Airfoil_Generator, create_modification_combinations
from file_mainp import File_Manipulation, xfoil_rel_path

# For custom package:
# Get the path to the parent directory (xfoil_project).
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
# Add this path to the list of places Python looks for modules.
sys.path.insert(0, project_root)
# Define folders
airfoils_dir = project_root / "airfoils"
force_files_dir = project_root / "force files"
polars_dir = project_root / "polars"
AI_results_dir = project_root / "AI results"
support_files_dir = project_root / "support_files"
geo_files_dir = project_root / "geometry files"

# === Geometric Generation === #
base_airfoil = os.path.join(support_files_dir,'naca0012_UNIT_0ALFA.csv')
n_cst = 3

# Ranges for values that modify CST Curves (starting_value, end_value, step)
ranges = {
    'first': (-1, 1, 0.25),   # range for value closest to LE
    'middle': (-1, 1, 0.25),  # range for middle points
    'last': (-1, 1, 0.25)     # range for value closest to TE
}
cmb_datalist = create_modification_combinations(n_cst, ranges, ranges)
# Random number for test
random_number = random.randint(0, int(cmb_datalist.shape[0]))
print(cmb_datalist[random_number])

def Gemoetric_Database_Generation(base_airfoil, n_cst, cmb_datalist, random_number):

    airfoil_creator = Airfoil_Generator(
        base_airfoil_csv=base_airfoil,
        n_cst=n_cst,
        cst_coeffs=cmb_datalist[random_number],
        output_dir=airfoils_dir,
        file_name=f"delete_test",
        show_plot=True  # Set to True to see the plot
    )
    # Run the generation process
    airfoil_creator.generate()

    return

if __name__ == "__main__":
    Gemoetric_Database_Generation(base_airfoil, n_cst, cmb_datalist, random_number)