import os
import sys
from pathlib import Path
# Import custom package class and methods
from geometric_support import Airfoil_Generator

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
    cmb = [-0.125, 0.125, 0.125, 0.125, 0.125, # Lower surface
           0.125, 0.125, 0.125, 0.125, 0.125] # Upper surface
    
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