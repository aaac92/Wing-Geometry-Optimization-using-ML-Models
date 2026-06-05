import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
# Import custom package class and methods
from geometric_support import Airfoil_Tools_Exctract
from cst_modeling.section import cst_foil_fit, cst_foil

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
    #base_airfoil_path = os.path.join(support_files_dir,'naca0012_UNIT_0ALFA.csv')
    #base_airfoil_path = os.path.join(support_files_dir,'naca653218_unit.csv')
    base_airfoil_path = os.path.join(support_files_dir,'naca0012_unit.csv')

    n_cst = 5

    ate = Airfoil_Tools_Exctract()
    ate.file = base_airfoil_path
    ate.airfoil_coords(verbose=False)
    airfoil_base_name = ate.name
    chord = ate.C
    #ate.ate_verif()
    #ate.ate_plt()
    x_upper_norm = np.array(ate.X_UPPER) / chord
    z_upper_norm = np.array(ate.Z_UPPER) / chord
    x_lower_norm = np.array(ate.X_LOWER) / chord
    z_lower_norm = np.array(ate.Z_LOWER) / chord
    # Fit original airfoil with CST
    x_len = 150
    x_values = np.linspace(0, 1, x_len)
    z_upper_flip = np.flip(z_upper_norm)
    x_upper_flip = np.flip(x_upper_norm)
    #a_gain_u, a_gain_l = cst_foil_fit(x_upper_norm, z_upper_flip * (-1), x_lower_norm, z_lower_norm, n_cst)
    a_gain_u, a_gain_l = cst_foil_fit(x_upper_flip, z_upper_flip, x_lower_norm, z_lower_norm, n_cst)
    # Reconstruct the airfoil from CST coefficients
    x_cst, z_cst_u, z_cst_l, _, _ = cst_foil(x_len, a_gain_u, a_gain_l, x=x_values)

    plt.figure(figsize=(15, 6))
    plt.plot(x_cst, z_cst_u, label='CST Airfoil', color='gray', linestyle='--')
    plt.plot(x_cst, z_cst_l, color='gray', linestyle='--')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title('Airfoil Visualization')
    plt.xlabel('x/c')
    plt.ylabel('z/c')
    plt.show()