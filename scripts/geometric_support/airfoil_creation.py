import matplotlib.pyplot as plt
import numpy as np
import os
from cst_modeling.section import cst_foil_fit, cst_foil
from cst_modeling.foil import FoilModification, check_validity
from .Airfoil_Data_Extract import Airfoil_Tools_Exctract

class Airfoil_Generator:
    """
    A class to generate modified airfoils using the CST method.

    This class takes a base airfoil from a .csv file, applies CST parameterization,
    modifies it with given coefficients, and saves the new airfoil to a .dat file.
    It can also optionally display a plot of the original and modified airfoils.

    Args:
        base_airfoil_csv (str): The file path for the base airfoil .csv file.
        n_cst (int): The order of the CST parameterization.
        cst_coeffs (list or np.ndarray): A list of 6 coefficients for modification.
                                         [c_l1, c_l2, c_l3, c_u1, c_u2, c_u3]
        output_dir (str): The directory where the output .dat file will be saved.
        file_name (str): The base name for the output file.
        x_len (int, optional): The number of points for the airfoil coordinates. Defaults to 150.
        show_plot (bool, optional): If True, displays a plot of the airfoils. Defaults to False.
    """
    def __init__(self, base_airfoil_csv, n_cst, cst_coeffs, output_dir, file_name, x_len=150, show_plot=False):
        if not isinstance(cst_coeffs, (list, np.ndarray)) or len(cst_coeffs) != n_cst*2:
            raise ValueError(f"cst_coeffs must be a list or numpy array of {n_cst*2} elements.")

        self.base_airfoil_csv = base_airfoil_csv
        self.n_cst = n_cst
        self.cmb_l = np.array(cst_coeffs[:n_cst])
        self.cmb_u = np.array(cst_coeffs[n_cst:])
        self.output_dir = output_dir
        self.file_name = file_name
        self.x_len = x_len
        self.show_plot = show_plot

    def generate(self):
        """
        Runs the full airfoil generation process.
        """
        # 1. Extract coordinates from the base airfoil file
        ate = Airfoil_Tools_Exctract()
        ate.file = self.base_airfoil_csv
        ate.airfoil_coords()

        # 2. Normalize coordinates
        chord = ate.C
        x_upper_norm = np.array(ate.X_UPPER) / chord
        z_upper_norm = np.array(ate.Z_UPPER) / chord
        x_lower_norm = np.array(ate.X_LOWER) / chord
        z_lower_norm = np.array(ate.Z_LOWER) / chord

        # 3. Fit original airfoil with CST
        x_values = np.linspace(0, 1, self.x_len)
        z_upper_flip = np.flip(z_upper_norm)
        a_gain_u, a_gain_l = cst_foil_fit(x_upper_norm, z_upper_flip * (-1), x_lower_norm, z_lower_norm, self.n_cst)
        
        # 4. Reconstruct the airfoil from CST coefficients
        x_cst, z_cst_u, z_cst_l, _, _ = cst_foil(self.x_len, a_gain_u, a_gain_l, x=x_values)
        
        # 5. Modify the CST-reconstructed airfoil
        if not check_validity(x_cst, z_cst_u, z_cst_l):
            print("Warning: The CST-reconstructed airfoil may not be valid.")
        
        mod_airfoil = FoilModification(x_cst, z_cst_u, z_cst_l)
        mod_airfoil.add_cst_incremental_curves(self.cmb_u, self.cmb_l, True)
        
        x_mod = mod_airfoil.x
        yu_mod = mod_airfoil.yu
        yl_mod = mod_airfoil.yl
        
        if not check_validity(x_mod, yu_mod, yl_mod):
            print("Warning: The modified airfoil may not be valid.")

        # 6. Save the modified airfoil to a file
        self._create_dat_file(x_mod, yu_mod, yl_mod, ate.name)

        # 7. Optionally display a plot
        if self.show_plot:
            self._plot_airfoils(x_cst, z_cst_u, z_cst_l, x_mod, yu_mod, yl_mod)

    def _create_dat_file(self, x, yu, yl, base_name):
        """Helper method to write the airfoil coordinates to a .dat file."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Construct the full filename with coefficients
        coeff_str = f"[{self.cmb_l[0]:.4f}, {self.cmb_l[1]:.4f}, {self.cmb_l[2]:.4f}, {self.cmb_u[0]:.4f}, {self.cmb_u[1]:.4f}, {self.cmb_u[2]:.4f}]"
        full_file_name = f"{self.file_name}.dat"
        file_path = os.path.join(self.output_dir, full_file_name)

        with open(file_path, "w") as f:
            f.write(f"{self.file_name} based on {base_name} using coefficients: {coeff_str}\n")
            # Write upper surface from trailing edge to leading edge
            for xi, yi in zip(x[::-1], yu[::-1]):
                f.write(f"{xi:.6f} {yi:.6f}\n")
            # Write lower surface from leading edge to trailing edge
            # Skip the first point to avoid duplicating the leading edge
            for xi, yi in zip(x[1:], yl[1:]):
                f.write(f"{xi:.6f} {yi:.6f}\n")
        print(f"Successfully created airfoil file: {file_path}")

    def _plot_airfoils(self, x_cst, z_cst_u, z_cst_l, x_mod, yu_mod, yl_mod):
        """Helper method to visualize the airfoils."""
        plt.figure(figsize=(15, 6))
        plt.plot(x_cst, z_cst_u, label='CST Airfoil', color='red', linestyle='--')
        plt.plot(x_cst, z_cst_l, color='red', linestyle='--')
        plt.plot(x_mod, yu_mod, label='Modified Airfoil', color='green')
        plt.plot(x_mod, yl_mod, color='green')
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.title('Airfoil Visualization')
        plt.xlabel('x/c')
        plt.ylabel('z/c')
        plt.show()