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
    It can also optionally display a plot of the original and modified airfoils (drawn by custom_graphs.py).

    Args:
        base_airfoil_csv (str): The file path for the base airfoil .csv file.
        n_cst (int): The order of the CST parameterization.
        cst_coeffs (list or np.ndarray): A list of 6 coefficients for modification.
                                         [c_l1, c_l2, c_l3, c_u1, c_u2, c_u3]
        output_dir (str): The directory where the output .dat file will be saved.
        file_name (str): The base name for the output file.
        x_len (int, optional): The number of points for the airfoil coordinates. Defaults to 150.
        show_plot (bool, optional): If True, displays a plot of the airfoils. Defaults to False.
        verbose(bool, optional): True to toggle debug and error/warning messages.
    """
    def __init__(self, base_airfoil_csv, n_cst, x_len=150, a0=0.0079, a1=0.96, verbose=False):

        self.base_airfoil_csv = base_airfoil_csv
        self.x_len = x_len
        self.n_cst = n_cst
        self.verbose = verbose
        self.a0 = a0
        self.a1 = a1
        # Extract coordinates from the base airfoil file
        ate = Airfoil_Tools_Exctract()
        ate.file = self.base_airfoil_csv
        ate.airfoil_coords(verbose=self.verbose)
        self.airfoil_base_name = ate.name
        # Normalize coordinates
        chord = ate.C
        x_upper_norm = np.array(ate.X_UPPER) / chord
        z_upper_norm = np.array(ate.Z_UPPER) / chord
        x_lower_norm = np.array(ate.X_LOWER) / chord
        z_lower_norm = np.array(ate.Z_LOWER) / chord
        # Fit original airfoil with CST
        x_values = np.linspace(0, 1, self.x_len)
        z_upper_flip = np.flip(z_upper_norm)
        x_upper_flip = np.flip(x_upper_norm)
        a_gain_u, a_gain_l = cst_foil_fit(x_upper_flip, z_upper_flip, x_lower_norm, z_lower_norm, n_cst)
        # Reconstruct the airfoil from CST coefficients
        self.x_cst, self.z_cst_u, self.z_cst_l, _, _ = cst_foil(self.x_len, a_gain_u, a_gain_l, x=x_values, a0=self.a0, a1=self.a1)

    def generate(self, cst_coeffs, show_plot=False, show_fancy_plot = False, debug=False):
        """
        Runs the full airfoil generation process.
        """
        # Check for correct format
        if not isinstance(cst_coeffs, (list, np.ndarray)) or len(cst_coeffs) != self.n_cst*2:
            raise ValueError(f"cst_coeffs must be a list or numpy array of {self.n_cst*2} elements.")
        # Create modification object from CST-reconstructed airfoil for higher resolution
        if not check_validity(self.x_cst, self.z_cst_u, self.z_cst_l, verbose=self.verbose):
            print("Warning: The CST-reconstructed airfoil may not be valid.")
        if debug:
            print('[DEBUG] Creating new modification object...')
        mod_airfoil = FoilModification(self.x_cst, self.z_cst_u, self.z_cst_l, verbose=self.verbose)
        if debug:
            print('[DEBUG] New modification object created!!')
        
        #output_dir = output_dir
        #file_name = file_name
        self.cmb_l = np.array(cst_coeffs[:self.n_cst])
        self.cmb_u = np.array(cst_coeffs[self.n_cst:])
        if debug:
            print(f"[DEBUG] Using cmb_l: {self.cmb_l} and cmb_u: {self.cmb_u}")

        mod_airfoil.add_cst_incremental_curves(self.cmb_u, self.cmb_l)
        
        x_mod = mod_airfoil.x
        yu_mod = mod_airfoil.yu
        yl_mod = mod_airfoil.yl

        # Optionally display a plot
        if show_plot:
            self._plot_airfoils(x_mod, yu_mod, yl_mod)

        if show_fancy_plot:
            self._fancy_plot_airfoils(x_mod, yu_mod, yl_mod)

        return x_mod, yu_mod, yl_mod

    def _create_dat_file(self, output_dir, n_cst, x, yu, yl, file_name, debug=False):
        """Helper method to write the airfoil coordinates to a .dat file."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Construct the full filename with coefficients
        coeff_str = "[" + ", ".join(f"{coeff:.4f}" for coeff in list(self.cmb_l) + list(self.cmb_u)) + "]"
        full_file_name = f"{file_name}.dat"
        file_path = os.path.join(output_dir, full_file_name)

        with open(file_path, "w") as f:
            f.write(f"{file_name} based on {self.airfoil_base_name} using coefficients: {coeff_str}\n")
            # Write upper surface from trailing edge to leading edge
            for xi, yi in zip(x[::-1], yu[::-1]):
                f.write(f"{xi:.6f} {yi:.6f}\n")
            # Write lower surface from leading edge to trailing edge
            # Skip the first point to avoid duplicating the leading edge
            for xi, yi in zip(x[1:], yl[1:]):
                f.write(f"{xi:.6f} {yi:.6f}\n")
        if debug:
            print(f"Successfully created airfoil file: {file_path}")

    def _plot_airfoils(self, x_mod, yu_mod, yl_mod):
        """Base (CST reconstruction) and modified airfoil; drawn by custom_graphs.plot_cst_comparison."""
        from .custom_graphs import plot_cst_comparison  # imported here: geometric_support loads this module first
        plot_cst_comparison(self.x_cst, self.z_cst_u, self.z_cst_l, x_mod, yu_mod, yl_mod)

    def _fancy_plot_airfoils(self, x_mod, yu_mod, yl_mod):
        """Same comparison with the modified airfoil smoothed for exports; drawn by custom_graphs.plot_cst_comparison_smooth."""
        from .custom_graphs import plot_cst_comparison_smooth
        plot_cst_comparison_smooth(self.x_cst, self.z_cst_u, self.z_cst_l, x_mod, yu_mod, yl_mod)