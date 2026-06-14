import os
import sys
from pathlib import Path
# Import custom package class and methods
from cst_modeling.foil import FoilGeoFeatures
from geometric_support import Airfoil_Generator, Validate_Airfoil, monotonic_airfoil_thickness, LE_angular_curvature_continuity, plot_thickness_behavior, plot_LE_slopes, create_modification_combinations

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
# Define final directories
airfoils_dir = project_root / "data" / "singular_test_data" / "singular_airfoils"
support_files_dir = project_root / "data" / "support"

if __name__ == "__main__":
    '''
    Use this script to visualize the limits created by the CST coefficients ranges. This helps visualize validity and find all posible airfoil combinations.
    '''
    # Choose a base aifoil to modify
    #base_airfoil_path = os.path.join(support_files_dir,'naca0012_UNIT_0ALFA.csv')
    base_airfoil_path = os.path.join(support_files_dir,'naca653218_unit.csv')
    #base_airfoil_path = os.path.join(support_files_dir,'naca0012_unit.csv')
    
    name = 'Manual Test'
    n_cst = 5
    cmb = [-0.03125, -0.125, -0.125, -0.125, -0.0625, # Lower surface
           -0.03125, -0.125, -0.125, -0.125, -0.0625] # Upper surface
    
    # Ranges are used here for ease of visualization, and later to claculate the possible combinations
    ranges_upper = {
        'LE': (-0.03125, 0.03125, 0.03125),     # range for value closest to LE
        'inter': (-0.125, 0.125, 0.125),        # range for intermediate points
        'TE': (-0.0625, 0.0625, 0.0625)         # range for value closest to TE
    }
    ranges_lower = {
        'LE':  (-0.03125, 0.03125, 0.03125),    # range for value closest to LE
        'inter': (-0.125, 0.125, 0.125),        # range for intermediate points
        'TE': (-0.0625, 0.0625, 0.0625)         # range for value closest to TE
    }

    airfoil = Airfoil_Generator(
            base_airfoil_csv=base_airfoil_path,
            n_cst=n_cst,
            x_len=150,
            verbose=False    
        )
    
    x, yu, yl = airfoil.generate(
            cst_coeffs=cmb,
            show_fancy_plot=True,
            debug=False
            )
    
    foil = FoilGeoFeatures(x, yu, yl)
    va = Validate_Airfoil(x, yu, yl, cmb, n_cst, name)
    LE_AC_Tolerance = 8
    LE_Radius_Threshold = 0.01
    acc_va, _ = va.sharpness_filter(curv_cont=True, tolerance=LE_AC_Tolerance, th_rLE=LE_Radius_Threshold) # Simulation of actual filter
    acc, deviations = LE_angular_curvature_continuity(x, yu, yl) # Values of LE Continuity
    mat_va, _ = va.monotonic_thickness_filter()
    mat, x_dip = monotonic_airfoil_thickness(x, yu, yl)
    t_max, t_x, _ = foil.get_maximum_thickness()
    rLE = foil.get_leading_edge_radius()
    c_max, _ , _ = foil.get_maximum_camber()
    c_min, _ , _ = foil.get_minimum_camber() # Custom function, finds maximum negative camber

    print(f"Maximum thickness: {t_max}")
    print(f"Maximum thickness location: {t_x}")
    print(f"Maximum camber: {c_max}")
    print(f"Minimum camber: {c_min}")
    print(f"Leading edge radius: {rLE}")
    if acc==True and acc_va=='invalid':
        print(f"Sharp angle point, airfoil is: {acc_va}")
        print(f"Leading edge angular change upper: {deviations['deviation_upper']}")
        print(f"Leading edge angular change lower: {deviations['deviation_lower']}")
        print(f"Leading edge angular difference: {deviations['deviation_diff']}")
        print(f"Surface combined angular change (threshold): {deviations['threshold']}")
    elif acc==True or acc_va=='invalid':
        print("Angular continuity mismach, check LE radius parameter or tolerance parameter")
        print(f"LE Radius Threshold: {LE_Radius_Threshold}")
        print(f"LE Angular Continuity Tolerance: {LE_AC_Tolerance}")
        print(f"Actual filter return value: {acc_va}")
        if acc == True:
            print(f"Raw filter return value with default tolerance of 5° is {acc}, therfore the airfoil is INVALID.")
        else:
            print(f"Raw filter return value with default tolerance of 5° is {acc}, therfore the airfoil is VALID.")
    else:
        print(f"Angular continuity pass")
    if mat==True and mat_va=='invalid':
        print(f"Airfoil is not monotonic at {x_dip}, airfoil is: {mat_va}")
    elif mat==True or mat_va=='invalid':
        print("Monotonic filter mismach, wth")
    else:
        print(f"Airfoil thickness is monotonic, pass")

    plot_LE_slopes(x, yu, yl)
    plot_thickness_behavior(x, yu, yl)

    cmb_datalist = create_modification_combinations(n_cst, ranges_upper, ranges_lower, debug=True)

    airfoil._create_dat_file(airfoils_dir, n_cst, x, yu, yl, name)