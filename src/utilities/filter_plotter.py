import os
from pathlib import Path

from geometric_support import Airfoil_Generator
from geometric_support.custom_graphs import plot_thickness_filter, plot_LE_slopes, plot_thickness_behavior, plot_camber_filter
from geometric_support.filter_settings import filter_parameters

utilities_dir = Path(__file__).resolve().parent
src_dir = utilities_dir.parent
project_root = src_dir.parent
support_files_dir = project_root / "data" / "support"
single_files_dir = project_root / "data" / "singular_test_data" / "singular_airfoils"

# Filter parameters, as used in the main pipeline (geometric_support/filter_settings.py, flat form for the plots)
geo_filt_parm = filter_parameters()

TF_COLORS = {
    'tf_airfoil': '#484C68',
    'tf_af_fill': '#FCF0DA',
    'tf_max_t': '#D1B8E1',
    'tf_min_t': '#4A4063',
    'tf_x_max': '#3B9574',
    'tf_x_min': '#81C6AA',
    'tf_valid_region': '#EBD7EE',
    'tf_indicator': '#2A4E68',
}

LE_COLORS = {
    'le_upper': '#D1B8E1',
    'le_lower': "#3B9574",
    'le_vector_upper': '#4A4063',
    'le_vector_lower': '#3F6D75',
    'le_vertical_ref': '#000000',
    'le_fill': '#FCF0DA',
    'le_radius': '#2A4E68',
    'le_threshold': '#000000',
}

TB_COLORS = {
    'tb_surface': '#484C68',
    'tb_camber': '#D1B8E1',
    'tb_thickness': '#3B9574',
    'tb_fail': '#FF0000',
    'tb_fill': '#FCF0DA',
}

C_COLORS = {
    'cf_airfoil': '#484C68',
    'cf_af_fill': '#FCF0DA',
    'cf_camber': '#484C68',
    'cf_max_c': '#3B9574',
    'cf_min_c': '#4A4063',
}

if __name__ == "__main__":
    
    #base_airfoil_path = os.path.join(support_files_dir, 'naca653218_unit.csv')
    base_airfoil_path = os.path.join(support_files_dir,'naca2412_unit.csv')
    #base_airfoil_path = os.path.join(single_files_dir, 'naca2412_unit_non-mono.csv')
    # Used only for resolution, n_cst = 3 should be good enough
    n_cst = 5
 
    # Zero CST coefficients modificators means an unmodified base airfoil
    cmb_zero = [0.0] * (2 * n_cst)

    # Used to upscale de airfoil
    airfoil = Airfoil_Generator(
        base_airfoil_csv=base_airfoil_path,
        n_cst=n_cst,
        x_len=150,
        verbose=False
    )

    x, zu, zl = airfoil.generate(
        cst_coeffs=cmb_zero,
        show_plot=False,
        debug=False
    )
 
    plot_thickness_filter(x, zu, zl, geo_filt_parm, colors=TF_COLORS)
    plot_LE_slopes(x, zu, zl, geo_filt_parm, colors=LE_COLORS)
    plot_thickness_behavior(x, zu, zl, colors=TB_COLORS)
    plot_camber_filter(x, zu, zl, geo_filt_parm, colors=C_COLORS)