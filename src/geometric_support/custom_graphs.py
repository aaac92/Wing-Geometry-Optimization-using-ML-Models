import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Arc
from cst_modeling.foil import FoilGeoFeatures
from .custom_filters import compute_true_thickness, LE_angular_curvature_continuity, monotonic_airfoil_thickness

# Shared color palette (hex). Keys are namespaced per function (le_, tb_, tf_)
# so each plot keeps its own established look while pulling from one dict.
# Swap values here; every function merges this with any `colors` override passed in.
COLORS = {
    # plot_LE_slopes
    'le_upper': '#0000FF',
    'le_lower': '#008000',
    'le_vector_upper': '#FF0000',
    'le_vector_lower': '#FFA500',
    'le_vertical_ref': '#000000',
    'le_fill': '#D3D3D3',
    'le_radius': '#9467BD',
    'le_threshold': '#000000',

    # plot_thickness_behavior
    'tb_surface': '#000000',
    'tb_camber': '#808080',
    'tb_thickness': '#0000FF',
    'tb_fail': '#FF0000',
    'tb_fill': '#D3D3D3',
 
    # plot_thickness_filter
    'tf_airfoil': '#000000',
    'tf_af_fill': '#D3D3D3',
    'tf_max_t': '#D62728',
    'tf_min_t': '#FF7F0E',
    'tf_x_max': '#1F77B4',
    'tf_x_min': '#17BECF',
    'tf_valid_region': '#ADD8E6',
    'tf_indicator': '#2CA02C',

    # plot_camber_filter
    'cf_airfoil': '#000000',
    'cf_af_fill': '#D3D3D3',
    'cf_camber': '#808080',
    'cf_max_c': '#D62728',
    'cf_min_c': '#FF7F0E',
}

def plot_LE_slopes(x, zu, zl, geo_filt_parm, n_points=10, colors=None, zoom_margin=0.15):
    """
    Plot slope vectors and angles at the LE for upper and lower surfaces,
    zoomed to the LE region with a padded, square-aspect view. Also draws
    the LE radius circle and the angular deviations used by the sharpness
    filter, with a title reporting the validity of both checks.

    Parameters:
        x, zu, zl     : arrays of x and z coordinates
        geo_filt_parm : filter parameter dict (used here for threshold_rLE)
        n_points      : number of points near LE to plot, also passed to
                        LE_angular_curvature_continuity
        colors        : optional dict overriding entries in the module COLORS dict
        zoom_margin   : fractional padding added around the LE data span
    """
    c = {**COLORS, **(colors or {})}

    _, ax = plt.subplots(figsize=(8, 8))

    x_u, z_u = x[:n_points], zu[:n_points]
    x_l, z_l = x[:n_points], zl[:n_points]

    ax.plot(x, zu, label="Superficie Superior", color=c['le_upper'])
    ax.plot(x, zl, label="Superficie Inferior", color=c['le_lower'])
    ax.fill_between(x, zu, zl, color=c['le_fill'], alpha=0.4)
    ax.scatter(x_u, z_u, color=c['le_upper'])
    ax.scatter(x_l, z_l, color=c['le_lower'])

    # First segments
    def plot_vector(x_start, z_start, vec, color, label=None):
        ax.quiver(x_start, z_start, vec[0], vec[1], angles='xy', scale_units='xy', scale=1,
                  color=color, width=0.005, label=label, zorder=5)

    # Upper and lower vectors
    vec_u = np.array([x_u[1] - x_u[0], z_u[1] - z_u[0]])
    vec_l = np.array([x_l[1] - x_l[0], z_l[1] - z_l[0]])

    # Plot slope vectors
    plot_vector(x_u[0], z_u[0], vec_u, c['le_vector_upper'], 'Vector LE Superior')
    plot_vector(x_l[0], z_l[0], vec_l, c['le_vector_lower'], 'Vector LE Inferior')

    z_top = z_u.max()
    z_bot = z_l.min()
    ax.axvline(0, linestyle='--', color=c['le_vertical_ref'], label='Referencia Vertical')

    # LE radius: circle through the origin and the first point of each surface ---
    Bx, By = x_u[1], z_u[1]
    Cx, Cy = x_l[1], z_l[1]
    D = 2 * (Bx * Cy - Cx * By)
    center_x = ((Bx**2 + By**2) * Cy - (Cx**2 + Cy**2) * By) / D
    center_z = (-(Bx**2 + By**2) * Cx + (Cx**2 + Cy**2) * Bx) / D
    radius = np.hypot(center_x, center_z)  # origin lies on the circle, so this is exact

    ax.add_patch(Circle((center_x, center_z), radius, fill=False, color=c['le_radius'], linewidth=1.5, zorder=3))
    ax.plot(center_x, center_z, 'o', color=c['le_radius'], zorder=5)
    ax.annotate('', xy=(0, 0), xytext=(center_x, center_z),
                arrowprops=dict(arrowstyle='->', color=c['le_radius'], linewidth=1.5), zorder=4)

    # --- Angular deviations from the sharpness filter ---
    has_sharp_LE, deviations = LE_angular_curvature_continuity(x, zu, zl, n_points=n_points)
    dev_u = deviations['deviation_upper']
    dev_l = deviations['deviation_lower']
    dev_diff = deviations['deviation_diff']
    threshold = deviations['threshold']
    curvature_valid = not has_sharp_LE
    radius_valid = radius >= geo_filt_parm['threshold_rLE']

    # Arcs showing each vector's deviation from its vertical reference
    arc_radius = 0.5 * min(np.linalg.norm(vec_u), np.linalg.norm(vec_l))
    angle_vec_u = np.degrees(np.arctan2(vec_u[1], vec_u[0]))
    angle_vec_l = np.degrees(np.arctan2(vec_l[1], vec_l[0]))
    theta1_u, theta2_u = sorted([angle_vec_u, 90])
    theta1_l, theta2_l = sorted([angle_vec_l, -90])

    ax.add_patch(Arc((0, 0), 2 * arc_radius, 2 * arc_radius, angle=0,
                      theta1=theta1_u, theta2=theta2_u, color=c['le_vector_upper'], linewidth=2, zorder=3))
    ax.add_patch(Arc((0, 0), 2 * arc_radius, 2 * arc_radius, angle=0,
                      theta1=theta1_l, theta2=theta2_l, color=c['le_vector_lower'], linewidth=2, zorder=3))

    # Bottom-left readouts, color coded to match their corresponding element
    text_bbox = dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor="#B1B1B1", alpha=1)
    ax.text(0.02, 0.02, f"Radio LE = {radius:.5f}", transform=ax.transAxes,
            color=c['le_radius'], fontsize=9, ha='left', va='bottom', zorder=6, bbox=text_bbox)
    comparator = '<' if curvature_valid else '>'
    ax.text(0.02, 0.09, f"Differencia Des: {dev_diff:.2f}° {comparator} umbral: ({threshold:.2f}°)", transform=ax.transAxes,
            color=c['le_threshold'], fontsize=9, ha='left', va='bottom', zorder=6, bbox=text_bbox)
    ax.text(0.02, 0.16, f"Desviación inferior = {dev_l:.2f}°", transform=ax.transAxes,
            color=c['le_vector_lower'], fontsize=9, ha='left', va='bottom', zorder=6, bbox=text_bbox)
    ax.text(0.02, 0.23, f"Desviación superior = {dev_u:.2f}°", transform=ax.transAxes,
            color=c['le_vector_upper'], fontsize=9, ha='left', va='bottom', zorder=6, bbox=text_bbox)

    # Auto zoom: bound the n_points LE region, pad it, then force a square
    # window (equal x/z span) so the plot reads like a true zoomed-in view.
    x_min = min(x_u.min(), x_l.min(), 0.0)
    x_max = max(x_u.max(), x_l.max())
    z_min = min(z_bot, 0.0)
    z_max = max(z_top, 0.0)

    span = max(x_max - x_min, z_max - z_min)
    span += 2 * span * zoom_margin

    x_mid = (x_max + x_min) / 2
    z_mid = (z_max + z_min) / 2

    ax.set_xlim(x_mid - span / 2, x_mid + span / 2)
    ax.set_ylim(z_mid - span / 2, z_mid + span / 2)
    ax.set_aspect('equal', adjustable='box')

    ax.set_title(f"Filtro de Agudeza (Continuidad de curvatura: {'válida' if curvature_valid else 'inválida'}, "
                 f"Radio de LE: {'válido' if radius_valid else 'inválido'})")
    ax.set_xlabel("x/c")
    ax.set_ylabel("z/c")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

def plot_thickness_behavior(x, zu, zl, colors=None):
    """
    Plot the airfoil profile, camber line, and true thickness distribution,
    flagging a non-monotonicity point if present. Monotonicity is checked
    internally via monotonic_airfoil_thickness, which also drives the
    validity status shown in the title.

    Parameters:
        x, zu, zl   : arrays of x and z coordinates
        colors      : optional dict overriding entries in the module COLORS dict
    """
    c = {**COLORS, **(colors or {})}

    thickness = compute_true_thickness(x, zu, zl)
    zc = 0.5 * (zu + zl)
    has_dip, x_dip = monotonic_airfoil_thickness(x, zu, zl)

    _, ax = plt.subplots(figsize=(10, 6))

    # Plot airfoil profile
    ax.plot(x, zu, label='Perfil Aerodinámico', color=c['tb_surface'], zorder=5)
    ax.plot(x, zl, color=c['tb_surface'], zorder=5)
    ax.fill_between(x, zu, zl, color=c['tb_fill'], alpha=0.4, zorder=1)

    # Plot camber line and thickness curve
    ax.plot(x, zc, label='Línea de Curvatura', color=c['tb_camber'], linestyle=':', linewidth=3, zorder=4)
    ax.plot(x, thickness, label='Espesor Real', color=c['tb_thickness'], linestyle='--', linewidth=2, zorder=4)

    # Axii
    ax.axvline(0, linestyle='--', linewidth=1, color='black', zorder=2)
    ax.axhline(0, linestyle='--', linewidth=1, color='black', zorder=2)

    if x_dip is not None:
        # Highlight failure point
        fail_thickness = np.interp(x_dip, x, thickness)
        ax.axvline(x_dip, color=c['tb_fail'], linestyle='--', label=f'No monotonicidad en x={x_dip:.3f}')
        ax.scatter(x_dip, fail_thickness, color=c['tb_fail'], zorder=4)

    else:
        ax.legend(["Perfil Aerodinámico", "Línea de Curvatura", "Espesor Real", "El espesor del perfil es monótono"])

    ax.set_xlabel('x/c')
    ax.set_ylabel('z/c - Espesor Real')
    status = 'válido' if not has_dip else 'inválido'
    ax.set_title(f"Filtro de Espesor Monótono — {status}")
    ax.grid(True)
    ax.legend()
    ax.set_ylim(-0.25, 0.25)
    ax.set_xlim(-0.05, 1.05)
    plt.tight_layout()
    plt.show()

def plot_thickness_filter(x, zu, zl, geo_filt_parm, colors=None):
    """
    Visualize the thickness magnitude/location filters against the airfoil
    profile. Thresholds are centered on the camber line. t_max/t_x are
    computed internally via FoilGeoFeatures.
 
    Parameters:
        x, zu, zl     : arrays of x and z coordinates
        geo_filt_parm : filter parameter dict (threshold_max_t, threshold_min_t,
                        threshold_t_x_max, threshold_t_x_min, ...)
        colors        : optional dict overriding entries in the module COLORS dict
    """
    c = {**COLORS, **(colors or {})}
 
    foil = FoilGeoFeatures(x, zu, zl)
    t_max, t_x, _ = foil.get_maximum_thickness()  # third return value unused for this filter
 
    # Camber line — thickness thresholds are centered on this, not on z=0
    zc = 0.5 * (zu + zl)
 
    # Surface positions at t_x, for the thickness annotation
    zu_at_tx = np.interp(t_x, x, zu)
    zl_at_tx = np.interp(t_x, x, zl)
 
    # Pass/fail, mirrors Validate_Airfoil.thickness_filters bounds logic
    t_pass = geo_filt_parm['threshold_min_t'] <= t_max <= geo_filt_parm['threshold_max_t']
    x_pass = geo_filt_parm['threshold_t_x_min'] <= t_x <= geo_filt_parm['threshold_t_x_max']
    overall_pass = t_pass and x_pass
 
    _, ax = plt.subplots(figsize=(10, 6))
 
    ax.plot(x, zu, color=c['tf_airfoil'], linewidth=1.5, label='Perfil Aerodinámico', zorder=4)
    ax.plot(x, zl, color=c['tf_airfoil'], linewidth=1.5, zorder=4)
    ax.fill_between(x, zu, zl, color=c['tf_af_fill'], alpha=0.4, zorder=1)
 
    # Thickness magnitude thresholds, centered on the camber line zc, plotted as ranges (+/-2)
    # Max T
    ax.plot(x, zc + geo_filt_parm['threshold_max_t']/2, color=c['tf_max_t'], linestyle='--', linewidth=2,
            label=f"threshold_max_t = {geo_filt_parm['threshold_max_t']}", zorder=3)
    ax.plot(x, zc - geo_filt_parm['threshold_max_t']/2, color=c['tf_max_t'], linestyle='--', linewidth=2, zorder=3)
    # Min T
    ax.plot(x, zc + geo_filt_parm['threshold_min_t']/2, color=c['tf_min_t'], linestyle='--', linewidth=2,
            label=f"threshold_min_t = {geo_filt_parm['threshold_min_t']}", zorder=3)
    ax.plot(x, zc - geo_filt_parm['threshold_min_t']/2, color=c['tf_min_t'], linestyle='--', linewidth=2, zorder=3)
 
    # Thickness location thresholds (vertical, independent of camber)
    ax.axvline(geo_filt_parm['threshold_t_x_max'], color=c['tf_x_max'], linestyle=':', linewidth=1,
               label=f"threshold_t_x_max = {geo_filt_parm['threshold_t_x_max']}", zorder=3)
    ax.axvline(geo_filt_parm['threshold_t_x_min'], color=c['tf_x_min'], linestyle=':', linewidth=1,
               label=f"threshold_t_x_min = {geo_filt_parm['threshold_t_x_min']}", zorder=3)
    
    # Axii
    xmin, xmax = ax.get_xlim()
    ax.axvline(0, linestyle='--', color='black')
    ax.plot([xmin, 0], [0, 0], color='k', linestyle='--', linewidth=1, zorder=1)
    ax.plot([1, xmax], [0, 0], color='k', linestyle='--', linewidth=1, zorder=1)
 
    # Valid regions: bound by threshold_t_x_min/max (x) and the camber-offset thickness range (y)
    x_lo = geo_filt_parm['threshold_t_x_min']
    x_hi = geo_filt_parm['threshold_t_x_max']
    mask = (x >= x_lo) & (x <= x_hi)
 
    ax.fill_between(x[mask], zc[mask] + geo_filt_parm['threshold_min_t']/2, zc[mask] + geo_filt_parm['threshold_max_t']/2,
                     facecolor=c['tf_valid_region'], alpha=0.4, edgecolor='none',
                     label='Región válida', zorder=2)
    ax.fill_between(x[mask], zc[mask] - geo_filt_parm['threshold_max_t']/2, zc[mask] - geo_filt_parm['threshold_min_t']/2,
                     facecolor=c['tf_valid_region'], alpha=0.4, edgecolor='none', zorder=2)
 
    # Extracted max-thickness value, drawn as a dimension arrow at x = t_x
    ax.annotate(
        '', xy=(t_x, zu_at_tx), xytext=(t_x, zl_at_tx),
        arrowprops=dict(arrowstyle='<->', color=c['tf_indicator'], linewidth=1.5), zorder=5
    )
    # t_max / t_x readout, bottom-right corner, axes-relative coordinates
    ax.text(0.98, 0.02, f"t_max = {t_max:.4f}\nt_x = {t_x:.4f}",
            transform=ax.transAxes, color=c['tf_indicator'], fontsize=9,
            ha='right', va='bottom', zorder=5)
 
    ax.set_xlabel('x/c')
    ax.set_ylabel('z/c')
    status = "valido" if overall_pass else "inválida"
    ax.set_title(f"Filtro de espesor — {status} "
                 f"(magnitud: {'valida' if t_pass else 'inválida'}, posición: {'valida' if x_pass else 'inválida'})")
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.25, 0.25)
    ax.set_xlim(-0.05, 1.05)
 
    plt.tight_layout()
    plt.show()

def plot_camber_filter(x, zu, zl, geo_filt_parm, colors=None):
    """
    Visualize the camber magnitude filters against the airfoil profile and
    camber line. c_max/c_min are computed internally via FoilGeoFeatures.

    Parameters:
        x, zu, zl     : arrays of x and z coordinates
        geo_filt_parm : filter parameter dict (threshold_max_camber, threshold_min_camber)
        colors        : optional dict overriding entries in the module COLORS dict
    """
    c = {**COLORS, **(colors or {})}

    foil = FoilGeoFeatures(x, zu, zl)
    c_max, x_c_max, _ = foil.get_maximum_camber()
    c_min, _, _ = foil.get_minimum_camber()

    zc = 0.5 * (zu + zl)

    # Pass/fail, mirrors Validate_Airfoil.camber_filters bounds logic
    max_pass = c_max <= geo_filt_parm['threshold_max_camber']
    min_pass = c_min >= geo_filt_parm['threshold_min_camber']
    overall_pass = max_pass and min_pass

    _, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x, zu, color=c['cf_airfoil'], linewidth=1.5, label='Perfil Aerodinámico', zorder=4)
    ax.plot(x, zl, color=c['cf_airfoil'], linewidth=1.5, zorder=4)
    ax.fill_between(x, zu, zl, color=c['cf_af_fill'], alpha=0.4, zorder=1)

    ax.plot(x, zc, color=c['cf_camber'], linestyle=':', linewidth=3, label='Línea de Curvatura', zorder=4)
    ax.plot(x_c_max, c_max, 'o', color=c['cf_max_c'], zorder=5)

    # Camber magnitude thresholds
    ax.axhline(geo_filt_parm['threshold_max_camber'], color=c['cf_max_c'], linestyle='--', linewidth=2,
               label=f"threshold_max_camber = {geo_filt_parm['threshold_max_camber']}", zorder=3)
    ax.axhline(geo_filt_parm['threshold_min_camber'], color=c['cf_min_c'], linestyle='--', linewidth=2,
               label=f"threshold_min_camber = {geo_filt_parm['threshold_min_camber']}", zorder=3)

    # Axii
    ax.axvline(0, linestyle='--', linewidth=1, color='black', zorder=2)
    ax.axhline(0, linestyle='--', linewidth=1, color='black', zorder=2)

    # Extracted camber values, bottom-right, color coded to their threshold
    ax.text(0.98, 0.02, f"c_min = {c_min:.4f}", transform=ax.transAxes,
            color=c['cf_min_c'], fontsize=9, ha='right', va='bottom', zorder=5)
    ax.text(0.98, 0.08, f"c_max = {c_max:.4f}", transform=ax.transAxes,
            color=c['cf_max_c'], fontsize=9, ha='right', va='bottom', zorder=5)

    ax.set_xlabel('x/c')
    ax.set_ylabel('z/c')
    status = "válido" if overall_pass else "inválido"
    ax.set_title(f"Filtro de Curvatura — {status} "
                 f"(máximo: {'válido' if max_pass else 'inválido'}, mínimo: {'válido' if min_pass else 'inválido'})")
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.25, 0.25)
    ax.set_xlim(-0.05, 1.05)

    plt.tight_layout()
    plt.show()