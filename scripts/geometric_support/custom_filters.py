import numpy as np
from itertools import product
from cst_modeling.foil import FoilGeoFeatures, check_validity

def LE_angular_curvature_continuity(x, zu, zl, n_points=10):
    """
    Check if the LE has a sharp slope deviation relative to the average slope changes.

    Parameters:
        x        : array of x-coordinates
        zu, zl   : arrays of upper and lower surface z-coordinates
        n_points : number of points near LE to evaluate

    Returns:
        has_sharp_LE (bool): True if sharp angle is detected
        deviations (dict)  : contains deviation from vertical and average angle change
    """
    # Use first n_points
    x_u, z_u = x[:n_points], zu[:n_points]
    x_l, z_l = x[:n_points], zl[:n_points]

    def compute_slopes_and_angles(x_arr, z_arr):
        vecs = np.diff(np.stack((x_arr, z_arr), axis=1), axis=0)
        angles = []
        for i in range(1, len(vecs)):
            v1 = vecs[i - 1]
            v2 = vecs[i]
            dot = np.dot(v1, v2)
            norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
            angle = np.arccos(np.clip(dot / norm_product, -1.0, 1.0))
            angles.append(np.degrees(angle))
        return vecs[0], np.mean(angles) if angles else 0

    # Get first vector and average angle change
    first_vec_u, avg_angle_u = compute_slopes_and_angles(x_u, z_u)
    first_vec_l, avg_angle_l = compute_slopes_and_angles(x_l, z_l)

    threshold = np.mean([avg_angle_l, avg_angle_u])

    vertical = np.array([0, 1])

    def deviation_from_vertical(vec):
        dot = np.dot(vec, vertical)
        norm_product = np.linalg.norm(vec) * np.linalg.norm(vertical)
        angle = np.arccos(np.clip(dot / norm_product, -1.0, 1.0))
        return np.degrees(angle)

    dev_u = deviation_from_vertical(first_vec_u)
    dev_l = abs(deviation_from_vertical(first_vec_l) - 180)

    dev_diff = abs(dev_l-dev_u)

    has_sharp_LE = dev_diff > threshold

    deviations = {
        "deviation_upper": dev_u,
        "deviation_lower": dev_l,
        "deviation_diff": dev_diff,
        "average_angle_upper": avg_angle_u,
        "average_angle_lower": avg_angle_l,
        "threshold": threshold
    }

    return has_sharp_LE, deviations

def compute_true_thickness(x, zu, zl):
    """
    Computes the ehicknes along the camber line of the airfoil

    Parameters:
        x                : x-coordinates
        zu               : upper surface z
        zl               : lower surface z
    
    Returns:
        thickness
    """
    # Camber line
    zc = 0.5 * (zu + zl)
    
    # Derivative of camber line (dz/dx) using central differences
    dzc_dx = np.gradient(zc, x)

    # Unit normal vector components
    norm_x = -dzc_dx
    norm_z = np.ones_like(dzc_dx)
    norm_mag = np.sqrt(norm_x**2 + norm_z**2)
    norm_x /= norm_mag
    norm_z /= norm_mag

    # Vector between upper and lower surface points
    dx_ul = 0  # x is shared, so dx = 0
    dz_ul = zu - zl  # vertical distance

    # Project difference vector onto normal
    # Since dx_ul = 0, dot product simplifies:
    thickness = dz_ul * norm_z

    return thickness

def monotonic_airfoil_thickness(x, zu, zl):
    """
    Detects an unnatural internal dip in thickness.

    Parameters:
        x                : x-coordinates
        zu               : upper surface z
        zl               : lower surface z

    Returns:
        has_dip (bool), x_dip (float or None)
    """
    thickness = compute_true_thickness(x, zu, zl)
    max_idx = np.argmax(thickness)

    # Check for second increase after max thickness
    decreasing_part = thickness[max_idx:]
    diffs = np.diff(decreasing_part)
    fail_indices = np.where(diffs > 0)[0]

    if fail_indices.size > 0:
        fail_idx = max_idx + fail_indices[0]
        return True, x[fail_idx]
    
    return False, None