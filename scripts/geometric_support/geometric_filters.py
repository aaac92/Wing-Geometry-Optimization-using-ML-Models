import numpy as np
from itertools import product

# Filter Function to create combinations of values given a range for critical areas
def create_modification_combinations(n_cst, ranges_upper, ranges_lower, debug=False):
    """
    Creates all valid airfoil modification combinations for upper and lower surfaces.

    Each row in the output array represents one complete airfoil modification,
    with the first n_cst columns for the lower surface and the next n_cst
    columns for the upper surface.

    Parameters:
    - n_cst (int): The number of CST coefficients for a single surface.
    - ranges_upper (dict): Ranges for the upper surface.
                           Keys 'first', 'middle', 'last' map to (start, end, step).
    - ranges_lower (dict): Ranges for the lower surface, same format as above.

    Returns:
    - numpy.ndarray: An array of all valid combinations with shape
                     (total_combinations, n_cst * 2).
    """
    # Helper function to generate combinations for a single surface
    def _get_surface_combos(n, ranges):
        # Generate value arrays for each position type
        start, stop, step = ranges['first']
        vals_first = np.linspace(start, stop, int(round((stop - start) / step)) + 1)
        start, stop, step = ranges['middle']
        vals_middle = np.linspace(start, stop, int(round((stop - start) / step)) + 1)
        start, stop, step = ranges['last']
        vals_last = np.linspace(start, stop, int(round((stop - start) / step)) + 1)

        # Construct a list of value options for each coefficient position
        val_options = [vals_first] + [vals_middle] * (n - 2) + [vals_last]

        # Return an iterator of all combinations for the surface
        return product(*val_options)

    # Generate iterators for upper and lower surface combinations
    combos_upper_iter = _get_surface_combos(n_cst, ranges_upper)
    combos_lower_iter = _get_surface_combos(n_cst, ranges_lower)
    
    # Eagerly evaluate to report counts before final combination
    combos_upper = list(combos_upper_iter)
    combos_lower = list(combos_lower_iter)
    
    if debug:
        print(f"[INFO] Combinations for Upper Surface: {len(combos_upper)}")
        print(f"[INFO] Combinations for Lower Surface: {len(combos_lower)}")
        print("-" * 30)

    # Create all pairs of (upper_combo, lower_combo) and concatenate them
    # This is the most memory-efficient way to build the final large array
    final_combos = np.array([
        np.concatenate(combo_pair)
        for combo_pair in product(combos_lower, combos_upper)
    ])

    # Reporting
    if debug:
        total_combos = final_combos.shape[0]
        print(f"[INFO] Total airfoil combinations generated: {total_combos}")

    return final_combos