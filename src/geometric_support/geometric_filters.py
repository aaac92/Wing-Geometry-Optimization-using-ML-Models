import numpy as np
from itertools import product
from cst_modeling.foil import FoilGeoFeatures, check_validity
from .custom_filters import LE_angular_curvature_continuity, monotonic_airfoil_thickness

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
        start, stop, step = ranges['LE']
        vals_le = np.linspace(start, stop, int(round((stop - start) / step)) + 1)
        start, stop, step = ranges['inter']
        vals_inter = np.linspace(start, stop, int(round((stop - start) / step)) + 1)
        start, stop, step = ranges['TE']
        vals_te = np.linspace(start, stop, int(round((stop - start) / step)) + 1)

        # Construct a list of value options for each coefficient position
        val_options = [vals_le] + [vals_inter] * (n - 2) + [vals_te]

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
    final_combos = np.array([
        np.concatenate(combo_pair)
        for combo_pair in product(combos_lower, combos_upper)
    ])

    # Reporting
    if debug:
        total_combos = final_combos.shape[0]
        print(f"[INFO] Total airfoil combinations generated: {total_combos}")

    return final_combos

class Validate_Airfoil:
    '''
    Master object to apply filters for a particular airfoil, given it's coordinates.

    Parameters
    ----------
    x : np.ndarray
        list of x coordinates, from 0 to 1
    
    yu : np.array
        corresponding y upper surface coordinates
    
    yl : np.array
        corresponding y lower surface coordinates
    
    cst_coeffs : list, np.ndarray
        A list of 6 coefficients for modification.
            [c_l1, c_l2, c_l3, c_u1, c_u2, c_u3]
    
    n_cst : int
        Order of CST coefficients for airfoil

    name : str
            Name of the airfoil, used for logging purposes.
    
    verbose : bool; optional
        True to toggle debug and error/warning messages.
    '''
    def __init__(self, x, yu, yl, cst_coeffs, n_cst, name, verbose=False):

        self.n_cst = n_cst
        # Check for correct format
        if not isinstance(cst_coeffs, (list, np.ndarray)) or len(cst_coeffs) != self.n_cst*2:
            raise ValueError(f"cst_coeffs must be a list or numpy array of {self.n_cst*2} elements.")
        
        self.cmb_l = np.array(cst_coeffs[:self.n_cst])
        self.cmb_u = np.array(cst_coeffs[self.n_cst:])
        
        self.x = x
        self.yu = yu
        self.yl = yl
        self.name = name
        self.verbose = verbose

        try:
            self.foil = FoilGeoFeatures(x, yu, yl, verbose=self.verbose)
        except Exception as e:
            raise ValueError(f"{self.name}.dat: cmb_l = {self.cmb_l}, cmb_u = {self.cmb_u} | Reason: feature extraction failed ({str(e)})\n")
        
    def thickness_filters(self, th_t_min=None, th_t_max=None, th_t_x_max=None, th_t_x_min=None):
        '''
        Activates and applies thickness filters, if a reason is found airfoil should be rejected. If any thershold is empty or None, then that 
        filter is deactivated.

        Parameters:
        -------------
        th_t_min : float
            Minimum thickness threshold value, test real airfoils for reference values.

        th_t_x : float
            Furthest permisible position to find the airfoil's maximum thickness from 0 to 1.
        '''
        if th_t_min==None and th_t_max==None and th_t_x_max==None and th_t_x_min==None:
            if self.verbose:
                print(f"[INFO] Thresholds are empty, no filters applied, returning valid status.")
            return ("valid", None)

        t_max, t_x, _ = self.foil.get_maximum_thickness()

        reason = None
        if th_t_min:
            if t_max < th_t_min:
                reason = f"Airfoil total thickness under threshold: {th_t_min}"
                log_line = f"{self.name}.dat: cmb_l = {self.cmb_l}, cmb_u = {self.cmb_u} | Reason: {reason}\n"
                return ("invalid", log_line)
        else:
            if self.verbose:
                print(f"[INFO] Minimum thickness threshold is {th_t_min} filter is inactive")
        if th_t_max:
            if t_max > th_t_max:
                reason = f"Airfoil total thickness over threshold: {th_t_max}"
                log_line = f"{self.name}.dat: cmb_l = {self.cmb_l}, cmb_u = {self.cmb_u} | Reason: {reason}\n"
                return ("invalid", log_line)
        else:
            if self.verbose:
                print(f"[INFO] Maximum thickness threshold is {th_t_min} filter is inactive")

        if th_t_x_max:
            if t_x > th_t_x_max:
                reason = f"Maximum thickness location found further back than {th_t_x_max}"
                log_line = f"{self.name}.dat: cmb_l = {self.cmb_l}, cmb_u = {self.cmb_u} | Reason: {reason}\n"
                return ("invalid", log_line)
        else:
            if self.verbose:
                print(f"[INFO] Maximum thickness location back threshold is {th_t_x_max} filter is inactive")
        if th_t_x_min:
            if t_x < th_t_x_min:
                reason = f"Maximum thickness location found further front than {th_t_x_min}"
                log_line = f"{self.name}.dat: cmb_l = {self.cmb_l}, cmb_u = {self.cmb_u} | Reason: {reason}\n"
                return ("invalid", log_line)
        else:
            if self.verbose:
                print(f"[INFO] Maximum thickness location front threshold is {th_t_x_min} filter is inactive")
        

        return ("valid", None)
    
    def sharpness_filter(self, curv_cont=False, tolerance=5, th_rLE=None):
        '''
        Activates and applies sharpness filters, if a reason is found airfoil should be rejected. If any thershold is empty or None, then that 
        filter is deactivated.

        Parameters:
        ------------
        curv_cont : bool
            Activates the curvature continuity filter for the airfoil.

        th_rLE : float
            Minimum threshold value for the radius of the LE, test real airfoils for reference values.
        '''
        if curv_cont==False and th_rLE==None:
            if self.verbose:
                print(f"[INFO] Thresholds are empty, no filters applied, returning valid status.")
            return ("valid", None)

        sharp_point, _ = LE_angular_curvature_continuity(self.x, self.yu, self.yl, tolerance=tolerance)
        rLE = self.foil.get_leading_edge_radius()

        if curv_cont:
            if sharp_point:
                reason = "Angular curvature continuity failed"
                log_line = f"{self.name}.dat: cmb_l = {self.cmb_l}, cmb_u = {self.cmb_u} | Reason: {reason}\n"
                return ("invalid", log_line)
        else:
            if self.verbose:
                print(f"[INFO] Curvature continuity toggle is {curv_cont}, filter is inactive.")

        if th_rLE:
            if rLE < th_rLE:
                reason = f"LE radius under threshold: {th_rLE}"
                log_line = f"{self.name}.dat: cmb_l = {self.cmb_l}, cmb_u = {self.cmb_u} | Reason: {reason}\n"
                return ("invalid", log_line)
        else:
            if self.verbose:
                print(f"[INFO] Threshold for LE radius is {th_rLE}, filter is inactive.")

        return ("valid", None)
    
    def monotonic_thickness_filter(self):
        '''
        Checks for compliance in thickness monotonicity, this checks for "dips" in the thickness which may lead to airfoils whose thickness 
        grows, then shrinks and then grows again.
        '''
        mono_t, _ = monotonic_airfoil_thickness(self.x, self.yu, self.yl)

        if mono_t:
            reason = "Monotonic thickness test failed"
            log_line = f"{self.name}.dat: cmb_l = {self.cmb_l}, cmb_u = {self.cmb_u} | Reason: {reason}\n"
            return ("invalid", log_line)
        
        return ("valid", None)
    
    def camber_filters(self, th_c_max=None, th_c_min=None, th_c_f60=[None, None], th_c_r40=[None, None]):
        '''
        Applies various camber filters including independent maximum and minimum camber filters as well as average filters for the front 
        60% and rear 40% of the airfoil.
        Parameters:
        -------------
        th_c_max : float
            Maximum acceptable value for maximum camber in terms of x/c. This value can exceed 1.

        th_c_min : float
            Minimum acceptable value for minimum camber in terms of x/c. This value can exceed -1.
            
        th_c_f60 : list of floats
            th_c_f60[0] Minimum acceptable value for front 60% camber, while th_c_f60[1] is the maximum acceptable value.
        '''
        if th_c_max == None and th_c_min == None and th_c_f60[0] == None and th_c_f60[1] == None and th_c_r40[0] == None and th_c_r40[1] == None:
            if self.verbose:
                print(f"[INFO] Thresholds are empty, no filters applied, returning valid status.")
            return ("valid", None)
        
        c_max, _, _ = self.foil.get_maximum_camber()
        c_min, _, _ = self.foil.get_minimum_camber()
        c_r40 = self.foil.get_average_camber_rear_40p()
        c_f60 = self.foil.get_average_camber_front_60p()

        # Maximum camber filter
        if th_c_max:
            if c_max > th_c_max:
                reason = f"Invalid maximum camber (Greater than {th_c_max*100}%)"
                log_line = f"{self.name}.dat: cmb_l = {self.cmb_l}, cmb_u = {self.cmb_u} | Reason: {reason}\n"
                return ("invalid", log_line)
        else:
            if self.verbose:
                print(f"Threshold for maximum camber is: {th_c_max}, filter is disabled.")

        # Minimum camber filter
        if th_c_min:
            if c_min < th_c_min:
                reason = f"Invalid minimum camber (Smaller than {th_c_min*-100}%)"
                log_line = f"{self.name}.dat: cmb_l = {self.cmb_l}, cmb_u = {self.cmb_u} | Reason: {reason}\n"
                return ("invalid", log_line)
        else:
            if self.verbose:
                print(f"Threshold for minimum camber is: {th_c_min}, filter is disabled.")

        # Front 60% camber filter
        if th_c_f60[0]:
            if c_f60 < th_c_f60[0]:
                reason = f"Invalid minimum front 60% camber (Smaller than {th_c_f60[0]})"
                log_line = f"{self.name}.dat: cmb_l = {self.cmb_l}, cmb_u = {self.cmb_u} | Reason: {reason}\n"
                return ("invalid", log_line)
        else:
            if self.verbose:
                print(f"Threshold for minimum front 60% camber is: {th_c_f60[0]}, filter is disabled.")
        if th_c_f60[1]:
            if c_f60 > th_c_f60[1]:
                reason = f"Invalid maximum front 60% camber (Greater than {th_c_f60[1]})"
                log_line = f"{self.name}.dat: cmb_l = {self.cmb_l}, cmb_u = {self.cmb_u} | Reason: {reason}\n"
                return ("invalid", log_line)
        else:
            if self.verbose:
                print(f"Threshold for maximum front 60% camber is: {th_c_f60[1]}, filter is disabled.")

        # Rear 40% camber filter
        if th_c_r40[0]:
            if c_r40 < th_c_r40[0]:
                reason = f"Invalid minimum rear 40% camber (Smaller than {th_c_r40[0]})"
                log_line = f"{self.name}.dat: cmb_l = {self.cmb_l}, cmb_u = {self.cmb_u} | Reason: {reason}\n"
                return ("invalid", log_line)
        else:
            if self.verbose:
                print(f"Threshold for minimum rear 40% camber is: {th_c_r40[0]}, filter is disabled.")
        if th_c_r40[1]:
            if c_r40 > th_c_r40[1]:
                reason = f"Invalid maximum rear 40% camber (Greater than {th_c_r40[1]})"
                log_line = f"{self.name}.dat: cmb_l = {self.cmb_l}, cmb_u = {self.cmb_u} | Reason: {reason}\n"
                return ("invalid", log_line)
        else:
            if self.verbose:
                print(f"Threshold for maximum rear 40% camber is: {th_c_r40[1]}, filter is disabled.")
        return ("valid", None)