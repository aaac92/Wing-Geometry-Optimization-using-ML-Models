import hashlib
import os
import re
import warnings
import numpy as np
from datetime import datetime
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from tqdm import tqdm

# Suppress warnings from the Gaussian Process Regressor
warnings.filterwarnings("ignore", category=UserWarning)

# Custom exception for handling files with not enough data
class InsufficientPointsError(Exception):
    pass

class PoorDistributionError(ValueError):
    """Custom exception for poorly distributed data points."""
    pass

class Key_Point_Extraction:
    """
    A class to process XFOIL polar files, interpolate the data using a
    Gaussian Process Regressor, and extract key aerodynamic points.
    """

    def __init__(self, polar_file_paths: list, verbose=False, random_state=None):
        """
        Initializes the processor with a list of paths to polar files.

        Args:
            polar_file_paths (list): A list of string paths to the .txt polar files.
            random_state (int, optional): Seed for the Gaussian Process optimizer restarts. None (default)
                keeps the original unseeded behavior, where repeated fits of the same polar can differ.
        """

        self.verbose = verbose
        self.random_state = random_state
        if not isinstance(polar_file_paths, list):
            raise TypeError("polar_file_paths must be a list of strings.")
        self.polar_file_paths = polar_file_paths
        self.results = []
        self.errors = []

    def _polar_extract(self, polar_path):
        # Open and read the polar file
        with open(polar_path, "r") as file:
            lines = file.readlines()
        data_start = -1
        # Find where data starts (after header dashed line)
        for i, line in enumerate(lines):
            if line.strip().startswith("------"):
                data_start = i + 1
                break

        # Store all raw lines from the data section
        raw_data_lines = []

        for line in lines[data_start:]:
            if "XFOIL process completed successfully" in line:
                continue
            raw_data_lines.append(line.rstrip())  # remove trailing newline

        alpha_list = []
        cl_list = []
        cd_list = []

        for line in raw_data_lines:
            # Skip lines that are clearly too short
            if len(line.strip()) < 30:
                continue

            try:
                # Get alpha
                alpha_end = 8
                alpha = line[1:alpha_end].strip()

                # Get CL (9 characters after alpha)
                cl_start = alpha_end
                cl_end = cl_start + 9
                cl = line[cl_start:cl_end].strip()

                # CD is at least 2 spaces after CL
                # So we scan for the first non-space character after 2 spaces from cl_end
                cd_start = cl_end + 2
                cd_end = cd_start + 8
                cd = line[cd_start:cd_end].strip()

                alpha_list.append(alpha)
                cl_list.append(cl)
                cd_list.append(cd)
            except IndexError:
                continue  # Skip malformed lines
        return alpha_list, cl_list, cd_list

    def _polar_cleanup(self, alpha_list: list, cl_list: list, cd_list: list):
        """Cleans the extracted data, removing non-converged points."""
        cl_corrupted = ['*' in cl for cl in cl_list]
        cd_corrupted = ['*' in cd for cd in cd_list]
        final_mask = [cl_err or cd_err for cl_err, cd_err in zip(cl_corrupted, cd_corrupted)]

        alpha_clean = [a for a, bad in zip(alpha_list, final_mask) if not bad]
        cl_clean = [cl for cl, bad in zip(cl_list, final_mask) if not bad]
        cd_clean = [cd for cd, bad in zip(cd_list, final_mask) if not bad]

        alpha_array = np.array([float(a) for a in alpha_clean])
        cl_array = np.array([float(cl) for cl in cl_clean])
        cd_array = np.array([float(cd) for cd in cd_clean])

        return alpha_array, cl_array, cd_array

    def _critical_point_extract(self, cl_array: np.ndarray, cd_array: np.ndarray):
        """Finds the min/max CL and min CD points from the arrays."""
        low_cl_idx = np.argmin(cl_array)
        high_cl_idx = np.argmax(cl_array)
        min_cd_idx = np.argmin(cd_array)

        low_cl_point = (cl_array[low_cl_idx], cd_array[low_cl_idx])
        high_cl_point = (cl_array[high_cl_idx], cd_array[high_cl_idx])
        min_cd_point = (cl_array[min_cd_idx], cd_array[min_cd_idx])

        return low_cl_point, high_cl_point, min_cd_point

    def _gpr_polar_fit(self, alpha_array: np.ndarray, cl_array: np.ndarray, cd_array: np.ndarray, x_len: int = 1000):
        """Fits a Gaussian Process Regressor to the polar data."""
        if len(alpha_array) < 2:
            return (np.array([]), np.array([]))

        alpha_X = alpha_array.reshape(-1, 1)
        kernel = 1.0 * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(noise_level=1e-5)
        
        gpr_cl = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=self.random_state)
        gpr_cd = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=self.random_state)

        gpr_cl.fit(alpha_X, cl_array)
        gpr_cd.fit(alpha_X, cd_array)

        alpha_interp = np.linspace(np.min(alpha_array), np.max(alpha_array), x_len).reshape(-1, 1)
        CL_values = gpr_cl.predict(alpha_interp)
        CD_values = gpr_cd.predict(alpha_interp)

        return CD_values, CL_values

    def _interpolate_and_extract(self, path_to_file: str, min_points=40):
        """Orchestrates the full processing pipeline for a single file."""
        alpha_list, cl_list, cd_list = self._polar_extract(path_to_file)
        alpha_array, cl_array, cd_array = self._polar_cleanup(alpha_list, cl_list, cd_list)

        # Check for minimum amount of points
        if len(alpha_array) < min_points:
            raise InsufficientPointsError(f"File has only {len(alpha_array)} valid data points, which is less than the required {min_points}.")
        
        # Check for point distribution in key alpha ranges
        conditions = {
            "2 points between (0, 10]": np.sum((alpha_array > 0) & (alpha_array <= 10)) >= 6,
            "2 points between [-10, 0)": np.sum((alpha_array >= -10) & (alpha_array < 0)) >= 6,
            "1 point between [11, 15]": np.sum((alpha_array >= 11) & (alpha_array <= 15)) >= 3,
            "1 point between [-15, -11]": np.sum((alpha_array >= -15) & (alpha_array <= -11)) >= 3,
            "1 point between [16, 20]": np.sum((alpha_array >= 16) & (alpha_array <= 20)) >= 1,
            "1 point between [-20, -16]": np.sum((alpha_array >= -20) & (alpha_array <= -16)) >= 1,
        }

        # Check if all conditions are met
        if not all(conditions.values()):
            # Find which conditions failed to provide a detailed error message
            failed_conditions = [desc for desc, met in conditions.items() if not met]
            error_message = f"Poor data point distribution. Missing: {'; '.join(failed_conditions)}."
            raise PoorDistributionError(error_message)

        CD_values, CL_values = self._gpr_polar_fit(alpha_array, cl_array, cd_array)

        return self._critical_point_extract(CL_values, CD_values)

    def process_file(self, file_path):
        """
        Key points of one polar file. Returns (result dict, None) on success or (None, error dict).
        The result dict has the format process_files() collects.
        """
        try:
            low_cl_point, high_cl_point, min_cd_point = self._interpolate_and_extract(file_path)
            return {
                'file_path': os.path.basename(file_path),
                'min_cl_point': {'CL': float(low_cl_point[0]), 'CD': float(low_cl_point[1])},
                'max_cl_point': {'CL': float(high_cl_point[0]), 'CD': float(high_cl_point[1])},
                'min_cd_point': {'CL': float(min_cd_point[0]), 'CD': float(min_cd_point[1])}
            }, None
        except Exception as e:
            return None, {'file_path': os.path.basename(file_path), 'error': str(e)}

    def polar_cd_min(self, file_path):
        """(CD, alpha, CL) of the converged polar point with the lowest CD, or None if the polar has no points."""
        try:
            alpha, cl, cd = self._polar_cleanup(*self._polar_extract(file_path))
        except (OSError, ValueError):
            return None
        if not len(cd):
            return None
        i = int(np.argmin(cd))
        return float(cd[i]), float(alpha[i]), float(cl[i])

    def process_files(self, progress_callback=None):
        """
        Processes all files provided during initialization.

        Loops through each file path, extracts key points, and stores them.
        Errors are stored in the `self.errors` list.

        Args:
            progress_callback (callable, optional): Called once (no args) after each file. When given, the
                caller owns the progress display and no tqdm bar is created.

        Returns:
            list: A list of dictionaries, where each dictionary contains the
                  key points for a successfully processed file.
        """
        self.results = []
        self.errors = []

        files = self.polar_file_paths
        if progress_callback is None:
            files = tqdm(files, desc="Extracting Key Points", disable=not self.verbose)
        for file_path in files:
            result, error = self.process_file(file_path)
            if result is not None:
                self.results.append(result)
            else:
                self.errors.append(error)
            if progress_callback is not None:
                progress_callback()

        return self.results
    
    def save_error_log(self, log_path: str):
        """
        Saves a detailed log of the processing session to a text file.

        This method should be called after `process_files()` has been run.

        Args:
            log_path (str): The full path where the .txt log file will be saved.
        """
        # Ensure the directory for the log file exists before writing
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        total_attempted = len(self.polar_file_paths)
        total_success = len(self.results)
        total_errors = len(self.errors)

        with open(log_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("      Key Point Processing Log\n")
            f.write("=" * 60 + "\n")
            f.write(f"Log generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("--- Summary ---\n")
            f.write(f"Total files attempted: {total_attempted}\n")
            f.write(f"Successfully processed: {total_success}\n")
            f.write(f"Failed to process: {total_errors}\n\n")

            f.write("--- Successful Files ---" + "-" * 39 + "\n")
            if self.results:
                for i, result in enumerate(self.results, 1):
                    f.write(f"{i}. {result['file_path']}\n")
            else:
                f.write("None\n")
            f.write("\n")

            f.write("--- Failed Files & Errors ---" + "-" * 32 + "\n")
            if self.errors:
                for i, error_info in enumerate(self.errors, 1):
                    f.write(f"{i}. File: {error_info['file_path']}\n")
                    f.write(f"   Error: {error_info['error']}\n")
            else:
                f.write("None\n")
            f.write("\n" + "=" * 60 + "\n")

# =============================================================================================== #
# === Key-point files: the CDCL polar of the AVL winglet sections (read by geo_file_creation) === #
# Moved here from utilities/kp_extraction.py (2026-11-16) so utilities/airfoil_kp_exporter.py uses them too.
# =============================================================================================== #

KEY_POINT_FILE_HEADER = "CLmin CDCLmin CLCDmin CDmin CLmax CDCLmax"  # line 2 of a key-point file


def extract_polar_metadata(polar_path: Path) -> dict:
    """Extracts aerodynamic flow regime metadata from the XFOIL polar file header."""
    metadata = {'name': 'Unknown', 'mach': '0.0', 're': '0.0', 'ncrit': '0.0'}
    
    with open(polar_path, "r", encoding="utf-8") as f:
        # Limit search to the first 15 lines to avoid scanning the entire data array
        for _ in range(15):
            line = f.readline()
            if not line:
                break
            
            # Extract airfoil name
            if "Calculated polar for:" in line:
                metadata['name'] = line.split("Calculated polar for:")[1].strip()
            
            # Extract flow regime parameters
            if "Mach =" in line and "Re =" in line:
                mach_match = re.search(r"Mach\s*=\s*([0-9\.]+)", line)
                re_match = re.search(r"Re\s*=\s*([0-9\.\s]+e\s*[0-9]+|[0-9\.]+)", line)
                ncrit_match = re.search(r"Ncrit\s*=\s*([0-9\.]+)", line)
                
                if mach_match:
                    metadata['mach'] = mach_match.group(1).strip()
                if ncrit_match:
                    metadata['ncrit'] = ncrit_match.group(1).strip()
                if re_match:
                    # Strip internal whitespace from scientific notation
                    raw_re = re_match.group(1).replace(" ", "")
                    try:
                        re_float = float(raw_re)
                        metadata['re'] = f"{re_float:g}"  # Formats to standard float representation
                    except ValueError:
                        metadata['re'] = raw_re
                        
    return metadata


def read_key_points_file(path) -> list:
    """The six values of a key-point file (its last non-empty line, the one pasted after CDCL in the .avl files)."""
    lines = [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    values = [float(v) for v in lines[-1].split()] if lines else []
    if len(values) != 6:
        raise ValueError(f"{path}: expected six key-point values in its last line, found {len(values)}.")
    return values


def key_points_hash(path) -> str:
    """Short hash of the six values of a key-point file. Not of its bytes: line endings differ between machines."""
    return hashlib.sha1(" ".join(repr(v) for v in read_key_points_file(path)).encode()).hexdigest()[:12]


def key_points_result(kp) -> dict:
    """The six key points in file order (as polar_key_points returns them) in the format of process_file()."""
    return {'min_cl_point': {'CL': kp[0], 'CD': kp[1]}, 'min_cd_point': {'CL': kp[2], 'CD': kp[3]},
            'max_cl_point': {'CL': kp[4], 'CD': kp[5]}}


def write_key_points_file(result_dict: dict, output_path: Path, meta: dict):
    """Writes the extracted key points and regime metadata to a text file."""
    with open(output_path, "w", encoding="utf-8") as f:
        # Line 1: Flow regime metadata header
        f.write(f"{meta['name']} Mach={meta['mach']} Re={meta['re']} Ncrit={meta['ncrit']}\n")
        
        # Line 2: Data structure header
        f.write(KEY_POINT_FILE_HEADER + "\n")
        
        # Line 3: Extracted aerodynamic values in raw floating-point format
        cl_min = result_dict['min_cl_point']['CL']
        cd_cl_min = result_dict['min_cl_point']['CD']
        
        cl_cd_min = result_dict['min_cd_point']['CL']
        cd_min = result_dict['min_cd_point']['CD']
        
        cl_max = result_dict['max_cl_point']['CL']
        cd_cl_max = result_dict['max_cl_point']['CD']
        
        f.write(f"{cl_min} {cd_cl_min} {cl_cd_min} {cd_min} {cl_max} {cd_cl_max}\n")
