import os
import warnings
import numpy as np
from datetime import datetime
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from tqdm import tqdm

# Suppress warnings from the Gaussian Process Regressor
warnings.filterwarnings("ignore", category=UserWarning)

# Custom exception for handling files with not enough data
class InsufficientPointsError(Exception):
    pass

class Key_Point_Extraction:
    """
    A class to process XFOIL polar files, interpolate the data using a
    Gaussian Process Regressor, and extract key aerodynamic points.
    """

    def __init__(self, polar_file_paths: list):
        """
        Initializes the processor with a list of paths to polar files.

        Args:
            polar_file_paths (list): A list of string paths to the .txt polar files.
        """
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
        
        gpr_cl = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        gpr_cd = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

        gpr_cl.fit(alpha_X, cl_array)
        gpr_cd.fit(alpha_X, cd_array)

        alpha_interp = np.linspace(np.min(alpha_array), np.max(alpha_array), x_len).reshape(-1, 1)
        CL_values = gpr_cl.predict(alpha_interp)
        CD_values = gpr_cd.predict(alpha_interp)

        return CD_values, CL_values

    def _interpolate_and_extract(self, path_to_file: str):
        """Orchestrates the full processing pipeline for a single file."""
        alpha_list, cl_list, cd_list = self._polar_extract(path_to_file)
        alpha_array, cl_array, cd_array = self._polar_cleanup(alpha_list, cl_list, cd_list)

        if len(alpha_array) < 10:
            raise InsufficientPointsError(f"File has only {len(alpha_array)} valid data points, which is less than the required 10.")

        CD_values, CL_values = self._gpr_polar_fit(alpha_array, cl_array, cd_array)
        return self._critical_point_extract(CL_values, CD_values)

    def process_files(self):
        """
        Processes all files provided during initialization.

        Loops through each file path, extracts key points, and stores them.
        Errors are stored in the `self.errors` list.

        Returns:
            list: A list of dictionaries, where each dictionary contains the
                  key points for a successfully processed file.
        """
        self.results = []
        self.errors = []
        
        for file_path in tqdm(self.polar_file_paths, desc="Extracting Key Points"):
            try:
                low_cl_point, high_cl_point, min_cd_point = self._interpolate_and_extract(file_path)
                
                result_dict = {
                    'file_path': os.path.basename(file_path),
                    'min_cl_point': {'CL': low_cl_point[0], 'CD': low_cl_point[1]},
                    'max_cl_point': {'CL': high_cl_point[0], 'CD': high_cl_point[1]},
                    'min_cd_point': {'CL': min_cd_point[0], 'CD': min_cd_point[1]}
                }
                self.results.append(result_dict)
                
            except Exception as e:
                self.errors.append({'file_path': os.path.basename(file_path), 'error': str(e)})
        
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