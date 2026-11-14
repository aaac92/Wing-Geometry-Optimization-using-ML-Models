import logging
import math
import os
import re
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Optional, Literal

class Training_Data_Generator:
    """
    A class to generate training data for both regression and classification models
    from airfoil simulation results.

    Regression file:     one row per airfoil shape and wing configuration with a successful simulation.
    Classification file: one row per airfoil shape; valid = 1 when the shape passed the geometric filters.
    """
    CST_DECIMALS = 4   # decimals of the CST columns of the classification file (the .dat header precision)

    def __init__(self,
                 force_files_dir: str,
                 airfoils_dir: str,
                 invalid_airfoils_path: str,
                 geo_files_dir: str,
                 output_dir: str,
                 n_cst: int,
                 verbose: bool = False,
                 output_mode: Literal['Efficiency', 'Coefficients'] = 'Efficiency',
                 append_kp: bool = False):
        """
        Initializes the TrainingDataGenerator.

        Args:
            force_files_dir (str): Directory containing force files for VALID airfoils.
            airfoils_dir (str): Directory with .dat files for VALID airfoils.
            invalid_airfoils_path (str): Path to the .txt log file for INVALID airfoils.
            geo_files_dir (str): Directory containing AVL geometry files (.avl).
            output_dir (str): Directory where the output CSVs and log will be saved.
            verbose (bool, optional): Enables console printing. Defaults to False.
            output_mode (Literal['Efficiency', 'Coefficients'], optional):
                         Determines the output for the regression file.
                         Defaults to 'Efficiency'.
            append_kp (bool, optional): Toggles the inclusion of key points data. Defaults to False.
        """
        self.force_files_dir = Path(force_files_dir)
        self.airfoils_dir = Path(airfoils_dir)
        self.invalid_airfoils_path = Path(invalid_airfoils_path)
        self.geo_files_dir = Path(geo_files_dir)
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.skipped = []
        self.row_counts = {}
        self.output_mode = output_mode
        self.append_kp = append_kp

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / 'data_generator.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=self.log_path,
            filemode='a'
        )
        # CST headers
        cst_headers = []
        for i in range(n_cst):
            header = f"cst_l_{i+1}"
            cst_headers.append(header)
        for i in range(n_cst):
            header = f"cst_u_{i+1}"
            cst_headers.append(header)
        self.cst_headers = cst_headers
        # Key Point Headers
        self.kp_headers = ["CLneg", "CDneg", "CLcdmin", "CDmin", "CLpos", "CDpos"]
        # Wing Configuration Headers
        self.config_headers = ["rec", "flat", "60DEG"]


    def _log_and_print(self, message: str, level: str = 'info'):
        """Logs a message and prints it to the console if verbose is True."""
        if self.verbose:
            print(message)
        getattr(logging, level)(message)

    def _extract_force_data(self, file_path: Path) -> Optional[Tuple[str, float, float]]:
        """
        Extracts airfoil name, CL, and CD from a force file. CL = 0 is a valid result (e.g. a symmetric section at
        zero incidence, E = 0); only a missing or non-numeric value makes the file unreadable. Until 2026-11-13 a
        zero value counted as missing, which dropped the symmetric airfoils of dataset 17.5 (Test_257, Test_49281
        and the base airfoil Test_98305) as "force file could not be read".
        """
        airfoil, cl_tot, cd_tot = None, None, None
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if "Configuration:" in line and " for " in line:
                        airfoil = line.split(" for ", 1)[1].strip()
                    elif 'CLtot =' in line:
                        cl_tot = float(line.split('=')[1].strip())
                    elif 'CDtot =' in line:
                        cd_tot = float(line.split('=')[1].strip())
            if airfoil and cl_tot is not None and cd_tot is not None and math.isfinite(cl_tot) and math.isfinite(cd_tot):
                return airfoil, cl_tot, cd_tot
            self._log_and_print(f"Incomplete data in {file_path}. Skipping.", 'warning')
            return None
        except FileNotFoundError:
            self._log_and_print(f"Force file not found: {file_path}", 'error')
            return None
        except (ValueError, IndexError) as e:
            self._log_and_print(f"Error parsing force data in {file_path}: {e}", 'error')
            return None

    def _extract_cst_from_dat(self, file_path: Path) -> Optional[List[float]]:
        """Extracts CST coefficients from a valid airfoil .dat file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                match = re.search(r"coefficients:\s*\[([^\]]*)\]", first_line)
                if match:
                    coeffs_str = match.group(1)
                    return [float(c.strip()) for c in coeffs_str.split(',')]
                self._log_and_print(f"CST coefficients not found in {file_path}. Header read: '{first_line}'", 'warning')
                return None
        except FileNotFoundError:
            self._log_and_print(f"Airfoil .dat file not found: {file_path}", 'error')
            return None
        except Exception as e:
            self._log_and_print(f"Error parsing CST from {file_path}: {e}", 'error')
            return None

    def _extract_kp_from_avl(self, file_path: Path) -> Optional[List[float]]:
        """
        Extracts key point values from an AVL geometry file.
 
        Reads the six space-separated floats on the line immediately following
        the CDCL keyword, corresponding to:
        [CLneg, CDneg, CLcdmin, CDmin, CLpos, CDpos]
 
        Args:
            file_path (Path): Path to the .avl geometry file.
 
        Returns:
            A list of 6 floats, or None if parsing fails.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                if line.strip() == 'CDCL':
                    values = lines[i + 1].strip().split()
                    if len(values) != len(self.kp_headers):
                        self._log_and_print(
                            f"Expected {len(self.kp_headers)} key point values in {file_path}, "
                            f"got {len(values)}. Skipping.", 'warning'
                        )
                        return None
                    return [float(v) for v in values]
            self._log_and_print(f"CDCL section not found in {file_path}.", 'warning')
            return None
        except FileNotFoundError:
            self._log_and_print(f"AVL geometry file not found: {file_path}", 'error')
            return None
        except (ValueError, IndexError) as e:
            self._log_and_print(f"Error parsing key points from {file_path}: {e}", 'error')
            return None
        
    def _extract_config_encoding(self, geometry_name: str) -> Optional[List[int]]:
        """
        Derives a one-hot configuration encoding from a geometry name.
 
        Iterates self.config_headers and checks whether geometry_name ends with
        '_{config}'. The first match produces a one-hot list aligned to
        self.config_headers.
 
        Args:
            geometry_name (str): The force file stem with the 'force_' prefix
                                 removed, e.g. 'Test_771_rec'.
 
        Returns:
            A one-hot list of ints, or None if no known config tag is found.
        """
        for i, config in enumerate(self.config_headers):
            if geometry_name.endswith(f"_{config}"):
                encoding = [0] * len(self.config_headers)
                encoding[i] = 1
                return encoding
        self._log_and_print(
            f"No matching configuration tag found in geometry name: '{geometry_name}'. Skipping.",
            'warning'
        )
        return None

    def count_force_files(self) -> int:
        """Number of force files generate_training_files() will read (one progress step each)."""
        return sum(1 for _ in self.force_files_dir.glob('*.txt'))

    def _process_valid_airfoils(self, progress_callback=None) -> List[List]:
        """Processes valid airfoils and returns their CSTs and performance metrics."""
        self._log_and_print("--- Processing valid airfoils ---")
        processed_data = []
        force_files = list(self.force_files_dir.glob('*.txt'))
        
        iterable = tqdm(force_files, desc="Processing valid airfoils") if self.verbose and progress_callback is None else force_files

        for force_file_path in iterable:
            if progress_callback is not None:
                progress_callback()
            force_data = self._extract_force_data(force_file_path)
            if not force_data:
                self.skipped.append((force_file_path.name, "force file could not be read"))
                continue

            airfoil_name, cl, cd = force_data

            geometry_name = force_file_path.stem.replace("force_", "", 1)
            config_encoding = self._extract_config_encoding(geometry_name)
            if not config_encoding:
                self.skipped.append((force_file_path.name, "no wing configuration in the file name"))
                continue

            airfoil_dat_path = self.airfoils_dir / f"{airfoil_name}.dat"
            cst_params = self._extract_cst_from_dat(airfoil_dat_path)
            if not cst_params:
                self.skipped.append((force_file_path.name, "CST coefficients missing from the .dat file"))
                continue

            # Modified execution conditionally reliant on self.append_kp toggle
            if self.append_kp:
                avl_path = self.geo_files_dir / f"{geometry_name}_geo.avl"
                kp_data = self._extract_kp_from_avl(avl_path)
                if not kp_data:
                    continue
            else:
                kp_data = []

            if self.output_mode == 'Efficiency':
                if cd == 0:
                    self._log_and_print(f"CD is zero for {airfoil_name}, cannot compute E. Skipping.", "warning")
                    self.skipped.append((force_file_path.name, "CD is zero"))
                    continue
                performance_metric = [cl / cd + 0.0]   # + 0.0 stores E = -0.0 (CL = -0.0) as 0.0
            else:  # 'Coefficients'
                performance_metric = [cl, cd]
            
            processed_data.append(cst_params + kp_data + config_encoding + performance_metric)
        
        return processed_data

    def _filter_valid_csts(self) -> List[List[float]]:
        """CST coefficients of every airfoil that passed the geometric filters: one .dat file each in airfoils_dir."""
        rows = []
        for dat_path in sorted(self.airfoils_dir.glob('*.dat')):
            cst = self._extract_cst_from_dat(dat_path)
            if cst:
                rows.append(cst)
        return rows

    def _process_invalid_airfoils(self) -> List[List[float]]:
        """Processes invalid airfoils and returns only their CST parameters."""
        self._log_and_print("--- Processing invalid airfoils ---")
        if not self.invalid_airfoils_path.is_file():
            self._log_and_print(f"Path for invalid airfoils is not a file: '{self.invalid_airfoils_path}'.", 'error')
            return []

        processed_data = []
        pattern = re.compile(r"cmb_l = \[([^\]]*)\]\s*,\s*cmb_u = \[([^\]]*)\]")
        try:
            with open(self.invalid_airfoils_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            matches = pattern.finditer(content)
            iterable = tqdm(list(matches), desc="Processing invalid airfoils") if self.verbose else matches

            for match in iterable:
                try:
                    cmb_l_list = [float(n) for n in match.group(1).strip().split()]
                    cmb_u_list = [float(n) for n in match.group(2).strip().split()]
                    processed_data.append(cmb_l_list + cmb_u_list)
                except ValueError as e:
                    self._log_and_print(f"Could not convert numbers in invalid log line: {match.group(0)}. Error: {e}", 'warning')
        except Exception as e:
            self._log_and_print(f"Error reading invalid airfoils file: {e}", 'error')

        return processed_data

    def _write_csv(self, file_path: Path, header: List[str], data_rows: List[List]):
        """Writes data to a CSV file."""
        try:
            with open(file_path, 'w', newline='') as f:
                f.write(",".join(header) + '\n')
                for row in data_rows:
                    if len(row) == len(header):
                        f.write(",".join(map(str, row)) + '\n')
                    else:
                        self._log_and_print(f"Skipping row due to column mismatch. Expected {len(header)}, got {len(row)}. Row: {row}", "warning")

            self._log_and_print(f"Successfully wrote {len(data_rows)} rows to {file_path}")
        except IOError as e:
            self._log_and_print(f"Error writing to CSV file {file_path}: {e}", 'error')

    def generate_training_files(self, regression_filename: str, classification_filename: str, progress_callback=None):
        """
        Generates two training files: one for regression and one for classification.

        Args:
            regression_filename (str): Filename for the regression model (valid data only).
            classification_filename (str): Filename for the classification model (all data).
            progress_callback (callable, optional): Called once per force file read (see count_force_files).

        Afterwards `self.skipped` lists (force file, reason) for every force file left out of the regression
        file and `self.row_counts` the rows written to each file.
        """
        self._log_and_print("--- Starting data generation ---")
        self.skipped = []
        
        valid_data_for_regression = self._process_valid_airfoils(progress_callback)
        invalid_csts_for_classification = self._process_invalid_airfoils()

        # --- Generate the REGRESSION file (valid data only) ---
        if valid_data_for_regression:
            # Reconstruct list variable condition based on the append_kp parameter
            kp_header_list = self.kp_headers if self.append_kp else []

            if self.output_mode == 'Efficiency':
                regr_header = self.cst_headers + kp_header_list + self.config_headers + ['E']
            else:
                regr_header = self.cst_headers + kp_header_list + self.config_headers + ['CL', 'CD']
            
            regr_path = self.output_dir / regression_filename
            self._write_csv(regr_path, regr_header, valid_data_for_regression)
        else:
            self._log_and_print("No valid data processed for regression file.", 'warning')

        # --- Generate the CLASSIFICATION file (combined data) ---
        # One row per airfoil shape. valid = 1: the shape passed the geometric filters (it has a .dat file), whether
        # or not its simulation succeeded later; valid = 0: the filters rejected it. The classifier is a model of the
        # geometric filters only. Both classes are written with the same number of decimals (CST_DECIMALS).
        # Until 2026-11-10 the valid rows were copied from the regression rows: one per wing configuration (three
        # rows per shape) and with fewer decimals than the invalid rows.
        n_cst_columns = len(self.cst_headers)
        seen = set()

        def one_per_shape(cst_rows, label):
            rows = []
            for cst in cst_rows:
                key = tuple(round(float(v), self.CST_DECIMALS) + 0.0 for v in cst[:n_cst_columns])
                if len(key) == n_cst_columns and key not in seen:
                    seen.add(key)
                    rows.append(list(key) + [label])
            return rows

        valid_csts_for_classification = one_per_shape(self._filter_valid_csts(), 1)
        invalid_data_for_classification = one_per_shape(invalid_csts_for_classification, 0)

        all_classification_data = valid_csts_for_classification + invalid_data_for_classification
        self.row_counts = {"regression": len(valid_data_for_regression), "classification_valid": len(valid_csts_for_classification),
                           "classification_invalid": len(invalid_data_for_classification)}
        
        if all_classification_data:
            class_header = self.cst_headers + ['valid']
            class_path = self.output_dir / classification_filename
            self._write_csv(class_path, class_header, all_classification_data)
        else:
            self._log_and_print("No data available to create classification file.", 'warning')

        self._log_and_print("--- Data generation complete. ---")
        logging.shutdown()