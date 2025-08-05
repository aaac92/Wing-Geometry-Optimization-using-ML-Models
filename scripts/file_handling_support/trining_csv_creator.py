import logging
import os
import re
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Optional, Literal

class Training_Data_Generator:
    """
    A class to generate training data for both regression and classification models
    from airfoil simulation results.
    """

    def __init__(self,
                 force_files_dir: str,
                 airfoils_dir: str,
                 invalid_airfoils_path: str,
                 output_dir: str,
                 n_cst: int,
                 verbose: bool = False,
                 output_mode: Literal['Efficiency', 'Coefficients'] = 'Efficiency'):
        """
        Initializes the TrainingDataGenerator.

        Args:
            force_files_dir (str): Directory containing force files for VALID airfoils.
            airfoils_dir (str): Directory with .dat files for VALID airfoils.
            invalid_airfoils_path (str): Path to the .txt log file for INVALID airfoils.
            output_dir (str): Directory where the output CSVs and log will be saved.
            verbose (bool, optional): Enables console printing. Defaults to False.
            output_mode (Literal['Efficiency', 'Coefficients'], optional):
                         Determines the output for the regression file.
                         Defaults to 'Efficiency'.
        """
        self.force_files_dir = Path(force_files_dir)
        self.airfoils_dir = Path(airfoils_dir)
        self.invalid_airfoils_path = Path(invalid_airfoils_path)
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.output_mode = output_mode

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
        #self.cst_headers = ['cst_l_1', 'cst_l_2', 'cst_l_3', 'cst_u_1', 'cst_u_2', 'cst_u_3']


    def _log_and_print(self, message: str, level: str = 'info'):
        """Logs a message and prints it to the console if verbose is True."""
        if self.verbose:
            print(message)
        getattr(logging, level)(message)

    def _extract_force_data(self, file_path: Path) -> Optional[Tuple[str, float, float]]:
        """Extracts airfoil name, CL, and CD from a force file."""
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
            if all((airfoil, cl_tot, cd_tot)):
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

    def _process_valid_airfoils(self) -> List[List]:
        """Processes valid airfoils and returns their CSTs and performance metrics."""
        self._log_and_print("--- Processing valid airfoils ---")
        processed_data = []
        force_files = list(self.force_files_dir.glob('*.txt'))
        
        iterable = tqdm(force_files, desc="Processing valid airfoils") if self.verbose else force_files

        for force_file_path in iterable:
            force_data = self._extract_force_data(force_file_path)
            if not force_data:
                continue

            airfoil_name, cl, cd = force_data
            airfoil_dat_path = self.airfoils_dir / f"{airfoil_name}.dat"
            cst_params = self._extract_cst_from_dat(airfoil_dat_path)
            if not cst_params:
                continue

            if self.output_mode == 'Efficiency':
                if cd == 0:
                    self._log_and_print(f"CD is zero for {airfoil_name}, cannot compute E. Skipping.", "warning")
                    continue
                performance_metric = [cl / cd]
            else:  # 'Coefficients'
                performance_metric = [cl, cd]
            
            processed_data.append(cst_params + performance_metric)
        
        return processed_data

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

    def generate_training_files(self, regression_filename: str, classification_filename: str):
        """
        Generates two training files: one for regression and one for classification.

        Args:
            regression_filename (str): Filename for the regression model (valid data only).
            classification_filename (str): Filename for the classification model (all data).
        """
        self._log_and_print("--- Starting data generation ---")
        
        valid_data_for_regression = self._process_valid_airfoils()
        invalid_csts_for_classification = self._process_invalid_airfoils()

        # --- 1. Generate the REGRESSION file (valid data only) ---
        if valid_data_for_regression:
            if self.output_mode == 'Efficiency':
                regr_header = self.cst_headers + ['E']
            else:
                regr_header = self.cst_headers + ['CL', 'CD']
            
            regr_path = self.output_dir / regression_filename
            self._write_csv(regr_path, regr_header, valid_data_for_regression)
        else:
            self._log_and_print("No valid data processed for regression file.", 'warning')

        # --- 2. Generate the CLASSIFICATION file (combined data) ---
        # Get CSTs from the valid data and add the 'valid=1' flag
        valid_csts_for_classification = [row[:len(self.cst_headers)] + [1] for row in valid_data_for_regression]
        
        # Add the 'valid=0' flag to the invalid data
        invalid_data_for_classification = [cst + [0] for cst in invalid_csts_for_classification]
        
        all_classification_data = valid_csts_for_classification + invalid_data_for_classification
        
        if all_classification_data:
            class_header = self.cst_headers + ['valid']
            class_path = self.output_dir / classification_filename
            self._write_csv(class_path, class_header, all_classification_data)
        else:
            self._log_and_print("No data available to create classification file.", 'warning')

        self._log_and_print("--- Data generation complete. ---")
        logging.shutdown()
