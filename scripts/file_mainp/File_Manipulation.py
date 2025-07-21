import os
import re
from pathlib import Path

class File_Manipulation:
    """
    A class for handling file manipulation tasks.
    """

    def read_AI_result(self, file_path: str) -> tuple[list[float], float]:
        """
        Reads file containing AI prediction results to extract the values
        for C1-C6 and the predicted output.

        Args:
            file_path: The path to the input file.

        Returns:
            A tuple containing:
                - A list of six floats representing the values of C1 to C6.
                - A float representing the 'Predicted Output'.
        """
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        c_values = []
        predicted_output = 0.0

        # It's good practice to handle potential file errors
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    # Split the line into parts to analyze it
                    parts = line.split()
                    
                    # Skip empty or irrelevant lines
                    if not parts:
                        continue
                    
                    # Use regex to find lines starting with C1, C2, ..., C6
                    if re.match(r'C[1-6]', parts[0]):
                        c_values.append(float(parts[1]))
                    
                    # Find the line with the predicted output
                    elif "Predicted" in parts and "Output:" in parts:
                        predicted_output = float(parts[-1])

        except FileNotFoundError:
            print(f"Error: The file at {file_path} was not found.")
            return [], 0.0
        except Exception as e:
            print(f"An error occurred: {e}")
            return [], 0.0
            
        return c_values, predicted_output, file_name
    
    def read_avl_forces(self, directory_path: str) -> list[dict]:
        """
        Reads all AVL force output files in a given directory, extracting key aerodynamic data.

        Args:
            directory_path: The path to the folder containing the force files.

        Returns:
            A list of dictionaries, where each dictionary represents a file and
            contains the airfoil name, CLtot, and CDtot.
            Example: [{'airfoil': 'LGBM_V2.7_Inter_V2.1', 'CLtot': 0.42322, 'CDtot': 0.01598}]
        """
        folder_path = Path(directory_path)
        # Force files have a common extension, e.g., '.txt', customizable in the simulation part
        force_files = self.get_all_files_from(folder_path, extension="txt") 
        
        results_list = []

        for file_path in force_files:
            airfoil = None
            cl_tot = None
            cd_tot = None

            try:
                with open(file_path, 'r') as f:
                    for line in f:
                        # Find airfoil name
                        if "Configuration:" in line:
                            parts = line.split(" for ", 1)
                            if len(parts) > 1:
                                airfoil = parts[1].strip()
                        
                        clean_line = line.strip()
                        # Find CLtot value
                        if clean_line.startswith('CLtot ='):
                            value_str = clean_line.split('=')[1].strip()
                            cl_tot = float(value_str)
                        # Find CDtot value
                        elif clean_line.startswith('CDtot ='):
                            value_str = clean_line.split('=')[1].strip()
                            cd_tot = float(value_str)

                        # Once all data is found, no need to read the rest of the file
                        if airfoil and cl_tot is not None and cd_tot is not None:
                            break
                
                if airfoil and cl_tot is not None and cd_tot is not None:
                    results_list.append({
                        'airfoil': airfoil,
                        'CLtot': cl_tot,
                        'CDtot': cd_tot
                    })

            except Exception as e:
                print(f"Could not process file {file_path.name}: {e}")
        
        return results_list
    
    def get_all_files_from(self, folder_path, extension="*", names_only=False):
        files = list(folder_path.glob(f"*.{extension}")) if extension != "*" else list(folder_path.glob("*"))

        if names_only:
            return [file.name for file in files]
        else:
            return files
        
def xfoil_rel_path(absolute_paths, case):
    """
    Converts a list of absolute airfoil paths to XFOIL-friendly relative paths.
    Parameters:
        absolute_paths (list of str or Path): List of absolute file paths.
        case (str): Relative path prefix from the XFOIL executable to the desired folder.
    Returns:
        list of str: Relative paths in the format XFOIL accepts (with forward slashes).
    """
    if case == 'airfoil':
        relative_prefix="../airfoils/"
    elif case == 'polar':
        relative_prefix="../polars/"
    else:
        print("[Error] Case not recognized, returning empty list")
        return []
    relative_paths = []
    for abs_path in absolute_paths:
        filename = Path(abs_path).name
        rel_path = Path(relative_prefix) / filename
        # Convert to string and replace backslashes with slashes for XFOIL compatibility
        rel_path_str = str(rel_path).replace("\\", "/")
        relative_paths.append(rel_path_str)
    return relative_paths
    
