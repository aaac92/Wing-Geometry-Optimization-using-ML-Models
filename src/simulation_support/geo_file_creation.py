import os
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# You can place this class in the same file as Key_Point_Processing or in a new file.
# If in a new file, you would need to import the other class.

class RecWingGeo_Creation:
    """
    Creates AVL geometry files for rectangular wings based on key aerodynamic points.
    """

    def __init__(self, key_points_data: list, airfoil_abs_paths: list, output_dir: str, verbose=False, pb=False):
        """
        Initializes the geometry file creator.

        Args:
            key_points_data (list): The list of dictionaries from Key_Point_Processing.
            airfoil_abs_paths (list): A list of absolute paths (string or Path objects)
                                      to the airfoil .dat files.
            output_dir (str): The path to the directory where .avl geometry
                              files will be saved.
            verbose (bool): Toggles tqdm progress bars and information prints
        """
        self.verbose = verbose
        self.pb = pb
        self.key_points_data = key_points_data
        self.output_dir = Path(output_dir)
        self.errors = []
        
        # Create a dictionary for fast lookups, mapping airfoil names to their full paths.
        self.airfoil_path_map = {Path(p).stem: p for p in airfoil_abs_paths}

    def create_geo_files(self):
        """
        Loops through the provided airfoil paths and creates a corresponding
        AVL geometry file if matching key point data is found.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # --- Create a lookup map from the key points data ---
        key_data_map = {
            item['file_path'].replace("key_polar_", "").replace("polar_", "").replace(".txt", ""): item
            for item in self.key_points_data
        }

        # --- Iterate through the trusted airfoil paths ---
        for airfoil_name, airfoil_dat_path in tqdm(self.airfoil_path_map.items(), desc="Creating Geometry Files", disable=not self.pb):
            try:
                # Find the key point data that matches the current airfoil name
                item = key_data_map.get(airfoil_name)

                if not item:
                    raise FileNotFoundError(f"Key point data not found for airfoil '{airfoil_name}'")

                # --- Extract Data and Write File ---
                CLmin = item['min_cl_point']['CL']
                CDCLmin = item['min_cl_point']['CD']
                CLmax = item['max_cl_point']['CL']
                CDCLmax = item['max_cl_point']['CD']
                CLCDmin = item['min_cd_point']['CL']
                CDmin = item['min_cd_point']['CD']
                
                geo_file_path = self.output_dir / f"{airfoil_name}_geo.avl"

                # The rest of the file writing logic remains the same
                with open(geo_file_path, "w", encoding="utf-8") as geo:
                    geo.write(f"GeoFile for {airfoil_name}     ! case title (any string)\n")
                    geo.write(f"0.15              Mach number (incompressible)\n")
                    geo.write(f"0 0 0.0           iYsym iZsym Zsym  \n")
                    geo.write(f"4.0 0.8 5.0       Sref Cref Bref \n")
                    geo.write(f"0.0 0.0 0.0       Xref Yref Zref\n")
                    geo.write(f"0.000             CDp (zero for no constant profile drag)\n")
                    geo.write(f"\n")
                    geo.write(f"SURFACE\n")
                    geo.write(f"RectWingSurf     ! surface name\n")
                    geo.write(f"24 1.0 24 -2.0    Nchord Cspace Nspan Sspace \n")
                    geo.write(f"\n")
                    geo.write(f"YDUPLICATE\n")
                    geo.write(f"0.0\n")
                    geo.write(f"\n")
                    geo.write(f"CDCL\n")
                    geo.write(f"{CLmin} {CDCLmin}  {CLCDmin} {CDmin}  {CLmax} {CDCLmax}\n")
                    geo.write(f"\n")
                    geo.write(f"ANGLE\n")
                    geo.write(f"5.0\n")
                    geo.write(f"\n")
                    geo.write(f"SECTION\n")
                    geo.write(f"  0.0  0.0 0.0  0.8  0.0\n")
                    geo.write(f"AFIL\n")
                    geo.write(f"{airfoil_dat_path}\n")
                    geo.write(f"\n")
                    geo.write(f"SECTION\n")
                    geo.write(f"  0.0  2.5 0.0  0.8  0.0\n")
                    geo.write(f"AFIL\n")
                    geo.write(f"{airfoil_dat_path}\n")
                    geo.write(f"\n")
                    geo.write(f"END              ! end of file")

            except Exception as e:
                self.errors.append({'airfoil': airfoil_name, 'error': str(e)})
                continue
        
        return len(self.errors)

    def save_error_log(self, log_path: str):
        """Saves a log of any errors encountered during file creation."""
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("      AVL Geometry Creation Log\n")
            f.write("=" * 60 + "\n")
            f.write(f"Log generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            if not self.errors:
                f.write("--- Process Completed with No Errors ---")
                return

            f.write(f"--- Encountered {len(self.errors)} Error(s) ---" + "-" * 26 + "\n")
            for i, error_info in enumerate(self.errors, 1):
                f.write(f"{i}. Airfoil: {error_info['airfoil']}\n")
                f.write(f"   Error: {error_info['error']}\n")

class SkyeyeGeo_Creation:
    """
    Creates AVL geometry files for a Spyeye 4500 UAV based wing.
    """

    def __init__(self, key_points_data: list, airfoil_abs_paths: list, output_dir: str, verbose=False, pb=False):
        """
        Initializes the geometry file creator.

        Args:
            key_points_data (list): The list of dictionaries from Key_Point_Processing.
            airfoil_abs_paths (list): A list of absolute paths (string or Path objects)
                                      to the airfoil .dat files.
            output_dir (str): The path to the directory where .avl geometry
                              files will be saved.
            verbose (bool): Toggles tqdm progress bars and information prints
        """
        self.verbose = verbose
        self.pb = pb
        self.key_points_data = key_points_data
        self.output_dir = Path(output_dir)
        self.errors = []
        
        # Create a dictionary for fast lookups, mapping airfoil names to their full paths.
        self.airfoil_path_map = {Path(p).stem: p for p in airfoil_abs_paths}

    def create_geo_files(self):
        """
        Loops through the provided airfoil paths and creates a corresponding
        AVL geometry file if matching key point data is found.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create a lookup map from the key points data
        key_data_map = {
            item['file_path'].replace("key_polar_", "").replace("polar_", "").replace(".txt", ""): item
            for item in self.key_points_data
        }

        # --- Iterate through the trusted airfoil paths ---
        for airfoil_name, airfoil_dat_path in tqdm(self.airfoil_path_map.items(), desc="Creating Geometry Files", disable=not self.pb):
            try:
                # Find the key point data that matches the current airfoil name
                item = key_data_map.get(airfoil_name)

                if not item:
                    raise FileNotFoundError(f"Key point data not found for airfoil '{airfoil_name}'")

                # --- Extract Data and Write File ---
                CLmin = item['min_cl_point']['CL']
                CDCLmin = item['min_cl_point']['CD']
                CLmax = item['max_cl_point']['CL']
                CDCLmax = item['max_cl_point']['CD']
                CLCDmin = item['min_cd_point']['CL']
                CDmin = item['min_cd_point']['CD']
                
                geo_file_path = self.output_dir / f"{airfoil_name}_geo.avl"

                # The rest of the file writing logic remains the same
                with open(geo_file_path, "w", encoding="utf-8") as geo:
                    geo.write(f"GeoFile for {airfoil_name}     ! case title (any string)\n")
                    geo.write(f"0.15              Mach number (incompressible)\n")
                    geo.write(f"0 0 0.0           iYsym iZsym Zsym  \n")
                    geo.write(f"1.8 0.4 4.5       Sref Cref Bref \n")
                    geo.write(f"0.0 0.0 0.0       Xref Yref Zref\n")
                    geo.write(f"0.000             CDp (zero for no constant profile drag)\n")
                    geo.write(f"\n")
                    geo.write(f"SURFACE\n")
                    geo.write(f"WingSurf     ! surface name\n")
                    geo.write(f"24 1.0 24 -2.0    Nchord Cspace Nspan Sspace \n")
                    geo.write(f"\n")
                    geo.write(f"YDUPLICATE\n")
                    geo.write(f"0.0\n")
                    geo.write(f"\n")
                    geo.write(f"CDCL\n")
                    geo.write(f"{CLmin} {CDCLmin}  {CLCDmin} {CDmin}  {CLmax} {CDCLmax}\n")
                    geo.write(f"\n")
                    geo.write(f"ANGLE\n") # Determines wing twist
                    geo.write(f"0.0\n")
                    geo.write(f"\n")
                    geo.write(f"SECTION\n")
                    geo.write(f"  0.0  0.0 0.0  0.4  0.0\n")
                    geo.write(f"AFIL\n")
                    geo.write(f"{airfoil_dat_path}\n")
                    geo.write(f"\n")
                    geo.write(f"SECTION\n")
                    geo.write(f"  0.0  0.640  0.0  0.4  0.0\n")
                    geo.write(f"AFIL\n")
                    geo.write(f"{airfoil_dat_path}\n")
                    geo.write(f"\n")
                    geo.write(f"SECTION\n")
                    geo.write(f"  0.076  2.25  0.0  0.2  0.0\n")
                    geo.write(f"AFIL\n")
                    geo.write(f"{airfoil_dat_path}\n")
                    geo.write(f"\n")
                    geo.write(f"END              ! end of file")

            except Exception as e:
                self.errors.append({'airfoil': airfoil_name, 'error': str(e)})
                continue
        
        return len(self.errors)

    def save_error_log(self, log_path: str):
        """Saves a log of any errors encountered during file creation."""
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("      AVL Geometry Creation Log\n")
            f.write("=" * 60 + "\n")
            f.write(f"Log generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            if not self.errors:
                f.write("--- Process Completed with No Errors ---")
                return

            f.write(f"--- Encountered {len(self.errors)} Error(s) ---" + "-" * 26 + "\n")
            for i, error_info in enumerate(self.errors, 1):
                f.write(f"{i}. Airfoil: {error_info['airfoil']}\n")
                f.write(f"   Error: {error_info['error']}\n")

class HossainWingRec_Creation:
    """
    Creates AVL geometry files for Hossain's wing without winglets.
    """

    def __init__(self, key_points_data: list, airfoil_abs_paths: list, output_dir: str, verbose=False, pb=False):
        """
        Initializes the geometry file creator.

        Args:
            key_points_data (list): The list of dictionaries from Key_Point_Processing.
            airfoil_abs_paths (list): A list of absolute paths (string or Path objects)
                                      to the airfoil .dat files.
            output_dir (str): The path to the directory where .avl geometry
                              files will be saved.
            verbose (bool): Toggles tqdm progress bars and information prints
        """
        self.verbose = verbose
        self.pb = pb
        self.key_points_data = key_points_data
        self.output_dir = Path(output_dir)
        self.errors = []
        
        # Create a dictionary for fast lookups, mapping airfoil names to their full paths.
        self.airfoil_path_map = {Path(p).stem: p for p in airfoil_abs_paths}

    def create_geo_files(self):
        """
        Loops through the provided airfoil paths and creates a corresponding
        AVL geometry file if matching key point data is found.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create a lookup map from the key points data
        key_data_map = {
            item['file_path'].replace("key_polar_", "").replace("polar_", "").replace(".txt", ""): item
            for item in self.key_points_data
        }

        # --- Iterate through the trusted airfoil paths ---
        for airfoil_name, airfoil_dat_path in tqdm(self.airfoil_path_map.items(), desc="Creating Geometry Files", disable=not self.pb):
            try:
                # Find the key point data that matches the current airfoil name
                item = key_data_map.get(airfoil_name)

                if not item:
                    raise FileNotFoundError(f"Key point data not found for airfoil '{airfoil_name}'")

                # --- Extract Data and Write File ---
                CLmin = item['min_cl_point']['CL']
                CDCLmin = item['min_cl_point']['CD']
                CLmax = item['max_cl_point']['CL']
                CDCLmax = item['max_cl_point']['CD']
                CLCDmin = item['min_cd_point']['CL']
                CDmin = item['min_cd_point']['CD']
                
                geo_file_path = self.output_dir / f"{airfoil_name}_rec_geo.avl"

                # The rest of the file writing logic remains the same
                with open(geo_file_path, "w", encoding="utf-8") as geo:
                    geo.write(f"Hossin_Rec_GeoFile for {airfoil_name} \n")
                    geo.write(f"0.09367               Mach number (incompressible)\n")
                    geo.write(f"0 0 0.0               iYsym iZsym Zsym  \n")
                    geo.write(f"0.07986 0.121 0.66    Sref Cref Bref \n")
                    geo.write(f"0.0 0.0 0.0           Xref Yref Zref\n")
                    geo.write(f"0.000                 CDp (zero for no constant profile drag)\n")
                    geo.write(f"\n")
                    geo.write(f"SURFACE\n")
                    geo.write(f"Hossin_WingSurf     ! surface name\n")
                    geo.write(f"24 1.0 24 -2.0    Nchord Cspace Nspan Sspace \n")
                    geo.write(f"\n")
                    geo.write(f"YDUPLICATE\n")
                    geo.write(f"0.0\n")
                    geo.write(f"\n")
                    geo.write(f"CDCL\n")
                    geo.write(f"{CLmin} {CDCLmin}  {CLCDmin} {CDmin}  {CLmax} {CDCLmax}\n")
                    geo.write(f"\n")
                    geo.write(f"ANGLE\n") # Determines wing twist
                    geo.write(f"0.0\n")
                    geo.write(f"\n")
                    geo.write(f"SECTION\n")
                    geo.write(f"  0.0  0.0 0.0  0.121  0.0\n")
                    geo.write(f"AFIL\n")
                    geo.write(f"{airfoil_dat_path}\n")
                    geo.write(f"\n")
                    geo.write(f"SECTION\n")
                    geo.write(f"  0.0  0.66  0.0  0.121  0.0\n")
                    geo.write(f"AFIL\n")
                    geo.write(f"{airfoil_dat_path}\n")
                    geo.write(f"\n")
                    geo.write(f"END              ! end of file")

            except Exception as e:
                self.errors.append({'airfoil': airfoil_name, 'error': str(e)})
                continue
        
        return len(self.errors)

    def save_error_log(self, log_path: str):
        """Saves a log of any errors encountered during file creation."""
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("      AVL Geometry Creation Log\n")
            f.write("=" * 60 + "\n")
            f.write(f"Log generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            if not self.errors:
                f.write("--- Process Completed with No Errors ---")
                return

            f.write(f"--- Encountered {len(self.errors)} Error(s) ---" + "-" * 26 + "\n")
            for i, error_info in enumerate(self.errors, 1):
                f.write(f"{i}. Airfoil: {error_info['airfoil']}\n")
                f.write(f"   Error: {error_info['error']}\n")

class HossainWing60DEG_Creation:
    """
    Creates AVL geometry files for Hossain's wing with flat winglets.
    """

    def __init__(self, key_points_data: list, airfoil_abs_paths: list, output_dir: str, verbose=False, pb=False):
        """
        Initializes the geometry file creator.

        Args:
            key_points_data (list): The list of dictionaries from Key_Point_Processing.
            airfoil_abs_paths (list): A list of absolute paths (string or Path objects)
                                      to the airfoil .dat files.
            output_dir (str): The path to the directory where .avl geometry
                              files will be saved.
            verbose (bool): Toggles tqdm progress bars and information prints
        """
        self.verbose = verbose
        self.pb = pb
        self.key_points_data = key_points_data
        self.output_dir = Path(output_dir)
        self.errors = []
        
        # Create a dictionary for fast lookups, mapping airfoil names to their full paths.
        self.airfoil_path_map = {Path(p).stem: p for p in airfoil_abs_paths}

    def create_geo_files(self):
        """
        Loops through the provided airfoil paths and creates a corresponding
        AVL geometry file if matching key point data is found.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create a lookup map from the key points data
        key_data_map = {
            item['file_path'].replace("key_polar_", "").replace("polar_", "").replace(".txt", ""): item
            for item in self.key_points_data
        }

        # --- Iterate through the trusted airfoil paths ---
        for airfoil_name, airfoil_dat_path in tqdm(self.airfoil_path_map.items(), desc="Creating Geometry Files", disable=not self.pb):
            try:
                # Find the key point data that matches the current airfoil name
                item = key_data_map.get(airfoil_name)

                if not item:
                    raise FileNotFoundError(f"Key point data not found for airfoil '{airfoil_name}'")

                # --- Extract Data and Write File ---
                CLmin = item['min_cl_point']['CL']
                CDCLmin = item['min_cl_point']['CD']
                CLmax = item['max_cl_point']['CL']
                CDCLmax = item['max_cl_point']['CD']
                CLCDmin = item['min_cd_point']['CL']
                CDmin = item['min_cd_point']['CD']
                
                geo_file_path = self.output_dir / f"{airfoil_name}_60DEG_geo.avl"

                # The rest of the file writing logic remains the same
                with open(geo_file_path, "w", encoding="utf-8") as geo:
                    geo.write(f"Hossin_60DEG_GeoFile for {airfoil_name} \n")
                    geo.write(f"0.09367                   Mach number (incompressible)\n")
                    geo.write(f"0 0 0.0                   iYsym iZsym Zsym  \n")
                    geo.write(f"0.09136 0.1102 0.781    Sref Cref Bref \n")
                    geo.write(f"0.0 0.0 0.0               Xref Yref Zref\n")
                    geo.write(f"0.000                     CDp (zero for no constant profile drag)\n")
                    geo.write(f"\n")
                    geo.write(f"SURFACE\n")
                    geo.write(f"Hossin_WingSurf     ! surface name\n")
                    geo.write(f"24 1.0 24 -2.0    Nchord Cspace Nspan Sspace \n")
                    geo.write(f"\n")
                    geo.write(f"YDUPLICATE\n")
                    geo.write(f"0.0\n")
                    geo.write(f"\n")
                    geo.write(f"CDCL\n")
                    geo.write(f"{CLmin} {CDCLmin}  {CLCDmin} {CDmin}  {CLmax} {CDCLmax}\n")
                    geo.write(f"\n")
                    geo.write(f"ANGLE\n") # Determines wing twist
                    geo.write(f"0.0\n")
                    geo.write(f"\n")
                    geo.write(f"SECTION\n")
                    geo.write(f"  0.0  0.0 0.0  0.121  0.0\n")
                    geo.write(f"AFIL\n")
                    geo.write(f"{airfoil_dat_path}\n")
                    geo.write(f"\n")
                    geo.write(f"SECTION\n")
                    geo.write(f"  0.0  0.33 0.0  0.121  0.0\n")
                    geo.write(f"AFIL\n")
                    geo.write(f"{airfoil_dat_path}\n")
                    geo.write(f"\n")
                    geo.write(f"SECTION\n")
                    geo.write(f"  0.00413  0.3456 0.0271  0.11687  0.0\n")
                    geo.write(f"NACA\n")
                    geo.write(f"0012\n")
                    geo.write(f"\n")
                    geo.write(f"SECTION\n")
                    geo.write(f"  0.01621  0.36025 0.05239  0.1048  0.0\n")
                    geo.write(f"NACA\n")
                    geo.write(f"0012\n")
                    geo.write(f"\n")
                    geo.write(f"SECTION\n")
                    geo.write(f"  0.03544  0.3727 0.07409  0.08556  0.0\n")
                    geo.write(f"NACA\n")
                    geo.write(f"0012\n")
                    geo.write(f"\n")
                    geo.write(f"SECTION\n")
                    geo.write(f"  0.0605  0.3823 0.09075  0.0605  0.0\n")
                    geo.write(f"NACA\n")
                    geo.write(f"0012\n")
                    geo.write(f"\n")
                    geo.write(f"SECTION\n")
                    geo.write(f"    0.08968  0.3884 0.101218  0.03131  0.0\n")
                    geo.write(f"NACA\n")
                    geo.write(f"0012\n")
                    geo.write(f"\n")
                    geo.write(f"SECTION\n")
                    geo.write(f"  0.121  0.3905 0.10479  0.0  0.0\n")
                    geo.write(f"NACA\n")
                    geo.write(f"0012\n")
                    geo.write(f"\n")
                    geo.write(f"END              ! end of file")

            except Exception as e:
                self.errors.append({'airfoil': airfoil_name, 'error': str(e)})
                continue
        
        return len(self.errors)

    def save_error_log(self, log_path: str):
        """Saves a log of any errors encountered during file creation."""
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("      AVL Geometry Creation Log\n")
            f.write("=" * 60 + "\n")
            f.write(f"Log generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            if not self.errors:
                f.write("--- Process Completed with No Errors ---")
                return

            f.write(f"--- Encountered {len(self.errors)} Error(s) ---" + "-" * 26 + "\n")
            for i, error_info in enumerate(self.errors, 1):
                f.write(f"{i}. Airfoil: {error_info['airfoil']}\n")
                f.write(f"   Error: {error_info['error']}\n")

class HossainWingFlat_Creation:
    """
    Creates AVL geometry files for Hossin's wing with 60° winglets.
    """

    def __init__(self, key_points_data: list, airfoil_abs_paths: list, output_dir: str, verbose=False, pb=False):
        """
        Initializes the geometry file creator.

        Args:
            key_points_data (list): The list of dictionaries from Key_Point_Processing.
            airfoil_abs_paths (list): A list of absolute paths (string or Path objects)
                                      to the airfoil .dat files.
            output_dir (str): The path to the directory where .avl geometry
                              files will be saved.
            verbose (bool): Toggles tqdm progress bars and information prints
        """
        self.verbose = verbose
        self.pb = pb
        self.key_points_data = key_points_data
        self.output_dir = Path(output_dir)
        self.errors = []
        
        # Create a dictionary for fast lookups, mapping airfoil names to their full paths.
        self.airfoil_path_map = {Path(p).stem: p for p in airfoil_abs_paths}

    def create_geo_files(self):
        """
        Loops through the provided airfoil paths and creates a corresponding
        AVL geometry file if matching key point data is found.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create a lookup map from the key points data
        key_data_map = {
            item['file_path'].replace("key_polar_", "").replace("polar_", "").replace(".txt", ""): item
            for item in self.key_points_data
        }

        # --- Iterate through the trusted airfoil paths ---
        for airfoil_name, airfoil_dat_path in tqdm(self.airfoil_path_map.items(), desc="Creating Geometry Files", disable=not self.pb):
            try:
                # Find the key point data that matches the current airfoil name
                item = key_data_map.get(airfoil_name)

                if not item:
                    raise FileNotFoundError(f"Key point data not found for airfoil '{airfoil_name}'")

                # --- Extract Data and Write File ---
                CLmin = item['min_cl_point']['CL']
                CDCLmin = item['min_cl_point']['CD']
                CLmax = item['max_cl_point']['CL']
                CDCLmax = item['max_cl_point']['CD']
                CLCDmin = item['min_cd_point']['CL']
                CDmin = item['min_cd_point']['CD']
                
                geo_file_path = self.output_dir / f"{airfoil_name}_flat_geo.avl"

                # The rest of the file writing logic remains the same
                with open(geo_file_path, "w", encoding="utf-8") as geo:
                    geo.write(f"Hossin_Flat_GeoFile for {airfoil_name} \n")
                    geo.write(f"0.09367                     Mach number (incompressible)\n")
                    geo.write(f"0 0 0.0                     iYsym iZsym Zsym  \n")
                    geo.write(f"0.102858 0.0964 0.902       Sref Cref Bref \n")
                    geo.write(f"0.0 0.0 0.0                 Xref Yref Zref\n")
                    geo.write(f"0.000                       CDp (zero for no constant profile drag)\n")
                    geo.write(f"\n")
                    geo.write(f"SURFACE\n")
                    geo.write(f"Hossin_WingSurf     ! surface name\n")
                    geo.write(f"24 1.0 24 -2.0    Nchord Cspace Nspan Sspace \n")
                    geo.write(f"\n")
                    geo.write(f"YDUPLICATE\n")
                    geo.write(f"0.0\n")
                    geo.write(f"\n")
                    geo.write(f"CDCL\n")
                    geo.write(f"{CLmin} {CDCLmin}  {CLCDmin} {CDmin}  {CLmax} {CDCLmax}\n")
                    geo.write(f"\n")
                    geo.write(f"ANGLE\n") # Determines wing twist
                    geo.write(f"0.0\n")
                    geo.write(f"\n")
                    geo.write(f"SECTION\n")
                    geo.write(f"  0.0  0.33 0.0  0.121  0.0\n")
                    geo.write(f"AFIL\n")
                    geo.write(f"{airfoil_dat_path}\n")
                    geo.write(f"\n")
                    geo.write(f"SECTION\n")
                    geo.write(f"  0.00413  0.3613 0.0  0.11687  0.0\n")
                    geo.write(f"AFIL\n")
                    geo.write(f"{airfoil_dat_path}\n")
                    geo.write(f"\n")
                    geo.write(f"SECTION\n")
                    geo.write(f"  0.01621  0.3905 0.0  0.1048  0.0\n")
                    geo.write(f"AFIL\n")
                    geo.write(f"{airfoil_dat_path}\n")
                    geo.write(f"\n")
                    geo.write(f"SECTION\n")
                    geo.write(f"  0.03544  0.4156 0.0  0.08556  0.0\n")
                    geo.write(f"AFIL\n")
                    geo.write(f"{airfoil_dat_path}\n")
                    geo.write(f"\n")
                    geo.write(f"SECTION\n")
                    geo.write(f"  0.0605  0.4348 0.0  0.0605  0.0\n")
                    geo.write(f"AFIL\n")
                    geo.write(f"{airfoil_dat_path}\n")
                    geo.write(f"\n")
                    geo.write(f"SECTION\n")
                    geo.write(f"  0.08968  0.4469 0.0  0.03131  0.0\n")
                    geo.write(f"AFIL\n")
                    geo.write(f"{airfoil_dat_path}\n")
                    geo.write(f"\n")
                    geo.write(f"SECTION\n")
                    geo.write(f"  0.121  0.451 0.0  0.0  0.0\n")
                    geo.write(f"AFIL\n")
                    geo.write(f"{airfoil_dat_path}\n")
                    geo.write(f"\n")
                    geo.write(f"END              ! end of file")

            except Exception as e:
                self.errors.append({'airfoil': airfoil_name, 'error': str(e)})
                continue
        
        return len(self.errors)

    def save_error_log(self, log_path: str):
        """Saves a log of any errors encountered during file creation."""
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("      AVL Geometry Creation Log\n")
            f.write("=" * 60 + "\n")
            f.write(f"Log generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            if not self.errors:
                f.write("--- Process Completed with No Errors ---")
                return

            f.write(f"--- Encountered {len(self.errors)} Error(s) ---" + "-" * 26 + "\n")
            for i, error_info in enumerate(self.errors, 1):
                f.write(f"{i}. Airfoil: {error_info['airfoil']}\n")
                f.write(f"   Error: {error_info['error']}\n")