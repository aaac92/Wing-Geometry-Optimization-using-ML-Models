import json
import os
from pathlib import Path
from typing import List, Tuple
from file_handling_support import compression_Alma

class recovery_json:
    def __init__(self, directory: Path = None):
        if directory is not None:
            recovery_file_path = Path(directory) / "recovery_status.json"
        else:
            recovery_file_path = Path(__file__).resolve().parent.parent / "recovery_status.json"
        self.recovery_file = recovery_file_path

    def get_recovery_status(self):
        if self.recovery_file.exists():
            with open(self.recovery_file, 'r') as f:
                return json.load(f)
        return {}

    def save_recovery_status(self, status):
        with open(self.recovery_file, 'w') as f:
            json.dump(status, f, indent=4)

    def recover_cleanup(self):
        self.recovery_file.unlink()

class recovery_Fabius:
    """
    Manages the recovery logic for different stages of the data generation pipeline.
    """
    def __init__(self):
        self.rec_Alma = compression_Alma()

    def check_stage_status(self, stage_name: str, current_rar: Path, required_deps: List[Tuple[Path, Path]] = None) -> bool:
        """
        Checks if a stage is complete. If incomplete, ensures all required precursor 
        temporary files exist for execution, decompressing them if necessary.
        
        Args:
            stage_name: Name of the current stage.
            current_rar: Path to the current stage's output archive.
            required_deps: List of tuples containing (dependency_rar_path, dependency_temp_dir).
            
        Returns:
            bool: True if the stage can be skipped, False if it needs to run.
        """
        # If the current stage is already complete, skip it.
        # Decompression of this stage's archive will be handled by downstream stages if they require it.
        if current_rar.exists():
            print(f"--- Stage '{stage_name}' archive '{current_rar.name}' exists. Skipping execution. ---")
            return True
            
        # Current stage needs to run. Ensure all required precursor files are available.
        if required_deps:
            for dep_rar, dep_temp_dir in required_deps:
                # Check if the required temp data is already available (pulled by a previous stage)
                if not dep_temp_dir.exists() or not os.listdir(dep_temp_dir):
                    if dep_rar.exists():
                        print(f"Precursor temporary files missing for '{stage_name}'. Restoring from '{dep_rar.name}'...")
                        self.rec_Alma.decompress_archive(dep_rar, dep_temp_dir)
                    else:
                        print(f"[WARNING] Required precursor archive '{dep_rar.name}' not found. Stage '{stage_name}' may fail.")

        print(f"Stage '{stage_name}' has not been completed. Starting process...")
        return False