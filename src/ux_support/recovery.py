import json
import os
from pathlib import Path
from file_handling_support import compression_Alma

class recovery_json:
    def __init__(self):
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

    def check_stage_status(self, stage_name: str, rar_path: Path, temp_dir: Path, prev_rar_path: Path = None, prev_temp_dir: Path = None, next_rar_path: Path = None) -> bool:
        """
        Checks if a stage is complete. If so, restores files from the archive if needed for the NEXT stage.
        If not complete, ensures the previous stage's temporary files exist for execution.
        
        Returns:
            bool: True if the stage can be skipped, False if it needs to run.
        """
        if rar_path.exists():
            if next_rar_path and next_rar_path.exists():
                print(f"--- Stage '{stage_name}' is complete, and next stage archive '{next_rar_path.name}' exists. Skipping decompression. ---")
                return True
            elif not next_rar_path:
                print(f"--- Stage '{stage_name}' is complete. This is the final stage. Process is complete. ---")
                return True
            
            # Current stage archive exists, but next stage does not. Check if temp files are present for downstream processes.
            if not temp_dir.exists() or not os.listdir(temp_dir):
                print(f"--- Stage '{stage_name}' is complete. Restoring files from '{rar_path.name}' for next stage. ---")
                self.rec_Alma.decompress_archive(rar_path, temp_dir)
            else:
                print(f"--- Stage '{stage_name}' is complete. Using existing temporary files for next stage. ---")
            return True

        # Current stage is incomplete. Check if we need to restore precursor files.
        if prev_rar_path and prev_temp_dir:
            if not prev_temp_dir.exists() or not os.listdir(prev_temp_dir):
                if prev_rar_path.exists():
                    print(f"Precursor temporary files missing for '{stage_name}'. Restoring from '{prev_rar_path.name}'...")
                    self.rec_Alma.decompress_archive(prev_rar_path, prev_temp_dir)
                else:
                    print(f"[WARNING] Precursor archive '{prev_rar_path.name}' not found. Stage '{stage_name}' may fail.")

        print(f"Stage '{stage_name}' has not been completed before. Starting process...")
        return False