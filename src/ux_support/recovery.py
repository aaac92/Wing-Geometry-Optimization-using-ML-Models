import json
import os
import subprocess
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

    def check_stage_status(self, stage_name: str, rar_path: Path, temp_dir: Path) -> bool:
        """
        Checks if a stage is complete. If so, it restores files from the archive if needed.
        
        Returns:
            bool: True if the stage can be skipped, False if it needs to run.
        """
        if not rar_path.exists():
            # The final archive doesn't exist, so the stage must run.
            print(f"Stage '{stage_name}' has not been completed before. Starting process...")
            return False

        # The archive exists. Now check if the temp files are present.
        if not temp_dir.exists() or not os.listdir(temp_dir):
            # Temp dir is missing, so we must decompress the archive to proceed.
            print(f"--- Stage '{stage_name}' is complete. Restoring files from '{rar_path.name}' for next stage. ---")
            self.rec_Alma.decompress_archive(rar_path, temp_dir)
        else:
            # Files are present from a previous run.
            print(f"--- Stage '{stage_name}' is complete. Using existing temporary files. ---")

        # In both cases where the .rar exists, we can skip the main generation loop for this stage.
        return True