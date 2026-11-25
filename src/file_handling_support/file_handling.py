import os
import subprocess
import sys
from pathlib import Path

def get_all_files_from(folder_path, extension="*", names_only=False):
        files = list(folder_path.glob(f"*.{extension}")) if extension != "*" else list(folder_path.glob("*"))

        if names_only:
            return [file.name for file in files]
        else:
            return files
        
class compression_Alma:
    def __init__(self, rar_executable: str = r"C:\Program Files\WinRAR\Rar.exe"):
        '''
        Initializes the handler with the path to the RAR executable.
        '''
        if sys.platform != "win32":
            # For Linux/macOS, it's typically 'rar'
            self.rar_executable = "rar"
        else:
            self.rar_executable = rar_executable
    
    def compress_folder(self, source_folder_path: Path, dest_folder_path: Path, rar_filename: str, verbose: bool = False):
        """
        Compresses a folder into a .rar file.
        """
        if not rar_filename.endswith('.rar'):
            rar_filename += '.rar'
        
        output_rar_path = dest_folder_path / rar_filename
        
        if not source_folder_path.is_dir():
            if verbose:
                print(f"Error: Source folder not found at '{source_folder_path}'")
            return

        dest_folder_path.mkdir(exist_ok=True)
        
        command = [self.rar_executable, 'a', '-r', str(output_rar_path), '.']

        try:
            if verbose:
                print(f"Compressing '{source_folder_path.name}' to '{output_rar_path.name}'...")
            subprocess.run(command, check=True, capture_output=True, text=True, cwd=source_folder_path)
            if verbose:
                print(f"Success! Archive '{rar_filename}' created.")
        except FileNotFoundError:
            if verbose:
                print(f"Error: '{self.rar_executable}' not found. Make sure WinRAR is installed and its path is correct.")
        except subprocess.CalledProcessError as e:
            if verbose:
                print(f"Error during compression:\n{e.stderr}")

    def extract_file(self, rar_path: Path, member: str, destination_dir: Path):
        """
        Extracts one file from the root of a .rar archive. Raises RuntimeError if it could not be extracted.
        """
        destination_dir.mkdir(parents=True, exist_ok=True)
        command = [self.rar_executable, 'e', '-o+', str(rar_path), member, str(destination_dir) + os.sep]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0 or not (destination_dir / member).exists():
            raise RuntimeError(f"Could not extract {member} from {rar_path}:\n{result.stdout}\n{result.stderr}")

    def decompress_archive(self, rar_path: Path, destination_dir: Path, verbose: bool = False):
        """
        Decompresses a .rar file into a destination folder.
        """
        if not rar_path.exists():
            if verbose:
                print(f"Error: Archive not found at '{rar_path}'")
            return

        destination_dir.mkdir(exist_ok=True)
        
        # 'x' is the command to extract with full paths
        # '-o+' ensures any existing files are overwritten without prompting
        command = [self.rar_executable, 'x', '-o+', str(rar_path), str(destination_dir)]

        try:
            if verbose:
                print(f"Decompressing '{rar_path.name}' to '{destination_dir.name}'...")
            subprocess.run(command, check=True, capture_output=True, text=True)
            if verbose:
                print("Decompression successful.")
        except FileNotFoundError:
            if verbose:
                print(f"Error: '{self.rar_executable}' not found. Make sure WinRAR is installed and its path is correct.")
        except subprocess.CalledProcessError as e:
            if verbose:
                print(f"Error during decompression:\n{e.stderr}")