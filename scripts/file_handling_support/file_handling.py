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
        
def create_rar_from_folder(source_folder_path, dest_folder_path, rar_filename, verbose=False, rar_executable = r"C:\Program Files\WinRAR\Rar.exe"):
    """
    Compresses a folder into a .rar file and saves it to a destination folder.

    Args:
        source_folder_path (str): The absolute path of the folder to compress.
        dest_folder_path (str): The absolute path of the folder to save the .rar file in.
        rar_filename (str): The desired name for the output .rar file (without extension).
        verbose, optional (bool): Toggles prints for information.
        rar_executable, optional (rstr): Path to Rar.exe in a raw string.
    """
    # Ensure the RAR command-line tool is available
    # For Windows, the executable is typically 'Rar.exe'
    
    if sys.platform != "win32":
        # For Linux/macOS, it's typically 'rar'
        rar_executable = "rar"
    
    # Add the '.rar' extension to the filename
    if not rar_filename.endswith('.rar'):
        rar_filename += '.rar'

    # Construct the full, absolute path for the output .rar file
    output_rar_path = os.path.join(dest_folder_path, rar_filename)
    if verbose:
        print(f"Source folder: {source_folder_path}")
        print(f"Output path: {output_rar_path}")

    # Check if the source folder exists
    if not os.path.isdir(source_folder_path):
        if verbose:
            print(f"Error: Source folder not found at '{source_folder_path}'")
        return

    # Create the destination directory if it doesn't exist
    os.makedirs(dest_folder_path, exist_ok=True)
    
    # Construct the command for the command-line tool
    # 'a' is the command to add files to an archive
    # '-r' tells it to recurse into subdirectories
    command = [
        rar_executable,
        'a',
        '-r',
        output_rar_path,
        '.'
    ]

    try:
        if verbose:
            print("Compressing folder... Please wait.")
        # Execute the command
        # 'capture_output=True' and 'text=True' help capture the output for debugging
        subprocess.run(command, check=True, capture_output=True, text=True, cwd=source_folder_path)
        if verbose:
            print(f"Success! Archive '{rar_filename}' created in '{dest_folder_path}'")
            # print("Output from Rar:\n", result.stdout) # Uncomment for detailed output
    except FileNotFoundError:
        if verbose:
            print(f"Error: '{rar_executable}' not found. Make sure WinRAR is installed and in your system's PATH.")
    except subprocess.CalledProcessError as e:
        if verbose:
            print(f"Error during compression:")
            print(e.stderr) # Print the error output from the rar command

