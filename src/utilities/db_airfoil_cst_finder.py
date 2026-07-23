import os
import re
import struct
import shutil
import subprocess
from pathlib import Path
from file_handling_support import compression_Alma

# Get the path to the parent directory.
utilities_dir = Path(__file__).resolve().parent
src_dir = utilities_dir.parent
project_root = src_dir.parent

def read_dat_header(file_path: str, expected_count: int):
    """
    Reads the first `num_values` floats from a .dat file header.
    Assumes float32 (4 bytes each).
    """
    values = []
    try:
        with open(file_path, "rb") as f:
            raw = f.read(1024)  # just read the first KB, header should fit here
            text = raw.decode(errors="ignore")

        # Try to extract floats from text
        match = re.search(r"\[([^\]]+)\]", text)
        if match:
            numbers = match.group(1).split(",")
            values = [float(n.strip()) for n in numbers if n.strip() != ""]
            return values[:expected_count]

        # If no text match, try binary floats
        with open(file_path, "rb") as f:
            data = f.read(expected_count * 4)
            values = list(struct.unpack(f"{expected_count}f", data))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

    return values

def search_sequence(values, target, tol=1e-4):
    """
    Compare two float sequences with tolerance.
    """
    if len(values) < len(target):
        return False
    
    return all(abs(v - t) <= tol for v, t in zip(values, target))

def process_rar(rar_path: Path, target_sequence: list):
    # Create hidden temp folder in C:\
    temp_dir = Path(os.path.join("C:\\", "_rar_temp"))
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    subprocess.call(["attrib", "+H", temp_dir])

    # Decompress using your class
    rar_Alma = compression_Alma()
    rar_Alma.decompress_archive(rar_path, temp_dir)

    # Scan all .dat files
    found_files = []
    for root, _, files in os.walk(temp_dir):
        for file in files:
            if file.lower().endswith(".dat"):
                file_path = os.path.join(root, file)
                header_values = read_dat_header(file_path, expected_count=len(target_sequence))
                if search_sequence(header_values, target_sequence):
                    found_files.append(file_path)

    # Clean up temp directory
    shutil.rmtree(temp_dir, ignore_errors=True)

    return found_files

if __name__ == "__main__":
    '''
    This script is capable of finding an aifoil based on it's CST coefficients within a database.
    '''
    # Airfoil.rar file
    airfoils_dir = project_root / "data" / "dataset_build_data" / "airfoils"
    rar_path = airfoils_dir / "airfoils_dataset_16.rar"
    # Airfoil CST coefficients
    target_sequence = [0, 0.125, 0.0833, 0, 0, 0, 0.0833, 0, 0, 0]

    result = process_rar(rar_path, target_sequence)
    if result:
        print("Sequence found in files:")
        for f in result:
            print(f)
    else:
        print("No matching sequence found.")
