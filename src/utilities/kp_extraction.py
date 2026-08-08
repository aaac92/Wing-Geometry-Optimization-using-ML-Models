import argparse
import sys
import re
from pathlib import Path
from simulation_support import Key_Point_Extraction

def extract_polar_metadata(polar_path: Path) -> dict:
    """Extracts aerodynamic flow regime metadata from the XFOIL polar file header."""
    metadata = {'name': 'Unknown', 'mach': '0.0', 're': '0.0', 'ncrit': '0.0'}
    
    with open(polar_path, "r", encoding="utf-8") as f:
        # Limit search to the first 15 lines to avoid scanning the entire data array
        for _ in range(15):
            line = f.readline()
            if not line:
                break
            
            # Extract airfoil name
            if "Calculated polar for:" in line:
                metadata['name'] = line.split("Calculated polar for:")[1].strip()
            
            # Extract flow regime parameters
            if "Mach =" in line and "Re =" in line:
                mach_match = re.search(r"Mach\s*=\s*([0-9\.]+)", line)
                re_match = re.search(r"Re\s*=\s*([0-9\.\s]+e\s*[0-9]+|[0-9\.]+)", line)
                ncrit_match = re.search(r"Ncrit\s*=\s*([0-9\.]+)", line)
                
                if mach_match:
                    metadata['mach'] = mach_match.group(1).strip()
                if ncrit_match:
                    metadata['ncrit'] = ncrit_match.group(1).strip()
                if re_match:
                    # Strip internal whitespace from scientific notation
                    raw_re = re_match.group(1).replace(" ", "")
                    try:
                        re_float = float(raw_re)
                        metadata['re'] = f"{re_float:g}"  # Formats to standard float representation
                    except ValueError:
                        metadata['re'] = raw_re
                        
    return metadata

def write_key_points_file(result_dict: dict, output_path: Path, meta: dict):
    """Writes the extracted key points and regime metadata to a text file."""
    with open(output_path, "w", encoding="utf-8") as f:
        # Line 1: Flow regime metadata header
        f.write(f"{meta['name']} Mach={meta['mach']} Re={meta['re']} Ncrit={meta['ncrit']}\n")
        
        # Line 2: Data structure header
        f.write("CLmin CDCLmin CLCDmin CDmin CLmax CDCLmax\n")
        
        # Line 3: Extracted aerodynamic values in raw floating-point format
        cl_min = result_dict['min_cl_point']['CL']
        cd_cl_min = result_dict['min_cl_point']['CD']
        
        cl_cd_min = result_dict['min_cd_point']['CL']
        cd_min = result_dict['min_cd_point']['CD']
        
        cl_max = result_dict['max_cl_point']['CL']
        cd_cl_max = result_dict['max_cl_point']['CD']
        
        f.write(f"{cl_min} {cd_cl_min} {cl_cd_min} {cd_min} {cl_max} {cd_cl_max}\n")

def main():
    # Resolve project root dynamically to establish default paths
    utilities_dir = Path(__file__).resolve().parent
    src_dir = utilities_dir.parent
    project_root = src_dir.parent
    
    default_polar_dir = project_root / "data" / "support" / "naca_polars"
    default_output_dir = project_root / "data" / "support" / "naca_kp"

    parser = argparse.ArgumentParser(description="Extract aerodynamic key points from XFOIL polar files.")
    parser.add_argument(
        "--polar_dir",
        type=str,
        default=str(default_polar_dir),
        help="Target directory containing the source polar files."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(default_output_dir),
        help="Target directory to save the generated key point files."
    )
    parser.add_argument(
        "--save_log",
        action="store_true",
        help="Generate a processing log file in the output directory."
    )
    args = parser.parse_args()

    polar_dir = Path(args.polar_dir)
    output_dir = Path(args.output_dir)
    
    if not polar_dir.is_dir():
        print(f"Error: Source directory '{polar_dir}' does not exist.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Search for any .txt file containing the word 'polar'
    polar_files = list(polar_dir.glob("*polar*.txt"))
    files_to_process = []
    metadata_map = {}

    # Extract metadata prior to the extraction queue to construct target filenames
    for polar_path in polar_files:
        meta = extract_polar_metadata(polar_path)
        
        # Extract base name dynamically without standardizing on a single string position
        base_name = polar_path.stem.replace("_polar", "").replace("polar_", "").replace("polar", "")
        
        # Append Mach and Re parameters to establish the target KP file name
        kp_file_name = f"{base_name}_M{meta['mach']}_RE{meta['re']}.txt"
        kp_path = output_dir / kp_file_name

        if kp_path.exists():
            print(f"Skipped: '{polar_path.name}' (Target file '{kp_file_name}' already exists).")
        else:
            files_to_process.append(str(polar_path))
            # Store metadata dictionary utilizing the base filename as the key
            metadata_map[polar_path.name] = (meta, kp_file_name)

    if not files_to_process:
        print("No new polar files to process. Exiting.")
        sys.exit(0)

    print(f"\nFound {len(files_to_process)} file(s) to process.")
    extractor = Key_Point_Extraction(polar_file_paths=files_to_process, verbose=True)
    results = extractor.process_files()

    for result in results:
        polar_name = result['file_path']
        meta, kp_file_name = metadata_map[polar_name]
        output_path = output_dir / kp_file_name
        
        write_key_points_file(result, output_path, meta)

    if args.save_log:
        log_path = output_dir / "naca_key_point_extraction_log.txt"
        extractor.save_error_log(str(log_path))
        print(f"\nExecution finished. Output log generated at: {log_path.name}")
    else:
        print("\nExecution finished.")

if __name__ == "__main__":
    main()