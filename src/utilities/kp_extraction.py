import argparse
import sys
from pathlib import Path
from simulation_support.key_point_processing import Key_Point_Extraction, extract_polar_metadata, write_key_points_file

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