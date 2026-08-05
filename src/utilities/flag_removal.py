import argparse
import math
import os
import sys
import time
from tqdm import tqdm

COMPLETION_FLAG = "XFOIL process completed successfully"
MODES = ["nuke", "slick", "shakedown"]

def parse_arguments():
    """
    Parses command-line arguments to configure the execution of the utility.
    """
    parser = argparse.ArgumentParser(
        description="Utility to remove incorrect completion flags from XFOIL polar files.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "--dir",
        required=True,
        help="Directory containing the polar files."
    )
    
    parser.add_argument(
        "--last-alpha",
        required=True,
        type=float,
        help="Last angle of the alpha sequence used during the simulation."
    )
    
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="slick",
        help=(
            "Processing mode to use (default: slick).\n"
            "  nuke      : Blindly remove the completion flag if present.\n"
            "  slick     : Remove the flag unless the preceding line starts with --last-alpha.\n"
            "  shakedown : Perform slick logic but do not modify any files (dry run)."
        )
    )
    
    return parser.parse_args()

def find_last_meaningful_line(lines, start_index=None):
    """
    Scans a list of lines backwards to find the index and content of the 
    last non-empty, non-whitespace line.
    """
    if start_index is None:
        start_index = len(lines) - 1
        
    for i in range(start_index, -1, -1):
        if lines[i].strip():
            return i, lines[i]
            
    return -1, None

def check_alpha_match(line, target_alpha, tolerance=1e-4):
    """
    Extracts the first numeric value from a string (assuming space-separated columns)
    and checks if it matches the target alpha value within a tight tolerance.
    """
    tokens = line.strip().split()
    if not tokens:
        return False
        
    try:
        alpha_value = float(tokens[0])
        return math.isclose(alpha_value, target_alpha, abs_tol=tolerance)
    except ValueError:
        return False

def process_directory(directory_path, target_alpha, mode):
    """
    Iterates through all files in a flat directory and applies the specified
    flag removal logic, tracking progress and errors along the way.
    """
    if not os.path.isdir(directory_path):
        print(f"FATAL ERROR: The directory '{directory_path}' does not exist or is not a directory.")
        sys.exit(1)

    # Filter out directories, yielding only file paths. 
    # Avoiding recursion to strictly adhere to processing the specified folder.
    file_entries = [entry for entry in os.scandir(directory_path) if entry.is_file()]
    total_files = len(file_entries)
    
    if total_files == 0:
        print(f"No files found in '{directory_path}'. Exiting.")
        sys.exit(0)

    # Initialize tracking metrics
    stats = {
        "modified": 0,
        "skipped": 0,
        "preserved": 0
    }
    recoverable_errors = []

    progress_bar = tqdm(
        total=total_files,
        desc=f"Scanning Polars ({mode.upper()})",
        unit="file",
        dynamic_ncols=True
    )

    for entry in file_entries:
        try:
            # Using newline='' guarantees that we read the exact original line endings
            # (LF or CRLF) without Python translating them automatically.
            with open(entry.path, 'r', newline='') as f:
                lines = f.readlines()

            flag_idx, flag_line = find_last_meaningful_line(lines)
            
            if flag_idx != -1 and flag_line.strip() == COMPLETION_FLAG:
                should_remove_flag = False
                
                if mode == "nuke":
                    should_remove_flag = True
                else:
                    # 'slick' or 'shakedown' modes require checking the preceding line
                    prev_idx, prev_line = find_last_meaningful_line(lines, flag_idx - 1)
                    
                    if prev_idx == -1:
                        # Only the flag exists in the file, nothing to preserve.
                        should_remove_flag = True
                    else:
                        if check_alpha_match(prev_line, target_alpha):
                            should_remove_flag = False
                        else:
                            should_remove_flag = True
                
                if should_remove_flag:
                    stats["modified"] += 1
                    if mode != "shakedown":
                        # Removing only the specific flag line to preserve all original 
                        # whitespace and trailing empty lines exactly as they were.
                        del lines[flag_idx]
                        
                        # Writing back with newline='' ensures original line endings are kept
                        with open(entry.path, 'w', newline='') as f:
                            f.writelines(lines)
                else:
                    stats["preserved"] += 1
            else:
                stats["skipped"] += 1

        except Exception as e:
            # Capture file-specific errors (e.g., encoding, permissions) without 
            # interrupting the main scanning pipeline.
            recoverable_errors.append((entry.name, str(e)))

        # Update the live progress counters
        progress_bar.set_postfix(
            mod=stats["modified"], 
            pres=stats["preserved"], 
            err=len(recoverable_errors)
        )
        progress_bar.update(1)

    progress_bar.close()
    return total_files, stats, recoverable_errors

def print_summary(start_time, total_files, stats, errors, mode):
    """
    Outputs a clean, consolidated report after all processing has finished.
    """
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*50)
    print(" EXECUTION SUMMARY")
    print("="*50)
    
    print(f"Total files scanned : {total_files}")
    if mode == "shakedown":
        print(f"Would be modified   : {stats['modified']}")
    else:
        print(f"Files modified      : {stats['modified']}")
        
    print(f"Files skipped       : {stats['skipped']} (No completion flag found)")
    print(f"Flags preserved     : {stats['preserved']} (Valid completed simulations)")
    print(f"Recoverable errors  : {len(errors)}")
    print(f"Total execution time: {elapsed_time:.2f} seconds")
    
    if mode == "shakedown":
        print("-" * 50)
        print(" SHAKEDOWN MODE ACTIVE: No files were actually modified.")
        
    if errors:
        print("\n" + "="*50)
        print(" ERROR LOG")
        print("="*50)
        for filename, error_msg in errors:
            print(f"- {filename}: {error_msg}")
    else:
        print("\nProcessing completed successfully with no file errors.")

def main():
    args = parse_arguments()
    
    start_time = time.time()
    total_files, stats, errors = process_directory(args.dir, args.last_alpha, args.mode)
    print_summary(start_time, total_files, stats, errors, args.mode)

if __name__ == "__main__":
    main()