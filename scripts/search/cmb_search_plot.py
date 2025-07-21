import os
import numpy as np
import matplotlib.pyplot as plt
from search_for_cmb import find_matching_files

def plot_airfoils(folder_path, matched_files):
    """
    Plots the geometry of one or more airfoils from their .dat files.

    Args:
        folder_path (str): The path to the folder containing the airfoil files.
        matched_files (list): A list of .dat filenames to plot.
    """
    if not matched_files:
        print("No matching airfoil files to plot.")
        return

    # --- Plotting Setup ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    print("--- Matched Airfoil Files ---")
    
    # --- Read and Plot Each Airfoil ---
    for filename in matched_files:
        file_path = os.path.join(folder_path, filename)
        
        try:
            # Read header
            with open(file_path, 'r') as f:
                header = f.readline().strip()
            
            # Print file information
            print(f"\nFile: {filename}")
            print(f"Header: {header}")

            # Load coordinates, skipping the header row
            coords = np.loadtxt(file_path, skiprows=1)
            x, y = coords[:, 0], coords[:, 1]

            # Plot the airfoil
            ax.plot(x, y, marker='o', linestyle='-', markersize=3, label=filename.replace('.dat', ''))

        except Exception as e:
            print(f"Could not read or plot {filename}. Error: {e}")

    # --- Final Plot Adjustments ---
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X-coordinate (Chord)')
    ax.set_ylabel('Y-coordinate (Thickness)')
    ax.set_title('Comparison of Airfoil Geometries')
    ax.legend()
    ax.grid(True)
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":

    folder = r'P:\xfoil_tmp\generated_airfoils'  # Replace this with your actual path
    values = [-2, -5, -1, 2, 2, 2]

    # Find the matching files using the imported function
    found_files = find_matching_files(folder, values)
    
    # Plot the found airfoils
    plot_airfoils(folder, found_files)