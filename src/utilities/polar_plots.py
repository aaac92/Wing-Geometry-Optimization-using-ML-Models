import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from simulation_support import Key_Point_Extraction
from file_handling_support import get_all_files_from

'''
Use this script to plot any polar file into it's corresponding polar curves. This will also calculate it's key points.
'''

# Get the path to the project root directory.
utilities_dir = Path(__file__).resolve().parent
src_dir = utilities_dir.parent
project_root = src_dir.parent

# Find desired polar file
#polar_file = "polar_Test_141377.txt"
polar_file = "polar_Dataset Best.txt"
polar_path = project_root / "data" / "singular_test_data" / "singular_polar_files"
polar_file_path = polar_path / polar_file
polars_abs_paths = get_all_files_from(polar_file_path)

# Read polar file
KPE = Key_Point_Extraction(
        polar_file_paths=polars_abs_paths
        )

alpha_list, cl_list, cd_list = KPE._polar_extract(polar_file_path)

# Convert cleaned string lists to float arrays
alpha_array = np.array([float(a) for a in alpha_list])
cl_array = np.array([float(cl) for cl in cl_list])
cd_array = np.array([float(cd) for cd in cd_list])

# Find critical points
low_cl_point, high_cl_point, min_cd_point = KPE._critical_point_extract(cl_array, cd_array)

# Display AVL-ready points
print("\n=== Key Points for AVL ===")
print(f"Low CL Point:   CL = {low_cl_point[0]:.4f}, CD = {low_cl_point[1]:.5f}")
print(f"Min CD Point:   CL = {min_cd_point[0]:.4f}, CD = {min_cd_point[1]:.5f}")
print(f"High CL Point:  CL = {high_cl_point[0]:.4f}, CD = {high_cl_point[1]:.5f}")

# Plot 1: CL vs CD
plt.figure(figsize=(6, 4))
plt.plot(cd_array, cl_array, 'b-', linewidth=1.5)
plt.xlabel("CD (Drag Coefficient)")
plt.ylabel("CL (Lift Coefficient)")
plt.title("CL vs CD")
plt.grid(True)
plt.tight_layout()

# Plot 2: CL vs Alpha
plt.figure(figsize=(6, 4))
plt.plot(alpha_array, cl_array, 'g-', linewidth=1.5)
plt.xlabel("Alpha (Angle of Attack in deg)")
plt.ylabel("CL (Lift Coefficient)")
plt.title("CL vs Alpha")
plt.grid(True)
plt.tight_layout()

# Plot 3: CD vs Alpha
plt.figure(figsize=(6, 4))
plt.plot(alpha_array, cd_array, 'r-', linewidth=1.5)
plt.xlabel("Alpha (Angle of Attack in deg)")
plt.ylabel("CD (Drag Coefficient)")
plt.title("CD vs Alpha")
plt.grid(True)
plt.tight_layout()

plt.show()