import os
import re

def find_matching_files(folder_path, target_values):
    """
    Finds airfoil .dat files in a folder that match an ordered list of target 
    values extracted from the header.
    """
    matched_files = []

    # The target_values are kept as a list to preserve order.
    # No need to convert to a set.

    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found at '{folder_path}'")
        return matched_files

    for filename in os.listdir(folder_path):
        if filename.endswith('.dat'):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r') as f:
                    header = f.readline().strip()
                    
                    match = re.search(r'Airfoil_([-\d\._]+)_\[NACA', header)
                    if match:
                        numbers_str = match.group(1).split('_')
                        try:
                            # Convert extracted numbers to a list of ints
                            numbers = [int(float(x)) for x in numbers_str]
                            # Compare the lists directly to enforce order.
                            if numbers == target_values:
                                matched_files.append(filename)
                        except ValueError:
                            # Silently ignore files with non-convertible numbers
                            pass
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    return matched_files

# Example usage
folder = r'P:\xfoil_tmp\generated_airfoils'  # Replace this with your actual path
values = [-2, -5, -1, 2, 2, 2]
results = find_matching_files(folder, values)
print("Matching files:")
for f in results:
    print(f)
