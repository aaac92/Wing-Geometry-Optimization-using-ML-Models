import pandas as pd

search_values = [-2, -5, -1, 2, 2, 2]
csv_file_name = r'P:\avl_tmp\Training CSV\Inter_V2 Files\LGBM\Training_File_[1618_DP_Test_14_Inter_V2].csv'

search_columns = [0, 1, 2, 3, 4, 5]
try:
    # Read the CSV file.
    # header=None assumes no header row.
    # skipinitialspace=True is KEY: it removes leading whitespace from values.
    df = pd.read_csv(csv_file_name, header=None, skipinitialspace=True)

    # --- REVISED SEARCH LOGIC ---
    # This is more direct and robust.
    # It converts the DataFrame columns to integers for the comparison.
    condition = (df[0].astype(int) == search_values[0]) & \
                (df[1].astype(int) == search_values[1]) & \
                (df[2].astype(int) == search_values[2]) & \
                (df[3].astype(int) == search_values[3]) & \
                (df[4].astype(int) == search_values[4]) & \
                (df[5].astype(int) == search_values[5])

    # Filter the DataFrame based on the condition
    found_row = df[condition]

    # Print the result
    if not found_row.empty:
        print("Match found! Here is the data:")
        # We print the found row as a simple list
        for index, row in found_row.iterrows():
            print(row.values.tolist())
    else:
        print("No matching row found for the values:", search_values)

except FileNotFoundError:
    print(f"Error: The file '{csv_file_name}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
    print("This might happen if a column you are searching contains non-numeric text.")