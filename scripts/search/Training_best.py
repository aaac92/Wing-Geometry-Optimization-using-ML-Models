import csv

def find_row_with_max_value(file_path, column_index):
    """
    Finds and returns the row with the maximum value in a specified column of a CSV file.

    Args:
        file_path (str): The path to the CSV file.
        column_index (int): The zero-based index of the column to check for the max value.

    Returns:
        list: The row with the highest value in the specified column, or None if an error occurs.
    """
    max_value = -float('inf')  # Initialize with a very small number
    row_with_max_value = None
    
    try:
        with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            
            # Skip header row if it exists
            # next(reader, None) 
            # Note: Commented out for now. If your file has a header, uncomment the line above.

            for row in reader:
                try:
                    # Ensure the row has enough columns
                    if len(row) > column_index:
                        # Convert the value in the specified column to a float for comparison
                        current_value = float(row[column_index])
                        
                        # Check if this value is the new maximum
                        if current_value > max_value:
                            max_value = current_value
                            row_with_max_value = row
                except (ValueError, IndexError) as e:
                    # This will catch rows where the column isn't a number or is missing
                    # You can uncomment the print statement below for debugging purposes
                    # print(f"Could not process row: {row}. Error: {e}")
                    continue

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
        
    return row_with_max_value

# --- Configuration ---
# The user provided the filename 'Training_File_[1605_DP_Test_14].csv'
CSV_FILE_PATH = r'P:\avl_tmp\Training CSV\Inter_V2.1 Files\Training_File_[1603_data_points].csv'
# Column 7 corresponds to index 6 (since Python is 0-indexed)
COLUMN_TO_CHECK = 6 

# --- Execution ---
if __name__ == "__main__":
    # Find the row with the highest value
    result_row = find_row_with_max_value(CSV_FILE_PATH, COLUMN_TO_CHECK)

    # Display the result
    if result_row:
        print("Found row with the highest value in column 7:")
        print("---------------------------------------------")
        print(result_row)
        # You can also print specific values from the row like this:
        # print(f"Value from Column 1: {result_row[0]}")
        # print(f"Value from Column 2: {result_row[1]}")
        # ...and so on.
    else:
        print("Could not find the row or the file/column is invalid.")