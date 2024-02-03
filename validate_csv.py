import pandas as pd

def identify_lab_names_with_different_units(file_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Group by lab names and check for different units within each group
    labs_with_different_units = {}
    for lab_name, group in df.groupby('name'):
        unique_units = group['unit'].unique()
        if len(unique_units) > 1:
            labs_with_different_units[lab_name] = unique_units
    
    # Print out labs with different units, if any
    if labs_with_different_units:
        print("Labs with different units:")
        for lab_name, units in labs_with_different_units.items():
            units = [str(unit) for unit in units]
            print(f"{lab_name}: {', '.join(units)}")
    else:
        print("No labs with different units found.")

# Replace 'your_file_path.csv' with the path to your actual CSV file
file_path = 'outputs/labs_results.final.csv'
identify_lab_names_with_different_units(file_path)
