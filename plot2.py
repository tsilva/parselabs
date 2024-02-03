import csv
from datetime import datetime
import matplotlib.pyplot as plt
import json
import numpy as np
from slugify import slugify
from utils import find_most_similar_lab_spec

def plot(csv_file):
    lab_names = set()  # Use a set to collect unique lab names

    # First, collect all unique lab names from the CSV file
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            lab_names.add(row['name'])

    # Iterate over each lab name to plot
    for lab_name in lab_names:
        dates = []
        values = []
        unit = None
        units = []

        with open(csv_file, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['name'] == lab_name:
                    dates.append(datetime.strptime(row['date'], '%Y-%m-%d'))  # Adjust format if needed
                    values.append(float(row['value']))
                    unit = row['unit']
                    units.append(unit)

        if len(list(set(units))) != 1:
            print(f"{lab_name} - units are not the same for all values: " + json.dumps(list(set(units))))
            continue

        lab_spec = find_most_similar_lab_spec(lab_name)
        units = lab_spec['units']
        if not unit in units: raise Exception(f"{lab_name} - unit {unit} is not defined in lab_specs.json")
        unit_range = lab_spec['units'][unit]

        # Sort the data by date
        sorted_data = sorted(zip(dates, values))
        dates, values = zip(*sorted_data)

        # Convert dates to numerical format for regression
        dates_numeric = [date.toordinal() for date in dates]

        # Calculate linear regression
        slope, intercept = np.polyfit(dates_numeric, values, 1)
        regression_line = [slope * x + intercept for x in dates_numeric]

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(dates, values, marker='o', label='Data')
        for i in range(len(dates)):
            plt.text(dates[i], values[i], f'{values[i]:.2f}', fontsize=8, ha='right', va='bottom')
        plt.plot(dates, regression_line, label='Linear Regression', color='red')
        if unit_range['min'] is not None:
            plt.axhline(y=unit_range['min'], color='green', linestyle='--', label='Min Value')
        if unit_range['max'] is not None:
            plt.axhline(y=unit_range['max'], color='green', linestyle='--', label='Max Value')

        plt.title(lab_name)
        plt.xlabel('Date')
        plt.ylabel(f"Value ({unit})")
        plt.grid(True)
        plt.legend()

        # Save plot to a file
        image_name = slugify(lab_name)
        plt.savefig(f"outputs/plot_{image_name}.png")
        plt.close()

# Replace the lab_names list with a call to plot all labs in the CSV
plot("outputs/labs_results.final.csv")
