import csv
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import json
import numpy as np
from slugify import slugify
from utils import find_most_similar_lab_spec

# Configure logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def plot(csv_file, lab_names=None):
    if not lab_names:
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
            logging.error(f"{lab_name} - units are not the same for all values: " + json.dumps(list(set(units))))
            continue

        lab_spec = find_most_similar_lab_spec(lab_name)
        if not lab_spec:
            logging.error(f"{lab_name} - no lab spec found")
            continue

        units = lab_spec['units']
        if not unit in units: 
            logging.error(f"{lab_name} - unit {unit} is not defined in lab_specs.json")
            continue
        
        unit_range = lab_spec['units'][unit]

        # Sort the data by date
        sorted_data = sorted(zip(dates, values))
        dates, values = zip(*sorted_data)

        # Convert dates to numerical format for regression
        dates_numeric = [date.toordinal() for date in dates]

        # Calculate linear regression
        slope, intercept = np.polyfit(dates_numeric, values, 1)
        regression_line = [slope * x + intercept for x in dates_numeric]

        # Adjusted shading for min/max value range
        plt.figure(figsize=(10, 6))
        plt.plot(dates, values, marker='o', label='Data')
        for i in range(len(dates)):
            plt.text(dates[i], values[i], f'{values[i]:.2f}', fontsize=8, ha='right', va='bottom')
        plt.plot(dates, regression_line, label='Linear Regression', color='red')

        # Define a default lower and upper bound for shading if min or max is None
        default_lower_bound = min(values)  # Default to the minimum value in your dataset
        default_upper_bound = max(values)  # Default to the maximum value in your dataset

        # Conditionally set the lower and upper bounds for shading
        lower_bound = unit_range.get('min', default_lower_bound)
        upper_bound = unit_range.get('max', default_upper_bound)

        # Shading acceptable value range with checks to prevent TypeError
        if lower_bound is not None and upper_bound is not None:
            plt.fill_between(dates, lower_bound, upper_bound, color='green', alpha=0.1, label='Acceptable Range')
        elif lower_bound is not None:
            plt.fill_between(dates, lower_bound, default_upper_bound, color='green', alpha=0.1, label='Above Min Value')
        elif upper_bound is not None:
            plt.fill_between(dates, default_lower_bound, upper_bound, color='green', alpha=0.1, label='Below Max Value')

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
lab_names = [
    "Eritrograma - Eritrócitos",
    "Leucograma - Leucócitos",
    "Eritrograma - Velocidade de Sedimentação - 1a Hora",
    "Fator Reumatoide / Teste RA",
    "Hormona Tiro-Estimulante / TSH",
    "Triiodotironina Livre / T3 Livre",
    "Tiroxina Livre / T4 Livre",
    "Triiodotironina Total / T3 Total",
    "Vitamina D3 / 25-Hidroxicolecalciferol / Calcidiol", 
    "Ferritina",
    "Ferro",
    "Tiroxina Total / T4 Total"
]
plot("outputs/labs_results.final.csv", lab_names)

