import csv
from datetime import datetime
import matplotlib.pyplot as plt
import json
import numpy as np
from slugify import slugify

def plot(csv_file, lab_name):
    dates = []
    values = []
    unit = None

    # Open and read the CSV file
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
        raise Exception(f"{lab_name} - units are not the same for all values: " + json.dumps(units))

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

    # Adding text next to each dot
    for i in range(len(dates)):
        plt.text(dates[i], values[i], f'{values[i]:.2f}', fontsize=8, ha='right', va='bottom')

    plt.plot(dates, regression_line, label='Linear Regression', color='red')

    unit = units[0]
    plt.title(lab_name)
    plt.xlabel('Date')
    plt.ylabel(f"Value ({unit})")
    plt.grid(True)
    plt.legend()

    # Save plot to a file
    image_name = slugify(lab_name)
    plt.savefig(f"outputs/plot_{image_name}.png")

    # Optionally, close the plot if not displayed
    plt.close()

lab_names = [
    "Hormona Tiro-Estimulante / TSH",
    "Triiodotironina Livre / T3 Livre",
    "Tiroxina Livre / T4 Livre"
    #"Tiroxina Total / T4 Total",
    #"Triiodotironina Total / T3 Total",
    #"Vitamina D3 / 25-Hidroxicolecalciferol / Calcidiol", 
    #"Ferritina"
]
for lab_name in lab_names:
    plot("outputs/labs_results.final.csv", lab_name)
