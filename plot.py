import csv
import json
import logging
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from slugify import slugify
from utils import find_most_similar_lab_spec, lab_spec_by_name

FINAL_CSV_PATH = "outputs/labs_results.final.csv"

# Configure logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def get_lab_names(csv_file):
    lab_names = set()
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader: lab_names.add(row['name'])
    return lab_names

def moving_average(values, window_size):
    """Calculate the moving average using a simple sliding window approach."""
    averages = []

    # Calculate the moving average using a sliding window
    for i in range(len(values)):
        if i < window_size:
            # Not enough data points yet for the window; use what's available
            averages.append(sum(values[:i+1]) / (i+1))
        else:
            window = values[i-window_size+1:i+1]
            averages.append(sum(window) / window_size)

    return averages

def plot_labs(csv_file, lab_name):
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
        return

    lab_spec = find_most_similar_lab_spec(lab_name)
    if not lab_spec:
        logging.error(f"{lab_name} - no lab spec found")
        return

    units = lab_spec['units']
    if not unit in units: 
        logging.error(f"{lab_name} - unit {unit} is not defined in lab_specs.json")
        return
    
    unit_range = lab_spec['units'][unit]

    # Sort the data by date
    sorted_data = sorted(zip(dates, values))
    dates, values = zip(*sorted_data)

    # Convert dates to numerical format for regression
    dates_numeric = [date.toordinal() for date in dates]

    # Calculate linear regression
    slope, intercept = np.polyfit(dates_numeric, values, 1)
    regression_line = [slope * x + intercept for x in dates_numeric]

    # Calculate moving average
    window_size = 5
    moving_avg = moving_average(values, window_size)

    # Adjusted shading for min/max value range
    plt.figure(figsize=(10, 6))
    plt.plot(dates, values, marker='o', label='Data')
    plt.plot(dates, moving_avg, label='Moving Average', color='orange', linestyle='--')
    for i in range(len(dates)): plt.text(dates[i], values[i], f'{values[i]:.2f}', fontsize=8, ha='right', va='bottom')
    plt.plot(dates, regression_line, label='Linear Regression', color='red', linestyle='--')

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

def normalize_values(values, value_range):
    """Normalize values to a 0-1 scale based on the given value range."""
    min_val, max_val = value_range
    # Prevent division by zero if min_val equals max_val
    if min_val == max_val:
        return [0 if v != min_val else 1 for v in values]
    return [(value - min_val) / (max_val - min_val) for value in values]

def linear_interpolate(x0, y0, x1, y1, x):
    """Perform linear interpolation for a value x given two data points (x0, y0) and (x1, y1)."""
    return y0 + (y1 - y0) * ((x - x0) / (x1 - x0))

def interpolate_missing_dates(dates, values, all_dates):
    """Interpolate missing values for dates not present in the dataset."""
    dates_ordinal = [date.toordinal() for date in dates]
    all_dates_ordinal = [date.toordinal() for date in all_dates]
    interpolated_values = []

    for target_date in all_dates_ordinal:
        if target_date in dates_ordinal:
            # If the date is already in the list, use the corresponding value without interpolation
            interpolated_values.append(values[dates_ordinal.index(target_date)])
        else:
            # Find the closest dates before and after the target date
            before_date = max([d for d in dates_ordinal if d < target_date], default=None)
            after_date = min([d for d in dates_ordinal if d > target_date], default=None)

            if before_date is not None and after_date is not None:
                # Perform linear interpolation
                before_value = values[dates_ordinal.index(before_date)]
                after_value = values[dates_ordinal.index(after_date)]
                interpolated_value = linear_interpolate(before_date, before_value, after_date, after_value, target_date)
                interpolated_values.append(interpolated_value)
            elif before_date is not None:
                # If there is no date after, use the value from the closest date before
                interpolated_values.append(values[dates_ordinal.index(before_date)])
            elif after_date is not None:
                # If there is no date before, use the value from the closest date after
                interpolated_values.append(values[dates_ordinal.index(after_date)])
            else:
                # If there are no dates before or after, it's an extrapolation scenario
                # Here you might want to handle the edge case differently, e.g., using the closest known value
                interpolated_values.append(values[0])  # or some other logic for extrapolation

    return interpolated_values

def plot_correlation(csv_file, lab_name1, lab_name2):
    # Load data
    data = {lab_name1: {'dates': [], 'values': [], "unit" : None}, lab_name2: {'dates': [], 'values': [], "unit" : None}}
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['name'] in data:
                data[row['name']]['unit'] = row["unit"]
                data[row['name']]['dates'].append(datetime.strptime(row['date'], '%Y-%m-%d'))
                data[row['name']]['values'].append(float(row['value']))

    # Normalize values based on lab specs
    for lab_name in [lab_name1, lab_name2]:
        lab_spec = lab_spec_by_name(lab_name)
        unit = data[lab_name]['unit']
        unit_range = lab_spec["units"][unit]
        value_range = (unit_range['min'], unit_range['max'])
        if value_range[0] >= value_range[1]: raise ValueError(f"Invalid value range for {lab_name}: {value_range}")
        data[lab_name]['normalized_values'] = normalize_values(data[lab_name]['values'], value_range)

    # Find superset of dates
    all_dates = sorted(set(data[lab_name1]['dates']) | set(data[lab_name2]['dates']))

    # Interpolate missing values for each lab result set
    for lab_name in [lab_name1, lab_name2]:
        data[lab_name]['interpolated_values'] = interpolate_missing_dates(
            data[lab_name]['dates'], data[lab_name]['normalized_values'], all_dates)

    # Plot
    plt.figure(figsize=(10, 6))
    for lab_name, color in zip([lab_name1, lab_name2], ['blue', 'red']):
        plt.plot(all_dates, data[lab_name]['interpolated_values'], marker='o', label=lab_name, color=color)

    plt.title(f"Correlation between {lab_name1} and {lab_name2}")
    plt.xlabel('Date')
    plt.ylabel("Normalized Value")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"outputs/plot_{slugify(lab_name1)}_and_{slugify(lab_name2)}.png")
    plt.close()

labs = [
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
    "Tiroxina Total / T4 Total",
    ["Hormona Tiro-Estimulante / TSH", "Tiroxina Livre / T4 Livre"]
]
for lab in labs:
    if isinstance(lab, list): plot_correlation(FINAL_CSV_PATH, lab[0], lab[1])
    else: plot_labs(FINAL_CSV_PATH, lab)