#!/usr/bin/env python3
"""Detailed analysis of $UNKNOWN$ values to identify fixes needed."""

import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv
import json

load_dotenv()

OUTPUT_PATH = os.getenv("OUTPUT_PATH")

# Read the main aggregated CSV
csv_path = Path(OUTPUT_PATH) / "all.csv"
df = pd.read_csv(csv_path)

print("="*80)
print("UNKNOWN LAB NAMES - Detailed Analysis")
print("="*80)

unknown_names = df[df['lab_name_standardized'] == '$UNKNOWN$']
if len(unknown_names) > 0:
    for _, row in unknown_names.iterrows():
        print(f"\nRaw lab_name: {row['lab_name_raw']}")
        print(f"  Lab type: {row['lab_type']}")
        print(f"  Unit: {row['lab_unit_raw']}")
        print(f"  Value: {row['value_raw']}")
        print(f"  Date: {row['date']}")

print("\n" + "="*80)
print("UNKNOWN UNITS - Detailed Analysis")
print("="*80)

unknown_units = df[df['lab_unit_standardized'] == '$UNKNOWN$']
if len(unknown_units) > 0:
    # Get unique combinations
    unique_combos = unknown_units[['lab_name_raw', 'lab_unit_raw', 'lab_type', 'lab_name_standardized']].drop_duplicates()

    for _, row in unique_combos.iterrows():
        print(f"\nTest: {row['lab_name_raw']}")
        print(f"  Standardized as: {row['lab_name_standardized']}")
        print(f"  Raw unit: {row['lab_unit_raw']}")
        print(f"  Lab type: {row['lab_type']}")

# Read lab_specs.json to check what's missing
print("\n" + "="*80)
print("CHECKING LAB_SPECS.JSON")
print("="*80)

with open("config/lab_specs.json", "r") as f:
    lab_specs = json.load(f)

print(f"\nTotal standardized labs in config: {len(lab_specs)}")

# Check which standardized names from unknown_units are missing specs
if len(unknown_units) > 0:
    standardized_names_with_unknown_units = unknown_units['lab_name_standardized'].unique()
    print(f"\nStandardized names with unknown units:")
    for name in standardized_names_with_unknown_units:
        if name in lab_specs:
            spec = lab_specs[name]
            print(f"\n  ✓ '{name}' EXISTS in config")
            print(f"    Primary unit: {spec.get('primary_unit')}")
            print(f"    Alternatives: {[alt['unit'] for alt in spec.get('alternatives', [])]}")
        else:
            print(f"\n  ✗ '{name}' NOT FOUND in config")
