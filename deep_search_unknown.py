#!/usr/bin/env python3
"""Deep search through ALL generated CSV files for $UNKNOWN$ values."""

import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()

OUTPUT_PATH = Path(os.getenv("OUTPUT_PATH"))

# Find all CSV files (individual page CSVs and document CSVs)
all_csv_files = sorted(OUTPUT_PATH.rglob("*.csv"))

print("="*80)
print(f"DEEP SEARCH FOR $UNKNOWN$ VALUES")
print(f"Found {len(all_csv_files)} CSV files")
print("="*80)

unknown_lab_names = defaultdict(list)
unknown_units = defaultdict(list)

for csv_file in all_csv_files:
    try:
        df = pd.read_csv(csv_file)

        # Check for unknown lab names
        if 'lab_name_standardized' in df.columns:
            unknown_name_rows = df[df['lab_name_standardized'] == '$UNKNOWN$']
            for _, row in unknown_name_rows.iterrows():
                test_name = row.get('test_name', 'N/A')
                lab_type = row.get('lab_type', 'N/A')
                unit = row.get('unit', 'N/A')
                value = row.get('value', 'N/A')

                key = (test_name, lab_type)
                unknown_lab_names[key].append({
                    'file': str(csv_file.relative_to(OUTPUT_PATH)),
                    'unit': unit,
                    'value': value
                })

        # Check for unknown units
        if 'lab_unit_standardized' in df.columns:
            unknown_unit_rows = df[df['lab_unit_standardized'] == '$UNKNOWN$']
            for _, row in unknown_unit_rows.iterrows():
                test_name = row.get('test_name', 'N/A')
                lab_name_std = row.get('lab_name_standardized', 'N/A')
                lab_type = row.get('lab_type', 'N/A')
                unit = row.get('unit', 'N/A')
                value = row.get('value', 'N/A')

                key = (test_name, unit, lab_name_std, lab_type)
                unknown_units[key].append({
                    'file': str(csv_file.relative_to(OUTPUT_PATH)),
                    'value': value
                })

    except Exception as e:
        print(f"Error reading {csv_file}: {e}")

# Report unknown lab names
print(f"\n{'='*80}")
print(f"UNKNOWN LAB NAMES: {len(unknown_lab_names)} unique tests")
print(f"{'='*80}")

if unknown_lab_names:
    for (test_name, lab_type), occurrences in sorted(unknown_lab_names.items()):
        print(f"\n[{lab_type}] {test_name}")
        print(f"  Occurrences: {len(occurrences)}")
        print(f"  Files: {set(occ['file'] for occ in occurrences)}")

        # Show unique units and values
        units = set(str(occ['unit']) for occ in occurrences)
        values = [str(occ['value']) for occ in occurrences[:3]]  # First 3 values
        print(f"  Units seen: {units}")
        print(f"  Sample values: {values}")
else:
    print("\n✅ No unknown lab names found!")

# Report unknown units
print(f"\n{'='*80}")
print(f"UNKNOWN UNITS: {len(unknown_units)} unique combinations")
print(f"{'='*80}")

if unknown_units:
    for (test_name, unit, lab_name_std, lab_type), occurrences in sorted(unknown_units.items()):
        print(f"\n[{lab_type}] {test_name}")
        print(f"  Raw unit: '{unit}'")
        print(f"  Standardized as: {lab_name_std}")
        print(f"  Occurrences: {len(occurrences)}")
        print(f"  Files: {set(occ['file'] for occ in occurrences)}")

        # Show sample values
        values = [str(occ['value']) for occ in occurrences[:3]]
        print(f"  Sample values: {values}")
else:
    print("\n✅ No unknown units found!")

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"Unique unknown lab names: {len(unknown_lab_names)}")
print(f"Unique unknown unit combinations: {len(unknown_units)}")
print(f"Total CSV files scanned: {len(all_csv_files)}")
