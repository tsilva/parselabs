#!/usr/bin/env python3
"""Search for $UNKNOWN$ values in parsed CSV files and analyze root causes."""

import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

OUTPUT_PATH = os.getenv("OUTPUT_PATH")

# Read the main aggregated CSV
csv_path = Path(OUTPUT_PATH) / "all.csv"
print(f"Reading: {csv_path}")
df = pd.read_csv(csv_path)

print(f"\nTotal rows: {len(df)}")

# Check for unknown standardized lab names
unknown_names = df[df['lab_name_standardized'] == '$UNKNOWN$']
if len(unknown_names) > 0:
    print(f"\n{'='*80}")
    print(f"❌ UNKNOWN LAB NAMES: {len(unknown_names)} rows")
    print(f"{'='*80}")

    # Group by raw lab_name_raw to see patterns
    name_counts = unknown_names.groupby(['lab_name_raw', 'lab_type']).size().reset_index(name='count')
    name_counts = name_counts.sort_values('count', ascending=False)

    print("\nRaw test names that couldn't be standardized:")
    for _, row in name_counts.iterrows():
        print(f"  [{row['lab_type']}] {row['lab_name_raw']:60s} ({row['count']} occurrences)")
else:
    print("\n✅ No unknown lab names found!")

# Check for unknown standardized units
unknown_units = df[df['lab_unit_standardized'] == '$UNKNOWN$']
if len(unknown_units) > 0:
    print(f"\n{'='*80}")
    print(f"❌ UNKNOWN UNITS: {len(unknown_units)} rows")
    print(f"{'='*80}")

    # Group by lab_name_raw and lab_unit_raw to see patterns
    unit_counts = unknown_units.groupby(['lab_name_raw', 'lab_unit_raw', 'lab_type']).size().reset_index(name='count')
    unit_counts = unit_counts.sort_values('count', ascending=False)

    print("\nRaw units that couldn't be standardized:")
    for _, row in unit_counts.iterrows():
        print(f"  [{row['lab_type']}] {row['lab_name_raw']:50s} | Unit: {row['lab_unit_raw']:15s} ({row['count']} occurrences)")
else:
    print("\n✅ No unknown units found!")

# Summary statistics
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"Total rows: {len(df)}")
print(f"Rows with unknown lab names: {len(unknown_names)} ({len(unknown_names)/len(df)*100:.1f}%)")
print(f"Rows with unknown units: {len(unknown_units)} ({len(unknown_units)/len(df)*100:.1f}%)")
print(f"Unique lab names in data: {df['lab_name_raw'].nunique()}")
print(f"Unique standardized names: {df[df['lab_name_standardized'] != '$UNKNOWN$']['lab_name_standardized'].nunique()}")
