#!/usr/bin/env python3
"""Verify that pipeline re-run successfully fixed all $UNKNOWN$ values."""

import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

OUTPUT_PATH = Path(os.getenv("OUTPUT_PATH"))
csv_path = OUTPUT_PATH / "all.csv"

if not csv_path.exists():
    print("❌ all.csv not found. Please run: python main.py")
    exit(1)

print("="*80)
print("VERIFYING PIPELINE FIXES")
print("="*80)

# Read all.csv
df = pd.read_csv(csv_path)

print(f"\nTotal rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

# Check for required derived columns
required_cols = ["lab_type", "lab_name", "lab_unit", "lab_name_standardized", "lab_unit_standardized"]
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    print(f"\n❌ Missing columns: {missing_cols}")
    print("   The normalization step may not have run properly.")
else:
    print(f"\n✅ All required columns present: {required_cols}")

# Check for $UNKNOWN$ in lab_name_standardized
unknown_names = df[df['lab_name_standardized'] == '$UNKNOWN$']
if len(unknown_names) > 0:
    print(f"\n❌ STILL HAVE UNKNOWN LAB NAMES: {len(unknown_names)} rows")
    print("\nUnknown test names:")
    for test_name in unknown_names['test_name'].unique():
        count = len(unknown_names[unknown_names['test_name'] == test_name])
        print(f"  - {test_name} ({count} occurrences)")
else:
    print(f"\n✅ NO UNKNOWN LAB NAMES!")

# Check for $UNKNOWN$ in lab_unit_standardized
unknown_units = df[df['lab_unit_standardized'] == '$UNKNOWN$']
if len(unknown_units) > 0:
    print(f"\n❌ STILL HAVE UNKNOWN UNITS: {len(unknown_units)} rows")
    print("\nUnknown unit combinations:")
    for (test_name, unit) in unknown_units[['test_name', 'unit']].drop_duplicates().values:
        count = len(unknown_units[(unknown_units['test_name'] == test_name) & (unknown_units['unit'] == unit)])
        print(f"  - {test_name} | Unit: {unit} ({count} occurrences)")
else:
    print(f"\n✅ NO UNKNOWN UNITS!")

# Check lab_type distribution
if 'lab_type' in df.columns:
    print(f"\n✅ lab_type distribution:")
    for lab_type, count in df['lab_type'].value_counts().items():
        print(f"  - {lab_type}: {count} rows ({count/len(df)*100:.1f}%)")
else:
    print(f"\n❌ lab_type column not found")

# Check for null values in key columns
print(f"\n" + "="*80)
print("NULL VALUE CHECK")
print("="*80)

key_cols = ['lab_name_standardized', 'lab_unit_standardized', 'lab_type', 'lab_name', 'lab_unit']
for col in key_cols:
    if col in df.columns:
        null_count = df[col].isna().sum()
        if null_count > 0:
            print(f"⚠️  {col}: {null_count} null values ({null_count/len(df)*100:.1f}%)")
        else:
            print(f"✅ {col}: no null values")
    else:
        print(f"❌ {col}: column not found")

# Summary
print(f"\n" + "="*80)
print("SUMMARY")
print("="*80)

success = True
issues = []

if missing_cols:
    success = False
    issues.append(f"Missing columns: {missing_cols}")

if len(unknown_names) > 0:
    success = False
    issues.append(f"{len(unknown_names)} rows with unknown lab names")

if len(unknown_units) > 0:
    success = False
    issues.append(f"{len(unknown_units)} rows with unknown units")

if success:
    print("\n🎉 ALL FIXES VERIFIED SUCCESSFULLY!")
    print("\n✅ No unknown lab names")
    print("✅ No unknown units")
    print("✅ All derived columns present")
    print("✅ Pipeline is working correctly")
else:
    print("\n⚠️  ISSUES FOUND:")
    for issue in issues:
        print(f"   - {issue}")
    print("\nNext steps:")
    print("1. Check the logs for errors during pipeline execution")
    print("2. Verify lab_specs.json has all necessary entries")
    print("3. Run python analyze_root_causes.py to debug remaining issues")
