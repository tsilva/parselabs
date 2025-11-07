#!/usr/bin/env python3
"""Compare old vs new extraction results"""

import pandas as pd
import numpy as np

# Load both CSVs
old_df = pd.read_csv("test/outputs/test/test.csv.old_approach")
new_df = pd.read_csv("test/outputs/test/test.csv")

print("="*80)
print("COMPARISON: Old Approach (with transcription) vs New Approach (direct extraction)")
print("="*80)
print()

# Basic stats
print(f"Old approach: {len(old_df)} rows")
print(f"New approach: {len(new_df)} rows")
print()

# Compare critical fields
critical_fields = ['test_name', 'value', 'unit', 'reference_range']

# Check if test names match
old_names = set(old_df['test_name'].dropna())
new_names = set(new_df['test_name'].dropna())

print("TEST NAMES:")
print(f"  Old: {len(old_names)} unique test names")
print(f"  New: {len(new_names)} unique test names")
print()

if old_names != new_names:
    print("  ⚠️  Test names differ:")
    only_old = old_names - new_names
    only_new = new_names - old_names
    if only_old:
        print(f"    Only in old: {only_old}")
    if only_new:
        print(f"    Only in new: {only_new}")
else:
    print("  ✅ Test names match!")
print()

# Compare values for each test
print("VALUE COMPARISON:")
print("-"*80)

# Merge on test_name to compare values
merged = old_df.merge(
    new_df,
    on='test_name',
    how='outer',
    suffixes=('_old', '_new'),
    indicator=True
)

# Check for value differences
value_diffs = []
for idx, row in merged.iterrows():
    name = row['test_name']
    val_old = row.get('value_old')
    val_new = row.get('value_new')

    # Skip if both are NaN
    if pd.isna(val_old) and pd.isna(val_new):
        continue

    # Check if values differ
    if pd.isna(val_old) or pd.isna(val_new) or not np.isclose(val_old, val_new, rtol=1e-3, equal_nan=True):
        value_diffs.append({
            'test_name': name,
            'old_value': val_old,
            'new_value': val_new,
            'old_unit': row.get('unit_old'),
            'new_unit': row.get('unit_new')
        })

if value_diffs:
    print(f"  ⚠️  Found {len(value_diffs)} value differences:")
    for diff in value_diffs[:10]:  # Show first 10
        print(f"    {diff['test_name'][:50]}")
        print(f"      Old: {diff['old_value']} {diff['old_unit']}")
        print(f"      New: {diff['new_value']} {diff['new_unit']}")
    if len(value_diffs) > 10:
        print(f"    ... and {len(value_diffs) - 10} more")
else:
    print("  ✅ All values match!")
print()

# Compare standardized fields
print("STANDARDIZED FIELDS:")
print("-"*80)

# Compare lab_name_standardized
old_std = set(old_df['lab_name_standardized'].dropna())
new_std = set(new_df['lab_name_standardized'].dropna())

print(f"  Old: {len(old_std)} unique standardized names")
print(f"  New: {len(new_std)} unique standardized names")

if old_std == new_std:
    print("  ✅ Standardized names match!")
else:
    only_old_std = old_std - new_std
    only_new_std = new_std - old_std
    if only_old_std or only_new_std:
        print("  ⚠️  Standardized names differ:")
        if only_old_std:
            print(f"    Only in old: {only_old_std}")
        if only_new_std:
            print(f"    Only in new: {only_new_std}")
print()

# Summary
print("="*80)
print("SUMMARY:")
print("="*80)

issues = []
if len(old_df) != len(new_df):
    issues.append(f"Row count differs: {len(old_df)} vs {len(new_df)}")
if old_names != new_names:
    issues.append(f"Test names differ")
if value_diffs:
    issues.append(f"{len(value_diffs)} value differences found")

if not issues:
    print("✅ Results are equivalent! The simplified approach produces the same extraction quality.")
else:
    print("⚠️  Issues found:")
    for issue in issues:
        print(f"  - {issue}")
    print()
    print("Note: Minor differences in test_name prefixes or formatting are acceptable")
    print("as long as the core values and standardized fields match.")

print()
