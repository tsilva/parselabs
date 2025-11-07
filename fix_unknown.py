#!/usr/bin/env python3
"""Fix $UNKNOWN$ values by updating lab_specs.json and improving standardization."""

import json
from pathlib import Path

LAB_SPECS_PATH = Path("config/lab_specs.json")

# Load lab specs
with open(LAB_SPECS_PATH, "r") as f:
    lab_specs = json.load(f)

print("="*80)
print("APPLYING FIXES TO LAB_SPECS.JSON")
print("="*80)

# Fix 1: Add Hepatitis B e Antigen (HBeAg)
hbeag_name = "Blood - Hepatitis B e Antigen (HBeAg)"
if hbeag_name not in lab_specs:
    print(f"\n✓ Adding: {hbeag_name}")
    lab_specs[hbeag_name] = {
        "lab_type": "blood",
        "primary_unit": "unitless",
        "alternatives": [],
        "ranges": {}
    }
else:
    print(f"\n- Already exists: {hbeag_name}")

# Fix 2: Add "mm" as alternative unit for ESR tests
esr_tests = [
    "Blood - Erythrocyte Sedimentation Rate (ESR) - 1h",
    "Blood - Erythrocyte Sedimentation Rate (ESR) - 2h",
    "Blood - Erythrocyte Sedimentation Rate (ESR, 1 Hour)"
]

for test_name in esr_tests:
    if test_name in lab_specs:
        alternatives = lab_specs[test_name].get("alternatives", [])
        # Check if "mm" already exists
        has_mm = any(alt.get("unit") == "mm" for alt in alternatives)

        if not has_mm:
            print(f"\n✓ Adding 'mm' alternative to: {test_name}")
            alternatives.append({"unit": "mm", "factor": 1.0})
            lab_specs[test_name]["alternatives"] = alternatives
        else:
            print(f"\n- Already has 'mm': {test_name}")
    else:
        print(f"\n⚠ NOT FOUND: {test_name}")

# Save updated lab specs
with open(LAB_SPECS_PATH, "w") as f:
    json.dump(lab_specs, f, ensure_ascii=False, indent=2)

print("\n" + "="*80)
print("✅ Lab specs updated successfully!")
print("="*80)
print("\nNext steps:")
print("1. Re-run the extraction pipeline: python main.py")
print("2. The new lab specs will be used for standardization")
print("3. Re-check for $UNKNOWN$ values: python search_unknown.py")
