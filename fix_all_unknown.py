#!/usr/bin/env python3
"""Comprehensive fix for all $UNKNOWN$ issues found in generated CSVs."""

import json
from pathlib import Path

LAB_SPECS_PATH = Path("config/lab_specs.json")

# Load lab specs
with open(LAB_SPECS_PATH, "r") as f:
    lab_specs = json.load(f)

print("="*80)
print("APPLYING COMPREHENSIVE FIXES TO LAB_SPECS.JSON")
print("="*80)

# ============================================================================
# FIX 1: Add missing hepatitis markers
# ============================================================================
print("\n" + "="*80)
print("FIX 1: ADDING MISSING LAB SPECS")
print("="*80)

new_labs = {
    "Blood - Hepatitis A Antibody Total": {
        "lab_type": "blood",
        "primary_unit": "unitless",
        "alternatives": [],
        "ranges": {}
    },
    "Blood - Hepatitis A Antibody (IgM)": {
        "lab_type": "blood",
        "primary_unit": "unitless",
        "alternatives": [],
        "ranges": {}
    },
    "Blood - Hepatitis B e Antibody (Anti-HBe)": {
        "lab_type": "blood",
        "primary_unit": "unitless",
        "alternatives": [],
        "ranges": {}
    }
}

for lab_name, spec in new_labs.items():
    if lab_name not in lab_specs:
        print(f"  ✓ Adding: {lab_name}")
        lab_specs[lab_name] = spec
    else:
        print(f"  - Already exists: {lab_name}")

# ============================================================================
# FIX 2: Add unit alternatives
# ============================================================================
print("\n" + "="*80)
print("FIX 2: ADDING UNIT ALTERNATIVES")
print("="*80)

# Helper function to add alternative unit
def add_alternative(lab_name, unit, factor=1.0):
    if lab_name in lab_specs:
        alternatives = lab_specs[lab_name].get("alternatives", [])
        # Check if unit already exists
        has_unit = any(alt.get("unit") == unit for alt in alternatives)

        if not has_unit:
            print(f"  ✓ Adding '{unit}' to {lab_name}")
            alternatives.append({"unit": unit, "factor": factor})
            lab_specs[lab_name]["alternatives"] = alternatives
            return True
        else:
            print(f"  - Already has '{unit}': {lab_name}")
            return False
    else:
        print(f"  ❌ Lab not found: {lab_name}")
        return False

# Add E6/mm3 for Erythrocytes (1 E6/mm3 = 1 million/mm3 = 10^6/mm3)
# Primary is 10¹²/L, so 1 E6/mm3 = 1×10⁶/mm³ = 1×10⁶/(10^-3 L) = 1×10⁹/L = 0.001×10¹²/L
add_alternative("Blood - Erythrocytes", "E6/mm3", 0.001)
add_alternative("Blood - Erythrocytes", "10⁶/mm³", 0.001)

# Add x10¹/L for cell counts (likely a typo/OCR error for 10⁹/L)
# Since primary is 10⁹/L, this should be factor 1.0
add_alternative("Blood - Basophils", "x10¹/L", 1.0)
add_alternative("Blood - Eosinophils", "x10¹/L", 1.0)
add_alternative("Blood - Neutrophils", "x10¹/L", 1.0)
add_alternative("Blood - Leukocytes", "x10¹/L", 1.0)
add_alternative("Blood - Lymphocytes", "x10¹/L", 1.0)
add_alternative("Blood - Monocytes", "x10¹/L", 1.0)

# Add /mm3 for cell counts
# Primary is 10⁹/L, so 1 cell/mm³ = 1/(10^-3 L) = 1000/L = 0.001×10⁹/L
add_alternative("Blood - Basophils", "/mm3", 0.001)
add_alternative("Blood - Basophils", "cells/mm³", 0.001)
add_alternative("Blood - Eosinophils", "/mm3", 0.001)
add_alternative("Blood - Eosinophils", "cells/mm³", 0.001)
add_alternative("Blood - Neutrophils", "/mm3", 0.001)
add_alternative("Blood - Neutrophils", "cells/mm³", 0.001)
add_alternative("Blood - Leukocytes", "/mm3", 0.001)
add_alternative("Blood - Leukocytes", "cells/mm³", 0.001)
add_alternative("Blood - Lymphocytes", "/mm3", 0.001)
add_alternative("Blood - Lymphocytes", "cells/mm³", 0.001)
add_alternative("Blood - Monocytes", "/mm3", 0.001)
add_alternative("Blood - Monocytes", "cells/mm³", 0.001)

# Add /mm3 for Platelets (same conversion)
add_alternative("Blood - Platelets", "/mm3", 0.001)
add_alternative("Blood - Platelets", "10³/mm³", 1.0)  # 10³/mm³ = 10⁹/L

# Add mUI/L for TSH (case variation)
# Primary is µIU/mL, and mIU/L = µIU/mL (1000 µIU/mL = 1 mIU/mL, 1 mIU/L = 0.001 mIU/mL)
# Actually mUI/L and mIU/L are the same (milli-international units)
# 1 mIU/L = 1 µIU/mL
add_alternative("Blood - Thyroid Stimulating Hormone (TSH)", "mUI/L", 1.0)
add_alternative("Blood - Thyroid Stimulating Hormone (TSH)", "mIU/L", 1.0)

# Add mn for ESR (typo for mm)
add_alternative("Blood - Erythrocyte Sedimentation Rate (ESR) - 1h", "mn", 1.0)
if "Blood - Erythrocyte Sedimentation Rate (ESR) - 2h" in lab_specs:
    add_alternative("Blood - Erythrocyte Sedimentation Rate (ESR) - 2h", "mn", 1.0)

# Save updated lab specs
with open(LAB_SPECS_PATH, "w") as f:
    json.dump(lab_specs, f, ensure_ascii=False, indent=2)

print("\n" + "="*80)
print("✅ ALL FIXES APPLIED SUCCESSFULLY!")
print("="*80)
print(f"\nTotal labs in config: {len(lab_specs)}")
print("\nNext steps:")
print("1. Re-run the extraction pipeline: python main.py")
print("2. The updated lab specs will be used for standardization")
print("3. Re-check for $UNKNOWN$ values: python deep_search_unknown.py")
