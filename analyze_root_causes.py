#!/usr/bin/env python3
"""Analyze root causes of $UNKNOWN$ values found in generated CSVs."""

import json
from pathlib import Path
from config import LabSpecsConfig

print("="*80)
print("ROOT CAUSE ANALYSIS")
print("="*80)

lab_specs = LabSpecsConfig()

print("\n" + "="*80)
print("ISSUE 1: MISSING LAB SPECS")
print("="*80)

missing_specs = [
    ("IMUNOLOGIA - ANTICORPO ANTI-HAV", "Hepatitis A Antibody Total", "unitless"),
    ("IMUNOLOGIA - ANTICORPO ANTI-HAV (IgM)", "Hepatitis A Antibody (IgM)", "unitless"),
    ("IMUNOLOGIA - MARCADORES VIRICOS DA HEPATITE B - Anticorpo Anti-HBe", "Hepatitis B e Antibody (Anti-HBe)", "unitless"),
]

print("\nThese lab tests are not in lab_specs.json:")
for raw_name, suggested_name, suggested_unit in missing_specs:
    full_name = f"Blood - {suggested_name}"
    exists = full_name in lab_specs.specs
    print(f"\n  {raw_name}")
    print(f"    Suggested: {full_name}")
    print(f"    Primary unit: {suggested_unit}")
    print(f"    Exists: {'✅' if exists else '❌'}")

print("\n" + "="*80)
print("ISSUE 2: MISSING UNIT ALTERNATIVES")
print("="*80)

unit_issues = [
    ("Blood - Erythrocytes", "E6/mm3", "10⁶/mm³", "Should map to 10⁶/mm³"),
    ("Blood - Basophils", "x10¹/L", "10⁹/L", "Typo: should be 10⁹/L not 10¹/L"),
    ("Blood - Eosinophils", "x10¹/L", "10⁹/L", "Typo: should be 10⁹/L not 10¹/L"),
    ("Blood - Neutrophils", "x10¹/L", "10⁹/L", "Typo: should be 10⁹/L not 10¹/L"),
    ("Blood - Leukocytes", "x10¹/L", "10⁹/L", "Typo: should be 10⁹/L not 10¹/L"),
    ("Blood - Lymphocytes", "x10¹/L", "10⁹/L", "Typo: should be 10⁹/L not 10¹/L"),
    ("Blood - Monocytes", "x10¹/L", "10⁹/L", "Typo: should be 10⁹/L not 10¹/L"),
    ("Blood - Basophils", "/mm3", "cells/mm³", "Missing unit alternative"),
    ("Blood - Eosinophils", "/mm3", "cells/mm³", "Missing unit alternative"),
    ("Blood - Neutrophils", "/mm3", "cells/mm³", "Missing unit alternative"),
    ("Blood - Leukocytes", "/mm3", "cells/mm³", "Missing unit alternative"),
    ("Blood - Lymphocytes", "/mm3", "cells/mm³", "Missing unit alternative"),
    ("Blood - Monocytes", "/mm3", "cells/mm³", "Missing unit alternative"),
    ("Blood - Platelets", "/mm3", "10³/mm³", "Missing unit alternative"),
    ("Blood - Thyroid Stimulating Hormone (TSH)", "mUI/L", "mIU/L", "Case variation: mUI/L vs mIU/L"),
    ("Blood - Erythrocyte Sedimentation Rate (ESR) - 1h", "mn", "mm/h", "Typo: 'mn' should be 'mm'"),
]

print("\nThese units need to be added as alternatives:")
for lab_name, raw_unit, standardized_unit, note in unit_issues:
    spec = lab_specs.specs.get(lab_name)
    if spec:
        primary = spec.get('primary_unit')
        alternatives = [alt['unit'] for alt in spec.get('alternatives', [])]
        has_alternative = raw_unit in alternatives or standardized_unit in alternatives

        print(f"\n  {lab_name}")
        print(f"    Raw unit: '{raw_unit}' → {standardized_unit}")
        print(f"    Primary: {primary}")
        print(f"    Alternatives: {alternatives}")
        print(f"    Has alternative: {'✅' if has_alternative else '❌'}")
        print(f"    Note: {note}")
    else:
        print(f"\n  ❌ {lab_name} NOT FOUND IN SPECS")

print("\n" + "="*80)
print("ISSUE 3: LAB_TYPE IS N/A")
print("="*80)
print("\nAll rows show lab_type='N/A' instead of 'blood'")
print("This suggests the lab_type field is not being populated correctly")
print("during extraction or standardization.")

print("\n" + "="*80)
print("RECOMMENDED FIXES")
print("="*80)
print("""
1. Add missing hepatitis markers to lab_specs.json:
   - Blood - Hepatitis A Antibody Total
   - Blood - Hepatitis A Antibody (IgM)
   - Blood - Hepatitis B e Antibody (Anti-HBe)

2. Add unit alternatives to existing lab specs:
   - E6/mm3 → 10⁶/mm³ for Erythrocytes
   - x10¹/L → 10⁹/L for cell counts (LLM should recognize this typo)
   - /mm3 → cells/mm³ for cell counts
   - mUI/L → mIU/L for TSH
   - mn → mm for ESR (already added, but LLM needs to recognize)

3. Investigate lab_type field population issue
""")
