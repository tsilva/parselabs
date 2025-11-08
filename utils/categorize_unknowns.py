#!/usr/bin/env python3
"""Categorize $UNKNOWN$ values to determine which are real tests vs reference indicators."""

import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()
OUTPUT_PATH = Path(os.getenv("OUTPUT_PATH"))

# Read all.csv
csv_path = OUTPUT_PATH / "all.csv"
df = pd.read_csv(csv_path)

# Get all unknown lab names
unknown_names = df[df['lab_name_standardized'] == '$UNKNOWN$']

print("="*80)
print("CATEGORIZING UNKNOWN LAB NAMES")
print("="*80)

# Category 1: Reference range indicators (not actual lab tests)
reference_indicators = []
actual_tests = []

for lab_name_raw in unknown_names['lab_name_raw'].unique():
    test_lower = lab_name_raw.lower()

    # Check if it's a reference indicator
    if any(keyword in test_lower for keyword in [
        'deficiencia', 'insuficiencia', 'suficiencia', 'toxicidade',
        'ferropenia', 'alto risco', 'baixo risco',
        'avaliação de risco'
    ]):
        reference_indicators.append(lab_name_raw)
    else:
        actual_tests.append(lab_name_raw)

print(f"\n📊 CATEGORY 1: REFERENCE INDICATORS ({len(reference_indicators)})")
print("These are NOT lab tests, but interpretation thresholds:")
for name in sorted(reference_indicators):
    count = len(unknown_names[unknown_names['lab_name_raw'] == name])
    print(f"  - {name} ({count} occurrences)")

print(f"\n🔬 CATEGORY 2: ACTUAL LAB TESTS ({len(actual_tests)})")
print("These ARE real lab tests that need to be added to config:")

# Further categorize actual tests
by_category = defaultdict(list)

for name in sorted(actual_tests):
    count = len(unknown_names[unknown_names['lab_name_raw'] == name])

    # Determine category
    name_lower = name.lower()
    if 'anti-hav' in name_lower or 'hepatite a' in name_lower:
        category = "Hepatitis A"
    elif 'anti-hbe' in name_lower or 'hepatite b' in name_lower:
        category = "Hepatitis B"
    elif 'hiv' in name_lower or 'anti-hiv' in name_lower:
        category = "HIV"
    elif 'citomegalovirus' in name_lower or 'cmv' in name_lower:
        category = "Cytomegalovirus"
    elif 'epstein' in name_lower or 'ebv' in name_lower:
        category = "Epstein-Barr Virus"
    elif 'endomísio' in name_lower or 'endomisio' in name_lower:
        category = "Celiac Disease"
    elif 'treponema' in name_lower or 'tpha' in name_lower:
        category = "Syphilis"
    elif 'morfológico' in name_lower or 'morfologico' in name_lower:
        category = "Morphology"
    elif 'g6pd' in name_lower or 'glucose-6-fosfato' in name_lower:
        category = "Enzyme Assays"
    elif 'piruvatoquinase' in name_lower or 'pyruvate' in name_lower:
        category = "Enzyme Assays"
    elif 'coombs' in name_lower or 'antiglobulina' in name_lower:
        category = "Blood Bank Tests"
    elif 'hemoglobinúria' in name_lower or 'hpn' in name_lower or 'gpi' in name_lower:
        category = "Paroxysmal Nocturnal Hemoglobinuria"
    elif 'hplc' in name_lower:
        category = "Hemoglobin HPLC"
    elif 'hemossiderina' in name_lower:
        category = "Iron Studies"
    elif 'caracterização' in name_lower or 'produto' in name_lower:
        category = "Sample Information"
    elif 'linfoplasmocitárias' in name_lower:
        category = "Cell Counts"
    elif 'monócitos' in name_lower and 'fórmula' in name_lower:
        category = "Cell Counts"
    else:
        category = "Other"

    by_category[category].append((name, count))

for category in sorted(by_category.keys()):
    print(f"\n  {category}:")
    for name, count in by_category[category]:
        print(f"    - {name} ({count} occurrences)")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

print(f"""
1. REFERENCE INDICATORS ({len(reference_indicators)} tests)
   ❌ Do NOT add these to lab_specs.json
   ✅ These should be filtered out during extraction
   ✅ They represent interpretation ranges, not actual test results

2. ACTUAL LAB TESTS ({len(actual_tests)} tests)
   ✅ Add these to lab_specs.json
   ✅ Most need primary_unit: "unitless" or "boolean"
   ✅ Some need specialized units (UI/g Hb, Indice, etc.)

3. EXTRACTION IMPROVEMENT
   Consider updating extraction prompt to skip reference range indicators
   and interpretation thresholds that don't have actual measured values.
""")

# Analyze units
print("\n" + "="*80)
print("MISSING UNITS ANALYSIS")
print("="*80)

unknown_units = df[df['lab_unit_standardized'] == '$UNKNOWN$']
unit_combos = unknown_units.groupby(['lab_unit_raw']).size().reset_index(name='count')
unit_combos = unit_combos.sort_values('count', ascending=False)

print("\nUnits that need to be added to lab_specs.json:")
for _, row in unit_combos.iterrows():
    if pd.notna(row['lab_unit_raw']):
        print(f"  - '{row['lab_unit_raw']}' ({row['count']} occurrences)")

print("\n" + "="*80)
print("ACTION ITEMS")
print("="*80)
print("""
1. Update lab_specs.json with ~35-40 new lab tests
2. Add ~10 new unit alternatives (UI/g Hb, U/g Hb, Indice, L/L, etc.)
3. Consider filtering reference indicators during extraction
4. Re-run pipeline to apply fixes
""")
