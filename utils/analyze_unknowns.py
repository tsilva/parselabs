#!/usr/bin/env python3
"""
Consolidated tool for analyzing $UNKNOWN$ values in lab results.

Usage:
    python utils/analyze_unknowns.py --mode search      # Count and summarize unknowns
    python utils/analyze_unknowns.py --mode analyze     # Detailed row-by-row analysis
    python utils/analyze_unknowns.py --mode categorize  # Categorize unknowns with recommendations
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
import os
from dotenv import load_dotenv

load_dotenv(Path(".env.local"), override=True)


def load_data():
    """Load the main CSV file."""
    output_path = Path(os.getenv("OUTPUT_PATH", "output"))
    csv_path = output_path / "all.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    return pd.read_csv(csv_path)


def mode_search(df: pd.DataFrame):
    """Basic search: count and summarize unknowns."""
    print(f"\nTotal rows: {len(df)}")

    # Unknown lab names
    unknown_names = df[df['lab_name_standardized'] == '$UNKNOWN$']
    if len(unknown_names) > 0:
        print(f"\n{'='*80}")
        print(f"UNKNOWN LAB NAMES: {len(unknown_names)} rows")
        print(f"{'='*80}")

        name_counts = unknown_names.groupby(['lab_name_raw', 'lab_type']).size().reset_index(name='count')
        name_counts = name_counts.sort_values('count', ascending=False)

        print("\nRaw test names that couldn't be standardized:")
        for _, row in name_counts.iterrows():
            print(f"  [{row['lab_type']}] {row['lab_name_raw']:60s} ({row['count']} occurrences)")
    else:
        print("\n No unknown lab names found!")

    # Unknown units
    unknown_units = df[df['lab_unit_standardized'] == '$UNKNOWN$']
    if len(unknown_units) > 0:
        print(f"\n{'='*80}")
        print(f"UNKNOWN UNITS: {len(unknown_units)} rows")
        print(f"{'='*80}")

        unit_counts = unknown_units.groupby(['lab_name_raw', 'lab_unit_raw', 'lab_type']).size().reset_index(name='count')
        unit_counts = unit_counts.sort_values('count', ascending=False)

        print("\nRaw units that couldn't be standardized:")
        for _, row in unit_counts.iterrows():
            print(f"  [{row['lab_type']}] {row['lab_name_raw']:50s} | Unit: {row['lab_unit_raw']:15s} ({row['count']} occurrences)")
    else:
        print("\n No unknown units found!")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total rows: {len(df)}")
    print(f"Rows with unknown lab names: {len(unknown_names)} ({len(unknown_names)/len(df)*100:.1f}%)")
    print(f"Rows with unknown units: {len(unknown_units)} ({len(unknown_units)/len(df)*100:.1f}%)")
    print(f"Unique lab names in data: {df['lab_name_raw'].nunique()}")
    print(f"Unique standardized names: {df[df['lab_name_standardized'] != '$UNKNOWN$']['lab_name_standardized'].nunique()}")


def mode_analyze(df: pd.DataFrame):
    """Detailed analysis: row-by-row view and lab_specs check."""
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
        unique_combos = unknown_units[['lab_name_raw', 'lab_unit_raw', 'lab_type', 'lab_name_standardized']].drop_duplicates()

        for _, row in unique_combos.iterrows():
            print(f"\nTest: {row['lab_name_raw']}")
            print(f"  Standardized as: {row['lab_name_standardized']}")
            print(f"  Raw unit: {row['lab_unit_raw']}")
            print(f"  Lab type: {row['lab_type']}")

    # Check lab_specs.json
    print("\n" + "="*80)
    print("CHECKING LAB_SPECS.JSON")
    print("="*80)

    lab_specs_path = Path("config/lab_specs.json")
    if lab_specs_path.exists():
        with open(lab_specs_path, "r") as f:
            lab_specs = json.load(f)

        print(f"\nTotal standardized labs in config: {len(lab_specs)}")

        if len(unknown_units) > 0:
            standardized_names_with_unknown_units = unknown_units['lab_name_standardized'].unique()
            print(f"\nStandardized names with unknown units:")
            for name in standardized_names_with_unknown_units:
                if name in lab_specs:
                    spec = lab_specs[name]
                    print(f"\n  '{name}' EXISTS in config")
                    print(f"    Primary unit: {spec.get('primary_unit')}")
                    print(f"    Alternatives: {[alt['unit'] for alt in spec.get('alternatives', [])]}")
                else:
                    print(f"\n  '{name}' NOT FOUND in config")
    else:
        print(f"\nWarning: {lab_specs_path} not found")


def mode_categorize(df: pd.DataFrame):
    """Categorize unknowns into reference indicators vs actual tests."""
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

    print(f"\nCATEGORY 1: REFERENCE INDICATORS ({len(reference_indicators)})")
    print("These are NOT lab tests, but interpretation thresholds:")
    for name in sorted(reference_indicators):
        count = len(unknown_names[unknown_names['lab_name_raw'] == name])
        print(f"  - {name} ({count} occurrences)")

    print(f"\nCATEGORY 2: ACTUAL LAB TESTS ({len(actual_tests)})")
    print("These ARE real lab tests that need to be added to config:")

    # Further categorize actual tests
    by_category = defaultdict(list)

    for name in sorted(actual_tests):
        count = len(unknown_names[unknown_names['lab_name_raw'] == name])
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

    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    print(f"""
1. REFERENCE INDICATORS ({len(reference_indicators)} tests)
   Do NOT add these to lab_specs.json
   These should be filtered out during extraction
   They represent interpretation ranges, not actual test results

2. ACTUAL LAB TESTS ({len(actual_tests)} tests)
   Add these to lab_specs.json
   Most need primary_unit: "unitless" or "boolean"
   Some need specialized units (UI/g Hb, Indice, etc.)

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


def main():
    parser = argparse.ArgumentParser(
        description="Analyze $UNKNOWN$ values in lab results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  search      Count and summarize unknown values (quick overview)
  analyze     Detailed row-by-row analysis with config lookup
  categorize  Categorize unknowns and provide recommendations
        """
    )
    parser.add_argument(
        '--mode', '-m',
        choices=['search', 'analyze', 'categorize'],
        default='search',
        help='Analysis mode (default: search)'
    )

    args = parser.parse_args()

    try:
        df = load_data()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    if args.mode == 'search':
        mode_search(df)
    elif args.mode == 'analyze':
        mode_analyze(df)
    elif args.mode == 'categorize':
        mode_categorize(df)

    return 0


if __name__ == "__main__":
    exit(main())
