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
import logging
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv(Path(".env.local"), override=True)


def load_data():
    """Load the main CSV file."""

    output_path = Path(os.getenv("OUTPUT_PATH", "output"))
    csv_path = output_path / "all.csv"

    # Bail out if CSV doesn't exist
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    return pd.read_csv(csv_path)


def mode_search(df: pd.DataFrame):
    """Basic search: count and summarize unknowns."""

    logger.info(f"\nTotal rows: {len(df)}")

    # Unknown lab names
    unknown_names = df[df["lab_name_standardized"] == "$UNKNOWN$"]

    # Display unknown lab name details if any found
    if len(unknown_names) > 0:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"UNKNOWN LAB NAMES: {len(unknown_names)} rows")
        logger.info(f"{'=' * 80}")

        name_counts = unknown_names.groupby(["lab_name_raw", "lab_type"]).size().reset_index(name="count")
        name_counts = name_counts.sort_values("count", ascending=False)

        logger.info("\nRaw test names that couldn't be standardized:")
        for _, row in name_counts.iterrows():
            logger.info(f"  [{row['lab_type']}] {row['lab_name_raw']:60s} ({row['count']} occurrences)")
    # No unknown lab names
    else:
        logger.info("\n No unknown lab names found!")

    # Unknown units
    unknown_units = df[df["lab_unit_standardized"] == "$UNKNOWN$"]

    # Display unknown unit details if any found
    if len(unknown_units) > 0:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"UNKNOWN UNITS: {len(unknown_units)} rows")
        logger.info(f"{'=' * 80}")

        unit_counts = unknown_units.groupby(["lab_name_raw", "lab_unit_raw", "lab_type"]).size().reset_index(name="count")
        unit_counts = unit_counts.sort_values("count", ascending=False)

        logger.info("\nRaw units that couldn't be standardized:")
        for _, row in unit_counts.iterrows():
            logger.info(f"  [{row['lab_type']}] {row['lab_name_raw']:50s} | Unit: {row['lab_unit_raw']:15s} ({row['count']} occurrences)")
    # No unknown units
    else:
        logger.info("\n No unknown units found!")

    # Summary
    logger.info(f"\n{'=' * 80}")
    logger.info("SUMMARY")
    logger.info(f"{'=' * 80}")
    logger.info(f"Total rows: {len(df)}")
    logger.info(f"Rows with unknown lab names: {len(unknown_names)} ({len(unknown_names) / len(df) * 100:.1f}%)")
    logger.info(f"Rows with unknown units: {len(unknown_units)} ({len(unknown_units) / len(df) * 100:.1f}%)")
    logger.info(f"Unique lab names in data: {df['lab_name_raw'].nunique()}")
    logger.info(f"Unique standardized names: {df[df['lab_name_standardized'] != '$UNKNOWN$']['lab_name_standardized'].nunique()}")


def mode_analyze(df: pd.DataFrame):
    """Detailed analysis: row-by-row view and lab_specs check."""

    logger.info("=" * 80)
    logger.info("UNKNOWN LAB NAMES - Detailed Analysis")
    logger.info("=" * 80)

    unknown_names = df[df["lab_name_standardized"] == "$UNKNOWN$"]

    # Print details for each unknown lab name
    if len(unknown_names) > 0:
        for _, row in unknown_names.iterrows():
            logger.info(f"\nRaw lab_name: {row['lab_name_raw']}")
            logger.info(f"  Lab type: {row['lab_type']}")
            logger.info(f"  Unit: {row['lab_unit_raw']}")
            logger.info(f"  Value: {row['value_raw']}")
            logger.info(f"  Date: {row['date']}")

    logger.info("\n" + "=" * 80)
    logger.info("UNKNOWN UNITS - Detailed Analysis")
    logger.info("=" * 80)

    unknown_units = df[df["lab_unit_standardized"] == "$UNKNOWN$"]

    # Print unique test/unit combinations for unknown units
    if len(unknown_units) > 0:
        unique_combos = unknown_units[
            [
                "lab_name_raw",
                "lab_unit_raw",
                "lab_type",
                "lab_name_standardized",
            ]
        ].drop_duplicates()

        for _, row in unique_combos.iterrows():
            logger.info(f"\nTest: {row['lab_name_raw']}")
            logger.info(f"  Standardized as: {row['lab_name_standardized']}")
            logger.info(f"  Raw unit: {row['lab_unit_raw']}")
            logger.info(f"  Lab type: {row['lab_type']}")

    # Check lab_specs.json
    logger.info("\n" + "=" * 80)
    logger.info("CHECKING LAB_SPECS.JSON")
    logger.info("=" * 80)

    lab_specs_path = Path("config/lab_specs.json")

    # Cross-reference unknown units against lab_specs config
    if lab_specs_path.exists():
        with open(lab_specs_path, "r") as f:
            lab_specs = json.load(f)

        logger.info(f"\nTotal standardized labs in config: {len(lab_specs)}")

        # Check which standardized names have unknown units
        if len(unknown_units) > 0:
            standardized_names_with_unknown_units = unknown_units["lab_name_standardized"].unique()
            logger.info("\nStandardized names with unknown units:")
            for name in standardized_names_with_unknown_units:
                # Lab exists in config - show its unit info
                if name in lab_specs:
                    spec = lab_specs[name]
                    logger.info(f"\n  '{name}' EXISTS in config")
                    logger.info(f"    Primary unit: {spec.get('primary_unit')}")
                    logger.info(f"    Alternatives: {[alt['unit'] for alt in spec.get('alternatives', [])]}")
                # Lab not in config
                else:
                    logger.info(f"\n  '{name}' NOT FOUND in config")
    # Config file missing
    else:
        logger.warning(f"{lab_specs_path} not found")


def mode_categorize(df: pd.DataFrame):
    """Categorize unknowns into reference indicators vs actual tests."""

    unknown_names = df[df["lab_name_standardized"] == "$UNKNOWN$"]

    logger.info("=" * 80)
    logger.info("CATEGORIZING UNKNOWN LAB NAMES")
    logger.info("=" * 80)

    # Category 1: Reference range indicators (not actual lab tests)
    reference_indicators = []
    actual_tests = []

    for lab_name_raw in unknown_names["lab_name_raw"].unique():
        test_lower = lab_name_raw.lower()

        # Matches a reference/interpretation keyword
        if any(
            keyword in test_lower
            for keyword in [
                "deficiencia",
                "insuficiencia",
                "suficiencia",
                "toxicidade",
                "ferropenia",
                "alto risco",
                "baixo risco",
                "avaliação de risco",
            ]
        ):
            reference_indicators.append(lab_name_raw)
        # Not a reference indicator - treat as actual test
        else:
            actual_tests.append(lab_name_raw)

    logger.info(f"\nCATEGORY 1: REFERENCE INDICATORS ({len(reference_indicators)})")
    logger.info("These are NOT lab tests, but interpretation thresholds:")
    for name in sorted(reference_indicators):
        count = len(unknown_names[unknown_names["lab_name_raw"] == name])
        logger.info(f"  - {name} ({count} occurrences)")

    logger.info(f"\nCATEGORY 2: ACTUAL LAB TESTS ({len(actual_tests)})")
    logger.info("These ARE real lab tests that need to be added to config:")

    # Further categorize actual tests
    by_category = defaultdict(list)

    for name in sorted(actual_tests):
        count = len(unknown_names[unknown_names["lab_name_raw"] == name])
        name_lower = name.lower()

        # Hepatitis A markers
        if "anti-hav" in name_lower or "hepatite a" in name_lower:
            category = "Hepatitis A"
        # Hepatitis B markers
        elif "anti-hbe" in name_lower or "hepatite b" in name_lower:
            category = "Hepatitis B"
        # HIV markers
        elif "hiv" in name_lower or "anti-hiv" in name_lower:
            category = "HIV"
        # Cytomegalovirus markers
        elif "citomegalovirus" in name_lower or "cmv" in name_lower:
            category = "Cytomegalovirus"
        # Epstein-Barr virus markers
        elif "epstein" in name_lower or "ebv" in name_lower:
            category = "Epstein-Barr Virus"
        # Celiac disease markers
        elif "endomísio" in name_lower or "endomisio" in name_lower:
            category = "Celiac Disease"
        # Syphilis markers
        elif "treponema" in name_lower or "tpha" in name_lower:
            category = "Syphilis"
        # Morphology tests
        elif "morfológico" in name_lower or "morfologico" in name_lower:
            category = "Morphology"
        # G6PD enzyme assay
        elif "g6pd" in name_lower or "glucose-6-fosfato" in name_lower:
            category = "Enzyme Assays"
        # Pyruvate kinase enzyme assay
        elif "piruvatoquinase" in name_lower or "pyruvate" in name_lower:
            category = "Enzyme Assays"
        # Blood bank / Coombs tests
        elif "coombs" in name_lower or "antiglobulina" in name_lower:
            category = "Blood Bank Tests"
        # PNH-related markers
        elif "hemoglobinúria" in name_lower or "hpn" in name_lower or "gpi" in name_lower:
            category = "Paroxysmal Nocturnal Hemoglobinuria"
        # Hemoglobin HPLC
        elif "hplc" in name_lower:
            category = "Hemoglobin HPLC"
        # Iron studies
        elif "hemossiderina" in name_lower:
            category = "Iron Studies"
        # Sample/product information
        elif "caracterização" in name_lower or "produto" in name_lower:
            category = "Sample Information"
        # Cell count: lymphoplasmacytic
        elif "linfoplasmocitárias" in name_lower:
            category = "Cell Counts"
        # Cell count: monocyte formula
        elif "monócitos" in name_lower and "fórmula" in name_lower:
            category = "Cell Counts"
        # Uncategorized test
        else:
            category = "Other"

        by_category[category].append((name, count))

    for category in sorted(by_category.keys()):
        logger.info(f"\n  {category}:")
        for name, count in by_category[category]:
            logger.info(f"    - {name} ({count} occurrences)")

    # Recommendations
    logger.info("\n" + "=" * 80)
    logger.info("RECOMMENDATIONS")
    logger.info("=" * 80)

    logger.info(f"""
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
    logger.info("\n" + "=" * 80)
    logger.info("MISSING UNITS ANALYSIS")
    logger.info("=" * 80)

    unknown_units = df[df["lab_unit_standardized"] == "$UNKNOWN$"]
    unit_combos = unknown_units.groupby(["lab_unit_raw"]).size().reset_index(name="count")
    unit_combos = unit_combos.sort_values("count", ascending=False)

    logger.info("\nUnits that need to be added to lab_specs.json:")
    for _, row in unit_combos.iterrows():
        # Skip rows with missing unit values
        if pd.notna(row["lab_unit_raw"]):
            logger.info(f"  - '{row['lab_unit_raw']}' ({row['count']} occurrences)")


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Analyze $UNKNOWN$ values in lab results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  search      Count and summarize unknown values (quick overview)
  analyze     Detailed row-by-row analysis with config lookup
  categorize  Categorize unknowns and provide recommendations
        """,
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["search", "analyze", "categorize"],
        default="search",
        help="Analysis mode (default: search)",
    )

    args = parser.parse_args()

    # Load data, bail on missing file
    try:
        df = load_data()
    except FileNotFoundError as e:
        logger.error(f"{e}")
        return 1

    # Dispatch to the selected mode
    if args.mode == "search":
        mode_search(df)
    elif args.mode == "analyze":
        mode_analyze(df)
    elif args.mode == "categorize":
        mode_categorize(df)

    return 0


if __name__ == "__main__":
    exit(main())
