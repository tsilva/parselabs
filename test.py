import hashlib
import json
import sys
from collections import defaultdict
from functools import wraps

import pandas as pd
from dotenv import dotenv_values

# Load OUTPUT_PATH from .env
env = dotenv_values(".env.local")
OUTPUT_PATH = env.get("OUTPUT_PATH", "output")
ALL_FINAL_CSV = f"{OUTPUT_PATH}/all.csv"


def integrity_test(func):
    """Decorator that handles common test boilerplate: file loading, error handling, and reporting."""

    @wraps(func)
    def wrapper(report):
        file = ALL_FINAL_CSV
        errors = []

        try:
            df = pd.read_csv(file)
            func(df, report, errors)
        except FileNotFoundError as e:
            errors.append(f"Results file not found: {e}")
        except pd.errors.EmptyDataError as e:
            errors.append(f"Results file is empty: {e}")
        except pd.errors.ParserError as e:
            errors.append(f"Failed to parse CSV: {e}")
        except PermissionError as e:
            errors.append(f"Permission denied reading results file: {e}")
        # Let unexpected exceptions (AttributeError, TypeError, etc.) propagate
        # These indicate programming bugs, not test data issues

        # Record errors against the file in the report
        if errors:
            report.setdefault(file, []).extend(errors)

    return wrapper


@integrity_test
def test_all_rows_have_dates_and_no_duplicates(df, report, errors):

    # Check all rows have a non-null date
    if not df["date"].notnull().all():
        errors.append("Some rows have null date")

    # Check all rows have a non-empty date string
    if not (df["date"].astype(str).str.strip() != "").all():
        errors.append("Some rows have empty date")

    # Check for duplicate rows by hashing all columns per row
    row_hashes = df.apply(
        lambda row: hashlib.sha256(("|".join(row.astype(str))).encode("utf-8")).hexdigest(),
        axis=1,
    )
    duplicates = row_hashes[row_hashes.duplicated(keep=False)]

    # Report each duplicate row grouped by source file
    if not duplicates.empty:
        for dup_hash in duplicates.unique():
            dup_indices = duplicates[duplicates == dup_hash].index.tolist()
            for idx in dup_indices:
                row = df.loc[idx]
                source_file = row.get("source_file", "unknown")
                report.setdefault(source_file, []).append(f"Duplicate row at index {idx}: {row.to_dict()}")


@integrity_test
def test_lab_unit_percent_vs_lab_name(df, report, errors):

    # Guard: Skip if required columns are missing
    if "unit" not in df.columns or "lab_name" not in df.columns:
        return

    # Find rows where unit is % but lab name doesn't end with (%)
    mask = (df["unit"] == "%") & (~df["lab_name"].astype(str).str.endswith("(%)"))

    for idx in df[mask].index:
        row = df.loc[idx]
        source_file = row.get("source_file", "unknown")
        lab_name = row.get("lab_name", "")
        report.setdefault(source_file, []).append(f'Row at index {idx} (lab_name="{lab_name}") has unit="%" but lab_name="{lab_name}"')


@integrity_test
def test_lab_unit_percent_value_range(df, report, errors):

    # Guard: Skip if required columns are missing
    if "unit" not in df.columns or "value" not in df.columns:
        return

    # Find percentage values outside valid 0-100 range
    mask = (df["unit"] == "%") & ((df["value"] < 0) | (df["value"] > 100))

    for idx in df[mask].index:
        row = df.loc[idx]
        source_file = row.get("source_file", "unknown")
        val = row.get("value")
        lab_name = row.get("lab_name", "")
        report.setdefault(source_file, []).append(f'Row at index {idx} (lab_name="{lab_name}") has unit="%" but value={val} (should be between 0 and 100)')


@integrity_test
def test_lab_unit_boolean_value(df, report, errors):

    # Guard: Skip if required columns are missing
    if "unit" not in df.columns or "value" not in df.columns:
        return

    # Find boolean labs with values other than 0 or 1
    mask = (df["unit"] == "boolean") & (~df["value"].isin([0, 1]))

    for idx in df[mask].index:
        row = df.loc[idx]
        source_file = row.get("source_file", "unknown")
        val = row.get("value")
        lab_name = row.get("lab_name", "")
        report.setdefault(source_file, []).append(f'Row at index {idx} (lab_name="{lab_name}") has unit="boolean" but value={val} (should be 0 or 1)')


@integrity_test
def test_lab_name_unit_consistency(df, report, errors):

    # Guard: Skip if required columns are missing
    if "lab_name" not in df.columns or "unit" not in df.columns:
        return

    # Check each lab name uses a single consistent unit
    grouped = df.groupby("lab_name")["unit"].unique()
    for lab_name, units in grouped.items():
        units = [u for u in units if pd.notnull(u)]

        # Flag labs with multiple different units
        if len(units) > 1:
            indices = df[df["lab_name"] == lab_name].index.tolist()
            errors.append(f'lab_name="{lab_name}" has inconsistent unit values: {units} (rows: {indices})')


@integrity_test
def test_lab_value_outliers_by_lab_name(df, report, errors):

    # Guard: Skip if required columns are missing
    if "value" not in df.columns or "lab_name" not in df.columns:
        return

    # Filter to rows with non-null values
    df = df[pd.notnull(df["value"])]

    for lab_name, group in df.groupby("lab_name"):
        # Determine most frequent unit to compare against
        if "unit" in group.columns:
            unit_counts = group["unit"].value_counts()
            if unit_counts.empty:
                continue
            most_freq_unit = unit_counts.idxmax()
            values = group[group["unit"] == most_freq_unit]["value"]
        else:
            values = group["value"]
            most_freq_unit = "N/A"

        # Convert to numeric and compute statistics
        values = pd.to_numeric(values, errors="coerce").dropna()

        # Guard: Need at least 5 values for meaningful statistics
        if len(values) < 5:
            continue

        mean = values.mean()
        std = values.std()

        # Guard: Skip if no variation in values
        if std == 0 or pd.isnull(std):
            continue

        # Find values more than 3 standard deviations from mean
        if "unit" in group.columns:
            outliers = group[(group["unit"] == most_freq_unit) & ((group["value"] > mean + 3 * std) | (group["value"] < mean - 3 * std))]
        else:
            outliers = group[(group["value"] > mean + 3 * std) | (group["value"] < mean - 3 * std)]

        # Report outliers with source file context
        if not outliers.empty:
            source_files = set(outliers["source_file"].dropna().astype(str))
            outlier_values = outliers["value"].tolist()
            page_numbers = outliers["page_number"].tolist() if "page_number" in outliers.columns else ["unknown"] * len(outlier_values)
            errors.append(
                f'lab_name="{lab_name}", unit="{most_freq_unit}" has outlier value (>3 std from mean {mean:.2f}±{std:.2f}) in files: {list(sorted(source_files))} outlier values: {outlier_values} page numbers: {page_numbers}'
            )


@integrity_test
def test_unique_date_lab_name(df, report, errors):

    # Guard: Skip if required columns are missing
    if "date" not in df.columns or "lab_name" not in df.columns:
        return

    # Check for duplicate (date, lab_name) pairs
    duplicates = df.duplicated(subset=["date", "lab_name"], keep=False)

    # Report each duplicate pair with source file context
    if duplicates.any():
        dup_df = df[duplicates]
        for row in dup_df.itertuples():
            source_file = getattr(row, "source_file", "unknown") or "unknown"
            date_val = getattr(row, "date", "")
            lab_name = getattr(row, "lab_name", "")
            report.setdefault(source_file, []).append(f"Duplicate (date, lab_name) at index {row.Index}: date={date_val}, lab_name={lab_name}")


def test_loinc_critical_codes(report):
    """Ensure critical LOINC codes are correct."""
    file = "config/lab_specs.json"
    errors = []

    try:
        with open(file, "r") as f:
            config = json.load(f)

        # Expected LOINC codes for critical tests
        expected_codes = {
            "Blood - Alpha-1-Antitrypsin (AAT)": "1825-9",
            "Blood - Alkaline Phosphatase (ALP)": "6768-6",
            "Blood - Bilirubin Total": "1975-2",
            "Blood - Albumin (%)": "13980-8",
            "Blood - Alpha-1 Globulins (%)": "13978-2",
        }

        # Verify each critical test has the expected LOINC code
        for test_name, expected_code in expected_codes.items():
            if test_name in config:
                actual_code = config[test_name].get("loinc_code")
                if actual_code != expected_code:
                    errors.append(f"{test_name} code should be {expected_code}, got {actual_code}")

        # Hemolysis tests should NOT share Bilirubin's LOINC code 1975-2
        hemolysis_tests = [
            "Blood - Hemolysis (total, immediate) (%)",
            "Blood - Hemolysis (total, after incubation) (%)",
            "Blood - Hemolysis (initial, immediate) (%)",
            "Blood - Hemolysis (initial, after incubation) (%)",
        ]
        for test_name in hemolysis_tests:
            if test_name in config:
                code = config[test_name].get("loinc_code")
                if code == "1975-2":
                    errors.append(f"{test_name} should not share code 1975-2 with Bilirubin")

    except Exception as e:
        errors.append(f"Exception: {e}")
    if errors:
        report.setdefault(file, []).extend(errors)


def test_no_critical_loinc_duplicates(report):
    """Ensure completely different tests don't share LOINC codes."""
    file = "config/lab_specs.json"
    errors = []

    try:
        with open(file, "r") as f:
            config = json.load(f)

        # Build reverse mapping: LOINC code -> list of lab names
        loinc_to_tests = defaultdict(list)
        for name, spec in config.items():
            # Skip meta keys like _relationships
            if name.startswith("_"):
                continue
            code = spec.get("loinc_code")
            if code:
                loinc_to_tests[code].append(name)

        # Check for known bad duplications between unrelated test types
        critical_pairs = [
            ("Alkaline Phosphatase", "Alpha-1-Antitrypsin"),
            ("Bilirubin", "Hemolysis"),
            ("Albumin (%)", "Alpha-1 Globulin"),
        ]

        for code, tests in loinc_to_tests.items():
            for pair in critical_pairs:
                has_first = any(pair[0] in t for t in tests)
                has_second = any(pair[1] in t for t in tests)
                if has_first and has_second:
                    errors.append(f"LOINC {code} incorrectly shared between {pair[0]} and {pair[1]}: {tests}")

    except Exception as e:
        errors.append(f"Exception: {e}")
    if errors:
        report.setdefault(file, []).extend(errors)


def test_lab_specs_schema(report):
    """Validate lab_specs.json schema and completeness."""
    file = "config/lab_specs.json"
    errors = []

    try:
        sys.path.insert(0, "utils")
        from validate_lab_specs_schema import LabSpecsValidator

        validator = LabSpecsValidator()
        validator.validate()

        # Collect any schema validation errors
        if validator.errors:
            errors.extend(validator.errors)
    except Exception as e:
        errors.append(f"Exception during schema validation: {e}")

    # Record errors against the config file in the report
    if errors:
        report.setdefault(file, []).extend(errors)


def main():
    report = {}

    # Schema validation first (validates config structure)
    test_lab_specs_schema(report)

    # Then LOINC-specific validations
    test_loinc_critical_codes(report)
    test_no_critical_loinc_duplicates(report)

    # Then data integrity tests
    test_all_rows_have_dates_and_no_duplicates(report)
    test_lab_unit_percent_vs_lab_name(report)
    test_lab_unit_percent_value_range(report)
    test_lab_unit_boolean_value(report)
    test_lab_name_unit_consistency(report)
    test_lab_value_outliers_by_lab_name(report)
    test_unique_date_lab_name(report)

    # Print summary report
    print("\n=== Integrity Report ===")
    if not report:
        print("All checks passed.")
    else:
        for file, errors in report.items():
            print(f"\nFile: {file}")
            for err in errors:
                print(f"  - {err}")


if __name__ == "__main__":
    main()
