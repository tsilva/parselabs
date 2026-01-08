import pandas as pd
import hashlib
import json
from dotenv import dotenv_values
import os

# Load OUTPUT_PATH from .env
env = dotenv_values(".env")
OUTPUT_PATH = env.get("OUTPUT_PATH", "output")
ALL_FINAL_CSV = os.path.join(OUTPUT_PATH, "all.csv")

def test_all_rows_have_dates_and_no_duplicates(report):
    file = ALL_FINAL_CSV
    errors = []
    try:
        df = pd.read_csv(file)
        # 1. All rows have a non-empty date
        if not df['date'].notnull().all():
            errors.append("Some rows have null date")
        if not (df['date'].astype(str).str.strip() != '').all():
            errors.append("Some rows have empty date")
        # 2. No duplicate rows (hash all columns per row)
        row_hashes = df.apply(lambda row: hashlib.sha256(
            ('|'.join(row.astype(str))).encode('utf-8')
        ).hexdigest(), axis=1)
        duplicates = row_hashes[row_hashes.duplicated(keep=False)]
        if not duplicates.empty:
            for dup_hash in duplicates.unique():
                dup_indices = duplicates[duplicates == dup_hash].index.tolist()
                # For each duplicate group, collect info
                for idx in dup_indices:
                    row = df.loc[idx]
                    source_file = row.get('source_file', 'unknown')
                    report.setdefault(source_file, []).append(
                        f"Duplicate row at index {idx}: {row.to_dict()}"
                    )
    except Exception as e:
        errors.append(f"Exception: {e}")
    if errors:
        report.setdefault(file, []).extend(errors)

def test_lab_unit_percent_vs_lab_name(report):
    file = ALL_FINAL_CSV
    errors = []
    try:
        df = pd.read_csv(file)
        if 'lab_unit_primary' in df.columns and 'lab_name_standardized' in df.columns:
            mask = (df['lab_unit_primary'] == "%") & (~df['lab_name_standardized'].astype(str).str.endswith("(%)"))
            for idx in df[mask].index:
                row = df.loc[idx]
                source_file = row.get('source_file', 'unknown')
                lab_name_standardized = row.get('lab_name_standardized', '')
                report.setdefault(source_file, []).append(
                    f'Row at index {idx} (lab_name_standardized="{lab_name_standardized}") has lab_unit_primary="%" but lab_name_standardized="{lab_name_standardized}"'
                )
    except Exception as e:
        errors.append(f"Exception: {e}")
    if errors:
        report.setdefault(file, []).extend(errors)

def test_lab_unit_percent_value_range(report):
    file = ALL_FINAL_CSV
    errors = []
    try:
        df = pd.read_csv(file)
        if 'lab_unit_primary' in df.columns and 'value_primary' in df.columns:
            mask = (df['lab_unit_primary'] == "%") & (
                (df['value_primary'] < 0) | (df['value_primary'] > 100)
            )
            for idx in df[mask].index:
                row = df.loc[idx]
                source_file = row.get('source_file', 'unknown')
                val = row.get('value_primary')
                lab_name_standardized = row.get('lab_name_standardized', '')
                report.setdefault(source_file, []).append(
                    f'Row at index {idx} (lab_name_standardized="{lab_name_standardized}") has lab_unit_primary="%" but value_primary={val} (should be between 0 and 100)'
                )
    except Exception as e:
        errors.append(f"Exception: {e}")
    if errors:
        report.setdefault(file, []).extend(errors)

def test_lab_unit_boolean_value(report):
    file = ALL_FINAL_CSV
    errors = []
    try:
        df = pd.read_csv(file)
        if 'lab_unit_primary' in df.columns and 'value_primary' in df.columns:
            mask = (df['lab_unit_primary'] == "boolean") & (~df['value_primary'].isin([0, 1]))
            for idx in df[mask].index:
                row = df.loc[idx]
                source_file = row.get('source_file', 'unknown')
                val = row.get('value_primary')
                lab_name_standardized = row.get('lab_name_standardized', '')
                report.setdefault(source_file, []).append(
                    f'Row at index {idx} (lab_name_standardized="{lab_name_standardized}") has lab_unit_primary="boolean" but value_primary={val} (should be 0 or 1)'
                )
    except Exception as e:
        errors.append(f"Exception: {e}")
    if errors:
        report.setdefault(file, []).extend(errors)

def test_lab_name_enum_unit_consistency(report):
    file = ALL_FINAL_CSV
    errors = []
    try:
        df = pd.read_csv(file)
        if 'lab_name_standardized' in df.columns and 'lab_unit_primary' in df.columns:
            # Group by lab_name_standardized and collect unique units
            grouped = df.groupby('lab_name_standardized')['lab_unit_primary'].unique()
            for lab_name_standardized, units in grouped.items():
                units = [u for u in units if pd.notnull(u)]
                if len(units) > 1:
                    indices = df[df['lab_name_standardized'] == lab_name_standardized].index.tolist()
                    report.setdefault(file, []).append(
                        f'lab_name_standardized="{lab_name_standardized}" has inconsistent lab_unit_primary values: {units} (rows: {indices})'
                    )
    except Exception as e:
        errors.append(f"Exception: {e}")
    if errors:
        report.setdefault(file, []).extend(errors)

def test_lab_value_outliers_by_lab_name_enum(report):
    file = ALL_FINAL_CSV
    errors = []
    try:
        df = pd.read_csv(file)
        if 'value_primary' not in df.columns or 'lab_name_standardized' not in df.columns:
            return
        # Only consider rows with non-null value_primary
        df = df[pd.notnull(df['value_primary'])]
        for lab_name_standardized, group in df.groupby('lab_name_standardized'):
            # Find the most frequent lab_unit_primary
            if 'lab_unit_primary' in group.columns:
                unit_counts = group['lab_unit_primary'].value_counts()
                if unit_counts.empty:
                    continue
                most_freq_unit = unit_counts.idxmax()
                values = group[group['lab_unit_primary'] == most_freq_unit]['value_primary']
            else:
                values = group['value_primary']
                most_freq_unit = 'N/A'

            # Only consider numeric values
            values = pd.to_numeric(values, errors='coerce').dropna()
            if len(values) < 5:
                continue  # skip small groups
            mean = values.mean()
            std = values.std()
            if std == 0 or pd.isnull(std):
                continue

            if 'lab_unit_primary' in group.columns:
                outliers = group[
                    (group['lab_unit_primary'] == most_freq_unit) &
                    (
                        (group['value_primary'] > mean + 3 * std) |
                        (group['value_primary'] < mean - 3 * std)
                    )
                ]
            else:
                outliers = group[
                    (group['value_primary'] > mean + 3 * std) |
                    (group['value_primary'] < mean - 3 * std)
                ]

            if not outliers.empty:
                source_files = set(outliers['source_file'].dropna().astype(str))
                outlier_values = outliers['value_primary'].tolist()
                # Use "page_number" column for page numbers
                if 'page_number' in outliers.columns:
                    page_numbers = outliers['page_number'].tolist()
                else:
                    page_numbers = ['unknown'] * len(outlier_values)
                report.setdefault(file, []).append(
                    f'lab_name_standardized="{lab_name_standardized}", lab_unit_primary="{most_freq_unit}" has outlier value_primary (>3 std from mean {mean:.2f}±{std:.2f}) '
                    f'in files: {list(sorted(source_files))} outlier values: {outlier_values} page numbers: {page_numbers}'
                )
    except Exception as e:
        errors.append(f"Exception: {e}")
    if errors:
        report.setdefault(file, []).extend(errors)

def test_unique_date_lab_name_enum(report):
    file = ALL_FINAL_CSV
    errors = []
    try:
        df = pd.read_csv(file)
        if 'date' in df.columns and 'lab_name_standardized' in df.columns:
            # Check for duplicate (date, lab_name_standardized) pairs
            duplicates = df.duplicated(subset=['date', 'lab_name_standardized'], keep=False)
            if duplicates.any():
                dup_rows = df[duplicates][['date', 'lab_name_standardized']]
                for idx, row in dup_rows.iterrows():
                    source_file = df.loc[idx].get('source_file', 'unknown')
                    report.setdefault(source_file, []).append(
                        f"Duplicate (date, lab_name_standardized) at index {idx}: date={row['date']}, lab_name_standardized={row['lab_name_standardized']}"
                    )
    except Exception as e:
        errors.append(f"Exception: {e}")
    if errors:
        report.setdefault(file, []).extend(errors)

def main():
    report = {}
    test_all_rows_have_dates_and_no_duplicates(report)
    test_lab_unit_percent_vs_lab_name(report)
    test_lab_unit_percent_value_range(report)
    test_lab_unit_boolean_value(report)
    test_lab_name_enum_unit_consistency(report)
    test_lab_value_outliers_by_lab_name_enum(report)
    test_unique_date_lab_name_enum(report)
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