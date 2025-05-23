import pandas as pd
import hashlib
import json

def test_all_rows_have_dates_and_no_duplicates(report):
    file = "output/all.csv"
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

def test_lab_name_mappings_prefixes(report):
    file = "config/lab_names_mappings.json"
    errors = []
    try:
        with open(file, encoding="utf-8") as f:
            mappings = json.load(f)
        for k, v in mappings.items():
            if k.startswith("blood-"):
                if not v.startswith("Blood - "):
                    errors.append(f"Key '{k}' must have value starting with 'Blood - ', got '{v}'")
            elif k.startswith("urine-"):
                if not v.startswith("Urine - "):
                    errors.append(f"Key '{k}' must have value starting with 'Urine - ', got '{v}'")
            elif k.startswith("feces-"):
                if not v.startswith("Feces - "):
                    errors.append(f"Key '{k}' must have value starting with 'Feces - ', got '{v}'")
    except Exception as e:
        errors.append(f"Exception: {e}")
    if errors:
        report.setdefault(file, []).extend(errors)

def test_lab_unit_percent_vs_lab_name(report):
    file = "output/all.csv"
    errors = []
    try:
        df = pd.read_csv(file)
        mask = (df['lab_unit_enum'] == "%") & (~df['lab_name_enum'].astype(str).str.endswith("(%)"))
        for idx in df[mask].index:
            row = df.loc[idx]
            source_file = row.get('source_file', 'unknown')
            lab_name = row.get('lab_name_enum', '')
            report.setdefault(source_file, []).append(
                f'Row at index {idx} (lab_name="{lab_name}") has lab_unit_enum="%" but lab_name_enum="{lab_name}"'
            )
    except Exception as e:
        errors.append(f"Exception: {e}")
    if errors:
        report.setdefault(file, []).extend(errors)

def test_lab_names_mapping_percent_suffix(report):
    file = "config/lab_names_mappings.json"
    errors = []
    try:
        with open(file, encoding="utf-8") as f:
            mappings = json.load(f)
        for k, v in mappings.items():
            if "percent" in k and not str(v).strip().endswith("(%)"):
                errors.append(f'Key "{k}" has value "{v}" which does not end with "(%)"')
    except Exception as e:
        errors.append(f"Exception: {e}")
    if errors:
        report.setdefault(file, []).extend(errors)

def test_lab_unit_not_empty(report):
    file = "output/all.csv"
    errors = []
    try:
        df = pd.read_csv(file)
        mask = df['lab_unit'].isnull() | (df['lab_unit'].astype(str).str.strip() == "")
        for idx in df[mask].index:
            row = df.loc[idx]
            source_file = row.get('source_file', 'unknown')
            report.setdefault(source_file, []).append(
                f'Row at index {idx} has empty lab_unit'
            )
    except Exception as e:
        errors.append(f"Exception: {e}")
    if errors:
        report.setdefault(file, []).extend(errors)

def test_lab_unit_percent_value_range(report):
    file = "output/all.csv"
    errors = []
    try:
        df = pd.read_csv(file)
        mask = (df['lab_unit_enum'] == "%") & (
            (df['lab_value'] < 0) | (df['lab_value'] > 100)
        )
        for idx in df[mask].index:
            row = df.loc[idx]
            source_file = row.get('source_file', 'unknown')
            val = row.get('lab_value')
            lab_name = row.get('lab_name_enum', '')
            report.setdefault(source_file, []).append(
                f'Row at index {idx} (lab_name="{lab_name}") has lab_unit_enum="%" but lab_value={val} (should be between 0 and 100)'
            )
    except Exception as e:
        errors.append(f"Exception: {e}")
    if errors:
        report.setdefault(file, []).extend(errors)

def test_lab_unit_boolean_value(report):
    file = "output/all.csv"
    errors = []
    try:
        df = pd.read_csv(file)
        mask = (df['lab_unit_enum'] == "boolean") & (~df['lab_value'].isin([0, 1]))
        for idx in df[mask].index:
            row = df.loc[idx]
            source_file = row.get('source_file', 'unknown')
            val = row.get('lab_value')
            lab_name = row.get('lab_name_enum', '')
            report.setdefault(source_file, []).append(
                f'Row at index {idx} (lab_name="{lab_name}") has lab_unit_enum="boolean" but lab_value={val} (should be 0 or 1)'
            )
    except Exception as e:
        errors.append(f"Exception: {e}")
    if errors:
        report.setdefault(file, []).extend(errors)

def test_lab_name_enum_unit_consistency(report):
    file = "output/all.csv"
    errors = []
    try:
        df = pd.read_csv(file)
        # Group by lab_name_enum and collect unique units
        grouped = df.groupby('lab_name_enum')['lab_unit_final'].unique()
        for lab_name_enum, units in grouped.items():
            units = [u for u in units if pd.notnull(u)]
            if len(units) > 1:
                indices = df[df['lab_name_enum'] == lab_name_enum].index.tolist()
                report.setdefault(file, []).append(
                    f'lab_name_enum="{lab_name_enum}" has inconsistent lab_unit_enum values: {units} (rows: {indices})'
                )
    except Exception as e:
        errors.append(f"Exception: {e}")
    if errors:
        report.setdefault(file, []).extend(errors)

def main():
    report = {}
    test_all_rows_have_dates_and_no_duplicates(report)
    test_lab_name_mappings_prefixes(report)
    test_lab_unit_percent_vs_lab_name(report)
    test_lab_names_mapping_percent_suffix(report)
    test_lab_unit_not_empty(report)
    test_lab_unit_percent_value_range(report)
    test_lab_unit_boolean_value(report)
    test_lab_name_enum_unit_consistency(report)
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