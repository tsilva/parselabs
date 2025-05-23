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

def main():
    report = {}
    test_all_rows_have_dates_and_no_duplicates(report)
    test_lab_name_mappings_prefixes(report)
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