import pandas as pd
import hashlib
import json

def test_all_rows_have_dates_and_no_duplicates():
    df = pd.read_csv("output/all.csv")
    # 1. All rows have a non-empty date
    assert df['date'].notnull().all(), "Some rows have null date"
    assert (df['date'].astype(str).str.strip() != '').all(), "Some rows have empty date"

    # 2. No duplicate rows (hash all columns per row)
    row_hashes = df.apply(lambda row: hashlib.sha256(
        ('|'.join(row.astype(str))).encode('utf-8')
    ).hexdigest(), axis=1)
    assert row_hashes.is_unique, "There are duplicate rows in all.csv"

def test_lab_name_mappings_prefixes():
    with open("config/lab_names_mappings.json", encoding="utf-8") as f:
        mappings = json.load(f)
    for k, v in mappings.items():
        if k.startswith("blood-"):
            assert v.startswith("Blood - "), f"Key '{k}' must have value starting with 'Blood - ', got '{v}'"
        elif k.startswith("urine-"):
            assert v.startswith("Urine - "), f"Key '{k}' must have value starting with 'Urine - ', got '{v}'"
        elif k.startswith("feces-"):
            assert v.startswith("Feces - "), f"Key '{k}' must have value starting with 'Feces - ', got '{v}'"

test_all_rows_have_dates_and_no_duplicates()
test_lab_name_mappings_prefixes()