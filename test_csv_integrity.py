import pandas as pd
import hashlib

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

test_all_rows_have_dates_and_no_duplicates()