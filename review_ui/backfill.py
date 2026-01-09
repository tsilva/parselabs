"""
Backfill result_index column in all.csv from JSON files.

This script matches CSV rows to JSON entries by comparing key fields,
then adds the result_index column without re-running the full extraction pipeline.
"""

import os
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def backfill_result_index(output_path: str = None):
    """
    Backfill result_index column in all.csv by matching to JSON entries.

    Args:
        output_path: Path to output directory. If None, uses OUTPUT_PATH env var.
    """
    if output_path is None:
        output_path = os.getenv('OUTPUT_PATH', './output')

    output_path = Path(output_path)
    csv_path = output_path / "all.csv"

    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return

    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)

    if 'result_index' in df.columns and df['result_index'].notna().all():
        print("result_index column already fully populated.")
        return

    # Add result_index column if missing
    if 'result_index' not in df.columns:
        df['result_index'] = pd.NA

    # Group by source file and page to process each JSON
    matched = 0
    unmatched = 0

    for (source_file, page_number), group in df.groupby(['source_file', 'page_number']):
        # Construct JSON path
        stem = source_file.rsplit('.', 1)[0] if '.' in str(source_file) else source_file
        page_str = f"{int(page_number):03d}" if pd.notna(page_number) else "001"
        json_path = output_path / stem / f"{stem}.{page_str}.json"

        if not json_path.exists():
            print(f"  JSON not found: {json_path}")
            unmatched += len(group)
            continue

        try:
            data = json.loads(json_path.read_text(encoding='utf-8'))
            lab_results = data.get('lab_results', [])
        except Exception as e:
            print(f"  Error reading {json_path}: {e}")
            unmatched += len(group)
            continue

        # Match each CSV row to a JSON entry by comparing key fields
        for idx in group.index:
            row = df.loc[idx]

            # Skip if already has result_index
            if pd.notna(row.get('result_index')):
                matched += 1
                continue

            # Try to find matching entry in JSON
            best_match_idx = None
            best_match_score = 0

            for json_idx, json_entry in enumerate(lab_results):
                score = 0

                # Compare lab_name_raw (most reliable)
                if row.get('lab_name_raw') == json_entry.get('lab_name_raw') or \
                   row.get('lab_name_raw') == json_entry.get('test_name'):
                    score += 10

                # Compare value_raw
                csv_val = str(row.get('value_raw', '')) if pd.notna(row.get('value_raw')) else ''
                json_val = str(json_entry.get('value_raw', '') or json_entry.get('value', '') or '')
                if csv_val and json_val and csv_val == json_val:
                    score += 5

                # Compare unit
                csv_unit = str(row.get('lab_unit_raw', '')) if pd.notna(row.get('lab_unit_raw')) else ''
                json_unit = str(json_entry.get('lab_unit_raw', '') or json_entry.get('unit', '') or '')
                if csv_unit and json_unit and csv_unit == json_unit:
                    score += 3

                # Compare source_text
                csv_src = str(row.get('source_text', '')) if pd.notna(row.get('source_text')) else ''
                json_src = str(json_entry.get('source_text', '') or '')
                if csv_src and json_src and csv_src == json_src:
                    score += 5

                if score > best_match_score:
                    best_match_score = score
                    best_match_idx = json_idx

            if best_match_idx is not None and best_match_score >= 10:
                df.at[idx, 'result_index'] = best_match_idx
                matched += 1
            else:
                unmatched += 1
                print(f"  No match for row {idx}: {row.get('lab_name_raw', 'unknown')}")

    # Save updated CSV
    print(f"\nMatched: {matched}, Unmatched: {unmatched}")

    if matched > 0:
        # Ensure result_index is integer type
        df['result_index'] = df['result_index'].astype('Int64')
        df.to_csv(csv_path, index=False)
        print(f"Saved updated CSV: {csv_path}")
    else:
        print("No matches found, CSV not updated.")


if __name__ == "__main__":
    backfill_result_index()
