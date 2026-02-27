"""One-off migration: strip standardization fields from cached extraction JSONs and per-PDF CSVs.

After moving standardization from extraction to the CSV merge phase, this script removes
leftover `lab_name_standardized`, `lab_unit_standardized`, `lab_name`, and `unit` fields
from cached per-page JSON files and per-PDF CSVs so they only contain raw extraction data.

Usage:
    python utils/migrate_strip_standardization.py <output_dir>
    python utils/migrate_strip_standardization.py <output_dir> --dry-run

Example:
    python utils/migrate_strip_standardization.py /path/to/profile/output
    python utils/migrate_strip_standardization.py /path/to/profile/output --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Fields to remove from JSON lab_results entries
JSON_FIELDS_TO_STRIP = {"lab_name_standardized", "lab_unit_standardized", "lab_name", "unit"}

# Columns to remove from per-PDF CSVs (not all.csv — that gets regenerated)
CSV_COLUMNS_TO_STRIP = {"lab_name_standardized", "lab_unit_standardized"}


def strip_json_file(json_path: Path, dry_run: bool) -> bool:
    """Strip standardization fields from a single JSON file. Returns True if modified."""

    data = json.loads(json_path.read_text(encoding="utf-8"))
    lab_results = data.get("lab_results", [])

    # Check if any results have fields to strip
    modified = False
    for result in lab_results:
        for field in JSON_FIELDS_TO_STRIP:
            if field in result:
                if not dry_run:
                    del result[field]
                modified = True

    # Write back if modified
    if modified and not dry_run:
        json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    return modified


def strip_csv_file(csv_path: Path, dry_run: bool) -> bool:
    """Strip standardization columns from a per-PDF CSV. Returns True if modified."""

    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except pd.errors.EmptyDataError:
        return False

    cols_to_drop = [col for col in CSV_COLUMNS_TO_STRIP if col in df.columns]

    if not cols_to_drop:
        return False

    if not dry_run:
        df = df.drop(columns=cols_to_drop)
        df.to_csv(csv_path, index=False, encoding="utf-8")

    return True


def main():
    parser = argparse.ArgumentParser(description="Strip standardization fields from cached extraction data")
    parser.add_argument("output_dir", type=Path, help="Profile output directory to migrate")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without modifying files")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if not output_dir.is_dir():
        print(f"Error: {output_dir} is not a directory")
        sys.exit(1)

    prefix = "[DRY RUN] " if args.dry_run else ""

    # Find all per-document subdirectories (pattern: {stem}_{hash}/)
    doc_dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name != "logs"]

    json_modified = 0
    json_total = 0
    csv_modified = 0
    csv_total = 0

    for doc_dir in sorted(doc_dirs):
        # Process JSON files
        for json_path in sorted(doc_dir.glob("*.json")):
            json_total += 1
            if strip_json_file(json_path, args.dry_run):
                json_modified += 1
                print(f"{prefix}Stripped JSON: {json_path.relative_to(output_dir)}")

        # Process per-PDF CSVs (not all.csv)
        for csv_path in sorted(doc_dir.glob("*.csv")):
            csv_total += 1
            if strip_csv_file(csv_path, args.dry_run):
                csv_modified += 1
                print(f"{prefix}Stripped CSV:  {csv_path.relative_to(output_dir)}")

    # Summary
    print(f"\n{prefix}Summary:")
    print(f"  JSON files: {json_modified}/{json_total} modified")
    print(f"  CSV files:  {csv_modified}/{csv_total} modified")

    if args.dry_run and (json_modified or csv_modified):
        print("\nRe-run without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
