#!/usr/bin/env python3
"""
Fix encoding issues in lab_specs.json by converting Unicode escape sequences
to actual UTF-8 characters.

This script:
1. Creates a backup of the original file
2. Reads the JSON file (Python automatically decodes escape sequences)
3. Writes it back with ensure_ascii=False to use actual UTF-8 characters
4. Validates the result

Usage:
    python utils/fix_lab_specs_encoding.py
"""

import json
import shutil
from pathlib import Path


def fix_encoding(lab_specs_path: Path, create_backup: bool = True) -> None:
    """
    Fix encoding in lab_specs.json by converting escape sequences to UTF-8 characters.

    Args:
        lab_specs_path: Path to the lab_specs.json file
        create_backup: Whether to create a backup before modifying
    """
    if not lab_specs_path.exists():
        raise FileNotFoundError(f"File not found: {lab_specs_path}")

    # Create backup
    if create_backup:
        backup_path = lab_specs_path.with_suffix('.json.backup')
        shutil.copy2(lab_specs_path, backup_path)
        print(f"✓ Created backup: {backup_path}")

    # Read and parse JSON (Python automatically decodes escape sequences)
    print(f"Reading {lab_specs_path}...")
    with open(lab_specs_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Write back with actual UTF-8 characters (ensure_ascii=False)
    print(f"Writing back with UTF-8 encoding...")
    with open(lab_specs_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write('\n')  # Add trailing newline

    # Validate by reading it back
    print(f"Validating...")
    with open(lab_specs_path, 'r', encoding='utf-8') as f:
        validated_data = json.load(f)

    # Verify data integrity
    if data != validated_data:
        raise ValueError("Data integrity check failed - data changed after write!")

    print(f"✓ Successfully fixed encoding in {lab_specs_path}")
    print(f"✓ Data integrity verified")


def main():
    # Get the lab_specs.json path relative to this script
    utils_dir = Path(__file__).parent
    project_dir = utils_dir.parent
    lab_specs_path = project_dir / 'config' / 'lab_specs.json'

    try:
        fix_encoding(lab_specs_path)
        print("\n✓ Encoding fix completed successfully!")
        print("\nNext steps:")
        print("  1. Run: python utils/validate_lab_specs_schema.py")
        print("  2. Verify manually: grep -n 'μg/L' config/lab_specs.json | head -5")
        print("  3. Check no escapes remain: grep '\\\\u00' config/lab_specs.json")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise


if __name__ == '__main__':
    main()
