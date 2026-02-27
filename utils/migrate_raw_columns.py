#!/usr/bin/env python3
"""Migrate existing output data from _raw suffix to raw_ prefix column names.

Renames keys/columns in JSON, per-document CSV, and all.csv files.
Idempotent: skips files that already use the new naming convention.

Usage:
    python utils/migrate_raw_columns.py --profile tsilva
    python utils/migrate_raw_columns.py --profile tsilva --dry-run
    python utils/migrate_raw_columns.py  # process all profiles
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from labs_parser.config import ProfileConfig  # noqa: E402

logger = logging.getLogger(__name__)

# JSON key renames (inside lab_results arrays)
_JSON_RENAMES = {
    "lab_name_raw": "raw_lab_name",
    "value_raw": "raw_value",
    "lab_unit_raw": "raw_lab_unit",
    "reference_min_raw": "raw_reference_min",
    "reference_max_raw": "raw_reference_max",
    "reference_range": "raw_reference_range",
    "comments": "raw_comments",
}

# Per-document CSV column renames (internal names)
_CSV_INTERNAL_RENAMES = {
    "lab_name_raw": "raw_lab_name",
    "value_raw": "raw_value",
    "lab_unit_raw": "raw_lab_unit",
    "reference_min_raw": "raw_reference_min",
    "reference_max_raw": "raw_reference_max",
    "reference_range": "raw_reference_range",
    "comments": "raw_comments",
}

# all.csv column renames (export names)
_CSV_EXPORT_RENAMES = {
    "lab_name_raw": "raw_lab_name",
    "value_raw": "raw_value",
    "unit_raw": "raw_unit",
}


def _migrate_json_file(json_path: Path, dry_run: bool) -> bool:
    """Migrate a single JSON extraction file. Returns True if changes were made."""

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lab_results = data.get("lab_results", [])
    if not lab_results:
        return False

    changed = False
    for result in lab_results:
        if not isinstance(result, dict):
            continue
        for old_key, new_key in _JSON_RENAMES.items():
            if old_key in result and new_key not in result:
                if not dry_run:
                    result[new_key] = result.pop(old_key)
                changed = True

    if changed and not dry_run:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    return changed


def _migrate_csv_file(csv_path: Path, renames: dict, dry_run: bool) -> bool:
    """Migrate a single CSV file. Returns True if changes were made."""

    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except pd.errors.EmptyDataError:
        return False

    # Filter renames to only columns that exist and aren't already renamed
    applicable = {old: new for old, new in renames.items() if old in df.columns and new not in df.columns}

    if not applicable:
        return False

    if not dry_run:
        df = df.rename(columns=applicable)
        df.to_csv(csv_path, index=False, encoding="utf-8")

    return True


def migrate_profile(output_path: Path, dry_run: bool) -> dict:
    """Migrate all output files for a profile. Returns stats."""

    stats = {"json_migrated": 0, "csv_migrated": 0, "json_skipped": 0, "csv_skipped": 0}

    # Find all JSON extraction files
    for json_path in sorted(output_path.rglob("*.json")):
        # Skip lab_specs.json copies
        if json_path.name == "lab_specs.json":
            continue
        if _migrate_json_file(json_path, dry_run):
            stats["json_migrated"] += 1
            logger.info(f"  {'[DRY RUN] ' if dry_run else ''}Migrated: {json_path.relative_to(output_path)}")
        else:
            stats["json_skipped"] += 1

    # Find all per-document CSV files (in subdirectories)
    for csv_path in sorted(output_path.rglob("*.csv")):
        # Determine if this is all.csv (export names) or per-doc CSV (internal names)
        if csv_path.name == "all.csv" and csv_path.parent == output_path:
            renames = _CSV_EXPORT_RENAMES
        elif csv_path.parent != output_path:
            renames = _CSV_INTERNAL_RENAMES
        else:
            continue

        if _migrate_csv_file(csv_path, renames, dry_run):
            stats["csv_migrated"] += 1
            logger.info(f"  {'[DRY RUN] ' if dry_run else ''}Migrated: {csv_path.relative_to(output_path)}")
        else:
            stats["csv_skipped"] += 1

    return stats


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Migrate _raw suffix columns to raw_ prefix")
    parser.add_argument("--profile", "-p", help="Profile name (omit to process all profiles)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without modifying files")
    args = parser.parse_args()

    if args.profile:
        profile_path = ProfileConfig.find_path(args.profile)
        if not profile_path:
            logger.error(f"Profile '{args.profile}' not found")
            return 1
        profiles = [ProfileConfig.from_file(profile_path)]
    else:
        # Find all profiles
        profiles_dir = Path("profiles")
        if not profiles_dir.exists():
            logger.error("No profiles directory found")
            return 1
        profiles = []
        for p in sorted(profiles_dir.glob("*")):
            if p.suffix in (".yaml", ".yml", ".json") and p.stem != "_template":
                profiles.append(ProfileConfig.from_file(p))

    total_stats = {"json_migrated": 0, "csv_migrated": 0, "json_skipped": 0, "csv_skipped": 0}

    for profile in profiles:
        output_path = profile.output_path
        if not output_path.exists():
            logger.warning(f"Output path does not exist for profile '{profile.name}': {output_path}")
            continue

        logger.info(f"\nMigrating profile '{profile.name}' ({output_path})")
        stats = migrate_profile(output_path, args.dry_run)

        for key in total_stats:
            total_stats[key] += stats[key]

    # Summary
    prefix = "[DRY RUN] " if args.dry_run else ""
    logger.info(f"\n{prefix}Migration complete:")
    logger.info(f"  JSON files migrated: {total_stats['json_migrated']}")
    logger.info(f"  JSON files skipped (already migrated): {total_stats['json_skipped']}")
    logger.info(f"  CSV files migrated: {total_stats['csv_migrated']}")
    logger.info(f"  CSV files skipped (already migrated): {total_stats['csv_skipped']}")

    return 0


if __name__ == "__main__":
    exit(main())
