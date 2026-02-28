#!/usr/bin/env python3
"""Remove deprecated fields from existing per-page JSON extraction files.

Strips fields that were removed from the Pydantic models:
- raw_is_abnormal (never used downstream)
- raw_reference_notes (almost always null, never used)
- raw_source_text (debug-only bulk)
- review_confidence (ghost field, not in model)

Also strips default-valued review fields (only when at their defaults):
- review_needed: false → removed (validation sets this on the DataFrame, not JSON)
- review_reason: null → removed (same)
- review_status: null → removed (viewer writes non-null values directly)
- review_completed_at: null → removed (same)

Review fields with actual data (e.g., review_status="accepted") are preserved.

Usage:
    python utils/cleanup_removed_fields.py --profile tiago
    python utils/cleanup_removed_fields.py --profile tiago --dry-run
    python utils/cleanup_removed_fields.py  # process all profiles
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from parselabs.config import ProfileConfig  # noqa: E402

logger = logging.getLogger(__name__)

# Fields to always remove from each lab_result entry
_FIELDS_TO_REMOVE = {
    "raw_is_abnormal",
    "is_abnormal",
    "raw_reference_notes",
    "reference_notes",
    "raw_source_text",
    "source_text",
    "review_confidence",
    "page_number",
    "source_file",
    "result_index",
}

# Fields to remove only when at their default (no-op) values
_DEFAULT_FIELDS_TO_REMOVE = {
    "review_needed": False,
    "review_reason": None,
    "review_status": None,
    "review_completed_at": None,
}


def _cleanup_json_file(json_path: Path, dry_run: bool) -> dict:
    """Remove deprecated fields from a single JSON file. Returns per-field removal counts."""

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lab_results = data.get("lab_results", [])
    if not lab_results:
        return {}

    removals = {}
    for result in lab_results:
        if not isinstance(result, dict):
            continue

        # Remove always-deleted fields
        for field in _FIELDS_TO_REMOVE:
            if field in result:
                removals[field] = removals.get(field, 0) + 1
                if not dry_run:
                    del result[field]

        # Remove default-valued review fields (preserve non-default values)
        for field, default_value in _DEFAULT_FIELDS_TO_REMOVE.items():
            if field in result and result[field] == default_value:
                removals[field] = removals.get(field, 0) + 1
                if not dry_run:
                    del result[field]

    if removals and not dry_run:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    return removals


def cleanup_profile(output_path: Path, dry_run: bool) -> dict:
    """Clean up all JSON files for a profile. Returns stats."""

    stats = {"files_cleaned": 0, "files_skipped": 0, "removals": {}}

    for json_path in sorted(output_path.rglob("*.json")):
        if json_path.name == "lab_specs.json":
            continue

        removals = _cleanup_json_file(json_path, dry_run)
        if removals:
            stats["files_cleaned"] += 1
            for field, count in removals.items():
                stats["removals"][field] = stats["removals"].get(field, 0) + count
            logger.info(f"  {'[DRY RUN] ' if dry_run else ''}Cleaned: {json_path.relative_to(output_path)} ({', '.join(f'{f}:{c}' for f, c in removals.items())})")
        else:
            stats["files_skipped"] += 1

    return stats


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Remove deprecated fields from extraction JSON files")
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
        profiles_dir = Path("profiles")
        if not profiles_dir.exists():
            logger.error("No profiles directory found")
            return 1
        profiles = []
        for p in sorted(profiles_dir.glob("*")):
            if p.suffix in (".yaml", ".yml", ".json") and p.stem != "_template":
                profiles.append(ProfileConfig.from_file(p))

    total_stats = {"files_cleaned": 0, "files_skipped": 0, "removals": {}}

    for profile in profiles:
        output_path = profile.output_path
        if not output_path.exists():
            logger.warning(f"Output path does not exist for profile '{profile.name}': {output_path}")
            continue

        logger.info(f"\nCleaning profile '{profile.name}' ({output_path})")
        stats = cleanup_profile(output_path, args.dry_run)

        total_stats["files_cleaned"] += stats["files_cleaned"]
        total_stats["files_skipped"] += stats["files_skipped"]
        for field, count in stats["removals"].items():
            total_stats["removals"][field] = total_stats["removals"].get(field, 0) + count

    prefix = "[DRY RUN] " if args.dry_run else ""
    logger.info(f"\n{prefix}Cleanup complete:")
    logger.info(f"  JSON files cleaned: {total_stats['files_cleaned']}")
    logger.info(f"  JSON files skipped (already clean): {total_stats['files_skipped']}")
    if total_stats["removals"]:
        logger.info("  Fields removed:")
        for field, count in sorted(total_stats["removals"].items()):
            logger.info(f"    {field}: {count} occurrences")

    return 0


if __name__ == "__main__":
    exit(main())
