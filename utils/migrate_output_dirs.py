#!/usr/bin/env python3
"""Migrate legacy output directories to include file hash suffix.

Output directories are named {pdf_stem}_{hash} (SHA-256 first 8 hex chars).
This was introduced after some extractions already existed with just {pdf_stem}/.
This script batch-renames all legacy directories that lack the hash suffix.

Usage:
    python utils/migrate_output_dirs.py --profile tsilva
    python utils/migrate_output_dirs.py --profile tsilva --dry-run
    python utils/migrate_output_dirs.py  # process all profiles
"""

import argparse
import hashlib
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from labs_parser.config import ProfileConfig  # noqa: E402

logger = logging.getLogger(__name__)

# Matches directories that already have a _{8-hex-char} suffix
_HASH_SUFFIX_RE = re.compile(r"_[0-9a-f]{8}$")


def _compute_file_hash(file_path: Path, hash_length: int = 8) -> str:
    """Compute SHA-256 hash of a file, returning first `hash_length` hex chars.

    # Duplicated from main.py to avoid importing it (triggers side effects).
    """

    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:hash_length]


def _find_matching_pdf(stem: str, input_path: Path, input_file_regex: str | None) -> Path | None:
    """Find a single PDF in input_path matching the given stem.

    Returns the PDF path, or None if zero or multiple matches found.
    """

    # Use input_file_regex if set, otherwise default to *.pdf
    pattern = input_file_regex or "*.pdf"
    candidates = [p for p in input_path.glob(pattern) if p.stem == stem]

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        logger.warning(f"  Multiple PDFs match stem '{stem}': {[c.name for c in candidates]} — skipping")

    return None


def migrate_profile(profile: ProfileConfig, dry_run: bool) -> dict:
    """Migrate legacy output directories for a profile. Returns stats."""

    stats = {"renamed": 0, "already_migrated": 0, "pdf_not_found": 0, "target_exists": 0}

    output_path = profile.output_path
    input_path = profile.input_path

    # Guard: need both paths
    if not output_path or not input_path:
        logger.warning(f"  Profile '{profile.name}' missing input_path or output_path — skipping")
        return stats

    if not output_path.exists():
        return stats

    if not input_path.exists():
        logger.warning(f"  Input path does not exist: {input_path}")
        return stats

    # Scan subdirectories of output_path
    for subdir in sorted(output_path.iterdir()):
        if not subdir.is_dir():
            continue

        # Skip directories that already have a hash suffix
        if _HASH_SUFFIX_RE.search(subdir.name):
            stats["already_migrated"] += 1
            continue

        stem = subdir.name

        # Find matching source PDF
        pdf_path = _find_matching_pdf(stem, input_path, profile.input_file_regex)
        if not pdf_path:
            stats["pdf_not_found"] += 1
            logger.warning(f"  No source PDF found for '{stem}' in {input_path}")
            continue

        # Compute hash and build target name
        file_hash = _compute_file_hash(pdf_path)
        target_name = f"{stem}_{file_hash}"
        target_dir = output_path / target_name

        # Guard: target already exists
        if target_dir.exists():
            stats["target_exists"] += 1
            logger.warning(f"  Target already exists: {target_name} — skipping")
            continue

        # Rename
        prefix = "[DRY RUN] " if dry_run else ""
        logger.info(f"  {prefix}{stem}/ -> {target_name}/")

        if not dry_run:
            subdir.rename(target_dir)

        stats["renamed"] += 1

    return stats


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Migrate legacy output directories to include file hash")
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

    total_stats = {"renamed": 0, "already_migrated": 0, "pdf_not_found": 0, "target_exists": 0}

    for profile in profiles:
        output_path = profile.output_path
        if not output_path or not output_path.exists():
            logger.warning(f"Output path does not exist for profile '{profile.name}': {output_path}")
            continue

        logger.info(f"\nMigrating profile '{profile.name}' ({output_path})")
        stats = migrate_profile(profile, args.dry_run)

        for key in total_stats:
            total_stats[key] += stats[key]

    # Summary
    prefix = "[DRY RUN] " if args.dry_run else ""
    logger.info(f"\n{prefix}Migration complete:")
    logger.info(f"  Directories renamed: {total_stats['renamed']}")
    logger.info(f"  Already migrated (skipped): {total_stats['already_migrated']}")
    logger.info(f"  Source PDF not found (skipped): {total_stats['pdf_not_found']}")
    logger.info(f"  Target already exists (skipped): {total_stats['target_exists']}")

    return 0


if __name__ == "__main__":
    exit(main())
