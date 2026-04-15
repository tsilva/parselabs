"""Batch-update standardization caches using the shared refresh helpers.

Usage:
    python utils/update_standardization_caches.py --profile tsilva
    python utils/update_standardization_caches.py --profile tsilva --dry-run
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from parselabs.config import LabSpecsConfig, ProfileConfig
from parselabs.standardization_refresh import (
    refresh_standardization_caches_from_dataframe,
)

logger = logging.getLogger(__name__)


def _load_profile_dataframe(profile_name: str) -> tuple[ProfileConfig, pd.DataFrame]:
    """Load the merged review dataframe for the requested profile."""

    profile_path = ProfileConfig.find_path(profile_name)

    # Guard: The requested profile must exist before refresh can start.
    if not profile_path:
        raise ValueError(f"Profile '{profile_name}' not found")

    profile = ProfileConfig.from_file(profile_path)

    # Guard: The updater only works against an existing merged output root.
    if not profile.output_path:
        raise ValueError(f"Profile '{profile_name}' has no output_path defined")

    csv_path = profile.output_path / "all.csv"

    # Guard: There must already be a merged review CSV to scan.
    if not csv_path.exists():
        raise ValueError(f"No all.csv found at {csv_path}. Run extraction first.")

    dataframe = pd.read_csv(csv_path, encoding="utf-8")
    logger.info(f"Loaded {len(dataframe)} rows from {csv_path}")
    return profile, dataframe


def _log_refresh_result(result) -> None:
    """Log a concise refresh summary for CLI usage."""

    logger.info(f"Uncached names: {len(result.uncached_names)}")
    logger.info(f"Uncached unit pairs: {len(result.uncached_unit_pairs)}")

    # Guard: Dry runs and no-op scans stop after reporting the unresolved work.
    if not result.changed and not result.attempted:
        logger.info("All values are cached. Nothing to update.")
        return

    if result.pruned_name_entries:
        logger.info(
            f"Removed {result.pruned_name_entries} stale unknown name cache entr{'y' if result.pruned_name_entries == 1 else 'ies'}"
        )

    if result.pruned_unit_entries:
        logger.info(
            f"Removed {result.pruned_unit_entries} stale unknown unit cache entr{'y' if result.pruned_unit_entries == 1 else 'ies'}"
        )

    if result.name_updates:
        logger.info(f"Name cache updated with {result.name_updates} entries")

    if result.unit_updates:
        logger.info(f"Unit cache updated with {result.unit_updates} entries")

    if result.name_error:
        logger.warning(f"Name refresh failed: {result.name_error}")

    if result.unit_error:
        logger.warning(f"Unit refresh failed: {result.unit_error}")

    if result.unresolved_names:
        logger.warning(f"Unresolved names remaining: {len(result.unresolved_names)}")
        for raw_name, raw_section_name in result.unresolved_names:
            if raw_section_name:
                logger.warning(f"  - ({raw_name}, {raw_section_name})")
                continue

            logger.warning(f"  - {raw_name}")

    if result.unresolved_unit_pairs:
        logger.warning(f"Unresolved unit pairs remaining: {len(result.unresolved_unit_pairs)}")
        for raw_unit, lab_name in result.unresolved_unit_pairs:
            logger.warning(f"  - ({raw_unit}, {lab_name})")


def main() -> int:
    """Run the legacy updater against one profile's merged review CSV."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Update standardization caches using LLM")
    parser.add_argument("--profile", "-p", required=True, help="Profile name to scan for uncached values")
    parser.add_argument("--model", "-m", type=str, help="Model ID (overrides the profile value)")
    parser.add_argument("--dry-run", action="store_true", help="Show uncached values without calling LLM")
    args = parser.parse_args()

    try:
        profile, dataframe = _load_profile_dataframe(args.profile)
    except ValueError as exc:
        logger.error(str(exc))
        return 1

    lab_specs = LabSpecsConfig()

    # Guard: Standardization refresh requires the lab-spec candidate lists.
    if not lab_specs.exists:
        logger.error("lab_specs.json not found")
        return 1

    result = refresh_standardization_caches_from_dataframe(
        dataframe,
        lab_specs,
        model_id=args.model or profile.extract_model_id,
        base_url=profile.openrouter_base_url,
        api_key=profile.openrouter_api_key,
        dry_run=args.dry_run,
    )
    _log_refresh_result(result)
    return 0


if __name__ == "__main__":
    from parselabs.admin_commands import run_legacy_utility

    raise SystemExit(run_legacy_utility("update-standardization-caches"))
