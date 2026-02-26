"""Lab name and unit standardization using persistent cache (no LLM at runtime)."""

import json
import logging
from pathlib import Path

from labs_parser.config import UNKNOWN_VALUE

logger = logging.getLogger(__name__)

# Cache directory for standardization results (user-editable JSON files)
CACHE_DIR = Path("config/cache")


def load_cache(name: str) -> dict:
    """Load JSON cache file. User-editable for overriding decisions."""

    path = CACHE_DIR / f"{name}.json"

    # Cache file exists — attempt to load it
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        # Cache file is corrupted or unreadable
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load cache {name}: {e}")

    # No cache file or load failed — return empty dict
    return {}


def save_cache(name: str, cache: dict):
    """Save cache to JSON, sorted alphabetically for easy editing."""

    # Ensure cache directory exists
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    path = CACHE_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False, sort_keys=True)


def standardize_lab_names(
    raw_test_names: list[str],
    standardized_names: list[str],
) -> dict[str, str]:
    """
    Map raw test names to standardized lab names using cache-only lookup.

    Cache miss returns $UNKNOWN$ and logs a warning. Use utils/update_standardization_caches.py
    to batch-process new raw names through LLM and update the cache.

    Args:
        raw_test_names: List of raw test names from extraction
        standardized_names: List of valid standardized lab names (unused, kept for API compat)

    Returns:
        Dictionary mapping raw_test_name -> standardized_name
    """

    # No input names — nothing to standardize
    if not raw_test_names:
        return {}

    # Load cache
    cache = load_cache("name_standardization")

    def cache_key(name):
        return name.lower().strip()

    # Get unique raw names and check for cache misses
    unique_raw_names = list(set(raw_test_names))
    uncached_names = [n for n in unique_raw_names if cache_key(n) not in cache]

    # Log warning for cache misses
    if uncached_names:
        logger.warning(f"[name_standardization] {len(uncached_names)} uncached names (returning $UNKNOWN$). Run utils/update_standardization_caches.py to update cache.")
        for name in uncached_names[:10]:  # Log first 10 to avoid flooding
            logger.warning(f"  Cache miss: '{name}'")
        if len(uncached_names) > 10:
            logger.warning(f"  ... and {len(uncached_names) - 10} more")

    # Return results for all names from cache ($UNKNOWN$ for misses)
    return {name: cache.get(cache_key(name), UNKNOWN_VALUE) for name in raw_test_names}


def standardize_lab_units(
    unit_contexts: list[tuple[str, str]],
    standardized_units: list[str],
    lab_specs_config=None,
) -> dict[tuple[str, str], str]:
    """
    Map raw lab units to standardized units using cache-only lookup.

    Cache miss returns $UNKNOWN$ and logs a warning. Use utils/update_standardization_caches.py
    to batch-process new raw units through LLM and update the cache.

    Args:
        unit_contexts: List of (raw_unit, standardized_lab_name) tuples for context
        standardized_units: List of valid standardized units (unused, kept for API compat)
        lab_specs_config: LabSpecsConfig instance (unused, kept for API compat)

    Returns:
        Dictionary mapping (raw_unit, lab_name) -> standardized_unit
    """

    # No input contexts — nothing to standardize
    if not unit_contexts:
        return {}

    # Load cache
    cache = load_cache("unit_standardization")

    def cache_key(raw_unit, lab_name):
        return f"{str(raw_unit).lower().strip()}|{str(lab_name).lower().strip()}"

    # Get unique (raw_unit, lab_name) pairs
    unique_pairs = list(set(unit_contexts))

    # Build results from cache
    cached_results = {}
    uncached_pairs = []
    for raw_unit, lab_name in unique_pairs:
        key = cache_key(raw_unit, lab_name)
        # Cache hit — use cached result
        if key in cache:
            cached_results[(raw_unit, lab_name)] = cache[key]
        # Cache miss — log warning and use $UNKNOWN$
        else:
            uncached_pairs.append((raw_unit, lab_name))
            cached_results[(raw_unit, lab_name)] = UNKNOWN_VALUE

    # Log warning for cache misses
    if uncached_pairs:
        logger.warning(f"[unit_standardization] {len(uncached_pairs)} uncached pairs (returning $UNKNOWN$). Run utils/update_standardization_caches.py to update cache.")
        for raw_unit, lab_name in uncached_pairs[:10]:  # Log first 10
            logger.warning(f"  Cache miss: ('{raw_unit}', '{lab_name}')")
        if len(uncached_pairs) > 10:
            logger.warning(f"  ... and {len(uncached_pairs) - 10} more")

    # Return results for all input pairs from cache
    return {pair: cached_results.get(pair, UNKNOWN_VALUE) for pair in unit_contexts}
