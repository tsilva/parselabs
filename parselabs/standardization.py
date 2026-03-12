"""Lab name and unit standardization using persistent cache (no LLM at runtime)."""

import json
import logging

from parselabs.config import UNKNOWN_VALUE
from parselabs.paths import get_cache_dir

logger = logging.getLogger(__name__)

# Cache directory for standardization results (user-editable JSON files)
CACHE_DIR = get_cache_dir()


def normalize_unit_cache_key_component(raw_unit) -> str:
    """Normalize missing and textual raw-unit values to a stable cache key token."""

    normalized = str(raw_unit).lower().strip()

    # Collapse all blank-like unit tokens onto one cache key so runtime and updater agree.
    if normalized in {"", "nan", "none", "null"}:
        return "null"

    return normalized


def _drop_unknown_standardization_entries(name: str, cache: dict) -> dict:
    """Treat cached $UNKNOWN$ values as unresolved for standardization caches."""

    # Only the name/unit standardization caches should enforce this invariant.
    if name not in {"name_standardization", "unit_standardization"}:
        return cache

    pruned_cache = {key: value for key, value in cache.items() if value != UNKNOWN_VALUE}

    # Unit mappings without a resolved standardized lab name are not valid cache entries.
    if name == "unit_standardization":
        invalid_keys = []
        for key in pruned_cache:
            _, _, lab_name = key.partition("|")
            if lab_name.strip() in {"", UNKNOWN_VALUE.lower()}:
                invalid_keys.append(key)
        for key in invalid_keys:
            pruned_cache.pop(key, None)

    return pruned_cache


def load_cache(name: str) -> dict:
    """Load JSON cache file. User-editable for overriding decisions."""

    path = CACHE_DIR / f"{name}.json"

    # Cache file exists — attempt to load it
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                cache = json.load(f)
                return _drop_unknown_standardization_entries(name, cache)
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
) -> dict[str, str]:
    """
    Map raw test names to standardized lab names using cache-only lookup.

    Cache miss returns $UNKNOWN$ and logs a warning. Use utils/update_standardization_caches.py
    to batch-process new raw names through LLM and update the cache.

    Args:
        raw_test_names: List of raw test names from extraction
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
) -> dict[tuple[str, str], str]:
    """
    Map raw lab units to standardized units using cache-only lookup.

    Cache miss returns $UNKNOWN$ and logs a warning. Use utils/update_standardization_caches.py
    to batch-process new raw units through LLM and update the cache.

    Args:
        unit_contexts: List of (raw_unit, standardized_lab_name) tuples for context
    Returns:
        Dictionary mapping (raw_unit, lab_name) -> standardized_unit
    """

    # No input contexts — nothing to standardize
    if not unit_contexts:
        return {}

    # Load cache
    cache = load_cache("unit_standardization")

    def cache_key(raw_unit, lab_name):
        normalized_unit = normalize_unit_cache_key_component(raw_unit)
        return f"{normalized_unit}|{str(lab_name).lower().strip()}"

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
