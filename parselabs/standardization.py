"""Lab name and unit standardization using persistent cache (no LLM at runtime)."""

import json
import logging
import re
import unicodedata

from parselabs.config import UNKNOWN_VALUE
from parselabs.paths import get_cache_dir

logger = logging.getLogger(__name__)

# Cache directory for standardization results (user-editable JSON files)
CACHE_DIR = get_cache_dir()


def normalize_name_cache_key_component(raw_name) -> str:
    """Normalize raw lab-name text into a stable cache-key token."""

    return str(raw_name).lower().strip()


def _collapse_spaced_letter_tokens(text: str) -> str:
    """Collapse headers extracted as spaced single letters into one token."""

    tokens = text.split()

    # Guard: Normal multi-word labels should keep their original token boundaries.
    if len(tokens) < 3 or not all(len(token) == 1 and token.isalpha() for token in tokens):
        return text

    return "".join(tokens)


def _normalize_lookup_cache_key_component(value: str) -> str:
    """Fold accents and stylized spacing so lookups survive OCR/header quirks."""

    text = str(value).lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = _collapse_spaced_letter_tokens(text)
    text = re.sub(r"\s+", " ", text)
    return text


def _normalize_lookup_cache_key(key: str) -> str:
    """Normalize cache keys component-wise so contextual separators remain meaningful."""

    if "|" in key:
        raw_name, raw_section_name = key.split("|", 1)
        normalized_name = _normalize_lookup_cache_key_component(raw_name)
        normalized_section = _normalize_lookup_cache_key_component(raw_section_name)
        return f"{normalized_name}|{normalized_section}"

    if " - " in key:
        raw_section_name, raw_name = key.split(" - ", 1)
        normalized_section = _normalize_lookup_cache_key_component(raw_section_name)
        normalized_name = _normalize_lookup_cache_key_component(raw_name)
        return f"{normalized_section} - {normalized_name}"

    return _normalize_lookup_cache_key_component(key)


def normalize_section_cache_key_component(raw_section_name) -> str:
    """Normalize missing and textual section labels to a stable cache-key token."""

    normalized = str(raw_section_name).lower().strip()

    # Collapse all blank-like section values so sectionless rows share one fallback path.
    if normalized in {"", "nan", "none", "null"}:
        return ""

    return normalized


def build_name_cache_key(raw_name, raw_section_name=None) -> str:
    """Build the canonical cache key for a raw lab name and optional section label."""

    normalized_name = normalize_name_cache_key_component(raw_name)
    normalized_section = normalize_section_cache_key_component(raw_section_name)

    # Sectionless rows stay backward-compatible with the legacy bare-name key format.
    if not normalized_section:
        return normalized_name

    return f"{normalized_name}|{normalized_section}"


def build_legacy_section_name_cache_key(raw_name, raw_section_name) -> str:
    """Build the legacy ``section - raw_name`` cache key still present in older caches."""

    normalized_name = normalize_name_cache_key_component(raw_name)
    normalized_section = normalize_section_cache_key_component(raw_section_name)
    if not normalized_section:
        return normalized_name
    return f"{normalized_section} - {normalized_name}"


def normalize_unit_cache_key_component(raw_unit) -> str:
    """Normalize missing and textual raw-unit values to a stable cache key token."""

    normalized = str(raw_unit).lower().strip()

    # Collapse all blank-like unit tokens onto one cache key so runtime and updater agree.
    if normalized in {"", "nan", "none", "null"}:
        return "null"

    # Normalize compact scientific-notation variants like x109/L and x1012/L so
    # they share cache entries with the more explicit x10^9/L / x10^12/L forms.
    compact_exponent_match = re.fullmatch(r"(?P<prefix>x)(?P<body>10)(?P<exp>\d{1,2})/l", normalized.replace(" ", ""))
    if compact_exponent_match:
        prefix = "x" if compact_exponent_match.group("prefix") else ""
        exponent = compact_exponent_match.group("exp")
        return f"{prefix}10^{exponent}/l"

    return normalized


def _contextual_cache_values_for_name(cache: dict[str, str], normalized_name: str) -> set[str]:
    """Collect all contextual cache values that mention a given normalized raw name."""

    contextual_values: set[str] = set()
    canonical_prefix = f"{normalized_name}|"
    legacy_suffix = f" - {normalized_name}"

    for key, value in cache.items():
        if key.startswith(canonical_prefix) or key.endswith(legacy_suffix):
            contextual_values.add(value)

    return contextual_values


def _build_folded_cache_index(cache: dict[str, str]) -> dict[str, str]:
    """Build a normalized cache index for lookup-only compatibility matches."""

    folded_cache: dict[str, str] = {}
    ambiguous_keys: set[str] = set()

    for key, value in cache.items():
        folded_key = _normalize_lookup_cache_key(key)

        # Guard: Once two folded keys disagree, treat that normalized form as unsafe.
        if folded_key in ambiguous_keys:
            continue

        existing_value = folded_cache.get(folded_key)

        # Keep a unique folded match only while every source key agrees on the value.
        if existing_value is None or existing_value == value:
            folded_cache[folded_key] = value
            continue

        ambiguous_keys.add(folded_key)
        folded_cache.pop(folded_key, None)

    return folded_cache


def _lookup_name_cache_value(
    cache: dict[str, str],
    folded_cache: dict[str, str],
    *candidate_keys: str,
) -> str | None:
    """Resolve a cache entry from direct keys first, then normalized compatibility keys."""

    for candidate_key in candidate_keys:
        # Prefer exact keys so newer cache entries win without normalization heuristics.
        if candidate_key in cache:
            return cache[candidate_key]

    for candidate_key in candidate_keys:
        folded_key = _normalize_lookup_cache_key(candidate_key)

        # Guard: Skip folded keys that are absent or were marked ambiguous.
        if folded_key not in folded_cache:
            continue

        return folded_cache[folded_key]

    return None


def _safe_bare_name_fallback(
    cache: dict[str, str],
    folded_cache: dict[str, str],
    raw_name,
    raw_section_name,
) -> str | None:
    """Return a bare-name cache match only when contextual mappings do not conflict."""

    normalized_name = normalize_name_cache_key_component(raw_name)
    bare_value = _lookup_name_cache_value(cache, folded_cache, normalized_name)

    # Guard: no bare cache entry means there is nothing to reuse.
    if bare_value is None:
        return None

    contextual_values = _contextual_cache_values_for_name(cache, normalized_name)

    # No contextual overrides exist, or every contextual mapping agrees with the bare mapping.
    if not contextual_values or contextual_values == {bare_value}:
        return bare_value

    logger.warning(
        "[name_standardization] Refusing bare-name fallback for ('%s', '%s') due to conflicting contextual cache values: %s",
        raw_name,
        raw_section_name,
        sorted(contextual_values),
    )
    return None


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
    raw_test_names: list[tuple[str, str | None]],
) -> dict[tuple[str, str | None], str]:
    """
    Map raw test names plus optional section labels to standardized lab names using cache-only lookup.

    Cache miss returns $UNKNOWN$ and logs a warning.

    Args:
        raw_test_names: List of (raw_test_name, raw_section_name) tuples from extraction
    Returns:
        Dictionary mapping (raw_test_name, raw_section_name) -> standardized_name
    """

    # No input names — nothing to standardize
    if not raw_test_names:
        return {}

    # Load cache
    cache = load_cache("name_standardization")
    folded_cache = _build_folded_cache_index(cache)

    normalized_contexts: list[tuple[str, str | None]] = []
    for raw_name, raw_section_name in raw_test_names:
        normalized_contexts.append((str(raw_name), raw_section_name if raw_section_name is not None else None))

    # Get unique name contexts and check for cache misses.
    unique_name_contexts = list(set(normalized_contexts))
    uncached_names: list[tuple[str, str | None]] = []
    cached_results: dict[tuple[str, str | None], str] = {}

    for raw_name, raw_section_name in unique_name_contexts:
        cache_key = build_name_cache_key(raw_name, raw_section_name)
        legacy_cache_key = build_legacy_section_name_cache_key(raw_name, raw_section_name)

        cached_value = _lookup_name_cache_value(
            cache,
            folded_cache,
            cache_key,
            legacy_cache_key,
        )

        # Resolve exact or normalized contextual cache keys before any bare-name fallback.
        if cached_value is not None:
            cached_results[(raw_name, raw_section_name)] = cached_value
            continue

        fallback_value = _safe_bare_name_fallback(cache, folded_cache, raw_name, raw_section_name)
        if fallback_value is not None:
            cached_results[(raw_name, raw_section_name)] = fallback_value
            continue

        uncached_names.append((raw_name, raw_section_name))
        cached_results[(raw_name, raw_section_name)] = UNKNOWN_VALUE

    # Log neutral warnings so callers can decide whether to auto-refresh later.
    if uncached_names:
        logger.warning(f"[name_standardization] {len(uncached_names)} uncached names (using $UNKNOWN$ for this pass).")
        for raw_name, raw_section_name in uncached_names[:10]:  # Log first 10 to avoid flooding
            if normalize_section_cache_key_component(raw_section_name):
                logger.warning(f"  Cache miss: ('{raw_name}', '{raw_section_name}')")
                continue

            logger.warning(f"  Cache miss: '{raw_name}'")
        if len(uncached_names) > 10:
            logger.warning(f"  ... and {len(uncached_names) - 10} more")

    # Return results for all names from cache ($UNKNOWN$ for misses).
    return {context: cached_results.get(context, UNKNOWN_VALUE) for context in normalized_contexts}


def standardize_lab_units(
    unit_contexts: list[tuple[str, str]],
) -> dict[tuple[str, str], str]:
    """
    Map raw lab units to standardized units using cache-only lookup.

    Cache miss returns $UNKNOWN$ and logs a warning.

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

    # Log neutral warnings so callers can decide whether to auto-refresh later.
    if uncached_pairs:
        logger.warning(f"[unit_standardization] {len(uncached_pairs)} uncached pairs (using $UNKNOWN$ for this pass).")
        for raw_unit, lab_name in uncached_pairs[:10]:  # Log first 10
            logger.warning(f"  Cache miss: ('{raw_unit}', '{lab_name}')")
        if len(uncached_pairs) > 10:
            logger.warning(f"  ... and {len(uncached_pairs) - 10} more")

    # Return results for all input pairs from cache
    return {pair: cached_results.get(pair, UNKNOWN_VALUE) for pair in unit_contexts}
