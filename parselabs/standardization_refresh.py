"""Shared helpers for refreshing standardization caches from extracted rows."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import cast

import pandas as pd
from openai import APIError, OpenAI

from parselabs.config import UNKNOWN_VALUE, LabSpecsConfig
from parselabs.extraction import load_prompt_template
from parselabs.standardization import (
    build_name_cache_key,
    load_cache,
    normalize_name_cache_key_component,
    normalize_section_cache_key_component,
    normalize_unit_cache_key_component,
    save_cache,
)
from parselabs.types import StandardizationNameMatch, StandardizationUnitMatch
from parselabs.utils import parse_llm_json_response

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StandardizationRefreshResult:
    """Structured outcome for one cache-refresh pass."""

    uncached_names: tuple[tuple[str, str | None], ...]
    uncached_unit_pairs: tuple[tuple[str, str], ...]
    name_updates: int
    unit_updates: int
    unresolved_names: tuple[tuple[str, str | None], ...]
    unresolved_unit_pairs: tuple[tuple[str, str], ...]
    pruned_name_entries: int = 0
    pruned_unit_entries: int = 0
    name_error: str | None = None
    unit_error: str | None = None

    @property
    def attempted(self) -> bool:
        """Return whether the refresh found any uncached mappings."""

        return bool(self.uncached_names or self.uncached_unit_pairs)

    @property
    def rebuild_required(self) -> bool:
        """Return whether refreshed caches should be applied to rebuilt outputs."""

        return bool(self.name_updates or self.unit_updates)

    @property
    def changed(self) -> bool:
        """Return whether the refresh changed cache contents at all."""

        return self.rebuild_required or bool(self.pruned_name_entries or self.pruned_unit_entries)

def _render_prompt_template(template: str, **replacements: object) -> str:
    """Replace only known placeholder tokens in a prompt template."""

    rendered = template

    # Replace only explicit placeholder names so example JSON stays intact.
    for key, value in replacements.items():
        rendered = rendered.replace(f"{{{key}}}", str(value))

    return rendered


def _prune_unknown_cache_entries(cache: dict) -> tuple[dict, int]:
    """Remove cached $UNKNOWN$ entries so unresolved mappings stay discoverable."""

    pruned_cache = {}

    # Keep only resolved cache entries with valid unit keys.
    for key, value in cache.items():
        if value == UNKNOWN_VALUE:
            continue

        if "|" in str(key):
            _, _, lab_name = str(key).partition("|")
            if lab_name.strip() in {"", UNKNOWN_VALUE.lower()}:
                continue

        pruned_cache[key] = value

    return pruned_cache, len(cache) - len(pruned_cache)


def _standardize_names_with_llm(
    uncached_names: list[tuple[str, str | None]],
    standardized_names: list[str],
    client: OpenAI,
    model_id: str,
) -> dict[tuple[str, str | None], str]:
    """Call the LLM to standardize a batch of raw lab names."""

    items: list[StandardizationNameMatch] = [
        {
            "raw_lab_name": raw_name,
            "raw_section_name": raw_section_name,
        }
        for raw_name, raw_section_name in uncached_names
    ]
    system_prompt_template = load_prompt_template("name_standardization")
    system_prompt = _render_prompt_template(
        system_prompt_template,
        num_candidates=len(standardized_names),
        candidates=json.dumps(standardized_names, ensure_ascii=False, indent=2),
        unknown=UNKNOWN_VALUE,
    )
    user_prompt = f"""Map these contextual items to standardized values:

{json.dumps(items, ensure_ascii=False, indent=2)}

Return a JSON array of objects with raw_lab_name, raw_section_name, and standardized_name."""

    completion = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=4000,
    )

    response_text = (completion.choices[0].message.content or "").strip()
    result = parse_llm_json_response(response_text, fallback=[])

    validated: dict[tuple[str, str | None], str] = {}
    expected_by_key = {
        build_name_cache_key(item["raw_lab_name"], item.get("raw_section_name")): (
            item["raw_lab_name"],
            item.get("raw_section_name"),
        )
        for item in items
    }

    # Keep only known standardized names from the configured candidate set.
    if not isinstance(result, list):
        return validated

    for item in result:
        if not isinstance(item, dict):
            continue

        typed_item = cast(StandardizationNameMatch, item)
        raw_name = typed_item.get("raw_lab_name")
        raw_section_name = typed_item.get("raw_section_name")
        standardized_name = typed_item.get("standardized_name")

        # Guard: Rows without the identifying inputs cannot be matched back to the request.
        if raw_name is None:
            continue

        cache_key = build_name_cache_key(raw_name, raw_section_name)
        expected_context = expected_by_key.get(cache_key)

        # Guard: Ignore outputs for items that were not part of this prompt.
        if expected_context is None:
            continue

        if standardized_name in standardized_names:
            validated[expected_context] = standardized_name
            continue

        logger.warning(
            "LLM returned invalid standardization '%s' for raw name '%s' in section '%s'",
            standardized_name,
            raw_name,
            raw_section_name,
        )

    return validated


def _standardize_units_with_llm(
    uncached_pairs: list[tuple[str, str]],
    standardized_units: list[str],
    client: OpenAI,
    model_id: str,
    lab_specs: LabSpecsConfig,
) -> dict[str, str]:
    """Call the LLM to standardize a batch of raw lab units."""

    uncached_pairs = [
        (raw_unit, lab_name)
        for raw_unit, lab_name in uncached_pairs
        if lab_name and lab_name != UNKNOWN_VALUE
    ]

    # Guard: No resolved lab names means there is nothing to standardize.
    if not uncached_pairs:
        return {}

    items: list[StandardizationUnitMatch] = [{"raw_unit": raw_unit, "lab_name": lab_name} for raw_unit, lab_name in uncached_pairs]
    primary_units_map = {}

    # Build primary-unit context so missing raw units can still map correctly.
    for _, lab_name in uncached_pairs:
        primary_unit = lab_specs.get_primary_unit(lab_name)
        if primary_unit:
            primary_units_map[lab_name] = primary_unit

    primary_units_context = ""

    # Attach a compact mapping only when at least one unit context exists.
    if primary_units_map:
        primary_units_list = [f'  "{lab}": "{unit}"' for lab, unit in sorted(primary_units_map.items())]
        mapping_content = "\n".join(primary_units_list)
        primary_units_context = f"\nPRIMARY UNITS MAPPING (use this for null/missing units):\n{{\n{mapping_content}\n}}\n"

    system_prompt_template = load_prompt_template("unit_standardization")
    system_prompt = _render_prompt_template(
        system_prompt_template,
        num_candidates=len(standardized_units),
        candidates=json.dumps(standardized_units, ensure_ascii=False, indent=2),
        unknown=UNKNOWN_VALUE,
        primary_units_context=primary_units_context,
    )
    user_prompt = f"""Map these items to standardized values:

{json.dumps(items, ensure_ascii=False, indent=2)}

Return a JSON array with the standardized values."""

    completion = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=4000,
    )

    response_text = (completion.choices[0].message.content or "").strip()
    result_list = parse_llm_json_response(response_text, fallback=[])
    validated: dict[str, str] = {}

    # Keep only candidate units and normalize keys exactly as the runtime does.
    if isinstance(result_list, list):
        for item in result_list:
            if not isinstance(item, dict):
                continue

            typed_item = cast(StandardizationUnitMatch, item)
            raw_unit = typed_item.get("raw_unit")
            item_lab_name = typed_item.get("lab_name")
            standardized_unit = typed_item.get("standardized_unit")

            if raw_unit is None or item_lab_name is None:
                continue

            if standardized_unit not in standardized_units:
                continue

            normalized_unit = normalize_unit_cache_key_component(raw_unit)
            cache_key = f"{normalized_unit}|{str(item_lab_name).lower().strip()}"
            validated[cache_key] = standardized_unit

    return validated


def _resolve_effective_lab_name(
    raw_name: object,
    raw_section_name: object,
    current_lab_name: object,
    name_cache: dict[str, str],
) -> str | None:
    """Resolve the best standardized name available for a row."""

    current_name = str(current_lab_name or "").strip()

    # Prefer the already-standardized column when the row is resolved.
    if current_name and current_name != UNKNOWN_VALUE:
        return current_name

    cached_name = name_cache.get(build_name_cache_key(raw_name, raw_section_name))

    # Guard: Missing or unknown name mappings cannot seed unit standardization.
    if not cached_name or cached_name == UNKNOWN_VALUE:
        return None

    return cached_name


def _collect_uncached_names(df: pd.DataFrame, name_cache: dict[str, str]) -> list[tuple[str, str | None]]:
    """Collect unresolved raw lab names from the provided dataframe."""

    if "raw_lab_name" not in df.columns:
        return []

    raw_section_series = (
        df["raw_section_name"]
        if "raw_section_name" in df.columns
        else pd.Series([None] * len(df))
    )

    uncached_names: list[tuple[str, str | None]] = []
    seen_keys: set[str] = set()

    # Preserve first-seen spelling while deduplicating by normalized cache key.
    for raw_name, raw_section_name in zip(
        df["raw_lab_name"].fillna("").astype(str),
        raw_section_series.tolist(),
        strict=False,
    ):
        normalized_name = normalize_name_cache_key_component(raw_name)
        normalized_section = normalize_section_cache_key_component(raw_section_name)
        cache_key = build_name_cache_key(raw_name, raw_section_name)

        # Guard: Blank raw names cannot form a usable cache key.
        if not normalized_name or cache_key in seen_keys:
            continue

        # Guard: Cached contextual keys should not be refreshed again.
        if cache_key in name_cache:
            continue

        # Sectionless rows stay backward-compatible with legacy bare-name cache entries.
        if not normalized_section and normalized_name in name_cache:
            continue

        seen_keys.add(cache_key)
        uncached_names.append((raw_name, raw_section_name if normalized_section else None))

    return uncached_names


def _get_raw_unit_column(df: pd.DataFrame) -> str | None:
    """Return the raw-unit column available in the dataframe."""

    if "raw_unit" in df.columns:
        return "raw_unit"

    if "raw_lab_unit" in df.columns:
        return "raw_lab_unit"

    return None


def _collect_uncached_unit_pairs(
    df: pd.DataFrame,
    *,
    name_cache: dict[str, str],
    unit_cache: dict[str, str],
) -> list[tuple[str, str]]:
    """Collect unresolved raw-unit pairs using the latest working name cache."""

    raw_unit_col = _get_raw_unit_column(df)

    # Guard: Unit scanning needs both raw units and raw names.
    if raw_unit_col is None or "raw_lab_name" not in df.columns:
        return []

    uncached_pairs: list[tuple[str, str]] = []
    seen_keys: set[str] = set()
    current_lab_names = df["lab_name"] if "lab_name" in df.columns else pd.Series([""] * len(df))
    raw_section_series = (
        df["raw_section_name"]
        if "raw_section_name" in df.columns
        else pd.Series([None] * len(df))
    )

    # Resolve unit contexts row by row so freshly learned name mappings unlock unit scans.
    for raw_unit, raw_name, raw_section_name, current_lab_name in zip(
        df[raw_unit_col].fillna("").astype(str),
        df["raw_lab_name"].fillna("").astype(str),
        raw_section_series.tolist(),
        current_lab_names.fillna("").astype(str),
        strict=False,
    ):
        effective_lab_name = _resolve_effective_lab_name(raw_name, raw_section_name, current_lab_name, name_cache)
        if effective_lab_name is None:
            continue

        normalized_unit = normalize_unit_cache_key_component(raw_unit)
        cache_key = f"{normalized_unit}|{effective_lab_name.lower().strip()}"
        if cache_key in unit_cache or cache_key in seen_keys:
            continue

        seen_keys.add(cache_key)
        uncached_pairs.append((raw_unit, effective_lab_name))

    return uncached_pairs


def _build_client(
    *,
    base_url: str | None,
    api_key: str | None,
) -> OpenAI:
    """Create a client for on-demand standardization refreshes."""

    # Guard: Auto-refresh cannot call the model without API credentials.
    if not api_key:
        raise ValueError("Standardization refresh requires an OpenRouter API key.")

    return OpenAI(
        base_url=base_url or "https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def refresh_standardization_caches_from_dataframe(
    df: pd.DataFrame,
    lab_specs: LabSpecsConfig,
    *,
    model_id: str | None,
    base_url: str | None = None,
    api_key: str | None = None,
    client: OpenAI | None = None,
    dry_run: bool = False,
) -> StandardizationRefreshResult:
    """Refresh name and unit standardization caches from a merged review dataframe."""

    name_cache, removed_name_unknowns = _prune_unknown_cache_entries(load_cache("name_standardization"))
    unit_cache, removed_unit_unknowns = _prune_unknown_cache_entries(load_cache("unit_standardization"))
    working_name_cache = dict(name_cache)
    uncached_names = _collect_uncached_names(df, working_name_cache)
    name_updates = 0
    unit_updates = 0
    name_error = None
    unit_error = None

    # Persist pruned caches during real runs so stale $UNKNOWN$ entries do not linger.
    if not dry_run and removed_name_unknowns:
        save_cache("name_standardization", name_cache)

    # Persist pruned unit caches with the same eager cleanup behavior.
    if not dry_run and removed_unit_unknowns:
        save_cache("unit_standardization", unit_cache)

    # Guard: Scans can stop early in dry-run mode before any client setup.
    if dry_run:
        uncached_unit_pairs = _collect_uncached_unit_pairs(df, name_cache=working_name_cache, unit_cache=unit_cache)
        return StandardizationRefreshResult(
            uncached_names=tuple(uncached_names),
            uncached_unit_pairs=tuple(uncached_unit_pairs),
            name_updates=0,
            unit_updates=0,
            unresolved_names=tuple(uncached_names),
            unresolved_unit_pairs=tuple(uncached_unit_pairs),
            pruned_name_entries=removed_name_unknowns,
            pruned_unit_entries=removed_unit_unknowns,
        )

    # Build the API client lazily so no-op scans do not require runtime credentials.
    if uncached_names and client is None:
        try:
            client = _build_client(base_url=base_url, api_key=api_key)
        except ValueError as exc:
            name_error = str(exc)

    # Refresh uncached raw names first so unit discovery can use the newly mapped names.
    if uncached_names:
        if name_error:
            logger.warning(f"[standardization] Skipping name refresh: {name_error}")
        elif not model_id:
            name_error = "Standardization refresh requires an extraction model ID."
        else:
            try:
                assert client is not None
                refreshed_names = _standardize_names_with_llm(uncached_names, lab_specs.standardized_names, client, model_id)
            except (APIError, OSError, ValueError) as exc:  # pragma: no cover - exercised through pipeline warning path
                name_error = str(exc)
            else:
                for (raw_name, raw_section_name), standardized_name in refreshed_names.items():
                    cache_key = build_name_cache_key(raw_name, raw_section_name)
                    working_name_cache[cache_key] = standardized_name
                    name_cache[cache_key] = standardized_name
                name_updates = len(refreshed_names)

                # Save successful name updates immediately so partial refresh progress is not lost.
                if name_updates:
                    save_cache("name_standardization", name_cache)

    uncached_unit_pairs = _collect_uncached_unit_pairs(df, name_cache=working_name_cache, unit_cache=unit_cache)

    # Build the API client only when the unit step actually needs one.
    if uncached_unit_pairs and client is None:
        try:
            client = _build_client(base_url=base_url, api_key=api_key)
        except ValueError as exc:
            unit_error = str(exc)

    # Refresh unresolved unit contexts using the post-name-refresh working cache.
    if uncached_unit_pairs:
        if unit_error:
            logger.warning(f"[standardization] Skipping unit refresh: {unit_error}")
        elif not model_id:
            unit_error = "Standardization refresh requires an extraction model ID."
        else:
            try:
                assert client is not None
                refreshed_units = _standardize_units_with_llm(
                    uncached_unit_pairs,
                    lab_specs.standardized_units,
                    client,
                    model_id,
                    lab_specs,
                )
            except (APIError, OSError, ValueError) as exc:  # pragma: no cover - exercised through pipeline warning path
                unit_error = str(exc)
            else:
                for cache_key, standardized_unit in refreshed_units.items():
                    unit_cache[cache_key] = standardized_unit
                unit_updates = len(refreshed_units)

                # Save successful unit updates immediately so partial refresh progress is not lost.
                if unit_updates:
                    save_cache("unit_standardization", unit_cache)

    unresolved_names = _collect_uncached_names(df, working_name_cache)
    unresolved_unit_pairs = _collect_uncached_unit_pairs(df, name_cache=working_name_cache, unit_cache=unit_cache)
    return StandardizationRefreshResult(
        uncached_names=tuple(uncached_names),
        uncached_unit_pairs=tuple(uncached_unit_pairs),
        name_updates=name_updates,
        unit_updates=unit_updates,
        unresolved_names=tuple(unresolved_names),
        unresolved_unit_pairs=tuple(unresolved_unit_pairs),
        pruned_name_entries=removed_name_unknowns,
        pruned_unit_entries=removed_unit_unknowns,
        name_error=name_error,
        unit_error=unit_error,
    )
