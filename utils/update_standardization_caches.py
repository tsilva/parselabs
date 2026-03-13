"""Batch-update standardization caches using LLM.

Reads uncached raw names/units from extracted CSVs and processes them through
an LLM to update the name and unit standardization caches.

Usage:
    python utils/update_standardization_caches.py --profile tsilva
    python utils/update_standardization_caches.py --profile tsilva --dry-run
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd  # noqa: E402
from openai import OpenAI  # noqa: E402

from parselabs.config import UNKNOWN_VALUE, LabSpecsConfig, ProfileConfig  # noqa: E402
from parselabs.extraction import load_prompt_template  # noqa: E402
from parselabs.standardization import load_cache, normalize_unit_cache_key_component, save_cache  # noqa: E402
from parselabs.utils import parse_llm_json_response  # noqa: E402

logger = logging.getLogger(__name__)


def _load_prompt(name: str) -> str:
    return load_prompt_template(name)


def _render_prompt_template(template: str, **replacements: str) -> str:
    """Replace only known placeholder tokens in a prompt template."""

    rendered = template

    # Replace explicit placeholders so literal braces in examples remain untouched.
    for key, value in replacements.items():
        rendered = rendered.replace(f"{{{key}}}", str(value))

    return rendered


def _prune_unknown_cache_entries(cache: dict) -> tuple[dict, int]:
    """Remove cached $UNKNOWN$ entries so unresolved mappings stay discoverable."""

    pruned_cache = {}
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
    uncached_names: list[str],
    standardized_names: list[str],
    client: OpenAI,
    model_id: str,
) -> dict[str, str]:
    """Call LLM to standardize a batch of raw lab names."""

    items = {name: name for name in uncached_names}
    system_prompt_template = _load_prompt("name_standardization")
    system_prompt = _render_prompt_template(
        system_prompt_template,
        num_candidates=len(standardized_names),
        candidates=json.dumps(standardized_names, ensure_ascii=False, indent=2),
        unknown=UNKNOWN_VALUE,
    )
    user_prompt = f"""Map these items to standardized values:

{json.dumps(items, ensure_ascii=False, indent=2)}

Return a JSON object with the standardized values."""

    completion = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=4000,
    )

    response_text = completion.choices[0].message.content.strip()
    result = parse_llm_json_response(response_text, fallback={})

    # Validate results against candidates
    validated = {}
    for key in items:
        if key in result:
            std = result[key]
            if std in standardized_names:
                validated[key] = std
            else:
                logger.warning(f"LLM returned invalid or unknown name: '{std}' for '{key}', leaving unresolved")
    return validated


def _standardize_units_with_llm(
    uncached_pairs: list[tuple[str, str]],
    standardized_units: list[str],
    client: OpenAI,
    model_id: str,
    lab_specs: LabSpecsConfig,
) -> dict[str, str]:
    """Call LLM to standardize a batch of raw units. Returns dict keyed by cache key."""

    uncached_pairs = [
        (raw_unit, lab_name)
        for raw_unit, lab_name in uncached_pairs
        if lab_name and lab_name != UNKNOWN_VALUE
    ]
    if not uncached_pairs:
        return {}

    items = [{"raw_unit": raw_unit, "lab_name": lab_name} for raw_unit, lab_name in uncached_pairs]

    # Build primary units context
    primary_units_map = {}
    for _, lab_name in uncached_pairs:
        if lab_name and lab_name != UNKNOWN_VALUE:
            primary_unit = lab_specs.get_primary_unit(lab_name)
            if primary_unit:
                primary_units_map[lab_name] = primary_unit

    primary_units_context = ""
    if primary_units_map:
        primary_units_list = [f'  "{lab}": "{unit}"' for lab, unit in sorted(primary_units_map.items())]
        mapping_content = "\n".join(primary_units_list)
        primary_units_context = f"\nPRIMARY UNITS MAPPING (use this for null/missing units):\n{{\n{mapping_content}\n}}\n"

    system_prompt_template = _load_prompt("unit_standardization")
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

    response_text = completion.choices[0].message.content.strip()
    result_list = parse_llm_json_response(response_text, fallback=[])

    validated = {}
    if isinstance(result_list, list):
        for item in result_list:
            raw_unit = item.get("raw_unit")
            lab_name = item.get("lab_name")
            standardized = item.get("standardized_unit")
            if raw_unit is not None and lab_name is not None:
                normalized_unit = normalize_unit_cache_key_component(raw_unit)
                cache_key = f"{normalized_unit}|{str(lab_name).lower().strip()}"
                if standardized in standardized_units:
                    validated[cache_key] = standardized
    return validated


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Update standardization caches using LLM")
    parser.add_argument("--profile", "-p", required=True, help="Profile name to scan for uncached values")
    parser.add_argument("--model", "-m", type=str, help="Model ID (overrides the profile value)")
    parser.add_argument("--dry-run", action="store_true", help="Show uncached values without calling LLM")
    args = parser.parse_args()

    # Load profile
    profile_path = ProfileConfig.find_path(args.profile)
    if not profile_path:
        logger.error(f"Profile '{args.profile}' not found")
        sys.exit(1)
    profile = ProfileConfig.from_file(profile_path)
    if not profile.output_path:
        logger.error(f"Profile '{args.profile}' has no output_path defined")
        sys.exit(1)

    # Load lab specs
    lab_specs = LabSpecsConfig()
    if not lab_specs.exists:
        logger.error("lab_specs.json not found")
        sys.exit(1)

    # Load CSV
    csv_path = profile.output_path / "all.csv"
    if not csv_path.exists():
        logger.error(f"No all.csv found at {csv_path}. Run extraction first.")
        sys.exit(1)

    df = pd.read_csv(csv_path, encoding="utf-8")
    logger.info(f"Loaded {len(df)} rows from {csv_path}")

    # Find uncached names
    name_cache = load_cache("name_standardization")
    raw_names = df["raw_lab_name"].dropna().unique().tolist()
    uncached_names = [n for n in raw_names if n.lower().strip() not in name_cache]

    # Find uncached units
    unit_cache = load_cache("unit_standardization")
    unit_pairs = []
    raw_unit_col = "raw_unit" if "raw_unit" in df.columns else "raw_lab_unit" if "raw_lab_unit" in df.columns else None
    if raw_unit_col and "lab_name" in df.columns:
        for _, row in df[[raw_unit_col, "lab_name"]].dropna().drop_duplicates().iterrows():
            if str(row["lab_name"]).strip() == UNKNOWN_VALUE:
                continue
            raw_unit = str(row[raw_unit_col])
            normalized_unit = normalize_unit_cache_key_component(raw_unit)
            key = f"{normalized_unit}|{str(row['lab_name']).lower().strip()}"
            if key not in unit_cache:
                unit_pairs.append((raw_unit, str(row["lab_name"])))

    logger.info(f"Uncached names: {len(uncached_names)}")
    logger.info(f"Uncached unit pairs: {len(unit_pairs)}")

    if args.dry_run:
        if uncached_names:
            logger.info("Uncached names:")
            for n in sorted(uncached_names):
                logger.info(f"  - {n}")
        if unit_pairs:
            logger.info("Uncached unit pairs:")
            for raw_unit, lab_name in sorted(unit_pairs):
                logger.info(f"  - ({raw_unit}, {lab_name})")
        return

    if not uncached_names and not unit_pairs:
        logger.info("All values are cached. Nothing to update.")
        return

    # Initialize LLM client
    model_id = args.model or profile.extract_model_id
    if not model_id:
        logger.error(f"Profile '{args.profile}' has no extract_model_id defined. Use --model to override it.")
        sys.exit(1)
    if not profile.openrouter_api_key:
        logger.error(f"Profile '{args.profile}' has no openrouter_api_key defined.")
        sys.exit(1)

    client = OpenAI(
        base_url=profile.openrouter_base_url or "https://openrouter.ai/api/v1",
        api_key=profile.openrouter_api_key,
    )

    # Update name cache
    name_cache, removed_name_unknowns = _prune_unknown_cache_entries(name_cache)
    if uncached_names:
        logger.info(f"Standardizing {len(uncached_names)} names via LLM...")
        llm_results = _standardize_names_with_llm(uncached_names, lab_specs.standardized_names, client, model_id)
        for raw_name, std_name in llm_results.items():
            name_cache[raw_name.lower().strip()] = std_name
        save_cache("name_standardization", name_cache)
        logger.info(f"Name cache updated with {len(llm_results)} entries")
    elif removed_name_unknowns:
        save_cache("name_standardization", name_cache)
        logger.info(f"Removed {removed_name_unknowns} stale unknown name cache entr{'y' if removed_name_unknowns == 1 else 'ies'}")

    # Update unit cache
    unit_cache, removed_unit_unknowns = _prune_unknown_cache_entries(unit_cache)
    if unit_pairs:
        logger.info(f"Standardizing {len(unit_pairs)} unit pairs via LLM...")
        llm_results = _standardize_units_with_llm(unit_pairs, lab_specs.standardized_units, client, model_id, lab_specs)
        for cache_key, std_unit in llm_results.items():
            unit_cache[cache_key] = std_unit
        save_cache("unit_standardization", unit_cache)
        logger.info(f"Unit cache updated with {len(llm_results)} entries")
    elif removed_unit_unknowns:
        save_cache("unit_standardization", unit_cache)
        logger.info(f"Removed {removed_unit_unknowns} stale unknown unit cache entr{'y' if removed_unit_unknowns == 1 else 'ies'}")

    logger.info("Cache update complete.")


if __name__ == "__main__":
    from parselabs.admin import run_legacy_utility

    raise SystemExit(run_legacy_utility("update-standardization-caches"))
