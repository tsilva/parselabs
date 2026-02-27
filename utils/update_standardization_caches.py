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
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from labs_parser.utils import load_dotenv_with_env

load_dotenv_with_env()

import pandas as pd  # noqa: E402
from openai import OpenAI  # noqa: E402

from labs_parser.config import UNKNOWN_VALUE, LabSpecsConfig, ProfileConfig  # noqa: E402
from labs_parser.standardization import load_cache, save_cache  # noqa: E402
from labs_parser.utils import parse_llm_json_response  # noqa: E402

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")


def _standardize_names_with_llm(
    uncached_names: list[str],
    standardized_names: list[str],
    client: OpenAI,
    model_id: str,
) -> dict[str, str]:
    """Call LLM to standardize a batch of raw lab names."""

    items = {name: name for name in uncached_names}
    system_prompt_template = _load_prompt("name_standardization")
    system_prompt = system_prompt_template.format(
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
            if std == UNKNOWN_VALUE or std in standardized_names:
                validated[key] = std
            else:
                logger.warning(f"LLM returned invalid name: '{std}' for '{key}', using $UNKNOWN$")
                validated[key] = UNKNOWN_VALUE
        else:
            validated[key] = UNKNOWN_VALUE
    return validated


def _standardize_units_with_llm(
    uncached_pairs: list[tuple[str, str]],
    standardized_units: list[str],
    client: OpenAI,
    model_id: str,
    lab_specs: LabSpecsConfig,
) -> dict[str, str]:
    """Call LLM to standardize a batch of raw units. Returns dict keyed by cache key."""

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

    system_prompt_template = _load_prompt("unit_standardization").replace("{primary_units_context}", primary_units_context)
    system_prompt = system_prompt_template.format(
        num_candidates=len(standardized_units),
        candidates=json.dumps(standardized_units, ensure_ascii=False, indent=2),
        unknown=UNKNOWN_VALUE,
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
                cache_key = f"{str(raw_unit).lower().strip()}|{str(lab_name).lower().strip()}"
                if standardized and (standardized == UNKNOWN_VALUE or standardized in standardized_units):
                    validated[cache_key] = standardized
                else:
                    validated[cache_key] = UNKNOWN_VALUE
    return validated


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Update standardization caches using LLM")
    parser.add_argument("--profile", "-p", required=True, help="Profile name to scan for uncached values")
    parser.add_argument("--model", "-m", type=str, help="Model ID (default: EXTRACT_MODEL_ID from .env)")
    parser.add_argument("--dry-run", action="store_true", help="Show uncached values without calling LLM")
    args = parser.parse_args()

    # Load profile
    profile_path = ProfileConfig.find_path(args.profile)
    if not profile_path:
        logger.error(f"Profile '{args.profile}' not found")
        sys.exit(1)
    profile = ProfileConfig.from_file(profile_path)

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
    if "raw_unit" in df.columns and "lab_name" in df.columns:
        for _, row in df[["raw_unit", "lab_name"]].dropna().drop_duplicates().iterrows():
            key = f"{str(row['raw_unit']).lower().strip()}|{str(row['lab_name']).lower().strip()}"
            if key not in unit_cache:
                unit_pairs.append((str(row["raw_unit"]), str(row["lab_name"])))

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
    model_id = args.model or os.getenv("EXTRACT_MODEL_ID")
    if not model_id:
        logger.error("No model specified. Use --model or set EXTRACT_MODEL_ID.")
        sys.exit(1)

    client = OpenAI(
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    # Update name cache
    if uncached_names:
        logger.info(f"Standardizing {len(uncached_names)} names via LLM...")
        llm_results = _standardize_names_with_llm(uncached_names, lab_specs.standardized_names, client, model_id)
        for raw_name, std_name in llm_results.items():
            name_cache[raw_name.lower().strip()] = std_name
        save_cache("name_standardization", name_cache)
        logger.info(f"Name cache updated with {len(llm_results)} entries")

    # Update unit cache
    if unit_pairs:
        logger.info(f"Standardizing {len(unit_pairs)} unit pairs via LLM...")
        llm_results = _standardize_units_with_llm(unit_pairs, lab_specs.standardized_units, client, model_id, lab_specs)
        for cache_key, std_unit in llm_results.items():
            unit_cache[cache_key] = std_unit
        save_cache("unit_standardization", unit_cache)
        logger.info(f"Unit cache updated with {len(llm_results)} entries")

    logger.info("Cache update complete.")


if __name__ == "__main__":
    main()
