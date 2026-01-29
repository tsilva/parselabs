"""Lab name and unit standardization using LLM with persistent cache."""

import json
import logging
from pathlib import Path
from typing import Any
from openai import OpenAI

from config import UNKNOWN_VALUE
from utils import parse_llm_json_response

logger = logging.getLogger(__name__)

# Cache directory for LLM standardization results (user-editable JSON files)
CACHE_DIR = Path("config/cache")


def load_cache(name: str) -> dict:
    """Load JSON cache file. User-editable for overriding LLM decisions."""
    path = CACHE_DIR / f"{name}.json"
    if path.exists():
        try:
            with open(path, encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load cache {name}: {e}")
    return {}


def save_cache(name: str, cache: dict):
    """Save cache to JSON, sorted alphabetically for easy editing."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{name}.json"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2, ensure_ascii=False, sort_keys=True)


def standardize_with_llm(
    items: dict[Any, Any] | list[Any],
    candidates: list[str],
    system_prompt_template: str,
    model_id: str,
    client: OpenAI,
    temperature: float = 0.1
) -> dict[Any, str] | list[Any]:
    """
    Generic LLM-based standardization function.

    Args:
        items: Dictionary or list of items to standardize with their context
        candidates: List of valid standardized values
        system_prompt_template: System prompt with {candidates} and {unknown} placeholders
        model_id: Model to use for standardization
        client: OpenAI client instance
        temperature: Temperature for LLM sampling

    Returns:
        Dictionary or list (matching input type) mapping items to standardized values
    """
    if not items:
        return {} if isinstance(items, dict) else []

    is_list_input = isinstance(items, list)

    def fallback():
        return items if is_list_input else {key: UNKNOWN_VALUE for key in items.keys()}

    if not candidates:
        logger.warning("No candidates available, returning $UNKNOWN$ for all")
        return fallback()

    system_prompt = system_prompt_template.format(
        num_candidates=len(candidates),
        candidates=json.dumps(candidates, ensure_ascii=False, indent=2),
        unknown=UNKNOWN_VALUE
    )

    user_prompt = f"""Map these items to standardized values:

{json.dumps(items, ensure_ascii=False, indent=2)}

Return a JSON object/array with the standardized values."""

    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=4000
        )

        if not completion or not completion.choices:
            logger.error("Invalid completion response for standardization")
            return fallback()

        response_text = completion.choices[0].message.content.strip()
        result = parse_llm_json_response(response_text, fallback={})

        if not result:
            logger.error("Failed to parse standardization response")
            return fallback()

        # For list input, return as-is (already validated by the calling function)
        if is_list_input:
            return result

        # Validate results for dict input
        validated = {}
        for key in items.keys():
            if key in result:
                standardized = result[key]
                if standardized == UNKNOWN_VALUE or standardized in candidates:
                    validated[key] = standardized
                else:
                    logger.warning(f"LLM returned invalid value: '{standardized}' for '{key}', using $UNKNOWN$")
                    validated[key] = UNKNOWN_VALUE
            else:
                logger.warning(f"LLM didn't return mapping for '{key}', using $UNKNOWN$")
                validated[key] = UNKNOWN_VALUE

        return validated

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse standardization response as JSON: {e}")
        return fallback()
    except Exception as e:
        logger.error(f"Error during standardization: {e}")
        return fallback()


def standardize_lab_names(
    raw_test_names: list[str],
    model_id: str,
    standardized_names: list[str],
    client: OpenAI
) -> dict[str, str]:
    """
    Map raw test names to standardized lab names using LLM with cache.

    Results are cached in config/cache/name_standardization.json for:
    - Avoiding repeated LLM calls for known names
    - User override of LLM decisions by editing the JSON file

    Args:
        raw_test_names: List of raw test names from extraction
        model_id: Model to use for standardization
        standardized_names: List of valid standardized lab names
        client: OpenAI client instance

    Returns:
        Dictionary mapping raw_test_name -> standardized_name
    """
    if not raw_test_names:
        return {}

    # Load cache
    cache = load_cache("name_standardization")

    # Get unique raw names
    unique_raw_names = list(set(raw_test_names))

    # Split into cached and uncached
    def cache_key(name):
        return name.lower().strip()

    uncached_names = [n for n in unique_raw_names if cache_key(n) not in cache]

    # Call LLM only for uncached names
    if uncached_names:
        logger.info(f"[name_standardization] {len(uncached_names)} uncached names, calling LLM...")
        items = {name: name for name in uncached_names}

        system_prompt_template = """You are a medical lab test name standardization expert.

Your task: Map raw test names from lab reports to standardized lab names from a predefined list.

CRITICAL RULES:
1. Choose the BEST MATCH from the standardized names list
2. Consider semantic similarity and medical terminology
3. Account for language variations (Portuguese/English)
4. If NO good match exists, use exactly: "{unknown}"
5. Return a JSON object mapping each raw name to its standardized name

IMPORTANT - Portuguese lab report patterns:
Portuguese reports often have SECTION PREFIXES before the actual test name. Strip these prefixes when matching:
- "bioquímica - {{test}}" → match "{{test}}" to standardized list
- "bioquímica geral - {{test}}" → match "{{test}}"
- "hematologia - hemograma - {{test}}" → match "{{test}}"
- "hematologia - hemograma com contagem de plaquetas - {{test}}" → match "{{test}}"
- "química clínica - sangue - {{test}}" → match "{{test}}"
- "endocrinologia - {{test}}" → match "{{test}}"
- "hemograma - {{test}}" → match "{{test}}"
- "hemograma com fórmula - {{test}}" → match "{{test}}"
- "fórmula leucocitária - {{test}}" → match "{{test}}"
- "reticulócitos - {{test}}" → match "{{test}}"
- "velocidade de sedimentação - {{test}}" → match "{{test}}"
- "bilirrubina total e directa - {{test}}" → match "{{test}}"

The actual test name is usually the LAST part after the final " - " separator.

STANDARDIZED NAMES LIST ({num_candidates} names):
{candidates}

EXAMPLES:
- "Hemoglobina" → "Blood - Hemoglobin (Hgb)"
- "GLICOSE -jejum-" → "Blood - Glucose (Fasting)"
- "URINA - pH" → "Urine Type II - pH"
- "bioquímica - creatinina" → "Blood - Creatinine"
- "bioquímica - glicose" → "Blood - Glucose (Fasting)"
- "bioquímica - ureia" → "Blood - Urea"
- "hematologia - hemograma com contagem de plaquetas - hemoglobina" → "Blood - Hemoglobin (Hgb)"
- "hematologia - hemograma com contagem de plaquetas - leucócitos" → "Blood - Leukocytes"
- "hemograma com fórmula - eritrócitos" → "Blood - Erythrocytes"
- "hemograma com fórmula - hematócrito" → "Blood - Hematocrit (HCT) (%)"
- "reticulócitos - % reticulócitos" → "Blood - Reticulocyte Count (%)"
- "reticulócitos - conteúdo hemoglobina reticulócito" → "Blood - Reticulocyte Hemoglobin Content"
- "reticulócitos - nº total reticulócitos" → "Blood - Reticulocyte Count"
- "velocidade de sedimentação - 1ª hora" → "Blood - Erythrocyte Sedimentation Rate (ESR) - 1h"
- "bilirrubina total e directa - bilirrubina directa" → "Blood - Bilirubin Direct"
- "bilirrubina total e directa - bilirrubina total" → "Blood - Bilirubin Total"
- "não-hdl colesterol" → "Blood - Non-HDL Cholesterol"
- "plaquetócrito" → "Blood - Plateletcrit (PCT) (%)"
- "volume plaquetario médio" → "Blood - Mean Platelet Volume (MPV)"
- "indice distribuição plaquetas - pdw" → "Blood - Platelet Distribution Width (PDW)"
- "eritroblastos por 100 leucócitos" → "Blood - Nucleated Red Blood Cells (NRBC)"
- "Some Unknown Test" → "{unknown}"
"""

        llm_result = standardize_with_llm(
            items=items,
            candidates=standardized_names,
            system_prompt_template=system_prompt_template,
            model_id=model_id,
            client=client
        )

        # Update cache with LLM results
        for raw_name, std_name in llm_result.items():
            cache[cache_key(raw_name)] = std_name
        save_cache("name_standardization", cache)
        logger.info(f"[name_standardization] Cache updated with {len(llm_result)} entries")

    # Return results for all names from cache
    return {name: cache.get(cache_key(name), UNKNOWN_VALUE) for name in raw_test_names}


def standardize_lab_units(
    unit_contexts: list[tuple[str, str]],
    model_id: str,
    standardized_units: list[str],
    client: OpenAI,
    lab_specs_config=None
) -> dict[tuple[str, str], str]:
    """
    Map raw lab units to standardized units using LLM with cache.

    Results are cached in config/cache/unit_standardization.json for:
    - Avoiding repeated LLM calls for known unit/lab combinations
    - User override of LLM decisions by editing the JSON file

    Args:
        unit_contexts: List of (raw_unit, standardized_lab_name) tuples for context
        model_id: Model to use for standardization
        standardized_units: List of valid standardized units
        client: OpenAI client instance
        lab_specs_config: LabSpecsConfig instance for looking up primary units (optional)

    Returns:
        Dictionary mapping (raw_unit, lab_name) -> standardized_unit
    """
    if not unit_contexts:
        return {}

    # Load cache
    cache = load_cache("unit_standardization")

    def cache_key(raw_unit, lab_name):
        return f"{str(raw_unit).lower().strip()}|{str(lab_name).lower().strip()}"

    # Get unique (raw_unit, lab_name) pairs
    unique_pairs = list(set(unit_contexts))

    # Split into cached and uncached
    cached_results = {}
    uncached_pairs = []
    for raw_unit, lab_name in unique_pairs:
        key = cache_key(raw_unit, lab_name)
        if key in cache:
            cached_results[(raw_unit, lab_name)] = cache[key]
        else:
            uncached_pairs.append((raw_unit, lab_name))

    # If all cached, return early
    if not uncached_pairs:
        return {pair: cached_results.get(pair, UNKNOWN_VALUE) for pair in unit_contexts}

    logger.info(f"[unit_standardization] {len(uncached_pairs)} uncached pairs, calling LLM...")

    # Build primary units mapping for null unit inference (only for uncached)
    primary_units_map = {}
    if lab_specs_config and lab_specs_config.exists:
        for _, lab_name in uncached_pairs:
            if lab_name and lab_name != UNKNOWN_VALUE:
                primary_unit = lab_specs_config.get_primary_unit(lab_name)
                if primary_unit:
                    primary_units_map[lab_name] = primary_unit

    # Build items as list of dicts for context (only uncached)
    items = [
        {"raw_unit": raw_unit, "lab_name": lab_name}
        for raw_unit, lab_name in uncached_pairs
    ]

    # Build primary units context for prompt
    primary_units_context = ""
    if primary_units_map:
        primary_units_list = [f'  "{lab}": "{unit}"' for lab, unit in sorted(primary_units_map.items())]
        # Build the mapping and escape curly braces for .format()
        mapping_content = "\n".join(primary_units_list)
        primary_units_context = """
PRIMARY UNITS MAPPING (use this for null/missing units):
{{
""" + mapping_content + """
}}
"""

    system_prompt_template = """You are a medical laboratory unit standardization expert.

Your task: Map (raw_unit, lab_name) pairs to standardized units from a predefined list.

CRITICAL RULES:
1. Choose the BEST MATCH from the standardized units list
2. Handle case variations (e.g., "mg/dl" → "mg/dL", "u/l" → "IU/L")
3. Handle symbol variations (e.g., "µ" vs "μ", superscripts like ⁶ ⁹ ¹²)
4. Handle spacing variations (e.g., "mg / dl" → "mg/dL")
5. For null/missing units, look up the lab_name in the PRIMARY UNITS MAPPING (if provided)
6. If NO good match exists or lab not in mapping, use exactly: "{unknown}"
7. Return a JSON array with objects: {{"raw_unit": "...", "lab_name": "...", "standardized_unit": "..."}}

CRITICAL: DO NOT CONVERT UNITS - ONLY NORMALIZE FORMAT
The goal is to standardize unit NOTATION, NOT to convert between different units.
Unit conversions are handled separately by the system using conversion factors.

CORRECT FORMAT NORMALIZATION (same unit, different notation):
- "/mm3", "/mm³", "cells/mm³" → "/mm3" (keep as /mm3, do NOT convert to 10⁹/L)
- "x10E6/µl", "x10E6/ul", "x10^6/µL" → "x10E6/µL" (normalize symbols only)
- "x10E3/ul", "x10ˆ3/ul", "x10^3/µL" → "x10E3/µL" (normalize symbols only)
- "x10E9/L", "x10^9/L", "10^9/L", "10⁹/L", "109/L" → "10⁹/L" (these ARE the same unit)
- "x10E12/L", "x10^12/L", "10¹²/L" → "10¹²/L" (these ARE the same unit)

WRONG - DO NOT DO THIS:
- "/mm3" → "10⁹/L" (WRONG! This is a unit CONVERSION, not format normalization)
- "x10E3/µL" → "10⁹/L" (WRONG! These are different magnitude units)

Case normalization:
- "iu/l", "IU/l", "iu/L" → "IU/L"
- "fl", "FL" → "fL"
- "pg", "PG" → "pg"
- "mg/dl", "MG/DL" → "mg/dL"
- "g/dl", "G/DL" → "g/dL"

Special handling:
- "nan", "null", "None", empty string, "NaN" → look up from PRIMARY UNITS MAPPING
- "U/L" and "IU/L" are often interchangeable for enzyme activities (prefer IU/L if in list)
- "Leu/µL" for leukocytes → may need conversion context

STANDARDIZED UNITS LIST ({num_candidates} units):
{candidates}
""" + primary_units_context + """
EXAMPLES:
- {{"raw_unit": "mg/dl", "lab_name": "Blood - Glucose", "standardized_unit": "mg/dL"}}
- {{"raw_unit": "x10E6/µl", "lab_name": "Blood - Erythrocytes", "standardized_unit": "10¹²/L"}}
- {{"raw_unit": "x10^9/L", "lab_name": "Blood - Leukocytes", "standardized_unit": "10⁹/L"}}
- {{"raw_unit": "x10ˆ3/ul", "lab_name": "Blood - Platelets", "standardized_unit": "10⁹/L"}}
- {{"raw_unit": "U/L", "lab_name": "Blood - AST", "standardized_unit": "IU/L"}}
- {{"raw_unit": "fl", "lab_name": "Blood - MCV", "standardized_unit": "fL"}}
- {{"raw_unit": "null", "lab_name": "Blood - Albumin", "standardized_unit": "g/dL"}} (from PRIMARY UNITS MAPPING)
- {{"raw_unit": "nan", "lab_name": "Blood - Glucose", "standardized_unit": "mg/dL"}} (from PRIMARY UNITS MAPPING)
"""

    result_list = standardize_with_llm(
        items=items,
        candidates=standardized_units,
        system_prompt_template=system_prompt_template,
        model_id=model_id,
        client=client
    )

    # Convert list result to dict mapping (raw_unit, lab_name) -> standardized_unit
    # The LLM should return a list, so we need to handle that
    llm_results = {}
    if isinstance(result_list, list):
        for item in result_list:
            raw_unit = item.get("raw_unit")
            lab_name = item.get("lab_name")
            standardized = item.get("standardized_unit")

            if raw_unit is not None and lab_name is not None:
                if standardized and (standardized == UNKNOWN_VALUE or standardized in standardized_units):
                    llm_results[(raw_unit, lab_name)] = standardized
                else:
                    logger.warning(f"LLM returned invalid unit: '{standardized}' for ({raw_unit}, {lab_name})")
                    llm_results[(raw_unit, lab_name)] = UNKNOWN_VALUE
    else:
        logger.error("Expected list response from LLM for unit standardization")

    # Ensure all uncached pairs have a mapping
    for pair in uncached_pairs:
        if pair not in llm_results:
            logger.warning(f"LLM didn't return mapping for {pair}, using $UNKNOWN$")
            llm_results[pair] = UNKNOWN_VALUE

    # Update cache with LLM results
    for (raw_unit, lab_name), std_unit in llm_results.items():
        cache[cache_key(raw_unit, lab_name)] = std_unit
    save_cache("unit_standardization", cache)
    logger.info(f"[unit_standardization] Cache updated with {len(llm_results)} entries")

    # Merge cached and new results
    cached_results.update(llm_results)

    # Return results for all input pairs from merged cache
    return {pair: cached_results.get(pair, UNKNOWN_VALUE) for pair in unit_contexts}


def standardize_qualitative_values(
    raw_values: list[str],
    model_id: str,
    client: OpenAI
) -> dict[str, int | None]:
    """
    Map qualitative lab result text to boolean values (0/1/None) using LLM with cache.

    Results are cached in config/cache/qualitative_to_boolean.json for:
    - Avoiding repeated LLM calls for known qualitative values
    - User override of LLM decisions by editing the JSON file

    Args:
        raw_values: List of raw qualitative values from extraction
        model_id: Model to use for classification
        client: OpenAI client instance

    Returns:
        Dictionary mapping raw_value -> 0 (negative), 1 (positive), or None (not qualitative)
    """
    if not raw_values:
        return {}

    # Load cache
    cache = load_cache("qualitative_to_boolean")

    def cache_key(value):
        return str(value).lower().strip()

    # Get unique values, filter out empty/nan
    unique_values = list(set(
        cache_key(v) for v in raw_values
        if v is not None and str(v).strip() and str(v).lower() not in ('nan', 'none', '')
    ))

    # Split into cached and uncached
    uncached_values = [v for v in unique_values if v not in cache]

    # If all cached, return early
    if not uncached_values:
        return {v: cache.get(cache_key(v)) for v in raw_values if v is not None}

    logger.info(f"[qualitative_to_boolean] {len(uncached_values)} uncached values, calling LLM...")

    # Build items for LLM
    items = {v: v for v in uncached_values}

    system_prompt_template = """You are a medical lab result classifier.

Your task: Classify qualitative lab result text as boolean values.

CLASSIFICATION RULES:
- 0 (NEGATIVE): negativo, ausente, não detectado, normal, não reativo, negative, absent, not detected, non-reactive, nenhum, none, nil, clear, incolor, amarelo claro, amarelo, límpido, within normal limits
- 1 (POSITIVE): positivo, presente, detectado, anormal, reativo, positive, present, detected, reactive, turvo, abnormal, elevated, increased
- null: For values that are NOT qualitative (numbers, ranges, units, empty, or unclear)

IMPORTANT:
- Return a JSON object mapping each input value to 0, 1, or null
- Be case-insensitive
- Handle Portuguese and English terms
- When in doubt, return null

Return format: {{"value1": 0, "value2": 1, "value3": null, ...}}
"""

    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt_template},
                {"role": "user", "content": f"Classify these values:\n\n{json.dumps(items, ensure_ascii=False, indent=2)}"}
            ],
            temperature=0.1,
            max_tokens=2000
        )

        if not completion or not completion.choices:
            logger.error("Invalid completion response for qualitative classification")
            return {v: None for v in raw_values if v is not None}

        response_text = completion.choices[0].message.content.strip()
        result = parse_llm_json_response(response_text, fallback={})

        if not result:
            logger.error("Failed to parse qualitative classification response")
            return {v: None for v in raw_values if v is not None}

        # Validate and update cache
        for value in uncached_values:
            if value in result:
                classified = result[value]
                if classified in (0, 1, None):
                    cache[value] = classified
                else:
                    logger.warning(f"LLM returned invalid classification: '{classified}' for '{value}', using null")
                    cache[value] = None
            else:
                logger.warning(f"LLM didn't return classification for '{value}', using null")
                cache[value] = None

        save_cache("qualitative_to_boolean", cache)
        logger.info(f"[qualitative_to_boolean] Cache updated with {len(uncached_values)} entries")

    except Exception as e:
        logger.error(f"Error during qualitative classification: {e}")
        return {v: None for v in raw_values if v is not None}

    # Return results for all input values from cache
    return {v: cache.get(cache_key(v)) for v in raw_values if v is not None}
