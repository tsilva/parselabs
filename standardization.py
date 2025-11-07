"""Lab name and unit standardization using LLM."""

import json
import logging
from typing import Any
from openai import OpenAI

from config import UNKNOWN_VALUE
from utils import parse_llm_json_response

logger = logging.getLogger(__name__)


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

    # Determine if items is a list or dict
    is_list_input = isinstance(items, list)

    if not candidates:
        logger.warning("No candidates available, returning $UNKNOWN$ for all")
        if is_list_input:
            return items  # Return as-is for list
        else:
            return {key: UNKNOWN_VALUE for key in items.keys()}

    # Build system prompt with candidates
    system_prompt = system_prompt_template.format(
        num_candidates=len(candidates),
        candidates=json.dumps(candidates, ensure_ascii=False, indent=2),
        unknown=UNKNOWN_VALUE
    )

    # Build user prompt with items to standardize
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

        if not completion or not completion.choices or len(completion.choices) == 0:
            logger.error("Invalid completion response for standardization")
            return {key: UNKNOWN_VALUE for key in items.keys()}

        response_text = completion.choices[0].message.content.strip()
        result = parse_llm_json_response(response_text, fallback={})

        if not result:
            logger.error("Failed to parse standardization response")
            if is_list_input:
                return items
            else:
                return {key: UNKNOWN_VALUE for key in items.keys()}

        # For list input, return as-is (it's already validated by the calling function)
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
        if is_list_input:
            return items
        else:
            return {key: UNKNOWN_VALUE for key in items.keys()}
    except Exception as e:
        logger.error(f"Error during standardization: {e}")
        if is_list_input:
            return items
        else:
            return {key: UNKNOWN_VALUE for key in items.keys()}


def standardize_lab_names(
    raw_test_names: list[str],
    model_id: str,
    standardized_names: list[str],
    client: OpenAI
) -> dict[str, str]:
    """
    Map raw test names to standardized lab names using LLM.

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

    # Get unique raw names to avoid duplicate API calls
    unique_raw_names = list(set(raw_test_names))

    # Build items dict (simple mapping for names)
    items = {name: name for name in unique_raw_names}

    system_prompt_template = """You are a medical lab test name standardization expert.

Your task: Map raw test names from lab reports to standardized lab names from a predefined list.

CRITICAL RULES:
1. Choose the BEST MATCH from the standardized names list
2. Consider semantic similarity and medical terminology
3. Account for language variations (Portuguese/English)
4. If NO good match exists, use exactly: "{unknown}"
5. Return a JSON object mapping each raw name to its standardized name

STANDARDIZED NAMES LIST ({num_candidates} names):
{candidates}

EXAMPLES:
- "Hemoglobina" → "Blood - Hemoglobin"
- "GLICOSE -jejum-" → "Blood - Glucose"
- "URINA - pH" → "Urine Type II - pH"
- "Some Unknown Test" → "{unknown}"
"""

    return standardize_with_llm(
        items=items,
        candidates=standardized_names,
        system_prompt_template=system_prompt_template,
        model_id=model_id,
        client=client
    )


def standardize_lab_units(
    unit_contexts: list[tuple[str, str]],
    model_id: str,
    standardized_units: list[str],
    client: OpenAI,
    lab_specs_config=None
) -> dict[tuple[str, str], str]:
    """
    Map raw lab units to standardized units using LLM with lab name context.

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

    # Get unique (raw_unit, lab_name) pairs
    unique_pairs = list(set(unit_contexts))

    # Build primary units mapping for null unit inference
    primary_units_map = {}
    if lab_specs_config and lab_specs_config.exists:
        for _, lab_name in unique_pairs:
            if lab_name and lab_name != UNKNOWN_VALUE:
                primary_unit = lab_specs_config.get_primary_unit(lab_name)
                if primary_unit:
                    primary_units_map[lab_name] = primary_unit

    # Build items as list of dicts for context
    items = [
        {"raw_unit": raw_unit, "lab_name": lab_name}
        for raw_unit, lab_name in unique_pairs
    ]

    # Build primary units context for prompt
    primary_units_context = ""
    if primary_units_map:
        primary_units_list = [f'  "{lab}": "{unit}"' for lab, unit in sorted(primary_units_map.items())]
        primary_units_context = f"""
PRIMARY UNITS MAPPING (use this for null/missing units):
{{
{chr(10).join(primary_units_list)}
}}
"""

    system_prompt_template = f"""You are a medical laboratory unit standardization expert.

Your task: Map (raw_unit, lab_name) pairs to standardized units from a predefined list.

CRITICAL RULES:
1. Choose the BEST MATCH from the standardized units list
2. Handle case variations (e.g., "mg/dl" → "mg/dL", "u/l" → "IU/L")
3. Handle symbol variations (e.g., "µ" vs "μ", superscripts)
4. Handle spacing variations (e.g., "mg / dl" → "mg/dL")
5. For null/missing units, look up the lab_name in the PRIMARY UNITS MAPPING (if provided)
6. If NO good match exists or lab not in mapping, use exactly: "{{unknown}}"
7. Return a JSON array with objects: {{"raw_unit": "...", "lab_name": "...", "standardized_unit": "..."}}

STANDARDIZED UNITS LIST ({{num_candidates}} units):
{{candidates}}
{primary_units_context}
EXAMPLES:
- {{"raw_unit": "mg/dl", "lab_name": "Blood - Glucose", "standardized_unit": "mg/dL"}}
- {{"raw_unit": "U/L", "lab_name": "Blood - AST", "standardized_unit": "IU/L"}}
- {{"raw_unit": "fl", "lab_name": "Blood - MCV", "standardized_unit": "fL"}}
- {{"raw_unit": "null", "lab_name": "Blood - Albumin", "standardized_unit": "g/dL"}} (from PRIMARY UNITS MAPPING)
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
    if isinstance(result_list, list):
        result_dict = {}
        for item in result_list:
            raw_unit = item.get("raw_unit")
            lab_name = item.get("lab_name")
            standardized = item.get("standardized_unit")

            if raw_unit is not None and lab_name is not None:
                if standardized and (standardized == UNKNOWN_VALUE or standardized in standardized_units):
                    result_dict[(raw_unit, lab_name)] = standardized
                else:
                    logger.warning(f"LLM returned invalid unit: '{standardized}' for ({raw_unit}, {lab_name})")
                    result_dict[(raw_unit, lab_name)] = UNKNOWN_VALUE

        # Ensure all unique pairs have a mapping
        for pair in unique_pairs:
            if pair not in result_dict:
                logger.warning(f"LLM didn't return mapping for {pair}, using $UNKNOWN$")
                result_dict[pair] = UNKNOWN_VALUE

        return result_dict
    else:
        logger.error("Expected list response from LLM for unit standardization")
        return {pair: UNKNOWN_VALUE for pair in unique_pairs}
