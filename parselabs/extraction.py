"""Lab result extraction from images using vision models."""

import ast
import base64
import json
import logging
import re
from pathlib import Path
from typing import Annotated

from openai import APIError, OpenAI
from pydantic import BaseModel, Field, field_validator

from parselabs.paths import get_prompts_dir

logger = logging.getLogger(__name__)

BBOX_FIELDS = [
    "bbox_left",
    "bbox_top",
    "bbox_right",
    "bbox_bottom",
]

RAW_LAB_NAME_FIELD = Annotated[
    str,
    Field(
        description="Test name ONLY as written in the PDF. Must contain ONLY the test name - DO NOT include values, units, reference ranges, or field labels. WRONG: 'Glucose, raw_value: 100' CORRECT: 'Glucose'",
    ),
]
RAW_SECTION_NAME_FIELD = Annotated[
    str | None,
    Field(
        default=None,
        description="Nearest visible section or header name governing this test row, copied EXACTLY as shown. Use the most specific visible section header for the row. If no governing section is visible, use null.",
    ),
]
RAW_VALUE_FIELD = Annotated[
    str | None,
    Field(
        default=None,
        description="Result value ONLY. Must contain ONLY the numeric or text result - DO NOT include test names, units, or field labels. Examples: '5.2', '14.8', 'NEGATIVO', 'POSITIVO'",
    ),
]
RAW_LAB_UNIT_FIELD = Annotated[
    str | None,
    Field(
        default=None,
        description="Unit ONLY as written in PDF. Must contain ONLY the unit symbol - DO NOT include values or test names. Examples: 'mg/dL', '%', 'U/L'",
    ),
]
RAW_REFERENCE_RANGE_FIELD = Annotated[
    str | None,
    Field(
        default=None,
        description="Complete reference range text EXACTLY as shown.",
    ),
]
BBOX_LEFT_FIELD = Annotated[
    float | None,
    Field(
        default=None,
        description="Optional normalized left edge of the result bounding box on the page. Use the 0-1000 scale relative to full page width.",
    ),
]
BBOX_TOP_FIELD = Annotated[
    float | None,
    Field(
        default=None,
        description="Optional normalized top edge of the result bounding box on the page. Use the 0-1000 scale relative to full page height.",
    ),
]
BBOX_RIGHT_FIELD = Annotated[
    float | None,
    Field(
        default=None,
        description="Optional normalized right edge of the result bounding box on the page. Use the 0-1000 scale relative to full page width.",
    ),
]
BBOX_BOTTOM_FIELD = Annotated[
    float | None,
    Field(
        default=None,
        description="Optional normalized bottom edge of the result bounding box on the page. Use the 0-1000 scale relative to full page height.",
    ),
]
COLLECTION_DATE_FIELD = Annotated[
    str | None,
    Field(
        default=None,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Specimen collection date in YYYY-MM-DD format",
    ),
]
REPORT_DATE_FIELD = Annotated[
    str | None,
    Field(
        default=None,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Report issue date in YYYY-MM-DD format",
    ),
]
EXTRACTION_COLLECTION_DATE_FIELD = Annotated[
    str | None,
    Field(
        default=None,
        description="Specimen collection date in YYYY-MM-DD format",
    ),
]
EXTRACTION_REPORT_DATE_FIELD = Annotated[
    str | None,
    Field(
        default=None,
        description="Report issue date in YYYY-MM-DD format",
    ),
]
LAB_FACILITY_FIELD = Annotated[
    str | None,
    Field(default=None, description="Name of laboratory that performed tests"),
]


# ========================================
# Pydantic Models
# ========================================


class LabResult(BaseModel):
    """Single lab test result - optimized for extraction accuracy."""

    # Raw extraction (exactly as shown in PDF)
    raw_lab_name: RAW_LAB_NAME_FIELD
    raw_section_name: RAW_SECTION_NAME_FIELD
    raw_value: RAW_VALUE_FIELD
    raw_lab_unit: RAW_LAB_UNIT_FIELD
    raw_reference_range: RAW_REFERENCE_RANGE_FIELD
    raw_reference_min: float | None = Field(
        default=None,
        description="Minimum reference value as a PLAIN NUMBER ONLY. Parse from reference_range. Examples: '< 40' → null, '150 - 400' → 150, '26.5-32.6' → 26.5",
    )
    raw_reference_max: float | None = Field(
        default=None,
        description="Maximum reference value as a PLAIN NUMBER ONLY. Parse from reference_range. Examples: '< 40' → 40, '150 - 400' → 400, '26.5-32.6' → 32.6",
    )
    raw_comments: str | None = Field(
        default=None,
        description="Additional notes or remarks about the test (NOT the test result itself). Only use for extra information like methodology notes or special conditions.",
    )
    bbox_left: BBOX_LEFT_FIELD
    bbox_top: BBOX_TOP_FIELD
    bbox_right: BBOX_RIGHT_FIELD
    bbox_bottom: BBOX_BOTTOM_FIELD
    # Internal fields (added by pipeline, not by LLM) — excluded from JSON serialization
    lab_name_standardized: str | None = Field(default=None, exclude=True)
    lab_unit_standardized: str | None = Field(default=None, exclude=True)

    @field_validator("raw_value", mode="before")
    @classmethod
    def coerce_raw_value_to_string(cls, v):
        """Coerce numeric values to strings - LLMs often return floats instead of strings."""

        # Guard: None passthrough
        if v is None:
            return v

        # Coerce numeric types to string representation
        if isinstance(v, (int, float)):
            # Format without unnecessary decimal places for integers
            return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)

        # Already a string or other type - return as-is
        return v

    @field_validator("raw_lab_name", mode="before")
    @classmethod
    def validate_raw_lab_name(cls, v):
        """Reject malformed lab names with embedded metadata.

        Previously this cleaned malformed entries. Now it fails fast
        to force investigation when the model misbehaves.
        """

        # Guard: None passthrough
        if v is None:
            return v

        v_str = str(v)

        # Check for embedded metadata patterns (both old and new field name formats)
        malformed_patterns = [
            "section_name:",
            "raw_section_name:",
            "value_raw:",
            "raw_value:",
            "lab_unit_raw:",
            "raw_lab_unit:",
            "reference_range:",
            "raw_reference_range:",
            "reference_min_raw:",
            "raw_reference_min:",
            "reference_max_raw:",
            "raw_reference_max:",
        ]
        v_lower = v_str.lower()

        # Reject lab names that contain embedded field data from other columns
        if any(pattern in v_lower for pattern in malformed_patterns):
            raise ValueError(f"MALFORMED OUTPUT: raw_lab_name contains embedded field data. Model returned: '{v_str[:100]}...'. This indicates the extraction prompt needs improvement.")

        return v_str


class HealthLabReport(BaseModel):
    """Document-level lab report metadata."""

    collection_date: EXTRACTION_COLLECTION_DATE_FIELD
    report_date: EXTRACTION_REPORT_DATE_FIELD
    lab_facility: LAB_FACILITY_FIELD
    page_has_lab_data: bool | None = Field(
        default=None,
        description="True if page contains lab test results, False if page is cover/instructions/administrative with no lab data",
    )
    source_file: str | None = Field(default=None, description="Source PDF filename")
    lab_results: list[LabResult] = Field(
        default_factory=list,
        description="List of all lab test results extracted from this page/document",
    )

    @staticmethod
    def _clear_empty_strings(model: BaseModel):
        """Set empty string optional fields to None on a Pydantic model."""

        # Read field metadata from the model class to avoid Pydantic instance deprecation warnings.
        model_fields = type(model).model_fields

        # Inspect each declared field so optional empty strings can be normalized consistently.
        for field_name in model_fields:
            value = getattr(model, field_name)
            field_info = model_fields[field_name]

            # Replace empty strings with None for optional fields
            if value == "" and not field_info.is_required():
                setattr(model, field_name, None)

    def normalize_empty_optionals(self):
        """Convert empty strings to None for optional fields."""

        self._clear_empty_strings(self)
        for lab_result in self.lab_results:
            self._clear_empty_strings(lab_result)


# ========================================
# LLM-Facing Schema (subset of full model)
# ========================================


class LabResultExtraction(BaseModel):
    """LLM-facing schema — only fields the model should populate.

    Excludes pipeline-internal fields (review_*, page_number, source_file,
    result_index, lab_name_standardized, lab_unit_standardized) to reduce
    token usage and avoid confusing the model.
    """

    raw_lab_name: RAW_LAB_NAME_FIELD
    raw_section_name: RAW_SECTION_NAME_FIELD
    raw_value: RAW_VALUE_FIELD
    raw_lab_unit: RAW_LAB_UNIT_FIELD
    raw_reference_range: RAW_REFERENCE_RANGE_FIELD
    raw_reference_min: float | None = Field(
        default=None,
        description="Minimum reference value as a PLAIN NUMBER ONLY. Parse from reference_range.",
    )
    raw_reference_max: float | None = Field(
        default=None,
        description="Maximum reference value as a PLAIN NUMBER ONLY. Parse from reference_range.",
    )
    raw_comments: str | None = Field(
        default=None,
        description="Additional notes or remarks about the test (NOT the test result itself).",
    )
    bbox_left: BBOX_LEFT_FIELD
    bbox_top: BBOX_TOP_FIELD
    bbox_right: BBOX_RIGHT_FIELD
    bbox_bottom: BBOX_BOTTOM_FIELD


class HealthLabReportExtraction(BaseModel):
    """LLM-facing schema for function calling.

    Excludes pipeline-internal fields from HealthLabReport. Uses LabResultExtraction
    to reduce schema size (~41% smaller) and avoid confusing the model.
    """

    collection_date: COLLECTION_DATE_FIELD
    report_date: REPORT_DATE_FIELD
    lab_facility: LAB_FACILITY_FIELD
    page_has_lab_data: bool | None = Field(
        default=None,
        description="True if page contains lab test results, False if page is cover/instructions/administrative",
    )
    lab_results: list[LabResultExtraction] = Field(
        default_factory=list,
        description="List of all lab test results extracted from this page/document",
    )


# Tool definition for function calling — uses LLM-facing schema to reduce token usage
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "extract_lab_results",
            "description": "Extracts lab results from medical report image",
            "parameters": HealthLabReportExtraction.model_json_schema(),
        },
    }
]


# ========================================
# Extraction Prompts
# ========================================

_PROMPTS_DIR = get_prompts_dir()


def load_prompt_template(name: str) -> str:
    """Load one prompt template from the shared prompts directory."""

    return (_PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8").strip()


EXTRACTION_SYSTEM_PROMPT = load_prompt_template("extraction_system")
EXTRACTION_USER_PROMPT = load_prompt_template("extraction_user")


def _empty_report() -> dict:
    """Return a fresh empty lab report dict."""

    return HealthLabReport(lab_results=[]).model_dump(mode="json")


def _failed_report(reason: str) -> dict:
    """Return an empty report annotated with extraction failure metadata."""

    result = _empty_report()
    result["_extraction_failed"] = True
    result["_failure_reason"] = reason
    return result


# ========================================
# Extraction Function
# ========================================


def extract_labs_from_page_image(
    image_path: Path,
    model_id: str,
    client: OpenAI,
    temperature: float = 0.0,
    max_retries: int = 1,
    temperature_step: float = 0.2,
) -> dict:
    """
    Extract lab results from a page image using vision model.

    Uses temperature escalation retry strategy for malformed outputs.
    When validation fails due to malformed LLM output (e.g., all fields
    concatenated into a single string), retries with higher temperature
    to get the model to produce properly structured output.

    Args:
        image_path: Path to the preprocessed page image
        model_id: Vision model to use for extraction
        client: OpenAI client instance
        temperature: Initial temperature for sampling (default: 0.0)
        max_retries: Maximum retry attempts with escalating temperature (default: 1)
        temperature_step: Temperature increment per retry (default: 0.2)

    Returns:
        Dictionary with extracted report data (validated by Pydantic).
        On failure, includes _extraction_failed=True and _failure_reason.
    """

    # Load image data for API call
    with open(image_path, "rb") as img_file:
        img_data = base64.standard_b64encode(img_file.read()).decode("utf-8")

    # Attempt extraction with retry logic
    result = _attempt_extraction_with_retries(
        image_path=image_path,
        img_data=img_data,
        model_id=model_id,
        client=client,
        effective_system_prompt=EXTRACTION_SYSTEM_PROMPT,
        temperature=temperature,
        max_retries=max_retries,
        temperature_step=temperature_step,
    )

    return result


def _attempt_extraction_with_retries(
    image_path: Path,
    img_data: str,
    model_id: str,
    client: OpenAI,
    effective_system_prompt: str,
    temperature: float,
    max_retries: int,
    temperature_step: float,
) -> dict:
    """Execute extraction with temperature escalation retry logic."""

    current_temp = temperature
    last_error = None

    for attempt in range(max_retries + 1):  # +1 for initial attempt
        # Log retry attempt with escalated temperature
        if attempt > 0:
            current_temp = temperature + (attempt * temperature_step)
            logger.info(f"[{image_path.name}] Retry {attempt}/{max_retries} with temperature={current_temp:.1f} (previous error: malformed output)")

        # Attempt single extraction
        tool_result_dict = _execute_single_extraction(
            image_path=image_path,
            img_data=img_data,
            model_id=model_id,
            client=client,
            effective_system_prompt=effective_system_prompt,
            current_temp=current_temp,
        )

        # Guard: Extraction failed at API level
        if tool_result_dict is None:
            return _failed_report("Invalid extraction response from model")

        # Pre-process: Fix common LLM issues before Pydantic validation
        tool_result_dict = _fix_lab_results_format(tool_result_dict)

        # Validate and process the extraction result
        validation_result = _validate_extraction_result(
            tool_result_dict=tool_result_dict,
            image_path=image_path,
            attempt=attempt,
            current_temp=current_temp,
        )

        # Guard: Validation succeeded, return the result
        if validation_result["success"]:
            return validation_result["data"]

        # Malformed output - retry with higher temperature
        if validation_result["should_retry"]:
            last_error = validation_result["error_msg"]
            num_results = len(tool_result_dict.get("lab_results", []))
            logger.warning(f"[{image_path.name}] Malformed output detected ({num_results} results), attempt {attempt + 1}/{max_retries + 1}")
            continue

        # Non-retryable validation error - try to salvage what we can
        return validation_result["data"]

    # All retries exhausted - return failure marker
    logger.error(f"[{image_path.name}] Extraction failed after {max_retries + 1} attempts. Last error: {last_error[:200] if last_error else 'unknown'}")
    result = _failed_report(f"Malformed output after {max_retries + 1} attempts")
    result["_retry_count"] = max_retries + 1
    return result


def _execute_single_extraction(
    image_path: Path,
    img_data: str,
    model_id: str,
    client: OpenAI,
    effective_system_prompt: str,
    current_temp: float,
) -> dict | None:
    """Execute a single extraction attempt via API.

    Returns:
        Dict with tool result data, or None if extraction failed
    """

    # Call the vision model API
    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": effective_system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": EXTRACTION_USER_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_data}"},
                        },
                    ],
                },
            ],
            temperature=current_temp,
            max_tokens=16384,
            tools=TOOLS,
            tool_choice={
                "type": "function",
                "function": {"name": "extract_lab_results"},
            },
        )
    # API-level failure - propagate as RuntimeError
    except APIError as e:
        logger.error(f"API Error during lab extraction from {image_path.name}: {e}")
        raise RuntimeError(f"Lab extraction failed for {image_path.name}: {e}")

    # Guard: Invalid response structure from API
    if not completion or not completion.choices:
        logger.error("Invalid completion response structure")
        return None

    # Guard: Model did not use tool call
    if not completion.choices[0].message.tool_calls:
        logger.warning(f"No tool call by model for lab extraction from {image_path.name}")
        return None

    # Extract and parse tool arguments
    tool_args_raw = completion.choices[0].message.tool_calls[0].function.arguments
    try:
        return json.loads(tool_args_raw)
    # Malformed JSON from API response
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error for tool args: {e}")
        return None


def _validate_extraction_result(
    tool_result_dict: dict,
    image_path: Path,
    attempt: int,
    current_temp: float,
) -> dict:
    """Validate extraction result with Pydantic and check quality.

    Returns:
        Dict with keys: success (bool), data (dict), should_retry (bool), error_msg (str)
    """

    # Validate with Pydantic
    try:
        report_model = HealthLabReport(**tool_result_dict)
        report_model.normalize_empty_optionals()
        _assert_complete_result_bboxes(report_model)

        # Check extraction quality and log warnings
        _log_extraction_quality(report_model, image_path)

        # Success - return the validated result
        result = report_model.model_dump(mode="json")

        # Log if this was a retry success
        if attempt > 0:
            logger.info(f"[{image_path.name}] Extraction succeeded on retry {attempt} with temp={current_temp:.1f}")
        return {"success": True, "data": result, "should_retry": False, "error_msg": None}

    # Validation error - check if retryable (malformed) or not
    except ValueError as e:
        error_msg = str(e)

        # Malformed output - signal for retry with higher temperature
        if "MALFORMED OUTPUT" in error_msg:
            return {"success": False, "data": None, "should_retry": True, "error_msg": error_msg}

        # Non-retryable validation error - try to salvage
        else:
            num_results = len(tool_result_dict.get("lab_results", []))
            logger.error(f"Model validation error for report with {num_results} lab_results: {e}")
            return {"success": False, "data": _salvage_lab_results(tool_result_dict), "should_retry": False, "error_msg": error_msg}

    # Unexpected error - salvage what we can without retrying
    except Exception as e:
        num_results = len(tool_result_dict.get("lab_results", []))
        logger.error(f"Model validation error for report with {num_results} lab_results: {e}")
        return {"success": False, "data": _salvage_lab_results(tool_result_dict), "should_retry": False, "error_msg": str(e)}


def _assert_complete_result_bboxes(report_model: HealthLabReport) -> None:
    """Reject extraction payloads that omit any bbox coordinate for extracted rows."""

    # Pages without extracted rows are validated by the existing no-results path.
    if not report_model.lab_results:
        return

    missing_bbox_count = 0

    # Count rows that do not carry all four coordinates so the retry path stays explicit.
    for result in report_model.lab_results:
        if any(getattr(result, field_name) is None for field_name in BBOX_FIELDS):
            missing_bbox_count += 1

    # Guard: Fully boxed payloads can continue through normal quality checks.
    if missing_bbox_count == 0:
        return

    raise ValueError(
        "MALFORMED OUTPUT: extracted lab results missing complete bounding boxes. "
        f"Rows without full bbox data: {missing_bbox_count}/{len(report_model.lab_results)}."
    )


def _log_extraction_quality(report_model: HealthLabReport, image_path: Path) -> None:
    """Log warnings if extraction quality is poor."""

    # Guard: No lab results extracted
    if not report_model.lab_results:
        # Page explicitly marked as non-lab content
        if report_model.page_has_lab_data is False:
            logger.debug(f"Page confirmed to have no lab data:\n\t- {image_path}")
        # Possible extraction failure - zero results when lab data was expected
        else:
            logger.warning(f"Extraction returned 0 lab results. This may indicate a model extraction failure - image should be manually reviewed.\n\t- {image_path}")
        return

    # Calculate percentage of null values
    null_count = sum(1 for r in report_model.lab_results if r.raw_value is None)
    total_count = len(report_model.lab_results)
    null_pct = (null_count / total_count * 100) if total_count > 0 else 0

    # Guard: Warn if too many null values
    if null_pct > 50:
        logger.warning(f"Extraction quality issue: {null_count}/{total_count} ({null_pct:.0f}%) lab results have null values. This suggests the model failed to extract numeric values from the image.\n\t- {image_path}")


def _normalize_date_format(date_str: str | None) -> str | None:
    """
    Normalize date strings to YYYY-MM-DD format.

    Handles common formats:
    - DD/MM/YYYY (e.g., 20/11/2024 -> 2024-11-20)
    - DD-MM-YYYY (e.g., 20-11-2024 -> 2024-11-20)
    - YYYY-MM-DD (already correct)

    Args:
        date_str: Date string in various formats

    Returns:
        Date string in YYYY-MM-DD format, or None if invalid/null
    """

    # Guard: Empty or invalid date string
    if not date_str or date_str == "0000-00-00":
        return None

    # Already in correct format
    if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        return date_str

    # DD/MM/YYYY or DD-MM-YYYY format
    match = re.match(r"^(\d{2})[/-](\d{2})[/-](\d{4})$", date_str)
    if match:
        day, month, year = match.groups()
        return f"{year}-{month}-{day}"

    # Unable to parse - log and return None
    logger.warning(f"Unable to normalize date format: {date_str}")
    return None


def _clean_numeric_field(value) -> float | None:
    """Strip embedded metadata from numeric fields.

    Some LLMs embed extra field data into numeric reference fields:
    - '1.20, raw_comments: Confirmado por duplo ensaio (mesma amostra)'

    This function extracts just the numeric value.
    """

    # Guard: None passthrough
    if value is None:
        return None

    # Already numeric - return as float
    if isinstance(value, (int, float)):
        return float(value)

    # String value - strip embedded metadata and parse
    if isinstance(value, str):
        value = value.strip()
        # Strip embedded field markers
        for pattern in [
            ", comments:",
            ", raw_comments:",
        ]:
            if pattern.lower() in value.lower():
                value = value.split(",")[0].strip()
                break

        # Attempt numeric conversion
        try:
            # Handle Portuguese decimal format (comma as decimal separator)
            # But only if there's a single comma and no dot
            if "," in value and "." not in value:
                value = value.replace(",", ".")
            return float(value)
        except ValueError:
            logger.warning(f"Could not parse numeric field value: {value[:50]}")
            return None

    # Unsupported type
    return None


def _fix_lab_results_format(tool_result_dict: dict) -> dict:
    """Fix common LLM formatting issues in lab_results and dates."""

    # Fix date formats at report level
    for date_field in ["collection_date", "report_date"]:
        if date_field in tool_result_dict:
            tool_result_dict[date_field] = _normalize_date_format(tool_result_dict[date_field])

    # Guard: No lab_results to process
    if "lab_results" not in tool_result_dict:
        return tool_result_dict

    # Normalize stringified and sequence-style lab_results payloads into dict rows.
    tool_result_dict["lab_results"] = _normalize_lab_results_payload(tool_result_dict["lab_results"])

    # Guard: Unsupported lab_results payloads cannot be cleaned further.
    if not isinstance(tool_result_dict["lab_results"], list):
        return tool_result_dict

    # Clean numeric reference fields (strip embedded metadata like ", comments: ...")
    _clean_numeric_reference_fields(tool_result_dict)

    # Clean bounding-box coordinates so malformed boxes degrade to null instead of breaking the row.
    _clean_bbox_fields(tool_result_dict)

    # Filter out any items that still could not be normalized into row dicts.
    original_count = len(tool_result_dict["lab_results"])
    tool_result_dict["lab_results"] = [r for r in tool_result_dict["lab_results"] if isinstance(r, dict)]
    filtered_count = original_count - len(tool_result_dict["lab_results"])
    if filtered_count > 0:
        logger.warning(f"Filtered {filtered_count} non-dict lab_results items")

    return tool_result_dict


def _normalize_lab_results_payload(lab_results) -> list[dict] | object:
    """Normalize malformed lab_results payloads into a list of dict rows."""

    # Expand stringified top-level arrays or dicts before item-level normalization.
    expanded = _expand_lab_results_container(lab_results)

    # Guard: Only list payloads can be normalized row by row.
    if not isinstance(expanded, list):
        return expanded

    normalized_results: list[dict] = []

    # Normalize each row independently so one malformed item does not poison the page.
    for item in expanded:
        normalized_results.extend(_normalize_lab_result_item(item))

    return normalized_results


def _expand_lab_results_container(lab_results):
    """Expand top-level lab_results wrappers produced by malformed tool output."""

    # The happy path already provides a list of lab-result items.
    if isinstance(lab_results, list):
        return lab_results

    # A single dict means the model emitted one row without wrapping it in a list.
    if isinstance(lab_results, dict):
        return [lab_results]

    # Strings sometimes contain a JSON array or dict payload for the whole field.
    if isinstance(lab_results, str):
        parsed_value = _parse_serialized_value(lab_results)

        # Guard: Give up when the string is not structured data.
        if parsed_value is None:
            return lab_results

        return _expand_lab_results_container(parsed_value)

    # Unsupported container types must be handled by the caller's existing guards.
    return lab_results


def _normalize_lab_result_item(item) -> list[dict]:
    """Normalize one malformed lab_result item into zero or more dict rows."""

    # Existing well-formed rows pass through unchanged.
    if isinstance(item, dict):
        return [_normalize_lab_result_keys(item)]

    # Sequence payloads can encode key/value pairs or nested rows.
    if isinstance(item, (list, tuple)):
        return _normalize_sequence_item(item)

    # Strings may hold serialized dicts or label-packed field text.
    if isinstance(item, str):
        return _normalize_string_item(item)

    # Unknown payload shapes cannot be salvaged safely.
    return []


def _normalize_sequence_item(item: list | tuple) -> list[dict]:
    """Normalize list/tuple-based lab_result payloads."""

    # Key/value pair sequences can be turned into a single row dict.
    sequence_dict = _dict_from_sequence(item)
    if sequence_dict is not None:
        return [_normalize_lab_result_keys(sequence_dict)]

    normalized_results: list[dict] = []

    # Nested sequences may contain multiple encoded rows.
    for nested_item in item:
        normalized_results.extend(_normalize_lab_result_item(nested_item))

    return normalized_results


def _normalize_string_item(item: str) -> list[dict]:
    """Normalize string-based lab_result payloads."""

    stripped_item = item.strip()

    # Guard: Ignore blank string items.
    if not stripped_item:
        return []

    # Serialized JSON/Python literals are the most common malformed payload shape.
    parsed_value = _parse_serialized_value(stripped_item)
    if parsed_value is not None:
        return _normalize_lab_result_item(parsed_value)

    # Label-packed strings are the fallback shape produced by some tool-call failures.
    labeled_dict = _parse_labeled_lab_result(stripped_item)
    if labeled_dict is not None:
        return [_normalize_lab_result_keys(labeled_dict)]

    return []


def _parse_serialized_value(raw_value: str):
    """Parse JSON or Python-literal strings emitted inside tool arguments."""

    # JSON is the primary serialization format for tool-call payloads.
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        pass

    # Python literal syntax is a common Gemini fallback when JSON discipline slips.
    try:
        return ast.literal_eval(raw_value)
    except (SyntaxError, ValueError):
        return None


def _dict_from_sequence(item: list | tuple) -> dict | None:
    """Convert key/value sequences into a dict row when possible."""

    normalized_dict: dict = {}

    # Handle explicit [("field", value), ...] or [["field", value], ...] payloads.
    if item and all(isinstance(part, (list, tuple)) and len(part) == 2 for part in item):
        for key, value in item:
            # Guard: Non-string keys are not usable as field names.
            if not isinstance(key, str):
                return None

            normalized_dict[key] = value

        return normalized_dict

    # Handle alternating ["field", value, "field2", value2] payloads.
    if len(item) % 2 == 0 and item and all(isinstance(item[index], str) for index in range(0, len(item), 2)):
        for index in range(0, len(item), 2):
            normalized_dict[item[index]] = item[index + 1]

        return normalized_dict

    return None


def _parse_labeled_lab_result(raw_value: str) -> dict | None:
    """Parse strings that inline field labels inside one lab-result item."""

    field_pattern = re.compile(
        r"(?P<field>raw_lab_name|raw_section_name|raw_value|raw_lab_unit|raw_reference_range|raw_reference_min|raw_reference_max|raw_comments|bbox_left|bbox_top|bbox_right|bbox_bottom)\s*[:=]\s*",
        flags=re.IGNORECASE,
    )
    matches = list(field_pattern.finditer(raw_value))

    # Guard: Without explicit field labels there is nothing deterministic to recover.
    if not matches:
        return None

    parsed_dict: dict[str, str] = {}

    # Slice the string between successive field markers to preserve embedded punctuation.
    for index, match in enumerate(matches):
        field_name = match.group("field").lower()
        value_start = match.end()
        value_end = matches[index + 1].start() if index + 1 < len(matches) else len(raw_value)
        field_value = raw_value[value_start:value_end].strip(" \t\r\n,;|")
        parsed_dict[field_name] = field_value

    # Guard: A recovered row without a lab name is not usable downstream.
    if not parsed_dict.get("raw_lab_name"):
        return None

    return parsed_dict


def _normalize_lab_result_keys(row_dict: dict) -> dict:
    """Map known legacy field aliases onto the canonical extraction schema."""

    alias_map = {
        "section_name": "raw_section_name",
        "raw_unit": "raw_lab_unit",
        "lab_unit_raw": "raw_lab_unit",
        "reference_range": "raw_reference_range",
        "reference_min": "raw_reference_min",
        "reference_max": "raw_reference_max",
        "left": "bbox_left",
        "top": "bbox_top",
        "right": "bbox_right",
        "bottom": "bbox_bottom",
    }
    normalized_dict: dict = {}

    # Copy fields into canonical keys so downstream validation sees one stable schema.
    for key, value in row_dict.items():
        normalized_key = alias_map.get(key, key)
        normalized_dict[normalized_key] = value

    return normalized_dict


def _clean_numeric_reference_fields(tool_result_dict: dict) -> None:
    """Clean numeric reference fields by stripping embedded metadata."""

    for item in tool_result_dict["lab_results"]:
        if isinstance(item, dict):
            if "raw_reference_min" in item:
                item["raw_reference_min"] = _clean_numeric_field(item["raw_reference_min"])
            if "raw_reference_max" in item:
                item["raw_reference_max"] = _clean_numeric_field(item["raw_reference_max"])


def _clean_bbox_fields(tool_result_dict: dict) -> None:
    """Normalize bounding-box coordinates or drop malformed boxes entirely."""

    for item in tool_result_dict["lab_results"]:
        # Guard: Skip non-dict items because other cleanup passes already handle them.
        if not isinstance(item, dict):
            continue

        # Guard: Rows without any bbox data do not need normalization.
        if not any(field in item for field in BBOX_FIELDS):
            continue

        cleaned_values = {field: _clean_numeric_field(item.get(field)) for field in BBOX_FIELDS}

        # Guard: Partial boxes are not review-safe, so clear the whole box.
        if any(value is None for value in cleaned_values.values()):
            for field in BBOX_FIELDS:
                item[field] = None
            continue

        left = cleaned_values["bbox_left"]
        top = cleaned_values["bbox_top"]
        right = cleaned_values["bbox_right"]
        bottom = cleaned_values["bbox_bottom"]

        # Guard: Non-positive boxes are not usable in the viewer.
        if left < 0 or top < 0 or right <= left or bottom <= top:
            for field in BBOX_FIELDS:
                item[field] = None
            continue

        # Persist canonical numeric coordinates when the box is internally consistent.
        for field, value in cleaned_values.items():
            item[field] = value


def _salvage_lab_results(tool_result_dict: dict) -> dict:
    """Try to salvage valid lab results even if report validation fails."""

    # Guard: No lab_results to salvage
    if "lab_results" not in tool_result_dict or not isinstance(tool_result_dict["lab_results"], list):
        return HealthLabReport(lab_results=[]).model_dump(mode="json")

    # Validate each lab result individually
    valid_results = []
    for i, lr_data in enumerate(tool_result_dict["lab_results"]):
        # Skip non-dict items
        if not isinstance(lr_data, dict):
            logger.warning(f"Skipping non-dict lab_result[{i}] in salvage")
            continue

        # Try to validate this individual result
        try:
            lr_model = LabResult(**lr_data)
            valid_results.append(lr_model.model_dump(mode="json"))
        # Invalid result - skip and continue salvaging others
        except Exception as e:
            logger.error(f"Failed to validate lab_result[{i}] in salvage: {e}. Data: {lr_data}")

    logger.warning(f"Salvaged {len(valid_results)}/{len(tool_result_dict['lab_results'])} lab results")
    return HealthLabReport(lab_results=valid_results).model_dump(mode="json")
