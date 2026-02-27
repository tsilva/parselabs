"""Lab result extraction from images using vision models."""

import base64
import json
import logging
import re
from pathlib import Path

from openai import APIError, OpenAI
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)

# Sentinel for unknown standardized values (imported from config to avoid circular import)
_UNKNOWN_VALUE = "$UNKNOWN$"


# ========================================
# Pydantic Models
# ========================================


class LabResult(BaseModel):
    """Single lab test result - optimized for extraction accuracy."""

    model_config = ConfigDict(populate_by_name=True)

    # Raw extraction (exactly as shown in PDF)
    raw_lab_name: str = Field(
        alias="lab_name_raw",
        description="Test name ONLY as written in the PDF. Must contain ONLY the test name - DO NOT include values, units, reference ranges, or field labels. WRONG: 'Glucose, raw_value: 100' CORRECT: 'Glucose'",
    )
    raw_value: str | None = Field(
        default=None,
        alias="value_raw",
        description="Result value ONLY. Must contain ONLY the numeric or text result - DO NOT include test names, units, or field labels. Examples: '5.2', '14.8', 'NEGATIVO', 'POSITIVO'",
    )
    raw_lab_unit: str | None = Field(
        default=None,
        alias="lab_unit_raw",
        description="Unit ONLY as written in PDF. Must contain ONLY the unit symbol - DO NOT include values or test names. Examples: 'mg/dL', '%', 'U/L'",
    )
    raw_reference_range: str | None = Field(
        default=None,
        alias="reference_range",
        description="Complete reference range text EXACTLY as shown.",
    )
    raw_reference_min: float | None = Field(
        default=None,
        alias="reference_min_raw",
        description="Minimum reference value as a PLAIN NUMBER ONLY. Parse from reference_range. Examples: '< 40' → null, '150 - 400' → 150, '26.5-32.6' → 26.5",
    )
    raw_reference_max: float | None = Field(
        default=None,
        alias="reference_max_raw",
        description="Maximum reference value as a PLAIN NUMBER ONLY. Parse from reference_range. Examples: '< 40' → 40, '150 - 400' → 400, '26.5-32.6' → 32.6",
    )
    raw_comments: str | None = Field(
        default=None,
        alias="comments",
        description="Additional notes or remarks about the test (NOT the test result itself). Only use for extra information like methodology notes or special conditions.",
    )
    # Internal fields (added by pipeline, not by LLM)
    lab_name_standardized: str | None = Field(default=None, description="Standardized lab name")
    lab_unit_standardized: str | None = Field(default=None, description="Standardized lab unit")

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

    collection_date: str | None = Field(
        default=None,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Specimen collection date in YYYY-MM-DD format",
    )
    report_date: str | None = Field(
        default=None,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Report issue date in YYYY-MM-DD format",
    )
    lab_facility: str | None = Field(default=None, description="Name of laboratory that performed tests")
    page_has_lab_data: bool | None = Field(
        default=None,
        description="True if page contains lab test results, False if page is cover/instructions/administrative with no lab data",
    )
    lab_results: list[LabResult] = Field(
        default_factory=list,
        description="List of all lab test results extracted from this page/document",
    )
    source_file: str | None = Field(default=None, description="Source PDF filename")

    @staticmethod
    def _clear_empty_strings(model: BaseModel):
        """Set empty string optional fields to None on a Pydantic model."""

        for field_name in model.model_fields:
            value = getattr(model, field_name)
            field_info = model.model_fields[field_name]

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

    raw_lab_name: str = Field(
        description="Test name ONLY as written in the PDF. Must contain ONLY the test name - DO NOT include values, units, reference ranges, or field labels. WRONG: 'Glucose, raw_value: 100' CORRECT: 'Glucose'"
    )
    lab_name: str | None = Field(
        default=None,
        description="Standardized lab name from the provided list. Match the raw name to the CLOSEST entry. Use '$UNKNOWN$' if no match.",
    )
    raw_value: str | None = Field(
        default=None,
        description="Result value ONLY. Must contain ONLY the numeric or text result - DO NOT include test names, units, or field labels. Examples: '5.2', '14.8', 'NEGATIVO', 'POSITIVO'",
    )
    raw_lab_unit: str | None = Field(
        default=None,
        description="Unit ONLY as written in PDF. Must contain ONLY the unit symbol - DO NOT include values or test names. Examples: 'mg/dL', '%', 'U/L'",
    )
    unit: str | None = Field(
        default=None,
        description="Standardized unit from the provided list for this lab. Normalize FORMAT only (e.g., 'mg/dl' → 'mg/dL'). Do NOT convert units. Use '$UNKNOWN$' if no match.",
    )
    raw_reference_range: str | None = Field(
        default=None,
        description="Complete reference range text EXACTLY as shown.",
    )
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


class HealthLabReportExtraction(BaseModel):
    """LLM-facing schema for function calling.

    Excludes pipeline-internal fields from HealthLabReport. Uses LabResultExtraction
    to reduce schema size (~41% smaller) and avoid confusing the model.
    """

    collection_date: str | None = Field(
        default=None,
        description="Specimen collection date in YYYY-MM-DD format",
    )
    report_date: str | None = Field(
        default=None,
        description="Report issue date in YYYY-MM-DD format",
    )
    lab_facility: str | None = Field(default=None, description="Name of laboratory that performed tests")
    page_has_lab_data: bool | None = Field(
        default=None,
        description="True if page contains lab test results, False if page is cover/instructions/administrative",
    )
    lab_results: list[LabResultExtraction] = Field(
        default_factory=list,
        description="List of all lab test results extracted from this page/document",
    )
    source_file: str | None = Field(default=None, description="Source PDF filename")


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


def _build_standardized_names_section(standardized_names: list[str], lab_specs_data: dict) -> str:
    """Build the standardized names reference section for the extraction prompt.

    Args:
        standardized_names: Sorted list of all standardized lab names
        lab_specs_data: Raw lab specs dict for looking up primary units and lab types

    Returns:
        Formatted string to append to the system prompt
    """

    sections: dict[str, list[str]] = {
        "blood": [],
        "urine": [],
        "feces": [],
        "saliva": [],
        "other": [],
    }

    # Build entries grouped by lab type
    for name in standardized_names:
        spec = lab_specs_data.get(name, {})
        primary_unit = spec.get("primary_unit", "")
        lab_type = spec.get("lab_type", "blood")
        entry = f"- {name} [{primary_unit}]"
        sections.setdefault(lab_type, []).append(entry)

    # Format each non-empty lab type section
    result = "\n\nSTANDARDIZED LAB NAMES AND UNITS:\n"
    for lab_type in ["blood", "urine", "feces", "saliva", "other"]:
        entries = sections.get(lab_type, [])
        if entries:
            result += f"\n### {lab_type.title()}\n" + "\n".join(entries) + "\n"

    return result


# ========================================
# Extraction Prompts
# ========================================

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
EXTRACTION_SYSTEM_PROMPT = (_PROMPTS_DIR / "extraction_system.md").read_text(encoding="utf-8").strip()
EXTRACTION_USER_PROMPT = (_PROMPTS_DIR / "extraction_user.md").read_text(encoding="utf-8").strip()
TEXT_EXTRACTION_USER_PROMPT = (_PROMPTS_DIR / "text_extraction_user.md").read_text(encoding="utf-8").strip()


def _empty_report() -> dict:
    """Return a fresh empty lab report dict."""

    return HealthLabReport(lab_results=[]).model_dump(mode="json")


# ========================================
# Extraction Function
# ========================================


def extract_labs_from_page_image(
    image_path: Path,
    model_id: str,
    client: OpenAI,
    temperature: float = 0.0,
    max_retries: int = 3,
    temperature_step: float = 0.2,
    standardization_section: str | None = None,
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
        max_retries: Maximum retry attempts with escalating temperature (default: 3)
        temperature_step: Temperature increment per retry (default: 0.2)
        standardization_section: Optional standardized names/units list to append to system prompt.
            When provided, the model also populates lab_name and unit fields inline.

    Returns:
        Dictionary with extracted report data (validated by Pydantic).
        On failure, includes _extraction_failed=True and _failure_reason.
    """

    # Load image data for API call
    with open(image_path, "rb") as img_file:
        img_data = base64.standard_b64encode(img_file.read()).decode("utf-8")

    # Build effective system prompt with optional standardization section
    effective_system_prompt = EXTRACTION_SYSTEM_PROMPT + standardization_section if standardization_section else EXTRACTION_SYSTEM_PROMPT

    # Attempt extraction with retry logic
    result = _attempt_extraction_with_retries(
        image_path=image_path,
        img_data=img_data,
        model_id=model_id,
        client=client,
        effective_system_prompt=effective_system_prompt,
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
            return _empty_report()

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
    result = _empty_report()
    result["_extraction_failed"] = True
    result["_failure_reason"] = f"Malformed output after {max_retries + 1} attempts"
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


def extract_labs_from_text(
    text: str,
    model_id: str,
    client: OpenAI,
    temperature: float = 0.0,
    standardization_section: str | None = None,
) -> dict:
    """
    Extract lab results from text using a text-only LLM (no vision).

    This is a cost-optimized extraction path for PDFs with embedded text.
    Uses the same prompts and tool schema as vision extraction.

    Args:
        text: Extracted text from PDF (via pdftotext)
        model_id: Model to use for extraction
        client: OpenAI client instance
        temperature: Temperature for sampling
        standardization_section: Optional standardized names/units list to append to system prompt.

    Returns:
        Dictionary with extracted report data (validated by Pydantic)
    """

    # Build effective system prompt with optional standardization section
    effective_system_prompt = EXTRACTION_SYSTEM_PROMPT + standardization_section if standardization_section else EXTRACTION_SYSTEM_PROMPT

    # Build user prompt for text extraction
    user_prompt = _build_text_extraction_prompt(text, standardization_section)

    # Attempt extraction via API
    extraction_result = _execute_text_extraction_api_call(
        model_id=model_id,
        client=client,
        effective_system_prompt=effective_system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
    )

    # Guard: Extraction failed at API level
    if extraction_result is None:
        return HealthLabReport(lab_results=[]).model_dump(mode="json")

    tool_result_dict = extraction_result

    # Pre-process: Fix common LLM issues before Pydantic validation
    tool_result_dict = _fix_lab_results_format(tool_result_dict)

    # Validate and return results
    return _validate_text_extraction_result(tool_result_dict)


def _build_text_extraction_prompt(text: str, standardization_section: str | None) -> str:
    """Build the user prompt for text-based extraction."""

    # Add standardization reminder only when standardization is active
    std_reminder = "\nAlso set lab_name (standardized name from the provided list) and unit (standardized unit format) for each result.\n" if standardization_section else ""

    return TEXT_EXTRACTION_USER_PROMPT.format(std_reminder=std_reminder, text=text)


def _execute_text_extraction_api_call(
    model_id: str,
    client: OpenAI,
    effective_system_prompt: str,
    user_prompt: str,
    temperature: float,
) -> dict | None:
    """Execute the API call for text extraction.

    Returns:
        Parsed tool result dict or None on failure
    """

    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": effective_system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=16384,
            tools=TOOLS,
            tool_choice={
                "type": "function",
                "function": {"name": "extract_lab_results"},
            },
        )
    # API-level failure - propagate as RuntimeError
    except APIError as e:
        logger.error(f"API Error during text-based lab extraction: {e}")
        raise RuntimeError(f"Text-based lab extraction failed: {e}")

    # Guard: Invalid response structure
    if not completion or not completion.choices:
        logger.error("Invalid completion response structure for text extraction")
        return None

    # Guard: No tool call from model
    if not completion.choices[0].message.tool_calls:
        logger.warning("No tool call by model for text-based lab extraction")
        return None

    # Parse tool arguments
    tool_args_raw = completion.choices[0].message.tool_calls[0].function.arguments
    try:
        return json.loads(tool_args_raw)
    # Malformed JSON from API response
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error for text extraction tool args: {e}")
        return None


def _validate_text_extraction_result(tool_result_dict: dict) -> dict:
    """Validate text extraction result with Pydantic."""

    try:
        report_model = HealthLabReport(**tool_result_dict)
        report_model.normalize_empty_optionals()

        # Log extraction results
        if report_model.lab_results:
            logger.info(f"Text extraction successful: {len(report_model.lab_results)} lab results")
        # No results - check if page was intentionally empty
        else:
            # Page explicitly marked as non-lab content
            if report_model.page_has_lab_data is False:
                logger.debug("Document confirmed to have no lab data (text extraction)")
            # Possible extraction failure
            else:
                logger.warning("Text extraction returned 0 lab results")

        return report_model.model_dump(mode="json")

    # Validation failed - salvage individual valid results
    except Exception as e:
        num_results = len(tool_result_dict.get("lab_results", []))
        logger.error(f"Model validation error for text extraction with {num_results} lab_results: {e}")
        return _salvage_lab_results(tool_result_dict)


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


# Backward compatibility: mapping from old _raw suffix field names to new raw_ prefix names
_RAW_FIELD_RENAMES = {
    "lab_name_raw": "raw_lab_name",
    "value_raw": "raw_value",
    "lab_unit_raw": "raw_lab_unit",
    "reference_min_raw": "raw_reference_min",
    "reference_max_raw": "raw_reference_max",
    "reference_range": "raw_reference_range",
    "comments": "raw_comments",
}


def _normalize_raw_field_names(lab_results: list[dict]) -> None:
    """Rename old _raw suffix keys to new raw_ prefix keys in lab result dicts."""

    for item in lab_results:
        if not isinstance(item, dict):
            continue
        for old_key, new_key in _RAW_FIELD_RENAMES.items():
            if old_key in item and new_key not in item:
                item[new_key] = item.pop(old_key)


def _fix_lab_results_format(tool_result_dict: dict) -> dict:
    """Fix common LLM formatting issues in lab_results and dates."""

    # Fix date formats at report level
    for date_field in ["collection_date", "report_date"]:
        if date_field in tool_result_dict:
            tool_result_dict[date_field] = _normalize_date_format(tool_result_dict[date_field])

    # Guard: No lab_results to process
    if "lab_results" not in tool_result_dict or not isinstance(tool_result_dict["lab_results"], list):
        return tool_result_dict

    # Normalize old _raw suffix field names to new raw_ prefix names
    _normalize_raw_field_names(tool_result_dict["lab_results"])

    # Map LLM-populated standardization fields to internal fields
    _map_standardization_fields(tool_result_dict)

    # Clean numeric reference fields (strip embedded metadata like ", comments: ...")
    _clean_numeric_reference_fields(tool_result_dict)

    # Filter out any non-dict items (string results from malformed output)
    original_count = len(tool_result_dict["lab_results"])
    tool_result_dict["lab_results"] = [r for r in tool_result_dict["lab_results"] if isinstance(r, dict)]
    filtered_count = original_count - len(tool_result_dict["lab_results"])
    if filtered_count > 0:
        logger.warning(f"Filtered {filtered_count} non-dict lab_results items")

    return tool_result_dict


def _map_standardization_fields(tool_result_dict: dict) -> None:
    """Map LLM-populated standardization fields to internal field names."""

    # lab_name (standardized name) → lab_name_standardized
    # unit (standardized unit) → lab_unit_standardized
    for item in tool_result_dict["lab_results"]:
        if isinstance(item, dict):
            # Rename lab_name to lab_name_standardized
            if "lab_name" in item:
                std_name = item.pop("lab_name")
                item["lab_name_standardized"] = std_name if std_name else _UNKNOWN_VALUE

            # Rename unit to lab_unit_standardized (map $UNKNOWN$ to None)
            if "unit" in item:
                std_unit = item.pop("unit")
                # Map $UNKNOWN$ to None so unit inference in normalization can handle it
                item["lab_unit_standardized"] = std_unit if (std_unit and std_unit != _UNKNOWN_VALUE) else None


def _clean_numeric_reference_fields(tool_result_dict: dict) -> None:
    """Clean numeric reference fields by stripping embedded metadata."""

    for item in tool_result_dict["lab_results"]:
        if isinstance(item, dict):
            if "raw_reference_min" in item:
                item["raw_reference_min"] = _clean_numeric_field(item["raw_reference_min"])
            if "raw_reference_max" in item:
                item["raw_reference_max"] = _clean_numeric_field(item["raw_reference_max"])


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
